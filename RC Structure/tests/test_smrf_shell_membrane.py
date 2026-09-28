"""Hand equilibrium and independent minimization checks for the membrane kernel."""
import unittest,math
import numpy as np
from scipy.optimize import minimize_scalar
from Design.SMRF_Shell_Membrane import membrane_equilibrium,shell_layer_resultants,correct_steel_depths


class ShellMembraneTests(unittest.TestCase):
    def test_hand_cases_cover_tension_shear_and_compression(self):
        cases=[([10.,5.,0.],[10.,5.], [0.,0.,0.]),
               ([0.,0.,3.],[3.,3.],[-3.,-3.,3.]),
               ([10.,-8.,4.],[12.,0.],[-2.,-8.,4.]),
               ([-8.,10.,-4.],[0.,12.],[-8.,-2.,-4.]),
               ([-9.,-9.,3.],[0.,0.],[-9.,-9.,3.]),
               ([-2.,-2.,3.],[1.,1.],[-3.,-3.,3.]),
               ([2.,-7.,0.],[2.,0.],[0.,-7.,0.]),
               ([0.,0.,0.],[0.,0.],[0.,0.,0.])]
        for actions,steel,concrete in cases:
            with self.subTest(actions=actions):
                r=membrane_equilibrium(actions)
                np.testing.assert_allclose(r['steel_tension_kip_per_in'],steel,atol=1e-12)
                np.testing.assert_allclose(r['concrete_tensor_kip_per_in'],concrete,atol=1e-12)
                self.assertFalse(r['engineering_verified']);self.assertFalse(r['capacity_pairing_authorized'])

    def test_independent_scalar_optimization_and_tensor_equilibrium(self):
        # Independent formulation: choose concrete compression a in X.
        # Positive semidefinite -C requires a*b >= Nxy^2. The objective is
        # max(0,Nx+a)+max(0,Ny+b), with a >= max(0,-Nx).
        rng=np.random.default_rng(926)
        for nx,ny,q in rng.uniform(-20.,20.,(160,3)):
            r=membrane_equilibrium([float(nx),float(ny),float(q)])
            lower=max(1e-12,-nx)
            objective=lambda a:nx+a+max(0.,ny+q*q/a)
            optimum=minimize_scalar(objective,bounds=(lower,200.),method='bounded',options={'xatol':1e-12})
            candidate=min(objective(lower),optimum.fun)
            actual=sum(r['steel_tension_kip_per_in'])
            self.assertAlmostEqual(actual,candidate,delta=1e-5)
            cx,cy,cxy=r['concrete_tensor_kip_per_in'];tx,ty=r['steel_tension_kip_per_in']
            np.testing.assert_allclose([cx+tx,cy+ty,cxy],[nx,ny,q],rtol=1e-12,atol=1e-12)
            self.assertLessEqual(np.linalg.eigvalsh([[cx,cxy],[cxy,cy]])[-1],1e-10)

    def test_axis_swap_and_shear_sign_do_not_create_reinforcement_asymmetry(self):
        for n in ([10.,-2.,4.],[-9.,2.,3.],[1.,2.,-5.]):
            r=membrane_equilibrium(n)['steel_tension_kip_per_in']
            flipped=membrane_equilibrium([n[0],n[1],-n[2]])['steel_tension_kip_per_in']
            swapped=membrane_equilibrium([n[1],n[0],n[2]])['steel_tension_kip_per_in']
            np.testing.assert_allclose(flipped,r,atol=1e-12)
            np.testing.assert_allclose(swapped,r[::-1],atol=1e-12)

    def test_asymmetric_layer_centroids_reconstruct_all_six_shell_actions(self):
        n=[4.,-2.,3.];m=[-12.,5.,7.];ht,hb=2.,3.
        result=shell_layer_resultants(n,m,top_center_in=ht,bottom_center_in=hb)
        t=np.array(result['top_kip_per_in']);b=np.array(result['bottom_kip_per_in'])
        np.testing.assert_allclose(t,[4.8,-2.2,.4],atol=1e-12)
        np.testing.assert_allclose(t+b,n,atol=1e-12)
        np.testing.assert_allclose(-ht*t+hb*b,m,atol=1e-12)
        self.assertFalse(result['reinforcement_depth_corrected'])

    def test_bad_units_shapes_or_nonfinite_inputs_are_not_silently_accepted(self):
        for n in ([1,2],[1,2,True],[1,2,float('nan')],[1,'2',3]):
            with self.assertRaises(ValueError):membrane_equilibrium(n)
        with self.assertRaises(ValueError):shell_layer_resultants([0,0,0],[1,2,3],top_center_in=0,bottom_center_in=2)

    def test_published_2014_appendix_b_depth_correction_with_face_branch_change(self):
        # Colombo et al., printed p.67, rounded fixed-layer actions for the
        # third iteration of element 3. Convert tf/m and m to kip/in and in.
        length=1000/25.4;force=9.80665/4.4482216152605/length
        result=correct_steel_depths([v*force for v in (-265.71,-36.23,141.55)],
            [v*force for v in (2.09,312.59,-58.)],top_center_in=.5525*length,bottom_center_in=.6045*length,
            top_steel_centers_in=[.400*length,.450*length],bottom_steel_centers_in=[.400*length,.450*length])
        np.testing.assert_allclose(result['top_steel_tension_kip_per_in'],[0.,0.],atol=1e-10)
        np.testing.assert_allclose(np.array(result['bottom_steel_tension_kip_per_in'])/force,[72.98,427.70],atol=.03,rtol=0)
        np.testing.assert_allclose(np.array(result['top_concrete']['actions_kip_per_in'])/force,[-278.61,-93.34,141.55],atol=.03,rtol=0)
        self.assertAlmostEqual(result['top_concrete']['concrete_principal_kip_per_in'][0]/force,-355.14,delta=.03)
        self.assertLess(result['normalized_equilibrium_residual'],1e-9)
        self.assertFalse(result['compression_layer_thickness_iterated'])

    def test_actual_depths_preserve_combined_actions_and_reject_iteration_exhaustion(self):
        kwargs=dict(top_center_in=2.,bottom_center_in=2.5,top_steel_centers_in=[1.4,1.1],bottom_steel_centers_in=[1.6,1.3])
        for top,bottom in (([8.,9.,2.],[7.,10.,-1.]),([-5.,0.,0.],[5.,0.,0.]),([5.,0.,0.],[-5.,0.,0.])):
            r=correct_steel_depths(top,bottom,**kwargs)
            np.testing.assert_allclose(r['reconstructed_normal_kip_per_in'],np.array(top)+bottom,atol=1e-9)
            np.testing.assert_allclose(r['reconstructed_moment_kip_in_per_in'],-2*np.array(top)+2.5*np.array(bottom),atol=1e-9)
        with self.assertRaisesRegex(RuntimeError,'did not converge'):
            correct_steel_depths([-5.,0.,0.],[5.,0.,0.],max_iterations=1,**kwargs)


if __name__=='__main__':unittest.main()
