import unittest
import numpy as np
from Design.SMRF_Beam_Shell_Connection import rotation_map,connect_independent_beam_rotations
from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness,rigid_arm_matrix
from test_floor_generalized_stiffness import floor_fixture

def beam_block(length,ei):
    l=length
    return ei/l**3*np.array([[12,6*l,-12,6*l],[6*l,4*l*l,-6*l,2*l*l],[-12,-6*l,12,-6*l],[6*l,2*l*l,-6*l,4*l*l]])

class BeamShellConnectionTests(unittest.TestCase):
    def test_independent_constant_curvature_oracle_every_mesh(self):
        # Exact Euler beam v=k*x^2/2, rz=k*x, M=EI*k, Q=0.
        # A shell with ux=0 would have continuum curl=k*x/2 instead.
        for n in (2,4,8):
            length=24.;ei=900.;curvature=.0002;x=np.linspace(0,length,n+1)
            blocks=[beam_block(length/n,ei)]*n
            rotations=rotation_map(blocks)@np.r_[curvature*x*x/2,0.,curvature*length]
            np.testing.assert_allclose(rotations,curvature*x,rtol=0,atol=1e-16)
            energy=0.
            for i,b in enumerate(blocks):
                u=np.array([curvature*x[i]**2/2,rotations[i],curvature*x[i+1]**2/2,rotations[i+1]])
                np.testing.assert_allclose(b@u,[0,-ei*curvature,0,ei*curvature],atol=1e-13)
                energy+=u@b@u/2
            self.assertAlmostEqual(energy,.5*ei*curvature**2*length,places=14)
            tied=.5*curvature*x;tied[[0,-1]]=curvature*x[[0,-1]]
            bad_energy=0.
            for i,b in enumerate(blocks):
                u=np.array([curvature*x[i]**2/2,tied[i],curvature*x[i+1]**2/2,tied[i+1]])
                bad_energy+=u@b@u/2
            # For interior rotation error -k*x/2, stationary-energy expansion
            # gives U_tied/U_exact = 1+(n-1)^2: 2,10,50 for these meshes.
            self.assertAlmostEqual(bad_energy/energy,1+(n-1)**2,places=10)

    def test_both_axis_signs_and_unloaded_rotation_equilibrium(self):
        rng=np.random.default_rng(83)
        for sign in (1.,-1.):
            t=np.diag([sign,1,sign,1]);blocks=[t@beam_block(l,720.)@t for l in (2.,3.,5.,4.)]
            values=rng.normal(size=7)*.001;rot=rotation_map(blocks)@values
            sums=np.zeros(5)
            for i,b in enumerate(blocks):
                f=b@np.array([values[i],rot[i],values[i+1],rot[i+1]])
                sums[i]+=f[1];sums[i+1]+=f[3]
            np.testing.assert_allclose(sums[1:-1],0.,atol=1e-14)

    def test_floor_energy_forces_and_rigid_modes(self):
        floor=floor_fixture();connect_independent_beam_rotations(floor)
        model=GeneralizedFloorStiffness(floor,beam_bending_factor=.35,slab_bending_factors=.2,
            web_torsion_factor=.1,stiffness_basis='independent connection fixture',input_identity='connection83')
        u=np.random.default_rng(83).normal(size=len(floor.dof_ids))*.0001
        result=model.recover(u,floor.loads(-.002))
        self.assertLess(result['force_partition_relative_error'],1e-10)
        um,ub=model.fields(u)
        energy=um@(floor.parts['membrane']@um)+.35*ub@(floor.parts['membrane']@ub)
        for factor,key in ((.35,'web_bending'),(.2,'slab_bending'),(1.,'shear'),(.1,'web_torsion')):energy+=factor*u@(floor.parts[key]@u)
        self.assertAlmostEqual(u@(model.stiffness@u)/energy,1.,places=11)
        # Changing shell drilling at a clear-span node cannot turn a beam.
        support_ids={s['id'] for s in floor.supports};drill=np.zeros_like(u)
        for i,node in enumerate(floor.node_ids):
            if node not in support_ids:drill[6*i+5]=.003
        for row in floor.elements:
            if row['kind']=='web':np.testing.assert_allclose(row['transform']@drill[row['indices']],0.,atol=1e-15)
        q=np.array([1.,-2.,3.,.001,-.002,.003])
        rigid=np.concatenate([rigid_arm_matrix(floor.node_positions[n])@q for n in floor.node_ids])
        for row in floor.elements:
            if row['kind']=='web':
                local=row['transform']@rigid[row['indices']]
                expected=np.concatenate([rigid_arm_matrix(p)@q for p in row['points']])
                np.testing.assert_allclose(local,expected,atol=1e-14)
        self.assertTrue(all(len(r['indices'])>=12 for r in floor.elements if r['kind']=='web'))
        with self.assertRaisesRegex(ValueError,'already'):connect_independent_beam_rotations(floor)

    def test_bad_chain_input_rejected(self):
        with self.assertRaises(ValueError):rotation_map([])
        with self.assertRaises(ValueError):rotation_map([np.zeros((4,4)),np.zeros((4,4))])

if __name__=='__main__':unittest.main()
