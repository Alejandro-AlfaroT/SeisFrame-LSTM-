"""Independent secant, finite-volume and two-way stiffness algebra tests."""
import math
import unittest
import numpy as np
from Design.SMRF_Slab_Secant import LayeredSlabSecant,principal_secant_matrix
from Design.SMRF_Slab_Stiffness import SlabStripStiffness


def mats():
    return [dict(id=axis+face,axis=axis,z_in=sgn*z,area_in2_per_in=.31/8,
                 bar_area_in2=.31,fy_ksi=60.)
        for axis,z in [('x',3.),('y',2.25)] for face,sgn in [('top',1),('bottom',-1)]]


class SlabSecantTests(unittest.TestCase):
    def setUp(self):
        self.s=LayeredSlabSecant(8,5,mats(),ec_ksi=57*math.sqrt(5000))

    def test_six_actual_range_failures_use_direct_solve_without_changing_physical_rank(self):
        import json
        from pathlib import Path
        import mpmath as mp
        from Design.SMRF_Physical_Floor_Materials import CellVoidProfile
        cases=json.loads(Path(__file__).with_name('slab_range_151_fixture.json').read_text())
        for case in cases:
            empty=CellVoidProfile(case['box'],7.5,[],state_identity=case['point_id'])
            model=LayeredSlabSecant(7.5,5.,[],ec_ksi=self.s.ec,concrete_void_profile=empty,quadrature_order=12)
            out=model.evaluate(case['q'],branch='restricted_cracked');recovery=out['membrane_relaxation_recovery']
            self.assertTrue(recovery['applied']);self.assertEqual(recovery['retained_rank'],3)
            self.assertFalse(recovery['artificial_stiffness_added']);self.assertFalse(recovery['rank_threshold_changed'])
            self.assertTrue(out['constitutive_state_verified'],out['domain_reasons'])
            self.assertLessEqual(recovery['final_response_range_error'],recovery['tolerance'])
            a=np.array(out['A']);b=np.array(out['B']);response=np.array(out['zero_incremental_N_membrane_response'])
            np.testing.assert_array_equal(np.array(out['ABD'])@np.array(case['q'])-np.array(out['nominal_resultants']),out['secant_reconstruction_residual'])
            with mp.workdps(80):oracle=np.array((-(mp.matrix(a.tolist())**-1)*mp.matrix(b.tolist())).tolist(),float)
            self.assertLessEqual(float(np.max(abs(response-oracle))),1e-7+1e-7*float(np.max(abs(oracle))))
            d=np.array(out['zero_incremental_N_relaxed_D']);energy=out['relaxed_bending_energy_recovery']
            self.assertTrue(energy['applied']);self.assertTrue(energy['quadrature']['passed'])
            self.assertGreaterEqual(np.linalg.eigvalsh(d).min(),-1e-9*max(1.,float(np.max(abs(d)))))
            self.assertLessEqual(float(np.max(abs(a@response+b))),recovery['tolerance'])
        # The new fallback must preserve an actual zero-stiffness nullspace.
        empty=CellVoidProfile([0,1,0,1],7.5,[],state_identity='true_zero')
        zero=LayeredSlabSecant(7.5,5.,[],ec_ksi=self.s.ec,concrete_void_profile=empty).evaluate([.001,.001,0,0,0,0],branch='restricted_cracked')
        self.assertFalse(zero['membrane_relaxation_recovery']['applied'])
        np.testing.assert_array_equal(zero['zero_incremental_N_membrane_response'],np.zeros((3,3)))
        np.testing.assert_array_equal(zero['zero_incremental_N_relaxed_D'],np.zeros((3,3)))

    def test_stationary_energy_matches_schur_and_retains_true_zero_modes(self):
        for branch,q in (('uncracked_elastic',np.zeros(6)),('restricted_cracked',np.array([-.0002,.0004,.0006,.00004,-.000015,.00002]))):
            out=self.s.evaluate(q,branch=branch)
            energy=self.s._relaxed_energy(q,branch,self.s.kernel.order,np.array(out['zero_incremental_N_membrane_response']))
            np.testing.assert_allclose(energy,out['zero_incremental_N_relaxed_D'],rtol=1e-9,atol=1e-7)
            self.assertFalse(out['relaxed_bending_energy_recovery']['applied'])
        from Design.SMRF_Physical_Floor_Materials import CellVoidProfile
        empty=CellVoidProfile([0,1,0,1],7.5,[],state_identity='empty_material')
        model=LayeredSlabSecant(7.5,5,[],ec_ksi=self.s.ec,concrete_void_profile=empty,quadrature_order=12)
        out=model.evaluate([.001,.001,0,0,0,0],branch='restricted_cracked')
        np.testing.assert_array_equal(out['zero_incremental_N_relaxed_D'],np.zeros((3,3)))
        self.assertTrue(out['constitutive_state_verified'],out['domain_reasons'])

    def test_actual_unreinforced_edge_cancellation_against_high_precision_independent_integral(self):
        import mpmath as mp
        from Design.SMRF_Physical_Floor_Materials import CellVoidProfile
        q=np.array([.00011637994663600447,-1.3107168154228004e-6,7.193391042931358e-5,
                    -1.6651996662997751e-6,1.9880288842241364e-8,3.405618812330491e-6])
        empty=CellVoidProfile([15.,15.78125,-15.,-11.25],7.5,[],state_identity='saved136_edge17_0_gp0')
        model=LayeredSlabSecant(7.5,5.,[],ec_ksi=57*math.sqrt(5000),concrete_void_profile=empty,quadrature_order=12)
        out=model.evaluate(q,branch='restricted_cracked');recovery=out['relaxed_bending_energy_recovery']
        self.assertTrue(recovery['applied']);self.assertTrue(recovery['quadrature']['passed'])
        self.assertFalse(recovery['stiffness_floor_added']);self.assertFalse(recovery['eigenvalue_projection_applied'])
        self.assertTrue(out['constitutive_state_verified'],out['domain_reasons'])
        d=np.asarray(out['zero_incremental_N_relaxed_D']);self.assertGreaterEqual(np.linalg.eigvalsh(d).min(),-1e-9*max(1.,float(np.max(abs(d)))))
        # Independent analytic principal projector, 60-digit integration and
        # ordinary high-precision Schur elimination. No production constitutive
        # routine, quadrature, energy factors, or relaxation matrix is reused.
        with mp.workdps(60):
            qq=list(map(lambda x:mp.mpf(str(x)),q));fc=mp.mpf(5);ec=mp.mpf(str(model.ec));mpa=mp.mpf('6.894757293168')
            n0=mp.mpf('.8')+fc*mpa/17;peak=fc/ec*n0/(n0-1)
            def value(z,i,j,k):
                ex,ey,g=(qq[t]+z*qq[t+3] for t in range(3));mid=(ex+ey)/2;rad=mp.sqrt(((ex-ey)/2)**2+(g/2)**2)
                e1,e2=mid+rad,mid-rad
                if not e2<0<e1:raise AssertionError('Independent fixture has one tensile and one compressive principal mode')
                p=[(e1-ex)/(2*rad),(e1-ey)/(2*rad),-g/(4*rad)]
                effective=fc/max(1,mp.mpf('.8')+mp.mpf('.34')*e1/peak);n=mp.mpf('.8')+effective*mpa/17;r=-e2/peak
                modulus=effective*n*r/(n-1+r**n)/(-e2)
                return z**k*modulus*p[i]*p[j]
            matrices=[]
            for k in range(3):
                matrix=mp.matrix(3)
                for i in range(3):
                    for j in range(i,3):matrix[i,j]=matrix[j,i]=mp.quad(lambda z:value(z,i,j,k),[-mp.mpf('3.75'),0,mp.mpf('3.75')])
                matrices.append(matrix)
            a,b,c=matrices;exact=c-b.T*(a**-1)*b
            target=np.array(exact.tolist(),float)
        np.testing.assert_allclose(d,target,rtol=1e-5,atol=1e-7)

    def test_published_principal_secants_and_harmonic_shear(self):
        peak=.002;ex,ey=-.0004,-.0007
        n=.8+5*6.894757293168/17
        fx=-5*n*(-ex/peak)/(n-1+(-ex/peak)**n)
        fy=-5*n*(-ey/peak)/(n-1+(-ey/peak)**n)
        exsec,eysec=fx/ex,fy/ey
        out=principal_secant_matrix(np.diag([ex,ey]),5,peak_strain=peak,compression_strain_limit=peak)
        expected=np.diag([exsec,eysec,exsec*eysec/(exsec+eysec)])
        np.testing.assert_allclose(out['matrix'],expected,rtol=1e-12,atol=1e-12)
        # Rotate the actual strain. Compare virtual work using tensors, not
        # a copy of the production engineering transformation.
        angle=.37;c,s=math.cos(angle),math.sin(angle);rot=np.array([[c,-s],[s,c]])
        e=rot@np.diag([ex,ey])@rot.T
        got=principal_secant_matrix(e,5,peak_strain=peak,compression_strain_limit=peak)
        sigma=rot@np.diag([fx,fy])@rot.T
        vec=np.array([e[0,0],e[1,1],2*e[0,1]])
        np.testing.assert_allclose(got['matrix']@vec,[sigma[0,0],sigma[1,1],sigma[0,1]],atol=1e-11)
        self.assertAlmostEqual(float(vec@got['matrix']@vec),float(np.sum(e*sigma)),12)

    def test_uncracked_exact_finite_void_ABD_and_actual_steel(self):
        out=self.s.evaluate([0]*6,branch='uncracked_elastic')
        d=self.s.elastic
        A=8*d.copy();B=np.zeros((3,3));D=8**3/12*d.copy()
        for row in mats():
            area=row['area_in2_per_in'];z=row['z_in'];r2=row['bar_area_in2']/math.pi
            A-=area*d;B-=area*z*d;D-=area*(z*z+r2/4)*d
            j=0 if row['axis']=='x' else 1
            A[j,j]+=area*29000;B[j,j]+=area*29000*z;D[j,j]+=area*29000*z*z
        np.testing.assert_allclose(out['A'],A,atol=1e-8,rtol=1e-9)
        np.testing.assert_allclose(out['B'],B,atol=1e-8)
        np.testing.assert_allclose(out['D'],D,atol=1e-7,rtol=1e-9)
        self.assertTrue(out['constitutive_state_verified'],out)
        self.assertFalse(out['rank_deficient']);self.assertEqual(out['poisson'],.2)
        self.assertFalse(out['strength_phi_applied'])

    def test_external_beam_steel_void_removes_only_concrete(self):
        void=dict(id='beam/top/1',axis='x',z_in=.9,area_in2_per_in=1/16,
                  bar_area_in2=1.,db_in=1.128,fy_ksi=60.)
        owned=LayeredSlabSecant(8,5,mats(),ec_ksi=self.s.ec,void_only_layers=[void])
        original=self.s.evaluate([0]*6,branch='uncracked_elastic')
        reduced=owned.evaluate([0]*6,branch='uncracked_elastic')
        a,z,r2=1/16,.9,1/math.pi;d=self.s.elastic
        expected=np.block([[a*d,a*z*d],[a*z*d,a*(z*z+r2/4)*d]])
        np.testing.assert_allclose(np.array(original['ABD'])-reduced['ABD'],expected,atol=1e-7,rtol=1e-9)
        self.assertEqual(reduced['concrete_void_only_layer_ids'],['beam/top/1'])
        self.assertNotIn('beam/top/1',[r['id'] for r in reduced['material_state']['steel']])
        # Nonzero cracked state also excludes that external steel from the
        # independent nominal-response reconstruction, while removing its void.
        cracked=owned.evaluate([-.0001,.0003,.0002,.00001,0,0],branch='restricted_cracked')
        self.assertTrue(cracked['constitutive_state_verified'],cracked)

    def test_all_six_secant_reconstruction_PSD_and_axis_exchange(self):
        q=np.array([-.0002,.0004,.0006,.00004,-.000015,.00002])
        out=self.s.evaluate(q,branch='restricted_cracked')
        self.assertTrue(out['constitutive_state_verified'],out)
        np.testing.assert_allclose(np.array(out['ABD'])@q,out['nominal_resultants'],atol=1e-8)
        self.assertGreaterEqual(min(out['scaled_eigenvalues']),-1e-8)
        swapped=[dict(r,axis='y' if r['axis']=='x' else 'x') for r in mats()]
        sy=LayeredSlabSecant(8,5,swapped,ec_ksi=self.s.ec)
        idx=[1,0,2,4,3,5]
        rotated=sy.evaluate(q[idx],branch='restricted_cracked')
        np.testing.assert_allclose(np.array(out['ABD'])[idx][:,idx],rotated['ABD'],atol=1e-7)
        self.assertFalse(out['force_bound_claimed']);self.assertFalse(out['full_floor_equilibrium_converged'])

    def test_source_secant_twist_is_not_arbitrary_constant(self):
        a=self.s.evaluate([0,0,0,0,0,.00005],branch='restricted_cracked')
        b=self.s.evaluate([.001,.001,0,0,0,0],branch='restricted_cracked')
        self.assertTrue(a['constitutive_state_verified'],a)
        self.assertGreater(a['D'][2][2],0.)
        # Biaxial pure tension: no concrete/interlock, actual orthogonal bars
        # carry no direct shear. The true zero is retained, not regularized.
        self.assertEqual(b['D'][2][2],0.)
        self.assertTrue(b['rank_deficient']);self.assertEqual(b['rank'],4)
        self.assertEqual(b['material_state']['minimum_principal_concrete_shear_secant_ksi'],0.)

    def test_small_strain_zero_N_matches_transformed_cracked_strip(self):
        strip=SlabStripStiffness(8,5,mats(),ec_ksi=self.s.ec)
        p=strip.properties('x',1);z0=p['neutral_axis_z_in'];curvature=1e-9
        out=self.s.evaluate([-curvature*z0,0,0,curvature,0,0],branch='restricted_cracked')
        self.assertTrue(out['constitutive_state_verified'],out)
        self.assertAlmostEqual(out['zero_incremental_N_relaxed_D'][0][0]/self.s.ec,p['Icr_in4_per_in'],places=5)
        # Schur return also gives the membrane strain needed for zero N.
        response=np.array(out['zero_incremental_N_membrane_response'])@[curvature,0,0]
        self.assertAlmostEqual(response[0],-curvature*z0,places=12)

    def test_uncracked_to_cracked_and_material_domain_are_explicit(self):
        valid=self.s.evaluate([1e-6,0,0,0,0,0],branch='uncracked_elastic')
        self.assertTrue(valid['constitutive_state_verified'])
        cracked=self.s.evaluate([.001,0,0,0,0,0],branch='uncracked_elastic')
        self.assertFalse(cracked['constitutive_state_verified'])
        self.assertIn('elastic_predictor_exceeds_published_concrete_cracking_stress',cracked['domain_reasons'])
        failed=self.s.evaluate([-.004,0,0,0,0,0],branch='restricted_cracked')
        self.assertFalse(failed['constitutive_state_verified'])
        self.assertIn('concrete_compression_exceeds_declared_prepeak_limit',failed['domain_reasons'])
        with self.assertRaises(ValueError):self.s.evaluate([0]*6,branch='automatic')

    def test_eigenstrain_breakpoints_are_scale_invariant_at_tiny_twist(self):
        tiny=self.s.evaluate([0,0,0,0,0,-8.470329472543003e-22],branch='restricted_cracked')
        finite=self.s.evaluate([0,0,0,0,0,-1e-12],branch='restricted_cracked')
        self.assertTrue(tiny['constitutive_state_verified'],tiny['domain_reasons'])
        self.assertTrue(tiny['quadrature']['passed'])
        np.testing.assert_allclose(tiny['ABD'],finite['ABD'],rtol=1e-7,atol=1e-7)
        # Actual frozen15 witness: repeated eigenstrain root at z=-3in.
        repeated=self.s.evaluate([0,0,-5.082197683525802e-21,0,0,-1.6940658945086007e-21],branch='restricted_cracked')
        self.assertTrue(repeated['constitutive_state_verified'],repeated['domain_reasons'])


if __name__=='__main__':unittest.main()
