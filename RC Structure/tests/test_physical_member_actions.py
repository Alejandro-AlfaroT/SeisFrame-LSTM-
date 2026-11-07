"""Independent section-force transport and consumer witnesses; no native solve."""
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from tests.test_physical_floor_cuts import shell,beam
from Design.SMRF_Physical_Member_Actions import (METHOD,full_flange_gravity_field,member_from_polynomials,
    recover_polynomial_span,validated_member_span,add_primary_increment,column_member_action,physical_face_reactions,_section_values,PreparedMemberResponses)


def gravity_field(axis='x',faces=(1.,1.)):
    j=0 if axis=='x' else 1;c=np.zeros((4,8));c[0,j]=1.;c[0,j+3]=4.;c[0,2]=3.;c[0,5]=6.;c[0,6+j]=7.
    slab=shell('s',0.,10.,-5.,5.,c) if axis=='x' else shell('s',-5.,5.,0.,10.,c)
    response=dict(elements=[slab,beam(axis,n=7,q=-2,length=10,m0=0,v0=10,z=-6)],force_partition_relative_error=0.)
    return full_flange_gravity_field(response,member_id='beam',axis=axis,start_in=0.,end_in=10.,beam_line_in=0.,
        depth_in=12.,slab_thickness_in=2.,flange_interval_in=[-2.,2.],face_offsets_in=faces,
        uniform_web_loads={'beam:0':-2.},state_identity='state',source_identity='source',definition_sha256='definition',gravity_case_id='G')


class PhysicalMemberTests(unittest.TestCase):
    def test_prepared_recovery_matches_raw_and_fresh_reader_detects_change(self):
        c=np.zeros((4,8));c[0,0]=1.;c[0,3]=4.
        response=dict(elements=[shell('s',0.,10.,-5.,5.,c),beam(n=7,q=-2,length=10,m0=0,v0=10,z=-6)],force_partition_relative_error=0.)
        result=dict(definition_sha256='definition',source_identity='source',floors=[dict(level=1,responses=[response])])
        prepared=PreparedMemberResponses(result)
        args=dict(member_id='beam',axis='x',start_in=0.,end_in=10.,beam_line_in=0.,depth_in=12.,slab_thickness_in=2.,
            flange_interval_in=[-2.,2.],face_offsets_in=(1.,1.),uniform_web_loads={'beam:0':-2.},
            state_identity='state',source_identity='source',definition_sha256='definition',gravity_case_id='G')
        self.assertEqual(full_flange_gravity_field(response,**args),full_flange_gravity_field(response,**args,prepared_responses=prepared))
        with self.assertRaisesRegex(ValueError,'cross physical'):prepared.validate(copy.deepcopy(result))
        with self.assertRaisesRegex(ValueError,'not prepared'):prepared.digest(copy.deepcopy(response))
        # A fresh readback hashes the current raw bytes, even if somebody
        # retained and mutated an in-memory producer object after preparation.
        response['elements'][0]['raw_resultants'][0][0]+=1.
        fresh=PreparedMemberResponses(result)
        self.assertNotEqual(fresh.payload_sha256,prepared.payload_sha256)
        self.assertNotEqual(fresh.digest(response),prepared.digest(response))

    def test_full_flange_N_M_and_shear_retain_physical_reference_and_face_wrenches(self):
        for axis in ('x','y'):
            f=gravity_field(axis);span=recover_polynomial_span(f);p=f['pieces'][1]
            # width4 slab plus disjoint web: N=4+7, Mz0=16-10x+x²-42;
            # beam midpoint z=-5 gives Msag=-Mz0-5N=-x²+10x-29.
            for s in (.1,.37,.9):
                x=10*s
                self.assertAlmostEqual(np.polynomial.polynomial.polyval(s,p['P_kip']),-11.)
                self.assertAlmostEqual(np.polynomial.polynomial.polyval(s,p['M_kipin']),-x*x+10*x-29.)
                self.assertAlmostEqual(np.polynomial.polynomial.polyval(s,p['Vz_kip']),-18-2*x)
            row=member_from_polynomials(f);self.assertAlmostEqual(row['local_force_kip_kipin'][4],0.,places=10)
            self.assertAlmostEqual(row['local_force_kip_kipin'][10],56.,places=10)
            self.assertAlmostEqual(span['clear_span']['mu_negative_kip_in'],20.)
            self.assertFalse(span['physical_trace_qualification_inferred'])

    def test_torsion_is_the_actual_positive_face_cross_product_with_reference_transport(self):
        for axis,sign in (('x',1.),('y',-1.)):
            field=gravity_field(axis);row=member_from_polynomials(field)
            # Symmetric Q first moment vanishes; slab Mxy=24 and Nxy=12.
            # Torque at z=-5: positive-face cut T=-sign*(24+60).
            # The local i resisting moment has opposite sign.
            self.assertAlmostEqual(row['local_force_kip_kipin'][3],sign*84.,places=10)
            self.assertAlmostEqual(row['local_force_kip_kipin'][1],-sign*12.,places=10)

    def test_frozen_independent_native_slab_face_wrenches_have_distinct_web_Q_sign(self):
        # C19 native20x30x8 patch, fileSHA76bc3292e9c9e2825a9a5a1009897622
        # a4368336bfeb32422a9a23ccb860bd39. Values below are the independent
        # native RIGHT-face nodal wrench, not this consumer's sign formula.
        for axis,fz,twist in (('x',.08333333333333334,1.0666666666666682),
                              ('y',.05555555555555555,-.711111111111111)):
            j=0 if axis=='x' else 1;sign=1 if j==0 else -1
            integral=np.zeros(8);first=np.zeros(8);integral[6+j]=fz
            integral[5]=-sign*twist;first[6+j]=10*fz
            web=dict(V_inplane_kip=0.,Q_vertical_kip=2.,T_about_positive_axis_kipin=0.,
                transverse_reference_in=0.,centroid_z_in=-6.)
            cut=dict(slab=dict(axis=axis,integral=integral.tolist(),transverse_first_moment=first.tolist()),
                web=web,resultants=[0.]*6,Ms_transverse_first_moment_kipin=0.,transverse_reference_in=0.)
            actual=_section_values(cut,beam_midplane_z_in=0.,beam_line_in=0.)
            self.assertAlmostEqual(actual['Vz_kip'],2-fz,places=12)
            # Move the native face vertical resultant10in from its transverse
            # reference; r cross F adds +10Fz about X and -10Fz about Y.
            self.assertAlmostEqual(actual['T_kipin'],-twist-sign*10*fz,places=12)

    def test_unloaded_primary_increment_is_added_once_at_actual_beam_midplane(self):
        field=gravity_field();increment=[2.,4.,3.,5.,11.,7.,-2.,-4.,-3.,-5.,-41.,33.]
        actual=add_primary_increment(field,increment,increment_identity='exact_primary_increment')
        old=member_from_polynomials(field);new=member_from_polynomials(actual)
        expected=[2.,4.,3.,-15.,21.,7.,-2.,-4.,-3.,15.,-51.,33.]
        np.testing.assert_allclose(np.array(new['local_force_kip_kipin'])-old['local_force_kip_kipin'],expected,atol=1e-10)
        self.assertFalse(actual['external_diaphragm_branch_added'])
        bad=increment.copy();bad[8]+=1.
        with self.assertRaisesRegex(ValueError,'equilibrated'):add_primary_increment(field,bad,increment_identity='bad')

    def test_actual_column_keeps_both_concurrent_axial_ends_and_joint_face_weight(self):
        g=[100.,1.,2.,3.,4.,5.,-80.,-1.,-2.,-3.,6.,7.];e=[-10.,2.,3.,4.,5.,6.,10.,-2.,-3.,-4.,7.,8.]
        row=column_member_action(g,increment_local_force=e,face_offsets_in=[2.,3.],axial_weight_kip_per_in=2.)
        np.testing.assert_allclose(row['local_force_kip_kipin'],np.array(g)+e)
        self.assertEqual([row['centerline_axial_i_kip'],row['centerline_axial_j_kip']],[90.,70.])
        self.assertEqual([row['axial_i_kip'],row['axial_j_kip']],[86.,76.])

    def test_interior_axial_and_shear_peaks_reach_selection_and_capacity_consumers(self):
        field=gravity_field(faces=(0.,0.));template=field['pieces'][0];pieces=[]
        for a,b,p,v in ((0.,.4,[0.,25.],[0.,50.]),(.4,1.,[50/3,-50/3],[100/3,-100/3])):
            r=copy.deepcopy(template);r.update(start_fraction=a,end_fraction=b,P_kip=p,Vz_kip=v);pieces.append(r)
        field['pieces']=pieces;member=member_from_polynomials(field)
        from Design.Group_Checks import beam_demand_rows
        from Design.SMRF_Capacity_Design import beam_axial_envelope
        case=dict(id='G',analysis_succeeded=True,members={'1':member})
        row=beam_demand_rows([1],[case])[0]
        self.assertAlmostEqual(row[-1],10.);self.assertAlmostEqual(row[4],np.hypot(20.,12.))
        cap=beam_axial_envelope([('strength',case)],[1],2.,2.,5.,'g',expected_ids=['G'])
        self.assertEqual(cap['status'],'evaluated');self.assertAlmostEqual(cap['pu_max_kip'],10.)
        self.assertEqual(cap['governing_compression']['end'],'span_max')
        broken=copy.deepcopy(member);broken['span_bending']['full_span']['axial_max_kip']=0.
        with self.assertRaisesRegex(ValueError,'replay'):validated_member_span(broken)

    def test_capacity_gravity_release_preserves_original_clear_span_load_once(self):
        f=gravity_field();f['pieces']=[dict(start_fraction=0.,end_fraction=1.,P_kip=[0.,0.],
            M_kipin=[4.,130.,-100.],Vz_kip=[13.,-20.],Vy_kip=[0.,0.],T_kipin=[0.,0.],Mz_kipin=[0.,0.])]
        # Face moments differ by24; releasing them adds -3 to V, leaving
        # clear-span simply supported reactions8/8 under2kip/in for8in.
        member=member_from_polynomials(f);case=dict(id='G',dead=1.,live=0.,ex=0.,ey=0.,branch='physical_G_only',analysis_succeeded=True,
            state_identity='state',source_identity='source',physical_definition_sha256='definition',members={'1':member})
        transfer=dict(physical_gravity_member_actions=[case],expected_physical_gravity_case_ids=['G'],
            state_identity='state',source_identity='source',physical_definition_sha256='definition')
        with patch('Model.Member_Groups.active',return_value=SimpleNamespace(identity=lambda:'state')):
            reactions,basis=physical_face_reactions(transfer,SimpleNamespace(member_tag=1),1.,0.)
        np.testing.assert_allclose(reactions,[8.,8.],atol=1e-12)
        self.assertIn('included once',basis)

    def test_physical_gravity_family_collector_rejects_changed_same_family_fields(self):
        from Design.SMRF_Story_Adapter import collect_physical_gravity_cases
        base=dict(method=METHOD,id='G',dead=1.,live=0.,ex=0.,ey=0.,branch='physical_G_only',state_identity='s',source_identity='src',
            physical_definition_sha256='d',members={'1':member_from_polynomials(gravity_field())})
        pairs=[]
        for ident in ('Eplus','Eminus'):
            pairs.append(dict(action=dict(method=METHOD,id=ident,dead=1.,live=0.),gravity_action={**copy.deepcopy(base),'represented_design_combination_ids':[ident]}))
        cases=collect_physical_gravity_cases(pairs,['Eplus','Eminus']);self.assertEqual(len(cases),1)
        self.assertEqual(cases[0]['represented_design_combination_ids'],['Eplus','Eminus'])
        pairs[1]['gravity_action']['members']['1']['axial_i_kip']+=.1
        with self.assertRaisesRegex(ValueError,'differing'):collect_physical_gravity_cases(pairs,['Eplus','Eminus'])


if __name__=='__main__':unittest.main()
