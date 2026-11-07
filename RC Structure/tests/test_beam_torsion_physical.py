"""ACI22.7 independent algebra/physical reservation oracles; no solves."""
import copy,math,sys,unittest
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model.Member_Groups import MemberDesign
from Design import SMRF_Beam_Torsion as tor

D=MemberDesign("beam_x",18.,16.,5.,8,4,4,0,5,3,3.)
POINTS=[dict(id=f"{face}_0_{i}",face=face,source_face=face,row=0,index=i,x_in=x,y_in=y)
        for face,y in (("top",5.),("bottom",-5.)) for i,x in enumerate((-6.,-2.,2.,6.))]
SECTION=tor.solid_section([[-9.,9.,-8.,8.]],5.,section_id="explicit_18x16")
IDENTITY={"case_id":"synthetic_C06","cut_id":"middle","state_identity":"written_fixture","recovery_id":"algebra"}
SHEAR={a:{"vu_kip":0.,"capacity_ve_kip":0.,"vc_kip":0.,"d_in":13.,"width_in":18.} for a in ("vertical","inplane")}
LONG={"status":"pass",**IDENTITY,"strength_phi":.65,"all_concurrent_demands_included":True,
      "bars":[{"id":p["id"],"area_in2":.79,"fy_ksi":60.,"nominal_stress_ksi":20.,"strength_credited":True} for p in POINTS]}
DEV={"status":"pass","source_check_ids":["fixture_actual_full_span_and_end_anchorage"],
     "longitudinal_bar_ids":[p["id"] for p in POINTS],"longitudinal_extends_required_zone":True,
     "transverse_extends_required_zone":True,"both_supports_developed":True}


def assess(T=100.,N=0.,**changes):
    kwargs=dict(design=D,coordinates=POINTS,spacing_in=3.,clear_cover_in=1.5,shear_reservations=SHEAR,
                longitudinal_state=LONG,development=DEV,identity=IDENTITY)
    kwargs.update(changes)
    return tor.assess_torsion(SECTION,T,N,**kwargs)


class TorsionThreshold(unittest.TestCase):
    def test_rectangle_and_T_union_have_exact_area_and_perimeter(self):
        self.assertEqual((SECTION["Acp_in2"],SECTION["pcp_in"]),(288.,68.))
        t=tor.solid_section([[-5,5,-20,0],[-15,15,-5,0]],4.)
        self.assertEqual((t["Acp_in2"],t["pcp_in"]),(300.,100.))
        self.assertEqual(t["Ag_in2"],300.)

    def test_torsion_flange_uses_projection_and_four_slab_thicknesses(self):
        t=tor.beam_torsion_section(10.,40.,6.,{"minus":0.,"plus":100.},5.)
        self.assertEqual(t["overhangs_in"],{"minus":0.,"plus":24.})
        self.assertEqual(t["Acp_in2"],400.+24*6)
        self.assertEqual(t["pcp_in"],2*(10+40)+2*24)
        slender=tor.beam_torsion_section(4.,48.,1.,{"minus":30.,"plus":30.},5.)
        self.assertTrue(slender["flanges_omitted"])
        self.assertEqual(slender["Acp_in2"],192.)

    def test_axial_threshold_and_strict_equality(self):
        base=math.sqrt(5000)*288**2/68/1000
        tension=tor.threshold(SECTION,0.,40.)
        compression=tor.threshold(SECTION,0.,-40.)
        self.assertAlmostEqual(tension["Tth_kip_in"],base*math.sqrt(1-40000/(4*288*math.sqrt(5000))),12)
        self.assertAlmostEqual(compression["Tth_kip_in"],base*math.sqrt(1+40000/(4*288*math.sqrt(5000))),12)
        self.assertFalse(tor.threshold(SECTION,.75*base,0.)["below_threshold"])
        self.assertTrue(tor.threshold(SECTION,.75*base-1e-5,0.)["below_threshold"])

    def test_root_fc_cap_tension_domain_and_missing_N(self):
        high={**SECTION,"fc_ksi":16.}
        self.assertEqual(tor.threshold(high,0.,0.)["sqrt_fc_used_psi"],100.)
        self.assertEqual(tor.threshold(SECTION,1.,1000.)["status"],"not_evaluated")
        self.assertEqual(tor.threshold(SECTION,1.,None)["status"],"not_evaluated")

    def test_disconnected_and_hollow_do_not_get_solid_formula(self):
        self.assertEqual(tor.solid_section([[0,1,0,1],[2,3,0,1]],4.)["status"],"fail")
        hollow=tor.solid_section([[0,3,0,1],[0,3,2,3],[0,1,1,2],[2,3,1,2]],4.)
        self.assertEqual(hollow["status"],"not_evaluated")

    def test_below_threshold_needs_no_fabricated_torsion_capacity(self):
        out=assess(T=20.,longitudinal_state=None,development=None,shear_reservations=None)
        self.assertEqual((out["status"],out["branch"]),("pass","below_threshold"))
        self.assertNotIn("geometry",out)


class ProductionTorsionBinding(unittest.TestCase):
    def test_saved_torsion_requires_exact_input_state_and_payload_binding(self):
        from Design.Grouped_Record import _bound_common_torsion,_sha256
        from unittest.mock import patch
        package={'source_id':'explicit_recovery','state_identity':'grouped_identity','records':[{'Tu':12.}]}
        record={'member_groups':{'sha256':'grouped_identity'},'common_torsion':package,
            'common_torsion_input_binding':{'analysis_input_sha256':'input_identity','source_id':'explicit_recovery',
                'state_identity':'grouped_identity','common_torsion_sha256':_sha256(package)}}
        with patch('Design.SMRF_Design_Evidence.analysis_input_signature',return_value='input_identity'):
            self.assertEqual(_bound_common_torsion(record),package)
            reloaded=__import__('json').loads(__import__('json').dumps(record))
            self.assertEqual(_bound_common_torsion(reloaded),package)
            reloaded['common_torsion']['records'][0]['Tu']=0.
            with self.assertRaisesRegex(ValueError,'not bound'):_bound_common_torsion(reloaded)
            self.assertIsNone(_bound_common_torsion({}))

    @staticmethod
    def package(state):
        from Design.SMRF_Beam_Torsion_Adapter import actual_member_torsion_section
        tags=[t for g in state.groups.values() if g['member_type']!='column' for t in g['member_tags']]
        records=[]
        for tag in tags:
            section=actual_member_torsion_section(state.member(tag),4.)
            for case in ('Ex_plus','Ex_minus'):
                records.append(dict(member_tag=tag,case_id=case,cut_id='declared_face_i',state_identity=state.identity(),
                    recovery_id='synthetic_equilibrium_'+case,status='pass',section_id=section['section_id'],
                    same_section_axial_recovered=True,N_tension_kip=0.,T_about_positive_axis_kipin=0.,
                    Q_vertical_kip=0.,V_inplane_kip=0.))
        return dict(state_identity=state.identity(),source_id='explicit_zero_torsion_synthetic_fixture',
                    expected_case_ids=['Ex_plus','Ex_minus'],cut_inventory_complete=True,
                    expected_cuts_by_member={str(t):['declared_face_i'] for t in tags},slab_fc_ksi=4.,records=records)

    def test_legacy_absent_torsion_is_unknown_not_zero(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design.Group_Capacity import beam_torsion_evidence
        with gf.frame():
            state=mg.install(gf.uniform_state())
            out=beam_torsion_evidence(state,['Ex_plus','Ex_minus'],{})
            self.assertEqual(out['status'],'not_evaluated')
            self.assertEqual(len(out['checks']),8)
            self.assertTrue(all(c['id']=='beam.torsion_concurrent' and c['status']=='not_evaluated' for c in out['checks']))

    def test_complete_actual_threshold_cases_and_missing_duplicate_or_wrong_section(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design.Group_Capacity import beam_torsion_evidence
        with gf.frame():
            state=mg.install(gf.uniform_state());p=self.package(state)
            self.assertEqual(beam_torsion_evidence(state,p['expected_case_ids'],{},p)['status'],'pass')
            absent=copy.deepcopy(p);absent['records'].pop()
            duplicate=copy.deepcopy(p);duplicate['records'].append(copy.deepcopy(p['records'][0]))
            wrong=copy.deepcopy(p);wrong['records'][0]['section_id']='whole_flexural_strip_is_not_same_torsion_N'
            malformed=copy.deepcopy(p);malformed['expected_cuts_by_member'][str(p['records'][0]['member_tag'])]=None
            for bad in (absent,duplicate,wrong,malformed):
                self.assertEqual(beam_torsion_evidence(state,p['expected_case_ids'],{},bad)['status'],'not_evaluated')

    def test_adapter_reserves_full_capacity_shear_even_when_caller_supplies_zero(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design.Group_Capacity import beam_torsion_evidence
        from unittest.mock import patch
        with gf.frame():
            state=mg.install(gf.uniform_state());p=self.package(state)
            capacity={}
            for gid,g in state.groups.items():
                if g['member_type']=='column':continue
                capacity[gid]={'effective_depth_in':20.,'vc_kip':0.,
                    'members':[{'member_tag':t,'governing_ve_kip':200.} for t in g['member_tags']]}
            for record in p['records']:record['shear_reservations']=copy.deepcopy(SHEAR)
            with patch('Design.SMRF_Beam_Torsion_Adapter.assess_torsion',return_value={'status':'pass'}) as spy:
                beam_torsion_evidence(state,p['expected_case_ids'],capacity,p)
            self.assertTrue(spy.call_args_list)
            self.assertTrue(all(call.kwargs['shear_reservations']['vertical']['capacity_ve_kip']==200. for call in spy.call_args_list))


class ActualTorsionReserve(unittest.TestCase):
    def test_actual_closed_hoop_and_per_bar_reserves_pass_known_fixture(self):
        out=assess()
        self.assertEqual(out["status"],"pass",[(c["rule"],c["details"]) for c in out["checks"] if not c["passes"]])
        g=out["geometry"]
        self.assertAlmostEqual(g["Ao_in2"],.85*g["Aoh_in2"],12)
        self.assertGreater(g["ph_upper_in"],g["ph_plan_in"])
        detail=next(c["details"] for c in out["checks"] if c["rule"].startswith("torsion_longitudinal_reserve"))
        self.assertAlmostEqual(sum(r["required_nominal_tension_kip"] for r in detail["per_bar"]),detail["required_additional_nominal_tension_kip"],10)
        self.assertTrue(all(abs(r["available_nominal_tension_kip"]-.79*40)<1e-9 for r in detail["per_bar"]))
        transverse=next(c["details"] for c in out["checks"] if c["rule"].startswith("torsion_transverse_reserve"))
        self.assertAlmostEqual(transverse["phi_Tn_kip_in"],.75*2*g["Ao_in2"]*.31*g["hoop_projection_factor"]*60/3,10)

    def test_same_steel_cannot_be_spent_twice_after_yielding_for_N_M(self):
        state=copy.deepcopy(LONG)
        for b in state["bars"]:b["nominal_stress_ksi"]=60.
        out=assess(longitudinal_state=state)
        row=next(c for c in out["checks"] if c["rule"].startswith("torsion_longitudinal_reserve"))
        self.assertEqual(out["status"],"fail")
        self.assertTrue(all(r["available_nominal_tension_kip"]==0. for r in row["details"]["per_bar"]))

    def test_full_Mpr_shear_is_reserved_even_when_actual_shear_is_zero(self):
        shear=copy.deepcopy(SHEAR)
        shear["vertical"]["capacity_ve_kip"]=200.
        out=assess(shear_reservations=shear)
        self.assertEqual(out["transverse_reservations"]["vertical"]["vu_used_kip"],200.)
        self.assertEqual(out["status"],"fail")
        self.assertFalse(next(c["passes"] for c in out["checks"] if c["rule"].startswith("torsion_transverse_reserve")))

    def test_missing_or_wrong_concurrent_state_is_unknown(self):
        for state in (None,{**LONG,"case_id":"different"},{**LONG,"all_concurrent_demands_included":False}):
            with self.subTest(state=state):
                self.assertEqual(assess(longitudinal_state=state)["status"],"not_evaluated")

    def test_missing_zone_development_is_unknown(self):
        self.assertEqual(assess(development=None)["status"],"not_evaluated")

    def test_undeveloped_zero_stress_bar_is_not_free_torsion_reserve(self):
        state=copy.deepcopy(LONG)
        state['bars'][0].update(strength_credited=False,nominal_stress_ksi=0.)
        self.assertEqual(assess(longitudinal_state=state)['status'],'not_evaluated')

    def test_open_or_invalid_physical_cage_is_computed_failure(self):
        d=replace(D,stirrup_legs=2)
        self.assertEqual(assess(design=d)["status"],"fail")

    def test_torsion_sign_does_not_change_area_reservation(self):
        plus,minus=assess(),assess(T=-100.)
        self.assertEqual(plus["checks"],minus["checks"])

    def test_interior_crosstie_addition_does_not_raise_pure_torsion_capacity(self):
        out=assess()
        self.assertAlmostEqual(out["geometry"]["outer_single_leg_area_in2"],.31*out["geometry"]["hoop_projection_factor"],12)
        self.assertLess(out["geometry"]["outer_single_leg_area_in2"],D.stirrup_legs*.31/2)


if __name__=="__main__":unittest.main()
