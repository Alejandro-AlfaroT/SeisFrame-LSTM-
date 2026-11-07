"""Independent algebra/geometry oracles, protocol C-20261103-01; no solves."""
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model.Member_Groups import MemberDesign
from Design import SMRF_Transitions as tr
from Design import SMRF_Transition_Physical as ph
from Design.SMRF_Coupler_Catalog import lookup_coupler

D = MemberDesign("column",30.,30.,5.,10,4,4,2,4,4,4.,(4,4))
KW=dict(fy_ksi=60.,clear_cover_in=1.5,upper_clear_height_in=140.,joint_depth_in=28.)
CTX=dict(beam_segments=[],beam_geometry_complete=True,upper_hoop_spacing_in=6.,
         transverse_fy_ksi=60.,aggregate_max_size_in=.75,longitudinal_specification="ASTM A706 Grade 60",axial_max_kip=0.)


class CornerGeometry(unittest.TestCase):
    def test_rigid_fabrication_orientations_preserve_real_owners_curvature_and_steel(self):
        import itertools
        coordinates=[{**p,'id':f"{p['face']}_{p['index']}"} for p in tr.bar_coordinates(D,1.5)]
        original=ph.rectangular_cage_geometry(D,coordinates,1.5)
        self.assertTrue(original['passes'])
        for signs in itertools.product((False,True),repeat=4):
            actual=ph.oriented_cage(original,*signs)
            self.assertIsNotNone(actual)
            self.assertEqual(actual['total_length_in'],original['total_length_in'])
            self.assertEqual(actual['core_area_in2'],original['core_area_in2'])
            self.assertEqual(actual['inside_bend_radius_in'],original['inside_bend_radius_in'])
            self.assertTrue(ph.transverse_bar_connectivity(actual['segments'])['passes'])
            self.assertFalse(ph.transverse_collisions(actual['segments'],actual['segments'],same_set=True))
            for bar in coordinates:
                self.assertEqual(ph.endpoint_engagement(actual,[bar['x_in'],bar['y_in']],1.27,bar['id']),
                                 ph.endpoint_engagement(original,[bar['x_in'],bar['y_in']],1.27,bar['id']))
            for old,new in zip(original['segments'],actual['segments']):
                self.assertAlmostEqual(math.dist(old['start_in'],old['end_in']),math.dist(new['start_in'],new['end_in']),12)

    def test_old_sharp_corner_overlaps_actual_hoop_bend(self):
        old=1.5+.5+.5
        bend_center=1.5+.5+1.
        self.assertGreater(math.sqrt(2)*(bend_center-old)+.5,1.)
        new=sp.first_bar_offset_in(1.5,.5,1.,grid_in=None)
        self.assertAlmostEqual(new,3.-.5/math.sqrt(2),12)
        self.assertAlmostEqual(math.sqrt(2)*(bend_center-new)+.5,1.,12)

    def test_grid_is_conservative_and_hoop_selection_does_not_move_cage(self):
        self.assertEqual(sp.first_bar_offset_in(1.5,.5,.625,grid_in=.5),3.)
        self.assertEqual(sp.first_bar_offset_in(1.5,.625,.625,grid_in=.5),3.)
        self.assertEqual(sp.first_bar_offset_in(1.5,.5,1.27,grid_in=.5),3.)

    def test_larger_transverse_bend_and_large_bar_envelope(self):
        for dt,db in ((.75,1.),(.5,2.257)):
            radius=max((2 if dt<=.625 else 3)*dt,db/2)
            first=sp.first_bar_offset_in(1.5,dt,db,grid_in=None)
            center=1.5+dt+radius
            self.assertLessEqual(math.sqrt(2)*(center-first)+db/2,radius+1e-12)

    def test_nonfinite_and_undeclared_transverse_geometry_rejected(self):
        for args in ((float('nan'),.5,1.),(1.5,-.5,1.),(1.5,1.128,1.)):
            with self.assertRaises(ValueError):
                sp.first_bar_offset_in(*args)


class FiniteBends(unittest.TestCase):
    def test_no14_general_roof_development_has_actual_cb_and_no_invented_stress_factor(self):
        from Design.SMRF_Roof_Anchorage import straight_roof_development_bound
        design=replace(D,b_in=36.,h_in=36.,fc_ksi=10.,bar_size=14)
        out=straight_roof_development_bound(design,extension_in=36.,joint_depth_in=36.,fy_ksi=60.,clear_cover_in=1.5)
        db=1.693
        expected=3/40*60000/100/min(3.5/db,2.5)*db
        self.assertAlmostEqual(out['cb_in'],3.5,12)
        self.assertAlmostEqual(out['ld_in'],expected,12)
        self.assertEqual(out['ktr_in'],0.)
        self.assertEqual(out['stress_ksi'],60.)
        self.assertEqual(out['status'],'pass')
        short=straight_roof_development_bound(design,extension_in=10.,joint_depth_in=16.,fy_ksi=60.,clear_cover_in=1.5)
        self.assertEqual(short['status'],'fail')

    def test_hand_derived_two_arc_offset_height(self):
        # No.8 R=(6db+db)/2=3.5; a 2 in offset at theta=atan(1/6).
        theta=math.atan(1/6)
        length=(2-7*(1-math.cos(theta)))/math.sin(theta)
        expected=7*math.sin(theta)+length*math.cos(theta)
        bend=ph.offset_bend_geometry(2.,8)
        self.assertAlmostEqual(bend["height_in"],expected,11)
        self.assertGreater(expected,12.)
        self.assertAlmostEqual(bend["slope"],1/6,12)

    def test_common_height_preserves_tangent_endpoint_and_smaller_slope(self):
        bend=ph.offset_bend_geometry(.2,10,20.)
        r,t,L=bend["radius_in"],bend["angle_rad"],bend["straight_in"]
        self.assertAlmostEqual(2*r*(1-math.cos(t))+L*math.sin(t),.2,11)
        self.assertAlmostEqual(2*r*math.sin(t)+L*math.cos(t),20.,11)
        self.assertLess(math.tan(t),1/6)

    def test_grade80_bend_ties_do_not_cancel_grade60_transverse_strength(self):
        a=tr.column_transition(D,replace(D,b_in=28.,h_in=28.),**KW)
        b=tr.column_transition(D,replace(D,b_in=28.,h_in=28.),**{**KW,"fy_ksi":80.})
        for axis in ("x","y"):
            self.assertAlmostEqual(b["offset_bend"]["tie_area_required_in2"][axis]/a["offset_bend"]["tie_area_required_in2"][axis],4/3,12)

    def test_segment_distance_independent_crossing_parallel_and_point_oracles(self):
        self.assertEqual(ph.segment_distance([0,0,0],[2,0,0],[1,-1,0],[1,1,0]),0.)
        self.assertEqual(ph.segment_distance([0,0,0],[2,0,0],[0,3,0],[2,3,0]),3.)
        self.assertEqual(ph.segment_distance([0,0,0],[0,0,0],[3,4,0],[3,4,1]),5.)


class ProductAndLapEvidence(unittest.TestCase):
    def test_published_transition_dimensions_and_type2_grade(self):
        a=lookup_coupler(10,9,80.,"ASTM A706 Grade 80")
        self.assertEqual((a["status"],a["part_number"],a["length_in"],a["envelope_diameter_in"]),("pass","EL3228A2",4.125,1.75))
        self.assertEqual(lookup_coupler(10,9,80.,"ASTM A706 Grade 80",True)["status"],"fail")

    def test_hex_envelope_is_circumscribed_not_across_flats(self):
        a=lookup_coupler(6,5,60.,"ASTM A706 Grade 60")
        self.assertAlmostEqual(a["envelope_diameter_in"],1.0625/math.cos(math.pi/6),12)

    def test_unpublished_pair_is_unknown_not_fabricated_product(self):
        a=lookup_coupler(10,8,60.,"ASTM A706 Grade 60")
        self.assertEqual(a["status"],"not_evaluated")
        self.assertNotIn("part_number",a)

    def test_contact_lap_pair_is_one_group_but_adjacent_spacing_is_real(self):
        paths=[dict(path="lap_spliced",target_xy_in=[0,0],upper_xy_in=[1,0]),
               dict(path="straight",target_xy_in=[4,0],upper_xy_in=[4,0])]
        a=ph.lap_group_spacing(paths,1.,1.)
        self.assertTrue(a["passes"])
        paths[1]["upper_xy_in"]=[3,0]
        self.assertFalse(ph.lap_group_spacing(paths,1.,1.)["passes"])


class IntegratedPhysicalStatus(unittest.TestCase):
    def test_missing_actual_beam_context_is_unknown(self):
        a=tr.column_transition(D,D,**KW)
        self.assertTrue(a["supported"])
        self.assertFalse(a["physically_verified"])
        self.assertEqual(a["physical_detail"]["status"],"not_evaluated")

    def test_identical_straight_cage_needs_no_artificial_offset(self):
        a=tr.column_transition(D,D,**KW,physical_context=CTX)
        self.assertEqual(a["path_counts"]["straight"],12)
        self.assertEqual(a["bars_bent"],0)
        self.assertEqual(a["physical_detail"]["status"],"pass",a["physical_detail"])

    def test_real_beam_column_collision_is_computed_failure(self):
        bar=tr.bar_coordinates(D,1.5)[0]
        beam=dict(id="beam",start_in=[-20,bar["y_in"],-10],end_in=[20,bar["y_in"],-10],diameter_in=1.)
        a=tr.column_transition(D,D,**KW,physical_context={**CTX,"beam_segments":[beam]})
        self.assertFalse(a["supported"])
        self.assertIn("column_paths_clear_beam_bars",a["physical_detail"]["failure_rules"])

    def test_reduced_section_uses_other_continuity_without_unknown_approval(self):
        lower=replace(D,top_bars=3,bot_bars=3,side_bars=1,stirrup_bar_size=5,stirrup_legs=3,
                      stirrup_legs_by_direction=(3,3),stirrup_spacing_in=3.)
        a=tr.column_transition(lower,replace(lower,b_in=28.,h_in=28.),**KW,
                               physical_context={**CTX,"upper_hoop_spacing_in":3.})
        self.assertFalse(a["column_reinforcement_continuous"])
        self.assertEqual(a["continuity_branch"],"other_conservative_reduced_column")
        self.assertEqual(a["physical_detail"]["status"],"pass",a["physical_detail"].get("failure_rules"))

    def test_reduction_with_insufficient_actual_core_confinement_fails(self):
        lower=replace(D,top_bars=3,bot_bars=3,side_bars=1,stirrup_legs=3,stirrup_legs_by_direction=(3,3))
        a=tr.column_transition(lower,replace(lower,b_in=28.,h_in=28.),**KW,physical_context=CTX)
        self.assertIn("actual_joint_core_confinement",a["physical_detail"]["failure_rules"])


class FabricationAndSplices(unittest.TestCase):
    def test_winding_is_connected_and_has_finite_3d_radius(self):
        cage=ph.rectangular_cage_geometry(D,tr.bar_coordinates(D,1.5),1.5)
        check=next(c for c in cage["checks"] if c["rule"]=="continuous_winding_3d_bend_radius")
        self.assertEqual(check["pitch_end_slopes"],[0.,0.])
        self.assertGreaterEqual(check["minimum_centerline_radius_bound_in"],1.25)
        self.assertTrue(ph.transverse_bar_connectivity(cage["segments"])["passes"])
        self.assertGreater(cage["total_length_in"],sum(math.dist(s["start_in"],s["end_in"]) for s in cage["segments"]))

    def test_unrelated_ties_cannot_claim_same_owner_exemption(self):
        a={**ph._tube([-2,0,0],[2,0,0],.5,"a",["same_bar"]),"physical_bar_id":"a"}
        b={**ph._tube([0,-2,0],[0,2,0],.5,"b",["same_bar"]),"physical_bar_id":"b"}
        self.assertTrue(ph.transverse_collisions([a],[b]))

    def test_finite_cylinder_distance_independent_radial_and_axial_oracles(self):
        vertical=ph._tube([1.2,0,-1],[1.2,0,1],.5,"tube")
        self.assertAlmostEqual(ph.segment_cylinder_clearance(vertical,[0,0],[-2,2],1.),-.05,9)
        # Bounding-box separation may return infinity (also conservative
        # for collision classification); force the exact branch at contact.
        touching=ph._tube([-2,0,2.2],[2,0,2.2],.5,"tube")
        self.assertLess(ph.segment_cylinder_clearance(touching,[0,0],[-2,2],1.),0.)

    def test_documented_column_device_and_local_cage_pass(self):
        a=ph.verify_mechanical_splice(D,fy_ksi=60.,clear_cover_in=1.5,clear_height_in=140.,spacing_in=4.,specification="ASTM A706 Grade 60")
        self.assertEqual(a["status"],"pass",a["checks"])
        self.assertEqual(a["product"]["part_number"],"EL32A2")
        self.assertEqual(len(a["bodies"]),12)
        ratios=a["transverse_detail"]["ratios"]
        for provided,original in zip(ratios["provided_Av_per_in"],ratios["original_Av_per_in"]):
            self.assertGreaterEqual(provided,original)
        self.assertAlmostEqual(a["net_envelope_steel_volume_in3"],12*a["product"]["length_in"]*(math.pi*1.75**2/4-1.27),10)

    def test_unpublished_column_device_is_unknown(self):
        a=ph.verify_mechanical_splice(replace(D,bar_size=8),lower_bar_size=10,fy_ksi=60.,clear_cover_in=1.5,
                                     clear_height_in=140.,spacing_in=4.,specification="ASTM A706 Grade 60")
        self.assertEqual(a["status"],"not_evaluated")
        self.assertFalse(a["bodies"])


class RoofHeads(unittest.TestCase):
    def test_declared_roof_extension_keeps_straight_bars_and_prices_real_volume(self):
        from Design.SMRF_Roof_Anchorage import verify_declared_roof
        d=replace(D,stirrup_bar_size=5,stirrup_spacing_in=3.)
        a=verify_declared_roof(d,extension_in=30.,joint_depth_in=28.,fy_ksi=60.,clear_cover_in=1.5,spacing_in=3.,
            specification="ASTM A706 Grade 60",physical_context=CTX,axial_max_kip=0.)
        self.assertEqual(a["status"],"pass",a.get("failure_rules"))
        self.assertEqual(a["termination_type"],"straight_bar_at_declared_extension_top")
        self.assertEqual(a["extension_concrete_in3"],27000.)
        self.assertAlmostEqual(a["extension_longitudinal_steel_in3"],12*1.27*(30.-1.5-1.27/2),10)
        self.assertGreater(len(a["extension_transverse_sets"]),1)
        self.assertNotIn("product",a)

    def test_current_published_head_and_independent_development(self):
        from Design.SMRF_Roof_Anchorage import head_product,headed_development
        product=head_product(9,60.,"ASTM A706 Grade 60")
        self.assertEqual((product["part_number"],product["diameter_in"],product["length_in"]),("EL28D6",2.75,1.5))
        d=replace(D,bar_size=9)
        evidence=headed_development(d,tr.bar_coordinates(d,1.5),60.)
        expected=75000*1.25*(5000/15000+.6)/(75*math.sqrt(5000))*1.128**1.5
        self.assertAlmostEqual(evidence["tension_ldt_in"],expected,10)
        self.assertEqual(evidence["psi_p"],1.)
        self.assertAlmostEqual(evidence["compression_ldc_in"],.0003*60000*1.128,10)

    def test_roof_has_real_heads_and_owned_downward_cages(self):
        from Design.SMRF_Roof_Anchorage import verify_roof_anchorage
        d=replace(D,bar_size=9)
        a=verify_roof_anchorage(d,joint_depth_in=28.,fy_ksi=60.,clear_cover_in=1.5,spacing_in=4.,
            specification="ASTM A706 Grade 60",physical_context=CTX,axial_max_kip=0.)
        self.assertEqual(a["status"],"pass",a.get("failure_rules"))
        self.assertEqual(len(a["heads"]),12)
        self.assertTrue(all(s["winding"]=="downward" for s in a["tie_sets"]))
        self.assertGreater(a["development"]["provided_in"],a["development"]["compression_ldc_in"])

    def test_head_cover_failure_does_not_get_product_unknown(self):
        from Design.SMRF_Roof_Anchorage import verify_roof_anchorage
        a=verify_roof_anchorage(D,joint_depth_in=28.,fy_ksi=60.,clear_cover_in=1.5,spacing_in=4.,
            specification="ASTM A706 Grade 60",physical_context=CTX,axial_max_kip=0.)
        self.assertEqual(a["product"]["status"],"pass")
        self.assertEqual(a["status"],"fail")
        self.assertIn("roof_head_code_conditions",a["failure_rules"])

    def test_report_extra_design_requirements_remain_unknown(self):
        from Design.SMRF_Roof_Anchorage import head_product
        self.assertEqual(head_product(9,80.,"ASTM A706 Grade 80")["status"],"not_evaluated")
        self.assertEqual(head_product(14,60.,"ASTM A706 Grade 60")["status"],"not_evaluated")


class ProductionInventory(unittest.TestCase):
    def test_every_member_is_inventoried_without_solves_or_duplicate_quantity_owners(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Group_Capacity as gc,Grouped_Design as gd
        from unittest.mock import patch
        import openseespy.opensees as ops
        with patch.object(ops,"analyze",side_effect=AssertionError("No solves in C protocol")),gf.frame():
            state=mg.install(gf.uniform_state())
            splice=gc.group_splices(state)
            actual=[c["member_tag"] for s in splice.values() for c in s["member_contexts"]]
            expected=[tag for group in state.groups.values() for tag in group["member_tags"]]
            self.assertEqual(sorted(actual),sorted(expected))
            self.assertEqual(len(actual),len(set(actual)))
            q=gd.quantities(state,capacity={"splices":splice})
            self.assertTrue(q["physical_quantity_owners_unique"])
            expected_volume=sum(c["physical_detail"].get("net_envelope_steel_volume_in3",0.) for s in splice.values() for c in s["member_contexts"])
            self.assertAlmostEqual(q["splice_device_net_envelope_steel_in3"],expected_volume,8)

    def test_required_inventory_names_each_physical_joint_and_group(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Group_Capacity as gc,Code_Check_Contract as cc
        with gf.frame():
            state=mg.install(gf.uniform_state())
            expected=cc.expected_inventory(state)["exact"]
            for gid in state.groups:
                self.assertIn(("capacity","detailing.splice_physical_detail",gid),expected)
                if not state.designs[gid].is_column:
                    self.assertIn(("capacity","beam.slab_physical_inventory",gid),expected)
            for name,kind in gc.joint_inventory(state)["types"].items():
                self.assertIn(("capacity","beam.joint_physical_detail",name),expected)
                row="column.roof_anchorage_physical_detail" if kind["representative"]["above"] is None else "column.transition_physical_detail"
                self.assertIn(("capacity",row,name),expected)

    def test_beam_device_uses_actual_rows_and_preserves_shear_steel(self):
        from Design.SMRF_Splice_Physical import verify_beam_mechanical_splice
        design=MemberDesign("beam_x",30.,28.,5.,8,4,4,0,4,4,4.)
        points=[dict(id=f"{f}_0_{i}",face=f,source_face=f,row=0,index=i,x_in=x,y_in=y)
                for f,y in (("top",11.),("bottom",-11.)) for i,x in enumerate((-9.,-3.,3.,9.))]
        a=verify_beam_mechanical_splice(design,points,clear_span_in=240.,spacing_in=4.,fy_ksi=60.,
                                      specification="ASTM A706 Grade 60",clear_cover_in=1.5)
        self.assertEqual(a["status"],"pass",a["checks"])
        self.assertEqual(len(a["bodies"]),8)
        self.assertGreaterEqual(a["transverse_detail"]["provided_Av_per_in"],.2)
        for body in a["bodies"]:
            self.assertGreater(body["z_interval_in"][0],56.)
            self.assertLess(body["z_interval_in"][1],184.)

    def test_short_beam_cannot_hide_device_inside_hinge_zone(self):
        from Design.SMRF_Splice_Physical import verify_beam_mechanical_splice
        design=MemberDesign("beam_x",30.,28.,5.,8,4,4,0,4,4,4.)
        points=[dict(id=f"{f}_0_{i}",face=f,source_face=f,row=0,index=i,x_in=x,y_in=y)
                for f,y in (("top",11.),("bottom",-11.)) for i,x in enumerate((-9.,-3.,3.,9.))]
        a=verify_beam_mechanical_splice(design,points,clear_span_in=110.,spacing_in=4.,fy_ksi=60.,
                                      specification="ASTM A706 Grade 60",clear_cover_in=1.5)
        self.assertEqual(a["status"],"fail")
        self.assertFalse(a["bodies"])


class ProjectedHoopArea(unittest.TestCase):
    def test_actual_beam_selection_cannot_credit_unprojected_winding(self):
        from Design.SMRF_Actual_Hoop_Selection import beam_selection
        from Design import SMRF_Capacity_Design as cd
        d=MemberDesign("beam_x",30.,28.,5.,8,4,4,0,4,4,4.)
        points=[dict(id=f"{f}_0_{i}",face=f,source_face=f,row=0,index=i,x_in=x,y_in=y)
                for f,y in (("top",11.),("bottom",-11.)) for i,x in enumerate((-9.,-3.,3.,9.))]
        nominal=cd.select_beam_hoops(30.,25.,5.,60.,8,1.5,4,4,45.,45.)
        result=beam_selection(d,points,1.5,{**nominal,"effective_depth_in":25.})
        self.assertIsNotNone(result["hoops"],result["cage"])
        h=result["hoops"]
        self.assertLess(h["av_in2"],h["nominal_av_in2"])
        self.assertAlmostEqual(h["av_in2"],(h["legs"]-2+2*h["hoop_leg_projection_factor"])*sp.rebar_area(h["bar_size"]),12)
        self.assertGreaterEqual(h["phi_vn_kip"],45.)

    def test_actual_column_core_and_directional_projection_independent_oracle(self):
        from Design.SMRF_Actual_Hoop_Selection import column_selection
        stories={axis:{1:{"axial_max_kip":0.,"vs_required_kip":25.,"vc_kip":0.,"ve_kip":18.75}} for axis in ("x","y")}
        nominal={"stories":stories,"worst_by_direction":{a:v[1] for a,v in stories.items()},
                 "directions":{a:{"depth_in":30.,"width_in":30.,"legs_key":key} for a,key in
                               (("x","across_b_face"),("y","across_h_face"))},
                 "spacing_bounds_in":{},"cage":{},"confinement":{},"vs_limit_by_direction_kip":{"x":150.,"y":150.}}
        r=column_selection(D,1.5,nominal,3.)
        self.assertIsNotNone(r["hoops"],r["cage"])
        h,c=r["hoops"],r["confinement"]
        self.assertLess(c["ach_in2"],math.prod(c["bc_by_direction_in"].values()))
        ratio=max(.3*(900./c["ach_in2"]-1)*5/60,.09*5/60)
        self.assertAlmostEqual(c["ash_ratio_required"],ratio,12)
        for axis,v in h["by_direction"].items():
            self.assertAlmostEqual(v["av_in2"],(v["legs"]-2+2*c["hoop_leg_projection_factor"])*sp.rebar_area(h["bar_size"]),12)
            self.assertGreaterEqual(v["av_in2"]/h["spacing_in"],ratio*c["bc_by_direction_in"][axis])
        self.assertTrue(r["screen"]["all_pass"])


class ActualMatInventory(unittest.TestCase):
    def test_corner_hook_families_clear_fixed_mats_and_preserve_actual_straight_runs(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Group_Capacity as gc,SMRF_Transition_Context as context
        column=MemberDesign('column',30.,30.,5.,10,4,4,2,5,4,3.,(4,4))
        beam=MemberDesign('beam_x',30.,28.,5.,8,4,4,0,5,4,3.)
        with gf.frame(column=column,beam=beam,FY_KSI=60.,REINFORCEMENT_SPECIFICATION='ASTM A706 Grade 60'):
            state=mg.install(gf.uniform_state(column,beam))
            sp.SLAB_REINFORCEMENT={'layout':self.layout()}
            entry=next(e for e in gc.joint_inventory(state)['joints'] if e['id']=='joint_13')
            preliminary=tr.column_transition(column,column,**KW)
            detail=context.beam_context(entry,preliminary)
            self.assertEqual(detail['status'],'pass',detail['beam_geometry_checks'])
            self.assertTrue(all(f['available_ldh_in']>=f['required_ldh_in'] for f in detail['beam_hook_families']))
            # Fixed geometrical oracle: bottom X family moves eight inches
            # inward and still has19.815in of hook development (required13.054).
            member=entry['beams']['x'][0]
            family=next(f for f in detail['beam_hook_families'] if f['member_tag']==member.member_tag and f['face']=='bottom')
            self.assertEqual(family['stagger_in'],8.)
            runs=context.installed_beam_straight_intervals(state,member.member_tag)
            self.assertEqual(runs['status'],'pass',runs['end_evidence'])
            bar=next(b for b in runs['bars'] if b['source_lane_id']=='bottom_0_0')
            self.assertEqual(bar['id'],f'member_{member.member_tag}/beam/bottom_0_0')
            self.assertAlmostEqual(bar['straight_run_interval_in'][0],family['tangent_axial_local_in'],12)
            self.assertEqual(bar['straight_run_interval_in'][1],gf.BAY_X)

    @staticmethod
    def layout():
        return {"bar_size":4,"outer_axis":"x","layers":{
            f"{axis}_{face}":{"bar_size":4,"bar_area_in2":.2,"spacing_in":6.,
                "effective_depth_in":6. if face=="top" else 1.,"axis":axis,
                "layer":"outer" if axis=="x" else "inner","clear_cover_outer_mat_in":.75}
            for axis in ("x","y") for face in ("top","bottom")}}

    def test_floor_specific_outer_face_counts_hooks_and_all_joint_collisions(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg,Member_Properties as mp
        from Model.Floor_Footprint import physical_floor
        from Design import SMRF_Joint_Mat_Physical as mats,Group_Capacity as gc
        with gf.frame():
            sp.SLAB_REINFORCEMENT={"layout":self.layout()}
            state=mg.install(gf.uniform_state())
            placement=mats.global_mat_placement(state)
            self.assertEqual(placement["status"],"pass")
            for floor,record in placement["floors"].items():
                self.assertEqual(record["footprint"],physical_floor(int(floor),state))
                for mat in record["layers"].values():
                    lo,hi=mat["transverse_interval_in"]
                    phase,spacing=mat["phase_in"],mat["spacing_in"]
                    count=math.floor((hi-phase)/spacing)-math.ceil((lo-phase)/spacing)+1
                    self.assertEqual(len(mat["global_transverse_coordinates_in"]),count)
                    straight=mat["straight_run_interval_in"]
                    length=straight[1]-straight[0]+math.pi*mat["hook_radius_in"]+2*mat["hook_extension_in"]
                    self.assertAlmostEqual(mat["bar_cut_length_in"],length,10)
            for kind in gc.joint_inventory(state)["types"].values():
                detail=mats.joint_mat_segments(kind["representative"],placement)
                self.assertEqual(detail["status"],"pass",detail["mat_collisions"])
            q=mats.mat_quantities(placement)
            hand=sum(len(mat["global_transverse_coordinates_in"])*mat["bar_cut_length_in"]*.2
                     for floor in placement["floors"].values() for mat in floor["layers"].values())
            self.assertAlmostEqual(q["steel_volume_in3"],hand,8)
            member=mg.beam_at("beam_x",1,0,1)
            flange=mp.effective_flange(member)
            inv=mats.member_slab_inventory(member,flange,placement)
            family=mp.beam_strengths(member)["family"]
            self.assertEqual(inv["status"],"pass")
            self.assertAlmostEqual(family["slab_steel_in_flange_in2"],len(inv["bars"])*.2,12)
            self.assertTrue(all(inv["effective_flange_interval_in"][0]<=b["global_transverse_coordinate_in"]<=inv["effective_flange_interval_in"][1] for b in inv["bars"]))

    def test_actual_bottom_hook_stagger_reduces_both_datums(self):
        from Design.SMRF_Beam_Slab_Strength import perimeter_slab_bar_anchorage
        layout=self.layout()
        top=perimeter_slab_bar_anchorage(layout,"x",18.,1.5,.5,4.,60.)
        joint=perimeter_slab_bar_anchorage(layout,"x",18.,1.5,.5,4.,60.,embedment_available_in=20.)
        self.assertAlmostEqual(top["layers"][0]["embedment_available_in"],16.)
        self.assertAlmostEqual(top["layers"][1]["embedment_available_in"],14.5)
        self.assertAlmostEqual(joint["layers"][1]["embedment_available_in"],18.5)
        self.assertEqual(top["governing"]["face"],"bottom")

    def test_joint_types_do_not_merge_opposite_hooks_or_different_global_mat_phase(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Design import Group_Capacity as gc
        with gf.frame():
            layout=self.layout()
            for layer in layout["layers"].values():
                layer["spacing_in"]=10.
            sp.SLAB_REINFORCEMENT={"layout":layout}
            state=mg.install(gf.uniform_state())
            inventory=gc.joint_inventory(state)
            entry={tuple((e["level"],e["grid_i"],e["grid_j"])):e for e in inventory["joints"]}
            owner={joint:name for name,kind in inventory["types"].items() for joint in kind["joints"]}
            self.assertNotEqual(owner[entry[(1,0,0)]["id"]],owner[entry[(1,2,0)]["id"]])
            # Same left-edge incident groups, but 216 mod10=6 versus
            # 432 mod10=2: the local mat cylinders differ.
            self.assertNotEqual(owner[entry[(1,0,1)]["id"]],owner[entry[(1,0,2)]["id"]])

    def test_unknown_inventory_has_no_end_sizing_credit_but_upper_capacity_demand(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg,Member_Properties as mp
        from unittest.mock import patch
        with gf.frame():
            sp.SLAB_REINFORCEMENT={"layout":self.layout()}
            mg.install(gf.uniform_state())
            member=mg.beam_at("beam_x",1,0,1)
            with patch("Design.SMRF_Joint_Mat_Physical.global_mat_placement",return_value={"status":"not_evaluated","floors":{}}):
                strength=mp.beam_strengths(member)
            family=strength["family"]
            self.assertEqual(family["slab_physical_inventory"]["status"],"not_evaluated")
            self.assertGreater(strength["hogging"],family["mn_undeveloped_negative_kip_in"])
            self.assertEqual(strength["basis"]["sizing_hogging_i_kip_in"],family["mn_undeveloped_negative_kip_in"])

    def test_column_volume_subtracts_only_actual_floor_overlap(self):
        from tests import grouped_fixture as gf
        from Model import Member_Groups as mg
        from Model.Floor_Footprint import physical_floor,column_overlap_area_in2
        from Design import Grouped_Design as gd
        with gf.frame():
            state=mg.install(gf.uniform_state())
            q=gd.quantities(state)
            for gid,group in state.groups.items():
                d=state.designs[gid]
                if not d.is_column:
                    continue
                hand=sum(d.b_in*d.h_in*gf.STORY_H-column_overlap_area_in2(physical_floor(m.story_or_floor,state),
                    m.grid_i*gf.BAY_X,m.grid_j*gf.BAY_Y,d.h_in,d.b_in)*gf.SLAB for m in (state.member(tag) for tag in group["member_tags"]))
                self.assertAlmostEqual(q["groups"][gid]["concrete_in3"],hand,8)


def setUpModule():
    # These tests verify the 3D physical detail route, which the computed common-action route keeps (2026-11-07).
    global _DETAIL_BASIS
    _DETAIL_BASIS = sp.DETAIL_BASIS
    sp.DETAIL_BASIS = "physical_3d_v2"


def tearDownModule():
    sp.DETAIL_BASIS = _DETAIL_BASIS


if __name__=="__main__":
    unittest.main()
