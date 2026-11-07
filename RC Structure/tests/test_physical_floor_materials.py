"""Independent finite-cylinder ownership and frozen-reference witnesses."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.integrate import quad
from Design.SMRF_Physical_Floor_Materials import ClippedBar,CellVoidProfile,FrozenGravityReferenceBuilder,spatial_section_from_inventory,compare_frozen_references
from Design.SMRF_Slab_Secant import LayeredSlabSecant


def bar(ident='x',axis='x',s=0.,z=1.,area=math.pi/4,run=(-4.,7.),owner='slab'):
    return dict(id=ident,axis=axis,s_in=s,z_in=z,area_in2=area,db_in=2*math.sqrt(area/math.pi),
                fy_ksi=60.,es_ksi=29000.,straight_run_interval_in=list(run),operator_owner=owner)


class PhysicalFloorMaterialTests(unittest.TestCase):
    def test_actual_circle_limits_preserve_full_volume_at_translated_elevations(self):
        # Actual127 y-mat geometry exposed an endpoint roundoff loss despite
        # the entire circle being within the slab and the plan footprint.
        # Closed cylinder formulas are independent of the arc primitive.
        for z in (-2.25,2.25,-2.75,2.75,.1,-.1):
            for axis in ('x','y'):
                actual=bar('actual',axis,s=853.,z=z,area=.2,run=(-9.375,1017.375))
                out=ClippedBar(actual,(-15,1030,-15,1030),8).record()
                volume=.2*1026.75;along=504.;x,y=(along,853.) if axis=='x' else (853.,along)
                target=volume*np.array([1.,x,y,z,z*z+.2/(4*math.pi)])
                keys=('straight_volume_in3','first_x_in4','first_y_in4','first_z_in4','second_z_in5')
                np.testing.assert_allclose([out[k] for k in keys],target,rtol=2e-14,atol=1e-11)

    def test_full_and_half_circle_exact_volume_first_moments(self):
        b=bar();p=ClippedBar(b,(-2,3,-2,2),8);area=b['area_in2'];r=.5
        np.testing.assert_allclose(p.moments,[5*area,5*area,5*area*(1+r*r/4)],atol=1e-13)
        self.assertAlmostEqual(p.first_longitudinal,5*area*.5,12)
        self.assertAlmostEqual(p.first_transverse,0,12)
        half=ClippedBar(b,(-2,3,0,2),8)
        self.assertAlmostEqual(half.moments[0],5*area/2,12)
        self.assertAlmostEqual(half.first_transverse,5*2*r**3/3,12)
        by=bar(axis='y');rotated=ClippedBar(by,(0,2,-2,3),8).record()
        self.assertAlmostEqual(rotated['first_x_in4'],half.first_transverse,12)
        self.assertAlmostEqual(rotated['first_y_in4'],half.first_longitudinal,12)

    def test_partial_circle_and_run_match_independent_adaptive_quadrature(self):
        b=bar(s=.2,z=3.8,run=(.7,2.3));p=ClippedBar(b,(0,4,.1,.55),8)
        r=.5;length=1.6
        def edges(z):
            dr=math.sqrt(max(0,r*r-(z-3.8)**2))
            return max(.1,.2-dr),min(.55,.2+dr)
        def width(z):
            lo,hi=edges(z);return max(0,hi-lo)
        # Adaptive integration uses circle intersections independently of
        # production analytic antiderivatives and includes slab-top clipping.
        expected=[length*quad(lambda z,i=i:width(z)*z**i,3.3,4,epsabs=1e-11,points=p.breaks)[0] for i in range(3)]
        np.testing.assert_allclose(p.moments,expected,atol=2e-9,rtol=2e-9)
        first_s=length*quad(lambda z:(max(0,edges(z)[1]-edges(z)[0])*(edges(z)[0]+edges(z)[1])/2),
                            3.3,4,epsabs=1e-11,points=p.breaks)[0]
        self.assertAlmostEqual(p.first_transverse,first_s,9)
        self.assertAlmostEqual(p.first_longitudinal,p.moments[0]*1.5,12)
        outside=ClippedBar(b,(5,6,.1,.55),8)
        np.testing.assert_array_equal(outside.moments,[0,0,0])

    def test_irregular_cell_partition_preserves_all_material_moments(self):
        bars=[bar('x1',s=.1,z=2.8,run=(-1,5.4)),bar('y1','y',s=2.1,z=1.5,run=(-.6,2.9)),
              bar('beam','x',s=1.3,z=-3.9,run=(-.2,4.2),owner='web_line')]
        xs=[-1,.2,2.1,3.7,6];ys=[-1,.1,1.8,4]
        whole=CellVoidProfile((xs[0],xs[-1],ys[0],ys[-1]),8,bars,state_identity='state')
        keys=['straight_volume_in3','first_x_in4','first_y_in4','first_z_in4','second_z_in5']
        sums={b['id']:np.zeros(5) for b in bars}
        for x0,x1 in zip(xs[:-1],xs[1:]):
            for y0,y1 in zip(ys[:-1],ys[1:]):
                cell=CellVoidProfile((x0,x1,y0,y1),8,bars,state_identity='state')
                for row in cell.records:sums[row['id']]+=[row[k] for k in keys]
                self.assertNotIn('beam',[v['id'] for v in cell.slab_steel_layers()])
        for row in whole.records:np.testing.assert_allclose(sums[row['id']],[row[k] for k in keys],atol=1e-11)

    def test_custom_void_elastic_ABD_uses_exact_clipped_volume(self):
        bars=[bar('x',s=.1,z=2.8,area=.31,run=(.2,3.8)),
              bar('y','y',s=2.1,z=1.5,area=.31,run=(.4,3.6)),
              bar('beam','x',s=1.3,z=-3.9,run=(.1,3.4),owner='web_line')]
        profile=CellVoidProfile((0,4,0,4),8,bars,state_identity='state')
        sec=LayeredSlabSecant(8,5,profile.slab_steel_layers(),ec_ksi=4000,
                             concrete_void_profile=profile,quadrature_order=24)
        got=sec.evaluate(np.zeros(6),branch='uncracked_elastic')
        a,b,d=np.array([8,0,8**3/12])-profile.exact_depth_moments/profile.area
        expected=np.block([[a*sec.elastic,b*sec.elastic],[b*sec.elastic,d*sec.elastic]])
        for row in profile.slab_steel_layers():
            axis=0 if row['axis']=='x' else 1
            v=np.zeros(6);v[axis]=1;v[axis+3]=row['z_in']
            expected+=row['area_in2_per_in']*29000*np.outer(v,v)
        np.testing.assert_allclose(got['ABD'],expected,atol=2e-6,rtol=2e-9)
        self.assertTrue(got['constitutive_state_verified'],got['domain_reasons'])
        self.assertEqual(got['concrete_void_profile_identity'],profile.identity)

    def test_synthetic_existing_shell_reference_maps_all_four_signed_GPs(self):
        # Existing operator arrays, no OpenSees/native calls or floor solve.
        raw=np.zeros((4,8,24));scales=[1.,.8,1.1,.9]
        for gp,scale in enumerate(scales):
            raw[gp,:6,:6]=np.diag([1,1,1,-1,-1,-1])*scale
        element=dict(kind='slab',id='floor1:cell:0:0',points=np.array([[0,0,0],[4,0,0],[4,4,0],[0,4,0]]),
                     native={'strain':raw},transform=np.eye(24),indices=np.arange(24))
        floor=SimpleNamespace(id='floor1',x=[0,4],y=[0,4],thickness=8.,ec=4000.,nu=.2,
                              elements=[element],dof_ids=list(range(24)))
        bars=[bar('xtop',s=1,z=2.8,area=.31,run=(0,4)),bar('xbottom',s=2,z=-2.8,area=.31,run=(0,4)),
              bar('ytop','y',s=1,z=2,area=.31,run=(0,4)),bar('ybottom','y',s=2,z=-2,area=.31,run=(0,4))]
        inv=dict(state_identity='state',identity_sha256='synthetic_inventory',floors={'1':{'bars':bars}},
                 hook_stiffness_basis='none in synthetic witness')
        builder=FrozenGravityReferenceBuilder(inv,reference_identity='D+L_reference',quadrature_order=16)
        row=dict(assembly=floor,description={'level':1,'fc_ksi':5.})
        q=np.array([-.0001,-.00008,.00002,.000012,-.000003,.000001]);u=np.r_[q,np.zeros(18)]
        out=builder.build(row,{'retained_displacements':u,'case_id':'D+L'},reference_loading={'D':1.,'L':1.})
        self.assertEqual(out['status'],'reference_constructed',out['audit'])
        self.assertEqual(out['slab_bending_matrices'][element['id']].shape,(4,3,3))
        for gp,point in enumerate(out['audit']['cells'][0]['reference_points']):
            np.testing.assert_allclose(point['reference_generalized_strain'],q*scales[gp])
            self.assertGreater(np.linalg.eigvalsh(point['D']).min(),0)
        self.assertFalse(out['audit']['final_combination_nonlinear_equilibrium_required'])
        self.assertFalse(out['audit']['targeted_coupled_sensitivity_verified'])
        self.assertTrue(out['audit']['physical_material_ownership_verified'])
        self.assertTrue(all(p['constitutive_branch']=='uncracked_elastic' for p in out['audit']['cells'][0]['reference_points']))
        cracked_u=np.r_[[0,0,0,.00005,0,0],np.zeros(18)]
        cracked=builder.build(row,{'retained_displacements':cracked_u,'case_id':'reference_cracked'},reference_loading='cracking_witness')
        mask=cracked['audit']['cracked_point_ids'];self.assertEqual(len(mask),4)
        next_builder=FrozenGravityReferenceBuilder(inv,reference_identity='updated_D+L',prior_cracked_point_ids=mask)
        unloaded=next_builder.build(row,{'retained_displacements':np.zeros(24),'case_id':'zero'},reference_loading='history_witness')
        self.assertTrue(all(p['constitutive_branch']=='restricted_cracked' for p in unloaded['audit']['cells'][0]['reference_points']))

    def test_paired_reference_keeps_compression_hidden_below_rigid_translation_ulp(self):
        # An independently defined constant-strain four-node patch. Its
        # compression lives wholly below the ULP of the rigid translation.
        points=np.array([[0.,0.,0.],[4.,0.,0.],[4.,4.,0.],[0.,4.,0.]])
        raw=np.zeros((4,8,24))
        for gp in range(4):
            raw[gp,0,[0,6,12,18]]=[-1/8,1/8,1/8,-1/8]
            raw[gp,1,[1,7,13,19]]=[-1/8,-1/8,1/8,1/8]
            raw[gp,2,[1,7,13,19]]=[-1/8,1/8,1/8,-1/8]
            raw[gp,2,[0,6,12,18]]=[-1/8,-1/8,1/8,1/8]
        element=dict(kind='slab',id='patch',points=points,native={'strain':raw},transform=np.eye(24),indices=np.arange(24))
        floor=SimpleNamespace(id='floor1',x=[0,4],y=[0,4],thickness=8.,ec=4000.,nu=.2,elements=[element],
            node_ids=list(range(4)),node_positions=dict(enumerate(points.tolist())),dof_ids=[(n,d) for n in range(4) for d in range(1,7)])
        bars=[bar('x',s=1,z=2.8,area=.31,run=(0,4)),bar('y','y',s=1,z=-2.8,area=.31,run=(0,4))]
        inv=dict(state_identity='state',identity_sha256='paired_patch',floors={'1':{'bars':bars}},hook_stiffness_basis='none in numerical patch')
        high=np.zeros((4,6));low=np.zeros_like(high);high[:,0]=2.**40;high[:,1]=-2.**39
        low[:,0]=-1e-7*points[:,0];low[:,1]=-1e-7*points[:,1]
        self.assertTrue(np.array_equal(high+low,high))
        builder=FrozenGravityReferenceBuilder(inv,reference_identity='paired_compression',quadrature_order=16)
        out=builder.build(dict(assembly=floor,description={'level':1,'fc_ksi':5.}),
            dict(retained_displacements=high.ravel(),retained_displacements_lo=low.ravel(),case_id='compression'),reference_loading='same_actual_compression')
        self.assertEqual(out['status'],'reference_constructed');self.assertTrue(out['audit']['paired_reference_displacements_used'])
        for gp in out['audit']['cells'][0]['reference_points']:
            np.testing.assert_allclose(gp['reference_generalized_strain'],[-1e-7,-1e-7,0,0,0,0],rtol=1e-12,atol=1e-20)

    def test_invalid_or_duplicate_material_identity_refused(self):
        b=bar()
        with self.assertRaises(ValueError):CellVoidProfile((0,1,0,1),8,[b,b],state_identity='state')
        with self.assertRaises(ValueError):CellVoidProfile((0,1,0,1),8,[dict(b,operator_owner='unknown')],state_identity='state')
        with self.assertRaises(ValueError):ClippedBar(dict(b,area_in2=-1),(0,1,0,1),8)
        with self.assertRaises(ValueError):ClippedBar(dict(b,straight_run_interval_in=[2,1]),(0,1,0,1),8)
        empty=LayeredSlabSecant(8,5,[],ec_ksi=4000,
            concrete_void_profile=CellVoidProfile((0,1,0,1),8,[],state_identity='state'))
        self.assertTrue(empty.evaluate(np.zeros(6),branch='uncracked_elastic')['constitutive_state_verified'])

    def test_narrow_actual_cell_can_have_exact_all_steel_depth_slices(self):
        profile=CellVoidProfile((0,4,-.05,.05),8,[bar(z=1,run=(0,4))],state_identity='state')
        self.assertEqual(float(profile.occupied(np.array([1.]))[0]),1.)
        sec=LayeredSlabSecant(8,5,profile.slab_steel_layers(),ec_ksi=4000,concrete_void_profile=profile)
        out=sec.evaluate([-.00005,0,0,.00001,0,0],branch='restricted_cracked')
        self.assertTrue(out['constitutive_state_verified'],out['domain_reasons'])

    def test_actual_spatial_inventory_has_integer_rows_and_no_development_assertion(self):
        bars=[bar('xtop',s=1,z=2.8,area=.31,run=(0,10)),bar('xbottom',s=2,z=-2.8,area=.31,run=(0,10)),
              bar('ytop1','y',s=4,z=2,area=.31,run=(0,8)),bar('ytop2','y',s=6,z=2,area=.31,run=(0,8))]
        inv=dict(state_identity='actual',identity_sha256='inventory',slab_thickness_in=8.,floors={'1':{'bars':bars}})
        args=dict(floor=1,axis='x',station_in=5,interval_in=[0,4],axial_accounting_interval_in=[3,7],
                  fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65,'diaphragm':.6})
        out=spatial_section_from_inventory(inv,**args)
        self.assertEqual(out['status'],'inventory_constructed',out['issues'])
        self.assertEqual(out['development']['status'],'unknown')
        self.assertEqual(out['constructor']['reference_s_in'],2)
        row=out['constructor']['transverse_rows'][0]
        self.assertEqual(row['physical_bar_ids'],['ytop1','ytop2'])
        self.assertEqual(row['accounting_length_in'],4)
        self.assertEqual(out['section'].phi,.6)
        evidence=[dict(status='pass',evidence_id='separate_actual_anchorage',station_interval_in=[0,8],
                       state_identity='actual',material_inventory_id='inventory',
                       floor=1,axis='x',interval_in=[0,4],axial_accounting_interval_in=[3,7],
                       physical_bar_ids=out['physical_bar_ids'])]
        qualified=spatial_section_from_inventory(inv,**args,development_evidence=evidence)
        self.assertEqual(qualified['development']['status'],'pass')
        self.assertFalse(qualified['complete_design_qualified'])
        self.assertEqual(qualified['identity_sha256'],out['identity_sha256'])
        failed=spatial_section_from_inventory(inv,**args,development_evidence=[dict(evidence[0],status='fail')])
        self.assertEqual(failed['development']['status'],'fail')
        other=spatial_section_from_inventory(inv,**args,development_evidence=[dict(evidence[0],state_identity='other')])
        self.assertEqual(other['development']['status'],'unknown')

    def test_partial_longitudinal_or_transverse_boundary_refused(self):
        inv=dict(state_identity='actual',identity_sha256='inventory',slab_thickness_in=8.,
                 floors={'1':{'bars':[bar('long',s=1,z=2.8,run=(0,10)),bar('cross','y',s=4,z=2,run=(0,8))]}})
        args=dict(floor=1,axis='x',station_in=5,interval_in=[1,4],axial_accounting_interval_in=[3,7],
                  fc_ksi=5,ec_ksi=4000,phi_requirements={'NM':.65})
        out=spatial_section_from_inventory(inv,**args)
        self.assertIsNone(out['section']);self.assertEqual(out['status'],'unknown')
        self.assertEqual(out['issues'][0]['reason'],'longitudinal_circle_crosses_strip_boundary')
        out=spatial_section_from_inventory(inv,**dict(args,interval_in=[0,4],axial_accounting_interval_in=[4,7]))
        self.assertEqual(out['issues'][0]['reason'],'transverse_circle_crosses_axial_accounting_boundary')


if __name__=='__main__':unittest.main()
