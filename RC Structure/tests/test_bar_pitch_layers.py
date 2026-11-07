import copy,math,unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from Design.SMRF_Bar_Pitch_Layers import BarPitchLayer,installed_mat_layers

def fixture(axis='x'):
    bars=[dict(id='b'+str(i),axis=axis,operator_owner='slab',mat_name=axis+'_bottom',s_in=s,z_in=-2.,
        area_in2=.2,db_in=.5,fy_ksi=60.,es_ksi=29000.,straight_run_interval_in=[2.,28.]) for i,s in enumerate((5.,11.,17.,23.))]
    fp=dict(x_min_in=0.,x_max_in=30.,y_min_in=0.,y_max_in=30.)
    return dict(identity_sha256='explicit_mat_fixture',state_identity='installed',slab_thickness_in=8.,floors={'1':dict(bars=bars,footprint=fp)})

class BarPitchLayerTests(unittest.TestCase):
    def test_point_field_keeps_run_ends_gaps_and_one_sided_layer_ownership(self):
        from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
        for axis in ('x','y'):
            field=FixedPitchFloorMaterial(fixture(axis),1)
            def at(along,across,trace=('plus','plus')):
                return field.point_profile(*( (along,across) if axis=='x' else (across,along)),trace=trace)
            self.assertEqual(at(1.,5.).slab_steel_layers(),[])
            self.assertEqual(at(10.,1.).slab_steel_layers(),[])
            self.assertEqual(at(28.,5.).slab_steel_layers(),[])
            self.assertEqual(len(at(2.,5.).slab_steel_layers()),1)
            ids=lambda p:[r['id'] for r in p.slab_steel_layers()]
            self.assertEqual(ids(at(10.,8.)),['b1'])
            self.assertEqual(ids(at(10.,8.,('minus','minus'))),['b0'])
            for point in (at(10.,7.999),at(10.,8.),at(10.,8.001)):
                self.assertAlmostEqual(sum(r['area_in2_per_in'] for r in point.slab_steel_layers()),.2/6)
            self.assertEqual(ids(at(28.,5.,('minus','minus'))),['b0'])
            self.assertNotEqual(at(10.,8.).identity,at(10.,8.,('minus','minus')).identity)
            with self.assertRaisesRegex(ValueError,'outside'):field.point_profile(-1.,0.)
            with self.assertRaisesRegex(ValueError,'traces'):field.point_profile(1.,1.,trace=None)

    def test_point_field_integrates_to_cell_profile_with_exact_beam_cylinder_void(self):
        from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
        from Design.SMRF_Slab_Secant import LayeredSlabSecant
        for axis in ('x','y'):
            inv=fixture(axis);beam=dict(inv['floors']['1']['bars'][0],id='beam',operator_owner='web_line',z_in=2.,s_in=10.)
            inv['floors']['1']['bars'].append(beam);field=FixedPitchFloorMaterial(inv,1)
            total=np.zeros(3)
            # Piecewise constant actual mat field, integrated on its physical
            # intervals; independent closed-form beam area/depth moments.
            for along_lo,along_hi in ((0.,2.),(2.,28.),(28.,30.)):
                for lo,hi in zip((0.,2.,8.,14.,20.,26.),(2.,8.,14.,20.,26.,30.)):
                    along=(along_lo+along_hi)/2;across=(lo+hi)/2
                    point=field.point_profile(*((along,across) if axis=='x' else (across,along)))
                    mat=sum((np.array(r['depth_moments_per_unit_area']) for r in point.records if r['operator_owner']=='slab'),np.zeros(3))
                    total+=(along_hi-along_lo)*(hi-lo)*mat
            radius=math.sqrt(.2/math.pi)
            total+=26*.2*np.array([1.,2.,4.+radius*radius/4])
            cell=field.profile([0.,30.,0.,30.])
            np.testing.assert_allclose(total,cell.exact_depth_moments,rtol=1e-13,atol=1e-13)
            # Independently integrate exact local cylinder chords in a polar
            # coordinate, avoiding a singular end-point quadrature oracle.
            nodes,weights=leggauss(48);theta=nodes*math.pi/2;w=weights*math.pi/2
            integrated=np.zeros(3)
            for t,weight in zip(theta,w):
                across=10+radius*math.sin(t)
                p=field.point_profile(*((15.,across) if axis=='x' else (across,15.)))
                moments=next(np.array(r['depth_moments_per_unit_area']) for r in p.records if r['id']=='beam')
                integrated+=moments*weight*radius*math.cos(t)
            np.testing.assert_allclose(integrated,.2*np.array([1.,2.,4.+radius*radius/4]),rtol=1e-12,atol=1e-13)
            p=field.point_profile(*((15.,10.) if axis=='x' else (10.,15.)))
            self.assertEqual(p.void_only_ids,['beam'])
            self.assertNotIn('beam',[r['id'] for r in p.slab_steel_layers()])
            model=LayeredSlabSecant(8.,5.,p.slab_steel_layers(),ec_ksi=4000.,concrete_void_profile=p,quadrature_order=24)
            result=model.evaluate(np.zeros(6),branch='uncracked_elastic')
            a,b,d=np.array([8.,0.,8**3/12])-p.exact_depth_moments_per_unit_area
            exact=np.block([[a*model.elastic,b*model.elastic],[b*model.elastic,d*model.elastic]])
            for layer in p.slab_steel_layers():
                v=np.zeros(6);j=0 if layer['axis']=='x' else 1;v[j]=1;v[j+3]=layer['z_in']
                exact+=29000*layer['area_in2_per_in']*np.outer(v,v)
            np.testing.assert_allclose(result['ABD'],exact,rtol=2e-9,atol=2e-6)
            self.assertTrue(result['constitutive_state_verified'],result['domain_reasons'])

    def test_cell_average_is_not_claimed_to_equal_local_material_at_run_boundary(self):
        from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
        field=FixedPitchFloorMaterial(fixture(),1)
        average=sum(r['area_in2_per_in'] for r in field.profile([0.,4.,4.,6.]).slab_steel_layers())
        self.assertAlmostEqual(average,.2/12)
        self.assertEqual(field.point_profile(1.,5.).slab_steel_layers(),[])
        self.assertAlmostEqual(field.point_profile(3.,5.).slab_steel_layers()[0]['area_in2_per_in'],.2/6)
        invalid=fixture();bad=dict(invalid['floors']['1']['bars'][0],id='beam_nan',operator_owner='web_line',area_in2=float('nan'))
        invalid['floors']['1']['bars'].append(bad)
        with self.assertRaisesRegex(ValueError,'geometry'):FixedPitchFloorMaterial(invalid,1)

    def test_profile_separates_beam_void_and_nominal_layer_stiffness(self):
        from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
        from Design.SMRF_Physical_Floor_Materials import CellVoidProfile
        from Design.SMRF_Slab_Secant import LayeredSlabSecant
        inv=fixture();beam=dict(inv['floors']['1']['bars'][0],id='beam',operator_owner='web_line',z_in=2.,s_in=10.)
        inv['floors']['1']['bars'].append(beam);field=FixedPitchFloorMaterial(inv,1)
        profile=field.profile([0,30,0,30]);old=CellVoidProfile([0,30,0,30],8,inv['floors']['1']['bars'],state_identity='installed')
        np.testing.assert_allclose(profile.exact_depth_moments,old.exact_depth_moments,rtol=1e-13,atol=1e-13)
        self.assertEqual(profile.void_only_ids,['beam']);self.assertNotIn('beam',[p['id'] for p in profile.slab_steel_layers()])
        sec=LayeredSlabSecant(8,5,profile.slab_steel_layers(),ec_ksi=4000,concrete_void_profile=profile,quadrature_order=24)
        got=sec.evaluate(np.zeros(6),branch='uncracked_elastic')
        a,b,d=np.array([8,0,8**3/12])-profile.exact_depth_moments/profile.area
        exact=np.block([[a*sec.elastic,b*sec.elastic],[b*sec.elastic,d*sec.elastic]])
        for row in profile.slab_steel_layers():
            v=np.array([1.,0,0,row['z_in'],0,0]);exact+=row['area_in2_per_in']*29000*np.outer(v,v)
        np.testing.assert_allclose(got['ABD'],exact,rtol=2e-9,atol=2e-6)
        self.assertTrue(got['constitutive_state_verified'],got['domain_reasons'])
        self.assertNotEqual(profile.identity,old.identity)

    def test_explicit_frozen_reference_mapping_and_provenance(self):
        from types import SimpleNamespace
        from Design.SMRF_Bar_Pitch_Layers import METHOD
        from Design.SMRF_Physical_Floor_Materials import FrozenGravityReferenceBuilder
        inv=fixture();inv['hook_stiffness_basis']='actual straight portions only'
        raw=np.zeros((4,8,24))
        for i in range(4):raw[i,:6,:6]=np.diag([1,1,1,-1,-1,-1])
        rows=[]
        for i,(lo,hi) in enumerate(((0.,8.),(8.,19.),(19.,30.))):
            rows.append(dict(kind='slab',id=str(i),points=np.array([[0,lo,0],[30,lo,0],[30,hi,0],[0,hi,0]]),
                native=dict(strain=raw),transform=np.eye(24),indices=np.arange(24)))
        floor=SimpleNamespace(id='f',x=[0.,30.],y=[0.,8.,19.,30.],thickness=8.,ec=4000.,nu=.2,elements=rows,dof_ids=list(range(24)))
        row=dict(assembly=floor,description=dict(level=1,fc_ksi=5.));response=dict(retained_displacements=np.zeros(24),case_id='zero')
        builder=FrozenGravityReferenceBuilder(inv,reference_identity='fixed_fields',material_mapping=METHOD)
        result=builder.build(row,response,reference_loading='zero_analytical_witness')
        self.assertEqual(result['status'],'reference_constructed')
        self.assertTrue(result['audit']['ownership']['verified'])
        self.assertEqual(result['audit']['material_mapping'],METHOD)
        for cell in result['audit']['cells']:
            for point in cell['reference_points']:
                self.assertTrue(point['secant_method'].endswith('v3_stable_membrane_relaxation'))
                self.assertIn('final_response_range_error',point['membrane_relaxation_recovery'])
                self.assertIn('applied',point['relaxed_bending_energy_recovery'])
        self.assertFalse(result['audit']['material_mapping_engineering_qualification_inferred'])
        self.assertTrue(result['audit']['stiffness_basis'].endswith('/'+METHOD))
        legacy=FrozenGravityReferenceBuilder(inv,reference_identity='fixed_fields')
        original=legacy.build(row,response,reference_loading='zero_analytical_witness')
        self.assertNotEqual(original['audit']['ownership']['whole_floor_profile'],result['audit']['ownership']['whole_floor_profile'])
        with self.assertRaisesRegex(ValueError,'matching'):legacy.frozen_operator(row,result,beam_bending_factor=.5,web_torsion_factor=1.,source_identity='test')
        with self.assertRaisesRegex(ValueError,'mapping'):FrozenGravityReferenceBuilder(inv,reference_identity='r',material_mapping='assert_qualified')

    def test_irregular_mesh_conserves_each_bar_volume_centroid_and_depth_moment(self):
        for axis in ('x','y'):
            inv=fixture(axis);layers,audit=installed_mat_layers(inv,1)
            self.assertEqual(audit['families'][0]['uncovered_intervals_in'],[[0.,2.],[26.,30.]])
            grid=[0.,1.,2.3,5.1,8.,13.,17.6,25.8,29.,30.]
            keys=['straight_volume_in3','first_x_in4','first_y_in4','first_z_in4','second_z_in5']
            for layer in layers:
                records=[layer.cell([a,b,c,d]).record for a,b in zip(grid[:-1],grid[1:]) for c,d in zip(grid[:-1],grid[1:])]
                total=np.array([sum(r[k] for r in records) for k in keys]);bar=layer.bar;vol=.2*26
                x,y=(15.,bar['s_in']) if axis=='x' else (bar['s_in'],15.)
                target=vol*np.array([1.,x,y,-2.,4.+.2/math.pi/4])
                np.testing.assert_allclose(total,target,rtol=1e-13,atol=1e-13)
                # Constant and linear virtual-work fields retain exact bar
                # measures. Quadratic transverse work changes by the declared
                # pitch-layer idealization; it is not claimed exact.
                self.assertAlmostEqual(layer.cell([0,30,0,30]).record['whole_strip_transverse_second_moment_increment_per_unit_volume_in2'],3.-.2/math.pi/4)

    def test_local_density_is_mesh_independent_and_cross_axis_covariant(self):
        for axis in ('x','y'):
            layers,_=installed_mat_layers(fixture(axis),1)
            for lo,hi in ((4.,6.),(4.1,4.2),(7.9,8.1),(8.,11.),(12.9,13.1)):
                box=[10.,12.,lo,hi] if axis=='x' else [lo,hi,10.,12.]
                values=[p.cell(box) for p in layers]
                self.assertAlmostEqual(sum(v.record['layer_area_in2_per_in'] for v in values),.2/6,places=14)
                # Independent polar circle integration of z moments avoids
                # endpoint singularities in the finite-circle occupancy.
                nodes,weights=leggauss(48);angle=nodes*math.pi/2;w=weights*math.pi/2;r=math.sqrt(.2/math.pi)
                z=-2+r*np.sin(angle);dz=r*np.cos(angle)
                density=sum(v.occupied(z) for v in values)
                self.assertAlmostEqual(float(np.sum(w*dz*density)),.2/6,places=13)
                self.assertAlmostEqual(float(np.sum(w*dz*density*z*z)),.2/6*(4+r*r/4),places=13)

    def test_actual_edge_spacing_and_invalid_source_are_explicit(self):
        inv=fixture();inv['floors']['1']['bars'][0]['s_in']=1.
        layers,audit=installed_mat_layers(inv,1)
        self.assertEqual(layers[0].pitch,(0.,2.));self.assertEqual(layers[1].pitch,(8.,14.))
        self.assertIn([2.,8.],audit['families'][0]['uncovered_intervals_in'])
        changed=copy.deepcopy(inv);changed['floors']['1']['bars'][1]['s_in']=1.
        with self.assertRaises(ValueError):installed_mat_layers(changed,1)
        changed=copy.deepcopy(inv);changed['floors']['1']['bars'][0]['s_in']=.1
        with self.assertRaisesRegex(ValueError,'volume'):installed_mat_layers(changed,1)
        changed=copy.deepcopy(inv);changed['floors']['1']['bars'].append(copy.deepcopy(changed['floors']['1']['bars'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):installed_mat_layers(changed,1)
        changed=copy.deepcopy(inv);changed['floors']['1']['bars'][0].pop('mat_name')
        with self.assertRaisesRegex(ValueError,'family'):installed_mat_layers(changed,1)
        p=layers[0]
        with self.assertRaisesRegex(ValueError,'Symmetric'):BarPitchLayer(p.bar,[0.,3.],footprint=inv['floors']['1']['footprint'],slab_thickness_in=8.,inventory_identity='bound')
        for invalid in (None,'',False):
            with self.assertRaisesRegex(ValueError,'inventory'):BarPitchLayer(p.bar,p.pitch,footprint=inv['floors']['1']['footprint'],slab_thickness_in=8.,inventory_identity=invalid)
        bad=dict(p.bar,fy_ksi=float('nan'))
        with self.assertRaisesRegex(ValueError,'strengths'):BarPitchLayer(bad,p.pitch,footprint=inv['floors']['1']['footprint'],slab_thickness_in=8.,inventory_identity='bound')

if __name__=='__main__':unittest.main()
