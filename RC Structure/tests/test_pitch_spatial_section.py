import copy,math,unittest
import numpy as np
from Design.SMRF_Pitch_Spatial_Section import pitch_spatial_section
from Design.SMRF_Physical_Floor_Materials import _hash
from Design.SMRF_Actual_Bar_Development import METHOD as DEV_METHOD,side_development,full_fy_developed_domain


def fixture(axis='x',station=10.,irregular=False):
    other='y' if axis=='x' else 'x';bars=[]
    for orient,z,positions in ((axis,-2.,(5.,11.,17.,23.)),(other,2.,(5.,10.,19.,25.) if irregular else (5.,11.,17.,23.))):
        for i,s in enumerate(positions):
            bars.append(dict(id=orient+str(i),axis=orient,operator_owner='slab',mat_name=orient+str(z),s_in=s,z_in=z,
                area_in2=.2,db_in=.5,fy_ksi=60.,es_ksi=29000.,straight_run_interval_in=[2.,28.]))
    inv=dict(state_identity='actual_state',identity_sha256='actual_inventory',slab_thickness_in=8.,
        floors={'1':dict(footprint=dict(x_min_in=0.,x_max_in=30.,y_min_in=0.,y_max_in=30.),bars=bars)})
    spec=dict(floor=1,axis=axis,station_in=station,interval_in=[2.,26.],fc_ksi=5.,ec_ksi=4000.,phi_requirements={'NM':.65,'shear':.6},
        transverse_reference=0.,material_scope='slab_only')
    evidence=[]
    for bar in bars:
        stations=[station] if bar['axis']==axis else [2.,26.]
        lengths=dict(status='pass',ld_tension_in=5.,ld_compression_in=3.,mandatory_confinement_passes=True)
        checks=[side_development(where,bar['straight_run_interval_in'],lengths,None) for where in stations]
        domain=full_fy_developed_domain(checks[0])
        row=dict(method=DEV_METHOD,state_identity=inv['state_identity'],material_inventory_id=inv['identity_sha256'],
            floor=1,axis=axis,interval_in=[2.,26.],station_interval_in=[station,station],physical_bar_ids=[bar['id']],
            bar_axis=bar['axis'],full_fy_only=True,full_fy_development_domain=domain,checks=checks)
        row['evidence_id']=_hash(row);evidence.append(row)
    return inv,spec,evidence


class PitchSpatialTests(unittest.TestCase):
    def test_local_transverse_pitch_and_developed_run_match_independent_nonuniform_steel_forces(self):
        for axis in ('x','y'):
            inv,args,ev=fixture(axis);out=pitch_spatial_section(inv,**args,development_evidence=ev)
            self.assertEqual(out['status'],'inventory_constructed',out['issues']);self.assertEqual(out['development_status'],'pass')
            self.assertFalse(out['engineering_qualified']);self.assertFalse(out['axial_window_equivalence_claimed'])
            s=out['section'];self.assertEqual(len(s.longitudinal),4);self.assertEqual(len(s.transverse),1)
            row=s.transverse[0];self.assertEqual(row['physical_bar_ids'],[('y' if axis=='x' else 'x')+'1'])
            self.assertEqual(row['accounting_length_in'],6.);self.assertEqual(row['strength_run_interval_in'],(7.,23.))
            q=np.array([.0004,.0003,0.,.000005,-.000004,0.,.000003])
            if axis=='y':q=q[[1,0,2,4,3,5,6]]
            # In this nonuniform all-tension state concrete contributes zero;
            # actual discrete longitudinal forces vary with transverse position.
            # The transverse local layer uses exactly one bar per actual pitch,
            # with strength only over its independently computed developed run.
            exact=np.zeros(7);j=0 if axis=='x' else 1;k=1-j
            for b in inv['floors']['1']['bars']:
                if b['axis']!=axis:continue
                strain=q[j]+b['z_in']*q[j+3]+b['s_in']*q[6]
                f=.2*29000*strain;exact[j]+=f;exact[j+3]+=f*b['z_in'];exact[6]+=f*b['s_in']
            transverse_force=.2/6*16*29000*(q[k]+2*q[k+3]);exact[k]+=transverse_force;exact[k+3]+=2*transverse_force
            np.testing.assert_allclose(s.state(q)['nominal_resultants'],exact/24,rtol=1e-11,atol=1e-11)

    def test_irregular_pitch_gap_and_one_sided_traces_do_not_invent_transverse_steel(self):
        inv,args,ev=fixture(station=14.,irregular=True)
        out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertIsNotNone(out['section']);self.assertEqual(out['constructor']['transverse_rows'],[])
        inv,args,ev=fixture(station=7.5,irregular=True)
        a=pitch_spatial_section(inv,**args,development_evidence=ev)
        b=pitch_spatial_section(inv,**args,development_evidence=ev,trace='minus')
        self.assertEqual(a['constructor']['transverse_rows'][0]['physical_bar_ids'],['y1'])
        self.assertEqual(b['constructor']['transverse_rows'][0]['physical_bar_ids'],['y0'])
        self.assertEqual(a['constructor']['transverse_rows'][0]['accounting_length_in'],5.)
        self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])

    def test_missing_or_forged_development_is_not_full_strength_credit(self):
        inv,args,ev=fixture();missing=pitch_spatial_section(inv,**args)
        self.assertEqual(missing['development_status'],'unknown')
        self.assertTrue(all(not r['strength_credited'] for r in missing['constructor']['longitudinal_bars']))
        self.assertIsNone(missing['constructor']['transverse_rows'][0]['strength_run_interval_in'])
        bad=copy.deepcopy(ev);bad[0]['full_fy_development_domain']['interval_in']=[2.,28.]
        with self.assertRaisesRegex(ValueError,'hash'):pitch_spatial_section(inv,**args,development_evidence=bad)
        bad[0]['evidence_id']=_hash({k:v for k,v in bad[0].items() if k!='evidence_id'})
        with self.assertRaisesRegex(ValueError,'end checks'):pitch_spatial_section(inv,**args,development_evidence=bad)
        stale=copy.deepcopy(ev)
        for row in stale:
            row['material_inventory_id']='stale';row['evidence_id']=_hash({k:v for k,v in row.items() if k!='evidence_id'})
        self.assertEqual(pitch_spatial_section(inv,**args,development_evidence=stale)['development_status'],'unknown')

    def test_undeveloped_longitudinal_ends_retain_actual_concrete_voids(self):
        inv,args,ev=fixture(station=3.);out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertEqual(out['development_status'],'pass');self.assertIsNotNone(out['section'])
        self.assertEqual(len(out['section'].longitudinal),4)
        self.assertTrue(all(not b['strength_credited'] for b in out['section'].longitudinal))
        self.assertEqual(len(out['physical_bar_ids']),5)
        self.assertFalse(out['nominal_probable_capacity_inventory_modified'])

    def test_beam_void_ownership_and_exact_crossing_chord_subtraction(self):
        inv,args,ev=fixture();beam=dict(inv['floors']['1']['bars'][0],id='beam_x',operator_owner='web_line',z_in=0.,s_in=10.)
        inv['floors']['1']['bars'].append(beam)
        out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertIsNotNone(out['section'],out['issues']);self.assertIn('beam_x',out['excluded_beam_strength_ids'])
        self.assertNotIn('beam_x',out['physical_bar_ids'])
        self.assertFalse(next(b for b in out['section'].longitudinal if b['id']=='beam_x')['strength_credited'])
        before=out['section'];crossing=dict(beam,id='beam_y',axis='y',s_in=10.,z_in=1.)
        inv['floors']['1']['bars'].append(crossing)
        out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertIsNotNone(out['section'],out['issues']);self.assertIn('beam_y',out['excluded_beam_strength_ids'])
        self.assertNotIn('beam_y',out['physical_bar_ids'])
        after=out['section'];void=out['constructor']['slab_void_rectangles'][0];r=math.sqrt(.2/math.pi)
        np.testing.assert_allclose(void['z_interval_in'],[1-r,1+r],rtol=0,atol=1e-14)
        q=np.array([-.0002,-.0002,0.,0.,0.,0.,0.]);n=.8+5*6.894757293168/17;peak=5/4000*n/(n-1)
        ratio=.0002/peak;sigma=-5*n*ratio/(n-1+ratio**n)
        expected=-sigma*(2*r)*np.array([1.,1.,0.,1.,1.,0.,14.])
        np.testing.assert_allclose(np.array(after.state(q)['nominal_resultants'])-before.state(q)['nominal_resultants'],expected,rtol=1e-11,atol=1e-10)
        trial=np.array([-.0002,-.00015,.00002,.000001,.000002,.0000003,.0000001])
        np.testing.assert_allclose(after.fixed_quadrature_resultants_batch([trial],order=24)[0],after.state(trial,quadrature_order=24)['design_resultants'],rtol=2e-6,atol=1e-7)
        crossing['z_in']=0.;out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertIsNone(out['section']);self.assertEqual(out['status'],'unknown')
        self.assertIn('overlaps',out['issues'][0]['reason'])
        # A plane offset from the cylinder axis has the smaller actual chord;
        # its nominal diameter is not subtracted at every parallel section.
        crossing['z_in']=1.;crossing['s_in']=10+r/2
        out=pitch_spatial_section(inv,**args,development_evidence=ev)
        self.assertIsNotNone(out['section'],out['issues'])
        zlo,zhi=out['constructor']['slab_void_rectangles'][0]['z_interval_in']
        self.assertAlmostEqual(zhi-zlo,math.sqrt(3)*r,places=13)


if __name__=='__main__':unittest.main()
