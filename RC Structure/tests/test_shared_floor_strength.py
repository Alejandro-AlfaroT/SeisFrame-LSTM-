"""Independent atomic-cut/physical-inventory checks; no structural solves."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from Design.SMRF_Shared_Floor_Strength import (
    SharedFloorSection, mat_layers_from_global_placement,
)


def layers(width=16.):
    count=int(width/8)
    return [dict(id=axis+face,axis=axis,z_in=sgn*z,
        area_in2_per_in=count*.31/width,bar_area_in2=.31,fy_ksi=60.,
        physical_bar_ids=[f'{axis}{face}/{i}' for i in range(count)],
        density_width_in=width,material_inventory_id='actualmat',state_identity='state1')
        for axis,z in [('x',3.),('y',2.25)] for face,sgn in [('top',1),('bottom',-1)]]


def beams():
    return [dict(id='beam/'+face,axis='x',z_in=z,area_in2=2.,bar_area_in2=1.,
        fy_ksi=60.,physical_bar_ids=[f'beam/{face}/{i}' for i in range(2)],
        concrete_owner=owner,material_inventory_id='actualbeam',state_identity='state1')
        for face,z,owner in [('top',.9,'slab'),('bottom',-22.,'web')]]


def section(mode='actual_core',width=16.,phi=.65):
    return SharedFloorSection(8.,5.,ec_ksi=4000.,cut_width_in=width,mat_layers=layers(width),
        beam_rows=() if mode=='slab_patch' else beams(),
        drop_webs=() if mode=='slab_patch' else [dict(id='web',axis='x',width_in=16.,z_bottom_in=-24.,z_top_in=-4.)],
        mode=mode,phi_requirements={'axial_flexure':.65,'diaphragm_shear':phi},concrete_peak_strain=.002)


def cut(target,width=16.):
    identity=dict(state_identity='state1',combination_id='1.2D+L+Ex',location_id='L2/beam7/cut3',recovery_id='physical01')
    return dict(identity=identity,source_identities=[dict(identity,recovery_id='gravity01'),dict(identity,recovery_id='elfprojection01')],
        resultants=list(target),units='per_unit_width',reference_z_in=0.,cut_width_in=width,
        accounting=dict(method='disjoint_actual_fields',evidence_id='forcecut-ledger01',
            equilibrium_residual=[0.]*6,equilibrium_tolerance=[1e-7]*6,
            common_width_strain_projection_verified=True,core_remote_coverage_verified=True))


class SharedFloorStrengthTests(unittest.TestCase):
    def test_all_six_core_actions_from_independent_finite_volume_integrals(self):
        s=section();q=np.array([-.00025,.00045,.0006,-.000045,.000012,.000018])
        # Independent closed2x2 spectral projector, scalar adaptive integration,
        # exact source-law parameters, and circular-void theta integration.
        # This target never calls the production stress/section recovery.
        def stress(z):
            x,y,g=q[:3]+z*q[3:];mid=(x+y)/2
            radius=math.hypot((x-y)/2,g/2);low,high=mid-radius,mid+radius
            if low>=0:return np.zeros(3)
            self.assertGreater(high,0)
            fc=5/max(1,.8+.34*high/.002);n=.8+fc*6.894757293168/17;r=-low/.002
            sigma=-fc*n*r/(n-1+r**n)
            return sigma*np.array([(high-x)/(high-low),(high-y)/(high-low),-g/(2*(high-low))])
        def webstress(z):
            r=max(0.,-(q[0]+z*q[3])/.002);n=.8+5*6.894757293168/17
            return -5*n*r/(n-1+r**n)
        ref=np.array([quad(lambda z:stress(z)[j%3]*(z if j>=3 else 1),-4,4,epsabs=1e-10)[0] for j in range(6)])
        for moment in (False,True):
            ref[3 if moment else 0]+=quad(lambda z:webstress(z)*(z if moment else 1),-24,-4,
                points=[-q[0]/q[3]],epsabs=1e-10)[0]
        for row in layers()+beams():
            area=row.get('area_in2_per_in',row.get('area_in2',0)/16)
            radius=math.sqrt(row['bar_area_in2']/math.pi);count=area/row['bar_area_in2']
            for j in range(6):
                def void(theta):
                    z=row['z_in']+radius*math.sin(theta)
                    value=stress(z)[j%3] if row.get('concrete_owner','slab')=='slab' else (webstress(z) if j%3==0 else 0.)
                    return value*(z if j>=3 else 1)*2*radius**2*math.cos(theta)**2*count
                ref[j]-=quad(void,-math.pi/2,math.pi/2,epsabs=1e-10)[0]
            j=0 if row['axis']=='x' else 1
            f=area*np.clip(29000*(q[j]+row['z_in']*q[j+3]),-60,60)
            ref[j]+=f;ref[j+3]+=row['z_in']*f
        self.assertTrue(np.all(abs(ref)>1e-5))
        out=s.evaluate(cut(.65*ref),initial_strain=q*.92)
        self.assertEqual(out['status'],'pass',out)
        np.testing.assert_allclose(out['numerical_state']['generalized_strain'],q,atol=1e-9)
        np.testing.assert_allclose(out['numerical_state']['nominal_resultants'],ref,atol=1e-8,rtol=1e-8)

    def test_actual_core_tension_hand_result_and_one_inventory(self):
        s=section();r=np.zeros(6)
        # Independent direct axial bar forces, no concrete, constant strain.
        for row in layers()+beams():
            area=row.get('area_in2_per_in',row.get('area_in2',0)/16)
            j=0 if row['axis']=='x' else 1
            f=area*29000*[.001,.0005][j]
            r[j]+=f;r[j+3]+=row['z_in']*f
        out=s.evaluate(cut(.65*r),initial_strain=[.001,.0005,0,0,0,0])
        self.assertEqual(out['status'],'pass',out)
        np.testing.assert_allclose(out['numerical_state']['generalized_strain'],[.001,.0005,0,0,0,0],atol=1e-10)
        self.assertEqual(len([v for row in out['material_ownership'] for v in row['physical_bar_ids']]),12)
        self.assertFalse(out['nominal_probable_effective_flange_inventory_modified'])
        # Full-depth core explicitly contains slab ABOVE web, not just a drop.
        self.assertEqual(s.kernel.h,8.)
        self.assertEqual(s.kernel.webs[0]['z_top_in'],-4.)

    def test_pure_slab_membrane_shear_hand_phi_and_path(self):
        s=section('slab_patch',phi=.60)
        # Principal strains+.0012,-.0008 at45deg; exact source backbone.
        peak=5/(.8+.34*.0012/.002);n=.8+peak*6.894757293168/17;r=.4
        sigma=-peak*n*r/(n-1+r**n)
        concrete=8-4*.31/8
        normal=sigma/2*concrete+2*.31/8*29000*.0002
        target=.60*np.array([normal,normal,-sigma/2*concrete,0,0,0])
        out=s.evaluate(cut(target),initial_strain=[.0002,.0002,.002,0,0,0])
        self.assertEqual(out['status'],'pass',out)
        self.assertEqual(out['phi'],.60)
        # Separate simple pure-tension path exercises the optional continuation.
        target=.60*np.array([2*.31/8*29,2*.31/8*14.5,0,0,0,0])
        path=s.evaluate(cut(target),path_steps=3)
        self.assertEqual(path['status'],'pass',path)
        self.assertTrue(path['path']['proportional_force_path_verified'])

    def test_atomic_identity_and_duplicate_material_refused(self):
        s=section();bad=cut([0]*6)
        bad['source_identities'][1]['combination_id']='different sign'
        with self.assertRaises(ValueError):s.evaluate(bad)
        bad=cut([0]*6);bad['reference_z_in']=-10
        with self.assertRaises(ValueError):s.evaluate(bad)
        bad=cut([0]*6);bad['cut_width_in']=48
        with self.assertRaises(ValueError):s.evaluate(bad)
        mats=layers();bars=beams();bars[0]['physical_bar_ids'][0]=mats[0]['physical_bar_ids'][0]
        with self.assertRaisesRegex(ValueError,'credited twice'):
            SharedFloorSection(8,5,ec_ksi=4000,cut_width_in=16,mat_layers=mats,
                beam_rows=bars,drop_webs=section().kernel.webs,mode='actual_core',phi_requirements={'all':.65})

    def test_unknown_vs_proved_capacity_exceeded(self):
        s=section('slab_patch');target=[20.,0,0,0,0,0]
        # Tension capacity cannot exceed total x-steel*fy; concrete has no tension.
        upper=.65*(2*.31/8)*60
        self.assertAlmostEqual(s.support_upper_bound([1,0,0,0,0,0]),upper,12)
        out=s.evaluate(cut(target))
        self.assertEqual(out['status'],'capacity_exceeded')
        self.assertIsNone(out['numerical_state'])
        self.assertAlmostEqual(out['capacity_bound_violations'][0]['proved_design_upper_bound'],upper)
        missing=cut(target);missing['accounting']['core_remote_coverage_verified']=False
        self.assertEqual(s.evaluate(missing)['status'],'unknown')
        # A failed numerical root has no analogous proof. Deliberately inject a
        # solver failure to verify classification without an expensive search.
        with patch.object(s.kernel,'solve',return_value={'restricted_model_state_verified':False}):
            out=s.evaluate(cut([0]*6))
        self.assertEqual(out['status'],'unknown')
        self.assertEqual(out['section_status'],'unknown')
        missing=cut([0]*6);missing['accounting']['method']='fullT_moved_into_narrow_core'
        self.assertEqual(s.evaluate(missing)['status'],'unknown')

    def test_support_bound_independent_uniform_torsion_and_samples(self):
        s=section('slab_patch')
        # Gross compression-only stress max|tau|=fc/2; no diagonal rebar shear.
        self.assertAlmostEqual(s.support_upper_bound([0,0,1,0,0,0]),.65*5/2*8)
        self.assertAlmostEqual(s.support_upper_bound([0,0,0,0,0,1]),.65*5/2*16)
        # A few finite states check the algebraic inequality, including coupling.
        rng=np.random.default_rng(141)
        for mode in ('slab_patch','actual_core','actual_common_width'):
            s=section(mode,width=48 if mode=='actual_common_width' else 16)
            for _ in range(8):
                q=rng.uniform(-1,1,6)*[.0002,.0002,.0004,.00001,.00001,.00002]
                state=s.kernel.state(q)
                for _ in range(3):
                    d=rng.uniform(-1,1,6)
                    self.assertLessEqual(float(d@np.array(state['design_resultants'])),s.support_upper_bound(d)+1e-8)

    def test_axis_exchange_and_integrated_strip_units(self):
        a=section();mats=layers();bars=beams();web=dict(a.kernel.webs[0])
        for row in mats+bars:row['axis']='y' if row['axis']=='x' else 'x'
        web['axis']='y'
        b=SharedFloorSection(8,5,ec_ksi=4000,cut_width_in=16,mat_layers=mats,beam_rows=bars,
            drop_webs=[web],mode='actual_core',phi_requirements={'all':.65},concrete_peak_strain=.002)
        q=np.array([.001,.0005,0,0,0,0])
        target=np.zeros(6)
        for row in layers()+beams():
            j=0 if row['axis']=='x' else 1
            area=row.get('area_in2_per_in',row.get('area_in2',0)/16)
            f=.65*area*29000*q[j];target[j]+=f;target[j+3]+=row['z_in']*f
        ca=cut(target*16);ca['units']='integrated_strip'
        cb=cut(target[[1,0,2,4,3,5]])
        ra=a.evaluate(ca,initial_strain=q);rb=b.evaluate(cb,initial_strain=q[[1,0,2,4,3,5]])
        self.assertEqual(ra['status'],'pass');self.assertEqual(rb['status'],'pass')
        np.testing.assert_allclose(np.array(ra['numerical_state']['design_resultants'])[[1,0,2,4,3,5]],rb['numerical_state']['design_resultants'],atol=1e-10)

    def test_global_phase_actual_counts_and_top_to_mid_reference(self):
        placement=dict(status='pass',state_identity='state1',layout_sha256='hash1',layers={
            f'{axis}{face}':dict(status='pass',runs_along=axis,db_in=.625,bar_size=5,
                centroid_from_top_in=z,global_transverse_coordinates_in=[2,10,18,26],run_interval_in=[0,32])
            for axis,z0 in [('x',1),('y',1.75)] for face,z in [('top',z0),('bottom',8-z0)]})
        rows=mat_layers_from_global_placement(placement,bounds_in=[0,16,0,24],thickness_in=8,fy_ksi=60,
            bar_area_by_size={5:.31})
        x=next(v for v in rows if v['id']=='slab/xtop');y=next(v for v in rows if v['id']=='slab/ytop')
        self.assertEqual(x['physical_bar_ids'],['slab/xtop/0','slab/xtop/1','slab/xtop/2'])
        self.assertEqual(y['physical_bar_ids'],['slab/ytop/0','slab/ytop/1'])
        self.assertAlmostEqual(x['area_in2_per_in'],3*.31/24)
        self.assertAlmostEqual(y['area_in2_per_in'],2*.31/16)
        self.assertEqual(x['z_in'],3);self.assertEqual(y['z_in'],2.25)
        current=dict(placement);current.pop('layers');current['floors']={'2':{'layers':{
            name:{**mat,'straight_run_interval_in':mat['run_interval_in']} for name,mat in placement['layers'].items()}}}
        actual=mat_layers_from_global_placement(current,bounds_in=[0,16,0,24],thickness_in=8,fy_ksi=60,
            bar_area_by_size={5:.31},floor=2)
        self.assertEqual(actual[0]['physical_bar_ids'][0],'floor_2/slab/xtop/0')
        with self.assertRaisesRegex(ValueError,'floor identity'):
            mat_layers_from_global_placement(current,bounds_in=[0,16,0,24],thickness_in=8,fy_ksi=60,bar_area_by_size={5:.31})
        with self.assertRaisesRegex(ValueError,'boundary cuts'):
            mat_layers_from_global_placement(placement,bounds_in=[0,16,0,18],thickness_in=8,fy_ksi=60,bar_area_by_size={5:.31})
        with self.assertRaisesRegex(ValueError,'nominal bar area'):
            mat_layers_from_global_placement(placement,bounds_in=[0,16,0,24],thickness_in=8,fy_ksi=60)


if __name__=='__main__':unittest.main()
