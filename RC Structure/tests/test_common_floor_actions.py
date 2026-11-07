import copy
import unittest
import numpy as np
from Design.SMRF_Common_Floor_Actions import polynomial_slab_field,integrate_slab_field,full_t_normal_terms,physical_partition,compose_common_cut,physical_response_terms
from Design.SMRF_Full_T_Projection import FullTIncrementProjector


def projector(axis='x'):
    return FullTIncrementProjector(slab_thickness_in=8,beam_depth_in=24,flange_interval_in=[-30,50],
        web_interval_in=[-5,15],slab_E_ksi=4000,web_E_ksi=4000,axis=axis,transverse_reference_in=5)


def ident(location='core'):
    return dict(state_identity='actual_state',combination_id='1.2D+L+Ex',location_id=location,recovery_id='recovery')


def gravity(interval=(-5,15),location='core',web=True):
    a,b=interval;width=b-a;sref=(a+b)/2
    tensor=np.array([1.,2.,.2,3.,-1.,.4,.8,.9]);i=tensor*width;f=tensor*(b*b-a*a)/2
    slab=dict(axis='x',station_in=100.,trace='minus',interval_in=[a,b],integral=i.tolist(),transverse_first_moment=f.tolist(),
              pieces=[dict(interval_in=[a,b],integral=i.tolist(),transverse_first_moment=f.tolist())])
    w=dict(member_id='7',N_tension_kip=-4.,M_centroid_kipin=50.,centroid_z_in=-12.,
           Ms_transverse_first_moment_kipin=7.,transverse_reference_in=5.) if web else None
    r=i[:6].copy();ms=f[0]-sref*i[0]
    if w:r[0]-=4;r[3]+=98;ms+=7+(5-sref)*-4
    return dict(identity=ident(location),slab=slab,web=w,reference_z_in=0.,transverse_reference_in=sref,
                resultants=r.tolist(),Ms_transverse_first_moment_kipin=ms,
                accounting=dict(equilibrium_residual=[0.],equilibrium_tolerance=[1e-9]))


def p5(identity,branch='original',source='p5'):
    c=np.array([[.3,-.1,.2,0,0,0,0,0],[.01,.02,-.003,0,0,0,0,0]])
    samples=np.array([-5,15]);v=np.c_[np.ones(2),samples]@c
    return polynomial_slab_field(source_id=source,source_kind='p5_slab',identity=identity,axis='x',station_in=100,
        interval_in=[-5,15],origin_in=0,coefficients=c,sample_coordinates_in=samples,sample_tensors=v,branch_id=branch)


class CommonFloorActionTests(unittest.TestCase):
    def test_independent_full_tensor_polynomial_integrals_and_first_moments(self):
        c=np.zeros((3,8));c[:,0]=[2,3,4];c[:,1]=[5,-2,1];c[:,2]=[.4,.1,0]
        x=np.array([1.,2.,4.]);v=np.vander(x-1,3,increasing=True)@c
        field=polynomial_slab_field(source_id='p',source_kind='p5_slab',identity=ident(),axis='x',station_in=100,
            interval_in=[1,4],origin_in=1,coefficients=c,sample_coordinates_in=x,sample_tensors=v,branch_id='original')
        i,f=integrate_slab_field(field,[1,4])
        self.assertAlmostEqual(i[0],2*3+3*9/2+4*27/3,12)
        self.assertAlmostEqual(f[0],i[0]+2*9/2+3*27/3+4*81/4,12)
        np.testing.assert_array_equal(i[3:],[0]*5)

    def test_disjoint_core_remote_preserves_full_T_N_M_and_weak_first_moment(self):
        p=projector();projected=p.project(n_tension_kip=0,m_z_kip_in=120,m_s_kip_in=13,increment_id='actualELF')
        cuts=[gravity((-30,-5),'left',False),gravity(),gravity((15,50),'right',False)]
        partition=physical_partition(cuts,outer_interval_in=[-30,50]);sumdelta=np.zeros(3)
        for cut in cuts:
            fields=full_t_normal_terms(p,projected,identity=cut['identity'],source_id='primary7',member_id='7',station_in=100)
            out=compose_common_cut(cut,slab_fields=[fields['slab']],web_additions=[fields['web']],
                required_source_ids=['primary7/slab','primary7/web'],partition=partition)
            delta=np.array(out['resultants'])-np.r_[cut['resultants'],cut['Ms_transverse_first_moment_kipin']]
            sumdelta +=[delta[0],delta[3],delta[6]+(out['reference_s_in']-5)*delta[0]]
            self.assertTrue(out['accounting']['spatial_seven_field_projection_verified'])
            self.assertFalse(out['accounting']['elastic_ultimate_strain_equality_required'])
            self.assertEqual(out['local_transverse_shear_status'],'unknown')
        np.testing.assert_allclose(sumdelta,[0,120,13],atol=1e-11)

    def test_P5_load_plane_transport_does_not_add_spurious_global_M(self):
        cut=gravity();partition=physical_partition([cut],outer_interval_in=[-5,15]);field=p5(cut['identity'])
        web=dict(source_id='P5web',source_kind='p5_web',identity=cut['identity'],axis='x',station_in=100.,
            member_id='7',interval_in=[-5,15],reference_s_in=5.,reference_z_in=0.,n_tension_kip=10.,
            m_z_kip_in=0.,m_s_kip_in=0.,signed_axial_end_forces_kip=[-10.,10.],source_evidence_id='actualFE',branch_id='original')
        out=compose_common_cut(cut,slab_fields=[field],web_additions=[web],required_source_ids=['p5','P5web'],partition=partition)
        self.assertAlmostEqual(out['resultants'][3],cut['resultants'][3],12)
        self.assertAlmostEqual(out['resultants'][0]-cut['resultants'][0],.3*20+.01*(225-25)/2+10,12)
        self.assertEqual(out['local_transverse_shear_status'],'unknown')
        self.assertFalse(out['accounting']['physical_diaphragm_embedding_verified'])
        bad=copy.deepcopy(web);bad['reference_z_in']=-12
        with self.assertRaises(ValueError):compose_common_cut(cut,slab_fields=[field],web_additions=[bad],required_source_ids=['p5','P5web'],partition=partition)

    def test_orthogonal_primary_tensor_source_retains_Nyy_and_Myy(self):
        cut=gravity();partition=physical_partition([cut],outer_interval_in=[-5,15]);p=projector('y')
        projections=[p.project(n_tension_kip=0,m_z_kip_in=m,m_s_kip_in=0,increment_id=str(m)) for m in (50,90)]
        values=np.array([p.slab_field(q,10)+[0,0] for q in projections]);co=np.stack([values[0],(values[1]-values[0])/20])
        term=polynomial_slab_field(source_id='orthogonalY',source_kind='primary_elf_slab',identity=cut['identity'],axis='x',station_in=100,
            interval_in=[-5,15],origin_in=-5,coefficients=co,sample_coordinates_in=[-5,15],sample_tensors=values,
            proof=dict(full_T_projections=projections,axial_scope='zero_total_N_primary'))
        out=compose_common_cut(cut,slab_fields=[term],web_additions=[],required_source_ids=['orthogonalY'],partition=partition)
        delta=np.array(out['resultants'][:6])-cut['resultants']
        np.testing.assert_allclose(delta,10*(values[0,:6]+values[1,:6]),atol=1e-12)
        self.assertNotEqual(delta[1],0);self.assertNotEqual(delta[4],0)

    def test_nonconcurrent_duplicate_missing_branch_and_axis_are_refused(self):
        cut=gravity();partition=physical_partition([cut],outer_interval_in=[-5,15]);field=p5(cut['identity'])
        def call(fields,ids):return compose_common_cut(cut,slab_fields=fields,web_additions=[],required_source_ids=ids,partition=partition)
        with self.assertRaises(ValueError):call([field,field],['p5'])
        with self.assertRaises(ValueError):call([field],[])
        with self.assertRaises(ValueError):call([field,p5(cut['identity'],'collector','p52')],['p5','p52'])
        bad=copy.deepcopy(field);bad['identity']['combination_id']='other'
        with self.assertRaises(ValueError):call([bad],['p5'])
        bad=copy.deepcopy(field);bad['axis']='y'
        with self.assertRaises(ValueError):call([bad],['p5'])
        bad=copy.deepcopy(field);bad['coefficients'][0][0]+=1
        with self.assertRaises(ValueError):call([bad],['p5'])

    def test_partition_and_primary_axial_overlap_refused(self):
        with self.assertRaises(ValueError):physical_partition([gravity((-5,5)),gravity((4,15),'overlap',False)],outer_interval_in=[-5,15])
        p=projector();projection=p.project(n_tension_kip=1,m_z_kip_in=120,m_s_kip_in=0,increment_id='Nduplicate')
        with self.assertRaises(ValueError):full_t_normal_terms(p,projection,identity=ident(),source_id='P',member_id='7',station_in=100)

    def test_integrated_shear_has_interface_axial_transfer_and_exact_total(self):
        p=projector();out=p.integrated_shear_partition(resultant_derivatives=[0,-25,4],vertical_shear_kip=25,source_id='actual_derivatives')
        nfprime,mfprime,_=out['flange_resultant_derivatives'];_,mwprime,_=out['web_resultant_derivatives']
        self.assertAlmostEqual(out['slab_integrated_Q_kip'],-mfprime-4*nfprime,12)
        self.assertAlmostEqual(out['web_integrated_Q_kip'],-mwprime+4*nfprime,12)
        self.assertAlmostEqual(out['slab_integrated_Q_kip']+out['web_integrated_Q_kip'],25,12)
        self.assertFalse(out['local_transverse_shear_field_established'])
        with self.assertRaises(ValueError):p.integrated_shear_partition(resultant_derivatives=[1,-25,4],vertical_shear_kip=25,source_id='bad')

    def test_actual_physical_P5_tensor_and_web_companion_moment_transfer(self):
        from Design.SMRF_Physical_Floor_Cuts import common_strip_cut
        g=1/np.sqrt(3);gps=[(-g,-g),(g,-g),(g,g),(-g,g)]
        base=np.array([.3,-.2,.1,2.,-3.,.5,4.,-5.]);slope=np.array([.02,.01,-.003,.04,.03,-.02,.05,.06])
        slab=dict(kind='slab',id='cell',positions_in=[[90,-5,0],[110,-5,0],[110,15,0],[90,15,0]],
                  raw_resultants=[(base+slope*(5+10*eta)).tolist() for xi,eta in gps])
        forces=[-10,2,3,4,-50,7,10,-2,-3,-4,-10,33]
        web=dict(kind='web',id='webseg',member_id='7',axis='x',positions_in=[[90,5,-12],[110,5,-12]],
                 global_resisting_force=forces,section={'b_in':20.,'h_in':24.})
        response=dict(method='independent_affine_field_fixture',elements=[slab,web],force_partition_relative_error=0.)
        cut=common_strip_cut(response,axis='x',station_in=100,interval_in=[-5,15],trace='minus',web_id='7',
            uniform_web_loads={'webseg':0},**dict(state_identity='actual_state',combination_id='1.2D+L+Ex',location_id='core',recovery_id='physical'))
        terms=physical_response_terms(response,cut,branch_id='collectorEq',source_prefix='physicalP5')
        np.testing.assert_allclose(terms['reconstruction_residual'],[0]*7,atol=1e-12)
        self.assertEqual(terms['web_other_actions'],{'Q_vertical_kip':3.,'V_inplane_kip':-2.,'T_about_positive_axis_kipin':-4.})
        self.assertEqual(terms['web_additions'][0]['m_z_kip_in'],-100.)
        original=gravity();partition=physical_partition([original],outer_interval_in=[-5,15])
        # Same case/location; each original recovery identity remains recorded.
        out=compose_common_cut(original,slab_fields=terms['slab_fields'],web_additions=terms['web_additions'],
            required_source_ids=terms['required_source_ids'],partition=partition)
        np.testing.assert_allclose(np.array(out['resultants'])-np.r_[original['resultants'],original['Ms_transverse_first_moment_kipin']],
                                   terms['reference_resultants'],atol=1e-11)
        np.testing.assert_allclose(np.array(out['known_transverse_shear_integrals'])-np.array(original['slab']['integral'][6:]),
                                   20*(base[6:]+5*slope[6:]),atol=1e-12)
        self.assertEqual(out['local_transverse_shear_status'],'available')
        wrong=copy.deepcopy(cut);wrong['Ms_transverse_first_moment_kipin']+=1
        with self.assertRaises(ValueError):physical_response_terms(response,wrong,branch_id='collectorEq',source_prefix='wrong')


if __name__=='__main__':unittest.main()
