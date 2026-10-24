import contextlib, math, sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model import IMK_Hinges as h, IMK_Calibration as c
import openseespy.opensees as ops

class RCCorrections(unittest.TestCase):
    def tearDown(self): ops.wipe()

    def test_peak_signature_and_negative_backbone(self):
        with mock.patch.object(sp,'IMK_MATERIAL_TYPE','IMKPeakOriented'), mock.patch.object(ops,'uniaxialMaterial') as fn:
            h._define_imk_peak_material(1,1e6,100,{'theta_p':.02,'theta_pc':.06,'theta_u':.12,'theta_p_neg':.03,'lambda_opensees_rad':.5},80)
        a=fn.call_args.args
        self.assertEqual(len(a),25)
        self.assertEqual(a[9],.03)
        self.assertEqual(a[12],80)
        self.assertEqual(a[15:19],(.5,)*4)
        self.assertEqual(a[19:23],(1.,)*4)

    def test_energy_conversion_and_numerical_stiffness_independence(self):
        with contextlib.ExitStack() as st:
            for k,v in {'IMK_DETERIORATION_MODE':'haselton_2008','H_BEAM':18.,'BEAM_STIRRUP_SPACING':3.,'IMK_BEAM_THETA_Y':.005}.items():st.enter_context(mock.patch.object(sp,k,v))
            a=c.deterioration_for_member('beam_x',0)
            d=18.-sp.longitudinal_cover_in('beam')      # Eq. 3.20 uses s/d, d = h - d' (2026-10-23; s/h before)
            self.assertAlmostEqual(a['lambda_opensees_rad'],170.7*.10**(3./d)*.005)
            with mock.patch.object(sp,'IMK_HINGE_STIFFNESS_FACTOR',200.):
                self.assertEqual(a,c.deterioration_for_member('beam_x',0))
            self.assertTrue(a['deterioration_beam_extrapolation'])
            self.assertLess(c.deterioration_for_member('beam_x',.5)['lambda_opensees_rad'],a['lambda_opensees_rad'])

    def test_installed_edge_spring_and_transformation_constraints(self):
        with contextlib.ExitStack() as st:
            for k,v in {'IMK_MATERIAL_TYPE':'IMKPeakOriented','IMK_HINGE_STIFFNESS_MODE':'member_stiffness_factor','SLAB_THICKNESS_IN':5.,'B_BEAM':10.,'H_BEAM':18.,'NUM_BAY_X':2,'NUM_BAY_Y':6,'BAY_X':120.,'BAY_Y':168.}.items():st.enter_context(mock.patch.object(sp,k,v))
            for kind,coord,axis in [('beam_x',(120,0,0),3),('beam_y',(0,168,0),3),('column',(0,0,120),1)]:
                ops.wipe(); ops.model('basic','-ndm',3,'-ndf',6)
                ops.node(1,0,0,0);ops.node(2,*coord);ops.fix(1,1,1,1,1,1,1)
                ops.geomTransf('Linear',1,*( (1,0,0) if kind=='column' else (0,0,1)))
                h.reset_hinge_registry();h.create_imk_member(1,1,2,kind,1)
                reg=h.hinge_registry()[1]
                if kind!='column':
                    props=h._member_properties(kind,family=(kind[-1],'edge'))
                    expected=sp.IMK_HINGE_STIFFNESS_FACTOR*6*props['e']*props['iy']/math.dist((0,0,0),coord)
                    self.assertAlmostEqual(reg['ke_y_kip_in_per_rad']/expected,1.)
                    self.assertNotEqual(expected,h.imk_hinge_stiffness(kind,'rot_y',math.dist((0,0,0),coord),family=(kind[-1],'interior')))
                ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
                load=[0.]*6;load[axis-1]=1.;ops.load(2,*load)
                ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral')
                ops.test('NormDispIncr',1e-10,30);ops.algorithm('Newton');ops.integrator('LoadControl',.001);ops.analysis('Static')
                self.assertEqual(ops.analyze(1),0,kind)
                self.assertGreater(ops.nodeDisp(2,axis),0.)

    def test_cyclic_strength_loss_is_active_and_legacy_available(self):
        for kind in ['IMKBilin','IMKPeakOriented']:
            ops.wipe()
            with mock.patch.object(sp,'IMK_MATERIAL_TYPE',kind):
                h._define_imk_peak_material(1,1e6,100,{'theta_p':.08,'theta_pc':.1,'theta_u':.2,'lambda_opensees_rad':.5})
            ops.testUniaxialMaterial(1);old=0.;peaks=[]
            for _ in range(6):
                for target in [.01,0.,-.01,0.]:
                    for j in range(1,201):ops.setStrain(old+(target-old)*j/200)
                    old=target
                    self.assertTrue(math.isfinite(ops.getStress()))
                    if target==.01:peaks.append(ops.getStress())
            self.assertLess(peaks[-1],.99*peaks[0])

if __name__=='__main__':unittest.main()
