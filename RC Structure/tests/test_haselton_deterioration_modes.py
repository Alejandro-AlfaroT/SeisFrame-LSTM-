import unittest,sys,math
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model import IMK_Calibration as c, IMK_Hinges as h
from Model.IMK_Spring_Mapping import LEGACY_REPLAY
import openseespy.opensees as ops

class HaseltonModes(unittest.TestCase):
    def setUp(self):
        # pre-freeze rule, kept as the explicit legacy replay (frozen IMK V2 calibration, 2026-10-29): nominal
        # lambda x theta_y,member, direct mode, IMKBilin and direct low-level installs
        patcher=mock.patch.object(sp,'IMK_MEMBER_SPRING_CALIBRATION',LEGACY_REPLAY);patcher.start();self.addCleanup(patcher.stop)
    def tearDown(self):ops.wipe()
    def test_calibrated_strength_only_for_both_material_signatures(self):
        with mock.patch.object(sp,'IMK_DETERIORATION_MODE','haselton_2008'):
            for member in ['beam_x','column']:
                b=c.deterioration_for_member(member,.2)
                for kind,expected in [('IMKPeakOriented',[b['lambda_opensees_rad']]*2+[1e12]*2),('IMKBilin',[b['lambda_opensees_rad']]*2+[1e12])]:
                    with mock.patch.object(sp,'IMK_MATERIAL_TYPE',kind),mock.patch.object(ops,'uniaxialMaterial') as fn:h._define_imk_peak_material(1,1e6,100,b)
                    self.assertEqual(list(fn.call_args.args[15:15+len(expected)]),expected)
    def test_direct_mode_preserves_user_coefficients(self):
        with mock.patch.object(sp,'IMK_DETERIORATION_MODE','direct'),mock.patch.object(sp,'IMK_MATERIAL_TYPE','IMKPeakOriented'),mock.patch.object(ops,'uniaxialMaterial') as fn:
            b=c.deterioration_for_member('beam_x',0);h._define_imk_peak_material(1,1e6,100,b)
        self.assertEqual(fn.call_args.args[15:19],tuple(getattr(sp,'IMK_LAMBDA_'+mode) for mode in ['S','C','A','K']))
    def test_strength_loss_persists_with_constant_unloading_slope(self):
        with mock.patch.object(sp,'IMK_DETERIORATION_MODE','haselton_2008'):
            b=c.deterioration_for_member('beam_x',0)
        h._define_imk_peak_material(1,1e6,100,b);ops.testUniaxialMaterial(1)
        old=0;peaks=[];unloading=[]
        for cycle in range(6):
            for target in [.01,0.,-.01,0.]:
                start=old
                for j in range(1,161):
                    ops.setStrain(start+(target-start)*j/160)
                    self.assertTrue(math.isfinite(ops.getStress()))
                    if target==0. and start==.01 and j==1:unloading.append(ops.getTangent())
                old=target
                if target==.01:peaks.append(ops.getStress())
        self.assertLess(peaks[-1],peaks[0])
        self.assertLess(max(abs(k/1e6-1) for k in unloading),1e-8)

if __name__=='__main__':unittest.main()
