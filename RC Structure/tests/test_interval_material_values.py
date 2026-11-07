import itertools
import json
import unittest
import numpy as np
from Design.SMRF_Interval_Material import (Interval, Gradient, UnsupportedBox,
    concrete_stress_gradient,
    concrete_stress_bounds, steel_stress_bounds, concrete_density_value_enclosure,
    replay_density_value_enclosure)
from Design.SMRF_Layered_Slab_Section import concrete_response


class MaterialValueBoxTests(unittest.TestCase):
    def test_zero_tension_stress_does_not_bypass_backbone_domain(self):
        strain=np.array([[.001,0.],[0.,.001]])
        actual=concrete_response(strain,.1,peak_strain=.002,compression_strain_limit=.002,
                                 model='polak1993_thorenfeldt')
        self.assertFalse(actual['backbone_valid'])
        self.assertTrue(np.array_equal(actual['stress_ksi'],np.zeros((2,2))))
        with self.assertRaises(UnsupportedBox):
            concrete_stress_bounds(.001,.001,0.,fc_ksi=.1,peak_strain=.002)
        with self.assertRaises(UnsupportedBox):
            concrete_stress_gradient(*[Gradient.variable((v,v),i,3) for i,v in enumerate((.001,.001,0.))],
                                     fc_ksi=.1,peak_strain=.002)

    def test_branch_crossing_and_repeated_root_boxes(self):
        boxes = [((-.00001, .00001), (.0002, .0003), (-.00001, .00001)),
                 ((-.0011, -.0009), (-.0011, -.0009), (-.0001, .0001)),
                 ((-.0006, -.0004), (.0011, .0013), (-.0001, .0001)),
                 ((-.00001, .00001), (-.00001, .00001), (-.00001, .00001))]
        for box in boxes:
            bounds = concrete_stress_bounds(*map(lambda v: Interval(*v), box),
                                             fc_ksi=4., peak_strain=.002)
            for fractions in itertools.product((.25, .5, .75), repeat=3):
                ex, ey, g = [a + (b-a)*f for (a,b),f in zip(box, fractions)]
                value = concrete_response(np.array([[ex,g/2],[g/2,ey]]),4.,peak_strain=.002,
                    compression_strain_limit=.002,model='polak1993_thorenfeldt')['stress_ksi']
                for bound, actual in zip(bounds, value[(0,1,0),(0,1,1)]):
                    self.assertTrue(bound.contains(float(actual)))

    def test_repeated_compression_enclosure_shrinks_without_arbitrary_axis(self):
        widths=[]
        for radius in (1e-4,1e-5,1e-6):
            q=Interval(-.001-radius,-.001+radius)
            result=concrete_stress_bounds(q,q,Interval(-radius,radius),fc_ksi=4.,peak_strain=.002)
            widths.append(float(result[0].hi-result[0].lo))
            self.assertEqual(result[0],result[1])
            self.assertTrue(result[2].contains(0))
        self.assertGreater(widths[0],widths[1]);self.assertGreater(widths[1],widths[2])
        exact=concrete_stress_bounds(0,0,0,fc_ksi=4.,peak_strain=.002)
        self.assertTrue(all(v.contains(0) for v in exact))

    def test_steel_clipped_union_and_physical_domain(self):
        actual=steel_stress_bounds(Interval('-.003','.003'),fy_ksi=60.)
        self.assertEqual(actual,Interval(-60,60))
        for e in (-.003,-.002,0,.001,.003):
            self.assertTrue(actual.contains(float(np.clip(e*29000.,-60,60))))
        with self.assertRaises(UnsupportedBox):
            steel_stress_bounds(Interval('-.02','.003'),fy_ksi=60.)
        with self.assertRaises(UnsupportedBox):
            concrete_stress_bounds(-.003,0,0,fc_ksi=4.,peak_strain=.002)

    def test_seven_resultant_continuous_values_replay_without_capacity_claim(self):
        q=[(-.0011,-.0009),(-.0011,-.0009),(-.0001,.0001)]+[(0,0)]*4
        result=concrete_density_value_enclosure(q,(1,2),(-2,-1),axis='y',reference_s_in=0,
                                               fc_ksi=4.,peak_strain=.002)
        self.assertEqual(result['status'],'continuous_material_value_box_enclosed')
        self.assertFalse(result['derivative_enclosure_provided'])
        self.assertFalse(result['continuous_capacity_verified'])
        self.assertEqual(replay_density_value_enclosure(json.loads(json.dumps(result))),result)
        result['continuous_capacity_verified']=True
        with self.assertRaises(ValueError):replay_density_value_enclosure(result)


if __name__=='__main__':unittest.main()
