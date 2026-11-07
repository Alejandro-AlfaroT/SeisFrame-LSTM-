import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import json
import unittest

import numpy as np
from Design.SMRF_Interval_Material import (
    Gradient, Interval, UnsupportedBox, concrete_density_enclosure,
    concrete_stress_gradient, replay_density_enclosure, steel_stress_gradient)
from Design.SMRF_Layered_Slab_Section import concrete_response


class IntervalMaterialTests(unittest.TestCase):
    def test_outward_basic_arithmetic_against_exact_rationals(self):
        a, b = Interval(.1), Interval(7)
        exact_a = Fraction(.1)
        for enclosed, exact in ((a / b, exact_a / 7), (a * b, exact_a * 7),
                                (a + b, exact_a + 7), (b - a, 7 - exact_a)):
            self.assertLessEqual(Fraction(enclosed.lo), exact)
            self.assertGreaterEqual(Fraction(enclosed.hi), exact)
        squared = Interval('-2', '3').square()
        self.assertEqual(squared.record(), ['0', '9'])
        with self.assertRaises(ValueError):
            Interval(-1, 1).reciprocal()

    def test_sqrt_and_transcendentals_have_independent_oracles(self):
        enclosed = Interval(2).sqrt()
        self.assertLess(Fraction(enclosed.lo)**2, 2)
        self.assertGreater(Fraction(enclosed.hi)**2, 2)
        # exp(ln(2)) must contain the exact integer, despite both roundings.
        self.assertTrue(Interval(2).log().exp().contains(2))
        with localcontext() as ctx:
            ctx.prec = 110
            self.assertTrue(Interval('0.7').exp().contains(Decimal('0.7').exp()))
            self.assertTrue(Interval('1.3').log().contains(Decimal('1.3').ln()))

    def test_diagonal_material_and_tangent_against_closed_form(self):
        args = [Gradient.variable((v, v), i, 3)
                for i, v in enumerate((-.0004, .0002, 0.))]
        xx, yy, xy = concrete_stress_gradient(*args, fc_ksi=4., peak_strain=.002)
        with localcontext() as ctx:
            ctx.prec = 110
            fc, peak, e = map(Decimal, (4., .002, -.0004))
            n = Decimal(.8) + fc * Decimal(6.894757293168) / 17
            r = -e / peak
            rn = (n * r.ln()).exp()
            stress = -fc * n * r / (n - 1 + rn)
            derivative = fc * n * (n - 1) * (1 - rn) / (peak * (n - 1 + rn)**2)
            shear_derivative = -stress / (2 * (Decimal(.0002) - e))
        self.assertTrue(xx.value.contains(stress))
        self.assertTrue(xx.derivatives[0].contains(derivative))
        self.assertTrue(xy.derivatives[2].contains(shear_derivative))
        self.assertTrue(yy.value.contains(0))
        self.assertTrue(xy.value.contains(0))

    def test_rotated_softened_and_biaxial_boxes_contain_production_stresses(self):
        centers = [(-.0005, .00025, .0003), (-.0005, .0018, .0004),
                   (-.0008, -.0003, .0002), (.0003, -.0006, -.0003)]
        for center in centers:
            radius = 1e-6
            args = [Gradient.variable((v - radius, v + radius), i, 3)
                    for i, v in enumerate(center)]
            bounded = concrete_stress_gradient(*args, fc_ksi=4., peak_strain=.002)
            for offsets in itertools.product((-.5, 0., .5), repeat=3):
                ex, ey, g = np.array(center) + radius * np.array(offsets)
                e = np.array([[ex, g / 2], [g / 2, ey]])
                value = concrete_response(e, 4., peak_strain=.002,
                    compression_strain_limit=.002, model='polak1993_thorenfeldt')['stress_ksi']
                for item, actual in zip(bounded, value[(0, 1, 0), (0, 1, 1)]):
                    self.assertTrue(item.value.contains(float(actual)))

    def test_smooth_box_tangents_contain_independent_numerical_differences(self):
        center = np.array([-.0005, .0018, .0004]);radius = 2e-6
        bounded = concrete_stress_gradient(*[Gradient.variable((v-radius, v+radius), i, 3)
            for i, v in enumerate(center)], fc_ksi=4., peak_strain=.002)
        def response(x):
            ex, ey, g = x
            c = concrete_response(np.array([[ex, g/2], [g/2, ey]]), 4., peak_strain=.002,
                compression_strain_limit=.002, model='polak1993_thorenfeldt')['stress_ksi']
            return c[(0, 1, 0), (0, 1, 1)]
        for j in range(3):
            d = np.eye(3)[j] * 1e-9
            derivative = (response(center+d)-response(center-d))/(2e-9)
            for row, value in zip(bounded, derivative):
                self.assertTrue(row.derivatives[j].contains(float(value)))

    def test_repeated_tension_zero_and_unproved_compression_refusal(self):
        tensile = [Gradient.variable((v, v), i, 3) for i, v in enumerate((.001, .001, 0.))]
        bounded = concrete_stress_gradient(*tensile, fc_ksi=4., peak_strain=.002)
        self.assertTrue(all(v.value.record() == ['0', '0'] for v in bounded))
        compressive = [Gradient.variable((v, v), i, 3) for i, v in enumerate((-.001, -.001, 0.))]
        with self.assertRaises(UnsupportedBox):
            concrete_stress_gradient(*compressive, fc_ksi=4., peak_strain=.002)

    def test_physical_limits_and_branch_crossings_are_not_claimed_failures(self):
        for ex in [(-.003, -.0025), (-1e-6, 1e-6)]:
            args = [Gradient.variable(ex, 0, 3), Gradient.variable((.001, .001), 1, 3),
                    Gradient.variable((0, 0), 2, 3)]
            with self.assertRaises(UnsupportedBox):
                concrete_stress_gradient(*args, fc_ksi=4., peak_strain=.002)
        threshold = .2/.34*.002
        args = [Gradient.variable((-.0004, -.0004), 0, 3),
                Gradient.variable((threshold-1e-6, threshold+1e-6), 1, 3),
                Gradient.variable((0, 0), 2, 3)]
        with self.assertRaises(UnsupportedBox):
            concrete_stress_gradient(*args, fc_ksi=4., peak_strain=.002)

    def test_steel_elastic_yielded_and_unsupported_boxes(self):
        for bounds, expected, tangent in [((.001, .0011), (29., 31.9), 29000),
                                          ((.003, .004), (60, 60), 0),
                                          ((-.004, -.003), (-60, -60), 0)]:
            stress = steel_stress_gradient(Gradient.variable(bounds, 0, 1), fy_ksi=60.)
            # Decimal literals give exact engineering values for the endpoint oracle.
            if tangent == 0:
                self.assertTrue(all(stress.value.contains(v) for v in expected))
            self.assertTrue(stress.derivatives[0].contains(tangent))
        for bounds in ((.001, .003), (.009, .011)):
            with self.assertRaises(UnsupportedBox):
                steel_stress_gradient(Gradient.variable(bounds, 0, 1), fy_ksi=60.)

    def test_seven_resultant_density_axes_and_saved_reconstruction(self):
        q = [(-.0004, -.0004), (.0002, .0002), (0, 0)] + [(0, 0)] * 4
        record = concrete_density_enclosure(q, (3, 3), (2, 2), axis='x', reference_s_in=1,
                                            fc_ksi=4., peak_strain=.002)
        self.assertEqual(record['status'], 'smooth_material_box_enclosed')
        self.assertFalse(record['continuous_capacity_verified'])
        self.assertFalse(record['section_integration_verified'])
        self.assertEqual(replay_density_enclosure(json.loads(json.dumps(record))), record)
        a, b = record['nominal_density'][0], record['nominal_density'][3]
        self.assertEqual(Interval(*b), 2 * Interval(*a))
        self.assertEqual(record['nominal_density'][3], record['nominal_density'][6])
        altered = copy.deepcopy(record);altered['continuous_capacity_verified'] = True
        with self.assertRaises(ValueError):
            replay_density_enclosure(altered)
        web = concrete_density_enclosure(q, (3, 3), (2, 2), axis='x', reference_s_in=1,
                                         fc_ksi=4., peak_strain=.002, region='web')
        self.assertEqual(web['status'], 'smooth_material_box_enclosed')
        self.assertEqual(web['nominal_density'][1], ['0', '0'])

    def test_unproved_density_retains_unknown_and_invalid_inputs_refused(self):
        q = [(-.001, -.001), (-.001, -.001), (0, 0)] + [(0, 0)] * 4
        record = concrete_density_enclosure(q, (0, 1), (0, 1), axis='x', reference_s_in=0,
                                            fc_ksi=4., peak_strain=.002)
        self.assertEqual(record['status'], 'unknown')
        self.assertFalse(record['physical_failure_claimed'])
        with self.assertRaises(ValueError):
            concrete_density_enclosure(q, (0, 1), (0, 1), axis='x', reference_s_in=0,
                                       fc_ksi=-4., peak_strain=.002, region='web')
        for value in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                Interval(value)


if __name__ == '__main__':
    unittest.main()
