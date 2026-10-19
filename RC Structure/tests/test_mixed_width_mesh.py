"""Mixed-width coordinate regressions; no design acceptance is inferred."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Floor_Mesh import (graded_face_levels, graded_mixed_face_levels,
    resolve_recipe_plan, floor_mesh, nested_refinement)
from Design.SMRF_Slab_Refinement import _plan

POLICY = dict(recipe="graded_face_v1", levels=4, max_shells=130000,
              moment_tolerance=.05, shear_tolerance=.05, tolerance_basis="Implementation fixture")


def widths(nx, ny):
    return {"face_widths_in": {"x": [18.] + [[16., 18.] + [16.] * (ny - 2)] * (nx - 1) + [18.],
                              "y": [18.] + [[18., 16.] + [16.] * (nx - 2)] * (ny - 1) + [18.]}}


class MixedWidthMesh(unittest.TestCase):
    def test_single_width_retains_original_coordinates_exactly(self):
        for length in (216., 252., 342., 360.):
            for width in (12., 16., 18., 22., 30.):
                self.assertEqual(graded_mixed_face_levels(length, [width, width]),
                                 graded_face_levels(length, width))

    def test_case0003_all_four_levels_fit_without_changing_budget(self):
        g = dict(num_bay_x=7, num_bay_y=4, bay_x_in=252., bay_y_in=342.)
        s = widths(7, 4); before = copy.deepcopy(s)
        plan = resolve_recipe_plan(g, s, POLICY)
        self.assertEqual([r['shell_count'] for r in plan['resolved_levels']],
                         [5488, 18928, 75712, 129472])
        self.assertEqual(plan['dropped_levels'], [])
        grids, resolution = _plan(g, dict(POLICY, local_extension=dict(
            method="failed_witness_face_band_v1", max_extra_levels=2)), sections=s)
        self.assertEqual(resolution['status'], 'resolved')
        self.assertEqual(len(grids), 4)
        self.assertEqual(s, before)
        fine = grids[-1]
        self.assertGreaterEqual(fine['minimum_cell_width_in'], .25 - 1e-10)
        self.assertLess(fine['maximum_cell_aspect_ratio'], 58.)
        for a, b in zip(grids, grids[1:]): nested_refinement(a, b)

    def test_three_widths_no_float_slivers_and_every_face_refines(self):
        rows = graded_mixed_face_levels(252., [16., 12., 14., 12.])
        self.assertEqual([len(r)-1 for r in rows], [16, 28, 56, 76])
        for row in rows:
            self.assertTrue(all(b-a > 1e-8 for a,b in zip(row,row[1:])))
            self.assertTrue(all(f in row for f in (6.,7.,8.,244.,245.,246.)))
            self.assertTrue(all(any(abs(252.-x-y)<1e-10 for y in row) for x in row))
        # Real one-inch face gaps are bisected at both fine levels, never omitted to meet a budget.
        self.assertIn(6.5, rows[2]); self.assertIn(6.25, rows[3])

    def test_near_coincident_real_faces_refused_instead_of_silently_merged(self):
        with self.assertRaisesRegex(ValueError, 'wider than 1e-8'):
            graded_mixed_face_levels(252., [16., 16. + 1e-10])
        for bad in ([], [16., float('nan')], [16., -2.], [16., 130.]):
            with self.subTest(widths=bad), self.assertRaises(ValueError):
                graded_mixed_face_levels(252., bad)

    def test_all_3_to_7_bay_combinations_preserve_faces_nesting_and_caps(self):
        for nx in range(3, 8):
            for ny in range(3, 8):
                g = dict(num_bay_x=nx, num_bay_y=ny, bay_x_in=252., bay_y_in=342.)
                s = widths(nx, ny)
                p = resolve_recipe_plan(g, s, POLICY)
                self.assertEqual([v['shell_count'] for v in p['resolved_levels'] + p['dropped_levels']],
                                 [nx*ny*n*n for n in (14,26,52,68)])
                grids = [floor_mesh(nx,ny,252.,342.,4,mesh_spec=m) for m in p['meshes']]
                for grid in grids:
                    self.assertLessEqual(grid['shell_count'], 130000)
                    for axis,n,length in [('x',nx,252.),('y',ny,342.)]:
                        for line,entry in enumerate(s['face_widths_in'][axis]):
                            for w in entry if isinstance(entry,list) else [entry]:
                                if line: self.assertIn(line*length-w/2,grid[axis+'_coordinates_in'])
                                if line<n:self.assertIn(line*length+w/2,grid[axis+'_coordinates_in'])
                for a,b in zip(grids,grids[1:]):nested_refinement(a,b)
                if len(grids)<4:
                    _, resolved = _plan(g,dict(POLICY,local_extension=dict(
                        method='failed_witness_face_band_v1',max_extra_levels=2)),sections=s)
                    self.assertEqual(resolved['status'],'unresolved_budget')

    def test_axis_transposition_preserves_coordinates_and_shell_count(self):
        g=dict(num_bay_x=3,num_bay_y=5,bay_x_in=228.,bay_y_in=354.)
        s=widths(3,5);s['face_widths_in']['y']=[16.]*6
        a=resolve_recipe_plan(g,s,POLICY)
        b=resolve_recipe_plan(dict(num_bay_x=5,num_bay_y=3,bay_x_in=354.,bay_y_in=228.),
            {'face_widths_in':{'x':s['face_widths_in']['y'],'y':s['face_widths_in']['x']}},POLICY)
        for x,y in zip(a['meshes'],b['meshes']):
            self.assertEqual(x['x_offsets_in'],y['y_offsets_in'])
            self.assertEqual(x['y_offsets_in'],y['x_offsets_in'])


if __name__ == '__main__':unittest.main()
