"""Controlled bare-frame assumption checks for the fixed wider diagnostic.

These are sensitivity experiments, not alternative approved design choices.
All variants retain the same saved loads and are compared at every column end.
"""
from pathlib import Path
from unittest import mock
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import sys
import time

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC))
import openseespy.opensees as ops
import Structure_Parameters as sp
import Geometry_Overrides as go
from Design.Design_Driver import apply_design
from Design import SMRF_Elastic as elastic
from Design.SMRF_Floor_Analysis import _beam_inputs
from Analysis import Gravity as gravity
from Loads.Gravity_Loads import apply_gravity_loads
from Model.Build_Model import node_tag, diaphragm_master_tag


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


VARIANTS = {
    'saved_baseline': dict(transform='PDelta', constraints='saved', diaphragm='rigid', inertia='frame'),
    'linear_baseline': dict(transform='Linear', constraints='saved', diaphragm='rigid', inertia='frame'),
    'exact_constraints': dict(transform='Linear', constraints='Transformation', diaphragm='rigid', inertia='frame'),
    'free_inplane': dict(transform='Linear', constraints='Transformation', diaphragm='free', inertia='frame'),
    'floor_convention_inertia': dict(transform='Linear', constraints='Transformation', diaphragm='rigid', inertia='floor'),
    'floor_inertia_free_inplane': dict(transform='Linear', constraints='Transformation', diaphragm='free', inertia='floor'),
}


def frame(case, record, variant, floor_beam):
    sp.BEAM_STIFFNESS_MODIFIER = sp.COLUMN_STIFFNESS_MODIFIER = 1.
    pattern = case['live_pattern']
    if isinstance(pattern, list):
        ref = record['coupled_comparison']['asymmetric']
        if pattern != ref['case']['live_pattern']:
            raise ValueError('Requested panel inventory has no corresponding saved transfer')
        pattern = ref['pattern']['id']
    real_transform, real_inertia = ops.geomTransf, sp.beam_flexural_inertia_in4
    changed = []

    def transform(kind, tag, *args):
        if tag == sp.COL_TRANSF_TAG:
            changed.append((kind, variant['transform']))
            return real_transform(variant['transform'], tag, *args)
        return real_transform(kind, tag, *args)

    def free_floor():
        # These independent bookkeeping masters carry no static loads and
        # connect to no elements. Fix only them, leaving all structural joints free.
        for k in range(1, sp.NUM_FLOOR + 1):
            ops.fix(diaphragm_master_tag(k), 1, 1, 1, 1, 1, 1)

    def inertia(axis, position):
        if variant['inertia'] == 'frame':
            return real_inertia(axis, position)
        return floor_beam['line_inertia'][axis + '_' + position]['t_section_gross_in4']

    try:
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as log:
            stack.enter_context(mock.patch.object(ops, 'geomTransf', side_effect=transform))
            stack.enter_context(mock.patch.object(sp, 'beam_flexural_inertia_in4', side_effect=inertia))
            if variant['constraints'] != 'saved':
                stack.enter_context(mock.patch.object(gravity, 'apply_analysis_constraints',
                                                     side_effect=lambda: ops.constraints(variant['constraints'])))
            if variant['diaphragm'] == 'free':
                stack.enter_context(mock.patch.object(elastic, 'create_rigid_diaphragms', side_effect=free_floor))
            elastic.build_design_model()
            apply_gravity_loads(floor_factor=1., self_weight_factor=case['dead_factor'],
                               dead_factor=case['dead_factor'], live_factor=case['live_factor'], live_pattern=pattern)
            analysis = gravity.run_gravity_analysis()
            ops.reactions()
        if len(changed) != 1:
            raise RuntimeError('Column transformation inventory changed unexpectedly')
        columns = {str(tag): list(ops.eleResponse(tag, 'localForce'))
                   for tag, _, _, kind in elastic.physical_members() if kind == 'column'}
        bases = [dict(grid_i=i, grid_j=j, reaction=ops.nodeReaction(node_tag(0, i, j)))
                 for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1)]
        joints = [dict(story=k, grid_i=i, grid_j=j, displacement=ops.nodeDisp(node_tag(k, i, j)))
                  for k in range(1, sp.NUM_FLOOR + 1) for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1)]
        if any(not math.isfinite(v) for f in columns.values() for v in f):
            raise RuntimeError('Nonfinite column force')
        return dict(status='completed', analysis=analysis, variant=variant, columns=columns,
                    base_reactions=bases, joints=joints, log=log.getvalue())
    finally:
        ops.wipe()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=RC / 'outputs/validation_structure_20260924/design.json.gz')
    parser.add_argument('--references', type=Path, default=RC / 'outputs/slab_face_recovery_20260924/compatibility')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = gzip.decompress(args.design.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'f49291440328b9097efd44345246d11a420365a0bbd1dea8c810276890c43871':
        raise ValueError('Protocol bounded to the fixed wider candidate')
    record = json.loads(raw)
    g, slab = record['geometry'], record['slab']
    go.apply_geometry_overrides({upper: g[lower] for lower, upper in
                                [('num_bay_x', 'NUM_BAY_X'), ('num_bay_y', 'NUM_BAY_Y'), ('num_floor', 'NUM_FLOOR'),
                                 ('bay_x_in', 'BAY_X'), ('bay_y_in', 'BAY_Y'), ('story_h_in', 'STORY_H')]},
                               variant_name='fixed_wider_assumption_sensitivity', emit=False)
    sp.FLOOR_LIVE_LOAD_KSF = record['floor_loads']['floor_live_load_ksf']
    apply_design(record)
    floor_beam = _beam_inputs(record['sections'], 'flexible_beams', slab['thickness_in'], g['bay_x_in'], g['bay_y_in'])
    family_audit = {axis + '_' + pos: dict(frame=sp.beam_flexural_section(axis, pos),
                                         floor=floor_beam['line_inertia'][axis + '_' + pos])
                    for axis in ('x', 'y') for pos in ('edge', 'interior')}
    summary = dict(source_design_sha256=digest, production_enabled=False, engineering_verified=False,
                   source_hashes={str(args.design): sha(args.design)}, attempts=[], cases={},
                   family_stiffness=family_audit, variants=VARIANTS,
                   saved_constraint_handler='Penalty' if sp.ELEMENT_FORMULATION == 'imk' else 'Transformation',
                   saved_penalty_parameters=[sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP],
                   limitation='Variations are attribution experiments only, not corrected demands or member-capacity checks.')
    summary['source_hashes'].update({str(p.relative_to(RC)): sha(p) for p in (RC / 'Design').glob('*.py')})
    governing = {}
    for combo in record['design_actions']['combinations']:
        for tag, member in combo['members'].items():
            if member['member_type'] == 'column':
                for end, idx in (('i', (4, 5)), ('j', (10, 11))):
                    governing[tag, end] = max(governing.get((tag, end), 0.),
                                               math.hypot(*(member['local_force_kip_kipin'][i] for i in idx)))
    per_story = (g['num_bay_x'] + 1) * (g['num_bay_y'] + 1)
    for case in (record['coupled_comparison']['case'], record['coupled_comparison']['asymmetric']['case']):
        name = case['id']
        rp = args.references / (name + '_frame.json')
        cp = args.references / (name + '_coupled_m10.json.gz')
        summary['source_hashes'].update({str(p): sha(p) for p in (rp, cp)})
        baseline = json.loads(rp.read_text())
        coupled = json.loads(gzip.decompress(cp.read_bytes()))
        if coupled['inputs']['geometry'] != g or coupled['inputs']['floor_loadcases'] != [case] * g['num_floor']:
            raise ValueError('Coupled reference input mismatch')
        expected_total = coupled['equilibrium']['base_force_kip'][2]
        variants = {}
        summary['cases'][name] = variants
        for key, variant in VARIANTS.items():
            attempt = dict(case=name, variant=key, status='started')
            summary['attempts'].append(attempt)
            save(args.output / 'summary.json', summary)
            start = time.perf_counter()
            try:
                result = frame(case, record, variant, floor_beam)
                total = sum(r['reaction'][2] for r in result['base_reactions'])
                if abs(total / expected_total - 1) > 1e-8:
                    raise RuntimeError('Vertical load ledger mismatch')
                if key == 'saved_baseline':
                    difference = max(abs(f[i] - baseline['columns'][tag][i])
                                     for tag, f in result['columns'].items() for i in range(12))
                    if difference > 1e-7:
                        raise RuntimeError('Saved baseline did not reproduce')
                else:
                    difference = None
                comparisons = []
                for col in coupled['column_actions']:
                    tag = str((col['story'] - 1) * per_story + col['grid_j'] * (g['num_bay_x'] + 1) + col['grid_i'] + 1)
                    for end, idx in (('i', (4, 5)), ('j', (10, 11))):
                        a = [result['columns'][tag][i] for i in idx]
                        b = [col['local_force_kip_kip_in'][i] for i in idx]
                        original = [baseline['columns'][tag][i] for i in idx]
                        denominator = governing[tag, end]
                        if denominator <= 0:
                            raise ValueError('Governing moment required at every column end')
                        comparisons.append(dict(tag=int(tag), story=col['story'], end=end, frame_moments=a,
                                                coupled_moments=b, governing_saved_demand=denominator,
                                                gap_over_governing=math.dist(a, b) / denominator,
                                                change_from_saved_over_governing=math.dist(a, original) / denominator))
                result['comparisons'] = comparisons
                save(args.output / (name + '_' + key + '.json'), result)
                variants[key] = dict(column_end_count=len(comparisons),
                                     maximum_gap=max(comparisons, key=lambda r: r['gap_over_governing']),
                                     maximum_change=max(comparisons, key=lambda r: r['change_from_saved_over_governing']),
                                     vertical_balance_relative_error=abs(total / expected_total - 1),
                                     baseline_max_absolute_difference=difference)
                attempt['status'] = 'completed'
                print(json.dumps(dict(case=name, variant=key,
                                      max_gap=variants[key]['maximum_gap']['gap_over_governing'],
                                      max_change=variants[key]['maximum_change']['change_from_saved_over_governing'])), flush=True)
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                attempt['elapsed_seconds'] = time.perf_counter() - start
                save(args.output / 'summary.json', summary)
    if hashlib.sha256(gzip.decompress(args.design.read_bytes())).hexdigest() != digest:
        raise RuntimeError('Source design changed')
    return 0 if all(a['status'] == 'completed' for a in summary['attempts']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
