"""Isolate column PDelta versus Linear in the unchanged wider bare frame.

An isolated diagnostic process substitutes only the requested column transform
while retaining the saved transfer, diaphragm and gross stiffness. No coupled
forces are transplanted, no design demand is corrected, and no gate is set.
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
from Design.SMRF_Elastic import build_design_model, physical_members
from Loads.Gravity_Loads import apply_gravity_loads
from Analysis.Gravity import run_gravity_analysis
from Model.Build_Model import node_tag


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def frame(case, record, transform):
    sp.BEAM_STIFFNESS_MODIFIER = sp.COLUMN_STIFFNESS_MODIFIER = 1.
    pattern = case['live_pattern']
    if isinstance(pattern, list):
        reference = record['coupled_comparison']['asymmetric']
        if pattern != reference['case']['live_pattern']:
            raise ValueError('No saved transfer pattern matches the requested panels')
        pattern = reference['pattern']['id']
    actual = ops.geomTransf
    substitutions = []

    def declare(kind, tag, *args):
        if tag == sp.COL_TRANSF_TAG:
            if kind != 'PDelta':
                raise ValueError('Unexpected baseline column transformation')
            substitutions.append(dict(tag=tag, original=kind, used=transform))
            return actual(transform, tag, *args)
        return actual(kind, tag, *args)

    try:
        with mock.patch.object(ops, 'geomTransf', side_effect=declare), contextlib.redirect_stdout(io.StringIO()) as log:
            build_design_model()
            apply_gravity_loads(floor_factor=1., self_weight_factor=case['dead_factor'],
                               dead_factor=case['dead_factor'], live_factor=case['live_factor'], live_pattern=pattern)
            analysis = run_gravity_analysis()
            ops.reactions()
        if len(substitutions) != 1:
            raise RuntimeError('Expected exactly one column transformation declaration')
        return dict(status='completed', analysis=analysis, substitutions=substitutions,
                    columns={str(tag): list(ops.eleResponse(tag, 'localForce'))
                             for tag, _, _, kind in physical_members() if kind == 'column'},
                    base_reactions=[dict(grid_i=i, grid_j=j, reaction=ops.nodeReaction(node_tag(0, i, j)))
                                    for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1)],
                    log=log.getvalue())
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
        raise ValueError('This experiment is bounded to the unchanged wider candidate')
    record = json.loads(raw)
    g = record['geometry']
    go.apply_geometry_overrides({upper: g[lower] for lower, upper in
                                [('num_bay_x', 'NUM_BAY_X'), ('num_bay_y', 'NUM_BAY_Y'), ('num_floor', 'NUM_FLOOR'),
                                 ('bay_x_in', 'BAY_X'), ('bay_y_in', 'BAY_Y'), ('story_h_in', 'STORY_H')]},
                               variant_name='fixed_wider_transform_sensitivity', emit=False)
    sp.FLOOR_LIVE_LOAD_KSF = record['floor_loads']['floor_live_load_ksf']
    apply_design(record)
    governing = {}
    for combo in record['design_actions']['combinations']:
        for tag, member in combo['members'].items():
            if member['member_type'] != 'column':
                continue
            f = member['local_force_kip_kipin']
            for end, indexes in [('i', (4, 5)), ('j', (10, 11))]:
                key = (tag, end)
                governing[key] = max(governing.get(key, 0.), math.hypot(*(f[i] for i in indexes)))
    summary = dict(source_design_sha256=digest, engineering_verified=False, production_enabled=False,
                   attempts=[], cases={}, source_hashes={str(args.design): sha(args.design)},
                   interpretation='Controlled transformation sensitivity only; no corrected design actions or P-M checks.')
    summary['source_hashes'].update({str(p.relative_to(RC)): sha(p) for p in (RC / 'Design').glob('*.py')})
    per_story = (g['num_bay_x'] + 1) * (g['num_bay_y'] + 1)
    for case in (record['coupled_comparison']['case'], record['coupled_comparison']['asymmetric']['case']):
        name = case['id']
        reference_path = args.references / (name + '_frame.json')
        coupled_path = args.references / (name + '_coupled_m10.json.gz')
        summary['source_hashes'].update({str(p): sha(p) for p in (reference_path, coupled_path)})
        reference = json.loads(reference_path.read_text())
        coupled = json.loads(gzip.decompress(coupled_path.read_bytes()))
        if (coupled['inputs']['geometry'] != g or coupled['inputs']['floor_loadcases'] != [case] * g['num_floor']
                or coupled['status'] != 'diagnostic_complete'):
            raise ValueError('Saved coupled model does not match the fixed comparison')
        variants = {}
        for transform in ('PDelta', 'Linear'):
            attempt = dict(case=name, transform=transform, status='started')
            summary['attempts'].append(attempt)
            write(args.output / 'summary.json', summary)
            started = time.perf_counter()
            try:
                variants[transform] = frame(case, record, transform)
                write(args.output / (name + '_' + transform + '.json'), variants[transform])
                attempt['status'] = 'completed'
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                attempt['elapsed_seconds'] = time.perf_counter() - started
                write(args.output / 'summary.json', summary)
        baseline_difference = max(abs(f[i] - reference['columns'][tag][i])
                                  for tag, f in variants['PDelta']['columns'].items() for i in range(12))
        if baseline_difference > 1e-7:
            raise RuntimeError(f'Baseline does not reproduce transferred frame forces: {baseline_difference}')
        rows = []
        for col in coupled['column_actions']:
            tag = str((col['story'] - 1) * per_story + col['grid_j'] * (g['num_bay_x'] + 1) + col['grid_i'] + 1)
            for end, indexes in [('i', (4, 5)), ('j', (10, 11))]:
                p, l, c = [[f[i] for i in indexes] for f in (variants['PDelta']['columns'][tag],
                          variants['Linear']['columns'][tag], col['local_force_kip_kip_in'])]
                denominator = governing[tag, end]
                if denominator <= 0:
                    raise ValueError('A governing design moment is required for every compared end')
                rows.append(dict(tag=int(tag), story=col['story'], grid_i=col['grid_i'], grid_j=col['grid_j'],
                                 end=end, pdelta_moments=p, linear_moments=l, coupled_moments=c,
                                 governing_saved_design_moment=denominator,
                                 transform_change_over_governing=math.dist(p, l) / denominator,
                                 pdelta_gap_over_governing=math.dist(p, c) / denominator,
                                 linear_gap_over_governing=math.dist(l, c) / denominator))
        write(args.output / (name + '_all_column_ends.json'), rows)
        summary['cases'][name] = dict(column_end_count=len(rows), baseline_max_absolute_difference=baseline_difference,
                                    maximum_transform_change=max(rows, key=lambda r: r['transform_change_over_governing']),
                                    maximum_original_gap=max(rows, key=lambda r: r['pdelta_gap_over_governing']),
                                    maximum_matched_linear_gap=max(rows, key=lambda r: r['linear_gap_over_governing']))
        write(args.output / 'summary.json', summary)
        print(json.dumps({'case': name, **summary['cases'][name]}), flush=True)
    if hashlib.sha256(gzip.decompress(args.design.read_bytes())).hexdigest() != digest:
        raise RuntimeError('Source design changed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
