"""Plot the most damaged beam hinge from verified histories; scoring is unchanged."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

RC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RC))
from Data_Generation import Run_Weighted_Seismic_Calibration as runner
from Data_Generation import Collect_Weighted_Seismic_Calibration as collector
from Data_Generation.Seismic_Response_Score import default_policy


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


BEAM_TYPES = {'beam', 'beam_x', 'beam_y'}


def peak_row(backbone, member_scope='beam'):
    if member_scope not in ('beam', 'all'):
        raise ValueError('Unsupported hinge selection scope')
    if not backbone:
        raise ValueError('No hinge damage rows')
    tags = [int(r['hinge_ele_tag']) for r in backbone]
    damage = [float(r['damage_ratio']) for r in backbone]
    if len(set(tags)) != len(tags) or any(not math.isfinite(x) or x < 0 for x in damage):
        raise ValueError('Invalid or duplicate hinge damage rows')
    eligible = backbone
    if member_scope == 'beam':
        if any(r.get('member_type') not in BEAM_TYPES | {'column'} for r in backbone):
            raise ValueError('Missing or unsupported member type; cannot identify beam hinges')
        eligible = [r for r in backbone if r['member_type'] in BEAM_TYPES]
        if not eligible:
            raise ValueError('No beam hinges in saved backbone; no column fallback')
    chosen = min(eligible, key=lambda r: (-float(r['damage_ratio']), int(r['hinge_ele_tag'])))
    ties = sorted(int(r['hinge_ele_tag']) for r in eligible
                  if float(r['damage_ratio']) == float(chosen['damage_ratio']))
    return chosen, ties


def select_histories(arrays, springs, backbone_row):
    """Match by explicit hinge/axis identities, never by a maximum-moment column."""
    t = arrays['time_sec']
    rotation, moment = arrays['rotation_rad'], arrays['moment_kip_in']
    if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or rotation.ndim != 2 or rotation.shape != moment.shape or rotation.shape[0] != len(t)
            or not np.isfinite(rotation).all() or not np.isfinite(moment).all()):
        raise ValueError('Invalid or unsynchronized moment/rotation histories')
    count = rotation.shape[1]
    for key in ('spring_hinge_element_tag', 'spring_axis_direction', 'spring_member_tag', 'spring_end_id'):
        if arrays[key].shape != (count,):
            raise ValueError('Spring identity array shape mismatch')
    by_index = {int(row['spring_index']): row for row in springs}
    if len(by_index) != len(springs) or set(by_index) != set(range(count)):
        raise ValueError('Spring table does not cover the history columns uniquely')
    tag = int(backbone_row['hinge_ele_tag'])
    columns = np.flatnonzero(arrays['spring_hinge_element_tag'] == tag)
    if len(columns) != 2 or set(arrays['spring_axis_direction'][columns].tolist()) != {5, 6}:
        raise ValueError('Selected hinge needs exactly one local y and one local z spring')
    selected = {}
    for column in columns:
        column = int(column)
        direction = int(arrays['spring_axis_direction'][column])
        axis = {5: 'y', 6: 'z'}[direction]
        row = by_index[column]
        if (int(row['hinge_element_tag']) != tag or row['local_axis'] != axis
                or int(row['spring_local_direction']) != direction
                or int(row['physical_member_tag']) != int(arrays['spring_member_tag'][column])
                or int(row['end_id']) != int(arrays['spring_end_id'][column])
                or int(row['physical_member_tag']) != int(backbone_row['ele_tag'])
                or int(row['end_id']) != int(backbone_row['end_id'])):
            raise ValueError('Hinge/backbone/spring identities disagree')
        selected[axis] = {'column': column, 'spring': row,
                          'rotation': rotation[:, column], 'moment': moment[:, column]}
    observed = max(float(np.abs(v['rotation']).max()) for v in selected.values())
    if not math.isclose(observed, float(backbone_row['rot_abs_max']), rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError('Saved damage envelope does not match the scheduled hinge history')
    return selected


def location(ntha, spring):
    result = {'joint_node': spring.get('joint_node'), 'end': spring['end'],
              'member_type': spring.get('member_type'), 'physical_member_tag': int(spring['physical_member_tag'])}
    path = ntha/'nodes.csv'
    if path.exists():
        matches = [r for r in rows(path) if r['node_tag'] == spring.get('joint_node')]
        if len(matches) == 1:
            node = matches[0]
            result.update(floor=int(node['floor']), grid_i_zero_based=int(node['grid_i']),
                          grid_j_zero_based=int(node['grid_j']),
                          xyz_in=[float(node[k]) for k in ('x_in', 'y_in', 'z_in')])
    path = ntha/'elements.csv'
    if path.exists():
        members = [r for r in rows(path) if int(r['ele_tag']) == result['physical_member_tag']]
        if len(members) == 1:
            result['element'] = members[0]
            endpoint = members[0]['node_i' if spring['end'] == 'i' else 'node_j']
            if endpoint != spring.get('joint_node'):
                raise ValueError('Member end does not match the selected joint node')
    return result


def require_separate_output(output, protected):
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError('Plots must be outside the original analysis/device artifact directories')
    return output


def render_manifest(manifest_path, output, policy=None, context=None, member_scope='beam'):
    manifest_path = Path(manifest_path).resolve()
    output = require_separate_output(output, [manifest_path.parent])
    policy = default_policy() if policy is None else policy
    assessment = runner.assess_trial(manifest_path, policy)
    if not assessment.get('usable'):
        raise ValueError('Unusable trial: ' + ', '.join(assessment.get('rejection_reasons', [])))
    manifest = runner.read_json(manifest_path)
    ntha = manifest_path.parent/'ntha'
    files = [manifest_path, *[ntha/name for name in ('hinge_backbone.csv', 'hinge_springs.csv',
             'hinge_moment_rotation.npz', 'hinge_moment_rotation_schema.json', 'response_arrays.npz')]]
    files += [ntha/name for name in ('nodes.csv', 'elements.csv', 'global_parameters.json', 'status.json') if (ntha/name).exists()]
    hashes = {str(p): runner.digest(p) for p in files}
    backbone = rows(ntha/'hinge_backbone.csv')
    structure_peak, _ = peak_row(backbone, member_scope='all')
    if not math.isclose(float(structure_peak['damage_ratio']), assessment['metrics']['peak_hinge_damage_ratio'], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError('Peak hinge does not reproduce the calibration damage metric')
    chosen, ties = peak_row(backbone, member_scope=member_scope)
    schema = runner.read_json(ntha/'hinge_moment_rotation_schema.json')
    if (schema.get('schema_version') != 'hinge_moment_rotation_v1' or schema.get('available') is not True
            or schema['status'].get('failed') is not False or schema['status'].get('truncated') is not False):
        raise ValueError('Moment/rotation export is not a complete supported history')
    with np.load(ntha/'hinge_moment_rotation.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    selected = select_histories(arrays, rows(ntha/'hinge_springs.csv'), chosen)
    if any(v['spring'].get('member_type') != chosen.get('member_type') for v in selected.values()):
        raise ValueError('Backbone and spring member types disagree')
    t = arrays['time_sec']
    if len(t) != manifest['solver']['completed_steps'] or len(t) != schema['status']['completed_steps']:
        raise ValueError('History length differs from the completed analysis')
    # Independently match the plotting export to the arrays used by scoring.
    with np.load(ntha/'response_arrays.npz', allow_pickle=False) as compact:
        order = compact['hinge_tag_order'].tolist()
        index = order.index(int(chosen['hinge_ele_tag']))
        if (not np.allclose(t, compact['hinge_history_time'], rtol=0, atol=1e-9)
                or not np.array_equal(arrays['commit_count'], compact['hinge_history_commit_count'])):
            raise ValueError('Scoring and plotting histories use different committed states')
        for axis, offset in (('y', 0), ('z', 1)):
            for name, compact_name in (('rotation', 'hinge_rotation'), ('moment', 'hinge_moment')):
                if not np.allclose(selected[axis][name], compact[compact_name][:, 2*index+offset], rtol=2e-7, atol=1e-12):
                    raise ValueError('Plot values disagree with the scoring history')
    controlling = max(('y', 'z'), key=lambda a: float(np.abs(selected[a]['rotation']).max()))
    peak_index = int(np.argmax(np.abs(selected[controlling]['rotation'])))
    where = location(ntha, selected['y']['spring'])
    if 'element' in where and where['element'].get('element_type') != chosen.get('member_type'):
        raise ValueError('Backbone and element member types disagree')
    context = dict(context or {})
    identity = {'schema_version': 'peak_hinge_plot_v2', 'case_id': manifest.get('case_id'),
                'selection_scope': member_scope,
                'context': context, 'input_sha256': hashes, 'plotter_sha256': runner.digest(__file__)}
    report_path = output/'peak_hinge.json'
    if report_path.exists():
        old = runner.read_json(report_path)
        if old.get('identity') != identity:
            raise ValueError('Plot directory belongs to different inputs/settings; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'peak_hinge_history.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_sec', 'commit_count', 'rotation_y_rad', 'moment_y_kip_in',
                         'rotation_z_rad', 'moment_z_kip_in'])
        writer.writerows(zip(t, arrays['commit_count'], selected['y']['rotation'], selected['y']['moment'],
                            selected['z']['rotation'], selected['z']['moment']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    colors = {'y': '#176b87', 'z': '#b45b28'}
    for col, axis in enumerate(('y', 'z')):
        value = selected[axis]; theta = value['rotation']; moment = value['moment']/12
        ip = int(np.argmax(np.abs(theta)))
        name = f'Local {axis}' + (' | largest rotation' if axis == controlling else '')
        axes[0, col].plot(theta, moment, color=colors[axis], lw=.9)
        axes[0, col].scatter(theta[ip], moment[ip], color='#bf2747', s=32, zorder=3, label=f'Peak |rotation| at {t[ip]:.3f} s')
        axes[0, col].set(title=name, xlabel='Total spring rotation (rad)', ylabel='Conjugate moment (kip-ft)')
        axes[0, col].legend(fontsize=8)
        axes[1, col].plot(t, moment, color=colors[axis], lw=.9)
        axes[1, col].set(xlabel='Time (s)', ylabel='Conjugate moment (kip-ft)')
        axes[2, col].plot(t, theta, color=colors[axis], lw=.9)
        axes[2, col].set(xlabel='Time (s)', ylabel='Total spring rotation (rad)')
        for row in (1, 2): axes[row, col].axvline(t[ip], color='#bf2747', ls=':', lw=.9)
        for ax in axes[:, col]:
            ax.grid(alpha=.22); ax.axhline(0, color='#999999', lw=.5)
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 4), useOffset=False)
    loc = f"node {where.get('joint_node', '?')}"
    if 'xyz_in' in where:
        x, y, z = [v/12 for v in where['xyz_in']]
        loc = f"{'Base' if where['floor'] == 0 else 'Floor ' + str(where['floor'])}; (x, y, z) = ({x:g}, {y:g}, {z:g}) ft"
    trial_label = context.get('label') or manifest.get('label') or manifest_path.parent.name
    if 'scale' in context:
        trial_label = f"Pair {context['result_id']} | scale {context['scale']:g} | trial {context['trial']}"
        if context.get('search_stop_reason') == 'unusable_trial':
            trial_label += ' | usable trial from FAILED search'
        elif context.get('selected_trial'):
            trial_label += ' | selected calibration trial'
        elif not context.get('search_target_reached'):
            trial_label += ' | search target unmet'
    noun = 'beam hinge' if member_scope == 'beam' else 'hinge'
    title = f'Peak-damage {noun}' if float(chosen['damage_ratio']) > 0 else f'Zero damage proxy: representative tied {noun}'
    fig.suptitle(f"{manifest.get('case_id', 'Saved run')} | {title} {chosen['hinge_ele_tag']}\n"
                 f"{where['member_type']} {where['physical_member_tag']}, end {where['end']} | {loc}\n"
                 f"Damage proxy = {float(chosen['damage_ratio']):.4g} | {trial_label}", fontsize=12)
    fig.text(.5, .012, 'Scheduled committed snapshots; absolute spring response includes gravity. Damage is a rotation-demand proxy, not fatigue damage.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .96), h_pad=2.0)
    png = output/'peak_hinge_history.png'
    fig.savefig(png, dpi=160, facecolor='white'); plt.close(fig)
    if any(runner.digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('Input files changed during plotting; plots are not verified')
    report = {'identity': identity, 'hinge_element_tag': int(chosen['hinge_ele_tag']), 'location': where,
              'damage_ratio': float(chosen['damage_ratio']), 'peak_rotation_axis': controlling,
              'peak_rotation_time_sec': float(t[peak_index]), 'damage_tied_hinge_tags': ties,
              'selection_scope': member_scope,
              'selection_rule': f'maximum recorded damage ratio among {"beam hinge ends only" if member_scope == "beam" else "all hinge ends"}; exact ties use smallest hinge tag',
              'structure_peak_hinge_element_tag': int(structure_peak['hinge_ele_tag']),
              'structure_peak_hinge_damage_ratio': float(structure_peak['damage_ratio']),
              'damage_basis': 'max(0, rot_abs_max - theta_y_spring) / theta_p; hinge-end proxy shared across its axes',
              'history_basis': 'absolute, gravity included, scheduled committed snapshots; no sign reversal or interpolation',
              'samples': len(t), 'backbone_row': chosen, 'spring_rows': {a: v['spring'] for a, v in selected.items()},
              'score': assessment['score'], 'analysis_profile': manifest.get('analysis_profile'),
              'outputs_sha256': {p.name: runner.digest(p) for p in (png, csv_path)}, 'training_release': False}
    runner.write_json(report_path, report)
    return report


def batch(plan_path, roots, output, member_scope='beam'):
    plan = runner.validate_plan(runner.read_json(plan_path))
    output = require_separate_output(output, roots)
    report_path = output/'peak_hinge_plots.json'
    if report_path.exists():
        old = runner.read_json(report_path)
        if old.get('plan_sha256') != plan['plan_sha256'] or old.get('selection_scope', 'all') != member_scope:
            raise ValueError('Plot summary belongs to another plan or selection scope; choose a new output directory')
    jobs = {j['job_id']: j for j in plan['jobs']}
    summary = {'plan_sha256': plan['plan_sha256'], 'selection_scope': member_scope,
               'rendered': [], 'skipped': [], 'errors': [], 'training_release': False}
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        try:
            identity = runner.read_json(root/'identity.json')
            shard = identity['shard']
            if shard in seen or identity != {'plan_sha256': plan['plan_sha256'], 'shard': shard, 'frozen': plan['frozen']}:
                raise ValueError('Duplicate device or mismatching frozen device identity')
            seen.add(shard)
            progress = runner.read_json(root/'progress.json')
            if progress.get('plan_sha256') != plan['plan_sha256'] or progress.get('shard') != shard:
                raise ValueError('Progress identity mismatch')
            for saved in progress.get('searches', []):
                name = saved['job_id']; job = jobs[name]
                if job['shard'] != shard:
                    raise ValueError('Search belongs to another device')
                if saved.get('status') != 'completed':
                    summary['skipped'].append({'job_id': name, 'reason': 'search unfinished'})
                    continue
                search = collector.verify_search(root, job, progress, plan['score_policy'], plan['search'])
                for trial in search['trials']:
                    if not trial.get('usable'):
                        summary['skipped'].append({'job_id': name, 'trial': trial['trial'], 'reason': 'unusable trial'})
                        continue
                    context = {**job, 'trial': trial['trial'], 'scale': trial['scale'],
                               'selected_trial': trial == search.get('best_trial'), 'search_stop_reason': search['stop_reason'],
                               'search_target_reached': search['target_reached']}
                    folder = output/f'device_{shard:02d}'/name/f"trial_{trial['trial']:02d}"
                    try:
                        report = render_manifest(collector.artifact_path(root, trial['manifest']), folder, plan['score_policy'], context, member_scope)
                        summary['rendered'].append({'job_id': name, 'trial': trial['trial'], 'scale': trial['scale'],
                            'selected_trial': context['selected_trial'], 'hinge_element_tag': report['hinge_element_tag'],
                            'damage_ratio': report['damage_ratio'], 'member_type': report['location']['member_type'],
                            'structure_peak_hinge_damage_ratio': report['structure_peak_hinge_damage_ratio'],
                            'png': str(folder/'peak_hinge_history.png')})
                    except Exception as exc:
                        summary['errors'].append({'job_id': name, 'trial': trial['trial'], 'error': f'{type(exc).__name__}: {exc}'})
        except Exception as exc:
            summary['errors'].append({'device_root': str(root), 'error': f'{type(exc).__name__}: {exc}'})
    runner.write_json(report_path, summary)
    with (output/'peak_hinge_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['job_id', 'trial', 'scale', 'selected_trial', 'hinge_element_tag', 'damage_ratio',
                  'member_type', 'structure_peak_hinge_damage_ratio', 'png']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summary['rendered'])
    print(f"Peak-hinge plots: {len(summary['rendered'])} rendered, {len(summary['skipped'])} skipped, {len(summary['errors'])} errors. {output}")
    return 1 if summary['errors'] else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--plan', type=Path)
    p.add_argument('--device', type=int, choices=range(1, 5))
    p.add_argument('--device-root', action='append', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--label')
    p.add_argument('--member-scope', choices=('beam', 'all'), default='beam',
                   help='Default: most damaged beam hinge; all reproduces structure-wide selection. Scores never change.')
    args = p.parse_args(argv)
    if args.manifest:
        if not args.output: p.error('--manifest requires a separate --output directory')
        result = render_manifest(args.manifest, args.output, context={'label': args.label} if args.label else {}, member_scope=args.member_scope)
        print(f"Hinge {result['hinge_element_tag']}; damage {result['damage_ratio']:.6g}; {args.output}")
        return 0
    plan_path = runner.resolve(args.plan)
    plan = runner.read_json(plan_path)
    if args.device and args.device_root: p.error('Use --device or --device-root, not both')
    roots = [runner.resolve(r) for r in args.device_root] if args.device_root else [runner.resolve(plan['output_root'])/f'device_{n:02d}' for n in ([args.device] if args.device else range(1, plan['shards']+1))]
    default = runner.resolve(plan['output_root'] + ('_beam_plots' if args.member_scope == 'beam' else '_plots'))
    if args.device: default /= f'device_{args.device:02d}_report'
    return batch(plan_path, roots, runner.resolve(args.output) if args.output else default, args.member_scope)


if __name__ == '__main__':
    raise SystemExit(main())
