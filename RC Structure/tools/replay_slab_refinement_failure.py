"""Replay one saved slab-refinement failure at its exact sections and thickness."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe')
    parser.add_argument('--max-shells', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from Design.Verify_Designs import configure_case, probe_config
    from Design import Design_Driver as driver
    from Design.SMRF_Slab_Refinement import build_refined_slab_action_evidence, refinement_verified
    from Design.SMRF_Slab_Reinforcement import design_slab_reinforcement
    import Structure_Parameters as sp
    old = json.loads((args.case_dir / 'result.json').read_text(encoding='utf-8-sig'))
    failure = json.loads((args.case_dir / old['failure_evidence']).read_text(encoding='utf-8-sig'))
    configure_case(old['case'], old['profile_id'])
    driver._apply_rung(failure['column_section'], 'column')
    driver._apply_rung(failure['beam_section'], 'beam')
    cfg = probe_config(old['probe_date'])
    if args.recipe:
        cfg.floor_analysis.slab_refinement['recipe'] = args.recipe
    if args.max_shells is not None:
        cfg.floor_analysis.slab_refinement['max_shells'] = args.max_shells
    slab = driver._select_slab(cfg, failure['thickness_in'])
    assert slab['thickness_in'] == failure['thickness_in']
    inputs = driver._slab_strength_inputs_from_state(slab, cfg)
    sections = driver._state_record_core()['sections']
    request = dict(case=old['case'], slab=slab, sections=sections, policy=cfg.floor_analysis.slab_refinement,
                   source_sha256=driver.source_sha256(), original_failure=failure['refinement'])
    (args.output / 'input.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    start = time.perf_counter()
    evidence = build_refined_slab_action_evidence(slab, driver._slab_geometry(), sections,
        sp.FLOOR_LIVE_LOAD_KSF, inputs, cfg.floor_analysis.slab_refinement, assertions=asdict(cfg.slab_actions))
    with gzip.open(args.output / 'evidence.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(evidence, handle)
    report = evidence['refinement']
    reinforcement = design_slab_reinforcement(inputs, evidence, None, driver._slab_completion_context(slab, cfg))
    (args.output / 'slab_reinforcement.json').write_text(json.dumps(reinforcement, indent=2), encoding='utf-8')
    summary = dict(case=old['case']['case_id'], status=report['status'], verified=refinement_verified(evidence),
        elapsed_s=time.perf_counter()-start, source_sha256=request['source_sha256'],
        layout_selected=reinforcement['layout'] is not None,
        reinforcement_unresolved_checks=[dict(id=c['id'], status=c['status']) for c in reinforcement['checks']
                                        if c['status'] != 'pass'],
        levels=[dict(mesh=l['requested_mesh']['subdivisions_per_bay'], status=l['status'],
                     shells=l['requested_mesh']['shell_count'], elapsed_s=l['elapsed_seconds'], error=l.get('error'))
                for l in report['levels']], comparisons=[])
    for comparison in report['comparisons']:
        rows = comparison['comparisons']
        summary['comparisons'].append(dict(failed=sum(not r['within_tolerance'] for r in rows),
            total=len(rows), max_moment=max(r['relative_change'] or 0 for r in rows if r['metric']=='mu_kip_in_per_ft'),
            max_shear=max(r['relative_change'] or 0 for r in rows if r['metric']=='vu_kip_per_ft')))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'}, indent=2), flush=True)
    return 0 if summary['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
