"""Review M1 on a saved uniform design, without changing qualification flags.

Run from RC Structure:
  python tools/review_torsional_strength.py outputs/CASE/design.json NEW_OUTPUT

Both 65- and 129-point column P-M approximations are solved. The sidecar is
bound to the exact input bytes and reviewer source. A completed calculation is
supporting evidence for a scoped research-method decision, not a person's
applicability assertion or production release. Outputs must be new directories.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._story_strength_reference import Reference, column_interaction, benchmark

METHOD = 'first_order_planar_line_nominal_strength_reference_v1'
LIMITATIONS = [
    'Directional first-order planar lines with rigid floor translations; not a 3D collapse analysis.',
    'No P-Delta, biaxial P-M-M, finite plastic rotation, cyclic deterioration, or joint/slip yielding.',
    'Uniform sections/cages only; fixed bases; orthogonal framing supplies saved gravity actions only.',
    'Story couples and uniform/height patterns define this review, not a unique code-prescribed story strength.',
    'Numerical refinement is convergence evidence, not a rigorous physical uncertainty bound.',
    'An applicability assertion and production acceptance remain separate from this calculation.',
]


def validate_input(record):
    from Design.Design_Driver import source_sha256
    from Design.Verification_Integrity import validate_record
    from Design.SMRF_Qualification import qualify_design
    from Design.SMRF_Demands import validate_torsion_assessment
    if not __debug__:
        raise ValueError('This reviewer must run with Python assertions enabled.')
    if record.get('member_groups') is not None:
        raise ValueError('Grouped designs are outside this uniform-reference scope.')
    if record['reinforcement']['col_top_bars'] != record['reinforcement']['col_bot_bars']:
        raise ValueError('The column reference requires a symmetric cage.')
    identity = record['request_identity']
    validate_record(record, identity)
    if identity['source_sha256'] != source_sha256():
        raise ValueError('Design source differs from the current source; use its matching checkout.')
    qualification = qualify_design(record)
    if qualification['counts']['fail']:
        raise ValueError('Resolve failed design checks before this M1 review.')
    unknown = [c['id'] for c in qualification['checks']
               if c['status'] == 'not_evaluated' and c['id'] != 'demands.torsional_irregularity']
    if unknown:
        raise ValueError('Other unresolved inputs are outside this review scope: '+str(unknown))
    torsion = record['demand_basis']['torsion']
    assessment = validate_torsion_assessment(torsion, record['geometry']['num_floor'],
                                           identity['policy']['demands']['accidental_torsion_ratio'])
    if not assessment['valid']:
        raise ValueError('Invalid torsion evidence: '+assessment['reason'])
    return assessment

def run(path, out, knots):
    out.mkdir(parents=True, exist_ok=False)
    raw = path.read_bytes()
    record = json.loads(raw)
    curves = {axis: column_interaction(record, axis, knots) for axis in ('x', 'y')}
    cap = {}
    for combo in record['design_actions']['combinations']:
        for tag, member in combo['members'].items():
            if member['member_type'] != 'column':
                continue
            for axis, curve in curves.items():
                for end in ('i', 'j'):
                    p = member['axial_' + end + '_kip']
                    mn = float(np.interp(p, curve['dense_N'], curve['dense_M']))
                    key = axis, int(tag), end
                    if key not in cap:
                        cap[key] = dict(Mn=mn, Nmin=p, Nmax=p)
                    else:
                        cap[key]['Mn'] = min(cap[key]['Mn'], mn)
                        cap[key]['Nmin'] = min(cap[key]['Nmin'], p)
                        cap[key]['Nmax'] = max(cap[key]['Nmax'], p)
    metadata = dict(design_path=str(path), design_sha256=hashlib.sha256(raw).hexdigest(),
                    request_identity=record['request_identity'], knots=knots,
                    benchmark=benchmark(), status='diagnostic_only',
                    kernel_sha256=hashlib.sha256(Path(__file__).with_name('_story_strength_reference.py').read_bytes()).hexdigest())
    (out/'identity.json').write_text(json.dumps(metadata, indent=2))
    rows = []; gravity = []; started = time.perf_counter()
    for axis in ('x', 'y'):
        count = record['geometry']['num_bay_' + ('y' if axis == 'x' else 'x')] + 1
        for load in ('high', 'low'):
            for line in range(count):
                model = Reference(record, cap, axis, line, load, curves[axis])
                check = linprog(np.zeros(model.A.shape[1]), A_eq=model.A, b_eq=model.b,
                                A_ub=model.H, b_ub=model.h,
                                bounds=[(None,None)]*model.A.shape[1], method='highs')
                gravity.append(dict(axis=axis, line=line, gravity=load, success=bool(check.success),
                                    message=check.message, reconstruction_residual=model.frozen_action_reconstruction_residual))
                if not check.success:
                    (out/'gravity.json').write_text(json.dumps(gravity, indent=2))
                    raise RuntimeError(f'Gravity infeasible: {axis} {line} {load}: {check.message}')
                for pattern, stories in [('story_couple', range(1,record['geometry']['num_floor']+1)),
                                         ('uniform', [1]), ('height', [1])]:
                    for story in stories:
                        for sway in (1,-1):
                            result = model.solve(story, sway, pattern)
                            if not result['success']:
                                raise RuntimeError(str((axis,load,line,story,sway,result)))
                            rows.append(result)
                (out/f'{axis}_{load}_{line}.json').write_text(json.dumps(rows[-2*(record['geometry']['num_floor']+2):], allow_nan=False))
                print(path.parent.name, knots, axis, load, line, len(rows), round(time.perf_counter()-started,1), flush=True)
    (out/'gravity.json').write_text(json.dumps(gravity,indent=2))
    fractions = []
    for axis in ('x', 'y'):
        for load in ('high','low'):
            for pattern, stories in [('story_couple', range(1,record['geometry']['num_floor']+1)),('uniform',[1]),('height',[1])]:
                for story in stories:
                    for sway in (1,-1):
                        group=sorted([r for r in rows if (r['axis'],r['gravity'],r['pattern'],r['story'],r['sway'])==(axis,load,pattern,story,sway)],key=lambda r:r['line'])
                        lo=np.array([r['capacity_kip'] for r in group]);hi=np.array([r['continuous_span_upper_bound_kip'] for r in group])
                        assert np.all(lo>0) and np.all(hi>=lo-1e-6)
                        sides=[np.arange(len(lo))<=(len(lo)-1)/2, np.arange(len(lo))>=(len(lo)-1)/2]
                        fractions.append(dict(axis=axis, gravity=load, pattern=pattern, story=story, sway=sway,
                            line_strengths_kip=list(lo), fraction=max(float(lo[s].sum()/lo.sum()) for s in sides),
                            numerical_upper=max(float(hi[s].sum()/(hi[s].sum()+lo[~s].sum())) for s in sides)))
    summary=dict(**metadata, elapsed_s=time.perf_counter()-started, solutions=len(rows),
                 gravity_checks=len(gravity), maximum_fraction=max(fractions,key=lambda r:r['fraction']),
                 maximum_fraction_upper=max(fractions,key=lambda r:r['numerical_upper']),
                 maximum_equilibrium_residual=max(r['equilibrium_residual'] for r in rows),
                 maximum_primal_dual_gap=max(abs(r['primal_dual_gap_kip']) for r in rows),
                 maximum_span_ratio=max(r['maximum_beam_span_ratio'] for r in rows),
                 maximum_pm_ratio=max(r['maximum_column_dense_pm_ratio'] for r in rows),
                 shear_limited_solutions=sum(any(m['mode']=='shear' for m in r['mechanism']) for r in rows),
                 axial_limited_solutions=sum(any(m['mode']=='axial' for m in r['mechanism']) for r in rows))
    (out/'fractions.json').write_text(json.dumps(fractions,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('request_identity','benchmark')},indent=2))
    return summary, fractions


def compare_refinement(coarse, fine, floors, max_fraction_change=0.001):
    """Require complete rosters and stable classification; never turn missing runs into a pass."""
    keys = ('axis','gravity','pattern','story','sway')
    expected = {(a,g,p,s,d) for a in ('x','y') for g in ('high','low')
                for p in ('story_couple','uniform','height')
                for s in (range(1,floors+1) if p == 'story_couple' else [1]) for d in (1,-1)}
    maps = []
    for rows in (coarse, fine):
        lookup = {tuple(r[k] for k in keys):r for r in rows}
        if set(lookup) != expected or len(rows) != len(expected):
            raise ValueError('Incomplete or duplicate refinement roster.')
        for row in rows:
            if not (np.isfinite(row['fraction']) and np.isfinite(row['numerical_upper'])
                    and 0 <= row['fraction'] <= row['numerical_upper']+1e-9 <= 1+1e-9):
                raise ValueError('Invalid fraction or numerical bound.')
        maps.append(lookup)
    changes = {k:abs(maps[1][k]['fraction']-maps[0][k]['fraction']) for k in expected}
    envelope = max(max(maps[i][k]['numerical_upper'] for i in (0,1))+changes[k] for k in expected)
    stable = max(changes.values()) <= max_fraction_change
    return dict(maximum_fraction_change=max(changes.values()), fraction_change_tolerance=max_fraction_change,
                maximum_refinement_envelope=envelope, converged=stable,
                strength_absence_supported=stable and envelope <= .75,
                basis='Maximum of both numerical upper fractions plus their observed refinement difference; sensitivity screen, not a formal bound.')


def review(path, output):
    raw=path.read_bytes();record=json.loads(raw)
    assessment=validate_input(record)
    output.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),Path(__file__).with_name('_story_strength_reference.py')]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (output/'status.json').write_text(json.dumps({'status':'running','design_sha256':hashlib.sha256(raw).hexdigest()}))
    try:
        coarse, coarse_rows=run(path,output/'pm65',65)
        fine, fine_rows=run(path,output/'pm129',129)
        comparison=compare_refinement(coarse_rows,fine_rows,record['geometry']['num_floor'])
        if path.read_bytes()!=raw or any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[p.name] for p in source_files):
            raise ValueError('Design or reviewer changed during calculation.')
        validate_input(record)  # Refuse a concurrent design-source change as well.
        mechanism_clear=not any(r['shear_limited_solutions'] or r['axial_limited_solutions'] for r in (coarse,fine))
        supported=comparison['strength_absence_supported'] and mechanism_clear
        absence=supported and assessment['tir']<=1.2
        result=dict(method=METHOD, status='review_supported' if supported else 'needs_engineering_review',
                    design_sha256=hashlib.sha256(raw).hexdigest(), design_path=str(path.resolve()),
                    reviewer_sha256=hashes, refinement=comparison, torsion_assessment=assessment,
                    gravity_checks=fine['gravity_checks'], final_solutions=fine['solutions'],
                    reference_maximum_fraction=fine['maximum_fraction']['fraction'],
                    shear_or_axial_mechanism=not mechanism_clear,
                    classification_under_declared_reference='none' if absence else ('type_1' if assessment['tir']>1.2 else 'unresolved'),
                    original_qualification_changed=False, applicability_assertion_added=False,
                    production_acceptance=False, limitations=LIMITATIONS)
        (output/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (output/'status.json').write_text(json.dumps({'status':'complete','review_status':result['status']}))
        return result
    except Exception as exc:
        (output/'status.json').write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'}))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('design',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=review(args.design,args.output)
    print(json.dumps({k:result[k] for k in ('status','reference_maximum_fraction','classification_under_declared_reference')},indent=2))
