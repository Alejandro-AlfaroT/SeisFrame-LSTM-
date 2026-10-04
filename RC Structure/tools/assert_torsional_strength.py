"""Apply an explicitly adopted M1 research method to a completed per-design review.

Writes a qualification ADDENDUM in a new directory. It never rewrites the
historical design, its input identity, or its PROBE assertions. The legacy
beam-proxy verification Boolean is not the assertion adopted here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Common import assertion_provenance_valid, make_check, summarize_checks
from Design.SMRF_Qualification import qualify_design
from tools.review_torsional_strength import METHOD, LIMITATIONS, compare_refinement, validate_input

CHECK_ID = 'demands.torsional_irregularity'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_assertion(assertion, reviewer_hashes):
    if not assertion_provenance_valid(assertion):
        raise ValueError('A named, dated assertion with a basis is required.')
    if assertion.get('method') != METHOD or assertion.get('adopted_for_v2_research') is not True:
        raise ValueError('This reference method has not been adopted for V2 research.')
    if assertion.get('requires_per_design_evidence') is not True:
        raise ValueError('A blanket verification assertion is not supported.')
    if assertion.get('reviewer_sha256') != reviewer_hashes:
        raise ValueError('The reviewed implementation differs from the adopted implementation.')
    if assertion.get('production_release_authorized') is not False:
        raise ValueError('This assertion cannot authorize production release.')


def checked_fraction_rows(folder, geometry, design_hash, reviewer_hashes, knots):
    identity = read(folder/'identity.json')
    if (identity.get('design_sha256') != design_hash or identity.get('knots') != knots
            or identity.get('kernel_sha256') != reviewer_hashes['_story_strength_reference.py']):
        raise ValueError('Reference identity mismatch.')
    nf = geometry['num_floor']
    expected = {(axis, line, gravity, pattern, story, sway)
                for axis in ('x','y')
                for line in range(geometry['num_bay_'+('y' if axis=='x' else 'x')]+1)
                for gravity in ('high','low') for pattern in ('story_couple','uniform','height')
                for story in (range(1,nf+1) if pattern=='story_couple' else [1]) for sway in (1,-1)}
    keys = ('axis','line','gravity','pattern','story','sway')
    lookup = {}
    for path in sorted(folder.glob('*_*.json')):
        for row in read(path):
            key = tuple(row[k] for k in keys)
            if key in lookup or key not in expected:
                raise ValueError('Duplicate or unexpected reference solution.')
            lookup[key] = row
    if set(lookup) != expected:
        raise ValueError('Incomplete reference solution roster.')
    for row in lookup.values():
        cap, upper = row['capacity_kip'], row['continuous_span_upper_bound_kip']
        if not (math.isfinite(cap) and cap>0 and math.isfinite(upper) and upper>=cap-1e-6):
            raise ValueError('Invalid line capacity or numerical bracket.')
        for name, limit in [('equilibrium_residual',1e-6), ('yield_violation',1e-6),
                            ('compatibility_residual',1e-7), ('primal_dual_gap_kip',1e-6*max(1.,cap)),
                            ('maximum_beam_span_ratio',1+1e-8), ('maximum_column_dense_pm_ratio',1+1e-6)]:
            if not math.isfinite(row[name]) or abs(row[name])>limit:
                raise ValueError('Reference numerical verification failed: '+name)
        if row.get('success') is not True or not math.isfinite(row['load_work']) or abs(row['load_work']-1)>1e-8:
            raise ValueError('Unsuccessful reference solution.')
        if any(m['mode'] in ('shear','axial') for m in row['mechanism']):
            raise ValueError('A shear/axial mechanism needs an additional applicability review.')
    gravity = read(folder/'gravity.json')
    expected_gravity = {key[:3] for key in expected}
    actual_gravity = {(r['axis'],r['line'],r['gravity']) for r in gravity}
    if actual_gravity != expected_gravity or len(gravity)!=len(expected_gravity):
        raise ValueError('Incomplete or duplicate gravity-feasibility roster.')
    if any(r.get('success') is not True or not math.isfinite(r['reconstruction_residual'])
           or abs(r['reconstruction_residual'])>1e-6 for r in gravity):
        raise ValueError('Gravity feasibility or force reconstruction failed.')
    groups = sorted({(a,g,p,s,d) for a,l,g,p,s,d in expected})
    rows = []
    for axis,gravity,pattern,story,sway in groups:
        count=geometry['num_bay_'+('y' if axis=='x' else 'x')]+1
        selected=[lookup[(axis,line,gravity,pattern,story,sway)] for line in range(count)]
        low=[r['capacity_kip'] for r in selected];high=[r['continuous_span_upper_bound_kip'] for r in selected]
        sides=[{i for i in range(count) if i<=(count-1)/2}, {i for i in range(count) if i>=(count-1)/2}]
        fraction=max(sum(low[i] for i in side)/sum(low) for side in sides)
        upper=max(sum(high[i] for i in side)/(sum(high[i] for i in side)+sum(low[i] for i in range(count) if i not in side)) for side in sides)
        rows.append(dict(axis=axis,gravity=gravity,pattern=pattern,story=story,sway=sway,
                         fraction=fraction,numerical_upper=upper))
    return rows


def replace_m1(qualification, check):
    original=[c for c in qualification['checks'] if c['id']==CHECK_ID]
    if len(original)!=1 or original[0]['status']!='not_evaluated':
        raise ValueError('Expected exactly one open M1 item; other states need review.')
    checks=[check if c['id']==CHECK_ID else c for c in qualification['checks']]
    return {**qualification, 'checks':checks, **summarize_checks(checks)}


def apply_assertion(design_path, review_folder, assertion_path, output):
    design_path,review_folder,assertion_path,output=map(Path,(design_path,review_folder,assertion_path,output))
    if output.exists():
        raise FileExistsError('Choose a new addendum output directory.')
    inputs = [design_path,assertion_path,*sorted(p for p in review_folder.rglob('*.json'))]
    hashes = {str(p.resolve()):digest(p) for p in inputs}
    record=read(design_path);assertion=read(assertion_path);review=read(review_folder/'review.json')
    reviewer_hashes={name:digest(Path(__file__).with_name(name))
                     for name in ('review_torsional_strength.py','_story_strength_reference.py')}
    validate_assertion(assertion,reviewer_hashes)
    if (review.get('method')!=METHOD or review.get('status')!='review_supported'
            or review.get('design_sha256')!=digest(design_path)
            or review.get('reviewer_sha256')!=reviewer_hashes
            or read(review_folder/'status.json').get('status')!='complete'):
        raise ValueError('No completed matching review supports this assertion.')
    assessment=validate_input(record)
    if assessment['tir']>1.2:
        raise ValueError('This scoped absence assertion does not cover TIR above 1.2.')
    rows=[checked_fraction_rows(review_folder/f'pm{k}',record['geometry'],digest(design_path),reviewer_hashes,k) for k in (65,129)]
    refinement=compare_refinement(*rows,record['geometry']['num_floor'])
    if not refinement['strength_absence_supported']:
        raise ValueError('The per-design strength/refinement evidence does not support absence.')
    applied=record['demand_basis']['torsion']['amplification']
    if isinstance(applied,bool) or not isinstance(applied,(int,float)) or not math.isfinite(applied) or applied<1.0:
        raise ValueError('Applied torsion amplification is missing or insufficient.')
    details=dict(method=METHOD, assertion=assertion, design_sha256=digest(design_path),
                 review_sha256=digest(review_folder/'review.json'), classification='none',
                 tir=assessment['tir'], reference_strength_fraction=max(r['fraction'] for r in rows[1]),
                 refinement=refinement, scope='V2 research applicability only', limitations=LIMITATIONS,
                 legacy_beam_proxy_verified=False)
    check=make_check(CHECK_ID,'ASCE 7-22 Table 12.3-1 Type 1; adopted per-design research reference',
                     applied,1.,'>=','Ax',details=details)
    original=qualify_design(record)
    effective=replace_m1(original,check)
    if any(digest(p)!=sha for p,sha in hashes.items()):
        raise ValueError('Input changed during assertion application.')
    if reviewer_hashes != {name:digest(Path(__file__).with_name(name)) for name in reviewer_hashes}:
        raise ValueError('Reviewer source changed during assertion application.')
    result=dict(schema='seisframe_m1_qualification_addendum_v1',method=METHOD,
                design_sha256=digest(design_path), design_path=str(design_path.resolve()),
                assertion_sha256=digest(assertion_path), assertion=assertion,
                review_path=str((review_folder/'review.json').resolve()),
                evidence_sha256=hashes, evaluator_sha256=digest(__file__),
                original_counts=original['counts'], qualification=effective,
                m1_closed=check['status']=='pass', research_checklist_complete=effective['accepted'],
                production_acceptance=False, historical_design_modified=False,
                integration='Supplemental qualification; consumers must explicitly read this addendum. The historical design/result remain unchanged.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'qualification_addendum.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','review','assertion','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();result=apply_assertion(args.design,args.review,args.assertion,args.output)
    print(json.dumps({k:result[k] for k in ('m1_closed','research_checklist_complete','production_acceptance','original_counts')},indent=2))
    print('Supplemental counts:',result['qualification']['counts'])
