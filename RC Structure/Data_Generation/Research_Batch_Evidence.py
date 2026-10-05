"""Portable, fail-closed consumption of the adopted per-design M1 evidence.

Replays qualification arithmetic, not structural analysis or limit optimization.
Historical files and the adopted reviewer implementation are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import tempfile

RC_DIR = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_case_file(case_id, roots, filename):
    found = [Path(root) / case_id / filename for root in (roots or [])
             if (Path(root) / case_id / filename).is_file()]
    if len(found) > 1:
        raise ValueError(f"Duplicate {filename} for {case_id}.")
    return found[0] if found else None


def portable_evidence(addendum):
    """Map original-machine path labels to their role, without following them."""
    normalized = lambda value: str(value).replace('\\', '/')
    design = normalized(addendum['design_path'])
    review = normalized(addendum['review_path']).rsplit('/', 1)[0] + '/'
    result = {}
    for path, sha in addendum['evidence_sha256'].items():
        path = normalized(path)
        if path == design:
            key = 'design'
        elif path.startswith(review):
            suffix = path[len(review):]
            if any(part in ('', '.', '..') for part in suffix.split('/')):
                raise ValueError('Invalid reference evidence path.')
            key = 'review/' + suffix
        elif path.endswith('/torsional_strength_assertion.json') and sha == addendum['assertion_sha256']:
            key = 'assertion'
        else:
            raise ValueError('Unexpected reference evidence path.')
        if key in result:
            raise ValueError('Duplicate reference evidence role.')
        result[key] = sha
    return result


def validated_addendum(design_path, addendum_path, review_folder):
    from tools.assert_torsional_strength import apply_assertion
    addendum_path = Path(addendum_path)
    before = digest(addendum_path)
    saved = json.loads(addendum_path.read_text(encoding='utf-8'))
    assertion = RC_DIR / 'pilots/v2ResearchPilot100/torsional_strength_assertion.json'
    # apply_assertion independently reconstructs the roster, fraction bounds and
    # qualification. It also checks current source and exact adopted tool hashes.
    with tempfile.TemporaryDirectory(prefix='seisframe-m1-check-') as tmp:
        rebuilt = apply_assertion(design_path, review_folder, assertion, Path(tmp) / 'addendum')
    # Qualification helpers use tuples in memory; JSON artifacts encode lists.
    rebuilt = json.loads(json.dumps(rebuilt, allow_nan=False))
    ignored = {'design_path', 'review_path', 'evidence_sha256'}
    if ({k: v for k, v in saved.items() if k not in ignored}
            != {k: v for k, v in rebuilt.items() if k not in ignored}
            or portable_evidence(saved) != portable_evidence(rebuilt)):
        raise ValueError('M1 addendum differs from independently reconstructed qualification/evidence.')
    if before != digest(addendum_path):
        raise ValueError('M1 addendum changed during validation.')
    return {'sha256': before, 'method': saved['method'],
            'qualification': saved['qualification'], 'assertion': saved['assertion'],
            'evidence_sha256': portable_evidence(saved),
            'production_acceptance': False}


def load_case_addendum(case_id, design_path, addendum_roots=None, review_roots=None):
    path = find_case_file(case_id, addendum_roots, 'qualification_addendum.json')
    if path is None:
        return None
    review = find_case_file(case_id, review_roots, 'review.json')
    if review is None:
        raise ValueError(f'{case_id}: matching M1 review evidence is required with the addendum.')
    return validated_addendum(design_path, path, review.parent)


def effective_state(original, addendum):
    if addendum is None:
        return dict(original)
    q = addendum['qualification']
    return {**original, 'accepted': q['accepted'], 'counts': q['counts'],
            'failed_check_ids': [c['id'] for c in q['checks'] if c['status'] == 'fail'],
            'open_check_ids': [c['id'] for c in q['checks'] if c['status'] == 'not_evaluated']}
