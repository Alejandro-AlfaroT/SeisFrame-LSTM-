"""Count mesh coordinates from saved candidates without importing OpenSees or solving.

The common-interior/explicit-face comparison is a geometric diagnostic only.
It is not installed in the floor model and has no convergence qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RC))
from Design import SMRF_Floor_Mesh as mesh
from Design import SMRF_Floor_Sections as floors
from Model import Member_Groups as groups

POLICY = dict(recipe='graded_face_v1', levels=4, max_shells=130000)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def layout(name):
    return 'perimeter_core_lines' if 'core_lines' in name else 'perimeter_core' if 'core' in name else 'perimeter'


def geometry(case):
    return dict(num_bay_x=case['num_bay_x'], num_bay_y=case['num_bay_y'],
                bay_x_in=case['bay_x_width_ft'] * 12, bay_y_in=case['bay_y_width_ft'] * 12)


def description(case, sections, band, arrangement):
    nx, ny = case['num_bay_x'], case['num_bay_y']
    def beam(axis, line, span):
        i, j = (span, line) if axis == 'x' else (line, span)
        kind = 'beam_' + axis
        location = groups.beam_location(kind, i, j, nx, ny, arrangement)
        section = sections[f'{band}__{kind}__{location}']
        return {k: section[k] for k in ('b_in', 'h_in', 'fc_ksi')}
    lines = {a: [[beam(a, line, span) for span in range(nx if a == 'x' else ny)]
                 for line in range((ny if a == 'x' else nx) + 1)] for a in ('x', 'y')}
    schema = floors.SPAN_SCHEMA
    if all(all(b == row[0] for b in row) for a in lines for row in lines[a]):
        schema = floors.SCHEMA
        lines = {a: [row[0] for row in lines[a]] for a in lines}
    supports = [[{k: sections[f'{band}__column__{groups.column_location(i,j,nx,ny,arrangement)}'][k]
                  for k in ('b_in', 'h_in')} for i in range(nx + 1)] for j in range(ny + 1)]
    fc = next(v['fc_ksi'] for k, v in sections.items() if '__beam_' in k)
    return floors.validate(dict(schema=schema, beam_lines=lines, supports=supports, fc_beam_ksi=fc), nx, ny)


def values(entry):
    return entry if isinstance(entry, list) else [entry]


def metrics(geom, widths, plans):
    """Coordinate checks only. No finite elements, stiffness matrices or solvers."""
    results = []
    previous = None
    for level in range(4):
        axes, deltas = {}, {}
        missed = []; near_zero = []
        for axis in ('x', 'y'):
            n, length = geom['num_bay_' + axis], geom['bay_' + axis + '_in']
            offsets = plans[axis][level]
            axes[axis] = [i * length + x for i, row in enumerate(offsets) for x in row[:-1]] + [n * length]
            deltas[axis] = [b-a for a,b in zip(axes[axis], axes[axis][1:])]
            for line, entry in enumerate(widths[axis]):
                for width in values(entry):
                    for side in (-1, 1):
                        if (line == 0 and side == -1) or (line == n and side == 1): continue
                        face = line * length + side * width / 2
                        if not any(abs(x-face) < 1e-8 for x in axes[axis]): missed.append([axis, line, width, side])
            near_zero += [[axis, v] for v in deltas[axis] if v <= 1e-8]
        count = (len(axes['x']) - 1) * (len(axes['y']) - 1)
        nested = previous is None or all(set(previous[a]).issubset(axes[a]) for a in axes)
        results.append(dict(level=level, cells_per_bay=[len(plans[a][level][0])-1 for a in ('x','y')],
            shell_count=count, within_budget=count <= POLICY['max_shells'], all_physical_faces_present=not missed,
            nested_from_previous=nested, minimum_cell_width_in=min(min(v) for v in deltas.values()),
            maximum_cell_aspect_ratio=max(max(deltas['x'])/min(deltas['y']),max(deltas['y'])/min(deltas['x'])),
            near_zero_cells=near_zero, missing_faces=missed))
        previous = axes
    return results


def inspect(geom, sections):
    widths = floors.face_widths(sections, geom['num_bay_x'], geom['num_bay_y'])
    distinct = {a: sorted({w for entry in widths[a] for w in values(entry)}) for a in widths}
    current = {a: mesh._by_bay_levels(geom['bay_'+a+'_in'], widths[a], 4, mesh.GRADED_FACE_V1) for a in widths}
    levels = metrics(geom, widths, current)
    resolution = mesh.resolve_recipe_plan(geom, sections, POLICY)
    recorded = resolution['resolved_levels'] + resolution['dropped_levels']
    assert [x['shell_count'] for x in recorded] == [x['shell_count'] for x in levels]
    assert all(x['all_physical_faces_present'] and x['nested_from_previous'] for x in levels)
    # Keep one shared interior grading pattern; add every actual physical face.
    # This is an unvalidated cost comparison, not a replacement recipe.
    alternative = {}
    for a in widths:
        length = geom['bay_'+a+'_in']; n = geom['num_bay_'+a]
        base = mesh.graded_face_levels(length, min(distinct[a]))
        anchors = {x for w in distinct[a] for x in (w/2, length-w/2)}
        alternative[a] = [[sorted(set(row) | anchors) for _ in range(n)] for row in base]
    alt_levels = metrics(geom, widths, alternative)
    assert all(x['all_physical_faces_present'] and x['nested_from_previous'] and not x['near_zero_cells'] for x in alt_levels)
    return dict(face_widths_in=widths, distinct_widths_in=distinct,
        mixed_axis={a:any(isinstance(w,list) for w in widths[a]) for a in widths}, levels=levels,
        affordable_levels=len(resolution['meshes']), local_extension_four_level_gate=len(resolution['meshes'])==4,
        common_interior_explicit_faces_geometry_only=alt_levels)


def synthetic(nx, ny, arrangement, sfrs=18., gravity=16., depth_only=False):
    case=dict(num_bay_x=nx,num_bay_y=ny,bay_x_width_ft=21.,bay_y_width_ft=28.5)
    sections={}
    for kind, locations in [('column',('corner','x_boundary_edge','y_boundary_edge','interior','core_corner','core_edge')),
                            ('beam_x',('edge','interior','core')),('beam_y',('edge','interior','core'))]:
        for loc in locations:
            is_sfrs = loc != 'interior'
            width = 24. if kind=='column' else gravity if depth_only else sfrs if is_sfrs else gravity
            height = 24. if kind=='column' else 36. if is_sfrs else 30.
            sections['s01_03__'+kind+'__'+loc]=dict(b_in=width,h_in=height,fc_ksi=8.)
    return case, description(case, sections, 's01_03', arrangement)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--retry',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    source_names=['Design/SMRF_Floor_Mesh.py','Design/SMRF_Floor_Sections.py','Design/SMRF_Slab_Refinement.py',
                  'Design/Config.py','Model/Member_Groups.py','Structure_Parameters.py']
    before={name:sha(RC/name) for name in source_names}
    assert mesh.MAX_EXPLICIT_SHELLS==130000
    cache={};cases=[];inputs={};candidate_count=0
    for label,root in [('original',args.batch),('retry',args.retry)]:
        settings=read(root/'run_settings.json');manifest=read(Path(settings['manifest']))
        inputs[str(Path(settings['manifest']))]=sha(Path(settings['manifest']))
        for case in manifest['cases']:
            folder=root/case['case_id'];geom=geometry(case)
            for p in sorted(folder.glob('grouped_candidates*.jsonl')):
                inputs[str(p)]=sha(p)
                for row in [json.loads(line) for line in p.read_text().splitlines() if line.strip()]:
                    sec=(row.get('evaluation') or {}).get('sections')
                    if not sec:continue
                    candidate_count+=1
                    for band in sorted({k.split('__')[0] for k in sec}):
                        desc=description(case,sec,band,layout(p.name));signature=floors.signature(desc)
                        key=json.dumps([geom,signature],sort_keys=True)
                        if key not in cache:cache[key]=inspect(geom,desc)
                        cases.append(dict(run=label,case_id=case['case_id'],candidate_log=p.name,index=row['index'],
                            phase=row['phase'],decision=row['decision'],evaluation_status=row['evaluation'].get('status'),
                            saved_reason=row['evaluation'].get('reason'),band=band,geometry=geom,layout=layout(p.name),
                            floor_sections_sha256=signature,mesh=cache[key],
                            scope='coordinate reconstruction; no original result is replaced'))
    population=[]
    for nx in range(3,8):
        for ny in range(3,8):
            item=dict(bays=[nx,ny],layouts={})
            for arr in ('perimeter','perimeter_core','perimeter_core_lines'):
                c,d=synthetic(nx,ny,arr);item['layouts'][arr]=inspect(geometry(c),d)
            population.append(item)
    c,d=synthetic(7,4,'perimeter_core',depth_only=True)
    depth_only=inspect(geometry(c),d)
    assert [l['shell_count'] for l in depth_only['levels']]==[4032,16128,64512,100800]
    focal=[r for r in cases if r['run']=='retry' and r['case_id']=='case_0003' and r['candidate_log']=='grouped_candidates_perimeter_core.jsonl' and r['band']=='s01_03' and r['index'] in (4,5)]
    assert len(focal)==2
    f5=next(r for r in focal if r['index']==5)
    assert f5['floor_sections_sha256']=='65223b57913a61280711b521d99606ba583f4632361dfea54381f64c3eec6b44'
    assert [l['shell_count'] for l in f5['mesh']['levels']]==[13552,59248,247408,389872]
    sensitivity=[]
    for width in (16.5,17.,18.,20.,22.):
        c,d=synthetic(7,4,'perimeter_core',sfrs=width)
        sensitivity.append(dict(widths_in=[16.,width],mesh=inspect(geometry(c),d)))
    # A separate synthetic stress case detects float-near-duplicates, not the two-width case-0003 cause.
    stress=mesh._by_bay_levels(252.,[[12.,14.,16.],[12.,14.,16.]],4,mesh.GRADED_FACE_V1)
    stress_report=[dict(level=i,cells_per_bay=len(x[0])-1,near_zero_cell_widths_in=[b-a for a,b in zip(x[0],x[0][1:]) if b-a<=1e-8]) for i,x in enumerate(stress)]
    assert not any(name.startswith('opensees') for name in sys.modules)
    assert before=={name:sha(RC/name) for name in source_names}
    assert all(sha(Path(p))==digest for p,digest in inputs.items())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='mesh-coordinate investigation only; no structural analyses, source-policy changes or design acceptance',
        population_bays=[3,7],shell_budget=130000,source_sha256=before,input_sha256=inputs,
        source_and_saved_inputs_unchanged=True,opensees_imported=False,candidate_section_sets=candidate_count,
        candidate_band_reconstructions=len(cases),distinct_floor_geometries=len(cache),focal_case0003=focal,
        depth_only_control=depth_only,width_separation_sensitivity=sensitivity,synthetic_three_width_float_stress=stress_report,
        population_matrix=population,saved_candidate_reconstructions=cases,
        conclusions=['One mixed-width line triggers an axis-wide union of whole recipes; the same union is replicated in every bay.',
          'Two width recipes usually give 22/46/94/118 cells per bay per mixed axis instead of 12/24/48/60. Counts multiply across axes.',
          'Mixed depths with equal widths do not trigger the union.',
          'With local extension enabled, fewer than four affordable base levels causes unresolved_budget before the structural solves, even when two levels fit.',
          'The common-interior/explicit-face comparison retains physical faces and nested coordinates but is not a converged or accepted analysis mesh.',
          'Large plans can independently exceed 130000 shells even without the mixed-width inflation. Keep the 3-7 population and separate these two causes.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    slim={k:v for k,v in report.items() if k not in ('saved_candidate_reconstructions','population_matrix','width_separation_sensitivity')}
    (args.output/'summary.json').write_text(json.dumps(slim,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('candidate_section_sets','candidate_band_reconstructions','distinct_floor_geometries','source_and_saved_inputs_unchanged','opensees_imported')}))
    for r in focal:
        print('case0003 core-ring',r['index'],[(l['cells_per_bay'],l['shell_count']) for l in r['mesh']['levels']])
    print('geometry-only alternative',[(l['cells_per_bay'],l['shell_count']) for l in f5['mesh']['common_interior_explicit_faces_geometry_only']])


if __name__=='__main__':main()
