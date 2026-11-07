"""Actual physical compatibility fields in the existing member-design contract.

The full G+Cd/Ie response is recovered once. It is never added to primary
earthquake member forces. Every imposed floor displacement is independently
checked at every physical column node before any member strength is read.
"""
from __future__ import annotations
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Member_Actions import build_member_actions

METHOD='actual_physical_compatibility_member_adapter_v1'


def displacement_audit(definition,response,case_index,*,absolute_tolerance=1e-8):
    case=response['cases'][case_index];imposed=case['imposed'];frame=response['frame']
    values=np.asarray(frame['cases'][case_index]['physical_displacements'],float)
    ids=[tuple(v) for v in frame['physical_dof_ids']];index={v:i for i,v in enumerate(ids)}
    if len(index)!=len(ids) or values.shape!=(len(ids),) or not np.isfinite(values).all():
        raise ValueError('Complete unique finite physical displacement inventory required')
    g=definition['geometry'];origin=np.asarray(imposed['reference_xy_in'],float);sign=float(imposed['sign'])
    levels={r['floor']:r['design_in'] for r in imposed['levels']}
    if (not np.isfinite(absolute_tolerance) or absolute_tolerance<=0 or origin.shape!=(2,) or not np.isfinite(origin).all()
        or sign not in (-1.,1.) or len(imposed['levels'])!=g['nf'] or set(levels)!=set(range(1,g['nf']+1))
        or any(type(r['floor']) is not int for r in imposed['levels'])
        or any(set(row)!={'ux','uy','rz_rad'} or not np.isfinite(list(row.values())).all() for row in levels.values())):
        raise ValueError('Every signed design displacement level required')
    rows=[]
    for level in range(1,g['nf']+1):
        target={k:float(v)*sign for k,v in levels[level].items()};residuals=[];nodes=[]
        for node in definition['nodes']:
            x,y,z=node['xyz']
            if abs(z-level*g['story_h'])>1e-8:continue
            reference=np.array([target['ux']-(y-origin[1])*target['rz_rad'],
                                target['uy']+(x-origin[0])*target['rz_rad'],target['rz_rad']])
            actual=values[[index[(node['id'],d)] for d in (1,2,6)]]
            residuals.append(actual-reference);nodes.append(node['id'])
        if len(nodes)!=(g['nx']+1)*(g['ny']+1):raise ValueError('Actual compatibility floor-node inventory differs')
        residual=np.max(abs(np.asarray(residuals)),axis=0)
        if np.any(residual>absolute_tolerance):raise ValueError('Computed physical displacements differ from signed Cd/Ie targets')
        rows.append(dict(floor=level,target_in=target,actual_node_ids=nodes,
            residual_in=dict(ux=float(residual[0]),uy=float(residual[1]),rz_rad=float(residual[2])),
            tolerance=absolute_tolerance))
    return rows


def adapt_compatibility(state,definition,physical,*,cd,importance_factor):
    if not np.isfinite([cd,importance_factor]).all() or min(cd,importance_factor)<=0:
        raise ValueError('Finite positive actual Cd and importance factor required')
    response=physical['response']
    if response.get('status')!='equilibrated':raise ValueError('Complete equilibrated physical compatibility response required')
    cases=[];axes={a:dict(roof_design_displacement_in=0.,case_ids=[]) for a in ('x','y')}
    for i,case in enumerate(response['cases']):
        original=case['imposed'];axis=original['axis'];sign=original['sign']
        if axis not in axes:raise ValueError('Compatibility input must name the retained primary excitation axis')
        combination=dict(id=case['id'],family='compatibility_'+case['gravity_case_id'],
            dead=case['dead'],live=case['live'],live_pattern=case.get('live_pattern','all'),ex=0.,ey=0.)
        adapted=build_member_actions(state,definition,response,combination,gravity_case_id=case['id'])['action']
        audit=displacement_audit(definition,response,i)
        adapted.update(axis=axis,sign=sign,imposed=audit,branch='physical_deformation_compatibility',
            analysis_models={'all':'complete actual slab/web/columns with imposed primary Cd/Ie displacement'},
            compatibility_response_sha256=_hash(response))
        cases.append(adapted);axes[axis]['case_ids'].append(case['id'])
        roof=next(r['design_in']['u'+axis] for r in original['levels'] if r['floor']==definition['geometry']['nf'])
        axes[axis]['roof_design_displacement_in']=max(axes[axis]['roof_design_displacement_in'],abs(roof))
    if any(not row['case_ids'] for row in axes.values()):raise ValueError('Both primary excitation directions required')
    return dict(method=METHOD,cases=cases,solves=0,physical_solves=len(cases),
        design_displacements=dict(cd=cd,importance_factor=importance_factor,axes=axes,solves=0),
        response_sha256=_hash(response),basis='ACI318-19 18.14.2.1; ASCE7-22 12.12.4/.5; actual G+Cd/Ie physical fields')
