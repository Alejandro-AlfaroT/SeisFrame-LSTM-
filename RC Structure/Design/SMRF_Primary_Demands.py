"""Retained ELF torsion/drift on the force-verified actual-gravity primary.

The demand coefficients and fixed perimeter-system assignment are unchanged.
Only the physical gravity initial state and recovered displacement increments
replace the former tributary/transfer-load initial state. No gravity-only
displacement is counted as earthquake drift. All source solves are retained.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash,validate_definition
from Model.Floor_Footprint import from_perimeter_beams
from Design.SMRF_Demands import (story_drift_ratio,amplification_from_level_displacements,
    classify_torsional_irregularity,validate_torsion_assessment,evaluate_drift_and_stability)

METHOD='actual_gravity_primary_ELF_torsion_drift_v1'


def floor_loads(definition,elf,grid_masses,*,ex,ey,accidental_ratio,amplification=1.,eccentricity_sign=1.):
    """Signed ELF force and exact actual-CM transport about the grid master."""
    validate_definition(definition);g=definition['geometry'];nf=g['nf'];forces=elf['story_forces_kip'];weights=elf['story_weights_kip']
    if len(forces)!=nf or len(weights)!=nf or set(grid_masses)!={str(k) for k in range(1,nf+1)}:
        raise ValueError('ELF and actual mass must cover every floor')
    if not all(math.isfinite(float(x)) for x in (ex,ey,accidental_ratio,amplification,eccentricity_sign)) or not 0<=accidental_ratio<=1 or amplification<1 or eccentricity_sign not in (-1.,1.):
        raise ValueError('Explicit finite ELF factors and valid accidental-torsion policy required')
    mx,my=g['nx']*g['lx']/2,g['ny']*g['ly']/2;loads=[];ledger=[]
    for f,force,weight in zip(definition['floors'],forces,weights):
        rows=grid_masses[str(f['level'])];w=np.array([r['weight_kip'] for r in rows]);xy=np.array([r['xy_in'] for r in rows])
        if not np.isfinite(w).all() or not np.isfinite(xy).all() or min(w)<0 or not math.isclose(float(w.sum()),weight,rel_tol=1e-9,abs_tol=1e-8):
            raise ValueError('ELF floor weight differs from actual physical mass inventory')
        center=(w@xy)/w.sum();fp=from_perimeter_beams(g,f['beams']);fx,fy=float(ex*force),float(ey*force)
        accident=eccentricity_sign*accidental_ratio*amplification*((fp['y_max_in']-fp['y_min_in'])*fx+(fp['x_max_in']-fp['x_min_in'])*fy)
        transport=(center[0]-mx)*fy-(center[1]-my)*fx
        loads.append([fx,fy,float(accident+transport)])
        ledger.append(dict(floor=f['level'],mass_center_xy_in=center.tolist(),
            accidental_torque_kipin=accident,mass_transport_torque_kipin=float(transport),weight_kip=float(w.sum())))
    return dict(loads=loads,reference_xy_in=[mx,my],ledger=ledger,
        demand_basis_identity=_hash(dict(elf=elf,definition=definition['identity_sha256'],mass=grid_masses,
            ex=ex,ey=ey,accidental_ratio=accidental_ratio,amplification=amplification,eccentricity_sign=eccentricity_sign)))


def node_displacements(primary,increment):
    """Actual primary increment at every physical grid node; fixed base kept."""
    if increment.get('status')!='equilibrated' or increment.get('primary_definition_sha256')!=primary.definition['identity_sha256']:
        raise ValueError('Actual complete source-matched primary response required')
    g=primary.definition['geometry'];u=np.asarray(increment['displacement_increment']);index=primary.index
    output={k:{} for k in range(g['nf']+1)}
    for node in primary.nodes:
        x,y,z=node['xyz'];level=round(z/g['story_h']);i,j=round(x/g['lx']),round(y/g['ly'])
        output[level][f'{i},{j}']=dict(x=float(u[index[(node['id'],1)]]),y=float(u[index[(node['id'],2)]]))
    return output


def torsion_assessment(primary,definition,elf,mass,*,accidental_ratio,strength_distribution,saved_responses=None):
    rows=[];responses={};cases=[];bycase={};axlevels={};g=definition['geometry']
    for axis in ('x','y'):
        for sign in (1.,-1.):
            key=axis+('+' if sign>0 else '-');cases.append(key)
            load=floor_loads(definition,elf,mass,ex=float(axis=='x'),ey=float(axis=='y'),
                accidental_ratio=accidental_ratio,eccentricity_sign=sign)
            if saved_responses is not None and _hash(saved_responses[key]['load'])!=_hash(load):raise ValueError('Saved torsion load differs from actual ELF/mass/accidental-torsion basis')
            response=primary.solve_increment(load['loads'],**({} if saved_responses is None else dict(saved_increment=saved_responses[key]['response'])))
            nodes=node_displacements(primary,response)
            responses[key]=dict(load=load,response=response);worst=0.
            for k in range(1,g['nf']+1):
                edgea=[f'{i},0' for i in range(g['nx']+1)] if axis=='x' else [f'0,{j}' for j in range(g['ny']+1)]
                edgeb=[f"{i},{g['ny']}" for i in range(g['nx']+1)] if axis=='x' else [f"{g['nx']},{j}" for j in range(g['ny']+1)]
                da=max(abs(nodes[k][n][axis]-nodes[k-1][n][axis]) for n in edgea)
                db=max(abs(nodes[k][n][axis]-nodes[k-1][n][axis]) for n in edgeb)
                ua=max((nodes[k][n][axis] for n in edgea),key=abs);ub=max((nodes[k][n][axis] for n in edgeb),key=abs)
                tir=story_drift_ratio(da,db);level=amplification_from_level_displacements(ua,ub)
                rows.append(dict(story=k,direction=axis,case=key,eccentricity_sign=sign,
                    delta_end_a_in=da,delta_end_b_in=db,delta_max_over_avg=tir,
                    delta_level_a_in=ua,delta_level_b_in=ub,level_max_over_avg=level['ratio'],ax_level=level['ax']))
                worst=max(worst,tir);axlevels[k]=max(axlevels.get(k,0.),level['ax'])
            bycase[key]=worst
    maximum=max(bycase.values());envelope=max(axlevels.values())
    classification=classify_torsional_irregularity(maximum,strength_distribution.get('one_side_fraction'),
        strength_model_verified=False,independent_distribution_bound=strength_distribution.get('independent_distribution_bound'))
    # Retain the full calculated Ax envelope even while the independent
    # strength-distribution classification is unresolved. No demand reduction.
    symmetry=max(abs(a['delta_max_over_avg']-b['delta_max_over_avg']) for a in rows for b in rows
        if a['story']==b['story'] and a['direction']==b['direction'] and a['eccentricity_sign']!=b['eccentricity_sign'])
    result=dict(method=METHOD,ratio=accidental_ratio,assessment_amplification=1.,base='fixed',
        amplification=envelope,amplification_required=envelope,amplification_by_level=axlevels,
        amplification_envelope_12_8_4_3=envelope,amplification_basis='Computed all-case Ax envelope retained without demand reduction',
        tir=maximum,max_drift_ratio=maximum,tir_by_case=bycase,torsional_irregularity=classification['label'],
        classification=classification,strength_distribution=strength_distribution,cases=cases,stories=rows,
        sign_symmetry_residual_max=symmetry,lateral_design_model=primary.definition['lateral_design_model'],
        primary_definition_sha256=primary.definition['identity_sha256'],source_identity=primary.source_identity,
        basis='ASCE7-22 12.3.2.1.1;12.8.4.2/.3, actual-P primary earthquake increments',
        gravity_transfer=primary.baseline['column_axial_ledger'],qualification_asserted=False)
    audit=validate_torsion_assessment(result,g['nf'],policy_ratio=accidental_ratio)
    if not audit['valid']:raise ValueError('Computed actual-P torsion record failed independent replay: '+audit['reason'])
    return result,responses


def drift_screen(primary,definition,elf,mass,*,accidental_ratio,amplification,basis,sdc,cd,gravity_above_by_story,saved_responses=None):
    """Node drift envelope of both actual accidental-eccentricity signs."""
    g=definition['geometry'];stories=[];responses={}
    for axis in ('x','y'):
        node_cases=[]
        for sign in (1.,-1.):
            key=axis+('+' if sign>0 else '-')
            load=floor_loads(definition,elf,mass,ex=float(axis=='x'),ey=float(axis=='y'),
                accidental_ratio=accidental_ratio,amplification=amplification,eccentricity_sign=sign)
            if saved_responses is not None and _hash(saved_responses[key]['load'])!=_hash(load):raise ValueError('Saved drift load differs from actual ELF/mass/amplification basis')
            increment=primary.solve_increment(load['loads'],**({} if saved_responses is None else dict(saved_increment=saved_responses[key]['response'])))
            node_cases.append(node_displacements(primary,increment))
            responses[key]=dict(load=load,response=increment)
        for k in range(1,g['nf']+1):
            node_ids=list(node_cases[0][k]);delta={}
            for node in node_ids:
                delta[node]={d:max((v[k][node][d]-v[k-1][node][d] for v in node_cases),key=abs) for d in ('x','y')}
            stories.append(dict(id=f'{k}/Q{axis}',height_in=g['story_h'],node_deltas_in=delta,expected_node_ids=node_ids,
                directions=[axis],story_shear_kip={axis:sum(elf['story_forces_kip'][k-1:])},gravity_above_kip=gravity_above_by_story[k]))
    limit_sdc=sdc if sdc in ('D','E','F') else 'D'
    result=evaluate_drift_and_stability(stories,cd=cd,importance_factor=basis['importance_factor'],
        risk_category=basis['risk_category'],drift_limit_ratio=basis['allowable_story_drift_ratio'],
        redundancy_factor=basis['redundancy_factor'],seismic_design_category=limit_sdc,second_order_included=True)
    result['assumptions'].update(cd=cd,rho_for_drift_load=1.,rho_for_limit=basis['redundancy_factor'],
        risk_category=basis['risk_category'],importance_factor=basis['importance_factor'],
        base_drift_limit_ratio=basis['allowable_story_drift_ratio'],effective_drift_limit_ratio=basis['effective_story_drift_ratio'],
        low_rise_allowance_asserted=False,limit_seismic_design_category=limit_sdc,seismic_design_category=sdc,
        accidental_torsion_included=accidental_ratio>0,gravity='Actual physical full D+L column initial forces',
        model='Retained primary stiffness/releases, actual-P PDelta, both eccentricity-sign node envelope',
        lateral_design_model=primary.definition['lateral_design_model'])
    result.update(method=METHOD,source_identity=primary.source_identity,primary_definition_sha256=primary.definition['identity_sha256'])
    return result,responses
