"""Bounded coupled floor secant equilibrium with actual-material residuals.

The update callback computes actual nominal constitutive operators from the
current physical strain field and returns its domain/quadrature/ownership
audits. Convergence uses NEW physical material forces at the current solution,
including every retained interior node and every free column/floor interface.
Small displacement change or a successful linear solver is not the criterion.
No material strength, experiment qualification or code acceptance is asserted.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from Design.SMRF_Paired_Arithmetic import add_pair
from Design.SMRF_Paired_Floor_Operator import paired_floor_model

METHOD='actual_material_force_coupled_floor_secant_iteration_v2_paired_actions'


def solve_constitutive_case(system,update,*,iteration_identity,maximum_iterations,
                            maximum_seconds,imposed_case=None,pdelta=True,
                            force_absolute_tolerance_kip=1e-7,
                            moment_absolute_tolerance_kipin=1e-6,relative_tolerance=1e-7):
    """One coherent gravity or imposed-motion branch, with no hidden retries.

update(floor_row,response,iteration)->{operator,audit}. The audit needs named
method/state identity and computed booleans material_domain_valid,
quadrature_converged,physical_material_ownership_verified. Missing/false audit
keeps the result unresolved. Each call is recorded; iteration/time exhaustion
is incomplete numerical evidence, never a physical capacity failure.
"""
    if len(system.cases)!=1 or not iteration_identity or not callable(update):
        raise ValueError('One simultaneous actual load case, updater and iteration identity required')
    if type(maximum_iterations) is not int or maximum_iterations<1:raise ValueError('Explicit positive iteration bound required')
    values=(maximum_seconds,force_absolute_tolerance_kip,moment_absolute_tolerance_kipin,relative_tolerance)
    if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Explicit finite positive iteration/time/residual bounds required')
    started=time.perf_counter();history=[];last=None
    for iteration in range(1,maximum_iterations+1):
        if time.perf_counter()-started>maximum_seconds:
            reason='declared_constitutive_iteration_time_limit';break
        current=(system.solve(pdelta=pdelta) if imposed_case is None else
                 system.solve_imposed_floor_displacements([imposed_case],pdelta=pdelta))
        last=current
        if current['status']!='equilibrated':
            reason='coupled_linearized_frame_not_equilibrated';break
        frame=current['frame'];state=frame['cases'][0]
        ids=list(map(tuple,frame['physical_dof_ids']));index={v:i for i,v in enumerate(ids)}
        actual_global=np.array(state['physical_residual_or_reaction'],float)
        old_global=actual_global.copy();global_scale=np.maximum(1.,abs(actual_global))
        operators={};audits={};floor_results=[];interior_max=0.;constitutive_ok=True
        for row,record in zip(system.floors,current['floors']):
            floor=row['assembly'];response=record['responses'][0]
            if row['description']['id']!=record['id']:raise ValueError('Coupled result changed floor ordering or identity')
            produced=update(row,response,iteration);model=produced['operator'];audit=produced.get('audit',{})
            if getattr(model,'floor',None) is not floor:raise ValueError('Updated constitutive field has different physical geometry')
            model=paired_floor_model(model,input_identity={'iteration':iteration_identity,'step':iteration,'floor':floor.id})
            if not audit.get('method') or not audit.get('state_identity'):raise ValueError('Actual material audit method/state provenance required')
            flags=('material_domain_valid','quadrature_converged','physical_material_ownership_verified')
            constitutive_ok &=all(audit.get(flag) is True for flag in flags)
            operators[floor.id]=model;audits[floor.id]=audit
            u=np.asarray(response['retained_displacements'],float);ul=np.asarray(response['retained_displacements_lo'],float);loads=row['loads'][0]['loads']
            oh,ol=row['model'].apply_pair(u,ul);ah,al=model.apply_pair(u,ul)
            actual_h,actual_l=add_pair(ah,al,-loads);actual=actual_h+actual_l
            dh,dl=add_pair(ah,al,-oh,-ol);delta=dh+dl
            scale=np.maximum(1.,np.maximum(abs(oh+ol),abs(ah+al)))
            tolerance=np.array([moment_absolute_tolerance_kipin if dof>=4 else force_absolute_tolerance_kip
                                for _,dof in floor.dof_ids])+relative_tolerance*scale
            boundary=set(floor.boundary_dofs)
            interior=[i for i,dof in enumerate(floor.dof_ids) if dof not in boundary]
            ratio=float(np.max(abs(actual[interior])/tolerance[interior],initial=0.))
            interior_max=max(interior_max,ratio)
            for i,dof in enumerate(floor.dof_ids):
                if dof not in boundary:continue
                at=index[dof];actual_global[at]+=delta[i];global_scale[at]+=scale[i]
            actual_response=model.recover_pair(u,ul,row['loads'][0])
            actual_response.update(case_id=response['case_id'],retained_displacements=u.tolist(),retained_displacements_lo=ul.tolist(),
                                   actual_material_audit=audit)
            floor_results.append({**record,'responses':[actual_response],'stiffness':model.metadata,
                'actual_material_interior_residual_ratio':ratio,
                'linearized_condensation':record['condensation'],
                'condensation':None,
                'equilibrium':{'maximum_normalized_actual_material_interior_force_residual':ratio,
                               'basis':'actual material forces at current coupled solution; no stale linearized condensation claim'}})
        nodes={n['id']:n for n in system.definition['nodes']}
        free=[i for i,(node,dof) in enumerate(ids) if nodes[node]['xyz'][2]>0 and
              (imposed_case is None or dof in (3,4,5))]
        tolerance=np.array([moment_absolute_tolerance_kipin if dof>=4 else force_absolute_tolerance_kip
                            for _,dof in ids])+relative_tolerance*global_scale
        interface_max=float(np.max(abs(actual_global[free])/tolerance[free],initial=0.))
        converged=max(interior_max,interface_max)<=1.
        record=dict(iteration=iteration,interior_residual_ratio=interior_max,interface_residual_ratio=interface_max,
                    actual_material_equilibrium_converged=converged,material_audits_passed=constitutive_ok,
                    elapsed_seconds=time.perf_counter()-started,audits=audits)
        history.append(record)
        if converged:
            if not constitutive_ok:
                reason='equilibrium_found_but_material_domain_quadrature_or_ownership_unverified';break
            result=copy.deepcopy(current);result['floors']=floor_results
            result['frame']['cases'][0]['physical_residual_or_reaction']=actual_global.tolist()
            result['frame']['cases'][0]['linearized_physical_residual_or_reaction']=old_global.tolist()
            result.update(status='constitutive_equilibrated',constitutive_iteration=dict(method=METHOD,
                identity=iteration_identity,history=history,converged=True,
                elapsed_seconds=time.perf_counter()-started,force_absolute_tolerance_kip=force_absolute_tolerance_kip,
                moment_absolute_tolerance_kipin=moment_absolute_tolerance_kipin,relative_tolerance=relative_tolerance),
                qualification_asserted=False)
            return result
        system.set_floor_operators(operators,iteration_identity=f'{iteration_identity}/iteration_{iteration}')
    else:reason='declared_constitutive_iteration_count_limit'
    return dict(method=METHOD,status='unresolved',reason=reason,constitutive_iteration=dict(
        identity=iteration_identity,history=history,converged=False,elapsed_seconds=time.perf_counter()-started),
        last_linearized_result=last,qualification_asserted=False,official_pilot_started=False)
