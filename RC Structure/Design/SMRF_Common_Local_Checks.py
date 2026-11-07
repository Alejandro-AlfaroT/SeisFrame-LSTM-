"""Complete actual-support slab shear and local steel inventory.

All physical supports, complete simultaneous cases and both element traces
are retained. The prescribed ACI critical-section stress is the strength
quantity; unsmoothed FE peaks remain separate refinement observations.
Full-fy development is a sufficient credit test, not a demand-based failure.
"""
from __future__ import annotations
import math
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Slab_Local_Shear import (critical_perimeter,recover_slab_perimeter,
    check_two_way_strength,local_minimum_steel)

METHOD='all_actual_supports_local_shear_and_steel_v1'


def _bars(material,axis,face,interval,station):
    lo,hi=interval
    return [b for b in material['bars'] if b['operator_owner']=='slab' and b['axis']==axis
        and b['mat_name']==axis+'_'+face and lo+b['db_in']/2-1e-9<=b['s_in']<=hi-b['db_in']/2+1e-9
        and b['straight_run_interval_in'][0]-1e-9<=station<=b['straight_run_interval_in'][1]+1e-9]


def credited_steel(bars,development,station):
    """Return proved full-fy area and the actual installed upper inventory."""
    evidence={i:r for r in development for i in r['physical_bar_ids']};credited=[];unknown=[]
    for bar in bars:
        row=evidence.get(bar['id']);domain=(row or {}).get('full_fy_development_domain',{})
        allowed=bool(row and (row.get('status')=='pass' or (domain.get('status')=='computed'
            and domain['interval_in'][0]-1e-9<=station<=domain['interval_in'][1]+1e-9)))
        (credited if allowed else unknown).append(bar)
    return dict(provided_developed_in2=sum(b['area_in2'] for b in credited),
        installed_area_in2=sum(b['area_in2'] for b in bars),
        credited_bar_ids=[b['id'] for b in credited],unproved_bar_ids=[b['id'] for b in unknown])


def minimum_area_status(required,credit):
    if required<=credit['provided_developed_in2']+1e-10:return 'pass'
    # Even every installed bar at full fy is insufficient: a real bound.
    if required>credit['installed_area_in2']+1e-10:return 'fail'
    return 'unknown'


def support_steel_specs(definition,inventory):
    """Actual support-face/local bands and core-crossing mat stations."""
    specs=[];groups=[];h=inventory['slab_thickness_in'];column_by_node={str(c['j']):c for c in definition['columns']}
    for floor in definition['floors']:
        k=floor['level'];mat=inventory['floors'][str(k)];fp=mat['footprint']
        for support in floor['supports']:
            col=column_by_node[str(support['id'])];group=dict(floor=k,support_id=support['id'],axes={})
            for axis,cross,dimension in (('x','y','b_in'),('y','x','h_in')):
                center=support[axis];across=support[cross];half=support[dimension]/2
                band=[max(fp[cross+'_min_in'],across-half-1.5*h),min(fp[cross+'_max_in'],across+half+1.5*h)]
                axial_half=support['h_in' if axis=='x' else 'b_in']/2
                stations=[center-axial_half,center+axial_half]
                stations=[s for s in stations if fp[axis+'_min_in']+1e-8<s<fp[axis+'_max_in']-1e-8]
                if not stations:raise ValueError('No physical local slab transfer face at a support')
                group['axes'][axis]=[]
                for face in ('top','bottom'):
                    for station in stations:
                        bars=_bars(mat,axis,face,band,station)
                        name=f'local/{k}/{support["id"]}/{axis}/{face}/{station:.12g}'
                        spec=dict(cut_id=name,floor=k,axis=axis,station_in=station,interval_in=band,
                            axial_accounting_interval_in=[fp[axis+'_min_in'],fp[axis+'_max_in']],
                            physical_bar_ids=[b['id'] for b in bars])
                        # Empty explicit steel sets must not expand to all bars in the provider.
                        if bars:specs.append(spec)
                        group['axes'][axis].append(dict(face=face,station_in=station,cut_id=name,bars=bars))
            group['column']=col;groups.append(group)
    return specs,groups


def assess_local_checks(state,definition,inventory,response_bundles,*,assessment_identity,
                        slab_fc_ksi,slab_fy_ksi,development_result=None,selected_floor_case_pairs=None):
    if not assessment_identity or definition['state_identity']!=inventory['state_identity']:
        raise ValueError('Exact named physical state and local assessment required')
    universe=[(b['source_id'],f['level'],c['id']) for b in response_bundles for f in definition['floors'] for c in b['response']['cases']]
    selected=universe if selected_floor_case_pairs is None else [tuple(p) for p in selected_floor_case_pairs]
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(universe):raise ValueError('Explicit nonempty subset of actual local floor/case universe required')
    selected_set=set(selected)
    specs,groups=support_steel_specs(definition,inventory)
    if development_result is None:
        from Design.SMRF_Actual_Bar_Development import actual_bar_development
        development_result=actual_bar_development(state,inventory,specs,slab_fc_ksi=slab_fc_ksi)
    dev={r['specification']['cut_id']:r for r in development_result['cuts']}
    if set(dev)!={r['cut_id'] for r in specs}:raise ValueError('Local development cut inventory differs')
    groupmap={(r['floor'],r['support_id']):r for r in groups};descriptions={r['level']:r for r in definition['floors']}
    h=inventory['slab_thickness_in'];records=[];required=[];seen=set()
    for bundle in response_bundles:
        source_id=bundle['source_id'];response=bundle['response']
        if response['definition_sha256']!=definition['identity_sha256'] or response['status']!='equilibrated':
            raise ValueError('Complete exact physical local response required')
        cases=[r['id'] for r in response['cases']]
        if len(set(cases))!=len(cases):raise ValueError('Duplicate physical local case')
        if {r['level'] for r in response['floors']}!=set(descriptions):raise ValueError('Every actual local floor required')
        for floor in response['floors']:
            k=floor['level'];material=inventory['floors'][str(k)];fp=material['footprint']
            if sorted(r['case_id'] for r in floor['responses'])!=sorted(cases):raise ValueError('Local floor case inventory differs')
            depths={}
            for axis in ('x','y'):
                bars=[b for b in material['bars'] if b['operator_owner']=='slab' and b['axis']==axis]
                if not bars:raise ValueError('Actual orthogonal slab mats required')
                depths[axis]=min(h/2+abs(b['z_in']) for b in bars)
            for support in descriptions[k]['supports']:
                group=groupmap[(k,support['id'])];credit={}
                for axis,rows in group['axes'].items():
                    for face in ('top','bottom'):
                        evidence=[credited_steel(r['bars'],dev.get(r['cut_id'],{}).get('evidence',[]),r['station_in']) for r in rows if r['face']==face]
                        credit[axis+'_'+face]=dict(provided_developed_in2=min(r['provided_developed_in2'] for r in evidence),
                            installed_area_in2=min(r['installed_area_in2'] for r in evidence),face_evidence=evidence)
                perimeter=critical_perimeter(fp,column_x_in=support['x'],column_y_in=support['y'],
                    column_h_x_in=support['h_in'],column_b_y_in=support['b_in'],
                    effective_depth_x_in=depths['x'],effective_depth_y_in=depths['y'])
                for case in floor['responses']:
                    if (source_id,k,case['case_id']) not in selected_set:continue
                    for trace in ('inside','outside'):
                        key=f'{source_id}/{k}/{support["id"]}/{case["case_id"]}/{trace}'
                        if key in seen:raise ValueError('Duplicate actual local support/case/trace identity')
                        seen.add(key);required.append(key);recovery=recover_slab_perimeter(case,perimeter,trace=trace)
                        shear=check_two_way_strength(perimeter,fc_ksi=slab_fc_ksi,
                            slab_resultant_kip_kipin=recovery['slab_resultant_kip_kipin'],
                            direct_shear_peak_ksi=recovery['maximum_direct_slab_shear_ksi'])
                        minimum=local_minimum_steel(perimeter,fp,slab_thickness_in=h,fc_ksi=slab_fc_ksi,fy_ksi=slab_fy_ksi,
                            vuv_ksi=shear['vuv_ksi'],provided_developed_area_in2={n:r['provided_developed_in2'] for n,r in credit.items()})
                        for name,row in minimum['checks'].items():
                            row.update(credit=credit[name],status=minimum_area_status(row['required_in2'],credit[name]))
                        statuses=[r['status'] for r in minimum['checks'].values()]
                        minimum['status']='fail' if 'fail' in statuses else 'unknown' if 'unknown' in statuses else 'pass'
                        records.append(dict(id=key,source_id=source_id,floor=k,support_id=support['id'],case_id=case['case_id'],
                            trace=trace,recovery=recovery,shear=shear,local_minimum=minimum,
                            flexural_transfer_required_kipin=shear['flexural_transfer_required_kipin']))
    if not records:raise ValueError('Nonempty complete physical local response inventory required')
    payload=dict(method=METHOD,assessment_identity=assessment_identity,definition=definition,inventory=inventory,
        response_bundles=response_bundles,slab_fc_ksi=slab_fc_ksi,slab_fy_ksi=slab_fy_ksi,
        development_result=development_result,selected_floor_case_pairs=[list(p) for p in selected])
    return dict(method=METHOD,state_identity=inventory['state_identity'],required_ids=required,records=records,
        development_result=development_result,payload_sha256=_hash(payload),replay_payload=payload,
        scope=dict(full_floor_case_universe=[list(p) for p in universe],selected_floor_case_pairs=[list(p) for p in selected],complete_required_inventory=set(selected)==set(universe)),
        local_flexure_qualification_inferred=False,mesh_qualification_inferred=False)


def replay_local_checks(record):
    p=record['replay_payload']
    if record['payload_sha256']!=_hash(p):raise ValueError('Saved local primitive payload changed')
    return assess_local_checks(None,p['definition'],p['inventory'],p['response_bundles'],
        assessment_identity=p['assessment_identity'],slab_fc_ksi=p['slab_fc_ksi'],slab_fy_ksi=p['slab_fy_ksi'],
        development_result=p['development_result'],selected_floor_case_pairs=p['selected_floor_case_pairs'])
