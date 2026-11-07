"""Actual local material/development boundaries, not a capacity theorem."""
from __future__ import annotations
import copy
import math
from Design.SMRF_Common_Floor_Actions import _hash
from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
from Design.SMRF_Actual_Bar_Development import full_fy_developed_domain,METHOD as DEVELOPMENT_METHOD

METHOD='actual_local_material_and_development_event_partition_v1'


def _replayed_domain(evidence):
    domains=[full_fy_developed_domain(c) for c in evidence['checks']]
    if domains and all(d['status']=='computed' for d in domains):
        result={**domains[0],'interval_in':[max(d['interval_in'][0] for d in domains),min(d['interval_in'][1] for d in domains)]}
        if result['interval_in'][1]<result['interval_in'][0]-1e-8:result['status']='fail'
        return result
    return dict(status='unknown',station_domains=domains)


def local_material_events(inventory,floor,development,*,analysis_axes=None):
    """Reconstruct every installed bar's domain and fixed local ownership events.

    The complete one-bar/own-axis development inventory is deliberate: a
    subset cannot silently omit a bar that starts carrying force elsewhere.
    The underlying saved checks still require the caller's geometry replay.
    """
    if inventory.get('identity_sha256')!=_hash({k:v for k,v in inventory.items() if k!='identity_sha256'}):
        raise ValueError('Actual material inventory content differs')
    if (development.get('method')!=DEVELOPMENT_METHOD or development.get('state_identity')!=inventory['state_identity']
        or development.get('material_inventory_id')!=inventory['identity_sha256'] or development.get('full_fy_only') is not True
        or development.get('evidence_id')!=_hash({k:v for k,v in development.items() if k!='evidence_id'})):
        raise ValueError('Complete actual full-fy development source differs')
    floor=int(floor);material=inventory['floors'][str(floor)];bars=material['bars'];fp=material['footprint']
    byid={b['id']:b for b in bars}
    if not bars or len(byid)!=len(bars):raise ValueError('Unique nonempty installed physical bar inventory required')
    domains={};unknown=[];seen=set()
    for row in development['cuts']:
        spec=row['specification'];ids=spec.get('physical_bar_ids',[])
        if (int(spec['floor'])!=floor or spec.get('bar_selection_basis')!='explicit_local_physical_ids'
            or len(ids)!=1 or ids[0] not in byid or ids[0] in seen or row['missing_physical_bar_ids'] or len(row['evidence'])!=1):
            raise ValueError('Every floor bar needs exactly one complete own-axis development record')
        bid=ids[0];bar=byid[bid];e=row['evidence'][0];seen.add(bid)
        cross='y' if bar['axis']=='x' else 'x';run=bar['straight_run_interval_in']
        if (spec['axis']!=bar['axis'] or spec['station_in']!=sum(run)/2
            or spec['interval_in']!=[fp[cross+'_min_in'],fp[cross+'_max_in']]
            or e.get('physical_bar_ids')!=[bid] or e.get('full_fy_only') is not True
            or e.get('state_identity')!=inventory['state_identity'] or e.get('material_inventory_id')!=inventory['identity_sha256']
            or e.get('evidence_id')!=_hash({k:v for k,v in e.items() if k!='evidence_id'})):
            raise ValueError('Exact bar geometry/source/end-check evidence required')
        domain=_replayed_domain(e)
        if domain!=e['full_fy_development_domain']:raise ValueError('Full-fy domain differs from original end-check calculation')
        if domain['status']=='computed':
            domains[bid]=dict(axis=bar['axis'],interval_in=domain['interval_in'],physical_run=run,evidence_id=e['evidence_id'])
        else:unknown.append(dict(bar_id=bid,domain=domain))
    if seen!=set(byid):raise ValueError('Development evidence omits actual physical bars')
    axes={} if analysis_axes is None else copy.deepcopy(analysis_axes)
    if axes and set(axes)!={'x_in','y_in'}:raise ValueError('Both actual analysis coordinate axes required')
    field=FixedPitchFloorMaterial(inventory,floor);events={}
    for axis in ('x','y'):
        lo,hi=fp[axis+'_min_in'],fp[axis+'_max_in'];rows=[]
        def event(value,kind,ident):
            value=float(value)
            if not math.isfinite(value):raise ValueError('Finite actual material event coordinate required')
            if lo<=value<=hi:rows.append(dict(coordinate_in=value,kind=kind,physical_id=ident))
        for v in (lo,hi):event(v,'physical_footprint_boundary',None)
        grid=axes.get(axis+'_in',[])
        if grid and (len(grid)<2 or any(not math.isfinite(float(v)) for v in grid)
                     or any(b<=a for a,b in zip(grid[:-1],grid[1:])) or grid[0]!=lo or grid[-1]!=hi):
            raise ValueError('Ordered full-footprint analysis axes required')
        for value in grid:event(value,'existing_analysis_cell_face',None)
        for bar in bars:
            if bar['axis']==axis:
                for value in bar['straight_run_interval_in']:event(value,'longitudinal_physical_run_end',bar['id'])
                if bar['id'] in domains:
                    for value in domains[bar['id']]['interval_in']:event(value,'longitudinal_full_fy_credit_change',bar['id'])
            elif bar['operator_owner']=='web_line':
                radius=math.sqrt(bar['area_in2']/math.pi)
                for value in (bar['s_in']-radius,bar['s_in'],bar['s_in']+radius):event(value,'crossing_nominal_cylinder_chord_event',bar['id'])
        for layer in field.layers:
            if layer.axis!=axis:
                for value in layer.pitch:event(value,'transverse_fixed_pitch_owner_change',layer.bar['id'])
        coordinates=sorted(set(r['coordinate_in'] for r in rows))
        events[axis]=dict(events=rows,coordinates_in=coordinates,intervals=len(coordinates)-1)
    result=dict(method=METHOD,state_identity=inventory['state_identity'],inventory_identity=inventory['identity_sha256'],
        field_identity=field.identity,floor=floor,events=events,domains=domains,unknown_domains=unknown,
        analysis_axes=axes,development=copy.deepcopy(development),
        continuous_capacity_verified=False,curved_crossing_voids_still_vary_inside_intervals=True)
    result['payload_sha256']=_hash(result)
    return result


def verify_local_material_events(record,inventory):
    if record.get('method')!=METHOD or record.get('payload_sha256')!=_hash({k:v for k,v in record.items() if k!='payload_sha256'}):
        raise ValueError('Saved local material event content differs')
    rebuilt=local_material_events(inventory,record['floor'],record['development'],analysis_axes=record['analysis_axes'])
    if rebuilt!=record:raise ValueError('Saved material events differ from actual inventory/end checks')
    return rebuilt


def installed_local_material_events(state,inventory,floor,*,slab_fc_ksi,analysis_axes=None):
    from Design.SMRF_Actual_Bar_Development import actual_bar_development
    floor=int(floor);material=inventory['floors'][str(floor)];fp=material['footprint'];specs=[]
    for bar in material['bars']:
        axis=bar['axis'];cross='y' if axis=='x' else 'x'
        specs.append(dict(floor=floor,axis=axis,station_in=sum(bar['straight_run_interval_in'])/2,
            interval_in=[fp[cross+'_min_in'],fp[cross+'_max_in']],physical_bar_ids=[bar['id']],bar_selection_basis='explicit_local_physical_ids'))
    development=actual_bar_development(state,inventory,specs,slab_fc_ksi=slab_fc_ksi)
    return local_material_events(inventory,floor,development,analysis_axes=analysis_axes)
