"""Explicit local transverse-layer alternative for a physical seven-action cut.

This maps transverse mat bars using the already frozen actual bar pitches,
instead of counting bars in a new axial averaging window. Longitudinal bars
remain actual discrete circles at their real transverse coordinates. A pitch
is a local smeared-reinforcement idealization, not proof that arbitrary axial
force fields equal their window average. Existing default assessment and its
window checks are unchanged. Qualification/continuous action coverage remain
upstream and cannot be supplied by this constructor's return status.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial,_inside_trace
from Design.SMRF_Physical_Floor_Materials import _hash
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection

METHOD='local_fixed_actual_pitch_transverse_seven_section_v2_crossing_voids_unqualified'


def developed_pitch_spatial_section(state,inventory,*,normalweight=True,**section_arguments):
    """Bind the local section's exact physical IDs to actual end-development.

    Geometry is selected before development, so missing evidence cannot cause
    a bar to disappear. The same constructor then applies computed full-fy
    domains while preserving every installed concrete void. This is an explicit
    alternate API; it does not activate a qualified production profile.
    """
    if 'development_evidence' in section_arguments:raise ValueError('This adapter computes actual development evidence itself')
    first=pitch_spatial_section(inventory,**section_arguments)
    if first['status']!='inventory_constructed' or not first['physical_bar_ids']:return first
    from Design.SMRF_Actual_Bar_Development import actual_bar_development
    spec=dict(first['identity']['specification'],physical_bar_ids=first['physical_bar_ids'],
        bar_selection_basis='explicit_local_physical_ids')
    development=actual_bar_development(state,inventory,[spec],slab_fc_ksi=section_arguments['fc_ksi'],normalweight=normalweight)
    result=pitch_spatial_section(inventory,**section_arguments,development_evidence=development['cuts'][0]['evidence'])
    if result['physical_bar_ids']!=first['physical_bar_ids']:raise ValueError('Development must not change actual physical bar selection')
    result['actual_development_evidence']=development
    return result


def _development_domain(evidence,inventory,spec,bar):
    """Re-scope computed full-fy evidence; an ID or pass flag alone is insufficient."""
    matches=[]
    for row in evidence:
        if bar['id'] not in row.get('physical_bar_ids',[]):continue
        if not row.get('evidence_id') or row['evidence_id']!=_hash({k:v for k,v in row.items() if k!='evidence_id'}):
            raise ValueError('Development evidence content hash differs')
        span=row.get('station_interval_in',[])
        scoped=(row.get('state_identity')==inventory['state_identity'] and
            row.get('material_inventory_id')==inventory['identity_sha256'] and
            row.get('floor')==spec['floor'] and row.get('axis')==spec['axis'] and
            row.get('interval_in')==spec['interval_in'] and len(span)==2 and span[0]<=spec['station_in']<=span[1] and
            row.get('full_fy_only') is True and row.get('bar_axis')==bar['axis'])
        if not scoped:continue
        from Design.SMRF_Actual_Bar_Development import METHOD as DEVELOPMENT_METHOD,full_fy_developed_domain
        if row.get('method')!=DEVELOPMENT_METHOD:raise ValueError('Authoritative actual bar-development method required')
        domain=row.get('full_fy_development_domain',{})
        if domain.get('status')!='computed':continue
        checks=row.get('checks',[]);recomputed=[full_fy_developed_domain(c) for c in checks]
        if not recomputed or any(d['status']!='computed' for d in recomputed):
            raise ValueError('Computed domain lacks reproducible underlying development checks')
        span=[max(d['interval_in'][0] for d in recomputed),min(d['interval_in'][1] for d in recomputed)]
        if domain.get('interval_in')!=span or any(d['physical_straight_run_interval_in']!=domain.get('physical_straight_run_interval_in') for d in recomputed):
            raise ValueError('Claimed full-fy domain differs from the actual end checks')
        if domain.get('basis')!='full_fy_both_signs_both_sides_no_25_4_10_reduction':
            raise ValueError('Explicit existing full-fy both-signs development basis required')
        run=domain.get('physical_straight_run_interval_in',[]);credit=domain.get('interval_in')
        if (len(run)!=2 or not np.isfinite(run).all() or run[1]<=run[0] or
                credit is not None and (len(credit)!=2 or not np.isfinite(credit).all() or credit[1]<=credit[0] or credit[0]<run[0]-1e-9 or credit[1]>run[1]+1e-9)):
            raise ValueError('Actual finite physical run and contained developed interval required')
        actual=bar['straight_run_interval_in']
        if run[0]>actual[0]+1e-9 or run[1]<actual[1]-1e-9:
            raise ValueError('Development physical run does not cover the installed bar')
        if bar['operator_owner']=='slab' and not np.allclose(run,actual,rtol=0,atol=1e-9):
            raise ValueError('Mat development run differs from actual installed straight run')
        if run!=actual and bar['operator_owner']=='web_line':
            checks=row.get('checks',[])
            if not checks or any(bar['id'] not in c.get('continuous_physical_bar_owners',[]) for c in checks):
                raise ValueError('Extended beam development requires its actual continuous owner chain')
        matches.append((credit,row['evidence_id']))
    if not matches:return dict(status='unknown',reason='bound_full_fy_domain_missing',interval_in=None)
    if any(m[0]!=matches[0][0] for m in matches):raise ValueError('Conflicting developed domains for one physical bar')
    return dict(status='computed',interval_in=matches[0][0],evidence_ids=sorted({m[1] for m in matches}))


def pitch_spatial_section(inventory,*,floor,axis,station_in,interval_in,fc_ksi,ec_ksi,
        phi_requirements,trace='plus',member_id=None,material_scope='common_slab_web',
        transverse_reference=None,development_evidence=(),quadrature_order=12):
    if axis not in ('x','y') or trace not in ('plus','minus'):
        raise ValueError('Physical section axis and explicit one-sided trace required')
    if material_scope not in ('common_slab_web','slab_only') or material_scope=='slab_only' and member_id is not None:
        raise ValueError('Disjoint common-section or slab-only ownership required')
    station=float(station_in);lo,hi=map(float,interval_in);reference=(lo+hi)/2 if transverse_reference is None else float(transverse_reference)
    if not all(math.isfinite(v) for v in (station,lo,hi,reference)) or hi<=lo:
        raise ValueError('Finite positive actual cut geometry required')
    field=FixedPitchFloorMaterial(inventory,floor);fp=field.footprint;cross='y' if axis=='x' else 'x'
    if not fp[axis+'_min_in']<=station<=fp[axis+'_max_in'] or lo<fp[cross+'_min_in'] or hi>fp[cross+'_max_in']:
        raise ValueError('Cut must lie within its actual floor footprint')
    material=inventory['floors'][str(floor)];h=field.h;slab_only=material_scope=='slab_only'
    spec=dict(floor=int(floor),axis=axis,station_in=station,interval_in=[lo,hi],trace=trace)
    longitudinal=[];transverse=[];selected=[];excluded=[];issues=[];development={};web=None;voids=[]
    strength_fc=float(fc_ksi);strength_ec=float(ec_ksi);grades=dict(slab_fc_ksi=strength_fc)
    if member_id is not None:
        member_id=str(member_id);g=inventory['beam_geometry'][member_id]
        if g['axis']!=axis or int(g['floor'])!=int(floor):raise ValueError('Actual member axis/floor differs')
        grades['web_fc_ksi']=float(g['fc_ksi']);strength_fc=min(strength_fc,g['fc_ksi']);strength_ec=min(strength_ec,g['ec_ksi'])
        web=dict(id='member_'+member_id,interval_in=[g['line_in']-g['b_in']/2,g['line_in']+g['b_in']/2],z_interval_in=[h/2-g['h_in'],-h/2])
    for bar in material['bars']:
        if bar['axis']!=axis:
            if bar['operator_owner']=='web_line':
                radius=math.sqrt(bar['area_in2']/math.pi)
                if abs(station-bar['s_in'])<radius and min(hi,bar['straight_run_interval_in'][1])>max(lo,bar['straight_run_interval_in'][0]):
                    half=math.sqrt(radius*radius-(station-bar['s_in'])**2)
                    bottom,top=max(-h/2,bar['z_in']-half),min(h/2,bar['z_in']+half)
                    if top>bottom:
                        if member_id is not None:
                            issues.append(dict(reason='crossing_beam_joint_requires_disjoint_web_geometry',bar_id=bar['id']))
                        else:
                            voids.append(dict(id=bar['id'],interval_in=[max(lo,bar['straight_run_interval_in'][0]),min(hi,bar['straight_run_interval_in'][1])],z_interval_in=[bottom,top]))
                            excluded.append(bar['id'])
            continue
        if not _inside_trace(station,bar['straight_run_interval_in'],trace):continue
        radius=bar['db_in']/2;s=bar['s_in']
        if s+radius<=lo or s-radius>=hi:continue
        void_only=slab_only and bar['operator_owner']=='web_line'
        if void_only and (bar['z_in']+radius<=-h/2 or bar['z_in']-radius>=h/2):continue
        if s-radius<lo-1e-9 or s+radius>hi+1e-9:
            issues.append(dict(reason='longitudinal_circle_crosses_strip_boundary',bar_id=bar['id']));continue
        if not slab_only and bar['operator_owner']=='web_line' and str(bar.get('member_id'))!=member_id:
            issues.append(dict(reason='longitudinal_beam_steel_has_no_matching_web',bar_id=bar['id']));continue
        r={k:bar[k] for k in ('id','s_in','z_in','area_in2','db_in','fy_ksi','es_ksi')}
        if void_only:
            r['strength_credited']=False;excluded.append(bar['id'])
        else:
            selected.append(bar['id']);d=_development_domain(development_evidence,inventory,spec,bar);development[bar['id']]=d
            span=d['interval_in'];r['strength_credited']=span is not None and span[0]<=station<=span[1]
        longitudinal.append(r)
    for layer in field.layers:
        if layer.axis==axis or not _inside_trace(station,layer.pitch,trace):continue
        a,b=max(lo,layer.run[0]),min(hi,layer.run[1])
        if b<=a:continue
        bar=layer.bar;selected.append(bar['id']);d=_development_domain(development_evidence,inventory,spec,bar);development[bar['id']]=d
        span=d['interval_in'];credit=None if span is None else [max(a,span[0]),min(b,span[1])]
        if credit is not None and credit[1]<=credit[0]:credit=None
        transverse.append(dict(id='pitch_row/'+bar['id'],physical_bar_ids=[bar['id']],
            **{k:bar[k] for k in ('area_in2','db_in','fy_ksi','es_ksi','z_in')},
            accounting_length_in=layer.width,run_interval_in=[a,b],strength_run_interval_in=credit))
    arguments=dict(thickness_in=h,fc_ksi=strength_fc,ec_ksi=strength_ec,axis=axis,
        slab_interval_in=[lo,hi],web=web,reference_s_in=reference,longitudinal_bars=longitudinal,
        transverse_rows=transverse,phi_requirements=dict(phi_requirements),quadrature_order=quadrature_order,slab_void_rectangles=voids)
    binding=dict(method=METHOD,state_identity=inventory['state_identity'],inventory_identity=inventory['identity_sha256'],
        field_identity=field.identity,specification=spec,member_id=member_id,material_scope=material_scope,
        constructor=arguments,development=development)
    result=dict(status='unknown' if issues else 'inventory_constructed',section=None,constructor=arguments,
        identity=binding,identity_sha256=_hash(binding),issues=issues,physical_bar_ids=sorted(selected),
        excluded_beam_strength_ids=sorted(excluded),development=development,
        development_status='pass' if selected and all(v['status']=='computed' for v in development.values()) else 'unknown',
        concrete_strength_basis=grades,material_mapping=METHOD,
        axial_window_equivalence_claimed=False,continuous_action_coverage_verified=False,
        nominal_probable_capacity_inventory_modified=False,engineering_qualified=False)
    if not issues:
        try:result['section']=SpatialFloorSection(**arguments)
        except ValueError as exc:result.update(status='unknown',issues=[dict(reason=str(exc))])
    return result
