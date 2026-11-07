"""Source-bound complete adjacent-panel corner moment envelopes.

ACI318-19 8.7.3 uses the maximum positive Mu in the adjacent slab PANEL.
The absolute moment boundary maximum below is an explicit upper bound on
that positive value. It is not a corner-region sample or an invented zero.
Every bilinear cell's clipped rectangle vertices bound its complete field.
Primary strength guards remain a separately required part of this envelope.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Floor_Cuts import _bounds,slab_tensor

METHOD='actual_adjacent_panel_corner_moment_upper_envelope_v1'


def adjacent_panel_bounds(definition,floor,corner):
    sx,sy=corner.split('_');g=definition['geometry'];fp=floor['footprint']
    if sx not in ('minus','plus') or sy not in ('minus','plus'):raise ValueError('Actual physical building corner required')
    return [fp['x_min_in'] if sx=='minus' else (g['nx']-1)*g['lx'],
            g['lx'] if sx=='minus' else fp['x_max_in'],
            fp['y_min_in'] if sy=='minus' else (g['ny']-1)*g['ly'],
            g['ly'] if sy=='minus' else fp['y_max_in']]


def panel_moment_bound(response,bounds):
    x0,x1,y0,y1=bounds;witnesses={a:None for a in ('x','y')};maximum={a:0. for a in ('x','y')};area=0.
    for element in response['elements']:
        if element['kind']=='web':continue
        ex0,ex1,ey0,ey1=_bounds(element)
        a,b,c,d=max(x0,ex0),min(x1,ex1),max(y0,ey0),min(y1,ey1)
        if b<=a or d<=c:continue
        area+=(b-a)*(d-c)
        for x,y in ((a,c),(b,c),(b,d),(a,d)):
            resultants=slab_tensor(element,x,y)
            for axis,at in (('x',3),('y',4)):
                value=12*abs(float(resultants[at]))
                if witnesses[axis] is None or value>maximum[axis]:
                    maximum[axis]=value;witnesses[axis]=dict(element_id=element['id'],xy_in=[x,y],signed_moment_kipin_per_ft=12*float(resultants[at]))
    if abs(area-(x1-x0)*(y1-y0))>1e-7*max(1.,area) or any(v is None for v in witnesses.values()):
        raise ValueError('Actual slab cells do not cover the entire adjacent corner panel exactly once')
    return dict(maximum_absolute_moment_kipin_per_ft=maximum,witnesses=witnesses,area_in2=area,bounds_in=bounds)


def projected_panel_bound(response,bounds,*,axis,axial_interval_in,flange_interval_in,coefficients):
    """Exact |bilinear physical M + quadratic projected M| on each clipped cell.

    Coefficients use t=(axial-a)/(b-a). The projected slab moment is uniform
    across its effective flange. At fixed axial coordinate the function is
    affine transversely, so two transverse edges and the axial quadratic's
    stationary points suffice, including both traces at a cell boundary.
    """
    j=0 if axis=='x' else 1;a,b=map(float,axial_interval_in);coef=np.asarray(coefficients,float)
    if axis not in ('x','y') or not b>a or coef.shape!=(3,) or not np.isfinite(coef).all():raise ValueError('Finite actual quadratic projection required')
    maximum=0.;witness=None;cells=0
    for e in response['elements']:
        if e['kind']!='slab':continue
        ex0,ex1,ey0,ey1=_bounds(e);clip=[max(bounds[0],ex0),min(bounds[1],ex1),max(bounds[2],ey0),min(bounds[3],ey1)]
        clip[2*j]=max(clip[2*j],a);clip[2*j+1]=min(clip[2*j+1],b)
        k=1-j;clip[2*k]=max(clip[2*k],flange_interval_in[0]);clip[2*k+1]=min(clip[2*k+1],flange_interval_in[1])
        if clip[1]<=clip[0] or clip[3]<=clip[2]:continue
        cells+=1;lo,hi=clip[2*j:2*j+2]
        for cross in clip[2*k:2*k+2]:
            def physical(at):
                xy=[at,cross] if j==0 else [cross,at]
                return float(slab_tensor(e,*xy)[3+j])
            slope=(physical(hi)-physical(lo))/(hi-lo)
            linear=slope*(b-a)+coef[1];locations=[lo,hi]
            if abs(coef[2])>1e-16:
                root=a+(b-a)*(-linear/(2*coef[2]))
                if lo<root<hi:locations.append(root)
            for at in locations:
                t=(at-a)/(b-a);value=physical(at)+float(np.array([1.,t,t*t])@coef)
                if witness is None or abs(value)*12>maximum:
                    maximum=abs(value)*12;witness=dict(element_id=e['id'],axial_in=at,transverse_in=cross,
                        physical_moment_kipin_per_in=physical(at),projected_moment_kipin_per_in=value-physical(at),
                        signed_total_moment_kipin_per_ft=12*value)
    return dict(maximum_absolute_moment_kipin_per_ft=maximum,witness=witness,intersecting_cells=cells)


def primary_corner_envelopes(state,common,*,prepared_sources=None):
    """Compute complete adjacent-panel bounds for all primary/P5 moment guards."""
    import Structure_Parameters as sp
    from Model import Member_Properties as mp
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    from Design.SMRF_Common_Guard_Plan import _PhysicalSource,primary_guard_increment
    from Design.SMRF_Physical_Action_Readout import installed_full_t_projector
    from Design.SMRF_Primary_Moment_Guards import _normal
    from Design.SMRF_Common_Guard_Plan import PreparedCommonGuardSources
    if prepared_sources is not None and not isinstance(prepared_sources,PreparedCommonGuardSources):raise ValueError('Invocation-local common source context required')
    context=None if prepared_sources is None else prepared_sources.validate(common)
    bundles=physical_response_bundles(common) if context is None else context.bundles
    sources={r['source_id']:_PhysicalSource(r,common) for r in bundles} if context is None else context.sources
    gravity=sources['actual_frozen_gravity'];descs={f['level']:f for f in common['definition']['floors']}
    seismic={c['id'] for c in common['combinations'] if c['ex'] or c['ey']};corners=('minus_minus','minus_plus','plus_minus','plus_plus')
    records={};proofs=[];seen=set();expected=set()
    for sid,source in sources.items():
        if not sid.startswith('physical_ELF/'):continue
        for cid in source.cases:
            if cid not in seismic:raise ValueError('Unexpected physical ELF corner case')
            gid=common['gravity_case_mapping'][cid]
            for member in [m for m in state.members() if not m.is_column]:
                floor=member.story_or_floor;projector=installed_full_t_projector(member);axis=projector.axis
                start=(member.grid_i if axis=='x' else member.grid_j)*(sp.BAY_X if axis=='x' else sp.BAY_Y)
                fi,fj=mp.beam_face_distances_in(member);lo,hi=start+fi,start+mp.beam_span_in(member)-fj
                field=source.fields[(floor,cid)][0];grid=source.floors[floor]['grid'][axis+'_in']
                breaks=sorted({lo,hi}|{float(v) for v in grid if lo<float(v)<hi})
                scope=(sid,cid,floor,str(member.member_tag));expected.add(scope)
                if scope in seen:raise ValueError('Duplicate actual primary corner member/case')
                seen.add(scope)
                for a,b in zip(breaks[:-1],breaks[1:]):
                    values=[];samples=[]
                    for t,trace in ((0.,'plus'),(.5,'minus'),(1.,'minus'),(.25,'minus')):
                        station=a+t*(b-a);spec=dict(cut_id=f'corner/{cid}/{member.member_tag}/{station}/{trace}',floor=floor,
                            axis=axis,station_in=station,interval_in=list(projector.flange),trace=trace,member_id=str(member.member_tag))
                        p=source.cut(spec,cid);g=gravity.cut(spec,gid);inc=primary_guard_increment(common,cid,member.member_tag,station)
                        delta=_normal(g,axis,projector.sref)+np.array([0.,inc['m_z_kip_in'],inc['m_s_kip_in']])-_normal(p,axis,projector.sref)
                        row=[]
                        for mode,dz,ds in (('strong',delta[1],0.),('weak',0.,delta[2]),('both',delta[1],delta[2]),('P5_both',delta[1],inc['m_s_kip_in'])):
                            projection=projector.project(n_tension_kip=0.,m_z_kip_in=dz,m_s_kip_in=ds,increment_id=mode)
                            row.append(projector.ef*projector.t**3/12*projection['generalized_normal_strain'][1])
                        values.append(row);samples.append(dict(station_in=station,trace=trace,delta_normal=delta.tolist(),primary_increment_sha256=inc['primary_increment_sha256']))
                    values=np.array(values);coef=np.linalg.solve(np.array([[1,0,0],[1,.5,.25],[1,1,1]]),values[:3])
                    residual=np.array([1,.25,.0625])@coef-values[3]
                    if np.any(abs(residual)>1e-8+1e-8*np.maximum(abs(values[3]),abs(coef).max(axis=0))):
                        raise ValueError('Primary corner moment projection failed independent polynomial witness')
                    for corner in corners:
                        key=(floor,corner,cid);record=records.setdefault(key,dict(floor=floor,corner=corner,case_id=cid,
                            mu_positive_x_kip_in_per_ft=0.,mu_positive_y_kip_in_per_ft=0.,proof_ids=[]))
                        bounds=adjacent_panel_bounds(common['definition'],descs[floor],corner)
                        for i,mode in enumerate(('strong','weak','both','P5_both')):
                            bound=projected_panel_bound(field,bounds,axis=axis,axial_interval_in=[a,b],flange_interval_in=projector.flange,coefficients=coef[:,i])
                            if not bound['intersecting_cells']:continue
                            proof=dict(source_id=sid,source_payload_sha256=source.bundle['response_payload_sha256'],case_id=cid,floor=floor,
                                corner=corner,member_id=str(member.member_tag),axis=axis,mode=mode,axial_interval_in=[a,b],
                                flange_interval_in=list(projector.flange),samples=samples,coefficients=coef[:,i].tolist(),
                                independent_quarter_residual=float(residual[i]),**bound)
                            proofs.append(proof);record['proof_ids'].append(_hash(proof));name='mu_positive_'+axis+'_kip_in_per_ft'
                            record[name]=max(record[name],bound['maximum_absolute_moment_kipin_per_ft'])
    required={(f,c,cid) for f in descs for c in corners for cid in seismic}
    if set(records)!=required or seen!=expected:raise ValueError('Primary corner floor/case/member inventory incomplete')
    output=[]
    for _,r in sorted(records.items()):r['recovery_id']=_hash(r);output.append(r)
    result=dict(method=METHOD+'/primary',analysis_input_identity=common['input_signature'],state_identity=common['state_identity'],
        material_inventory_id=common['inventory']['identity_sha256'],records=output,proofs=proofs,
        required_member_cases=[list(v) for v in sorted(expected)],complete_verified_inventory=True)
    result['payload_sha256']=_hash(result);return result


def corner_demands(common,*,guard_envelopes=None):
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    definition=common['definition'];descriptions={f['level']:f for f in definition['floors']};records={};proofs=[]
    for bundle in physical_response_bundles(common):
        for floor in bundle['response']['floors']:
            level=floor['level'];description=descriptions[level]
            for case in floor['responses']:
                for corner in ('minus_minus','minus_plus','plus_minus','plus_plus'):
                    row=panel_moment_bound(case,adjacent_panel_bounds(definition,description,corner))
                    key=(level,corner,case['case_id']);proof=dict(source_id=bundle['source_id'],response_payload_sha256=bundle['response_payload_sha256'],
                        floor=level,corner=corner,case_id=case['case_id'],**row);proofs.append(proof)
                    old=records.setdefault(key,dict(floor=level,corner=corner,case_id=case['case_id'],
                        mu_positive_x_kip_in_per_ft=0.,mu_positive_y_kip_in_per_ft=0.,recovery_ids=[]))
                    for a in ('x','y'):old['mu_positive_'+a+'_kip_in_per_ft']=max(old['mu_positive_'+a+'_kip_in_per_ft'],row['maximum_absolute_moment_kipin_per_ft'][a])
                    old['recovery_ids'].append(_hash(proof))
    # A guard bound must be a computed per-case/floor/corner moment envelope,
    # with its raw projection witnesses retained by the producer. Absence
    # stays explicit; the physical-only envelope cannot qualify final steel.
    guard_complete=False
    if guard_envelopes is not None:
        expected={(r['floor'],r['corner'],r['case_id']) for r in guard_envelopes['records']}
        if not expected<=set(records):raise ValueError('Primary corner guards have different case/floor scope')
        for row in guard_envelopes['records']:
            target=records[(row['floor'],row['corner'],row['case_id'])]
            for a in ('x','y'):target['mu_positive_'+a+'_kip_in_per_ft']=max(target['mu_positive_'+a+'_kip_in_per_ft'],row['mu_positive_'+a+'_kip_in_per_ft'])
            target['recovery_ids'].append(row['recovery_id'])
        seismic={c['id'] for c in common['combinations'] if c['ex'] or c['ey']}
        required={(f,corner,c) for f in descriptions for corner in ('minus_minus','minus_plus','plus_minus','plus_plus') for c in seismic}
        if (expected!=required or guard_envelopes.get('payload_sha256')!=_hash({k:v for k,v in guard_envelopes.items() if k!='payload_sha256'})
            or guard_envelopes.get('analysis_input_identity')!=common['input_signature']
            or guard_envelopes.get('material_inventory_id')!=common['inventory']['identity_sha256']):
            raise ValueError('Primary corner bound is not the complete actual source/material/case inventory')
        guard_complete=bool(guard_envelopes.get('proofs')) and guard_envelopes.get('complete_verified_inventory') is True
    output=[]
    for key,row in sorted(records.items()):
        row['region_basis']='entire_adjacent_slab_panel';row['recovery_id']=_hash(row);output.append(row)
    result=dict(method=METHOD,state_identity=common['state_identity'],source_id=common['analysis_source_identity'],
        material_inventory_id=common['inventory']['identity_sha256'],analysis_input_identity=common['input_signature'],
        expected_case_ids=sorted({r['case_id'] for r in output}),records=output,physical_witnesses=proofs,
        primary_guard_coverage_complete=guard_complete,primary_guard_envelopes=guard_envelopes,
        demand_basis='Absolute bilinear field maximum on the complete adjacent slab panel, bounding positive Mu;12in/ft',
        qualification_inferred=False)
    result['payload_sha256']=_hash(result);return result
