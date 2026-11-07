"""Actual physical gravity section fields and primary member increments.

No planar P5 force field, tributary beam proxy, or gravity C0 is added. The
complete physical floor supplies G; the force-verified primary frame supplies
only its identical-baseline-subtracted increment. Qualification of this
declared member-analysis branch and of mesh traces remains a separate gate.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from Design.SMRF_Physical_Floor_Cuts import common_strip_cut,identity
from Design.SMRF_Story_Load_Field import METHOD as STORY_FIELD_METHOD

METHOD='physical_full_flange_gravity_primary_member_fields_v2_native_slab_Q'
SPAN_METHOD='physical_piecewise_section_force_extrema_v2_native_slab_Q'
_FIELDS={'P_kip':2,'M_kipin':3,'Vz_kip':2,'Vy_kip':2,'T_kipin':2,'Mz_kipin':2}


def _value(piece,key,s):return float(np.polynomial.polynomial.polyval(float(s),piece[key]))


def _trace(pieces,s,side='right'):
    if s<=0:return pieces[0]
    if s>=1:return pieces[-1]
    for p in pieces:
        a,b=p['start_fraction'],p['end_fraction']
        if a<s<b or (side=='right' and abs(a-s)<1e-12) or (side=='left' and abs(b-s)<1e-12):return p
    raise ValueError('No physical member trace at requested station')


def validate_polynomials(field):
    if field.get('method')!=METHOD or not field.get('source_identity') or not field.get('state_identity'):
        raise ValueError('Bound physical member source and state identities required')
    length=float(field['length_in']);offsets=field['face_offsets_in']
    if not math.isfinite(length) or length<=0 or len(offsets)!=2 or min(offsets)<0 or sum(offsets)>=length:
        raise ValueError('Positive actual beam length and clear face interval required')
    cursor=0.
    for p in field['pieces']:
        a,b=p['start_fraction'],p['end_fraction']
        if not math.isfinite(a+b) or abs(a-cursor)>1e-12 or not a<b<=1:raise ValueError('Complete nonoverlapping physical beam polynomial partition required')
        for key,size in _FIELDS.items():
            v=np.asarray(p[key],float)
            if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('Explicit complete finite physical member polynomial coefficients required')
        cursor=b
    if abs(cursor-1)>1e-12:raise ValueError('Physical member field does not reach both centerline references')
    return field


def recover_polynomial_span(field):
    """Exact component extrema, including both sides of every field jump."""
    validate_polynomials(field);length=float(field['length_in']);all_samples=[]
    for p in field['pieces']:
        a,b=p['start_fraction'],p['end_fraction'];stations={a,b}
        for key in _FIELDS:
            c=p[key]
            if len(c)==3 and c[2]:
                root=-c[1]/(2*c[2])
                if a<root<b:stations.add(root)
        for x in sorted(stations):all_samples.append(dict(x_in=x*length,trace='plus' if x==a else 'minus' if x==b else 'interior',
            **{key:_value(p,key,x) for key in _FIELDS}))
    def envelope(start,stop):
        rows=[r for r in all_samples if start-1e-9<=r['x_in']<=stop+1e-9]
        # Faces need not coincide with a supplied piece boundary.
        for at,side in ((start,'right'),(stop,'left')):
            s=at/length;p=_trace(field['pieces'],s,side);rows.append(dict(x_in=at,trace=side,**{k:_value(p,k,s) for k in _FIELDS}))
        low=min(rows,key=lambda r:r['M_kipin']);high=max(rows,key=lambda r:r['M_kipin'])
        return dict(mu_positive_kip_in=max(0.,high['M_kipin']),mu_negative_kip_in=max(0.,-low['M_kipin']),
            positive_x_in=high['x_in'],negative_x_in=low['x_in'],interval_in=[start,stop],
            axial_min_kip=min(r['P_kip'] for r in rows),axial_max_kip=max(r['P_kip'] for r in rows),
            maximum_abs_vertical_shear_kip=max(abs(r['Vz_kip']) for r in rows),
            maximum_abs_inplane_shear_kip=max(abs(r['Vy_kip']) for r in rows),
            maximum_abs_torsion_kipin=max(abs(r['T_kipin']) for r in rows),
            maximum_abs_weak_moment_kipin=max(abs(r['Mz_kipin']) for r in rows))
    fi,fj=field['face_offsets_in']
    return dict(method_version=SPAN_METHOD,length_in=length,face_offsets_in=[fi,fj],
        full_span=envelope(0.,length),clear_span=envelope(fi,length-fj),critical_sections=all_samples,
        polynomial_field=field,polynomial_field_sha256=identity(field),
        equilibrium=dict(passed=True,basis='exact supplied physical polynomial transfer; upstream field equilibrium/refinement separately required'),
        physical_trace_qualification_inferred=False)


def _section_values(cut,*,beam_midplane_z_in,beam_line_in):
    """Full actual cut to the project's local beam end-action conventions."""
    j=0 if cut['slab']['axis']=='x' else 1;sign=1. if j==0 else -1.
    slab=cut['slab'];web=cut['web'];n=float(cut['resultants'][j]);m=float(cut['resultants'][j+3]);z=float(beam_midplane_z_in)
    across_n=float(slab['integral'][2])+web['V_inplane_kip']
    q=web['Q_vertical_kip']-float(slab['integral'][6+j])
    weak=float(cut['Ms_transverse_first_moment_kipin'])+(cut['transverse_reference_in']-beam_line_in)*n
    first_q=float(slab['transverse_first_moment'][6+j])-beam_line_in*float(slab['integral'][6+j])
    # Native shell +normal-face vertical traction is +raw_Q; the physical
    # web's diagram convention gives -Q_web. C19 independent native nodal
    # virtual-work witnesses establish both Q directions and the Mxy couple.
    # Mxy=int(z*tau_xy); transport each component to the actual beam reference.
    torque=sign*(first_q-float(slab['integral'][5])+z*float(slab['integral'][2]))
    torque+=web['T_about_positive_axis_kipin']-sign*((web['transverse_reference_in']-beam_line_in)*web['Q_vertical_kip']+
             (web['centroid_z_in']-z)*web['V_inplane_kip'])
    return dict(P_kip=-n,M_kipin=-m+z*n,Vz_kip=q,Vy_kip=-sign*across_n,T_kipin=-torque,Mz_kipin=sign*weak)


class PreparedMemberResponses:
    """Invocation-local immutable response digest table; never saved acceptance.

    One member-action build hashes each complete floor/case field once. A
    fresh reader builds its own table from the saved raw bytes. The table
    cannot be used with another response object or material result.
    """
    def __init__(self,result):
        self.result=result;self.payload_sha256=identity(result)
        self.binding=(result.get('definition_sha256'),result.get('source_identity'))
        self.responses={}
        for floor in result['floors']:
            for response in floor['responses']:
                if id(response) in self.responses:raise ValueError('Duplicate actual response object')
                self.responses[id(response)]=(response,identity(response))

    def validate(self,result):
        if result is not self.result or self.binding!=(result.get('definition_sha256'),result.get('source_identity')):
            raise ValueError('Prepared member response cannot cross physical input results')
        return self

    def digest(self,response):
        record=self.responses.get(id(response))
        if record is None or record[0] is not response:raise ValueError('Actual floor response was not prepared in this invocation')
        return record[1]


def full_flange_gravity_field(response,*,member_id,axis,start_in,end_in,beam_line_in,depth_in,
        slab_thickness_in,flange_interval_in,face_offsets_in,uniform_web_loads,state_identity,
        source_identity,definition_sha256,gravity_case_id,prepared_responses=None):
    """Exact clear-span polynomials from full effective-flange physical cuts.

    Shell fields are bilinear inside their own cells. An aligned full-width
    cut is affine along the beam; web M is quadratic and N/V affine. Three
    points reconstruct each piece; a separate fourth point checks transfer.
    Rigid face-to-centerline extensions preserve the complete face wrench.
    These extensions carry no fictitious gravity load or added capacity.
    """
    if axis not in ('x','y') or end_in<=start_in:raise ValueError('Actual positive aligned member geometry required')
    length=float(end_in-start_in);fi,fj=map(float,face_offsets_in);a,b=start_in+fi,end_in-fj
    if fi<0 or fj<0 or b<=a:raise ValueError('Physical clear beam span required')
    lo,hi=map(float,flange_interval_in)
    if hi<=lo or not lo<=beam_line_in<=hi:raise ValueError('Full actual nominal effective-flange interval required')
    if prepared_responses is not None and not isinstance(prepared_responses,PreparedMemberResponses):raise ValueError('Actual invocation-local member response table required')
    normal=0 if axis=='x' else 1;breaks={a,b};response_hash=identity(response) if prepared_responses is None else prepared_responses.digest(response)
    for e in response['elements']:
        positions=np.asarray(e['positions_in'])
        if e['kind']=='slab' or (e['kind']=='web' and str(e['member_id'])==str(member_id)):
            breaks.update(float(v) for v in positions[:,normal] if a<float(v)<b)
    breaks=sorted(breaks);pieces=[];proof=[];midplane=(float(slab_thickness_in)-float(depth_in))/2
    def sample(at,trace):
        cut=common_strip_cut(response,axis=axis,station_in=at,interval_in=[lo,hi],trace=trace,web_id=str(member_id),
            uniform_web_loads=uniform_web_loads,state_identity=state_identity,combination_id=gravity_case_id,
            location_id=f'beam:{member_id}:{at:.15g}:{trace}',recovery_id=response_hash)
        return _section_values(cut,beam_midplane_z_in=midplane,beam_line_in=beam_line_in),cut['accounting']['evidence_id']
    for left,right in zip(breaks,breaks[1:]):
        stations=[left,(left+right)/2,right];values=[sample(x,'minus' if k==2 else 'plus')[0] for k,x in enumerate(stations)]
        s=np.array([(x-start_in)/length for x in stations]);piece=dict(start_fraction=float(s[0]),end_fraction=float(s[-1]))
        test=(left+right)/2+(right-left)/math.sqrt(19);actual,cut_hash=sample(test,'plus');errors={}
        for key,size in _FIELDS.items():
            c=np.polynomial.polynomial.polyfit(s,[v[key] for v in values],size-1)
            piece[key]=c.tolist();pred=float(np.polynomial.polynomial.polyval((test-start_in)/length,c))
            errors[key]=pred-actual[key]
            if abs(errors[key])>1e-8+1e-9*max(abs(pred),abs(actual[key])):raise ValueError('Physical cut field is outside its exact polynomial transfer domain')
        pieces.append(piece);proof.append(dict(interval_in=[left,right],independent_station_in=test,cut_evidence_id=cut_hash,errors=errors))
    def extension(s,first,last):
        p=pieces[0] if first==0. else pieces[-1];values={k:_value(p,k,s) for k in _FIELDS}
        out=dict(start_fraction=first,end_fraction=last)
        for key,size in _FIELDS.items():out[key]=[values[key]]+[0.]*(size-1)
        out['M_kipin']=[values['M_kipin']-values['Vz_kip']*length*s,values['Vz_kip']*length,0.]
        out['Mz_kipin']=[values['Mz_kipin']+values['Vy_kip']*length*s,-values['Vy_kip']*length]
        return out
    if fi:pieces.insert(0,extension(fi/length,0.,fi/length))
    if fj:pieces.append(extension(1-fj/length,1-fj/length,1.))
    field=dict(method=METHOD,member_id=str(member_id),axis=axis,length_in=length,face_offsets_in=[fi,fj],
        flange_interval_in=[lo,hi],capacity_width_in=hi-lo,moment_reference='gross_depth_midplane_sagging_positive',
        beam_midplane_z_in=midplane,beam_line_in=float(beam_line_in),pieces=pieces,
        state_identity=state_identity,source_identity=source_identity,definition_sha256=definition_sha256,
        gravity_case_id=gravity_case_id,physical_response_sha256=response_hash,piece_transfer_checks=proof,
        physical_trace_qualification_inferred=False)
    validate_polynomials(field);return field


def member_from_polynomials(field,*,group_id=None):
    """Legacy endpoint fields plus a polymorphic exact whole-span record."""
    span=recover_polynomial_span(field);length=field['length_in'];fi,fj=field['face_offsets_in']
    left={k:_value(field['pieces'][0],k,0.) for k in _FIELDS};right={k:_value(field['pieces'][-1],k,1.) for k in _FIELDS}
    force=[left[k] for k in ('P_kip','Vy_kip','Vz_kip','T_kipin','M_kipin','Mz_kipin')]
    force.extend(-right[k] for k in ('P_kip','Vy_kip','Vz_kip','T_kipin','M_kipin','Mz_kipin'))
    pbase=(left['P_kip']+right['P_kip'])/2;mi,mj=left['M_kipin'],-right['M_kipin'];fixed=[]
    for p in field['pieces']:
        fixed.append(dict(start_fraction=p['start_fraction'],end_fraction=p['end_fraction'],
            P_offset_kip=[p['P_kip'][0]-pbase,p['P_kip'][1]],
            M_offset_kipin=[p['M_kipin'][0]-mi,p['M_kipin'][1]+mi+mj,p['M_kipin'][2]],
            V_offset_kip=[p['Vz_kip'][0]+(mi+mj)/length,p['Vz_kip'][1]]))
    field_for_story=dict(method=STORY_FIELD_METHOD,source_identity=identity(field),pieces=fixed)
    from Design.SMRF_Story_Load_Field import validate
    validate(field_for_story)
    return dict(member_type='beam_'+field['axis'],group_id=group_id,local_force_kip_kipin=force,
        centerline_axial_i_kip=left['P_kip'],centerline_axial_j_kip=right['P_kip'],
        axial_i_kip=_value(_trace(field['pieces'],fi/length,'right'),'P_kip',fi/length),
        axial_j_kip=_value(_trace(field['pieces'],1-fj/length,'left'),'P_kip',1-fj/length),
        joint_face_offsets_in=[fi,fj],axial_line_load_kip_per_in=None,span_bending=span,
        story_fixed_section_field=field_for_story,story_baseline_member_actions=dict(P=pbase,Mi=mi,Mj=mj),
        story_field_capacity_width_in=field['capacity_width_in'],story_field_moment_reference=field['moment_reference'],
        physical_field_sha256=identity(field),physical_trace_qualification_inferred=False)


def validated_member_span(member):
    """Replay the saved exact physical polynomial envelope, never trust labels."""
    from Design.SMRF_Beam_Actions import _matches
    saved=member.get('span_bending') or {}
    if saved.get('method_version')!=SPAN_METHOD:raise ValueError('Actual physical polynomial span record required')
    field=saved['polynomial_field'];actual=recover_polynomial_span(field)
    if member.get('physical_field_sha256')!=identity(field) or not _matches(saved,actual):
        raise ValueError('Saved physical member field/envelope does not replay')
    return actual


def physical_face_reactions(transfer,member,dead,live):
    """Actual G clear-span V after releasing both face moments to zero.

    A change of face moments adds a constant shear. Physical side tractions
    and couples represented by the original section field stay unchanged.
    Member self-weight is already in G and is never added a second time.
    """
    from Model import Member_Groups as mg
    cases=transfer['physical_gravity_member_actions'];expected=transfer['expected_physical_gravity_case_ids']
    if (len(set(expected))!=len(expected) or sorted(c['id'] for c in cases)!=sorted(expected) or
        transfer.get('state_identity')!=mg.active().identity()):raise ValueError('Complete actual physical gravity transfer case/state inventory required')
    matching=[c for c in cases if float(c['dead'])==float(dead) and float(c['live'])==float(live)]
    if not matching:raise ValueError('Actual simultaneous physical gravity load family is missing')
    rows=[]
    for c in matching:
        if c.get('ex',0.) or c.get('ey',0.) or c.get('branch')!='physical_G_only' or c.get('analysis_succeeded') is not True:
            raise ValueError('Capacity gravity reactions require exact gravity-only physical responses')
        if any(c.get(k)!=transfer.get(k) for k in ('state_identity','source_identity','physical_definition_sha256')):
            raise ValueError('Physical capacity gravity case/definition/source binding differs')
        saved=c['members'][str(member.member_tag)];span=validated_member_span(saved);field=span['polynomial_field']
        if any(field.get(k)!=c.get(k if k!='definition_sha256' else 'physical_definition_sha256') for k in ('state_identity','source_identity','definition_sha256')):
            raise ValueError('Physical beam gravity field identity differs from its actual case')
        length=field['length_in'];fi,fj=field['face_offsets_in'];left,right=fi/length,1-fj/length
        pi,pj=_trace(field['pieces'],left,'right'),_trace(field['pieces'],right,'left')
        mi,mj=_value(pi,'M_kipin',left),_value(pj,'M_kipin',right);shift=(mi-mj)/(length-fi-fj)
        rows.append([_value(pi,'Vz_kip',left)+shift,-_value(pj,'Vz_kip',right)-shift])
    return np.max(rows,axis=0).tolist(),('actual physical full-flange gravity section V with both face moments released; '
        'all matching simultaneous live patterns retained; original member weight included once; exact polynomial source replay')


def add_primary_increment(gravity_field,local_force_increment,*,increment_identity):
    """Add the unloaded full effective-beam increment about the same reference.

    The physical gravity cut uses the gross-depth midplane; the primary beam
    centerline is the slab midplane. Exact rigid wrench transport retains its
    axial/shear eccentric terms. No external diaphragm branch is appended.
    """
    validate_polynomials(gravity_field);f=np.asarray(local_force_increment,float)
    if f.shape!=(12,) or not np.isfinite(f).all() or not increment_identity:raise ValueError('Complete identified primary local force increment required')
    length=gravity_field['length_in'];z=gravity_field['beam_midplane_z_in']
    residual=np.r_[f[:4]+f[6:10],f[4]+f[10]+f[2]*length,f[5]+f[11]-f[1]*length]
    if max(abs(residual))>1e-8*max(1.,max(abs(f)),length*max(abs(f[1:3]))):
        raise ValueError('Primary beam increment is not an equilibrated zero-element-load branch')
    add=dict(P_kip=[f[0],0.],M_kipin=[f[4]-z*f[0],f[2]*length,0.],
             Vz_kip=[f[2],0.],Vy_kip=[f[1],0.],T_kipin=[f[3]+z*f[1],0.],Mz_kipin=[f[5],-f[1]*length])
    result=copy.deepcopy(gravity_field)
    for p in result['pieces']:
        for key in _FIELDS:p[key]=(np.asarray(p[key])+add[key]).tolist()
    result.update(gravity_field_sha256=identity(gravity_field),primary_increment_identity=increment_identity,
        primary_increment_local_forces=f.tolist(),branch='actual_physical_G_plus_baseline_subtracted_primary_member_increment',
        external_diaphragm_branch_added=False,primary_force_composition_qualification_inferred=False)
    return result


def column_member_action(gravity_local_force,*,increment_local_force=None,face_offsets_in,
                         axial_weight_kip_per_in,member_type='column',group_id=None):
    """Exact concurrent column local12, including both actual axial end forces."""
    g=np.asarray(gravity_local_force,float);d=np.zeros(12) if increment_local_force is None else np.asarray(increment_local_force,float)
    offsets=np.asarray(face_offsets_in,float);weight=float(axial_weight_kip_per_in)
    if g.shape!=(12,) or d.shape!=(12,) or offsets.shape!=(2,) or not np.isfinite(np.r_[g,d,offsets,weight]).all() or min(offsets)<0 or weight<0:
        raise ValueError('Finite actual column forces, face offsets and downward weight required')
    f=g+d
    if abs(d[0]+d[6])>1e-8*max(1.,max(abs(d))):raise ValueError('Primary increment invents an axial element load')
    return dict(member_type=member_type,group_id=group_id,local_force_kip_kipin=f.tolist(),
        centerline_axial_i_kip=float(f[0]),centerline_axial_j_kip=float(-f[6]),
        axial_i_kip=float(f[0]-weight*offsets[0]),axial_j_kip=float(-f[6]+weight*offsets[1]),
        joint_face_offsets_in=offsets.tolist(),axial_line_load_kip_per_in=weight,
        physical_gravity_local_forces=g.tolist(),primary_increment_local_forces=d.tolist())


def build_member_actions(state,definition,physical_result,combination,*,gravity_case_id,
                         primary_definition=None,primary_increment=None,primary_transfer=None,saved_input_binding=None,prepared_responses=None):
    """Actual installed member inventory for selection/capacity/strength.

    Call once for each original signed combination, using its exact D/L/live
    pattern physical gravity case. Primary inputs are all supplied or all
    absent (gravity only). Source/state/geometry and every transferred column
    axial end are checked numerically. This does not qualify mesh convergence
    or authorize adding an independent P5 physical load branch.
    """
    from Design.SMRF_Coupled_Floor_Design import validate_definition
    from Design.SMRF_Physical_Action_Readout import installed_full_t_projector
    from Design.Group_Checks import joint_beam_depth_in
    from Model import Member_Properties as mp
    validate_definition(definition)
    if definition['state_identity']!=state.identity() or physical_result.get('definition_sha256')!=definition['identity_sha256']:
        raise ValueError('Physical member actions belong to a different installed state or definition')
    source=physical_result.get('source_identity')
    if prepared_responses is not None and not isinstance(prepared_responses,PreparedMemberResponses):raise ValueError('Actual invocation-local member response table required')
    context=PreparedMemberResponses(physical_result) if prepared_responses is None else prepared_responses.validate(physical_result)
    physical_hash=context.payload_sha256
    if not source or physical_result.get('status') not in ('equilibrated','constitutive_equilibrated'):
        raise ValueError('Identified equilibrated physical gravity result required')
    matches=[i for i,c in enumerate(physical_result['cases']) if c['id']==gravity_case_id]
    if len(matches)!=1:raise ValueError('Unique simultaneous physical gravity case required')
    at=matches[0];gcase=physical_result['cases'][at]
    if any(float(gcase[k])!=float(combination[k]) for k in ('dead','live')) or gcase.get('live_pattern','all')!=combination.get('live_pattern','all'):
        raise ValueError('Physical gravity D/L/pattern differs from signed design combination')
    frame=physical_result['frame'];base=frame['cases'][at]
    if base.get('status')!='equilibrated' or not base['maximum_free_equilibrium_residual']<=base['equilibrium_tolerance']:
        raise ValueError('Physical gravity frame equilibrium did not close')
    active=primary_increment is not None
    increment_hash=identity(primary_increment) if active else None
    if active!=(primary_definition is not None) or active!=(primary_transfer is not None):raise ValueError('Complete primary definition, increment and actual-P transfer proof required together')
    if not active and (combination.get('ex',0.) or combination.get('ey',0.)):raise ValueError('Signed seismic combination omits its primary increment')
    if active:
        if saved_input_binding is not None:
            if (saved_input_binding.get('source_identity')!=source or saved_input_binding.get('payload_sha256')!=physical_hash or
                not saved_input_binding.get('input_manifest_sha256') or primary_transfer.get('saved_input_binding')!=saved_input_binding):
                raise ValueError('Explicit saved physical gravity content/source binding does not replay')
        elif primary_increment.get('source_identity')!=source:
            raise ValueError('Primary and gravity sources differ without an explicit prior-input binding')
        if (primary_increment.get('status')!='equilibrated' or not primary_increment.get('source_identity') or
            primary_increment.get('physical_gravity_case_id')!=gravity_case_id or
            primary_increment.get('primary_definition_sha256')!=primary_definition.get('identity_sha256') or
            primary_definition.get('physical_definition_sha256')!=definition['identity_sha256'] or
            primary_definition.get('state_identity')!=state.identity()):
            raise ValueError('Primary increment and exact simultaneous physical state/source differ')
        ledger=primary_transfer.get('column_axial_ledger',[]);bytag={str(r['column_id']):r for r in ledger}
        if len(bytag)!=len(ledger) or set(bytag)!=set(base['columns']):raise ValueError('Actual-P primary transfer omits or duplicates a physical column')
        for tag,column in base['columns'].items():
            actual=np.asarray(column['local_force'])[[0,6]];proof=bytag[tag]
            target=np.asarray(proof['target_signed_local_end_axials_kip']);transferred=np.asarray(proof['actual_signed_local_end_axials_kip']);tol=np.asarray(proof['tolerance_kip'])
            if (target.shape!=(2,) or transferred.shape!=(2,) or tol.shape!=(2,) or not np.isfinite(np.r_[actual,target,transferred,tol]).all()
                or np.any(tol<=0) or np.any(abs(actual-target)>1e-9) or np.any(abs(transferred-target)>tol)):
                raise ValueError('Primary transfer does not reproduce both actual physical column axial ends')
        if set(primary_increment['columns'])!=set(base['columns']):raise ValueError('Primary column increment inventory differs')
    floors={r['level']:r for r in physical_result['floors']};members={};gravity_members={};g=definition['geometry']
    physical_columns={str(c['id']):c for c in definition['columns']}
    for member in state.members():
        tag=str(member.member_tag);d=member.design
        if member.is_column:
            col=base['columns'][tag];k,i,j=member.story_or_floor,member.grid_i,member.grid_j
            offsets=[.5*joint_beam_depth_in(k-1,i,j),.5*joint_beam_depth_in(k,i,j)]
            weight=float(gcase['dead'])*mp.column_self_weight_kip_per_in(member)
            original=column_member_action(col['local_force'],face_offsets_in=offsets,axial_weight_kip_per_in=weight,group_id=member.group_id)
            actual=column_member_action(col['local_force'],increment_local_force=primary_increment['columns'][tag]['local_force_increment'] if active else None,
                face_offsets_in=offsets,axial_weight_kip_per_in=weight,group_id=member.group_id)
            actual['physical_gravity_global_forces']=col['global_resisting_force']
            if active:actual['primary_increment_global_forces']=primary_increment['columns'][tag]['global_force_increment']
        else:
            axis=mp.beam_axis(member);normal=0 if axis=='x' else 1;level=member.story_or_floor
            row=floors[level];responses=[r for r in row['responses'] if r['case_id']==gravity_case_id]
            if len(responses)!=1:raise ValueError('Physical floor response omits/duplicates the simultaneous gravity case')
            response=responses[0];projector=installed_full_t_projector(member)
            start=(member.grid_i if axis=='x' else member.grid_j)*(g['lx'] if axis=='x' else g['ly']);length=mp.beam_span_in(member)
            line=(member.grid_j if axis=='x' else member.grid_i)*(g['ly'] if axis=='x' else g['lx'])
            webloads={e['id']:-float(gcase['dead'])*definition['gamma_kip_in3']*d.b_in*(d.h_in-definition['slab_thickness_in'])
                      for e in response['elements'] if e['kind']=='web' and str(e['member_id'])==tag}
            field=full_flange_gravity_field(response,member_id=tag,axis=axis,start_in=start,end_in=start+length,
                beam_line_in=line,depth_in=d.h_in,slab_thickness_in=definition['slab_thickness_in'],
                flange_interval_in=projector.flange,face_offsets_in=mp.beam_face_distances_in(member),uniform_web_loads=webloads,
                state_identity=state.identity(),source_identity=source,definition_sha256=definition['identity_sha256'],gravity_case_id=gravity_case_id,prepared_responses=context)
            original=member_from_polynomials(field,group_id=member.group_id)
            actual=member_from_polynomials(add_primary_increment(field,primary_increment['beams'][tag]['local_force_increment'],
                increment_identity=increment_hash) if active else field,group_id=member.group_id)
        gravity_members[tag]=original;members[tag]=actual
    if set(base['columns'])!=set(physical_columns) or set(members)!={str(m.member_tag) for m in state.members()}:
        raise ValueError('Physical full member inventory differs from installed candidate')
    common=dict(analysis_succeeded=True,axial_reference='joint_faces',state_identity=state.identity(),source_identity=source,
        physical_definition_sha256=definition['identity_sha256'],physical_gravity_result_sha256=physical_hash,
        physical_gravity_case_id=gravity_case_id,method=METHOD,physical_trace_qualification_inferred=False)
    action={**combination,**common,'members':members,'primary_increment_sha256':increment_hash,
            'primary_source_identity':primary_increment.get('source_identity') if active else None,'saved_input_binding':saved_input_binding,
            'branch':'physical_G_plus_primary_increment' if active else 'physical_G_only'}
    gravity={**gcase,**common,'ex':0.,'ey':0.,'members':gravity_members,'family':'directional_story_strength_gravity',
             'represented_design_combination_ids':[combination['id']],'branch':'physical_G_only'}
    return dict(action=action,gravity_action=gravity)
