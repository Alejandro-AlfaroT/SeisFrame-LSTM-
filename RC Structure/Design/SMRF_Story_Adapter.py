"""Final grouped-state adapter for directional nominal story strength.

FEMA P-2012 G.3/G.4.2 permits a directional strength measure without a
three-dimensional plastic analysis. Exact gravity states and full member
strength interactions feed the directional equilibrium model. Signed 100/30
cases remain mandatory for the separate actual biaxial member design checks.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from Design.SMRF_Story_Strength import (METHOD, analyze_gravity_family, rectangular_interaction,
    layered_interaction, reverse_moment_interaction, adjacent_strength_bounds,
    _gravity_shear, _axial_shift, verify_solution)

ADAPTER_METHOD='final_grouped_state_physical_field_equilibrated_plane_story_bounds_v2'


class StoryStrengthPremiseUnavailable(ValueError):
    """A physical premise of the directional story model is not established for this design (not a binding, identity
    or inventory error): the story strength stays not calculated and its regularity row not evaluated."""
    def __init__(self,message,**evidence):
        super().__init__(message);self.evidence=evidence
VERIFICATION_METHOD='reconstructed_full_primal_dual_story_certificates_v1'


def collect_gravity_cases(actions,solver):
    """Exact existing gravity solves, covering every supplied D/L/pattern.

    No seismic-force averaging and no new OpenSees solve. The two-system
    combination solver already cached the continuous-frame gravity response.
    """
    from Design.Perimeter_Systems import CONTINUOUS_MODEL
    unique={};cases=[]
    for action in actions:
        key=(action['dead'],action['live'],action.get('live_pattern','all'))
        unique.setdefault(key,[]).append(action)
    for (dead,live,pattern),family in unique.items():
        exact=next((c for c in family if not c['ex'] and not c['ey']),None)
        cached=getattr(solver,'_cache',{}).get((CONTINUOUS_MODEL,dead,live,pattern))
        if exact is None and cached is None:
            raise ValueError(f'No exact solved gravity state for D={dead}, L={live}, pattern={pattern}; seismic averaging is not permitted')
        members=exact['members'] if exact is not None else cached['members']
        cases.append(dict(id=f'gravity_state_{len(cases)+1:03d}',dead=dead,live=live,ex=0.,ey=0.,live_pattern=pattern,
            family='directional_story_strength_gravity',analysis_succeeded=True,axial_reference='joint_faces',members=members,
            represented_design_combination_ids=[c['id'] for c in family],
            provenance='existing exact continuous gravity solve; no recombination or new solve'))
    return cases


def collect_physical_gravity_cases(adapted_combinations,expected_ids):
    """Deduplicate exact actual-coupled G records; never infer G from +/-E."""
    from Design.SMRF_Physical_Member_Actions import METHOD as PHYSICAL_METHOD
    ids=[r['action']['id'] for r in adapted_combinations]
    if sorted(ids)!=sorted(expected_ids) or len(set(ids))!=len(ids):raise ValueError('Complete original physical design combination inventory required')
    unique={};bindings=set()
    for pair in adapted_combinations:
        action,gravity=pair['action'],pair['gravity_action']
        if action.get('method')!=PHYSICAL_METHOD or gravity.get('method')!=PHYSICAL_METHOD or gravity.get('branch')!='physical_G_only':
            raise ValueError('Actual physical member producer is required for the gravity collector')
        bindings.add(tuple(gravity.get(k) for k in ('state_identity','source_identity','physical_definition_sha256')))
        if gravity['represented_design_combination_ids']!=[action['id']]:raise ValueError('Physical gravity represented-case binding differs')
        key=(action['dead'],action['live'],action.get('live_pattern','all'))
        if key!=(gravity['dead'],gravity['live'],gravity.get('live_pattern','all')) or gravity['ex'] or gravity['ey']:
            raise ValueError('Actual gravity and signed combination load families differ')
        payload={k:v for k,v in gravity.items() if k!='represented_design_combination_ids'}
        if key not in unique:unique[key]=[{**payload,'represented_design_combination_ids':[]},_hash(payload)]
        if unique[key][1]!=_hash(payload):raise ValueError('Same gravity family has differing actual solved fields')
        unique[key][0]['represented_design_combination_ids'].append(action['id'])
    if len(bindings)!=1 or any(v is None for v in next(iter(bindings))):raise ValueError('Physical gravity family mixes or omits candidate/source bindings')
    return [row for row,digest in unique.values()]


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_identity():
    root=Path(__file__).resolve().parents[1]
    names=['Design/SMRF_Story_Strength.py','Design/SMRF_Story_Adapter.py','Design/SMRF_Biaxial_Section.py',
           'Design/SMRF_Story_Load_Field.py',
           'Design/SMRF_Physical_Member_Actions.py','Design/SMRF_Physical_Floor_Cuts.py',
           'Design/SMRF_Uniaxial_Section.py','Design/SMRF_Beam_Axial.py','Design/SMRF_Beam_Slab_Strength.py',
           'Design/SMRF_Transitions.py','Model/Member_Properties.py','Model/Member_Bar_Layers.py','Structure_Parameters.py']
    names+=['Design/Grouped_Design.py','Design/Group_Capacity.py','Design/Group_Checks.py','Design/SMRF_Common.py','Model/Member_Groups.py']
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    return {'sha256':_hash(files),'files':files}


def joint_traction_bounds(*,node,joint_id,column_below_id,column_length_in,left=None,right=None,nominal_vn_kip):
    """Bounds from the actual horizontal cut below diaphragm force injection.

    Beam compression P is positive. For the lower-half axial resultant F,
    -Ty_bottom <= F <= P+Ty_top follows from nonnegative concrete compression
    and bounded steel tension. Q=Vc+F_left-F_right. The inner model requires
    every Q in this interval to be safe; the outer model only requires that
    the interval meet [-Vn,+Vn]. No guessed moment/lever-arm relation is used.
    """
    low={column_below_id:{'Mi':1/column_length_in,'Mj':1/column_length_in}}
    high={column_below_id:{'Mi':1/column_length_in,'Mj':1/column_length_in}}
    lower_constant=upper_constant=0.
    if left:
        high[left['id']]={'P':1.}
        lower_constant-=left['bottom_tension_kip'];upper_constant+=left['top_tension_kip']+left.get('axial_offset_kip',0.)
    if right:
        low[right['id']]={'P':-1.}
        lower_constant-=right['top_tension_kip']+right.get('axial_offset_kip',0.);upper_constant+=right['bottom_tension_kip']
    rows=[]
    for bound,pairs in [('inner',[(1,high,upper_constant),(-1,low,lower_constant)]),
                        ('outer',[(1,low,lower_constant),(-1,high,upper_constant)])]:
        for sense,terms,constant in pairs:
            rows.append(dict(id=f'{joint_id}/{bound}/{sense:+d}',node=str(node),sense=sense,bound=bound,
                nominal=True,basis='Horizontal panel cut below diaphragm injection: Vc + Fbottom,left - Fbottom,right; nominal steel force bounds and nonnegative concrete compression',
                terms={mid:{key:sense*v for key,v in values.items()} for mid,values in terms.items()},
                constant=sense*constant,limit=float(nominal_vn_kip)))
    return rows


def _layer_tension(section):
    # All slab mats above the beam midplane are in the top partition. Steel
    # centroids on the cut are refused, rather than assigned arbitrarily.
    if np.isclose(section.layers[:,0],section.h/2,atol=1e-9,rtol=0).any():
        raise StoryStrengthPremiseUnavailable('A longitudinal steel centroid lies on the selected joint cut')
    ty=section.layers[:,1]*section.layers[:,3]
    return {'top_tension_kip':float(ty[section.layers[:,0]<section.h/2].sum()),
            'bottom_tension_kip':float(ty[section.layers[:,0]>section.h/2].sum())}


def _nominal_steel_shear(member,axis,capacity):
    """Guaranteed steel-only nominal shear; concrete is relaxed in outer LP.

    The selected physical hoop's Vs is capped at the existing ACI section
    steel limit. Taking Vc=0 is valid over the whole variable axial domain.
    A demand-envelope Vc is not reused at a different collapse axial load.
    """
    import Structure_Parameters as sp
    from Model import Member_Bar_Layers as bl
    d=member.design
    if member.is_column:
        data=capacity['columns'][member.group_id];hoops=data['hoops']
        row=hoops['by_direction'][axis]
        vs=row['av_in2']*sp.FYT_KSI*row['d_in']/hoops['spacing_in']
        limit=data['vs_limit_by_direction_kip'][axis]
    else:
        rows=bl.group_rows(member.group_id)
        depth=d.h_in-max(rows['top']['centroid_in'],rows['bottom']['centroid_in'])
        vs=d.stirrup_legs*sp.rebar_area(d.stirrup_bar_size)*sp.FYT_KSI*depth/d.stirrup_spacing_in
        limit=8*math.sqrt(d.fc_ksi*1000)*d.b_in*depth/1000
    if min(vs,limit)<=0:
        raise StoryStrengthPremiseUnavailable('No physical positive nominal transverse-steel strength',
                                              member_tag=member.member_tag,axis=axis)
    return min(vs,limit)


def build_frame(state,case,capacity,axis,*,divisions=128,curve_cache=None,source=None):
    """Reconstruct a compatible planar projection from one saved load case.

    External node forces are recovered from the selected members' actual end
    forces and their exact saved element loads. Thus excluded frame reactions
    and floor-direct loads remain in this projection, not guessed tributary
    weights. All parallel SFRS lines share each floor horizontal DOF.
    """
    import Structure_Parameters as sp
    from Model import Member_Groups as mg, Member_Properties as mp, Member_Bar_Layers as bl
    from Design.SMRF_Biaxial_Section import BiaxialSection
    from Design.SMRF_Transitions import bar_coordinates
    from Design.SMRF_Beam_Axial import installed_member_sections
    from Design.Group_Capacity import joint_inventory
    from Design.Group_Checks import joint_beam_depth_in
    cache={} if curve_cache is None else curve_cache
    nx,ny,nf=sp.NUM_BAY_X,sp.NUM_BAY_Y,sp.NUM_FLOOR
    layout=mg.layout_of(state.policy_id);system=mg.is_system_policy(state.policy_id)
    selected=[]
    for member in state.members():
        if member.is_column:
            include=not system or axis in mg.column_sfrs_directions(member.grid_i,member.grid_j,nx,ny,layout)
        else:
            include=member.member_type==f'beam_{axis}' and (not system or mg.member_system(member.member_type,member.grid_i,member.grid_j,nx,ny,layout)==mg.SYSTEM_SFRS)
        if include:selected.append(member)
    nodes={};members=[];by_section={};baseline={};orthogonal=[]
    def node(member,end):
        floor=member.story_or_floor-(member.is_column and end=='i')
        along=member.grid_i if axis=='x' else member.grid_j
        line=member.grid_j if axis=='x' else member.grid_i
        if not member.is_column and end=='j':along+=1
        nid=f'{line}/{floor}/{along}'
        nodes.setdefault(nid,dict(id=nid,floor=int(floor),x_in=along*(sp.BAY_X if axis=='x' else sp.BAY_Y),
                                  z_in=floor*sp.STORY_H,fixed_base=floor==0,load={'x_kip':0.,'z_kip':0.,'moment_kip_in':0.}))
        return nid,line
    for member in selected:
        d=member.design;mid=str(member.member_tag);saved=case['members'][mid]
        f=list(map(float,saved['local_force_kip_kipin']))
        if len(f)!=12 or not all(math.isfinite(v) for v in f):raise ValueError('Incomplete/nonfinite saved member force vector')
        ni,line=node(member,'i');nj,_=node(member,'j')
        item=dict(id=mid,kind='column' if member.is_column else 'beam',i=ni,j=nj,line_id=str(line),
                  capacity_reference='joint_faces',nominal_shear_kip=_nominal_steel_shear(member,axis,capacity),
                  outer_nominal_shear_kip=None,shear_bound_basis='inner: installed capped nominal Vs, Vc=0 over all P; outer: shear relaxed, never a false upper bound')
        if member.is_column:
            bars=[(p['x_in'],p['y_in'],d.bar_area_in2) for p in bar_coordinates(d,sp.COL_CLEAR_COVER_IN)]
            key=_hash(['column',d.b_in,d.h_in,d.fc_ksi,sp.FY_KSI,sp.ES_KSI,bars,axis,divisions])
            if key not in cache:cache[key]=rectangular_interaction(BiaxialSection(d.b_in,d.h_in,d.fc_ksi,sp.FY_KSI,bars,sp.ES_KSI),axis='h' if axis=='x' else 'b',divisions=divisions)
            item.update(interaction_i=cache[key],interaction_j=cache[key],transverse_load_kip_per_in=0.,
                        axial_load_kip_per_in=case['dead']*mp.column_self_weight_kip_per_in(member),
                        face_offset_i_in=0. if member.story_or_floor==1 else .5*joint_beam_depth_in(member.story_or_floor-1,member.grid_i,member.grid_j),
                        face_offset_j_in=.5*joint_beam_depth_in(member.story_or_floor,member.grid_i,member.grid_j))
            component=4 if axis=='x' else 5;other=5 if axis=='x' else 4
            mi,mj=-f[component],-f[component+6]
            orthogonal.append(dict(member_tag=member.member_tag,orthogonal_moment_i_kip_in=f[other],orthogonal_moment_j_kip_in=f[other+6]))
        else:
            if sp.SLAB_THICKNESS_IN is not None and sp.SLAB_THICKNESS_IN>=d.h_in/2:
                raise StoryStrengthPremiseUnavailable('Directional joint cut requires the complete diaphragm slab above the beam midplane',
                                                      member_tag=member.member_tag,beam_depth_in=d.h_in,slab_thickness_in=sp.SLAB_THICKNESS_IN)
            sections=installed_member_sections(member,bl.group_rows(member.group_id));by_section[mid]=sections
            curves={}
            for end,section in sections.items():
                key=_hash(['beam',section.h,section.regions.tolist(),section.layers.tolist(),divisions])
                if key not in cache:cache[key]=layered_interaction(section,divisions=divisions)
                curves[end]=cache[key]
            fi,fj=mp.beam_face_distances_in(member)
            item.update(interaction_i=curves['i'],interaction_j=reverse_moment_interaction(curves['j']),span_interaction=curves['span'],
                        face_offset_i_in=fi,face_offset_j_in=fj)
            field=saved.get('story_fixed_section_field')
            if field is not None:
                from Design.SMRF_Story_Load_Field import validate
                validate(field);item['fixed_section_field']=field
                actual=saved.get('story_baseline_member_actions')
                if not isinstance(actual,dict) or set(actual)!={'P','Mi','Mj'} or not all(math.isfinite(float(v)) for v in actual.values()):
                    raise ValueError('Physical section fields require their exact reference end-resultant baseline')
                if (saved.get('story_field_capacity_width_in')!=sections['span'].regions[0,2] or
                    saved.get('story_field_moment_reference')!='gross_depth_midplane_sagging_positive'):
                    raise ValueError('Physical story gravity must use the actual full nominal flange and moment reference')
                mi,mj=actual['Mi'],actual['Mj']
            else:
                span=saved.get('span_bending')
                if not span or span.get('equilibrium',{}).get('passed') is not True:raise ValueError('Exact saved beam-load equilibrium is required')
                item.update(transverse_load_kip_per_in=-float(span['uniform_z_kip_per_in']),
                            transverse_point_loads=[[float(a),-float(p)] for a,p in span['point_z_loads']])
                mi,mj=f[4],f[10]
        baseline[mid]={'P':(saved['story_baseline_member_actions']['P'] if not member.is_column and saved.get('story_fixed_section_field') is not None else (f[0]-f[6])/2),
                       'Mi':mi,'Mj':mj};members.append(item)
    # End-action-compatible external forces, including exact element loading.
    for m in members:
        ni,nj=nodes[m['i']],nodes[m['j']];length=math.hypot(nj['x_in']-ni['x_in'],nj['z_in']-ni['z_in'])
        cx=(nj['x_in']-ni['x_in'])/length;cz=(nj['z_in']-ni['z_in'])/length;sy,sz=-cz,cx
        b=baseline[m['id']];v=(b['Mi']+b['Mj'])/length
        for n,sign,vs,end in [(ni,-1.,1.,'i'),(nj,1.,-1.,'j')]:
            reaction=_gravity_shear(m,length,0.,side='left') if end=='i' else -_gravity_shear(m,length,1.)
            pshift=_axial_shift(m,length,0. if end=='i' else 1.,side='right' if end=='i' else 'left')
            n['load']['x_kip']+=-sign*(b['P']+pshift)*cx-vs*v*sy+reaction*sy
            n['load']['z_kip']+=-sign*(b['P']+pshift)*cz-vs*v*sz+reaction*sz
            n['load']['moment_kip_in']-=b['Mi' if end=='i' else 'Mj']
    by_member={m['id']:m for m in members};joints=[];joint_evidence=[]
    for entry in joint_inventory(state)['joints']:
        col=str(entry['below'].member_tag)
        if col not in by_member:continue
        endpoints={}
        for side,label,end in [('minus','left','j'),('plus','right','i')]:
            beam=entry['joint'][f'beam_{axis}_{side}']
            if beam is not None and str(beam.member_tag) in by_member:
                mid=str(beam.member_tag);m=by_member[mid];length=sp.BAY_X if axis=='x' else sp.BAY_Y
                fraction=m['face_offset_i_in']/length if end=='i' else 1-m['face_offset_j_in']/length
                endpoints[label]={'id':mid,**_layer_tension(by_section[mid][end]),
                                  'axial_offset_kip':_axial_shift(m,length,fraction,side='right' if end=='i' else 'left')}
        if not endpoints:continue
        record=capacity['joint_types'][entry['type_id']]['shear'][axis]
        if record.get('capacity_topology_and_confinement_checked') is not True:
            raise StoryStrengthPremiseUnavailable('Joint nominal strength lacks topology/confinement evidence',
                                                  joint_id=entry['id'],joint_type=entry['type_id'],axis=axis)
        joints.extend(joint_traction_bounds(node=by_member[col]['j'],joint_id=entry['id'],column_below_id=col,
                      column_length_in=sp.STORY_H,nominal_vn_kip=record['nominal_vn_kip'],**endpoints))
        joint_evidence.append(dict(joint_id=entry['id'],type_id=entry['type_id'],nominal_vn_kip=record['nominal_vn_kip'],**endpoints))
    return dict(state_identity=state.identity(),source_identity=source or source_identity(),combination_id=case['id'],
                nodes=list(nodes.values()),members=members,joint_constraints=joints,
                adapter_method=ADAPTER_METHOD,direction=axis,load_factors={key:case.get(key) for key in ('dead','live','ex','ey','live_pattern')},
                actual_saved_actions_sha256=_hash(case),orthogonal_column_actions=orthogonal,joint_bound_inputs=joint_evidence,
                baseline_member_actions=baseline,
                scope=['FEMA P-2012 G.3/G.4.2 directional nominal story-strength measure; no rigorous three-dimensional plastic analysis is required for this measure.',
                       'Rigid diaphragm action is assumed; diaphragm strength and three-dimensional signed biaxial member design remain separate required production checks.',
                       'First-order nominal resistance; required ELF drift/PDelta stability checks remain separate.'])


def _validate_inventory(state,actions,expected_ids,gravity_actions):
    from Design.SMRF_Common import action_inventory_problems
    _,problems=action_inventory_problems(actions,expected_ids,member_tags=[m.member_tag for m in state.members()])
    if problems:raise ValueError('Story strength action inventory: '+'; '.join(problems))
    represented=[cid for c in gravity_actions for cid in c['represented_design_combination_ids']]
    if sorted(represented)!=sorted(expected_ids):raise ValueError('Gravity state coverage does not match the complete signed design inventory')
    required={(a['dead'],a['live'],a.get('live_pattern','all')) for a in actions}
    supplied={(a['dead'],a['live'],a.get('live_pattern','all')) for a in gravity_actions}
    if supplied!=required or len(supplied)!=len(gravity_actions):raise ValueError('Gravity load-family inventory is incomplete or duplicated')
    by_id={a['id']:a for a in actions}
    for case in gravity_actions:
        if case['ex'] or case['ey']:raise ValueError('Directional story-strength gravity state must have no seismic preload')
        _,problems=action_inventory_problems([case],[case['id']],member_tags=[m.member_tag for m in state.members()])
        if problems:raise ValueError('Gravity state member inventory: '+'; '.join(problems))
        key=(case['dead'],case['live'],case.get('live_pattern','all'))
        if any((by_id[cid]['dead'],by_id[cid]['live'],by_id[cid].get('live_pattern','all'))!=key for cid in case['represented_design_combination_ids']):
            raise ValueError('Represented design combinations have different gravity factors')


def _certificate_payload(result):
    keys=('method','adapter_method','stories','cases','adjacent_ratio_bounds','state_identity','source_identity',
          'case_inventory','gravity_family_evidence','section_divisions','signed_actions_sha256','gravity_actions_sha256',
          'unreduced_sds_for_vertical_ground_motion','unreduced_sds')
    return {k:result.get(k) for k in keys}


def computed_verification_valid(result):
    """In-process gate; saved records additionally replay every certificate."""
    proof=(result or {}).get('verification') or {}
    return (result.get('method')==METHOD and result.get('adapter_method')==ADAPTER_METHOD
            and result.get('applicability',{}).get('status')=='verified'
            and proof.get('method')==VERIFICATION_METHOD and proof.get('passed') is True
            and proof.get('certificate_payload_sha256')==_hash(_certificate_payload(result))
            and proof.get('source_identity')==source_identity())


def verify_saved_result(state,actions,expected_ids,capacity,*,gravity_actions,result):
    """Rebuild physical input models and replay all primal/dual proofs; no solve.

    Each gravity state's feasible inner force field is necessary. A certified
    optimized outer bound from any one state bounds their minimum from above.
    Stored aggregate strengths, labels and classification ratios are checked
    against these independently reconstructed force certificates.
    """
    _validate_inventory(state,actions,expected_ids,gravity_actions)
    if result.get('method')!=METHOD or result.get('adapter_method')!=ADAPTER_METHOD:raise ValueError('Story strength method differs')
    if result['state_identity']!=state.identity() or result['source_identity']!=source_identity():raise ValueError('Story strength state/source identity differs')
    if result['case_inventory']!=list(expected_ids) or result['signed_actions_sha256']!=_hash(actions) or result['gravity_actions_sha256']!=_hash(gravity_actions):
        raise ValueError('Story strength saved load inventory/hash differs')
    cases={(c['direction'],c['combination_id']):c for c in result['cases']}
    if len(cases)!=len(result['cases']) or set(cases)!={(a,g['id']) for a in ('x','y') for g in gravity_actions}:raise ValueError('Directional gravity case coverage differs')
    import Structure_Parameters as sp
    high=[a for a in actions if a.get('family')=='seismic_high_gravity'];low=[a for a in actions if a.get('family')=='seismic_low_gravity']
    full_ev=bool(high and low) and all(math.isclose(a['dead'],1.2+.2*sp.ASCE_SDS,rel_tol=1e-10) for a in high) and all(math.isclose(a['dead'],.9-.2*sp.ASCE_SDS,rel_tol=1e-10) for a in low)
    if result['unreduced_sds_for_vertical_ground_motion']!=full_ev or result['unreduced_sds']!=sp.ASCE_SDS:raise ValueError('Unreduced vertical-ground-motion evidence differs')
    cache={};certificates=[];reconstructed={a:[] for a in ('x','y')}
    def same(a,b):
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7)
    for axis in ('x','y'):
        all_rows={};reference=result['gravity_family_evidence'][axis]['reference_case']
        if reference not in {g['id'] for g in gravity_actions}:raise ValueError('Missing optimized reference gravity state')
        for gravity in gravity_actions:
            build_cache={}
            model=build_frame(state,gravity,capacity,axis,divisions=result['section_divisions'],curve_cache=cache,source=result['source_identity'])
            case=cases[(axis,gravity['id'])]
            if case['model_sha256']!=_hash(model) or case['represented_design_combination_ids']!=gravity['represented_design_combination_ids']:
                raise ValueError('Saved directional case input binding differs')
            by_story={r['story']:r for r in case['stories']}
            if len(by_story)!=len(case['stories']) or set(by_story)!=set(range(1,sp.NUM_FLOOR+1)):raise ValueError('Story certificate inventory differs')
            for story,row in by_story.items():
                senses={s['sway']:s for s in row['senses']}
                if len(row['senses'])!=2 or set(senses)!={-1,1}:raise ValueError('Both sway certificates are required')
                for sway,sense in senses.items():
                    inner=sense['inner']
                    if (inner['story'],inner['sway'],inner['bound'])!=(story,sway,'inner'):raise ValueError('Saved inner certificate target differs')
                    certificates.append(verify_solution(model,inner,build_cache=build_cache))
                    if not same(sense['lower_kip'],inner['strength_kip']):raise ValueError('Saved lower bound differs from its force certificate')
                    if gravity['id']==reference:
                        outer=sense['outer']
                        if outer['status']=='solved':
                            if (outer['story'],outer['sway'],outer['bound'])!=(story,sway,'outer'):raise ValueError('Saved outer certificate target differs')
                            certificates.append(verify_solution(model,outer,build_cache=build_cache))
                            if not same(sense['upper_kip'],outer['strength_kip']) or inner['strength_kip']>outer['strength_kip']+1e-5:raise ValueError('Saved optimized bounds are inconsistent')
                        elif outer['status']!='unresolved' or sense['upper_kip'] is not None:
                            raise ValueError('Uncertified outer bound or inconsistent infeasibility cannot be used')
                lo=min(s['inner']['strength_kip'] for s in senses.values())
                upper_values=[s['outer']['strength_kip'] for s in senses.values() if s['outer']['status']=='solved']
                if gravity['id']==reference and not upper_values:raise ValueError('At least one optimized sway upper certificate is required')
                hi=min(upper_values) if gravity['id']==reference else None
                if not same(row['lower_kip'],lo) or (hi is not None and not same(row['upper_kip'],hi)):raise ValueError('Saved case strength differs from certificates')
                all_rows[(gravity['id'],story)]=(lo,hi)
        for story in range(1,sp.NUM_FLOOR+1):
            lo=min(all_rows[(g['id'],story)][0] for g in gravity_actions);hi=all_rows[(reference,story)][1]
            saved=next(r for r in result['stories'] if r['story']==story)['by_direction'][axis]
            if not same(saved['story_strength_lower_kip'],lo) or not same(saved['story_strength_upper_kip'],hi) or not same(saved['story_strength_kip'],lo):
                raise ValueError('Saved family strength differs from its certified cases')
            if not 0.<lo<=hi+1e-5:raise ValueError('Nonpositive/inconsistent family resistance bounds')
            if not same(all_rows[(saved['governing_lower_case'],story)][0],lo):raise ValueError('Saved governing gravity case differs')
            reconstructed[axis].append({'story':story,'status':'bounded','lower_kip':lo,'upper_kip':hi})
    if len(result['stories'])!=sp.NUM_FLOOR or len({r['story'] for r in result['stories']})!=sp.NUM_FLOOR:raise ValueError('Family story inventory differs')
    ratios={a:adjacent_strength_bounds(reconstructed[a]) for a in ('x','y')}
    if ratios!=result['adjacent_ratio_bounds']:raise ValueError('Saved adjacent-strength intervals differ')
    report={'method':VERIFICATION_METHOD,'passed':True,'certificates_checked':len(certificates),
            'source_identity':result['source_identity'],'certificate_payload_sha256':_hash(_certificate_payload(result)),
            'maximum_recomputed_residual':max((v for row in certificates for k,v in row.items() if k!='exact_span_extrema_checked'),default=0.),
            'basis':'Physical model hashes, complete primal equilibrium/strength/span bounds and optimized dual upper proofs independently replayed; every gravity state and both senses retained'}
    return report


def calculate(state,actions,expected_ids,capacity,*,gravity_actions,divisions=128,time_limit_seconds=60.):
    """Exact gravity inventory covers every original signed-case load family."""
    _validate_inventory(state,actions,expected_ids,gravity_actions)
    source=source_identity();identity=state.identity();cache={};results=[];gaps=set();families={}
    base_tags=[str(m.member_tag) for m in state.members() if m.is_column and m.story_or_floor==1]
    ordered=sorted(gravity_actions,key=lambda c:sum(c['members'][tag]['centerline_axial_i_kip'] for tag in base_tags),reverse=True)
    for axis in ('x','y'):
        models=[]
        for case in ordered:
            if case['ex'] or case['ey']:raise ValueError('Directional story-strength gravity state must have no seismic preload')
            models.append(build_frame(state,case,capacity,axis,divisions=divisions,curve_cache=cache,source=source))
        family=analyze_gravity_family(models,time_limit_seconds=time_limit_seconds);families[axis]=family
        for case,model,result in zip(ordered,models,family['cases']):
            result.update(direction=axis,load_factors=model['load_factors'],actual_saved_actions_sha256=model['actual_saved_actions_sha256'],
                          orthogonal_column_actions=model['orthogonal_column_actions'],joint_bound_inputs=model['joint_bound_inputs'])
            result['scope_basis']=model['scope'];result['represented_design_combination_ids']=case['represented_design_combination_ids'];results.append(result)
        if family['status']!='bounded':gaps.add('At least one directional gravity-family strength bracket is incomplete')
    if state.identity()!=identity or source_identity()!=source:raise RuntimeError('Story strength inputs changed during calculation')
    nf=max(row['story'] for result in results for row in result['stories']);stories=[]
    for k in range(1,nf+1):
        item={'story':k,'by_direction':{}}
        for axis in ('x','y'):
            row=next(s for s in families[axis]['stories'] if s['story']==k);bounded=row['status']=='bounded'
            lo,hi=row['lower_kip'],row['upper_kip']
            item['by_direction'][axis]={'status':'bounded' if bounded else 'incomplete','story_strength_kip':lo,
                'story_strength_lower_kip':lo,'story_strength_upper_kip':hi,
                'governing_lower_case':row['governing_lower_case'],
                'bounds_scope':'directional nominal plane-frame resistance under the complete exact gravity load-family inventory'}
        stories.append(item)
    ratios={axis:adjacent_strength_bounds([{'story':r['story'],'status':r['by_direction'][axis]['status'],
                    'lower_kip':r['by_direction'][axis]['story_strength_lower_kip'],'upper_kip':r['by_direction'][axis]['story_strength_upper_kip']} for r in stories]) for axis in ('x','y')}
    for axis,rows in ratios.items():
        for row in rows:
            if row.get('status')!='bounded' or any(t in row.get('threshold_crossings',[]) for t in (.65,.8)):
                gaps.add(f'Direction {axis} story {row["story"]}: ratio is incomplete or still crosses a classification threshold')
    import Structure_Parameters as sp
    high=[a for a in actions if a.get('family')=='seismic_high_gravity'];low=[a for a in actions if a.get('family')=='seismic_low_gravity']
    full_ev=bool(high and low) and all(math.isclose(a['dead'],1.2+.2*sp.ASCE_SDS,rel_tol=1e-10) for a in high) and all(math.isclose(a['dead'],.9-.2*sp.ASCE_SDS,rel_tol=1e-10) for a in low)
    output={'method':METHOD,'adapter_method':ADAPTER_METHOD,'stories':stories,'cases':results,'adjacent_ratio_bounds':ratios,
            'section_divisions':divisions,'signed_actions_sha256':_hash(actions),'gravity_actions_sha256':_hash(gravity_actions),
            'unreduced_sds_for_vertical_ground_motion':full_ev,'unreduced_sds':sp.ASCE_SDS,
            'state_identity':identity,'source_identity':source,'case_inventory':list(expected_ids),'direction_cases':len(results),
            'gravity_family_evidence':{axis:{k:v for k,v in family.items() if k not in ('cases','stories')} for axis,family in families.items()},
            'unique_section_curves':len(cache),'applicability':{'status':'incomplete' if gaps else 'computed',
                'verified':False,'technical_gaps':sorted(gaps),'basis':'Computed directional method; complete saved force and dual certificates must independently replay'},
            'model':'full-plane static equilibrium with nominal N-M interaction, developed slabs, installed Vs, bounded joint traction, actual base/roof and adjacent-story mechanisms',
            'basis':'ASCE 7-22 Table12.3-2 story lateral strength; conditional generalized story force-pair resistance; retained brackets and signed load cases'}
    if not gaps:
        output['verification']=verify_saved_result(state,actions,expected_ids,capacity,gravity_actions=gravity_actions,result=output)
        output['applicability'].update(status='verified',verified=True)
    return output
