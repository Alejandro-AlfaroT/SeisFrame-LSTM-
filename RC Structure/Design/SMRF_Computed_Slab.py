"""Physical mat detailing and nominal development for the common-action route.

These calculations never reserve/reduce Mn or Mpr because a concurrent action
fails. Actual concurrent strength and method qualification remain separate checks.
No legacy zero-membrane assertion is read or created.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

from Design.SMRF_Common import make_check, not_evaluated, summarize_checks
from Design.SMRF_Common_Action_Contract import sha256

METHOD = "computed_physical_slab_detailing_and_nominal_development_v1"
_CACHE = {}


def _status(values):
    values=list(values)
    return "fail" if "fail" in values else "pass" if values and all(v=="pass" for v in values) else "unknown"


def _row(name,clause,status,reason,location="",details=None):
    row=(make_check(name,clause,int(status=="pass"),1,"==",location=location) if status in ("pass","fail")
         else not_evaluated(name,clause,reason,location))
    row["details"].update(details or {})
    return row


def _sources():
    root=Path(__file__).resolve().parents[1]
    names=("Design/SMRF_Computed_Slab.py","Design/SMRF_Slab_Reinforcement.py",
           "Design/SMRF_Actual_Bar_Development.py","Design/SMRF_Joint_Mat_Physical.py",
           "Model/Member_Properties.py","Structure_Parameters.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def nominal_development(state,inventory,slab_inputs,placement):
    """Full actual fy, both signs/sides, every actual bar in each effective flange.

    Support faces use the actual transverse beam widths, joint faces the actual
    joint core. Exterior hook datums remain distinct. The same immutable physical
    inventory used by common strength is required, never legacy strip assertions.
    """
    import Structure_Parameters as sp
    from Model import Member_Groups as mg,Member_Properties as mp
    from Design.SMRF_Joint_Mat_Physical import member_slab_inventory
    from Design.SMRF_Actual_Bar_Development import _mat_station,_nearest
    if not isinstance(inventory,dict) or inventory.get('state_identity')!=state.identity():
        return {'status':'unknown','members':{},'reason':'Exact material inventory absent or belongs to another state'}
    if inventory.get('identity_sha256')!=sha256({k:v for k,v in inventory.items() if k!='identity_sha256'}):
        return {'status':'unknown','members':{},'reason':'Material inventory digest differs'}
    if inventory.get('layout_sha256')!=placement.get('layout_sha256') or inventory.get('slab_thickness_in')!=sp.SLAB_THICKNESS_IN:
        return {'status':'unknown','members':{},'reason':'Material inventory mat layout/thickness differs'}
    key=sha256([state.identity(),inventory['identity_sha256'],slab_inputs,placement,_sources()])
    if key in _CACHE:return copy.deepcopy(_CACHE[key])
    members=[m for m in state.members() if not m.is_column];out={};nearest={};station_cache={}
    for member in members:
        axis=mp.beam_axis(member);other='y' if axis=='x' else 'x';floor=str(member.story_or_floor)
        layer_inventory=member_slab_inventory(member,mp.effective_flange(member),placement)
        floor_inventory=inventory.get('floors',{}).get(floor,{})
        bars=floor_inventory.get('bars',[]);by_id={b['id']:b for b in bars}
        selected=layer_inventory['bars'];missing=[b['id'] for b in selected if b['id'] not in by_id]
        actual=placement.get('floors',{}).get(floor,{})
        floor_members=[m for m in members if str(m.story_or_floor)==floor]
        fc=min([slab_inputs['fc_ksi']]+[m.design.fc_ksi for m in floor_members])
        depth=max([sp.SLAB_THICKNESS_IN]+[m.design.h_in for m in floor_members])
        nodes={'i':(member.grid_i,member.grid_j),'j':(member.grid_i+int(axis=='x'),member.grid_j+int(axis=='y'))}
        depths=dict(zip(('i','j'),mp.beam_end_joint_depths(member)));ends={}
        for end,(gi,gj) in nodes.items():
            joint=mg.joint_members(member.story_or_floor,gi,gj)
            crossing=[b for b in (joint[f'beam_{other}_minus'],joint[f'beam_{other}_plus']) if b is not None]
            origin=gi*sp.BAY_X if axis=='x' else gj*sp.BAY_Y;sign=1 if end=='i' else -1
            datums={'support_face':None if not crossing else origin+sign*min(b.design.b_in for b in crossing)/2,
                    'joint_face':origin+sign*depths[end]/2}
            proof={}
            for name,station in datums.items():
                records=[];problems=list(missing)
                if station is None:problems.append('No actual transverse beam support face')
                for expected in selected:
                    if expected['id'] not in by_id or station is None:continue
                    bar=by_id[expected['id']]
                    if (bar.get('operator_owner')!='slab' or bar.get('axis')!=axis or
                        any(abs(float(bar.get(k,1e100))-float(v))>1e-8 for k,v in
                            (('s_in',expected['global_transverse_coordinate_in']),('z_in',sp.SLAB_THICKNESS_IN/2-expected['centroid_from_top_in']),
                             ('db_in',expected['db_in']),('area_in2',expected['area_in2']),('fy_ksi',expected['fy_ksi']))) or
                        bar.get('straight_run_interval_in')!=expected['straight_run_interval_in']):
                        problems.append('Physical bar differs: '+expected['id']);continue
                    mat=actual['layers'][f'{axis}_{expected["face"]}']
                    k=(bar['id'],station,fc,depth)
                    if k not in station_cache:
                        if bar['id'] not in nearest:nearest[bar['id']]=_nearest(bar,bars)
                        _,result=_mat_station(bar,station,mat,actual['footprint'],sp.SLAB_THICKNESS_IN,fc,depth,bars,True,
                                              nearest_center_spacing_in=nearest[bar['id']])
                        station_cache[k]=result
                    records.append({'physical_bar_id':bar['id'],'check':station_cache[k]})
                status=_status([r['check']['status'] for r in records]+(['unknown'] if problems or not records else [])+
                               [layer_inventory['status']])
                proof[name]={'status':status,'station_in':station,'bars':records,'problems':problems}
            statuses=[proof[k]['status'] for k in ('support_face','joint_face')]
            ends[end]={**proof,'status':_status(statuses),'developed_support_face':statuses[0]=='pass',
                'developed_joint_face':statuses[1]=='pass','developed_either':'pass' in statuses,
                'developed_both':all(s=='pass' for s in statuses),'exterior':mp.exterior_ends(member)[end]}
        out[str(member.member_tag)]={'axis':axis,'floor':member.story_or_floor,'ends':ends,
            'status':_status(e['status'] for e in ends.values()),'physical_bar_ids':[b['id'] for b in selected]}
    result={'method':METHOD,'status':_status(v['status'] for v in out.values()),'members':out,
            'state_identity':state.identity(),'material_inventory_id':inventory['identity_sha256'],
            'full_fy_only':True,'concurrent_demand_failure_established':False,'strength_reduction_factor_applied':False}
    result['identity']=sha256(result)
    if len(_CACHE)>8:_CACHE.clear()
    _CACHE[key]=copy.deepcopy(result)
    return result


def physical_integrity(state,placement):
    """Count actual globally phased bottom bars crossing each confined column core."""
    import Structure_Parameters as sp
    from Design.Group_Capacity import joint_inventory
    from Design.SMRF_Transitions import bar_coordinates
    rows=[]
    for joint in joint_inventory(state)['joints']:
        core=joint['core'].design;level=joint['level'];mats=placement.get('floors',{}).get(str(level),{}).get('layers',{})
        for axis in ('x','y'):
            mat=mats.get(axis+'_bottom') or {};db=mat.get('db_in');center=(joint['grid_j']*sp.BAY_Y if axis=='x' else joint['grid_i']*sp.BAY_X)
            if db is None or mat.get('phase_in') is None:
                rows.append({'joint_id':joint['id'],'axis':axis,'status':'unknown','actual_bar_ids':[]});continue
            coordinates=bar_coordinates(core,sp.COL_CLEAR_COVER_IN)
            key='x_in' if axis=='x' else 'y_in'
            lo=min(p[key] for p in coordinates)+sp.rebar_diameter(core.bar_size)/2+db/2
            hi=max(p[key] for p in coordinates)-sp.rebar_diameter(core.bar_size)/2-db/2
            selected=[f'floor_{level}/slab/{axis}_bottom/{i}' for i,s in enumerate(mat['global_transverse_coordinates_in'])
                      if lo-1e-8<=s-center<=hi+1e-8]
            rows.append({'joint_id':joint['id'],'axis':axis,'status':'pass' if len(selected)>=2 else 'fail',
                         'actual_bar_ids':selected,'count':len(selected),'required':2,'core_center_in':center,
                         'usable_transverse_interval_in':[center+lo,center+hi],
                         'basis':'Inside actual column longitudinal bar envelopes; exterior anchorage checked separately'})
    return {'status':_status(r['status'] for r in rows),'joints':rows}


def computed_record(slab_inputs,layout,*,state,material_inventory,common_actions=None,procedure=None,policy=None,slab_geometry_record=None):
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Design import SMRF_Slab_Reinforcement as sr
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    from Design.SMRF_Slab import evaluate_slab
    inputs=sr._inputs(slab_inputs);resolved=sr._policy(policy)
    if mg.active().identity()!=state.identity():raise ValueError('The actual grouped state must be installed')
    if (sp.SLAB_REINFORCEMENT or {}).get('layout')!=layout or sp.SLAB_THICKNESS_IN!=inputs['thickness_in']:
        raise ValueError('The exact proposed slab and mat layout must be installed before physical evaluation')
    checks=[];layers={}
    for name in ('x_top','x_bottom','y_top','y_bottom'):
        layer=(layout or {}).get('layers',{}).get(name) or {};axis,face=name.split('_')
        bar=layer.get('bar_size');spacing=layer.get('spacing_in')
        if bar not in resolved['bar_sizes'] or spacing not in resolved['spacing_options_in']:
            raise ValueError('Selected physical mat is outside the unchanged slab catalog')
        expected=sr._layout(inputs,resolved,bar,axis,spacing)
        matches=expected is not None and all(layer.get(k)==v for k,v in expected.items() if k not in ('flexure','one_way_shear'))
        checks.append(make_check('slab_computed_layout_definition','Declared actual bar geometry and quantities',int(matches),1,'==',location=name))
        if expected is None:continue
        layers[name]=expected
        for check_id,clause,demand,capacity,comparison in (
            ('slab_strip_minimum_flexural_steel','ACI 318-19 8.6.1.1',expected['area_in2_per_ft'],expected['minimum_area_in2_per_ft'],'>='),
            ('slab_strip_maximum_spacing','ACI 318-19 8.7.2.2',spacing,expected['maximum_spacing_in'],'<='),
            ('slab_strip_horizontal_clear_spacing','ACI 318-19 25.2.1',expected['horizontal_clear_spacing_in'],expected['minimum_clear_spacing_in'],'>=')):
            checks.append(make_check(check_id,clause,demand,capacity,comparison,location=name))
        maximum=min(15*(40/(2/3*inputs['fy_ksi']))-2.5*resolved['clear_cover_in'],12*(40/(2/3*inputs['fy_ksi'])))
        checks.append(make_check('slab_crack_control_spacing','ACI 318-19 24.3.2 with fs=2/3fy',spacing,maximum,'<=','in',name))
    bar=(layout or {}).get('bar_size');diameter=sp.rebar_diameter(bar)
    checks.append(make_check('slab_strip_cover','ACI 318-19 Table20.5.1.3.1',resolved['clear_cover_in'],.75,'>=','in'))
    gap=inputs['thickness_in']-2*resolved['clear_cover_in']-4*diameter
    checks.append(make_check('slab_strip_face_mat_clearance','ACI 318-19 25.2',gap,max(1.,4/3*inputs['max_aggregate_size_in']),'>=','in'))
    placement=global_mat_placement(state)
    checks.append(_row('slab_computed_physical_mat_inventory','Actual outer-face footprint/global phase/hook geometry',placement['status'],
                       'Installed mat placement is incomplete',details={'layout_sha256':placement.get('layout_sha256')}))
    nominal=nominal_development(state,material_inventory,inputs,placement)
    checks.append(_row('slab_bar_development','ACI 318-19 25.4.2/25.4.3/25.4.9; actual full-fy nominal member credit',
        'pass' if nominal['status']=='pass' else 'unknown','Full-fy nominal development credit is not established; actual concurrent failure is not inferred',details={'evidence_identity':nominal.get('identity'),
        'full_fy_proof_status':nominal['status'],
        'full_fy_sufficiency_only':True,'actual_concurrent_demand_failure_established':False}))
    for axis in ('x','y'):
        ends=[end for m in nominal['members'].values() if m['axis']==axis for end in m['ends'].values() if end['exterior']]
        status=_status(end['support_face']['status'] for end in ends)
        checks.append(_row('slab_perimeter_bar_anchorage','ACI 318-19 25.4;8.7.4.1.1(b), actual support face',status,
                           'Actual perimeter development is incomplete',axis))
    integrity=physical_integrity(state,placement)
    checks.append(_row('slab_integrity_bottom_bars_through_column','ACI 318-19 8.7.4.2',integrity['status'],
                       'Actual phased bottom-bar/core inventory is incomplete',details=integrity))
    continuous=(placement['status']=='pass' and (layout or {}).get('uniform_all_panels_and_floors') is True and
                all(m.get('straight_run_interval_in',[0,0])[1]>m.get('straight_run_interval_in',[0,0])[0]
                    for f in placement.get('floors',{}).values() for m in f.get('layers',{}).values()))
    checks.append(_row('slab_continuity_and_extensions','ACI 318-19 8.7.4.1.3',
                       'pass' if continuous else placement['status'] if placement['status']!='pass' else 'fail',
                       'Actual continuous mat path is incomplete'))
    thickness=evaluate_slab(slab_geometry_record)
    checks.append(_row('slab_deflection_control','ACI 318-19 8.3.2.1 via8.3.1.2 minimum thickness',
                       _status(c['status'] for c in thickness),'Actual thickness screen does not reproduce',details={'recomputed':thickness}))
    # Actual corner capacity/demand records are supplied by the common response
    # producer. Until their exact same-case inventory exists this remains unknown.
    corner=corner_checks(state,placement,inputs,layout,(common_actions or {}).get('slab_corner_demands'),material_inventory)
    checks.extend(corner['checks'])
    saved_inputs={'slab':inputs,'policy':resolved,'context':None,'material_inventory':copy.deepcopy(material_inventory),
                  'slab_geometry_record':copy.deepcopy(slab_geometry_record),
                  'slab_corner_demands':copy.deepcopy((common_actions or {}).get('slab_corner_demands'))}
    return {'method_version':METHOD,'stage':'computed_common_physical_slab','accepted':False,
            'strip_screen_passed':False,'screen_passed':all(c['status']=='pass' for c in checks),
            'layout':copy.deepcopy(layout),'inputs':saved_inputs,'checks':checks,'summary':summarize_checks(checks),
            'state_identity':state.identity(),'source_sha256':_sources(),'nominal_development':nominal,
            'physical_integrity':integrity,'corner_inventory':corner['inventory'],
            'common_action_identity':None if common_actions is None else common_actions.get('payload_sha256'),
            'procedure_identity':None if procedure is None else procedure.get('identity'),
            'capacity_design_policy':'Full actual nominal/probable beam/slab strengths; no shared-state reserve deduction'}


def corner_checks(state,placement,inputs,layout,demands,material_inventory=None):
    """Actual phase counts/coverage; missing same-case positive demand stays unknown."""
    import Structure_Parameters as sp
    from Design.SMRF_Slab_Reinforcement import rectangular_strip_strength
    from Design.SMRF_Actual_Bar_Development import _mat_station
    inventory=[];checks=[];extent=max(sp.BAY_X,sp.BAY_Y)/5
    for floor,record in placement.get('floors',{}).items():
        fp=record['footprint']
        for xs in ('minus','plus'):
            for ys in ('minus','plus'):
                for name,mat in record['layers'].items():
                    axis=name[0];cross='y' if axis=='x' else 'x';side=ys if axis=='x' else xs
                    edge=fp[f'{cross}_min_in'] if side=='minus' else fp[f'{cross}_max_in']
                    band=[edge,edge+extent] if side=='minus' else [edge-extent,edge]
                    bars=[f'floor_{floor}/slab/{name}/{i}' for i,s in enumerate(mat.get('global_transverse_coordinates_in',[]))
                          if band[0]<=s<=band[1]]
                    inventory.append({'floor':int(floor),'corner':xs+'_'+ys,'layer':name,'region_extent_in':extent,
                        'transverse_interval_in':band,'physical_bar_ids':bars,'actual_area_in2_per_ft':len(bars)*sp.rebar_area(mat['bar_size'])*12/extent,
                        'spacing_in':mat['spacing_in'],'status':placement['status']})
    by={};problems=[]
    try:
        if not isinstance(demands,dict) or demands.get('state_identity')!=state.identity() or not demands.get('source_id'):
            raise ValueError('Exact source/state-bound corner panel actions are absent')
        if demands.get('primary_guard_coverage_complete') is not True:
            raise ValueError('Complete physical and primary-guard adjacent-panel corner envelope required')
        cases=demands['expected_case_ids']
        if not isinstance(cases,list) or not cases or len(set(cases))!=len(cases):raise ValueError('Unique actual corner cases are required')
        expected={(int(floor),xs+'_'+ys,case) for floor in placement['floors'] for xs in ('minus','plus') for ys in ('minus','plus') for case in cases}
        for row in demands['records']:
            key=(row['floor'],row['corner'],row['case_id'])
            if key in by:raise ValueError('Duplicate corner case')
            if not row.get('recovery_id') or row.get('region_basis')!='entire_adjacent_slab_panel':raise ValueError('Corner demand must envelope its entire adjacent panel')
            for axis in ('x','y'):
                value=row['mu_positive_'+axis+'_kip_in_per_ft']
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative positive-panel moments required')
            by[key]=row
        if set(by)!=expected:raise ValueError('Actual floor/corner/case inventory is incomplete')
        if not isinstance(material_inventory,dict) or material_inventory.get('state_identity')!=state.identity():
            raise ValueError('Actual corner development inventory missing')
    except (ValueError,TypeError,KeyError) as exc:problems.append(str(exc))
    for name in ('x_top','y_top','x_bottom','y_bottom'):
        assessments=[]
        if not problems:
            for region in [r for r in inventory if r['layer']==name]:
                floor=str(region['floor']);actual=placement['floors'][floor];mat=actual['layers'][name];fp=actual['footprint']
                bars=material_inventory['floors'][floor]['bars'];by_id={b['id']:b for b in bars};axis=name[0]
                side=region['corner'].split('_')[0 if axis=='x' else 1]
                width=fp['perimeter_beam_widths_in'][axis+'_'+side]
                station=fp[axis+'_min_in']+width if side=='minus' else fp[axis+'_max_in']-width
                floor_members=[m for m in state.members() if not m.is_column and m.story_or_floor==int(floor)]
                fc=min([inputs['fc_ksi']]+[m.design.fc_ksi for m in floor_members]);depth=max([inputs['thickness_in']]+[m.design.h_in for m in floor_members])
                development=[]
                for bar_id in region['physical_bar_ids']:
                    if bar_id not in by_id:development.append({'status':'unknown','reason':'Actual corner bar absent'});continue
                    status,proof=_mat_station(by_id[bar_id],station,mat,fp,inputs['thickness_in'],fc,depth,bars,True)
                    development.append({'status':status,'physical_bar_id':bar_id,'proof':proof})
                strength=rectangular_strip_strength(inputs['fc_ksi'],inputs['fy_ksi'],layout['layers'][name]['effective_depth_in'],region['actual_area_in2_per_ft'])
                governing=max((by[(region['floor'],region['corner'],case)] for case in cases),
                              key=lambda r:max(r['mu_positive_x_kip_in_per_ft'],r['mu_positive_y_kip_in_per_ft']))
                demand=max(governing['mu_positive_x_kip_in_per_ft'],governing['mu_positive_y_kip_in_per_ft'])
                assessments.append({'floor':region['floor'],'corner':region['corner'],'case_id':governing['case_id'],
                    'recovery_id':governing['recovery_id'],'demand_kip_in_per_ft':demand,
                    'capacity_kip_in_per_ft':strength['phi_mn_kip_in_per_ft'],'development_status':_status(x['status'] for x in development),
                    'development':development,'regional_strength':strength,'actual_bar_ids':region['physical_bar_ids']})
        if problems or not assessments or placement['status']!='pass' or any(r['development_status']!='pass' for r in assessments):
            check=not_evaluated('slab_computed_corner_reinforcement','ACI 318-19 8.7.3.1--8.7.3.1.3',
                '; '.join(problems) or 'Full-fy regional corner capacity/development is not established; actual demand failure is not inferred',name)
        else:
            worst=max(assessments,key=lambda r:r['demand_kip_in_per_ft']/max(r['capacity_kip_in_per_ft'],1e-30))
            check=make_check('slab_computed_corner_reinforcement','ACI 318-19 8.7.3.1--8.7.3.1.3',
                worst['demand_kip_in_per_ft'],worst['capacity_kip_in_per_ft'],'<=','kip-in/ft',name)
        check['details'].update(actual_regions=assessments,primary_basis='ACI318-19 printed114/PDF116; orthogonal mats permitted by8.7.3.1.3',
            strength_basis='Actual bar count divided by entire Llong/5 region width, fullfy developed at actual support face; full slab field still checked concurrently')
        checks.append(check)
    return {'checks':checks,'inventory':inventory,'supplied_demand_present':demands is not None}


def replay(record):
    from Model import Member_Groups as mg
    data=record['inputs']
    result=computed_record(data['slab'],record['layout'],state=mg.active(),material_inventory=data['material_inventory'],
        policy=data['policy'],slab_geometry_record=data.get('slab_geometry_record'),
        common_actions={'slab_corner_demands':data.get('slab_corner_demands'),'payload_sha256':record.get('common_action_identity')},
        procedure={'identity':record.get('procedure_identity')})
    return result


def eligibility(record,current_context=None):
    from Model import Member_Groups as mg
    current=replay(record)
    fresh=current==record
    nominal=current['nominal_development'];complete=all(c['status']!='not_evaluated' for c in current['checks'])
    development=nominal['status'] if fresh else 'unknown'
    return {'version':METHOD,'record_present':True,'layout_selected':current.get('layout') is not None,
            'slab_checks_complete':complete,'slab_checks_passed':fresh and complete and all(c['status']=='pass' for c in current['checks']),
            'interior_development':development,'perimeter_anchorage':{axis:_status(e['support_face']['status'] for m in nominal['members'].values()
                if m['axis']==axis for e in m['ends'].values() if e['exterior']) for axis in ('x','y')},
            'context_current':fresh and mg.active().identity()==record['state_identity'],
            'credit':'developed' if fresh and development=='pass' else 'not_established',
            'reason_code':'computed_actual_nominal_development' if fresh else 'computed_slab_record_stale',
            'reason':'Nominal development is computed per actual member/end/datum; shared strength does not reduce Mn/Mpr.',
            'member_development':nominal['members'] if fresh else {},'signature':sha256(current),
            'failed_checks':[c['id'] for c in current['checks'] if c['status']=='fail'],
            'unknown_checks':[c['id'] for c in current['checks'] if c['status']=='not_evaluated'],
            'review_dependencies':[],'scope_exclusions':[]}
