"""Final-state coordination for the explicit computed common-action route.

The original catalogs and reinforcement selector are retained. A changed
cage or mat invalidates the whole mechanical input. A bounded iteration or
cycle is unresolved; neither can reuse old actions as final-design evidence.
"""
from __future__ import annotations
import copy
import Structure_Parameters as sp
from Model import Member_Groups as mg,Member_Bar_Layers as bar_layers
from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Coupled_Floor_Design import _hash

METHOD='computed_common_final_state_coordination_v3_layout_reselection'
# The one inventory refusal a lane-clear slab layout can repair: slab bars in the planes of hooked beam bars.
STRAIGHT_INTERVALS_UNRESOLVED='Authoritative installed beam straight intervals are unresolved'


def active(cfg):
    return (getattr(cfg.floor_analysis,'common_action_procedure',None) or {}).get('kind')=='computed_common_actions_v1'


def design_signature(state):
    return _hash(dict(state=state.identity(),slab_thickness_in=sp.SLAB_THICKNESS_IN,
        layout=(sp.SLAB_REINFORCEMENT or {}).get('layout')))


def bounded_layout_proposals(candidates,place,*,limit=32,lane_clear_only=False,known=()):
    """Walk the catalog in its own order until ``place`` passes one layout or ``limit`` phase searches were made.

    ``place(candidate)`` returns (status, {layer name: 'pass'|'columns_only'|'fail'} or None, attempt record). A layer
    the search failed is remembered by (name, bar size, spacing, largest bar of the layout), the quantities its phase
    search depends on, and later layouts that contain it are passed over without another search. With
    ``lane_clear_only`` a layer whose phase clears the column bars only ('columns_only') is remembered and passed over
    in the same way, and a layout is taken only when every layer is 'pass'. ``known``: layer keys already found
    unusable. Returns (selected candidate or None, attempts, passed over, searches made, unusable layer keys)."""
    def keys(candidate):
        rows=candidate['layout']['layers'];default=candidate['layout'].get('bar_size')
        sizes={n:int(r.get('bar_size',default)) for n,r in rows.items()};largest=max(sizes.values())
        return {n:(n,sizes[n],float(r['spacing_in']),largest) for n,r in rows.items()}
    refused=('fail','columns_only') if lane_clear_only else ('fail',)
    attempts=[];unplaceable={tuple(k) for k in known};passed_over=0;searches=0
    for candidate in candidates:
        layer_keys=keys(candidate)
        if any(k in unplaceable for k in layer_keys.values()):
            passed_over+=1;continue
        if searches>=limit:break
        searches+=1
        status,layer_status,attempt=place(candidate)
        attempts.append(attempt)
        if layer_status:
            unplaceable.update(k for n,k in layer_keys.items() if layer_status.get(n) in refused)
        if status=='pass' and not (lane_clear_only and any(v in refused for v in (layer_status or {}).values())):
            return candidate,attempts,passed_over,searches,sorted(unplaceable)
    return None,attempts,passed_over,searches,sorted(unplaceable)


def reselect_layout(state,catalog,current_layout,*,limit=32,lane_clear_only=False):
    """A cage change left the installed slab mat layout without a placement: choose again from the original catalog.

    Candidates are the catalog layouts, in catalog order, that are not weaker than the installed layout in any layer
    (a cage change must not thin the slab steel). The first whose global placement passes on the new cages is installed
    and its inventory built; if that inventory is refused and a layer clears the column bars only, one further walk
    takes the first layout whose every layer also clears the beam lanes. Returns (candidate or None, evidence); with
    None the earlier layout is left installed. Nothing here accepts a mat: its actions and every check follow."""
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    from Design.SMRF_Physical_Floor_Materials import installed_inventory,PhysicalInventoryUnresolved
    previous=sp.SLAB_REINFORCEMENT
    area={n:r['area_in2_per_ft'] for n,r in current_layout['layers'].items()}
    eligible=[c for c in catalog['candidates']
              if all(c['layout']['layers'][n]['area_in2_per_ft']>=area[n]-1e-10 for n in area)]
    rules={}
    def install(candidate):
        sp.SLAB_REINFORCEMENT=dict(layout=copy.deepcopy(candidate['layout']),accepted=False,
                                   stage='computed_cage_change_layout_reselection')
    def place(candidate):
        install(candidate)
        audit=bar_layers.row_audit(state);bar_layers.raise_arrangement_errors(audit)
        if not audit['valid']:
            return 'actual_beam_rows_do_not_fit',None,dict(candidate_id=candidate['candidate_id'],status='actual_beam_rows_do_not_fit')
        placement=global_mat_placement(state)
        status={n:('columns_only' if r.get('status')=='pass' and r.get('phase_rule')=='columns_only' else r.get('status'))
                for n,r in (placement.get('layers') or {}).items()}
        rules[candidate['candidate_id']]=status
        return placement['status'],status,dict(candidate_id=candidate['candidate_id'],status=placement['status'],
                                               layout_sha256=candidate['layout_sha256'])
    evidence=dict(method=METHOD+'/layout_reselection_after_cage_change',previous_layout_sha256=_hash(current_layout),
        state_identity=state.identity(),eligible_not_weaker_count=len(eligible),
        original_catalog_count=catalog['enumerated_count'],walks=[])
    known=();chosen=None
    try:
        # ``lane_clear_only``: the installed layout places but its bars lie in planes of hooked beam bars; only the
        # lane-clear walk can help.
        for lane_clear in ((True,) if lane_clear_only else (False,True)):
            candidate,attempts,passed_over,searches,unusable=bounded_layout_proposals(eligible,place,limit=limit,
                lane_clear_only=lane_clear,known=known)
            walk=dict(lane_clear_only=lane_clear,phase_searches=searches,passed_over=passed_over,attempts=attempts[-8:],
                selected=None if candidate is None else candidate['candidate_id'],unusable_layers=[list(k) for k in unusable])
            evidence['walks'].append(walk)
            if candidate is None:break
            install(candidate)
            try:installed_inventory(state)
            except PhysicalInventoryUnresolved as exc:
                retry=(not lane_clear and str(exc)==STRAIGHT_INTERVALS_UNRESOLVED
                       and 'columns_only' in (rules.get(candidate['candidate_id']) or {}).values())
                # The walk that ends the re-selection keeps the whole refusal; one that is retried keeps its summary.
                walk['inventory']=dict(status='unresolved',reason=str(exc),
                    evidence={k:v for k,v in (exc.evidence or {}).items() if not retry or k!='members'})
                if not retry:break
                known=unusable;continue
            walk['inventory']=dict(status='established');chosen=candidate;break
    finally:
        # Whatever ends the walks, a trial layout is never left installed unless it was chosen.
        if chosen is None:sp.SLAB_REINFORCEMENT=previous
    evidence.update(selected_candidate_id=None if chosen is None else chosen['candidate_id'],
        selected_layout_sha256=None if chosen is None else chosen['layout_sha256'])
    return chosen,evidence


def prepare_slab(cfg,context):
    from Design.Grouped_Design import CandidateInfeasible,_slab_policy
    from Design import Design_Driver as driver
    from Design.SMRF_Slab import choose_slab,SlabSizingError
    from Design.SMRF_Physical_Slab_Selection import catalog_proposals
    from Design.SMRF_Slab_Reinforcement import computed_layout,computed_slab_reinforcement
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    from Design.SMRF_Physical_Floor_Materials import installed_inventory
    description=floor_sections.by_floor()
    try:slab=choose_slab(driver._slab_geometry(),description,_slab_policy(cfg))
    except SlabSizingError as exc:raise CandidateInfeasible('slab_thickness',str(exc)) from exc
    driver._apply_slab(slab);sp.FLOOR_TRANSFER=None;sp.SLAB_ACTIONS=None
    inputs=driver._slab_strength_inputs_from_state(slab,cfg);catalog=catalog_proposals(inputs)
    state=mg.active();selected=None;layouts={};layer_rules={}
    # This is an explicit bounded proposal stage, not a smaller catalog.
    # Exhaustion of this allowance is reported incomplete with all untried
    # original alternatives still available to subsequent coordination.
    def place(candidate):
        layout=computed_layout(inputs,candidate['layout']['bar_size'],
            {n:r['spacing_in'] for n,r in candidate['layout']['layers'].items()})
        sp.SLAB_REINFORCEMENT=dict(layout=layout,accepted=False,stage='computed_catalog_geometry_trial')
        audit=bar_layers.row_audit(state);bar_layers.raise_arrangement_errors(audit)
        if not audit['valid']:
            return 'actual_beam_rows_do_not_fit',None,dict(candidate_id=candidate['candidate_id'],status='actual_beam_rows_do_not_fit',audit=audit)
        placement=global_mat_placement(state);layouts[candidate['candidate_id']]=layout
        layer_status={n:('columns_only' if r.get('status')=='pass' and r.get('phase_rule')=='columns_only' else r.get('status'))
                      for n,r in (placement.get('layers') or {}).items()}
        layer_rules[candidate['candidate_id']]=layer_status
        return placement['status'],layer_status,dict(
            candidate_id=candidate['candidate_id'],status=placement['status'],layout_sha256=_hash(layout),
            placement_evidence_sha256=_hash(placement))
    chosen,attempts,passed_over,searches,unplaceable=bounded_layout_proposals(catalog['candidates'],place)
    if passed_over:
        attempts.append(dict(candidate_id=None,status='passed_over_known_unplaceable_layer',count=passed_over,
            layers=[list(k) for k in unplaceable]))
    if chosen is not None:
        selected=(chosen,layouts[chosen['candidate_id']])
        # The last trial installed is the selected one; state it again so nothing depends on the walk order.
        sp.SLAB_REINFORCEMENT=dict(layout=selected[1],accepted=False,stage='computed_catalog_geometry_trial')
    if selected is None:
        raise CandidateInfeasible('slab_reinforcement','Bounded actual mat placement did not establish an installed layout; no catalog infeasibility is inferred',
            dict(outcome='incomplete_physical_layout_selection',attempts=attempts,catalog_payload_sha256=catalog['payload_sha256'],
                original_catalog_count=catalog['enumerated_count'],phase_searches=searches,
                passed_over_known_unplaceable_layer=passed_over,unplaceable_layers=[list(k) for k in unplaceable],
                untried_candidates=max(0,len(catalog['candidates'])-searches-passed_over)))
    refusals=('Authoritative actual global mat placement is unresolved','Actual beam cage lanes are unresolved',
              'Authoritative installed beam straight intervals are unresolved')
    try:inventory=installed_inventory(state)
    except ValueError as exc:
        if str(exc) not in refusals:raise
        first=dict(candidate_id=selected[0]['candidate_id'],layout_sha256=_hash(selected[1]),reason=str(exc),
                   layer_phase_rules=layer_rules.get(selected[0]['candidate_id']),inventory_evidence=getattr(exc,'evidence',None))
        retry=None
        if str(exc)==STRAIGHT_INTERVALS_UNRESOLVED and 'columns_only' in (first['layer_phase_rules'] or {}).values():
            # The layout's bars lie in planes of hooked beam bars. One further bounded walk of the same catalog for the
            # first layout whose every layer also clears the beam lanes; the layers already known unplaceable carry over.
            chosen,more,passed,made,unusable=bounded_layout_proposals(catalog['candidates'],place,lane_clear_only=True,known=unplaceable)
            attempts.extend(more)
            attempts.append(dict(candidate_id=None,status='lane_clear_walk',phase_searches=made,passed_over=passed,
                selected=None if chosen is None else chosen['candidate_id'],unusable_layers=[list(k) for k in unusable]))
            if chosen is not None:
                selected=(chosen,layouts[chosen['candidate_id']])
                sp.SLAB_REINFORCEMENT=dict(layout=selected[1],accepted=False,stage='computed_catalog_geometry_trial')
                try:inventory=installed_inventory(state);retry='established'
                except ValueError as again:
                    if str(again) not in refusals:raise
                    retry=dict(candidate_id=chosen['candidate_id'],layout_sha256=_hash(selected[1]),reason=str(again),
                               inventory_evidence=getattr(again,'evidence',None))
        if retry!='established':
            raise CandidateInfeasible('slab_reinforcement',str(exc),dict(outcome='physical_inventory_not_established',
                state_identity=state.identity(),layout_sha256=first['layout_sha256'],attempts=attempts,
                inventory_evidence=first['inventory_evidence'],first_layout=first,lane_clear_layout=retry)) from exc
        # The lighter layout was passed over for this reason; the record keeps it.
        attempts.append(dict(candidate_id=first['candidate_id'],status='first_layout_inventory_refused',reason=first['reason'],
            layout_sha256=first['layout_sha256'],layer_phase_rules=first['layer_phase_rules']))
    procedure=cfg.floor_analysis.common_action_procedure
    reinforcement=computed_slab_reinforcement(inputs,selected[1],state=state,material_inventory=inventory,
        procedure=procedure,slab_geometry_record=slab)
    sp.SLAB_REINFORCEMENT=reinforcement
    context['computed_slab_catalog']=catalog
    return dict(slab=slab,description=description,transfer=None,evidence=dict(method=METHOD,
        physical_inventory_sha256=inventory['identity_sha256'],legacy_assertions_used=False),reinforcement=reinforcement,
        material_inventory=inventory,attempts=attempts,computed_catalog_candidate_id=selected[0]['candidate_id'],
        original_catalog_sha256=catalog['payload_sha256'],original_catalog_count=catalog['enumerated_count'])


def refresh_slab(state,slab,cfg,common):
    from Design.SMRF_Slab_Reinforcement import computed_slab_reinforcement
    from Design import Design_Driver as driver
    layout=copy.deepcopy(sp.SLAB_REINFORCEMENT['layout'])
    if common['state_identity']!=state.identity() or common['inventory']['layout_sha256']!=_hash(layout):
        raise ValueError('Final common actions and selected actual mat/cage differ')
    slab['material_inventory']=common['inventory']
    slab['reinforcement']=computed_slab_reinforcement(driver._slab_strength_inputs_from_state(slab['slab'],cfg),layout,
        state=state,material_inventory=common['inventory'],common_actions=common,
        procedure=cfg.floor_analysis.common_action_procedure,slab_geometry_record=slab['slab'])
    slab['transfer']=common['capacity_gravity_transfer']
    slab['evidence']=dict(method=METHOD,common_action_input_identity=common['input_signature'],
        source_identity=common['analysis_source_identity'],physical_inventory_sha256=common['inventory']['identity_sha256'],
        legacy_assertions_used=False)
    sp.SLAB_REINFORCEMENT=slab['reinforcement'];sp.FLOOR_TRANSFER=slab['transfer'];sp.SLAB_ACTIONS=slab['evidence']


def analyze(state,cfg,context,period):
    from Design.SMRF_Common_Action_Analysis import analyze_installed,current_source_identity
    from Design.SMRF_Common_Action_Procedure import material_basis_from_protocol
    import openseespy.opensees as ops
    # The period model of this evaluation is the evaluator's own and is not read again on this route: the native
    # floor kernel requires an empty domain and must never clear a model it does not own (2026-11-05).
    ops.wipe()
    protocol=cfg.floor_analysis.common_action_procedure['procedure']['protocol']
    return analyze_installed(state,cfg,source_identity=current_source_identity(),model_period_sec=period,
        limits=protocol['analysis_limits'],progress=context.get('common_progress'),final_assessment=False,
        **material_basis_from_protocol(protocol))


def _coordinate_one_layout(state,cfg,context,slab,period,common,*,maximum_passes=10):
    from Design import Grouped_Design as gd,Perimeter_Systems as systems,Group_Capacity as capacities,Group_Selection as selection
    from Design.SMRF_Physical_Floor_Materials import PhysicalInventoryUnresolved
    from Design.SMRF_Joints import evaluate_joints
    from Design.SMRF_Slab_Reinforcement import slab_strength_eligibility
    if type(maximum_passes) is not int or maximum_passes<1:raise ValueError('Positive finite actual coordination bound required')
    history=[];stable=False;report=None;reserve=[];period_solves=0;cycle=None;stop='pass_budget';reselections=[]
    def beam_cages():
        return {gid:dict(b_in=d.b_in,h_in=d.h_in,bar_size=d.bar_size,top_bars=d.top_bars,bot_bars=d.bot_bars,
                         stirrup_bar_size=d.stirrup_bar_size)
                for gid,d in sorted(state.designs.items()) if state.groups[gid]['member_type']!='column'}
    def update_actual(candidate):
        nonlocal state,period,common,period_solves
        state=candidate;mg.install(state)
        installed=(sp.SLAB_REINFORCEMENT or {}).get('layout')
        if installed is not None and common['state_identity']!=state.identity():
            # A cage change can leave the installed slab mat layout without an on-grid placement (the column bars
            # moved). That is a step of the coordination, not the candidate's outcome: the layout is chosen again for
            # the new cages before the actions are computed (2026-11-06).
            from Design.SMRF_Joint_Mat_Physical import global_mat_placement
            if bar_layers.row_audit(state)['valid'] and global_mat_placement(state)['status']=='fail':
                chosen,evidence=reselect_layout(state,context['computed_slab_catalog'],installed)
                reselections.append(evidence)
                if chosen is None:
                    last=next((w['inventory']['reason'] for w in reversed(evidence['walks']) if w.get('inventory')),None)
                    raise gd.CandidateInfeasible('slab_reinforcement','The installed slab mat layout has no placement on the selected '
                        'cages and the bounded re-selection established none; no catalog infeasibility is inferred'
                        +(f' (last inventory refusal: {last})' if last else ''),
                        dict(outcome='incomplete_physical_layout_selection',state_identity=state.identity(),
                            coordination_passes=len(history),layout_reselection=evidence,selected_beam_cages=beam_cages(),
                            layout_reselections=reselections))
                slab['computed_catalog_candidate_id']=chosen['candidate_id']
        if (common['state_identity']!=state.identity()
            or common['inventory']['layout_sha256']!=_hash((sp.SLAB_REINFORCEMENT or {}).get('layout'))
            or float(common['definition']['slab_thickness_in'])!=float(sp.SLAB_THICKNESS_IN)):
            period=systems.model_period(state);period_solves+=1
            try:common=analyze(state,cfg,context,period)
            except PhysicalInventoryUnresolved as exc:
                refusal=exc;again=None
                now=(sp.SLAB_REINFORCEMENT or {}).get('layout')
                if str(exc)==STRAIGHT_INTERVALS_UNRESOLVED and now is not None:
                    # The layout still places on the new cages, but only clear of the column bars, and its bars meet
                    # hooked beam bars: one lane-clear walk, as the slab stage gives a seed, before the outcome.
                    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
                    if any(r.get('phase_rule')=='columns_only' for r in (global_mat_placement(state).get('layers') or {}).values()):
                        chosen,evidence=reselect_layout(state,context['computed_slab_catalog'],now,lane_clear_only=True)
                        reselections.append(evidence)
                        if chosen is not None:
                            slab['computed_catalog_candidate_id']=chosen['candidate_id']
                            try:again=analyze(state,cfg,context,period)
                            except PhysicalInventoryUnresolved as second:refusal=second
                if again is None:
                    # The proposed cages have no established installed inventory (mat phase, lanes or hooked straight
                    # runs): a recorded outcome of this candidate with its evidence, never an evaluation exception.
                    raise gd.CandidateInfeasible('reinforcement_selection',str(refusal),dict(outcome='physical_inventory_not_established',
                        state_identity=state.identity(),layout_sha256=_hash((sp.SLAB_REINFORCEMENT or {}).get('layout')),
                        coordination_passes=len(history),selected_beam_cages=beam_cages(),inventory_evidence=refusal.evidence,
                        layout_reselections=reselections)) from refusal
                common=again
            if common.get('status')!='responses_computed':
                raise gd.CandidateInfeasible('member_actions','Changed actual cage/mat action calculation remains unresolved',
                    dict(outcome='incomplete_common_actions',common_actions=common,layout_reselections=reselections))
        refresh_slab(state,slab,cfg,common)
    def current_capacity():
        ids=[c['id'] for c in common['combinations']]
        return capacities.build_group_capacity_design(common['actions'],ids,cfg,common['capacity_gravity_transfer'],state,
            compatibility=common['compatibility_actions'],slab_eligibility=slab_strength_eligibility(slab['reinforcement']),
            common_torsion=common.get('common_torsion'),common_actions=common)
    refresh_slab(state,slab,cfg,common)
    for margin in (1.,1.03,1.06):
        visited={};states=[];stable=False;resolved=False
        for index in range(maximum_passes):
            before=design_signature(state);key=(before,tuple(sorted(reserve)))
            if key in visited:
                stop='cycle';cycle=dict(scwb_margin=margin,pass_index=index+1,repeats_pass=visited[key]+1,candidates=[])
                # Retain the existing candidate order and complete selection/
                # fixed-hoop/capacity rules, with NEW actions for each actual
                # proposed design. A prior state's force field is never reused.
                for proposal in gd._cycle_candidates(states[visited[key]:])[:gd.CYCLE_RESOLUTION_MAX_CANDIDATES]:
                    update_actual(proposal)
                    verified=selection.verify_installed_selection(state,common['actions'],cfg,margin,reserve)
                    capacity=current_capacity();updates=capacities.hoop_updates(capacity,state)
                    cycle['candidates'].append(dict(state_identity=state.identity(),common_input_identity=common['input_signature'],
                        selection_verified=verified['verified'],hoops_fixed=not updates,capacity_accepted=capacity['accepted']))
                    if verified['verified'] and not updates and capacity['accepted']:
                        resolved=True;report=selection.verified_selection_report(report,verified,reserve)
                        cycle['resolution']=dict(adopted_state_identity=state.identity(),verification=verified)
                        break
                break
            visited[key]=index;states.append(state)
            proposal,report=selection.select_reinforcement(state,common['actions'],cfg,scwb_margin=margin,
                **({'reserve':reserve} if reserve else {}))
            reserve=list(report.get('reserved_cores') or reserve)
            reselected=len(reselections)
            update_actual(proposal)
            capacity=current_capacity();updates=capacities.hoop_updates(capacity,state)
            entry=dict(scwb_margin=margin,pass_index=index+1,input_design_signature=before,
                output_design_signature=design_signature(state),common_input_signature=common['input_signature'],
                selection_settled=bool(report['settled']),hoops_settled=not updates,
                exhausted_groups=report['exhausted_groups'],hoops_changed=sorted(updates),
                terminal_arrangements=[t.get('status') for t in report.get('terminal_arrangements') or []],
                layout_reselected=len(reselections)>reselected)
            history.append(entry)
            # A pass that had the slab layout chosen again is not a fixed point: its cages were selected against the
            # earlier mat and its actions, so another pass follows.
            if not updates and report['settled'] and not entry['layout_reselected']:
                stable=True;stop='settled';break
            if updates:
                proposal=state.with_designs(updates)
                proposal=selection.close_terminal_arrangement(proposal,common['actions'],cfg,margin,report,reserve,
                    inputs=dict(call='computed_common_after_hoops',input_state=proposal,scwb_margin=margin,reserve=sorted(reserve)))
                update_actual(proposal)
        if report is None:raise ValueError('No actual reinforcement coordination pass occurred')
        capacity=current_capacity();joint_checks=evaluate_joints(gd.joint_check_inputs(capacity,state))
        failed=[r for r in joint_checks if r['id']=='scwb' and r['status']=='fail']
        if not failed or report['exhausted_groups']:break
    settled=dict(state=state,capacity=capacity,joint_checks=joint_checks,selection=report,
        settled=stable,selection_settled=stable and bool(report['settled']),
        hoops_settled=not capacities.hoop_updates(capacity,state),evidence_rebuilt_on_final_design=True,
        scwb_margin=margin,log=history,stop=stop,cycle=cycle,scwb=gd.scwb_constraint(joint_checks,capacity),
        resolution='settled' if stable else gd.contract.VERIFIED_CYCLE_RESOLUTION if resolved else
                   'cycle_unresolved' if stop=='cycle' else 'pass_budget')
    settled['common_coordination']=dict(method=METHOD,history=history,stable=stable,layout_reselections=reselections,
        maximum_passes=maximum_passes,final_design_signature=design_signature(state),final_common_signature=common['input_signature'],
        additional_period_solves=period_solves)
    from Design.SMRF_Common_Action_Analysis import assess_installed
    common=assess_installed(state,common,progress=context.get('common_progress'))
    refresh_slab(state,slab,cfg,common)
    # Recompute the final capacity evidence with the final slab check record;
    # nominal/probable steel remains full-fy under its independent development.
    capacity=current_capacity();settled['capacity']=capacity
    settled['joint_checks']=evaluate_joints(gd.joint_check_inputs(capacity,state))
    settled['scwb']=gd.scwb_constraint(settled['joint_checks'],capacity)
    settled['hoops_settled']=not capacities.hoop_updates(capacity,state)
    # Final slab eligibility can change the capacity evidence. All returned
    # aggregate constraints must belong to this final capacity/joint record.
    settled['settled']=stable and settled['hoops_settled']
    return dict(state=state,slab=slab,period=period,common=common,settled=settled,layout_settled=(stable or resolved) and settled['hoops_settled'])


def mat_feedback(common,reinforcement):
    """Computed failures guide a new trial; they do not qualify another mat."""
    names=('x_top','x_bottom','y_top','y_bottom');ratios={n:1. for n in names};failures=[]
    for r in common.get('local_checks',{}).get('records',[]):
        for name,check in r['local_minimum']['checks'].items():
            if check['status']=='fail':
                failures.append(dict(kind='local_minimum',record_id=r['id'],mat=name,required_in2=check['required_in2'],
                    installed_in2=check['credit']['installed_area_in2']))
                area=check['credit']['installed_area_in2']
                ratios[name]=max(ratios[name],check['required_in2']/area if area>0 else 2.)
    for row in reinforcement.get('checks',[]):
        if row['status']=='fail':failures.append(dict(kind='computed_slab_check',check_id=row['id'],location=row.get('location')))
    for family in ('common_strength','local_strength'):
        if (common.get(family) or {}).get('status')=='fail':failures.append(dict(kind='concurrent_strength',family=family,payload_sha256=common[family]['payload_sha256']))
    return dict(known_failures=failures,mat_density_target_ratios=ratios,
        interpretation='Proposal priority only; new installed mat/cage/actions and all checks are required',
        unknown_converted_to_capacity_failure=False)


def ordered_mat_trials(catalog,current_layout,visited,feedback):
    """Prioritize the original catalog, retaining every untried alternative.

    Actual integer support counts give density targets only as a search hint.
    The complete catalog stays in the saved context. No reinforcement gene,
    surrogate or narrowed proposal domain is introduced.
    """
    current={n:r['area_in2_per_ft'] for n,r in current_layout['layers'].items()}
    targets={n:a*feedback['mat_density_target_ratios'][n] for n,a in current.items()}
    def priority(candidate):
        values={n:r['area_in2_per_ft'] for n,r in candidate['layout']['layers'].items()}
        target=all(values[n]>=targets[n]-1e-10 for n in targets)
        increases=all(values[n]>=current[n]-1e-10 for n in current) and any(values[n]>current[n]+1e-10 for n in current)
        return (0 if target and increases else 1 if increases else 2,candidate['nominal_steel_area_sum_in2_per_ft'],candidate['candidate_id'])
    return sorted((r for r in catalog['candidates'] if r['layout_sha256'] not in visited),key=priority)


def coordinate(state,cfg,context,slab,period,common,*,maximum_passes=10,maximum_mat_trials=4,maximum_geometry_trials=32):
    """Finite actual cage/mat fixed point with original catalog and fresh actions."""
    from Design import Perimeter_Systems as systems
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    from Design.SMRF_Physical_Floor_Materials import installed_inventory,PhysicalInventoryUnresolved
    from Design.Grouped_Design import CandidateInfeasible
    if any(type(v) is not int or v<1 for v in (maximum_mat_trials,maximum_geometry_trials)):
        raise ValueError('Explicit positive finite mat/geometry coordination limits required')
    catalog=context.get('computed_slab_catalog')
    if not catalog or catalog['payload_sha256']!=slab['original_catalog_sha256']:raise ValueError('Original complete slab proposal catalog required')
    visited=set();history=[];geometry=[];additional_periods=0;cage_periods=0;reason='declared_mat_trial_bound';result=None
    reselections=[]
    for trial in range(maximum_mat_trials):
        layout=copy.deepcopy(sp.SLAB_REINFORCEMENT['layout']);visited.add(_hash(layout))
        try:result=_coordinate_one_layout(state,cfg,context,slab,period,common,maximum_passes=maximum_passes)
        except CandidateInfeasible as exc:
            # The refusal of a later mat trial keeps the re-selections of the earlier ones.
            if reselections:exc.evidence['layout_reselections_of_earlier_mat_trials']=list(reselections)
            raise
        reselections.extend(dict(r,mat_trial=trial+1) for r in result['settled']['common_coordination'].get('layout_reselections') or [])
        cage_periods+=result['settled']['common_coordination']['additional_period_solves']
        state,slab,period,common=(result[k] for k in ('state','slab','period','common'))
        # The coordination may have chosen the layout again after a cage change: read the layout installed now.
        installed_layout=copy.deepcopy(sp.SLAB_REINFORCEMENT['layout']);visited.add(_hash(installed_layout))
        feedback=mat_feedback(common,slab['reinforcement'])
        history.append(dict(trial=trial+1,layout_sha256=_hash(layout),final_layout_sha256=_hash(installed_layout),state_identity=state.identity(),
            common_input_signature=common['input_signature'],common_payload_sha256=common['payload_sha256'],
            cage_settled=result['layout_settled'],feedback=feedback))
        if not result['layout_settled']:reason='cage_coordination_unresolved';break
        if not feedback['known_failures']:reason='no_computed_mat_failure_requires_new_trial';break
        if trial+1==maximum_mat_trials:break
        proposals=ordered_mat_trials(catalog,installed_layout,visited,feedback);chosen=None
        for candidate in proposals[:maximum_geometry_trials]:
            sp.SLAB_REINFORCEMENT=dict(layout=copy.deepcopy(candidate['layout']),accepted=False,stage='computed_strength_guided_catalog_trial')
            visited.add(candidate['layout_sha256']);audit=bar_layers.row_audit(state);bar_layers.raise_arrangement_errors(audit)
            placement=global_mat_placement(state) if audit['valid'] else None
            row=dict(candidate_id=candidate['candidate_id'],layout_sha256=candidate['layout_sha256'],
                row_audit_sha256=_hash(audit),placement_sha256=None if placement is None else _hash(placement),
                status='pass' if placement and placement['status']=='pass' else 'actual_geometry_unresolved_or_failed')
            geometry.append(row)
            if row['status']=='pass':chosen=candidate;break
        if chosen is None:
            sp.SLAB_REINFORCEMENT=slab['reinforcement'];reason='bounded_mat_geometry_trials_unresolved';break
        # The selected phase, full steel volume and stiffness are rebuilt.
        # No previous mat's action/strength payload is paired with this mat.
        try:inventory=installed_inventory(state)
        except PhysicalInventoryUnresolved as exc:
            from Design.Grouped_Design import CandidateInfeasible
            raise CandidateInfeasible('slab_reinforcement',str(exc),dict(outcome='physical_inventory_not_established',
                state_identity=state.identity(),layout_sha256=chosen['layout_sha256'],mat_trial_history=history,
                inventory_evidence=exc.evidence,layout_reselections=reselections)) from exc
        slab['material_inventory']=inventory
        slab['computed_catalog_candidate_id']=chosen['candidate_id']
        period=systems.model_period(state);additional_periods+=1
        common=analyze(state,cfg,context,period)
        if common.get('status')!='responses_computed':
            from Design.Grouped_Design import CandidateInfeasible
            raise CandidateInfeasible('member_actions','Strength-guided actual mat action calculation remains unresolved',
                dict(outcome='incomplete_common_actions',common_actions=common,mat_trial_history=history,
                    layout_reselections=reselections))
    result['settled']['common_coordination']['additional_period_solves']=cage_periods+additional_periods
    result['settled']['common_coordination']['mat_coordination']=dict(method=METHOD,history=history,geometry_trials=geometry,
        stop_reason=reason,maximum_mat_trials=maximum_mat_trials,maximum_geometry_trials=maximum_geometry_trials,
        original_catalog_sha256=catalog['payload_sha256'],original_catalog_count=catalog['enumerated_count'],
        untried_geometry_candidate_count=sum(r['layout_sha256'] not in visited for r in catalog['candidates']),
        final_layout_sha256=_hash(sp.SLAB_REINFORCEMENT['layout']),no_catalog_feasibility_claim=True,
        layout_reselections=reselections)
    return result
