"""Complete prescribed-primary demand guards on actual physical cut material.

Original G, physical G+ELF and compatibility states remain unchanged. P5
states are specified design forces at the slab plane, not a 3D displacement
superposition. Full-T moment projections add no external N and alter no
nominal/probable steel inventory. Every extra state is named and retained.
"""
from __future__ import annotations
import copy
import numpy as np
from Design.SMRF_Coupled_Floor_Design import _hash
from Design.SMRF_Physical_Floor_Cuts import common_strip_cut,bar_clear_interval
from Design.SMRF_Physical_Action_Readout import primary_beam_cut,installed_full_t_projector
from Design.SMRF_Primary_Moment_Guards import (physical_cut_states,primary_moment_guards,
    primary_membrane_guards,primary_membrane_source_from_readout)
from Design.SMRF_Primary_Diaphragm_Batch import PrimaryDiaphragmReadout

METHOD='actual_physical_cut_complete_primary_demand_guard_plan_v4_explicit_local_binding'
LOCAL_PLAN_CONTRACT='actual_local_pitch_point_action_plan_v1'


def primary_guard_increment(common,combination_id,member_id,station_in):
    """Measured identical-baseline-subtracted full-T moment at one actual cut."""
    gid=common['gravity_case_mapping'][combination_id];source=common['primary'][gid]
    definition=source['definition'];increment=source['increments'][combination_id]
    prior=common.get('saved_physical_input_binding') or {}
    diagnostic_reuse=(bool(prior.get('input_manifest_sha256')) and bool(prior.get('prior_input_signature')) and
        increment.get('source_identity')==prior.get('prior_source_identity'))
    if (definition['state_identity']!=common['state_identity']
        or definition['physical_definition_sha256']!=common['definition_sha256']
        or increment.get('physical_gravity_case_id')!=gid
        or (increment.get('source_identity')!=common['analysis_source_identity'] and not diagnostic_reuse)):
        raise ValueError('Primary guard increment actual state/gravity/source differs')
    cut=primary_beam_cut(definition,increment,member_id,station_in)
    if abs(cut['N_tension_kip'])>1e-7:
        raise ValueError('Retained rigid primary beam has nonzero external axial force; zero-N guard premise unresolved')
    return dict(source_operation='released_loaded_minus_identical_released_gravity',
        state_identity=common['state_identity'],gravity_load_id=gid,physical_combination_id=combination_id,
        member_id=str(member_id),axis=cut['axis'],station_in=float(station_in),n_tension_kip=0.,
        measured_primary_N_tension_kip=cut['N_tension_kip'],m_z_kip_in=cut['M_centroid_kipin'],
        m_s_kip_in=cut['Ms_transverse_first_moment_kipin'],reference_z_in=0.,reference_s_in=cut['transverse_reference_in'],
        primary_definition_sha256=definition['identity_sha256'],primary_increment_sha256=_hash(increment),
        gravity_transfer_sha256=_hash(source['gravity_transfer']),actual_primary_cut=cut)


class _PhysicalSource:
    def __init__(self,bundle,common,prepared_responses=None):
        self.bundle=bundle;self.response=bundle['response'];self.common=common
        digest=_hash(self.response) if prepared_responses is None else prepared_responses.digest(bundle)
        if (bundle['response_payload_sha256']!=digest
            or bundle['material_inventory_id']!=common['inventory']['identity_sha256']
            or bundle['analysis_input_identity']!=common['input_signature']
            or self.response['definition_sha256']!=common['definition_sha256']):
            raise ValueError('Physical guard source differs from actual material/input/response')
        self.cases={c['id']:c for c in self.response['cases']};self.floors={r['level']:r for r in self.response['floors']}
        if len(self.cases)!=len(self.response['cases']):raise ValueError('Duplicate actual physical source case')
        self.fields={};self.axial_windows={}
        for level,floor in self.floors.items():
            for response in floor['responses']:
                case=self.cases[response['case_id']];definition=common['definition']
                q={e['id']:-float(case['dead'])*definition['gamma_kip_in3']*e['section']['b_in']*(e['section']['h_in']-definition['slab_thickness_in'])
                   for e in response['elements'] if e['kind']=='web'}
                key=(level,case['id'])
                if key in self.fields:raise ValueError('Duplicate actual floor response')
                self.fields[key]=(response,q)

    def cut(self,spec,case_id,*,location=None,interval=None,member='from_spec'):
        response,q=self.fields[(spec['floor'],case_id)]
        row=common_strip_cut(response,axis=spec['axis'],station_in=spec['station_in'],
            interval_in=spec['interval_in'] if interval is None else interval,trace=spec['trace'],
            web_id=spec.get('member_id') if member=='from_spec' else member,uniform_web_loads=q,
            state_identity=self.common['state_identity'],combination_id=case_id,
            location_id=spec['cut_id'] if location is None else location,
            recovery_id=self.bundle['source_id']+'/'+self.bundle['response_payload_sha256'])
        row.update(material_inventory_id=self.bundle['material_inventory_id'],analysis_input_identity=self.bundle['analysis_input_identity'],
            source_response_sha256=self.bundle['response_payload_sha256'],definition_sha256=self.common['definition_sha256'])
        if 'axial_accounting_interval_in' in spec:
            from Design.SMRF_Axial_Action_Accounting import physical_window_evidence
            normalized,_=physical_cut_states([row],outer_interval_in=row['slab']['interval_in'])
            actual_spec={**spec,'interval_in':row['slab']['interval_in'],'member_id':None if row['web'] is None else row['web']['member_id']}
            key=(spec['floor'],case_id,spec['axis'],tuple(actual_spec['interval_in']),
                 tuple(spec['axial_accounting_interval_in']),str(actual_spec['member_id']),
                 _hash(spec.get('axial_window_selection')))
            if key not in self.axial_windows:
                self.axial_windows[key]=physical_window_evidence(response,actual_spec,normalized[0],uniform_web_loads=q)
            row['axial_action_window']=copy.deepcopy(self.axial_windows[key])
        return row


class PreparedCommonGuardSources:
    """One evaluation/readback's fully hashed fields and one-floor P5 reader."""
    def __init__(self,common):
        from Design.SMRF_Common_Action_Analysis import physical_response_bundles
        from Design.SMRF_Physical_Floor_Strength import PreparedPhysicalResponses
        self.common=common;self.binding=(common['input_signature'],common['state_identity'],common['inventory']['identity_sha256'],common['definition_sha256'])
        self.bundles=physical_response_bundles(common);self.physical=PreparedPhysicalResponses(self.bundles)
        self.sources={b['source_id']:_PhysicalSource(b,common,self.physical) for b in self.bundles}
        self.readout=PrimaryDiaphragmReadout(common['p5'])
        self.p5_hash=common['p5']['payload_sha256']

    def validate(self,common):
        if self.common is not common or self.binding!=(common['input_signature'],common['state_identity'],common['inventory']['identity_sha256'],common['definition_sha256']) or self.p5_hash!=common['p5']['payload_sha256']:
            raise ValueError('Prepared common source cannot cross actual input/material/P5 evaluations')
        return self


def build_common_guard_plan(common,physical_cut_plan,p5_batch,*,state,maximum_cuts,allow_partial_plan=False,prepared_sources=None,local_pitch_actions=False):
    """Append every primary moment/P5/simultaneous section demand state.

    The input plan is the complete deterministic physical cut plan. Extra
    effective-flange guard partitions enlarge only their material boundaries
    enough to contain whole bars. Their projection and signed target retain
    the original code-effective width. No inverse or structural solve occurs.
    """
    from Design.SMRF_Common_Action_Analysis import physical_response_bundles
    if not isinstance(maximum_cuts,int) or maximum_cuts<1:raise ValueError('Explicit positive complete guard-cut bound required')
    plan=physical_cut_plan
    from Design.SMRF_Local_Primary_Guards import primary_guard_recipe,bind_local_primary_guard_action
    from Design.SMRF_Local_Pitch_Strength import bind_local_pitch_action
    if not isinstance(local_pitch_actions,bool):raise ValueError('Explicit boolean local action selection required')
    if local_pitch_actions != (plan.get('action_contract')==LOCAL_PLAN_CONTRACT):
        raise ValueError('Local pitch action option and declared physical plan contract must agree')
    def local_context(spec,source):
        return dict(floor=spec['floor'],inventory=common['inventory'],analysis_input_identity=common['input_signature'],
            source_response_sha256=source.bundle['response_payload_sha256'],definition_sha256=common['definition_sha256'])
    if (plan.get('payload_sha256')!=_hash({k:v for k,v in plan.items() if k!='payload_sha256'})
        or plan['state_identity']!=common['state_identity'] or state.identity()!=common['state_identity']
        or plan['definition_sha256']!=common['definition_sha256'] or plan['material_inventory_id']!=common['inventory']['identity_sha256']):
        raise ValueError('Complete unchanged actual physical cut plan required')
    if set(plan['required_cut_ids'])!=set(plan['cuts']) or [s['cut_id'] for s in plan['cut_specs']]!=plan['required_cut_ids']:
        raise ValueError('Complete original physical cut inventory differs')
    if prepared_sources is not None and not isinstance(prepared_sources,PreparedCommonGuardSources):raise ValueError('Invocation-local common source context required')
    context=None if prepared_sources is None else prepared_sources.validate(common)
    readout=PrimaryDiaphragmReadout(p5_batch) if context is None else context.readout
    if (p5_batch['common_input_signature']!=common['input_signature'] or p5_batch['source_identity']!=common['analysis_source_identity']
        or p5_batch['definition_sha256']!=common['definition_sha256'] or p5_batch['state_identity']!=common['state_identity']
        or p5_batch['combination_inventory_sha256']!=_hash(common['combinations'])):
        raise ValueError('P5 and physical guard actual input/source/combination bindings differ')
    bundles=physical_response_bundles(common) if context is None else context.bundles
    sources={b['source_id']:_PhysicalSource(b,common) for b in bundles} if context is None else context.sources
    if len(sources)!=len(bundles):raise ValueError('Unique actual physical source roster required')
    gravity=sources['actual_frozen_gravity'];descriptions={f['level']:f for f in common['definition']['floors']}
    seismic={c['id']:c for c in common['combinations'] if c['ex'] or c['ey']}
    specs=copy.deepcopy(plan['cut_specs']);cuts=copy.deepcopy(plan['cuts']);guards=[];seen=set();covered=set()
    if len(specs)>maximum_cuts:raise ValueError('Original required cut inventory exceeds complete guard budget')
    def append(branch,templates,origin,recipe=None):
        if len(branch['states'])!=len(templates):raise ValueError('Complete guard partition and template inventory required')
        for row,template in zip(branch['states'],templates):
            if local_pitch_actions:
                original=row
                row=bind_local_primary_guard_action(recipe,branch['branch_id'],row['identity']['location_id'],
                    **local_context(template,sources[template['source_id']]))
                if row['resultants']!=original['resultants']:raise ValueError('Local guard binding changed original signed demand')
            name=template['cut_id']+'|guard:'+branch['branch_id']
            if name in cuts:raise ValueError('Duplicate actual primary guard cut identity')
            spec={**template,'cut_id':name,'source_physical_cut_id':origin,'guard_branch_id':branch['branch_id'],
                  'sampling_kind':'additional_required_primary_strength_guard','physical_region_kind':template.get('physical_region_kind','actual_effective_flange_guard')}
            specs.append(spec);cuts[name]=row
            if len(specs)>maximum_cuts:raise ValueError('Complete required primary guards exceed declared cut budget; no truncation allowed')
    def p5_sources(raws,templates,floor,cid,branch):
        output=[]
        for raw,template in zip(raws,templates):
            row=readout.cut(floor,cid,branch,axis=raw['slab']['axis'],station_in=raw['slab']['station_in'],
                interval_in=raw['slab']['interval_in'],trace=raw['slab']['trace'],transverse_reference_in=raw['transverse_reference_in'])
            verified=primary_membrane_source_from_readout(row,identity=raw['identity'])
            from Design.SMRF_Axial_Action_Accounting import primary_membrane_axial_action
            verified['axial_action']=primary_membrane_axial_action(readout,template,raw,verified)
            output.append(verified)
        return output
    for spec in plan['cut_specs']:
        cid=plan['cuts'][spec['cut_id']]['identity']['combination_id']
        if cid not in seismic:continue
        covered.add(cid);source=sources[spec['source_id']];gid=common['gravity_case_mapping'][cid]
        raw=source.cut(spec,cid);baseline=gravity.cut(spec,gid)
        normalized=[bind_local_pitch_action(raw,**local_context(spec,source))] if local_pitch_actions else physical_cut_states([raw],outer_interval_in=spec['interval_in'])[0]
        if _hash(normalized[0])!=_hash(plan['cuts'][spec['cut_id']]):raise ValueError('Physical guard source does not reproduce the original required cut')
        witness=readout.floor(spec['floor'])['witnesses'][cid]
        branches=[c['id'] for c in witness['cases']]
        required={'diaphragm_structural_analysis','diaphragm_fpx','collector_elf_omega','collector_eq_omega',
                  'collector_minimum','structural_original_direct','collector_original_direct_omega'}
        if len(set(branches))!=len(branches) or not required<=set(branches):raise ValueError('Every original signed conventional/direct P5 branch required')
        for branch in branches:
            name=f'P5/{branch}'
            arguments=dict(physical_cuts=[raw],gravity_cuts=[baseline],primary_membrane_sources=p5_sources([raw],[spec],spec['floor'],cid,branch),
                outer_interval_in=spec['interval_in'],guard_identity=name)
            result=primary_membrane_guards(**arguments)
            recipe=primary_guard_recipe('membrane',**arguments) if local_pitch_actions else None
            append(result['branches'][1],[spec],spec['cut_id'],recipe)
        member=spec.get('member_id')
        if member is None:continue
        key=(spec['source_id'],cid,spec['floor'],str(member),float(spec['station_in']),spec['trace'])
        if key in seen:continue
        seen.add(key);projector=installed_full_t_projector(state.member(member));desc=descriptions[spec['floor']]
        if projector.axis!=spec['axis']:raise ValueError('Actual primary projector axis differs from physical beam')
        cross='y' if spec['axis']=='x' else 'x';fp=desc['footprint']
        bars=common['inventory']['floors'][str(spec['floor'])]['bars']
        families=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in bars if b['axis']==spec['axis']]
        outer=bar_clear_interval(projector.flange,[fp[cross+'_min_in'],fp[cross+'_max_in']],families)
        a,b=spec['interval_in']
        # A torsion-section material check can be wider than the flexural
        # code flange. Keep its entire physical core, and project zero
        # incremental primary action outside the original effective flange.
        outer=[min(outer[0],a),max(outer[1],b)]
        intervals=[]
        if a>outer[0]+1e-8:intervals.append(('left',[outer[0],a],None))
        intervals.append(('core',[a,b],member))
        if outer[1]>b+1e-8:intervals.append(('right',[b,outer[1]],None))
        templates=[];physical=[];gcuts=[]
        for label,interval,owner in intervals:
            template={**spec,'cut_id':spec['cut_id']+'|effective:'+label,'interval_in':interval,'member_id':owner}
            if local_pitch_actions:template['material_scope']='slab_only' if owner is None else 'common_slab_web'
            templates.append(template);physical.append(source.cut(template,cid));gcuts.append(gravity.cut(template,gid))
        pe=source.cut(spec,cid,location=spec['cut_id']+'|exact_effective',interval=projector.flange)
        ge=gravity.cut(spec,gid,location=spec['cut_id']+'|exact_effective',interval=projector.flange)
        increment=primary_guard_increment(common,cid,member,spec['station_in'])
        arguments=dict(physical_cuts=physical,gravity_effective_cut=ge,projector=projector,primary_increment=increment,
            guard_identity='primary_M',components=('strong','weak','both'),physical_effective_cut=pe,outer_interval_in=outer)
        moments=primary_moment_guards(**arguments)
        recipe=primary_guard_recipe('moment',**arguments) if local_pitch_actions else None
        # The additional wider physical sections are retained alongside their
        # demand guards; no state can disappear merely by creating a guard.
        append(moments['branches'][0],templates,spec['cut_id'],recipe)
        for branch in moments['branches'][1:]:append(branch,templates,spec['cut_id'],recipe)
        for branch in branches:
            arguments=dict(physical_cuts=physical,gravity_cuts=gcuts,primary_membrane_sources=p5_sources(physical,templates,spec['floor'],cid,branch),outer_interval_in=outer,
                guard_identity='P5_effective/'+branch,projector=projector,primary_increment=increment,
                physical_effective_cut=pe,gravity_effective_cut=ge)
            result=primary_membrane_guards(**arguments)
            recipe=primary_guard_recipe('membrane',**arguments) if local_pitch_actions else None
            for extra in result['branches'][1:]:append(extra,templates,spec['cut_id'],recipe)
        guards.append(dict(source_physical_cut_id=spec['cut_id'],combination_id=cid,floor=spec['floor'],member_id=str(member),
            station_in=spec['station_in'],trace=spec['trace'],effective_flange_interval_in=list(projector.flange),
            enclosing_bar_clear_interval_in=outer,primary_increment=increment,complete_P5_branch_ids=branches,
            partition_specs=templates,primary_moment_payload_sha256=moments['payload_sha256']))
    expected=set(seismic)
    if allow_partial_plan:
        scope=plan.get('scope')
        if not scope or scope.get('full_universe_sha256')!=_hash(scope['full_floor_case_universe']):
            raise ValueError('An explicitly scoped partial plan needs its full source/floor/case universe')
        expected={plan['cuts'][s['cut_id']]['identity']['combination_id'] for s in plan['cut_specs']} & set(seismic)
    elif plan.get('scope',{}).get('complete_required_universe') is False:
        raise ValueError('A partial physical plan cannot claim full primary guard coverage')
    if covered!=expected:raise ValueError('Every original signed seismic case in the declared scope needs its actual physical guard source')
    result={**copy.deepcopy(plan),'method':METHOD,'physical_plan_payload_sha256':plan['payload_sha256'],
        'common_input_signature':common['input_signature'],'p5_batch_payload_sha256':p5_batch['payload_sha256'],
        'cut_specs':specs,'cuts':cuts,'required_cut_ids':[s['cut_id'] for s in specs],'required_cut_count':len(specs),
        'original_physical_cut_ids':list(plan['required_cut_ids']),'primary_guard_inventory':guards,
        'original_signed_seismic_case_ids':list(seismic),'all_primary_demand_guards_produced':not allow_partial_plan,
        'scoped_signed_seismic_case_ids':sorted(expected),'scoped_primary_demand_guards_produced':True,
        'new_structural_or_inverse_solves':0,'physical_displacement_superposition_claimed':False,
        'nominal_probable_capacity_inventory_modified':False,'mesh_qualification_inferred':False,'complete_design_qualified':False}
    result.pop('payload_sha256');result['payload_sha256']=_hash(result);return result
