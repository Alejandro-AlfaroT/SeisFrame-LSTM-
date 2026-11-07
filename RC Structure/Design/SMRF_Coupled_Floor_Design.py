"""Actual grouped geometry adapter for coupled physical floor actions.

This is a bounded mechanical-analysis API. It accepts explicit stiffness
assumptions and source identity; it does not activate an evidence profile or
replace the retained ELF/system assignment by assertion. All physical floor
interface DOFs couple to the actual columns, including vertical shortening
and joint rotation. No C0 or frame beam moment is added to these actions.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
import numpy as np
from scipy import sparse
from Design.SMRF_Floor_Substructure import FloorAssembly,GeneralizedFloorStiffness,rectangle_torsion_constant
from Design.SMRF_Beam_Shell_Connection import connect_independent_beam_rotations
from Design.SMRF_Static_Substructure import condense
from Design.SMRF_Substructure_Frame import SubstructureFrame
from Design.SMRF_Paired_Floor_Operator import paired_floor_model
from Design.SMRF_Paired_Coupled_Equilibrium import PairedCoupledEquilibrium
from Design.SMRF_Paired_Arithmetic import add_pair
from Model.Floor_Footprint import from_perimeter_beams, column_overlap

METHOD='grouped_physical_floor_column_coupling_v4_paired_full_equilibrium'


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_recovered_fields(current,saved):
    """Compare rebuilt actual local fields and identities, excluding run timing."""
    def compare(a,b,label):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Saved physical field keys differ: '+label)
            for key in a:compare(a[key],b[key],label+'/'+str(key))
        elif isinstance(a,(list,tuple,np.ndarray)):
            if not isinstance(b,(list,tuple,np.ndarray)) or len(a)!=len(b):raise ValueError('Saved physical field dimensions differ: '+label)
            try:x=np.asarray(a,float);y=np.asarray(b,float)
            except (ValueError,TypeError):
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+'/'+str(i))
            else:
                if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(x,y,rtol=1e-10,atol=1e-8):
                    raise ValueError('Saved physical field differs from reconstructed operator: '+label)
        elif type(a) in (int,float) and type(b) in (int,float):
            if not np.isfinite([a,b]).all() or abs(a-b)>1e-8+1e-10*max(abs(a),abs(b)):
                raise ValueError('Saved physical scalar differs: '+label)
        elif a!=b:raise ValueError('Saved physical metadata differs: '+label)
    for key in ('definition_sha256','source_identity','cases'):compare(current[key],saved[key],key)
    if len(current['floors'])!=len(saved['floors']):raise ValueError('Saved physical floor inventory differs')
    for a,b in zip(current['floors'],saved['floors']):
        for key in ('id','level','grid','stiffness','responses'):compare(a[key],b[key],str(a['level'])+'/'+key)
    return dict(floors_verified=len(current['floors']),cases_verified=len(current['cases']),frame_solves=0)


def validate_definition(definition):
    """Bind identity to complete physical topology, not a trusted hash label."""
    if not isinstance(definition,dict) or definition.get('identity_sha256')!=_hash(
            {k:v for k,v in definition.items() if k!='identity_sha256'}):
        raise ValueError('Frozen physical floor definition identity does not match its contents')
    g=definition['geometry'];nx,ny,nf=(g[k] for k in ('nx','ny','nf'))
    if any(type(v) is not int or v<1 for v in (nx,ny,nf)):
        raise ValueError('Complete positive integer physical building dimensions required')
    lx,ly,sh=(float(g[k]) for k in ('lx','ly','story_h'))
    if any(not math.isfinite(v) or v<=0 for v in (lx,ly,sh)):
        raise ValueError('Finite positive physical spans/story heights required')
    nodes=definition['nodes'];byid={str(n['id']):n for n in nodes}
    positions={tuple(n['xyz']):n['id'] for n in nodes}
    expected={(i*lx,j*ly,k*sh) for k in range(nf+1) for j in range(ny+1) for i in range(nx+1)}
    if len(byid)!=len(nodes) or len(positions)!=len(nodes) or set(positions)!=expected:
        raise ValueError('Physical node inventory does not exactly cover the declared building')
    floors=definition['floors']
    if sorted(f['level'] for f in floors)!=list(range(1,nf+1)) or len({str(f['id']) for f in floors})!=nf:
        raise ValueError('Every actual elevated floor must occur exactly once')
    for floor in floors:
        level=floor['level']
        if floor['z_in']!=level*sh:raise ValueError('Floor elevation differs from physical node geometry')
        supports=floor['supports'];found={(s['x'],s['y']) for s in supports}
        if len(supports)!=(nx+1)*(ny+1) or found!={(i*lx,j*ly) for j in range(ny+1) for i in range(nx+1)}:
            raise ValueError('Floor support inventory is incomplete or duplicated')
        for s in supports:
            if s['id']!=positions[(s['x'],s['y'],level*sh)]:raise ValueError('Support identity does not name its physical frame node')
        actual={(b['axis'],b['line'],b['start'],b['end']) for b in floor['beams']}
        expected_beams={('x',j*ly,i*lx,(i+1)*lx) for j in range(ny+1) for i in range(nx)}
        expected_beams|={('y',i*lx,j*ly,(j+1)*ly) for i in range(nx+1) for j in range(ny)}
        if len(actual)!=len(floor['beams']) or actual!=expected_beams:
            raise ValueError('Floor beam inventory omits, duplicates or moves a physical span')
        footprint=from_perimeter_beams(g,floor['beams'])
        if 'footprint' in floor and floor['footprint']!=footprint:
            raise ValueError('Frozen slab footprint differs from actual perimeter beam outside faces')
    columns=definition['columns'];actual=set()
    for c in columns:
        key=(c['story'],c['grid_i'],c['grid_j']);k,i,j=key
        if not(1<=k<=nf and 0<=i<=nx and 0<=j<=ny):raise ValueError('Column lies outside declared topology')
        if c['i']!=positions[(i*lx,j*ly,(k-1)*sh)] or c['j']!=positions[(i*lx,j*ly,k*sh)]:
            raise ValueError('Column end node identities do not match its physical story/grid')
        actual.add(key)
    if len(actual)!=len(columns) or len(actual)!=nf*(nx+1)*(ny+1) or len({str(c['id']) for c in columns})!=len(columns):
        raise ValueError('Every physical column segment must occur exactly once')
    return True


def installed_definition(state=None):
    """Freeze every actual span, column and floor; no nominal representative."""
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Model.Build_Model import node_tag
    from Model.Member_Properties import joint_core_column
    state=state or mg.active()
    if state is None or not mg.is_grouped() or state.identity()!=mg.active().identity():
        raise ValueError('The exact grouped design must be installed before freezing floor geometry')
    h=float(sp.SLAB_THICKNESS_IN or 0.)
    if h<=0:raise ValueError('An actual selected slab thickness is required')
    nx,ny,nf=sp.NUM_BAY_X,sp.NUM_BAY_Y,sp.NUM_FLOOR
    lx,ly,sh=sp.BAY_X,sp.BAY_Y,sp.STORY_H
    columns=[];floors=[]
    for member in state.members('column'):
        d=member.design
        columns.append({'id':str(member.member_tag),'i':member.node_i,'j':member.node_j,
                        'story':member.story_or_floor,'grid_i':member.grid_i,'grid_j':member.grid_j,
                        'b_in':d.b_in,'h_in':d.h_in,'fc_ksi':d.fc_ksi,'E_ksi':sp.concrete_ec_ksi(d.fc_ksi),
                        'roof_extension_weight_kip':sp.roof_column_extension_weight_kip(d.b_in,d.h_in)
                            if member.story_or_floor==nf else 0.,
                        'group_id':member.group_id,'system':state.system_of(member.group_id)})
    for level in range(1,nf+1):
        supports=[];beams=[]
        for j in range(ny+1):
            for i in range(nx+1):
                d=joint_core_column(level,i,j).design
                supports.append({'id':node_tag(level,i,j),'x':i*lx,'y':j*ly,'b_in':d.b_in,'h_in':d.h_in})
        for member in state.members():
            if member.is_column or member.story_or_floor!=level:continue
            d=member.design;axis='x' if member.member_type=='beam_x' else 'y'
            start=(member.grid_i*lx if axis=='x' else member.grid_j*ly)
            beams.append({'id':str(member.member_tag),'axis':axis,
                          'line':member.grid_j*ly if axis=='x' else member.grid_i*lx,
                          'start':start,'end':start+(lx if axis=='x' else ly),
                          'b_in':d.b_in,'h_in':d.h_in,'E_ksi':sp.concrete_ec_ksi(d.fc_ksi),
                          'fc_ksi':d.fc_ksi,'group_id':member.group_id,'system':state.system_of(member.group_id)})
        grades={b['fc_ksi'] for b in beams}
        if len(grades)!=1:raise ValueError('A physical floor needs the declared common slab/beam concrete grade')
        fc=next(iter(grades))
        floors.append({'id':f'floor_{level}','level':level,'z_in':level*sh,'supports':supports,'beams':beams,
                       'fc_ksi':fc,'E_ksi':sp.concrete_ec_ksi(fc),
                       'footprint':from_perimeter_beams({'nx':nx,'ny':ny,'lx':lx,'ly':ly},beams)})
    nodes=[{'id':node_tag(k,i,j),'xyz':[i*lx,j*ly,k*sh]}
           for k in range(nf+1) for j in range(ny+1) for i in range(nx+1)]
    definition={'method':METHOD,'state_identity':state.identity(),'policy_id':state.policy_id,
                'geometry':{'nx':nx,'ny':ny,'nf':nf,'lx':lx,'ly':ly,'story_h':sh},
                'slab_thickness_in':h,'gamma_kip_in3':sp.CONCRETE_UNIT_WEIGHT_KCF/1728.,
                'floor_dead_load_ksf':sp.floor_dead_load_ksf(),'floor_live_load_ksf':sp.FLOOR_LIVE_LOAD_KSF,
                'nodes':nodes,'floors':floors,'columns':columns,
                'slab_plan':'actual perimeter beam outside faces; beam drops exclude joint footprints'}
    definition['identity_sha256']=_hash(definition)
    return definition


def physical_grid(floor,geometry,*,subdivisions_per_bay,include_beam_faces=True,punching_offset_in=None):
    """Nested explicit grids: uniform subdivisions plus exact physical faces.

    No face is snapped to a nearby FE point. Optional punching offset adds
    exact critical perimeter coordinates; the caller still integrates cuts.
    """
    n=subdivisions_per_bay
    if type(n) is not int or n<2:raise ValueError('At least two declared subdivisions per bay required')
    nx,ny,lx,ly=(geometry[k] for k in ('nx','ny','lx','ly'))
    fp=from_perimeter_beams(geometry,floor['beams'])
    x={float(i*lx/n) for i in range(nx*n+1)};y={float(j*ly/n) for j in range(ny*n+1)}
    x.update((fp['x_min_in'],fp['x_max_in']));y.update((fp['y_min_in'],fp['y_max_in']))
    for support in floor['supports']:
        for values,key,dim,low,high in ((x,'x','h_in',fp['x_min_in'],fp['x_max_in']),
                                       (y,'y','b_in',fp['y_min_in'],fp['y_max_in'])):
            values.add(float(support[key]))
            for sign in (-1,1):
                face=support[key]+sign*support[dim]/2
                if low<face<high:values.add(float(face))
                if punching_offset_in is not None:
                    p=face+sign*float(punching_offset_in)
                    if low<p<high:values.add(float(p))
    if include_beam_faces:
        for beam in floor['beams']:
            values,low,high=(y,fp['y_min_in'],fp['y_max_in']) if beam['axis']=='x' else (x,fp['x_min_in'],fp['x_max_in'])
            for sign in (-1,1):
                v=beam['line']+sign*beam['b_in']/2
                if low<v<high:values.add(float(v))
    # Exact decimal construction can produce sub-picometer floating duplicate
    # coordinates. Merge only <=1e-8 inch numerical duplicates, never faces.
    def clean(values):
        result=[]
        for value in sorted(values):
            if not result or value-result[-1]>1e-8:result.append(value)
        return result
    return clean(x),clean(y)


def _pressure(floor,geometry,definition,case,patterns):
    selected=case.get('live_pattern','all')
    if isinstance(selected,str) and selected not in ('all','none'):
        if selected not in patterns:raise ValueError(f'Unknown actual live-load pattern {selected}')
        selected=patterns[selected]
    panels=None if selected=='all' else set() if selected=='none' else {tuple(p) for p in selected}
    values=[]
    for element in floor.elements:
        if element['kind']!='slab':continue
        center=element['points'][:,:2].mean(axis=0)
        i=max(0,min(geometry['nx']-1,int(center[0]/geometry['lx'])))
        j=max(0,min(geometry['ny']-1,int(center[1]/geometry['ly'])))
        live=1. if panels is None or (i,j) in panels else 0.
        values.append(-(float(case['dead'])*definition['floor_dead_load_ksf']+
                        float(case['live'])*definition['floor_live_load_ksf']*live)/144.)
    return values


class CoupledFloorSystem:
    """One frozen geometry/stiffness, multiple explicit static load cases.

    Construction requires bounded mesh and source identity. Numerical success
    is recorded separately from physical model/strength qualification.
    """
    def __init__(self,definition,cases,*,source_identity,stiffness_basis,
                 subdivisions_per_bay,max_shells_per_floor,max_coupled_dofs,
                 beam_bending_factor,slab_bending_factor,web_torsion_factor,
                 column_bending_factor,poisson=.2,patterns=(),punching_offset_in=None,
                 initial_operator_factory=None):
        validate_definition(definition)
        if not source_identity or not stiffness_basis:raise ValueError('Source identity and explicit stiffness basis required')
        self.definition=definition;self.cases=[dict(c) for c in cases]
        if not self.cases or len({c['id'] for c in self.cases})!=len(self.cases):raise ValueError('Unique nonempty case inventory required')
        if any(float(c.get('ex',0.)) or float(c.get('ey',0.)) for c in self.cases):
            raise ValueError('This gravity API cannot silently substitute for the retained ELF/system model')
        if initial_operator_factory is None:
            factors=[beam_bending_factor,slab_bending_factor,web_torsion_factor,column_bending_factor]
        else:
            if not callable(initial_operator_factory) or len(self.cases)!=1:
                raise ValueError('Actual initial material operator requires a callable and one simultaneous case')
            if any(v is not None for v in (beam_bending_factor,slab_bending_factor,web_torsion_factor)):
                raise ValueError('Explicit material factory cannot also claim independent prescribed floor factors')
            factors=[column_bending_factor]
        if any(not math.isfinite(float(v)) or not 0<float(v)<=1 for v in factors):raise ValueError('Explicit stiffness factors must be in(0,1]')
        self.source_identity=source_identity;self.floors=[];interfaces=[];geometry=definition['geometry']
        patterns={p['id']:p['panels'] for p in patterns}
        footprints={f['level']:from_perimeter_beams(geometry,f['beams']) for f in definition['floors']}
        # A clipped slab/column overlap has a first moment about the column
        # center. Preserve that gravity wrench instead of silently moving the
        # residual material back onto its centerline.
        eccentric={level:{} for level in footprints}
        for col in definition['columns']:
            x,y=col['grid_i']*geometry['lx'],col['grid_j']*geometry['ly']
            overlap=column_overlap(footprints[col['story']],x,y,col['h_in'],col['b_in'])
            removed_weight=overlap['area_in2']*definition['slab_thickness_in']*definition['gamma_kip_in3']
            cx,cy=overlap['centroid_in']
            roof_weight=float(col.get('roof_extension_weight_kip',0.))
            if not math.isfinite(roof_weight) or roof_weight<0 or (roof_weight and col['story']!=geometry['nf']):
                raise ValueError('Physical roof extension weight must occur only at its actual roof joint')
            eccentric[col['story']][col['j']]=[0.,0.,-roof_weight,removed_weight*(cy-y),-removed_weight*(cx-x),0.]
        self._dead_joint_wrenches=eccentric
        for desc in definition['floors']:
            x,y=physical_grid(desc,geometry,subdivisions_per_bay=subdivisions_per_bay,punching_offset_in=punching_offset_in)
            floor=FloorAssembly(x,y,floor_id=desc['id'],thickness_in=definition['slab_thickness_in'],
                elastic_modulus_ksi=desc['E_ksi'],supports=desc['supports'],beams=desc['beams'],poisson=poisson,
                max_shells=max_shells_per_floor,concrete_weight_kip_in3=definition['gamma_kip_in3'])
            connect_independent_beam_rotations(floor)
            if initial_operator_factory is None:
                model=GeneralizedFloorStiffness(floor,beam_bending_factor=beam_bending_factor,
                    slab_bending_factors=slab_bending_factor,web_torsion_factor=web_torsion_factor,
                    stiffness_basis=stiffness_basis,input_identity=definition['identity_sha256'],
                    maximum_coupled_dofs=max_coupled_dofs)
            else:
                model=initial_operator_factory(floor,desc,definition,self.cases[0])
                if getattr(model,'floor',None) is not floor or model.stiffness.shape!=(len(floor.dof_ids),len(floor.dof_ids)):
                    raise ValueError('Initial constitutive operator belongs to different physical geometry')
            model=paired_floor_model(model,input_identity={'source':source_identity,'definition':definition['identity_sha256'],'floor':desc['id']})
            loads=[floor.loads(_pressure(floor,geometry,definition,c,patterns),beam_weight_factor=c['dead'],
                               joint_wrenches={tag:[v*float(c['dead']) for v in wrench]
                                               for tag,wrench in eccentric[desc['level']].items()}) for c in self.cases]
            rhs=np.column_stack([r['loads'] for r in loads])
            condensed=condense(model.stiffness,floor.dof_ids,floor.boundary_dofs,rhs,
                input_identity={'definition':definition['identity_sha256'],'source':source_identity,'floor':desc['id'],
                                'stiffness':model.metadata,'cases':self.cases})
            interfaces.append({'id':desc['id'],'dof_ids':floor.boundary_dofs,'stiffness':condensed.stiffness,
                               'loads':condensed.loads,'input_identity':condensed.record()['identity_sha256']})
            self.floors.append({'assembly':floor,'model':model,'condensed':condensed,'loads':loads,'description':desc})
        columns=[]
        for row in definition['columns']:
            b,h,e=row['b_in'],row['h_in'],row['E_ksi'];length=geometry['story_h']
            overlap=column_overlap(footprints[row['story']],row['grid_i']*geometry['lx'],
                                   row['grid_j']*geometry['ly'],h,b)
            volume=b*h*length-overlap['area_in2']*definition['slab_thickness_in']
            columns.append({'id':row['id'],'i':row['i'],'j':row['j'],'A':b*h,'E':e,'G':e/(2*(1+poisson)),
                'J':rectangle_torsion_constant(b,h)*column_bending_factor,
                'Iy':b*h**3/12*column_bending_factor,'Iz':h*b**3/12*column_bending_factor,'vecxz':[1.,0.,0.],
                'axial_load_kip_per_in':[-float(c['dead'])*volume*definition['gamma_kip_in3']/length for c in self.cases]})
        supports=[(n['id'],d) for n in definition['nodes'] if n['xyz'][2]==0 for d in range(1,7)]
        self._interfaces=interfaces;self._columns=columns
        self.frame=SubstructureFrame(definition['nodes'],interfaces,columns,
            input_identity={'method':METHOD,'definition':definition['identity_sha256'],'source':source_identity},supports=supports)

    def with_cases(self,cases,*,patterns=(),case_identity):
        """Reuse exact frozen operators with fresh coherent gravity load cases.

        No element query or stiffness mutation occurs. Every pressure, physical
        beam weight, column weight and eccentric joint wrench is rebuilt; a new
        condensed load vector and global frame bind the complete case identity.
        """
        cases=[dict(c) for c in cases]
        if not case_identity or not cases or len({c['id'] for c in cases})!=len(cases):
            raise ValueError('Fresh unique nonempty gravity cases and identity required')
        if any(float(c.get('ex',0.)) or float(c.get('ey',0.)) for c in cases):
            raise ValueError('Gravity rebinding cannot replace the retained ELF model')
        clone=object.__new__(CoupledFloorSystem)
        clone.definition=self.definition;clone.source_identity=self.source_identity;clone.cases=cases
        clone._dead_joint_wrenches=self._dead_joint_wrenches
        geometry=self.definition['geometry'];patterns={p['id']:p['panels'] for p in patterns}
        clone.floors=[];interfaces=[]
        for row in self.floors:
            floor=row['assembly'];desc=row['description'];model=row['model']
            loads=[floor.loads(_pressure(floor,geometry,self.definition,c,patterns),beam_weight_factor=c['dead'],
                joint_wrenches={tag:[v*float(c['dead']) for v in wrench] for tag,wrench in
                    self._dead_joint_wrenches[desc['level']].items()}) for c in cases]
            rhs=np.column_stack([v['loads'] for v in loads])
            sub=condense(model.stiffness,floor.dof_ids,floor.boundary_dofs,rhs,
                input_identity={'definition':self.definition['identity_sha256'],'source':self.source_identity,
                    'floor':desc['id'],'stiffness':model.metadata,'cases':cases,'case_identity':case_identity})
            interfaces.append(dict(id=desc['id'],dof_ids=floor.boundary_dofs,stiffness=sub.stiffness,
                                   loads=sub.loads,input_identity=sub.record()['identity_sha256']))
            clone.floors.append({**row,'condensed':sub,'loads':loads})
        columns=[]
        originals={str(c['id']):c for c in self.definition['columns']}
        footprints={r['level']:from_perimeter_beams(geometry,r['beams']) for r in self.definition['floors']}
        for c in self._columns:
            row=originals[str(c['id'])];b,h=row['b_in'],row['h_in'];length=geometry['story_h']
            overlap=column_overlap(footprints[row['story']],row['grid_i']*geometry['lx'],row['grid_j']*geometry['ly'],h,b)
            weight=(b*h*length-overlap['area_in2']*self.definition['slab_thickness_in'])*self.definition['gamma_kip_in3']/length
            columns.append({**c,'axial_load_kip_per_in':[-float(v['dead'])*weight for v in cases]})
        clone._interfaces=interfaces;clone._columns=columns
        supports=[(n['id'],d) for n in self.definition['nodes'] if n['xyz'][2]==0 for d in range(1,7)]
        clone.frame=SubstructureFrame(self.definition['nodes'],interfaces,columns,supports=supports,
            input_identity={'method':METHOD+'/frozen_case_rebind','definition':self.definition['identity_sha256'],
                            'source':self.source_identity,'case_identity':case_identity})
        return clone

    def solve(self,*,pdelta=True,recover_fields=True,saved_response=None):
        started=time.perf_counter()
        engine=PairedCoupledEquilibrium(self.frame,self.floors,input_identity={'method':METHOD,'source':self.source_identity,'cases':self.cases})
        if saved_response is None:frame=engine.solve(pdelta=pdelta)
        else:
            engine.verify_saved(saved_response['frame'],pdelta=pdelta);frame=saved_response['frame']
            if saved_response['cases']!=self.cases:raise ValueError('Saved physical gravity case inventory differs')
        result=self._recover(frame,self.cases,list(range(len(self.cases))),
                             recover_fields=recover_fields,started=started,stage='coupled_gravity_action_analysis')
        if saved_response is not None:verify_recovered_fields(result,saved_response)
        return result

    def set_floor_operators(self,operators,*,iteration_identity):
        """Install complete explicit constitutive operators for this analysis.

Geometry, supports, columns and applied loads remain the frozen actual inputs.
All floors are supplied together, so a secant update cannot leave an old floor
factorization paired with new material actions. This is a solver operation,
not an assertion of material convergence or strength acceptance. Iterations
for different load cases require separate one-case systems unless the caller
proves that their constitutive matrices are identical.
"""
        expected={r['description']['id'] for r in self.floors}
        if set(operators)!=expected or not iteration_identity:
            raise ValueError('Complete actual-floor operator inventory and iteration identity required')
        if len(self.cases)!=1:
            raise ValueError('A constitutive secant update requires one actual simultaneous load case')
        proposed=[];interfaces=[]
        for row in self.floors:
            floor=row['assembly'];model=operators[row['description']['id']]
            if getattr(model,'floor',None) is not floor:
                raise ValueError('Constitutive floor operator belongs to different physical geometry')
            if model.stiffness.shape!=(len(floor.dof_ids),len(floor.dof_ids)):
                raise ValueError('Constitutive floor DOF inventory differs from the frozen physical geometry')
            model=paired_floor_model(model,input_identity={'source':self.source_identity,'iteration':iteration_identity,'floor':floor.id})
            rhs=np.column_stack([r['loads'] for r in row['loads']])
            sub=condense(model.stiffness,floor.dof_ids,floor.boundary_dofs,rhs,
                input_identity={'definition':self.definition['identity_sha256'],'source':self.source_identity,
                                'floor':floor.id,'stiffness':model.metadata,'cases':self.cases,
                                'iteration_identity':iteration_identity})
            proposed.append((row,model,sub))
            interfaces.append({'id':floor.id,'dof_ids':floor.boundary_dofs,'stiffness':sub.stiffness,
                               'loads':sub.loads,'input_identity':sub.record()['identity_sha256']})
        supports=[(n['id'],d) for n in self.definition['nodes'] if n['xyz'][2]==0 for d in range(1,7)]
        frame=SubstructureFrame(self.definition['nodes'],interfaces,self._columns,
            input_identity={'method':METHOD+'/constitutive_update','definition':self.definition['identity_sha256'],
                            'source':self.source_identity,'iteration_identity':iteration_identity},supports=supports)
        for row,model,sub in proposed:row['model'],row['condensed']=model,sub
        self._interfaces=interfaces;self.frame=frame
        return {'iteration_identity':iteration_identity,'floor_operator_identities':
                {r['description']['id']:r['model'].metadata for r in self.floors},'qualification_asserted':False}

    def solve_additional_floor_loads(self,additional_loads,*,baseline_case_id,case_ids,input_identity,pdelta=True,saved_response=None):
        """Exact physical increments under explicit balanced floor free bodies.

        Supply one retained-DOF load matrix per actual floor, including explicit
        zero matrices on other floors. The original simultaneous gravity load
        and actual column P are retained in every case. All column DOFs remain
        compatible; every additional column force and slab/web M/Q accompanies
        the membrane increment. This mechanical API does not automatically
        authorize addition to a separate primary analysis branch.
        """
        expected={r['description']['id'] for r in self.floors};case_ids=list(case_ids)
        if not input_identity or set(additional_loads)!=expected or not case_ids or len(set(case_ids))!=len(case_ids):
            raise ValueError('Complete physical floor loads, unique cases and input identity required')
        matches=[i for i,c in enumerate(self.cases) if c['id']==baseline_case_id]
        if len(matches)!=1 or baseline_case_id in case_ids:raise ValueError('Unique original gravity baseline required')
        base_index=matches[0];base=self.cases[base_index];count=len(case_ids)
        clone=object.__new__(CoupledFloorSystem);clone.definition=self.definition;clone.source_identity=self.source_identity
        clone._dead_joint_wrenches=self._dead_joint_wrenches
        clone.cases=[base,*[{**base,'id':name,'gravity_case_id':baseline_case_id,'branch':'physical_additional_floor_load'} for name in case_ids]]
        clone.floors=[];interfaces=[];load_audits={};increments={}
        for row in self.floors:
            floor=row['assembly'];ident=row['description']['id'];extra=np.asarray(additional_loads[ident],float)
            if extra.shape!=(len(floor.dof_ids),count) or not np.isfinite(extra).all():
                raise ValueError('Actual finite retained floor load matrix dimensions differ')
            original=row['loads'][base_index]
            loads=[original,*[{**original,'loads':original['loads']+extra[:,j]} for j in range(count)]]
            rhs=np.column_stack([r['loads'] for r in loads]);fc=row['condensed'].condense_loads(rhs)
            positions=np.array([floor.node_positions[tag] for tag in floor.node_ids]);positions[:,2]+=row['description']['z_in'];wrenches=[]
            for j in range(count):
                physical_count=6*len(floor.node_ids);force=extra[:physical_count,j].reshape(-1,6)
                # Independent retained web-normal rotations are nodal Rz
                # couples; they have no translational force or lever arm.
                extra_normal=float(np.sum(extra[physical_count:,j]))
                if any(d!=6 for _,d in floor.dof_ids[physical_count:]):raise ValueError('Unknown additional retained rotation load')
                wrenches.append(np.r_[force[:,:3].sum(axis=0),
                    (force[:,3:]+np.cross(positions,force[:,:3])).sum(axis=0)+[0.,0.,extra_normal]].tolist())
            load_audits[ident]=dict(additional_loads_sha256=_hash(extra.tolist()),global_wrenches_kip_kipin=wrenches,
                                   dof_ids=floor.dof_ids,case_ids=case_ids)
            interfaces.append(dict(id=ident,dof_ids=floor.boundary_dofs,stiffness=row['condensed'].stiffness,loads=fc,
                input_identity={'original_stiffness':row['condensed'].record()['identity_sha256'],
                                'additional':load_audits[ident],'source':input_identity}))
            clone.floors.append({**row,'loads':loads})
        clone._interfaces=interfaces
        clone._columns=[{**c,'axial_load_kip_per_in':[c['axial_load_kip_per_in'][base_index]]*(count+1)} for c in self._columns]
        supports=[(n['id'],d) for n in self.definition['nodes'] if n['xyz'][2]==0 for d in range(1,7)]
        clone.frame=SubstructureFrame(self.definition['nodes'],interfaces,clone._columns,supports=supports,
            input_identity={'method':METHOD+'/additional_floor_free_body','definition':self.definition['identity_sha256'],
                            'source':self.source_identity,'additional':input_identity,'load_audits':load_audits})
        solved=clone.solve(pdelta=pdelta,saved_response=saved_response)
        if solved['status']!='equilibrated':return dict(status='unresolved',total_response=solved,load_audits=load_audits)
        for j,name in enumerate(case_ids,1):
            actual=solved['frame']['cases'][j];baseline=solved['frame']['cases'][0]
            columns={tag:{'global_force_increment':(np.array(r['global_resisting_force'])-
                         baseline['columns'][tag]['global_resisting_force']).tolist(),
                         'local_force_increment':(np.array(r['local_force'])-baseline['columns'][tag]['local_force']).tolist(),
                         'initial_basic_axial_tension_kip':baseline['columns'][tag]['pdelta_basic_axial_tension_positive'],
                         'loaded_basic_axial_tension_kip':r['pdelta_basic_axial_tension_positive']}
                     for tag,r in actual['columns'].items()}
            floors=[]
            for row,output in zip(clone.floors,solved['floors']):
                first=output['responses'][0];loaded=output['responses'][j]
                duh,dul=add_pair(np.array(loaded['retained_displacements']),np.array(loaded['retained_displacements_lo']),
                    -np.array(first['retained_displacements']),-np.array(first['retained_displacements_lo']))
                zero={**row['loads'][0],'loads':np.asarray(additional_loads[output['id']])[:,j-1],
                    'element_loads':{key:np.zeros_like(value) for key,value in row['loads'][0]['element_loads'].items()},
                    'beam_weight_factor':0.,'joint_wrenches':{}}
                field=row['model'].recover_pair(duh,dul,zero)
                before={r['id']:r for r in first['elements']};after={r['id']:r for r in loaded['elements']}
                error=max(float(np.max(abs(np.asarray(e['global_resisting_force'])-
                    (np.asarray(after[e['id']]['global_resisting_force'])-before[e['id']]['global_resisting_force']))))
                    for e in field['elements'])
                if error>1e-7:raise ValueError('Physical increment fields differ from exact same-baseline subtraction')
                floors.append(dict(id=output['id'],level=output['level'],response={**field,'case_id':name,
                    'retained_displacements':duh.tolist(),'retained_displacements_lo':dul.tolist()},field_subtraction_max_error_kip_kipin=error))
            djh,djl=add_pair(np.array(actual['physical_displacements']),np.array(actual['physical_displacements_lo']),
                -np.array(baseline['physical_displacements']),-np.array(baseline['physical_displacements_lo']))
            increments[name]=dict(columns=columns,floors=floors,displacement_increment=(np.array(
                djh)+djl).tolist(),displacement_increment_hi=djh.tolist(),displacement_increment_lo=djl.tolist(),
                all_six_column_force_components_retained=True)
        return dict(method=METHOD+'/additional_floor_free_body',status='equilibrated',
            definition_sha256=self.definition['identity_sha256'],source_identity=self.source_identity,
            baseline_case_id=baseline_case_id,input_identity=input_identity,load_audits=load_audits,
            total_response=solved,increments=increments,qualification_asserted=False,
            primary_force_composition_authorized=False)

    def solve_imposed_floor_displacements(self,imposed_cases,*,pdelta=True,recover_fields=True,saved_response=None):
        """Coherent gravity plus design-displacement compatibility branches.

        ASCE7-22 12.12.4 and ACI318-19 18.14.2.1: prescribe actual design
        floor translations and torsion, leaving every vertical displacement
        and horizontal-axis joint rotation compatible/free. This does not
        designate the gravity members as SFRS or replace primary ELF forces.

        Each row requires id,gravity_case_id,levels; levels contain floor and
        design_in={ux,uy,rz_rad}, already amplified by Cd/Ie. Optional sign is
        +/-1. Values are NOT amplified again. Every complete branch is kept;
        its whole moment is never added to a primary ELF moment at the same cut.
        """
        started=time.perf_counter();imposed_cases=list(imposed_cases)
        if not imposed_cases or len({r['id'] for r in imposed_cases})!=len(imposed_cases):
            raise ValueError('Distinct nonempty displacement-compatibility cases required')
        byid={c['id']:i for i,c in enumerate(self.cases)};indices=[];cases=[]
        nodes=self.definition['nodes'];geometry=self.definition['geometry']
        ids=[(n['id'],d) for n in nodes for d in range(1,7)]
        free=[i for i,(node,d) in enumerate(ids) if nodes[i//6]['xyz'][2]>0 and d in (3,4,5)]
        transform=sparse.csr_matrix((np.ones(len(free)),(free,np.arange(len(free)))),shape=(len(ids),len(free)))
        lift=np.zeros((len(ids),len(imposed_cases)));references=[]
        for column,row in enumerate(imposed_cases):
            if row['gravity_case_id'] not in byid:raise ValueError('Compatibility branch lacks its actual simultaneous gravity case')
            indices.append(byid[row['gravity_case_id']]);gravity=self.cases[indices[-1]]
            levels={r['floor']:r['design_in'] for r in row['levels']}
            if len(levels)!=len(row['levels']) or set(levels)!=set(range(1,geometry['nf']+1)):
                raise ValueError('Every actual floor design displacement must be supplied exactly once')
            sign=float(row.get('sign',1.))
            if sign not in (-1.,1.):raise ValueError('Compatibility sign must be +/-1')
            origin=np.asarray(row.get('reference_xy_in',[geometry['nx']*geometry['lx']/2,geometry['ny']*geometry['ly']/2]),float)
            if origin.shape!=(2,) or not np.isfinite(origin).all():raise ValueError('Finite actual floor reference required')
            references.append(origin.tolist())
            for n,node in enumerate(nodes):
                x,y,z=node['xyz']
                if z==0:continue
                level=int(round(z/geometry['story_h']));v=levels[level]
                values=np.array([v['ux'],v['uy'],v['rz_rad']],float)*sign
                if not np.isfinite(values).all():raise ValueError('Design displacements must be finite')
                lift[6*n,column]=values[0]-(y-origin[1])*values[2]
                lift[6*n+1,column]=values[1]+(x-origin[0])*values[2]
                lift[6*n+5,column]=values[2]
            cases.append({**gravity,'id':row['id'],'gravity_case_id':row['gravity_case_id'],
                          'branch':'physical_deformation_compatibility','imposed':row,
                          'basis':'ASCE7-22 12.12.4; ACI318-19 18.14.2.1'})
        interfaces=[{**f,'loads':np.asarray(f['loads'])[:,indices],
                     'input_identity':{'original':f['input_identity'],'case_indices':indices}} for f in self._interfaces]
        columns=[{**c,'axial_load_kip_per_in':[c['axial_load_kip_per_in'][i] for i in indices]} for c in self._columns]
        frame=SubstructureFrame(nodes,interfaces,columns,input_identity={'method':METHOD+'/imposed_design_displacements',
            'definition':self.definition['identity_sha256'],'source':self.source_identity,'cases':cases},
            kinematic_transform=transform,reduced_dof_ids=[ids[i] for i in free],support_lift=lift)
        engine=PairedCoupledEquilibrium(frame,self.floors,case_indices=indices,
            input_identity={'method':METHOD+'/imposed_design_displacements','source':self.source_identity,'cases':cases})
        if saved_response is None:solved=engine.solve(pdelta=pdelta)
        else:
            engine.verify_saved(saved_response['frame'],pdelta=pdelta);solved=saved_response['frame']
            if saved_response['cases']!=cases:raise ValueError('Saved imposed displacement case inventory differs')
        result=self._recover(solved,cases,indices,recover_fields=recover_fields,started=started,
                             stage='physical_deformation_compatibility')
        result['displacement_reference_xy_in']=references
        if saved_response is not None:verify_recovered_fields(result,saved_response)
        return result

    def _recover(self,frame,cases,case_indices,*,recover_fields,started,stage):
        output={'method':METHOD,'definition_sha256':self.definition['identity_sha256'],'source_identity':self.source_identity,
                'cases':cases,'frame':frame,'floors':[],'qualification_asserted':False,
                'stage':stage,'official_pilot_started':False}
        index={tuple(d):i for i,d in enumerate(frame['physical_dof_ids'])}
        if frame['status']!='equilibrated':
            output['status']='unresolved';return output
        for floor_index,row in enumerate(self.floors):
            floor,model,sub=row['assembly'],row['model'],row['condensed']
            states=[c['floor_states'][floor_index] for c in frame['cases']]
            if any(s['id']!=floor.id for s in states):raise ValueError('Paired physical floor case inventory differs')
            audit={'method':METHOD,'cases':[s['equilibrium'] for s in states],
                'maximum_interior_force_residual':max(s['equilibrium']['absolute_interior_force_residual'] for s in states),
                'maximum_stationary_potential_energy_residual':max(s['equilibrium']['stationary_potential_energy_residual'] for s in states),
                'stationary_potential_energy_basis':'full physical potential and boundary work; preconditioner energy not used',
                'boundary_reactions':np.array([s['boundary_reactions'] for s in states]).T.tolist(),
                'preconditioner_is_authoritative':False}
            recovered=[]
            if recover_fields:
                for i,case in enumerate(cases):
                    recovered.append({'case_id':case['id'],**model.recover_pair(states[i]['displacements_hi'],
                        states[i]['displacements_lo'],row['loads'][case_indices[i]])})
            output['floors'].append({'id':row['description']['id'],'level':row['description']['level'],
                'grid':{'x_in':floor.x.tolist(),'y_in':floor.y.tolist()},'stiffness':model.metadata,
                'condensation':sub.record(),'equilibrium':audit,'responses':recovered})
        output.update(status='equilibrated',elapsed_seconds=time.perf_counter()-started)
        return output
