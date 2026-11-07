"""Balance paired physical floors and columns, using Schur factors only to solve.

Both joint and floor-interior equations determine convergence. Existing floor
and frame factors are block preconditioners, never physical force authorities.
The optional PDelta law remains ElasticColumn's native basic-axial-force law.
This numerical mechanism does not qualify material or engineering premises.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import splu
from Design.SMRF_Paired_Arithmetic import add_pair, scale_pair, paired_dense_action, paired_sparse_action
from Design.SMRF_Static_Substructure import _json_hash
from Design.SMRF_Substructure_Frame import _token

METHOD='paired_full_floor_and_column_equilibrium_v1_unqualified'


def _norm(value):
    return float(np.max(np.abs(value),initial=0.))


def _work(uh,ul,fh,fl):
    """Paired work of work-conjugate physical displacements and forces."""
    a,b=paired_dense_action(np.asarray(uh)[None,:],fh,fl)
    c,d=paired_dense_action(np.asarray(ul)[None,:],fh,fl)
    h,l=add_pair(a,b,c,d);return h[0],l[0]


def paired_column_response(element,high,low,weight,*,pdelta):
    """Same local elastic and native PDelta law, retaining arithmetic low parts."""
    ulh,ull=paired_dense_action(element.transform,high,low)
    bh,bl=paired_dense_action(element.basic,ulh,ull)
    bh,bl=paired_dense_action(element.kb,bh,bl)
    bh[0],bl[0]=add_pair(bh[0],bl[0],-weight*element.length/2)
    lh,ll=paired_dense_action(element.local_stiffness,ulh,ull)
    equivalent=element.equivalent_load(weight)
    eh,el=paired_dense_action(element.transform,equivalent,np.zeros(12))
    lh,ll=add_pair(lh,ll,-eh,-el)
    fh,fl=paired_dense_action(element.transform.T,lh,ll)
    gh=gl=np.zeros(12)
    if pdelta:
        dh,dl=paired_dense_action(element.geometric_unit,high,low)
        gh,gl=add_pair(*scale_pair(dh,dl,bh[0]),*scale_pair(dh,dl,bl[0]))
        fh,fl=add_pair(fh,fl,gh,gl)
    response=dict(local_force=(lh+ll).tolist(),global_resisting_force=(fh+fl).tolist(),
        basic_force=(bh+bl).tolist(),pdelta_basic_axial_tension_positive=float(bh[0]+bl[0]),
        geometric_global_force=(gh+gl).tolist(),local_displacement=(ulh+ull).tolist())
    # Resistance already subtracts the element's own equivalent weight.
    return fh,fl,response


class PairedCoupledEquilibrium:
    def __init__(self,frame,floors,*,input_identity,case_indices=None,nodal_loads=None,tolerance_scale=1.):
        if not input_identity or not floors:raise ValueError('Explicit physical floor system identity required')
        if not np.isfinite(tolerance_scale) or not 0<tolerance_scale<=1.:
            raise ValueError('Equilibrium tolerances may only be tightened')
        self.tolerance_scale=float(tolerance_scale)
        self.frame=frame;self.floors=list(floors);self.count=frame.case_count
        self.case_indices=list(range(self.count)) if case_indices is None else list(case_indices)
        if len(self.case_indices)!=self.count or any(type(v) is not int or v<0 for v in self.case_indices):
            raise ValueError('Complete explicit simultaneous case mapping required')
        self.nodal=np.zeros_like(frame.loads) if nodal_loads is None else np.asarray(nodal_loads,float)
        if self.nodal.shape!=frame.loads.shape or not np.isfinite(self.nodal).all():raise ValueError('Complete finite frame nodal loads required')
        self.T=frame.transform.tocsr();self.Tt=self.T.T.tocsr();self.maps=[];identities=[]
        expected=self.nodal.copy()
        for row in self.floors:
            floor,model,sub=row['assembly'],row['model'],row['condensed']
            if not callable(getattr(model,'apply_pair',None)) or model.floor is not floor:
                raise ValueError('An explicit paired operator on the same floor is required')
            if floor.dof_ids!=sub.dof_ids or floor.boundary_dofs!=sub.boundary_dofs:
                raise ValueError('Physical floor and preconditioner inventories differ')
            inds=np.array([frame._index[_token(d)] for d in floor.boundary_dofs],dtype=int)
            if len(set(inds))!=len(inds):raise ValueError('Duplicate physical boundary DOF')
            if max(self.case_indices)>=len(row['loads']):raise ValueError('Missing simultaneous floor load case')
            rhs=np.column_stack([row['loads'][i]['loads'] for i in self.case_indices])
            if rhs.shape!=(len(floor.dof_ids),self.count) or not np.isfinite(rhs).all():raise ValueError('Complete finite physical floor loads required')
            # Validate correspondence to the frame preconditioner, including any
            # explicitly supplied nodal loads. This check is not equilibrium.
            expected[inds]+=sub.condense_loads(rhs)
            self.maps.append((inds,rhs))
            identities.append(dict(id=floor.id,model=model.metadata,preconditioner=sub.record()['identity_sha256'],
                physical_loads_sha256=_json_hash(rhs.tolist())))
        for element,indices,weights in frame.columns:
            for case in range(self.count):expected[indices,case]+=element.equivalent_load(weights[0 if len(weights)==1 else case])
        if _norm(expected-frame.loads)>1e-10*max(1.,_norm(frame.loads)):
            raise ValueError('Physical loads do not match frame/floor preconditioner case inventory')
        self.identity=dict(method=METHOD,input_identity=input_identity,frame=frame.identity_sha256,floors=identities,
            case_indices=self.case_indices,nodal_loads_sha256=_json_hash(self.nodal.tolist()),tolerance_scale=self.tolerance_scale,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.identity_sha256=_json_hash(self.identity)

    def _joint(self,qh,ql,case):
        return add_pair(*paired_sparse_action(self.T,qh,ql),self.frame.lift[:,case])

    def _evaluate(self,qh,ql,interiors,case,*,pdelta):
        frame=self.frame;jh,jl=self._joint(qh,ql,case)
        rh,rl=add_pair(-self.nodal[:,case],np.zeros(len(jh)),0.)
        columns={}
        for element,indices,weights in frame.columns:
            ch,cl,record=paired_column_response(element,jh[indices],jl[indices],weights[0 if len(weights)==1 else case],pdelta=pdelta)
            rh[indices],rl[indices]=add_pair(rh[indices],rl[indices],ch,cl)
            columns[str(element.id)]=record
        floor_states=[];merit=0.
        for row,(indices,loads),(ih,il) in zip(self.floors,self.maps,interiors):
            sub=row['condensed'];n=len(sub.dof_ids)
            uh=np.empty(n);ul=np.empty(n);uh[sub._b]=jh[indices];ul[sub._b]=jl[indices]
            uh[sub._i]=ih;ul[sub._i]=il
            ah,al=row['model'].apply_pair(uh,ul)
            ah,al=np.asarray(ah,float),np.asarray(al,float)
            if ah.shape!=(n,) or al.shape!=(n,) or not np.isfinite(ah).all() or not np.isfinite(al).all():
                raise ValueError('Complete finite paired floor physical action required')
            fh,fl=add_pair(ah,al,-loads[:,case]);fr=fh+fl
            rh[indices],rl[indices]=add_pair(rh[indices],rl[indices],fh[sub._b],fl[sub._b])
            scale=max(1.,_norm(loads[:,case]),_norm(fr[sub._b]));error=_norm(fr[sub._i])
            tolerance=1e-9*self.tolerance_scale
            audit=dict(absolute_interior_force_residual=error,force_scale=scale,relative_interior_force_residual=error/scale,
                tolerance=tolerance,passed=error/scale<=tolerance,preconditioner_is_authoritative=False)
            merit=max(merit,error/(tolerance*scale))
            floor_states.append(dict(high=uh,low=ul,action_hi=ah,action_lo=al,residual_hi=fh,residual_lo=fl,audit=audit))
        reduced_h,reduced_l=paired_sparse_action(self.Tt,rh,rl);reduced=reduced_h+reduced_l
        # Retain the existing frame criterion and its original case load scale.
        fr=self.Tt@(frame.loads[:,case]-frame.stiffness@frame.lift[:,case])
        limit=(1e-8+1e-10*max(1.,_norm(fr)))*self.tolerance_scale;error=_norm(reduced);merit=max(merit,error/limit)
        return dict(joint_hi=jh,joint_lo=jl,residual_hi=rh,residual_lo=rl,reduced_residual=reduced,
            frame_error=error,frame_limit=limit,merit=merit,floors=floor_states,columns=columns)

    def _step(self,state,case,*,pdelta):
        # Condense the FULL physical correction residual. Boundary residuals
        # already occur in rj; adding a separate floor rb here double counts.
        frame=self.frame;rhs=-(state['residual_hi']+state['residual_lo']);solved=[]
        for row,(indices,_),fs in zip(self.floors,self.maps,state['floors']):
            sub=row['condensed'];ri=-(fs['residual_hi'][sub._i]+fs['residual_lo'][sub._i])
            yi=sub._solve(ri);rhs[indices]-=sub._kbi@yi;solved.append(ri)
        reduced=self.Tt@rhs
        if pdelta:
            tangent=frame._geometric(state['joint_hi']+state['joint_lo'],case,tangent=True)[1]
            factor=splu((frame.reduced_stiffness+self.Tt@tangent@self.T).tocsc())
        else:factor=frame._linear_factor
        dq=factor.solve(np.asarray(reduced).ravel());dj=self.T@dq;di=[]
        if not np.isfinite(dq).all():raise ValueError('Nonfinite coupled correction')
        for row,(indices,_),ri in zip(self.floors,self.maps,solved):
            sub=row['condensed'];di.append(sub._solve(ri-sub._kib@dj[indices]))
        return dq,di

    def _record(self,state,qh,ql,case):
        for row,(_,loads),fs in zip(self.floors,self.maps,state['floors']):
            sub=row['condensed'];h,l=fs['high'],fs['low']
            elastic=_work(h,l,fs['action_hi'],fs['action_lo'])
            external=_work(h,l,loads[:,case],np.zeros(len(h)))
            boundary=_work(h[sub._b],l[sub._b],fs['residual_hi'][sub._b],fs['residual_lo'][sub._b])
            # Pi=.5*u.F-u.f. At stationary interior DOFs, it also equals
            # .5*ub.rb-.5*u.f. This uses full physical work, not the rounded
            # preconditioner's condensed energy or a claimed Schur authority.
            eh,el=add_pair(*elastic,-boundary[0],-boundary[1]);eh,el=add_pair(eh,el,-external[0],-external[1])
            fs['audit'].update(stationary_potential_energy_residual=abs(float(eh+el))*.5,
                stationary_potential_energy_basis='full physical potential versus boundary reaction work minus half applied work',
                physical_potential_energy_kipin=float(.5*(elastic[0]+elastic[1])-(external[0]+external[1])))
        return dict(status='equilibrated' if state['merit']<=1. else 'unresolved',case_index=case,
            physical_displacements=state['joint_hi'].tolist(),physical_displacements_lo=state['joint_lo'].tolist(),
            reduced_displacements=qh.tolist(),reduced_displacements_lo=ql.tolist(),
            physical_residual_or_reaction=(state['residual_hi']+state['residual_lo']).tolist(),
            maximum_free_equilibrium_residual=state['frame_error'],equilibrium_tolerance=state['frame_limit'],
            columns=state['columns'],floor_states=[dict(id=r['assembly'].id,displacements_hi=f['high'].tolist(),
                displacements_lo=f['low'].tolist(),equilibrium=f['audit'],boundary_reactions=(f['residual_hi'][r['condensed']._b]+
                f['residual_lo'][r['condensed']._b]).tolist()) for r,f in zip(self.floors,state['floors'])],
            maximum_normalized_physical_residual=state['merit'])

    def solve(self,*,pdelta=True,max_iterations=40):
        if type(max_iterations) is not int or not 1<=max_iterations<=40:raise ValueError('Original at most forty coupled iterations required')
        started=time.perf_counter();frame=self.frame;results=[]
        for case in range(self.count):
            qh=frame._linear_factor.solve(np.asarray(self.Tt@(frame.loads[:,case]-frame.stiffness@frame.lift[:,case])).ravel());ql=np.zeros_like(qh)
            jh,jl=self._joint(qh,ql,case);interiors=[]
            for row,(indices,loads) in zip(self.floors,self.maps):
                sub=row['condensed'];ih=sub._solve(loads[sub._i,case]-sub._kib@(jh[indices]+jl[indices]))
                interiors.append((ih,np.zeros_like(ih)))
            state=self._evaluate(qh,ql,interiors,case,pdelta=pdelta);history=[]
            for iteration in range(max_iterations):
                history.append(dict(iteration=iteration,normalized_physical_residual=state['merit']))
                if state['merit']<=1:break
                dq,di=self._step(state,case,pdelta=pdelta);accepted=False
                for power in range(12):
                    th,tl=add_pair(qh,ql,dq/(2**power));trial_i=[add_pair(ih,il,step/(2**power)) for (ih,il),step in zip(interiors,di)]
                    trial=self._evaluate(th,tl,trial_i,case,pdelta=pdelta)
                    if trial['merit']<state['merit']:
                        qh,ql,interiors,state=th,tl,trial_i,trial;accepted=True;break
                if not accepted:break
            record=self._record(state,qh,ql,case);record.update(residual_history=history,iterations=iteration+1)
            results.append(record)
        return dict(method=METHOD,identity_sha256=self.identity_sha256,physical_dof_ids=frame.dof_ids,
            status='equilibrated' if all(r['status']=='equilibrated' for r in results) else 'unresolved',
            pdelta=bool(pdelta),cases=results,elapsed_seconds=time.perf_counter()-started,qualification_asserted=False,
            stiffness_role='block_preconditioner_only',floor_and_joint_physical_equilibrium=True)

    def verify_saved(self,record,*,pdelta=True):
        """Recompute physical forces at every saved high/low state, with no solve."""
        if (record.get('method')!=METHOD or record.get('identity_sha256')!=self.identity_sha256 or
            record.get('pdelta')!=bool(pdelta) or record.get('status')!='equilibrated' or
            _json_hash(record.get('physical_dof_ids'))!=_json_hash(self.frame.dof_ids) or len(record.get('cases',[]))!=self.count):
            raise ValueError('Saved paired coupled method, source, case or DOF identity differs')
        for case,saved in enumerate(record['cases']):
            qh,ql=(np.asarray(saved[k],float) for k in ('reduced_displacements','reduced_displacements_lo'))
            if qh.shape!=(self.T.shape[1],) or ql.shape!=qh.shape or not np.isfinite(qh).all() or not np.isfinite(ql).all():
                raise ValueError('Saved paired reduced state is incomplete')
            if len(saved.get('floor_states',[]))!=len(self.floors):raise ValueError('Saved floor inventory differs')
            interiors=[]
            for row,fs in zip(self.floors,saved['floor_states']):
                sub=row['condensed'];h,l=(np.asarray(fs[k],float) for k in ('displacements_hi','displacements_lo'))
                if h.shape!=(len(sub.dof_ids),) or l.shape!=h.shape or not np.isfinite(h).all() or not np.isfinite(l).all():
                    raise ValueError('Saved paired floor state is incomplete')
                interiors.append((h[sub._i],l[sub._i]))
            state=self._evaluate(qh,ql,interiors,case,pdelta=pdelta);actual=self._record(state,qh,ql,case)
            if actual['status']!='equilibrated':raise ValueError('Saved state fails full physical coupled equilibrium')
            # Exact deterministic rebuild also checks boundary compatibility,
            # low parts and complete forces; timing/iteration history are not proof.
            if _json_hash(actual)!=_json_hash({k:saved.get(k) for k in actual}):
                raise ValueError('Saved paired physical fields differ from reconstructed state')
        return dict(passed=True,method=METHOD,physical_cases_verified=self.count,new_global_solves=0)
