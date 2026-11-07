"""Explicit paired-coefficient physical energy and bounded Dirichlet solve.

Numerical API only; it does not activate a production engineering profile.
The ordinary matrix is a preconditioner. Full physical high/low coefficient
actions govern all residuals; all gauge equations remain checked. Physical
displacement pairs must also survive through downstream local recovery.
"""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Implicit_Membrane_Energy import ImplicitMembraneEnergy
from Design.SMRF_Paired_Arithmetic import add_pair,scale_pair,paired_sparse_action,two_sum
from Design.SMRF_Accurate_Sparse_Action import _products

METHOD='paired_full_physical_equilibrium_v1_unqualified'

def _pair_action(high,low,xh,xl):
    return add_pair(*paired_sparse_action(high,xh,xl),*paired_sparse_action(low,xh,xl))

class PairedMembraneEnergy(ImplicitMembraneEnergy):
    """Same composite energy, preserving coefficient and field low parts."""
    def __init__(self,membrane_high,membrane_low,other_high,other_low,membrane_indices,**kwargs):
        if any(np.iscomplexobj(v.data if sparse.issparse(v) else v) for v in (membrane_high,membrane_low,other_high,other_low)):
            raise ValueError('Real physical coefficient pairs required')
        super().__init__(membrane_high,other_high,membrane_indices,**kwargs)
        self.Mlow=sparse.csr_matrix(membrane_low,dtype=float);self.Olow=sparse.csr_matrix(other_low,dtype=float)
        if any(v.shape!=self.shape or not np.isfinite(v.data).all() for v in (self.Mlow,self.Olow)):
            raise ValueError('Finite complete coefficient pairs required')
        for high,low in ((self.M,self.Mlow),(self.O,self.Olow)):
            scale=max(1.,float(np.max(abs(high.data),initial=0.)),float(np.max(abs(low.data),initial=0.)))
            if np.max(abs((high-high.T+low-low.T).data),initial=0.)>1e-10*scale:raise ValueError('Symmetric physical coefficient pairs required')
        scale=max(1.,float(np.max(abs(self.M.data),initial=0.)),float(np.max(abs(self.Mlow.data),initial=0.)))
        modes=np.asarray(kwargs['rigid_modes'],float)
        defect=float(np.max(abs((self.M[:,self.m]+self.Mlow[:,self.m])@modes),initial=0.))/scale
        if defect>self.tolerance:raise ValueError('Coefficient pairs violate the declared physical rigid modes')
        self.Alow=self.Mlow[self.m,:][:,self.m].tocsr();self.Blow=self.Mlow[self.m,:][:,self.b].tocsr()
        self.full_free_low=self.Alow[:,self.free].tocsr()
        self.metadata.update(method=METHOD,coefficient_pairs_retained=True,
            paired_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paired_rigid_mode_relative_defect=defect)

    def paired_fields(self,high,low):
        uh,ul=np.asarray(high,float),np.asarray(low,float)
        if uh.shape!=(self.shape[0],) or ul.shape!=uh.shape or not np.isfinite(uh).all() or not np.isfinite(ul).all():
            raise ValueError('Finite complete physical displacement pair required')
        bh,bl=_pair_action(self.B,self.Blow,uh[self.b],ul[self.b]);zh=self.factor.solve((bh+bl)[self.free]);zl=np.zeros_like(zh)
        def residual(h,l):
            ah,al=_pair_action(self.full_free,self.full_free_low,h,l)
            rh,rl=add_pair(bh,bl,-ah,-al)
            return rh+rl
        res=residual(zh,zl);error=float(np.max(abs(res),initial=0.));initial=error;history=[]
        for step in range(3):
            th,tl=add_pair(zh,zl,self.factor.solve(res[self.free]));candidate=residual(th,tl);value=float(np.max(abs(candidate),initial=0.));accepted=value<error
            history.append(dict(step=step+1,absolute_full_equation_residual=value,retained=accepted))
            if not accepted:break
            zh,zl,res,error=th,tl,candidate,value
        scale=max(1.,float(np.max(abs(bh+bl),initial=0.)))
        self.field_solves+=1;self.last_relaxation=dict(method=METHOD,maximum_steps=3,initial_absolute_residual=initial,
            final_absolute_residual=error,rhs_scale=scale,relative_full_equation_residual=error/scale,
            tolerance=self.tolerance,history=history,all_equation_rows_included=True)
        if error/scale>self.tolerance:raise ValueError('Paired full membrane equilibrium did not converge')
        sh=np.zeros(len(self.m));sl=np.zeros_like(sh);sh[self.free]=zh;sl[self.free]=zl
        mh=np.zeros_like(uh);ml=np.zeros_like(ul);mh[self.m],ml[self.m]=add_pair(uh[self.m],ul[self.m],sh,sl)
        wh,wl=scale_pair(uh,ul,self.beta)
        wh[self.m],wl[self.m]=add_pair(uh[self.m],ul[self.m],*scale_pair(sh,sl,1.-self.beta))
        return dict(projected_hi=uh.copy(),projected_lo=ul.copy(),membrane_hi=mh,membrane_lo=ml,
            weighted_hi=wh,weighted_lo=wl,shift_hi=sh,shift_lo=sl)

    def apply_pair(self,high,low):
        fields=self.paired_fields(high,low);self.applications+=1;self.last_fields=fields
        return add_pair(*_pair_action(self.M,self.Mlow,fields['weighted_hi'],fields['weighted_lo']),
                        *_pair_action(self.O,self.Olow,fields['projected_hi'],fields['projected_lo']))

    def fields(self,physical):
        raise ValueError('Paired physical energy requires explicit high/low field recovery')

    def apply(self,physical):
        raise ValueError('Paired physical energy requires explicit apply_pair')

def remove_rigid_pair(floor,high,low):
    """Subtract a chosen six-component rigid motion without losing low bits.

    Its estimated coefficients need not be exact: any physical rigid motion
    may be removed. The coordinate products/subtractions retain their errors.
    This function introduces no restraint and claims no element-nullspace proof.
    """
    h,l=np.asarray(high,float),np.asarray(low,float);n=len(floor.dof_ids);count=6*len(floor.node_ids)
    if h.shape!=(n,) or l.shape!=(n,) or not np.isfinite(h).all() or not np.isfinite(l).all():raise ValueError('Complete finite physical displacement pair required')
    coords=np.asarray([floor.node_positions[node] for node in floor.node_ids],float)
    if coords.shape!=(count//6,3) or not np.isfinite(coords).all():raise ValueError('Complete finite physical node coordinates required')
    reference=coords.mean(axis=0);centered=coords-reference;motion=h[:count].reshape(-1,6)
    translation=motion[:,:3].mean(axis=0);delta=motion[:,:3]-translation;x,y=centered[:,0],centered[:,1]
    denominator=float(np.sum(x*x+y*y));plane=np.column_stack((y,-x))
    if denominator<=0 or np.linalg.matrix_rank(plane)!=2:raise ValueError('Nondegenerate planar physical floor required')
    rz=float(np.sum(x*delta[:,1]-y*delta[:,0])/denominator);rx,ry=np.linalg.solve(plane.T@plane,plane.T@delta[:,2]);rotation=np.array([rx,ry,rz])
    dh=h.copy();dl=l.copy();xh,xl=two_sum(coords,-reference);vh,vl=dh[:count].reshape(-1,6),dl[:count].reshape(-1,6)
    for axis in range(3):
        vh[:,axis],vl[:,axis]=add_pair(vh[:,axis],vl[:,axis],-translation[axis])
        for ri,ci,sign in (((axis+1)%3,(axis+2)%3,-1.),((axis+2)%3,(axis+1)%3,1.)):
            ph,pl=_products(np.full(len(coords),sign*rotation[ri]),xh[:,ci]);qh,ql=_products(np.full(len(coords),sign*rotation[ri]),xl[:,ci])
            vh[:,axis],vl[:,axis]=add_pair(vh[:,axis],vl[:,axis],ph,pl)
            vh[:,axis],vl[:,axis]=add_pair(vh[:,axis],vl[:,axis],qh,ql)
        vh[:,axis+3],vl[:,axis+3]=add_pair(vh[:,axis+3],vl[:,axis+3],-rotation[axis])
    if count<n:
        extra=list(range(count,n))
        if extra!=getattr(floor,'web_rotation_connection',{}).get('internal_rotation_dof_indices') or any(floor.dof_ids[i][1]!=6 for i in extra):
            raise ValueError('Only explicit retained normal rotations may follow floor nodes')
        dh[count:],dl[count:]=add_pair(dh[count:],dl[count:],-rotation[2])
    return dh,dl,dict(translation=translation.tolist(),rotation=rotation.tolist(),reference=reference.tolist())

class PairedPhysicalConvergenceError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__('Paired complete physical equilibrium did not converge within its declared bound')

def solve_paired_dirichlet(operator,preconditioner,boundary_indices,boundary_displacements,loads,*,input_identity,
                          initial_pair=None,tolerance=1e-9,maximum_steps=3):
    k=sparse.csc_matrix(preconditioner,dtype=float);n=k.shape[0];b=np.asarray(boundary_indices);ub=np.asarray(boundary_displacements,float);f=np.asarray(loads,float)
    if (not input_identity or k.shape!=(n,n) or not np.isfinite(k.data).all() or b.ndim!=1 or b.dtype.kind not in 'iu'
        or not 0<len(b)<n or len(set(b.tolist()))!=len(b) or np.any(b<0) or np.any(b>=n) or ub.shape!=(len(b),)
        or f.shape!=(n,) or not np.isfinite(ub).all() or not np.isfinite(f).all()):raise ValueError('Explicit consistent finite physical system required')
    if not math.isfinite(tolerance) or not 0<tolerance<=1e-9 or type(maximum_steps) is not int or not 0<=maximum_steps<=3:
        raise ValueError('Original strict tolerance and at most three corrections required')
    interior=np.setdiff1d(np.arange(n),b);factor=splu(k[interior,:][:,interior].tocsc());actions=0
    def action(h,l):
        nonlocal actions
        ah,al=operator(h,l);ah,al=np.asarray(ah,float),np.asarray(al,float);actions+=1
        if ah.shape!=(n,) or al.shape!=(n,) or not np.isfinite(ah).all() or not np.isfinite(al).all():raise ValueError('Finite complete physical force pair required')
        rh,rl=add_pair(f,np.zeros(n),-ah,-al)
        return ah,al,rh+rl
    if initial_pair is None:
        uh=np.zeros(n);ul=np.zeros(n);uh[b]=ub;_,_,res=action(uh,ul);uh[interior]=factor.solve(res[interior])
    else:
        if len(initial_pair)!=2:raise ValueError('Explicit two-component initial state required')
        uh,ul=(np.asarray(v,float).copy() for v in initial_pair)
        if uh.shape!=(n,) or ul.shape!=(n,) or not np.isfinite(uh).all() or not np.isfinite(ul).all() or not np.array_equal(uh[b],ub) or np.any(ul[b]):
            raise ValueError('Initial guess must preserve the exact physical boundary')
    fh,fl,res=action(uh,ul);error=float(np.max(abs(res[interior])));initial=error;history=[]
    for step in range(maximum_steps):
        th,tl=uh.copy(),ul.copy();th[interior],tl[interior]=add_pair(uh[interior],ul[interior],factor.solve(res[interior]))
        ah,al,candidate=action(th,tl);value=float(np.max(abs(candidate[interior])));accepted=value<error
        history.append(dict(step=step+1,absolute_interior_force_residual=value,retained=accepted))
        if not accepted:break
        uh,ul,fh,fl,res,error=th,tl,ah,al,candidate,value
    scale=max(1.,float(np.max(abs(f))),float(np.max(abs(res[b]))))
    audit=dict(method=METHOD,input_identity=input_identity,tolerance=tolerance,maximum_steps=maximum_steps,
        initial_absolute_interior_force_residual=initial,absolute_interior_force_residual=error,force_scale=scale,
        relative_interior_force_residual=error/scale,history=history,operator_applications=actions,
        preconditioner_factorizations=1,preconditioner_is_authoritative=False,passed=error/scale<=tolerance,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result=dict(displacements_hi=uh,displacements_lo=ul,physical_action_hi=fh,physical_action_lo=fl,
        boundary_reactions=-res[b],audit=audit)
    if not audit['passed']:raise PairedPhysicalConvergenceError(result)
    return result
