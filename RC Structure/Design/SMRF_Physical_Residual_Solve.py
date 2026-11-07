"""Bounded Dirichlet refinement checked on the actual complete physical action.

A supplied sparse matrix is only a preconditioner. All residuals evaluate the
authoritative operator at the complete physical displacement, including the
prescribed boundary lift. This avoids cancellation from separately formed
Kii*ui and Kib*ub. Failure is incomplete numerical evidence, never capacity.
"""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

METHOD='bounded_full_physical_dirichlet_residual_refinement_v1'


class PhysicalResidualConvergenceError(RuntimeError):
    def __init__(self,audit,displacements,physical_action):
        self.audit=audit
        self.displacements=np.asarray(displacements,float).copy()
        self.physical_action=np.asarray(physical_action,float).copy()
        super().__init__('Full physical equilibrium did not converge within the declared refinement bound')


def solve_physical_dirichlet(operator,preconditioner,boundary_indices,boundary_displacements,loads,*,input_identity,
                             tolerance=1e-9,maximum_steps=3):
    k=sparse.csc_matrix(preconditioner,dtype=float);n=k.shape[0]
    b=np.asarray(boundary_indices);ub=np.asarray(boundary_displacements,float);f=np.asarray(loads,float)
    if (not input_identity or k.shape!=(n,n) or not np.isfinite(k.data).all() or b.ndim!=1 or b.dtype.kind not in 'iu'
        or not 0<len(b)<n or len(set(b.tolist()))!=len(b) or np.any(b<0) or np.any(b>=n)
        or ub.shape!=(len(b),) or f.shape!=(n,) or not np.isfinite(ub).all() or not np.isfinite(f).all()):
        raise ValueError('Finite complete physical system and distinct nonempty prescribed boundary required')
    if not math.isfinite(tolerance) or not 0<tolerance<=1e-9 or type(maximum_steps) is not int or not 0<=maximum_steps<=3:
        raise ValueError('Original strict tolerance and at most three residual corrections required')
    interior=np.setdiff1d(np.arange(n),b);factor=splu(k[interior,:][:,interior].tocsc());actions=0
    def action(u):
        nonlocal actions
        out=np.asarray(operator(u),float);actions+=1
        if out.shape!=(n,) or not np.isfinite(out).all():raise ValueError('Authoritative action returned invalid physical forces')
        return out
    u=np.zeros(n);u[b]=ub;u[interior]=factor.solve(f[interior]-action(u)[interior])
    force=action(u);residual=f-force;error=float(np.max(abs(residual[interior]),initial=0.));history=[]
    initial=error;corrections=0
    # Iterate on the actual force residual even if a coefficient-scaled
    # backward error would already pass. Retain only improvements.
    for step in range(maximum_steps):
        if error==0.:break
        trial=u.copy();trial[interior]+=factor.solve(residual[interior]);corrections+=1
        trial_force=action(trial);trial_residual=f-trial_force;candidate=float(np.max(abs(trial_residual[interior]),initial=0.))
        accepted=candidate<error;history.append(dict(step=step+1,absolute_interior_force_residual=candidate,retained=accepted))
        if not accepted:break
        u,force,residual,error=trial,trial_force,trial_residual,candidate
    # Force/reaction scale, not |K|*|u|: large cancelling coefficients cannot
    # manufacture acceptance. Boundary reactions are retained, not cancelled.
    scale=max(1.,float(np.max(abs(f),initial=0.)),float(np.max(abs((force-f)[b]),initial=0.)))
    audit=dict(method=METHOD,input_identity=input_identity,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerance=tolerance,maximum_steps=maximum_steps,correction_solves=corrections,operator_applications=actions,
        initial_absolute_interior_force_residual=initial,absolute_interior_force_residual=error,
        force_scale=scale,relative_interior_force_residual=error/scale,history=history,
        preconditioner_factorizations=1,preconditioner_is_authoritative=False,passed=error/scale<=tolerance)
    if not audit['passed']:raise PhysicalResidualConvergenceError(audit,u,force)
    return dict(displacements=u,physical_action=force,boundary_reactions=(force-f)[b],audit=audit)
