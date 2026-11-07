"""Existing composite energy applied through physical membrane equations.

Avoid forming D-B'A+ B. For u=(m,b), solve A z=B b with only the caller's
declared rigid gauges, set um=(m+z,0), ub=u-um, and apply M(um+beta*ub)+O*u.
This is the same energy as GeneralizedFloorStiffness in exact arithmetic.
No explicit condensed matrix is the authority for an action or residual here.
The optional numerical method does not qualify any engineering floor model.
"""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Linear_Refinement import refine_gauged_rhs
from Design.SMRF_Accurate_Sparse_Action import accurate_sparse_action, METHOD as ACTION_METHOD

METHOD='implicit_physical_composite_membrane_energy_v2_accurate_action_unqualified'


class ImplicitMembraneEnergy:
    def __init__(self,membrane,other,membrane_indices,*,beam_bending_factor,rigid_modes,gauge_indices,input_identity,tolerance=1e-9,membrane_factor=None):
        self.M=sparse.csr_matrix(membrane,dtype=float);self.O=sparse.csr_matrix(other,dtype=float);self.shape=self.M.shape
        n=self.shape[0];self.beta=float(beam_bending_factor);self.tolerance=float(tolerance)
        if not input_identity or not math.isfinite(self.beta) or not 0<self.beta<=1 or not math.isfinite(tolerance) or not 0<tolerance<=1e-9:
            raise ValueError('Explicit identity, beam factor and unchanged strict tolerance required')
        for matrix in (self.M,self.O):
            if matrix.shape!=(n,n) or not np.isfinite(matrix.data).all():raise ValueError('Finite square physical operators required')
            scale=max(1.,float(np.max(abs(matrix.data),initial=0.)))
            if np.max(abs((matrix-matrix.T).data),initial=0.)>1e-10*scale:raise ValueError('Symmetric physical operators required')
        self.m=np.asarray(membrane_indices)
        if self.m.ndim!=1 or self.m.dtype.kind not in 'iu' or not len(self.m) or len(set(self.m.tolist()))!=len(self.m) or np.any(self.m<0) or np.any(self.m>=n):
            raise ValueError('Distinct actual membrane coordinates required')
        self.b=np.setdiff1d(np.arange(n),self.m);self.A=self.M[self.m,:][:,self.m].tocsr();self.B=self.M[self.m,:][:,self.b].tocsr()
        modes=np.asarray(rigid_modes,float);gauges=np.asarray(gauge_indices)
        if modes.ndim!=2 or modes.shape[0]!=len(self.m) or not np.isfinite(modes).all() or np.linalg.matrix_rank(modes)!=modes.shape[1]:
            raise ValueError('Explicit independent physical rigid modes required')
        if gauges.ndim!=1 or gauges.dtype.kind not in 'iu' or len(gauges)!=modes.shape[1] or len(set(gauges.tolist()))!=len(gauges) or np.any(gauges<0) or np.any(gauges>=len(self.m)):
            raise ValueError('Exactly one independent gauge per declared rigid mode required')
        if len(gauges) and np.linalg.matrix_rank(modes[gauges])!=len(gauges):raise ValueError('Gauge coordinates do not fix the declared rigid modes')
        scale=max(1.,float(np.max(abs(self.M.data),initial=0.)))
        defect=float(np.max(abs(self.M[:,self.m]@modes),initial=0.))/scale
        if defect>tolerance:raise ValueError('Declared gauges are not physical membrane rigid modes')
        self.free=np.setdiff1d(np.arange(len(self.m)),gauges);self.full_free=self.A[:,self.free].tocsr()
        if membrane_factor is not None and not callable(getattr(membrane_factor,'solve',None)):
            raise ValueError('Reused membrane factor must supply a correction solve')
        # A supplied factor is only a preconditioner. All physical rows still
        # govern the existing refinement/acceptance check; no extra LU is made.
        self.factor=membrane_factor if membrane_factor is not None else splu(self.A[self.free,:][:,self.free].tocsc())
        self.applications=0;self.field_solves=0;self.last_relaxation=None
        self.metadata=dict(method=METHOD,input_identity=input_identity,beam_bending_factor=self.beta,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gauge_indices=gauges.tolist(),
            declared_rigid_mode_relative_defect=defect,membrane_factorizations=int(membrane_factor is None),membrane_factor_reused=membrane_factor is not None,explicit_condensed_couple_formed=False,
            physical_force_tolerance=tolerance,qualification_asserted=False,action_arithmetic=ACTION_METHOD)

    def fields(self,physical):
        u=np.asarray(physical,float)
        if u.shape!=(self.shape[0],) or not np.isfinite(u).all():raise ValueError('Complete finite physical displacement vector required')
        rhs=np.asarray(self.B@u[self.b])[:,None]
        answer,audit=refine_gauged_rhs(self.full_free,rhs,self.free,self.factor.solve(rhs[self.free]),self.factor.solve)
        error=audit['final_absolute_residuals'][0]/max(1.,float(np.max(abs(rhs),initial=0.)))
        self.field_solves+=1;self.last_relaxation=dict(refinement=audit,relative_full_equation_residual=error,tolerance=self.tolerance)
        if error>self.tolerance:raise ValueError('Implicit physical membrane equilibrium did not converge')
        shift=np.zeros(len(self.m));shift[self.free]=answer[:,0];um=np.zeros_like(u);um[self.m]=u[self.m]+shift
        return um,u-um

    def apply(self,physical):
        u=np.asarray(physical,float);um,ub=self.fields(u);self.applications+=1
        return accurate_sparse_action(self.M,um+self.beta*ub)+accurate_sparse_action(self.O,u)
