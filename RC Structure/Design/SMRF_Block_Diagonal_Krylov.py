"""Bounded full-equation GMRES with two separately factored diagonal blocks.

For the stationary-q mixed system H=[[P,C],[C',-Q]], P and Q are positive
after the declared physical supports and auxiliary gauges. Use diag(P,Q)^-1
as a preconditioner, retaining H in every matrix action and final residual.
No full coupled LU, incomplete factor, penalty or relaxed convergence is used.
This numerical component does not assert positivity or physical qualification
of arbitrary caller matrices. A failed or exhausted solve raises and is saved
in solve_audits; it cannot become a successful floor response.
"""
from __future__ import annotations
import math
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,gmres,splu

METHOD='bounded_two_block_exact_preconditioned_gmres_v1'


class KrylovConvergenceError(RuntimeError):
    """Incomplete numerical evidence; this does not establish singularity."""
    def __init__(self,audit):
        self.audit=dict(audit)
        super().__init__('Bounded full-equation Krylov solve did not converge: '+str(
            {k:v for k,v in self.audit.items() if k!='preconditioned_residual_history'}))


class BlockDiagonalKrylovFactor:
    def __init__(self,matrix,positive_size,*,relative_tolerance=1e-13,maximum_iterations=80,restart=40):
        self.matrix=sparse.csr_matrix(matrix,dtype=float);n=self.matrix.shape[0]
        if self.matrix.shape!=(n,n) or not np.isfinite(self.matrix.data).all():raise ValueError('Finite square coupled matrix required')
        if type(positive_size) is not int or not 0<positive_size<n:raise ValueError('Explicit nonempty positive and negative block sizes required')
        if not math.isfinite(relative_tolerance) or not 0<relative_tolerance<=1e-11:raise ValueError('Explicit relative tolerance at most1e-11 required')
        if type(maximum_iterations) is not int or type(restart) is not int or not 1<=restart<=maximum_iterations:raise ValueError('Finite positive iteration/restart bounds required')
        scale=max(1.,float(np.max(abs(self.matrix.data),initial=0.)))
        if np.max(abs((self.matrix-self.matrix.T).data),initial=0.)>1e-10*scale:raise ValueError('Declared stationary symmetric block structure required')
        self.positive_size=positive_size;self.shape=self.matrix.shape;self.relative_tolerance=relative_tolerance
        self.maximum_iterations=maximum_iterations;self.restart=restart;self.solves=0;self.solve_audits=[];self.preconditioner_actions=0
        p=self.matrix[:positive_size,:positive_size].tocsc();q=-self.matrix[positive_size:,positive_size:].tocsc()
        if np.any(p.diagonal()<=0) or np.any(q.diagonal()<=0):raise ValueError('Positive diagonal blocks required; no diagonal repair is permitted')
        self.positive_factor=splu(p,permc_spec='MMD_AT_PLUS_A');self.negative_factor=splu(q,permc_spec='MMD_AT_PLUS_A')
        self.nnz=int(self.positive_factor.nnz+self.negative_factor.nnz)
        def apply(v):
            self.preconditioner_actions+=1
            return np.r_[self.positive_factor.solve(v[:positive_size]),self.negative_factor.solve(v[positive_size:])]
        self.preconditioner=LinearOperator(self.shape,matvec=apply,dtype=float)
        self.metadata=dict(method=METHOD,equations=n,positive_block_equations=positive_size,negative_block_equations=n-positive_size,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sparse_factorizations=2,full_coupled_factorization=False,column_ordering='MMD_AT_PLUS_A',
            relative_tolerance=relative_tolerance,maximum_iterations=maximum_iterations,restart=restart,
            all_equation_rows_retained=True)

    def solve(self,rhs):
        values=np.asarray(rhs,float);vector=values.ndim==1
        if vector:values=values[:,None]
        if values.ndim!=2 or values.shape[0]!=self.shape[0] or values.shape[1]<1 or not np.isfinite(values).all():raise ValueError('Finite complete nonempty right-hand side required')
        answer=np.empty_like(values)
        for column in range(values.shape[1]):
            b=values[:,column];history=[]
            # legacy makes maxiter count inner iterations, not restart cycles.
            x,info=gmres(self.matrix,b,M=self.preconditioner,rtol=self.relative_tolerance,atol=0.,
                restart=self.restart,maxiter=self.maximum_iterations,callback=lambda value:history.append(float(value)),callback_type='legacy')
            residual=b-self.matrix@x;norm=float(np.linalg.norm(b));error=float(np.linalg.norm(residual))
            accepted=bool(info==0 and np.isfinite(x).all() and error<=self.relative_tolerance*norm)
            audit=dict(solve_batch=self.solves,column=column,iterations=len(history),info=int(info),
                preconditioned_residual_history=history,true_l2_residual=error,rhs_l2_norm=norm,
                maximum_absolute_residual=float(np.max(abs(residual),initial=0.)),
                true_l2_relative_residual=0. if norm==0 and error==0 else error/norm if norm else None,accepted=accepted)
            self.solve_audits.append(audit)
            if not accepted:raise KrylovConvergenceError(audit)
            answer[:,column]=x
        self.solves+=1
        return answer[:,0] if vector else answer

    @property
    def storage_bytes(self):
        total=self.matrix.data.nbytes+self.matrix.indices.nbytes+self.matrix.indptr.nbytes
        for factor in (self.positive_factor,self.negative_factor):
            for matrix in (factor.L,factor.U):total+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        return int(total)
