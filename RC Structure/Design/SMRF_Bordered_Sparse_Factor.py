"""Exact block solve for a sparse core and a small explicit equation border.

Dense gauge rows/columns can cause avoidable fill in a general sparse LU.
Factor the caller's core C and then the small Schur matrix D-B C^-1 E.
All border equations are retained; no penalty, discarded residual, support or
null-space projection is introduced. The caller still checks the full system
residual and chooses physically justified constraints before using this solver.
"""
from __future__ import annotations
import numpy as np
import hashlib
from pathlib import Path
from scipy import sparse
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse.linalg import splu

METHOD='exact_small_border_sparse_core_factor_v1'


class BorderedSparseFactor:
    def __init__(self,matrix,border_indices,*,maximum_border=12):
        a=sparse.csc_matrix(matrix,dtype=float)
        if a.shape[0]!=a.shape[1] or not np.isfinite(a.data).all():raise ValueError('Finite square matrix required')
        border=np.asarray(border_indices)
        if type(maximum_border) is not int or maximum_border<1:raise ValueError('Positive explicit border limit required')
        if border.ndim!=1 or border.dtype.kind not in 'iu' or not 0<len(border)<=maximum_border or len(set(border.tolist()))!=len(border) or np.any(border<0) or np.any(border>=a.shape[0]):
            raise ValueError('Distinct bounded actual border indices required')
        self.border=border.astype(int);self.core=np.setdiff1d(np.arange(a.shape[0]),self.border)
        if not len(self.core):raise ValueError('A nonempty sparse core is required')
        self.shape=a.shape;self.factor=splu(a[self.core,:][:,self.core].tocsc())
        self.lower=a[self.border,:][:,self.core].tocsr()
        self.influence=self.factor.solve(a[self.core,:][:,self.border].toarray())
        self.schur=a[self.border,:][:,self.border].toarray()-self.lower@self.influence
        # LAPACK's small dense pivoting is independent of the sparse ordering.
        # Reject a singular border explicitly; never use a pseudoinverse.
        self.small_factor=lu_factor(self.schur)
        if not np.isfinite(self.small_factor[0]).all() or np.any(np.diag(self.small_factor[0])==0.):
            raise ValueError('Singular bordered system')
        self.nnz=int(self.factor.nnz+self.schur.size);self.solves=0
        self.metadata=dict(method=METHOD,total_equations=a.shape[0],core_equations=len(self.core),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            border_indices=self.border.tolist(),border_equations=len(self.border),
            core_factor_nnz=int(self.factor.nnz),border_condition_number=float(np.linalg.cond(self.schur)),
            all_equation_rows_retained=True)
        if hasattr(self.factor,'metadata'):self.metadata['core_solver']=dict(self.factor.metadata)

    def solve(self,rhs):
        values=np.asarray(rhs,float);vector=values.ndim==1
        if vector:values=values[:,None]
        if values.ndim!=2 or values.shape[0]!=self.shape[0] or not np.isfinite(values).all():raise ValueError('Finite complete right-hand side required')
        y=self.factor.solve(values[self.core]);z=lu_solve(self.small_factor,values[self.border]-self.lower@y)
        result=np.empty_like(values);result[self.border]=z;result[self.core]=y-self.influence@z
        self.solves+=1
        if not np.isfinite(result).all():raise ValueError('Nonfinite bordered solution')
        return result[:,0] if vector else result

    @property
    def storage_bytes(self):
        total=self.influence.nbytes+self.schur.nbytes+self.small_factor[0].nbytes+self.small_factor[1].nbytes
        total+=self.core.nbytes+self.border.nbytes
        matrices=[self.lower]
        if hasattr(self.factor,'storage_bytes'):total+=self.factor.storage_bytes
        else:matrices.extend((self.factor.L,self.factor.U))
        for matrix in matrices:
            total+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        return int(total)
