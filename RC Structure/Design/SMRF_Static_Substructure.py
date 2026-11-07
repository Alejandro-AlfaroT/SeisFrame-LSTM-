"""Exact linear static substructuring on explicitly identified physical DOFs.

No structural model or support assumptions are created here. The caller
supplies a symmetric assembled stiffness, load vectors, retained boundary
DOFs and (when needed) an explicit affine kinematic map. Block elimination is
Kc=Kbb-Kbi solve(Kii,Kib), fc=fb-Kbi solve(Kii,fi), and
ui=solve(Kii,fi-Kib ub). A singular interior is refused; no penalty, weak
spring, pseudoinverse or dropped coupling repairs it silently.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Block_Diagonal_Krylov import KrylovConvergenceError

METHOD='exact_sparse_static_schur_complement_v1'


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sparse_hash(matrix):
    matrix=sparse.csr_matrix(matrix,dtype=np.float64,copy=True)
    matrix.sum_duplicates();matrix.sort_indices();matrix.eliminate_zeros()
    digest=hashlib.sha256()
    digest.update(np.asarray(matrix.shape,dtype='<i8').tobytes())
    for value,dtype in ((matrix.indptr,'<i8'),(matrix.indices,'<i8'),(matrix.data,'<f8')):
        digest.update(np.asarray(value,dtype=dtype).tobytes())
    return digest.hexdigest()


def _dof_inventory(dofs):
    values=list(dofs)
    tokens=[json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False) for d in values]
    if not values or len(tokens)!=len(set(tokens)):raise ValueError('A nonempty unique physical DOF inventory is required')
    return values,tokens


def _rhs(values,size,label):
    array=np.asarray(values,dtype=float)
    vector=array.ndim==1
    if vector:array=array[:,None]
    if array.ndim!=2 or array.shape[0]!=size or array.shape[1]<1 or not np.isfinite(array).all():
        raise ValueError(f'{label} must be finite with shape ({size},) or ({size},number_of_load_cases)')
    return array.copy(),vector


def _symmetric(matrix,tolerance):
    matrix=sparse.csr_matrix(matrix,dtype=float,copy=True)
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    if matrix.shape[0]!=matrix.shape[1] or not np.isfinite(matrix.data).all():raise ValueError('Finite square stiffness required')
    scale=max(1.,float(np.max(np.abs(matrix.data),initial=0.)))
    difference=matrix-matrix.T
    error=float(np.max(np.abs(difference.data),initial=0.))
    if error>tolerance*scale:raise ValueError('Stiffness is not symmetric within the declared relative tolerance')
    # Do not symmetrize away a physical coupling. Retain the exact supplied
    # matrix; its measured numerical asymmetry remains in the report.
    return matrix,error/scale


class StaticSubstructure:
    """One cached sparse interior factorization, with arbitrary load RHSs."""
    def __init__(self,stiffness,dof_ids,boundary_dofs,loads,*,input_identity,
                 symmetry_tolerance=1e-10,solve_tolerance=1e-9,block_size=64,interior_border_dofs=()):
        start=time.perf_counter()
        if not input_identity:raise ValueError('Assembly/source input identity is mandatory')
        if not all(math.isfinite(v) and v>0 for v in (symmetry_tolerance,solve_tolerance)):
            raise ValueError('Positive finite numerical tolerances are required')
        if type(block_size) is not int or block_size<1:raise ValueError('Positive integer factor solve block size required')
        self.dof_ids,tokens=_dof_inventory(dof_ids)
        self.boundary_dofs,boundary_tokens=_dof_inventory(boundary_dofs)
        inventory={token:i for i,token in enumerate(tokens)}
        if not set(boundary_tokens)<=set(tokens):raise ValueError('A retained boundary DOF is absent from the physical inventory')
        self._k,self.symmetry_residual=_symmetric(stiffness,symmetry_tolerance)
        if self._k.shape[0]!=len(tokens):raise ValueError('Stiffness size and physical DOF inventory differ')
        self._b=np.asarray([inventory[token] for token in boundary_tokens],dtype=int)
        kept=set(self._b);self._i=np.asarray([i for i in range(len(tokens)) if i not in kept],dtype=int)
        self.interior_dofs=[self.dof_ids[i] for i in self._i]
        border=list(interior_border_dofs)
        border_indices=None
        if border:
            _,border_tokens=_dof_inventory(border)
            interior_tokens={tokens[i]:j for j,i in enumerate(self._i)}
            if not set(border_tokens)<=set(interior_tokens):raise ValueError('Border equations must be actual interior DOFs')
            border_indices=np.array([interior_tokens[t] for t in border_tokens])
        self._f,self._vector=_rhs(loads,len(tokens),'Loads')
        self._kib=self._k[self._i,:][:,self._b].tocsr()
        self._kbi=self._k[self._b,:][:,self._i].tocsr()
        self._kii=self._k[self._i,:][:,self._i].tocsc()
        self._factor=None
        self._solve_tolerance=solve_tolerance
        self._factor_solves=0
        self._maximum_factor_residual=0.
        self._maximum_initial_factor_residual=0.
        self._residual_corrections=0
        if len(self._i):
            try:
                if border_indices is None:self._factor=splu(self._kii)
                else:
                    from Design.SMRF_Bordered_Sparse_Factor import BorderedSparseFactor
                    self._factor=BorderedSparseFactor(self._kii,border_indices)
            except KrylovConvergenceError:
                # Border setup solves its core influences. An exhausted bounded
                # solve is not evidence of a singular structural stiffness.
                raise
            except RuntimeError as exc:raise ValueError('Interior stiffness is singular; explicit boundary/support DOFs are insufficient') from exc
        condensed=self._k[self._b,:][:,self._b].toarray()
        for start_col in range(0,len(self._b),block_size):
            stop=min(start_col+block_size,len(self._b))
            response=self._solve(self._kib[:,start_col:stop].toarray())
            condensed[:,start_col:stop]-=self._kbi@response
        self._kc=condensed
        scale=max(1.,float(np.max(np.abs(condensed),initial=0.)))
        self.condensed_symmetry_residual=float(np.max(np.abs(condensed-condensed.T),initial=0.))/scale
        if self.condensed_symmetry_residual>max(symmetry_tolerance,10*solve_tolerance):
            raise ValueError('Condensed stiffness lost symmetry beyond numerical tolerance')
        self._fc=self._condense_rhs(self._f)
        self._input_identity=json.loads(json.dumps(input_identity,allow_nan=False))
        self._signature={'method':METHOD,'input_identity':self._input_identity,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_dof_sha256':_json_hash(self.dof_ids),'boundary_dof_sha256':_json_hash(self.boundary_dofs),
            'assembled_stiffness_sha256':_sparse_hash(self._k),'loads_sha256':_json_hash(self._f.tolist()),
            'condensed_stiffness_sha256':_sparse_hash(self._kc),'condensed_loads_sha256':_json_hash(self._fc.tolist())}
        if border:
            self._signature['interior_factor_method']=self._factor.metadata['method']
            self._signature['interior_factor_source_sha256']=self._factor.metadata['source_sha256']
            self._signature['interior_border_dofs']=border
            if 'core_solver' in self._factor.metadata:self._signature['interior_core_solver']=self._factor.metadata['core_solver']
        self.elapsed_seconds=time.perf_counter()-start

    def _solve(self,rhs):
        if not len(self._i):return np.zeros_like(rhs)
        answer=self._factor.solve(np.asarray(rhs,order='F'))
        self._factor_solves+=1
        # Backward error: exact block equation, scaled by actual matrix and
        # vector magnitudes, not by an arbitrary model-size absolute bound.
        history=[]
        for attempt in range(4):
            residual=self._kii@answer-rhs
            denominator=np.asarray(abs(self._kii)@np.abs(answer))+np.abs(rhs)
            error=float(np.max(np.abs(residual)/np.maximum(1.,denominator),initial=0.))
            history.append(error)
            if not np.isfinite(answer).all() or error<=self._solve_tolerance or attempt==3:break
            # Bounded same-factor iterative refinement solves the original
            # equation. It changes neither K nor the acceptance tolerance.
            answer-=self._factor.solve(np.asarray(residual,order='F'))
            self._factor_solves+=1;self._residual_corrections+=1
        self._maximum_initial_factor_residual=max(self._maximum_initial_factor_residual,history[0])
        self._maximum_factor_residual=max(self._maximum_factor_residual,error)
        if not np.isfinite(answer).all() or error>self._solve_tolerance:
            raise ValueError(f'Interior factor solve failed its equation residual check: {history}; tolerance={self._solve_tolerance}')
        return answer

    @property
    def stiffness(self):return self._kc.copy()

    @property
    def loads(self):return self._fc[:,0].copy() if self._vector else self._fc.copy()

    def _condense_rhs(self,loads):
        return loads[self._b]-self._kbi@self._solve(loads[self._i])

    def condense_loads(self,loads):
        values,vector=_rhs(loads,len(self.dof_ids),'Loads')
        answer=self._condense_rhs(values)
        return answer[:,0] if vector else answer

    def recover(self,boundary_displacements,loads=None):
        ub,vector=_rhs(boundary_displacements,len(self.boundary_dofs),'Boundary displacements')
        f=self._f if loads is None else _rhs(loads,len(self.dof_ids),'Loads')[0]
        if ub.shape[1]!=f.shape[1]:raise ValueError('Boundary displacement and load case counts differ')
        full=np.empty((len(self.dof_ids),ub.shape[1]));full[self._b]=ub
        full[self._i]=self._solve(f[self._i]-self._kib@ub)
        return full[:,0] if vector else full

    def residuals(self,boundary_displacements,loads=None):
        """Independent full equation, boundary reaction and energy identities."""
        ub,_=_rhs(boundary_displacements,len(self.boundary_dofs),'Boundary displacements')
        f=self._f if loads is None else _rhs(loads,len(self.dof_ids),'Loads')[0]
        full=_rhs(self.recover(ub,loads),len(self.dof_ids),'Recovered displacements')[0]
        residual=self._k@full-f
        boundary=self._kc@ub-self._condense_rhs(f)
        actual_energy=.5*np.sum(full*(self._k@full),axis=0)-np.sum(full*f,axis=0)
        reduced_energy=(.5*np.sum(ub*(self._kc@ub),axis=0)-np.sum(ub*self._condense_rhs(f),axis=0)
                        -.5*np.sum(f[self._i]*self._solve(f[self._i]),axis=0))
        return {'maximum_interior_force_residual':float(np.max(np.abs(residual[self._i]),initial=0.)),
                'maximum_boundary_reaction_residual':float(np.max(np.abs(residual[self._b]-boundary),initial=0.)),
                'maximum_stationary_potential_energy_residual':float(np.max(np.abs(actual_energy-reduced_energy),initial=0.)),
                'boundary_reactions':boundary.tolist(),'full_dof_forces':residual.tolist(),
                'factor_backward_error':self._maximum_factor_residual}

    def record(self):
        factor_bytes=0
        if self._factor is not None:
            if hasattr(self._factor,'storage_bytes'):factor_bytes=self._factor.storage_bytes
            else:
                for matrix in (self._factor.L,self._factor.U):factor_bytes+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        return {**self._signature,'identity_sha256':_json_hash(self._signature),
                'physical_dof_ids':self.dof_ids,'boundary_dof_ids':self.boundary_dofs,'interior_dof_ids':self.interior_dofs,
                'physical_dofs':len(self.dof_ids),'boundary_dofs':len(self.boundary_dofs),'interior_dofs':len(self.interior_dofs),
                'factorizations':int(self._factor is not None),'factor_solve_batches':self._factor_solves,
                'maximum_factor_backward_error':self._maximum_factor_residual,
                'maximum_initial_factor_backward_error':self._maximum_initial_factor_residual,
                'same_factor_residual_corrections':self._residual_corrections,
                'assembled_symmetry_residual':self.symmetry_residual,'condensed_symmetry_residual':self.condensed_symmetry_residual,
                'factor_storage_bytes':factor_bytes,'dense_boundary_storage_bytes':self._kc.nbytes,
                'elapsed_seconds':self.elapsed_seconds}


def condense(stiffness,dof_ids,boundary_dofs,loads,*,input_identity,**options):
    return StaticSubstructure(stiffness,dof_ids,boundary_dofs,loads,input_identity=input_identity,**options)


class AffineKinematics:
    """Explicit caller-supplied u=Tq+u0, including rigid/support constraints."""
    def __init__(self,stiffness,loads,physical_dofs,reduced_dofs,transform,*,lift=None,input_identity,
                 symmetry_tolerance=1e-10):
        self.physical_dofs,_=_dof_inventory(physical_dofs);self.reduced_dofs,_=_dof_inventory(reduced_dofs)
        self._k,_=_symmetric(stiffness,symmetry_tolerance)
        self._f,self._vector=_rhs(loads,len(self.physical_dofs),'Physical loads')
        self._t=sparse.csr_matrix(transform,dtype=float,copy=True)
        if (self._t.shape!=(len(self.physical_dofs),len(self.reduced_dofs)) or not np.isfinite(self._t.data).all()
                or self._k.shape!=(len(self.physical_dofs),len(self.physical_dofs))):raise ValueError('Explicit affine map dimensions/values differ')
        if np.asarray(abs(self._t).sum(axis=0)).ravel().min(initial=1.)<=0:raise ValueError('Affine map has an unsupported zero reduced DOF column')
        if lift is None:self._lift=np.zeros_like(self._f)
        else:
            self._lift,_=_rhs(lift,len(self.physical_dofs),'Prescribed support lift')
            if self._lift.shape[1]==1 and self._f.shape[1]>1:self._lift=np.repeat(self._lift,self._f.shape[1],axis=1)
            if self._lift.shape!=self._f.shape:raise ValueError('Prescribed support lift and load case counts differ')
        if not input_identity:raise ValueError('Affine map source/input identity is mandatory')
        self.stiffness=(self._t.T@self._k@self._t).tocsr()
        transformed=self._t.T@(self._f-self._k@self._lift)
        self.loads=transformed[:,0].copy() if self._vector else np.asarray(transformed)
        self.identity={'method':'explicit_affine_physical_dof_map_v1','input_identity':input_identity,
                       'physical_dofs':self.physical_dofs,'reduced_dofs':self.reduced_dofs,
                       'transform_sha256':_sparse_hash(self._t),'lift_sha256':_json_hash(self._lift.tolist()),
                       'assembled_stiffness_sha256':_sparse_hash(self._k),'loads_sha256':_json_hash(self._f.tolist())}

    def recover(self,reduced_displacements):
        q,vector=_rhs(reduced_displacements,len(self.reduced_dofs),'Reduced displacements')
        if q.shape[1]!=self._f.shape[1]:raise ValueError('Reduced displacement and load case counts differ')
        u=np.asarray(self._t@q)+self._lift
        return u[:,0] if vector else u

    def forces(self,reduced_displacements):
        u,vector=_rhs(self.recover(reduced_displacements),len(self.physical_dofs),'Physical displacements')
        forces=np.asarray(self._k@u)-self._f
        return forces[:,0] if vector else forces

