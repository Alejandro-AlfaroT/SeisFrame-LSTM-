"""Sparse stationary form of the existing cracked composite membrane energy.

For physical floor coordinates u=(m,b), gross membrane matrix M and A=M_mm,
the existing reduced energy scales C=M_bb-M_bm A+ M_mb by beta, preserving
gross A and M_mb. Introduce q=m+A+ M_mb b and use the stationary functional

 beta/2 u'Mu + 1/2 u'Ou + (1-beta)(q'M_mu - 1/2 q'Aq).

Eliminating q gives exactly that existing physical energy. O contains the
separately specified plate/web bending, shear and torsion terms. The augmented
matrix is intentionally indefinite; q is a numerical auxiliary, not extra
physical material or a new column restraint. Explicit mean constraints choose
only the declared null-space gauge of q. No penalty or diagonal regularization
is used. This algebraic alternative does not qualify a floor engineering model.
"""
from __future__ import annotations
import json
import math
import numpy as np
from scipy import sparse

METHOD='stationary_mixed_composite_membrane_energy_v2_explicit_coordinates_unqualified'


class MixedMembraneEnergy:
    def __init__(self,membrane,other,physical_dof_ids,membrane_indices,*,
                 beam_bending_factor,rigid_modes,input_identity,tolerance=1e-9,
                 auxiliary_coordinates='stationary_membrane'):
        beta=float(beam_bending_factor)
        if not math.isfinite(beta) or not 0<beta<=1:raise ValueError('Beam factor must be in(0,1]')
        if not input_identity:raise ValueError('Explicit physical/source identity required')
        if auxiliary_coordinates not in ('stationary_membrane','relaxation_shift'):raise ValueError('Explicit supported auxiliary coordinate basis required')
        self.coordinate_basis=auxiliary_coordinates
        if not math.isfinite(tolerance) or not 0<tolerance<=1e-9:raise ValueError('Positive tolerance at most1e-9 required')
        self.beta=beta;self.weight=1-beta;self.physical_dof_ids=list(physical_dof_ids)
        self.n=len(self.physical_dof_ids);self.m=np.asarray(membrane_indices)
        if self.m.ndim!=1 or self.m.dtype.kind not in 'iu' or len(self.m)==0 or len(set(self.m.tolist()))!=len(self.m) or np.any(self.m<0) or np.any(self.m>=self.n):
            raise ValueError('Distinct actual membrane DOF indices required')
        self.m=self.m.astype(int);self.M=sparse.csr_matrix(membrane,dtype=float);self.O=sparse.csr_matrix(other,dtype=float)
        for name,matrix in [('membrane',self.M),('other',self.O)]:
            if matrix.shape!=(self.n,self.n) or not np.isfinite(matrix.data).all():raise ValueError('Finite physical '+name+' matrix required')
            scale=max(1.,float(np.max(abs(matrix.data),initial=0.)))
            if np.max(abs((matrix-matrix.T).data),initial=0.)>1e-10*scale:raise ValueError('Physical '+name+' matrix must be symmetric')
        self.A=self.M[self.m,:][:,self.m].tocsr();self.C=self.M[:,self.m].tocsr();self.L=self.M[self.m,:].tocsr()
        rigid=np.asarray(rigid_modes,float)
        if rigid.ndim!=2 or rigid.shape[0]!=len(self.m) or not np.isfinite(rigid).all():raise ValueError('Explicit finite membrane rigid modes required')
        if rigid.shape[1]:
            if np.linalg.matrix_rank(rigid)!=rigid.shape[1]:raise ValueError('Independent rigid modes required')
            rigid=np.linalg.qr(rigid,mode='reduced')[0]
        self.R=rigid;scale=max(1.,float(np.max(abs(self.M.data),initial=0.)))
        defect=float(np.max(abs(self.C@rigid),initial=0.))/scale
        if defect>tolerance:raise ValueError('Declared auxiliary gauges are not physical rigid modes')
        if auxiliary_coordinates=='stationary_membrane':self.physical_block=(beta*self.M+self.O).tocsr()
        else:
            # Set q=m+z analytically before assembly. The m-z blocks cancel,
            # leaving only eccentric b-z coupling. The free gauge of z may
            # be fixed to zero mean independently of the gauge chosen for q.
            # This preserves A and B; only the physical bb block is beta*D.
            is_m=np.zeros(self.n,dtype=bool);is_m[self.m]=True
            physical=self.M.tocoo(copy=True)
            physical.data[~is_m[physical.row]&~is_m[physical.col]]*=beta
            self.physical_block=(physical.tocsr()+self.O).tocsr()
            c=self.C.tocoo();keep=~is_m[c.row]
            self.C=sparse.coo_matrix((c.data[keep],(c.row[keep],c.col[keep])),shape=c.shape).tocsr()
            lower=self.L.tocoo();keep=~is_m[lower.col]
            self.L=sparse.coo_matrix((lower.data[keep],(lower.row[keep],lower.col[keep])),shape=lower.shape).tocsr()
        self.dof_ids=list(self.physical_dof_ids)
        if self.weight:
            self.dof_ids += [('membrane_aux:'+json.dumps(self.physical_dof_ids[i],separators=(',',':')),0) for i in self.m]
            self.dof_ids += [('membrane_mean_constraint',i) for i in range(rigid.shape[1])]
            q=sparse.csr_matrix(rigid);ng=rigid.shape[1]
            self.stiffness=sparse.bmat([[self.physical_block,self.weight*self.C,sparse.csr_matrix((self.n,ng))],
                [self.weight*self.L,-self.weight*self.A,q],
                [sparse.csr_matrix((ng,self.n)),q.T,sparse.csr_matrix((ng,ng))]],format='csr')
        else:self.stiffness=self.physical_block
        tokens=[json.dumps(d,sort_keys=True) for d in self.dof_ids]
        if len(set(tokens))!=len(tokens):raise ValueError('Distinct physical and auxiliary DOF identities required')
        self.metadata=dict(method=METHOD,input_identity=input_identity,beam_bending_factor=beta,
            auxiliary_coordinate_basis=auxiliary_coordinates,physical_auxiliary_coupling_nnz=int(self.C.nnz),
            physical_dofs=self.n,auxiliary_membrane_coordinates=len(self.m) if self.weight else 0,
            auxiliary_mean_constraints=rigid.shape[1] if self.weight else 0,
            declared_rigid_mode_relative_defect=defect,explicit_condensed_couple_formed=False,
            auxiliary_coordinates_are_physical=False,qualification_asserted=False)

    def loads(self,physical):
        values=np.asarray(physical,float)
        if values.ndim not in (1,2) or values.shape[0]!=self.n or not np.isfinite(values).all():raise ValueError('Finite physical load vector/matrix required')
        return np.concatenate((values,np.zeros((len(self.dof_ids)-self.n,)+values.shape[1:])),axis=0)

    def coordinates(self,augmented):
        values=np.asarray(augmented,float)
        if values.shape!=(len(self.dof_ids),) or not np.isfinite(values).all():raise ValueError('Finite complete augmented displacement vector required')
        physical=values[:self.n]
        if not self.weight:return physical,(np.zeros(len(self.m)) if self.coordinate_basis=='relaxation_shift' else physical[self.m]),np.zeros(0)
        return physical,values[self.n:self.n+len(self.m)],values[self.n+len(self.m):]

    def physical_action(self,augmented):
        u,q,_=self.coordinates(augmented)
        return self.physical_block@u+self.weight*(self.C@q)

    def fields(self,augmented):
        u,q,_=self.coordinates(augmented);membrane=np.zeros_like(u);membrane[self.m]=q
        if self.coordinate_basis=='relaxation_shift':membrane[self.m]+=u[self.m]
        return membrane,u-membrane

    def audit(self,augmented):
        u,q,lagrange=self.coordinates(augmented)
        if not self.weight:return dict(auxiliary_stationarity_relative_error=0.,mean_constraint_relative_error=0.,unsubtracted_gauge_reaction_max=0.)
        left=self.L@u;right=self.A@q;reaction=self.R@lagrange
        residual=self.weight*(left-right)+reaction
        scale=max(1.,float(np.max(abs(left))),float(np.max(abs(right))))*self.weight
        return dict(auxiliary_stationarity_relative_error=float(np.max(abs(residual),initial=0.))/max(1.,scale),
            mean_constraint_relative_error=float(np.max(abs(self.R.T@q),initial=0.))/max(1.,float(np.max(abs(q),initial=0.))),
            unconstrained_membrane_equilibrium_max=float(np.max(abs(left-right),initial=0.)),
            unsubtracted_gauge_reaction_max=float(np.max(abs(reaction),initial=0.)))
