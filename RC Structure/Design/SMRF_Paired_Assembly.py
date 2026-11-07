"""Preserve low-order physical coefficients through congruence and assembly.

Both returned CSR matrices belong to one represented operator, high+low.
Neither low coefficients nor component energies may be discarded. This is
arithmetic only: no symmetry repair, stiffness floor or physical zero removal.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from scipy import sparse
from Design.SMRF_Paired_Arithmetic import paired_dense_action,two_sum

METHOD='paired_physical_congruence_and_duplicate_assembly_v1'

def paired_congruence(matrix,transform):
    """Return two components of T.T @ K @ T without premature rounding."""
    if np.iscomplexobj(matrix) or np.iscomplexobj(transform):raise ValueError('Real physical matrices required')
    k,t=np.asarray(matrix,float),np.asarray(transform,float)
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or t.ndim!=2 or t.shape[0]!=len(k) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError('Finite square component and consistent transform required')
    # A signed permutation changes only ordering and signs, exactly. Avoid
    # expensive products for the many unconstrained physical element nodes.
    if t.shape==k.shape and len(k):
        rows,cols=np.nonzero(t)
        if len(rows)==len(k) and np.array_equal(rows,np.arange(len(k))) and len(set(cols.tolist()))==len(k) and np.all(abs(t[rows,cols])==1.):
            order=np.argsort(cols);sign=t[rows,cols][order]
            return k[np.ix_(order,order)]*sign[:,None]*sign[None,:],np.zeros_like(k)
    rh=np.empty((len(k),t.shape[1]));rl=np.empty_like(rh)
    for col in range(t.shape[1]):rh[:,col],rl[:,col]=paired_dense_action(k,t[:,col],np.zeros(len(k)))
    h=np.empty((t.shape[1],t.shape[1]));l=np.empty_like(h)
    for col in range(t.shape[1]):h[:,col],l[:,col]=paired_dense_action(t.T,rh[:,col],rl[:,col])
    return h,l

def paired_triplet_matrix(high,low,rows,columns,*,shape):
    """Accurately combine duplicate physical entries, preserving sum errors."""
    if len(shape)!=2 or any(not isinstance(n,(int,np.integer)) or isinstance(n,(bool,np.bool_)) or n<0 for n in shape):
        raise ValueError('Explicit nonnegative integer matrix shape required')
    if np.iscomplexobj(high) or np.iscomplexobj(low):raise ValueError('Real coefficient pairs required')
    hh,ll=np.asarray(high,float),np.asarray(low,float);rr,cc=np.asarray(rows),np.asarray(columns)
    if any(a.ndim!=1 or a.shape!=hh.shape for a in (hh,ll,rr,cc)) or not np.isfinite(hh).all() or not np.isfinite(ll).all():
        raise ValueError('Consistent finite one-dimensional coefficient pairs required')
    if not len(hh):return sparse.csr_matrix(shape),sparse.csr_matrix(shape)
    if rr.dtype.kind not in 'iu' or cc.dtype.kind not in 'iu' or np.any(rr<0) or np.any(cc<0) or np.any(rr>=shape[0]) or np.any(cc>=shape[1]):
        raise ValueError('Integer coefficient coordinates must lie inside the matrix')
    order=np.lexsort((cc,rr));rr=rr[order];cc=cc[order];hh=hh[order];ll=ll[order]
    first=np.r_[0,1+np.flatnonzero((rr[1:]!=rr[:-1])|(cc[1:]!=cc[:-1]))];stop=np.r_[first[1:],len(hh)]
    oh,ol=two_sum(hh[first],ll[first])
    for i in np.flatnonzero(stop-first>1):
        a,b=first[i],stop[i]
        try:
            oh[i]=math.fsum(itertools.chain(hh[a:b],ll[a:b]));ol[i]=math.fsum(itertools.chain(hh[a:b],ll[a:b],(-oh[i],)))
        except OverflowError as exc:raise FloatingPointError('Physical coefficient sum overflow') from exc
    if not np.isfinite(oh).all() or not np.isfinite(ol).all():raise FloatingPointError('Nonfinite assembled coefficient pair')
    out=[]
    for values in (oh,ol):
        keep=values!=0.
        out.append(sparse.coo_matrix((values[keep],(rr[first[keep]],cc[first[keep]])),shape=shape).tocsr())
    return tuple(out)
