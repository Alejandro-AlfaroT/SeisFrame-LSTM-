"""High/low binary64 arithmetic for physical force and displacement recovery.

Ogita, Rump and Oishi (2005), SISC26(6), Algorithms3.1/3.3 supply error-free
sum/product transforms. Each pair represents high+low; never discard low
before a cancellation-sensitive action. Pair updates are rounded to two
components, not arbitrary precision. No structural qualification is implied.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from scipy import sparse
from Design.SMRF_Accurate_Sparse_Action import _products

METHOD='two_component_binary64_physical_arithmetic_v1'

def _finite(*values):
    if any(np.iscomplexobj(v) for v in values):raise ValueError('Real paired operands required')
    arrays=np.broadcast_arrays(*(np.asarray(v,dtype=float) for v in values))
    if any(not np.isfinite(a).all() for a in arrays):raise ValueError('Finite paired operands required')
    return arrays

def two_sum(a,b):
    """Error-free sum of finite binary64 operands, unless the sum overflows."""
    a,b=_finite(a,b)
    with np.errstate(over='raise',invalid='raise'):
        high=a+b;v=high-a;low=(a-(high-v))+(b-v)
    return high,low

def add_pair(high,low,increment_high,increment_low=0.):
    """Add two represented values and retain two components of the result."""
    high,low,increment_high,increment_low=_finite(high,low,increment_high,increment_low)
    for change in (increment_high,increment_low):
        high,error=two_sum(high,change)
        high,low=two_sum(high,error+low)
    return high,low

def scale_pair(high,low,factor):
    high,low,factor=_finite(high,low,factor)
    ph,pl=_products(high,factor);qh,ql=_products(low,factor)
    return add_pair(ph,pl,qh,ql)

def _vector_pair(high,low,size):
    if np.iscomplexobj(high) or np.iscomplexobj(low):raise ValueError('Real vector pair required')
    high,low=np.asarray(high,float),np.asarray(low,float)
    if high.shape!=(size,) or low.shape!=(size,) or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError('Finite complete vector pair required')
    return high,low

def _sum_pair(*parts):
    # First sum is rounded binary64. A second accurate sum includes its exact
    # negation, retaining the part lost by that first rounding.
    values=list(itertools.chain.from_iterable(parts))
    try:
        high=math.fsum(values);low=math.fsum(itertools.chain(values,(-high,)))
    except OverflowError as exc:raise FloatingPointError('Paired physical force overflow') from exc
    if not math.isfinite(high) or not math.isfinite(low):raise FloatingPointError('Nonfinite paired physical force')
    return high,low

def paired_dense_action(matrix,high,low):
    """Small physical transforms/operators; retain product and sum errors."""
    if np.iscomplexobj(matrix):raise ValueError('Real dense operator required')
    a=np.asarray(matrix,float)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Finite rectangular dense operator required')
    high,low=_vector_pair(high,low,a.shape[1])
    ph,pl=_products(a,np.broadcast_to(high,a.shape));qh,ql=_products(a,np.broadcast_to(low,a.shape))
    rh=np.empty(a.shape[0]);rl=np.empty_like(rh)
    for row in range(a.shape[0]):rh[row],rl[row]=_sum_pair(ph[row],pl[row],qh[row],ql[row])
    return rh,rl

def paired_sparse_action(matrix,high,low):
    """CSR physical action; retain duplicate entries and both vector parts."""
    if not sparse.issparse(matrix) or matrix.format!='csr' or np.iscomplexobj(matrix.data):
        raise ValueError('Explicit real CSR physical operator required')
    high,low=_vector_pair(high,low,matrix.shape[1])
    if not np.isfinite(matrix.data).all():raise ValueError('Finite physical operator required')
    rh=np.empty(matrix.shape[0]);rl=np.empty_like(rh)
    for first in range(0,matrix.shape[0],16384):
        last=min(first+16384,matrix.shape[0]);start,stop=matrix.indptr[first],matrix.indptr[last]
        data=matrix.data[start:stop];indices=matrix.indices[start:stop]
        ph,pl=_products(data,high[indices]);qh,ql=_products(data,low[indices])
        for row in range(first,last):
            a,b=matrix.indptr[row]-start,matrix.indptr[row+1]-start
            rh[row],rl[row]=_sum_pair(ph[a:b],pl[a:b],qh[a:b],ql[a:b])
    return rh,rl
