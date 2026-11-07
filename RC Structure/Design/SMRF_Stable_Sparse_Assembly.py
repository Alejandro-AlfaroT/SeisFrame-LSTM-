"""Accurately sum repeated finite-element matrix entries before CSR conversion.

COO permits duplicate coordinates; its conversion adds those contributions.
At rigid support patches, thousands of signed terms can share one coordinate.
Accumulate each group with math.fsum before conversion, retaining every finite
term and every nonzero result. This changes arithmetic, not the supplied local
operators, kinematic constraints, physical supports or acceptance tolerances.
No symmetry projection, stiffness floor or diagonal repair is performed.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import sparse

METHOD='finite_element_duplicate_triplets_fsum_v1'


def accurate_triplet_matrix(data,rows,columns,*,shape):
    if (len(shape)!=2 or any(not isinstance(n,(int,np.integer)) or isinstance(n,(bool,np.bool_)) or n<0 for n in shape)):
        raise ValueError('Explicit nonnegative integer matrix shape required')
    values=np.asarray(data,dtype=float);row=np.asarray(rows);column=np.asarray(columns)
    if (values.ndim!=1 or row.ndim!=1 or column.ndim!=1 or values.shape!=row.shape or values.shape!=column.shape
        or not np.isfinite(values).all()):raise ValueError('Equal finite one-dimensional triplet arrays required')
    if not len(values):return sparse.csr_matrix(shape,dtype=float)
    if row.dtype.kind not in 'iu' or column.dtype.kind not in 'iu':raise ValueError('Integer matrix coordinates required')
    if np.any(row<0) or np.any(column<0) or np.any(row>=shape[0]) or np.any(column>=shape[1]):raise ValueError('Matrix coordinates outside explicit shape')
    order=np.lexsort((column,row));row=row[order];column=column[order];values=values[order]
    first=np.r_[0,1+np.flatnonzero((row[1:]!=row[:-1])|(column[1:]!=column[:-1]))]
    stop=np.r_[first[1:],len(values)];summed=values[first].copy()
    for i in np.flatnonzero(stop-first>1):summed[i]=math.fsum(values[first[i]:stop[i]])
    if not np.isfinite(summed).all():raise ValueError('Nonfinite assembled coefficient')
    keep=summed!=0.
    # Coordinates are now sorted and unique, so CSR performs no further sums.
    return sparse.coo_matrix((summed[keep],(row[first[keep]],column[first[keep]])),shape=shape).tocsr()
