"""Compensated binary64 sparse actions for cancellation-sensitive force sums.

Ogita, Rump and Oishi (2005), Algorithms 3.2/3.3: split each product into
its rounded value and exact error, then sum both with math.fsum. The explicit
operand domain keeps *all* split products away from overflow and underflow.
This changes arithmetic only; it does not change an operator or its tolerance.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from scipy import sparse

METHOD = 'binary64_two_product_fsum_sparse_action_v1'
_LOWER = 2.0 ** -450
_UPPER = 2.0 ** 450


def _products(a, b):
    active = (a != 0) & (b != 0)
    if np.any(active & ((abs(a) < _LOWER) | (abs(a) > _UPPER)
                        | (abs(b) < _LOWER) | (abs(b) > _UPPER))):
        raise FloatingPointError('Accurate sparse action operand outside verified binary64 domain')
    # Zero products need no split; this also avoids splitting a huge coefficient
    # multiplying zero. Nonzero operands have at most 53 significand bits, so
    # the smallest nonzero partial product is >= 2**(-450-52)*2**(-450-52).
    aa = np.where(active, a, 0.)
    bb = np.where(active, b, 0.)
    p = aa * bb
    ca = 134217729. * aa
    cb = 134217729. * bb
    ah = ca - (ca - aa)
    al = aa - ah
    bh = cb - (cb - bb)
    bl = bb - bh
    error = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, error


def accurate_sparse_action(matrix, vector):
    """Return A*x using compensated products and sums; A must already be CSR.

    Requiring CSR avoids an implicit, ordinary-precision duplicate assembly.
    Existing duplicate CSR entries are summed accurately in their physical row.
    Finite real binary64 operands are required; each nonzero product operand
    must lie in [2**-450, 2**450] in magnitude. Numerical-domain failure raises
    FloatingPointError, never a structural-capacity failure or a relaxed check.
    """
    if not sparse.issparse(matrix) or matrix.format != 'csr':
        raise ValueError('Explicit preassembled CSR physical operator required')
    if np.iscomplexobj(matrix.data) or np.iscomplexobj(vector):
        raise ValueError('Real physical operator and displacement required')
    x = np.asarray(vector, dtype=float)
    if x.shape != (matrix.shape[1],) or not np.isfinite(x).all() or not np.isfinite(matrix.data).all():
        raise ValueError('Finite physical operator and complete displacement required')
    data = np.asarray(matrix.data, dtype=float)
    result = np.empty(matrix.shape[0])
    for first in range(0, matrix.shape[0], 16384):
        last = min(first + 16384, matrix.shape[0])
        start, stop = matrix.indptr[first], matrix.indptr[last]
        p, error = _products(data[start:stop], x[matrix.indices[start:stop]])
        for row in range(first, last):
            lo, hi = matrix.indptr[row] - start, matrix.indptr[row + 1] - start
            try:
                result[row] = math.fsum(itertools.chain(p[lo:hi], error[lo:hi]))
            except OverflowError as exc:
                raise FloatingPointError('Accurate physical force sum overflow') from exc
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite accurate physical force')
    return result
