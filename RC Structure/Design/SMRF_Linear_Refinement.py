"""Bounded residual correction with the original sparse factorization.

SuperLU Users' Guide (August 2011), sections 1.3.6--1.3.7, gives the
residual/correction iteration. Here the complete, ungauged equations choose
the retained iterate; solving only the free rows cannot hide gauge imbalance.
This improves arithmetic, not a structural model or its acceptance tolerance.
"""
from __future__ import annotations
import numpy as np

METHOD = 'full_equation_best_residual_refinement_v1'


def refine_gauged_rhs(operator, rhs, free_rows, initial, solve, *, maximum_steps=3):
    """Return a free-coordinate solution; never worsen a column's residual.

    operator contains ALL equation rows and only free unknown columns. rhs
    includes every row, including gauge constraints omitted in factorization.
    The unchanged full-equation infinity residual selects each column. A
    finite step budget/stagnation leaves the caller's original check in force.
    No forward-error or exact-arithmetic residual bound is claimed.
    """
    b=np.asarray(rhs,float);x=np.asarray(initial,float).copy();free=np.asarray(free_rows)
    if (b.ndim!=2 or x.ndim!=2 or operator.shape!=(b.shape[0],x.shape[0]) or
            x.shape[1]!=b.shape[1] or free.shape!=(x.shape[0],) or
            not np.issubdtype(free.dtype,np.integer) or len(set(free.tolist()))!=len(free) or
            np.any(free<0) or np.any(free>=b.shape[0]) or
            not np.isfinite(b).all() or not np.isfinite(x).all()):
        raise ValueError('Finite consistent complete equations and unique free rows required')
    if type(maximum_steps) is not int or not 0<=maximum_steps<=3:
        raise ValueError('Residual correction permits at most three steps')
    scale=np.maximum(1.,np.max(abs(b),axis=0,initial=0.))
    residual=b-operator@x
    if not np.isfinite(residual).all():raise ValueError('Nonfinite initial full-equation residual')
    errors=np.max(abs(residual),axis=0,initial=0.);initial_errors=errors.copy()
    # This is a work-saving trigger, not the caller's acceptance criterion.
    trigger=128*np.finfo(float).eps
    active=np.flatnonzero(errors/scale>trigger);history=[];corrections=0
    for step in range(maximum_steps):
        if not len(active):break
        change=np.asarray(solve(residual[free,:][:,active]),float);corrections+=1
        if change.shape!=(len(free),len(active)) or not np.isfinite(change).all():
            raise ValueError('Invalid finite residual correction')
        trial=x[:,active]+change;candidate=b[:,active]-operator@trial
        if not np.isfinite(candidate).all():raise ValueError('Nonfinite corrected full-equation residual')
        new=np.max(abs(candidate),axis=0,initial=0.);improved=new<errors[active]
        retained=active[improved]
        history.append(dict(step=step+1,attempted_columns=active.tolist(),
                            accepted_columns=retained.tolist()))
        x[:,retained]=trial[:,improved];residual[:,retained]=candidate[:,improved]
        errors[retained]=new[improved]
        # Do not spend additional solves on a column that has stagnated.
        active=retained[errors[retained]/scale[retained]>trigger]
    return x,dict(method=METHOD,maximum_steps=maximum_steps,correction_solves=corrections,
        trigger_relative_residual=trigger,initial_absolute_residuals=initial_errors.tolist(),
        final_absolute_residuals=errors.tolist(),rhs_scales=scale.tolist(),history=history,
        all_equation_rows_included=True,forward_error_bound_claimed=False)
