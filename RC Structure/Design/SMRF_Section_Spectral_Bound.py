"""Conservative simultaneous-action bound for compression-only section concrete.

The implemented no-enhancement material has -fc I <= stress <= 0. For a
dual strain-like tensor W, its enlarged-set support is
fc * sum(max(-eigenvalue(W), 0)). This convex function of W is integrated
from ABOVE: W is affine in transverse coordinate and depth, hence the
bilinear interpolation of its four corner tensors is exact on each rectangle.
Convexity bounds the integral by area times the mean corner support.
This is a geometric upper bound, not a quadrature convergence assertion.

Gross concrete, independent developed +/-fy steel and discarded compatibility
enlarge the capacity set. A separating direction can prove an exceedance of
the declared restricted material model. Absence of separation proves nothing.
No ACI qualification or complete member failure is inferred by this module.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize
from Design.SMRF_Spatial_Floor_Section import _positive_affine_rectangle

METHOD='gross_principal_compression_convex_corner_upper_bound_v1'


def _direction(value):
    a=np.asarray(value,float)
    if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Seven finite dual coefficients required')
    return a


def principal_stress_support_upper_bound(section,direction,*,subdivisions=16):
    """Upper bound in the section's design resultant units, including phi.

    Every finite partition is conservative; more subdivisions only tighten
    its concrete bound. Actual void subtraction is deliberately omitted.
    Developed domains/zero-credit bars remain exactly as declared by section.
    """
    d=_direction(direction);s=section
    if type(subdivisions) is not int or not 1<=subdivisions<=128:
        raise ValueError('Explicit integer convex partition count1..128 required')
    value=0.
    for region in s.regions:
        sa,sb=region['s'];za,zb=region['z']
        if region['id']!='slab':
            value+=s.fc*_positive_affine_rectangle(sa-s.sref,sb-s.sref,za,zb,-d[s.j],-d[6],-d[s.j+3])
            continue
        ss=np.linspace(sa-s.sref,sb-s.sref,subdivisions+1)[:,None]
        zz=np.linspace(za,zb,subdivisions+1)[None,:]
        xx=d[0]+zz*d[3]+(ss*d[6] if s.j==0 else 0.)
        yy=d[1]+zz*d[4]+(ss*d[6] if s.j==1 else 0.)
        xy=d[2]+zz*d[5]
        trace=xx+yy;radius=np.hypot(xx-yy,xy)/2
        support=np.maximum(np.maximum(0.,-trace),radius-trace/2)
        # Positive bilinear weights integrate exactly to one quarter each.
        cell=(support[:-1,:-1]+support[1:,:-1]+support[:-1,1:]+support[1:,1:])/4
        value+=s.fc*(sb-sa)*(zb-za)/subdivisions**2*float(np.sum(cell))
    for bar in s.longitudinal:
        if bar['strength_credited']:
            value+=bar['area_in2']*bar['fy_ksi']*abs(d[s.j]+bar['z_in']*d[s.j+3]+(bar['s_in']-s.sref)*d[6])
    for row in s.transverse:
        run=row['strength_run_interval_in']
        if run is not None:
            value+=row['area_in2']*row['count_per_length_in']*(run[1]-run[0])*row['fy_ksi']*abs(d[s.other]+row['z_in']*d[s.other+3])
    result=float(s.phi*value/s.width)
    if not math.isfinite(result):raise ValueError('Finite support bound required')
    return result


def principal_capacity_separation_certificate(section,factored_resultants,*,maximum_iterations=120,initial_directions=()):
    """Bounded dual search followed by independent finer upper-bound evaluation.

    The optimizer supplies a direction only. Its status never proves failure
    or adequacy. Up to three distinct starts may be supplied; each is clipped
    in dimensionless coordinates and every candidate is checked again.
    """
    target=_direction(factored_resultants);s=section
    if type(maximum_iterations) is not int or not 1<=maximum_iterations<=500:
        raise ValueError('Finite explicit dual-search iteration bound required')
    if len(initial_directions)>3:raise ValueError('At most three explicit additional dual starts required')
    depth=max(s.h,max(abs(z) for r in s.regions for z in r['z']))
    scale=np.array([1.,1.,1.,1/depth,1/depth,1/depth,1/s.width])
    normalized=target*scale;seed=normalized/max(1.,float(max(abs(normalized))))
    starts=[seed]
    for raw in initial_directions:
        x=_direction(raw)/scale;starts.append(x/max(1.,float(max(abs(x)))))
    def objective(x):
        d=x*scale
        return principal_stress_support_upper_bound(s,d,subdivisions=8)-float(d@target)
    candidates=[];searches=[]
    for initial in starts:
        fit=minimize(objective,initial,method='L-BFGS-B',bounds=[(-1.,1.)]*7,
            options={'maxiter':maximum_iterations,'maxfun':maximum_iterations*40,'ftol':1e-12,'gtol':1e-9,'maxls':30})
        candidates.extend([initial,np.asarray(fit.x,float)])
        searches.append(dict(success=bool(fit.success),iterations=int(fit.nit),evaluations=int(fit.nfev)))
    direction=min(candidates,key=objective)*scale
    upper=principal_stress_support_upper_bound(s,direction,subdivisions=32);demand=float(direction@target)
    allowance=1e-8+1e-9*max(abs(upper),abs(demand))
    return dict(method=METHOD,capacity_exceeded=bool(demand>upper+allowance),direction=direction.tolist(),demand=demand,
        proved_design_upper_bound=upper,floating_point_allowance=allowance,excess=demand-upper,convex_partition_subdivisions=32,
        searches=searches,search_partition_subdivisions=8,failure_inferred_from_optimizer=False,adequacy_inferred_when_no_separation=False,
        concrete_premise='minus_fc_I_le_stress_le_zero_no_biaxial_enhancement',gross_concrete_upper_bound=True,
        actual_developed_steel_domains_retained=True,engineering_qualification_inferred=False)
