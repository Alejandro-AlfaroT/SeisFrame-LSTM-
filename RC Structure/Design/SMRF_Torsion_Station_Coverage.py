"""Polynomial extrema of concurrent guard forces, including threshold domains.

These are demand sampling certificates, not continuous nonlinear material or
mesh qualification. No threshold capacity is extrapolated across an N domain
boundary. Exact physical/P5 mesh and installed hoop transitions split intervals.
"""
from __future__ import annotations
import math
import numpy as np

METHOD='quadratic_guard_force_threshold_station_coverage_v1'
FIELDS=('N_tension_kip','T_about_positive_axis_kipin','Q_vertical_kip','V_inplane_kip')


def _roots(coeff):
    coeff=np.asarray(coeff,float)
    # Trim only roundoff relative to this polynomial, not a force threshold.
    scale=max(abs(coeff),default=0.)
    while len(coeff)>1 and abs(coeff[-1])<=1e-12*scale:coeff=coeff[:-1]
    if len(coeff)<=1:return []
    return [float(z.real) for z in np.polynomial.polynomial.polyroots(coeff)
            if abs(z.imag)<=1e-9 and 1e-10<z.real<1-1e-10]


def polynomial_points(values,section):
    """All scalar extrema, T zeroes and extrema of T²/(1-N/k), degree<=2.

    Primary strong moment projection can introduce quadratic axial demand.
    Hence the physical affine-N simplification is deliberately not used here.
    """
    values=np.asarray(values,float)
    if values.shape!=(3,len(FIELDS)) or not np.isfinite(values).all():raise ValueError('Three finite complete guard force samples required')
    coeff=np.linalg.solve(np.array([[1,0,0],[1,.5,.25],[1,1,1]]),values)
    points={0.:{'interval_boundary'},.5:{'interval_midpoint'},1.:{'interval_boundary'}}
    def add(roots,why):
        for x in roots:points.setdefault(x,set()).add(why)
    for c in coeff.T:add(_roots(np.polynomial.polynomial.polyder(c)),'guard_scalar_stationary_point')
    n,t=coeff[:,0],coeff[:,1]
    scale=4*section['Ag_in2']*section['lambda']*min(100.,math.sqrt(section['fc_ksi']*1000))/1000
    r=np.array([1.,0.,0.])-n/scale
    add(_roots(r),'guard_threshold_axial_formula_domain_boundary')
    add(_roots(t),'guard_torque_zero')
    poly=np.polynomial.polynomial
    derivative=poly.polysub(2*poly.polymul(poly.polyder(t),r),poly.polymul(t,poly.polyder(r)))
    add([x for x in _roots(derivative) if poly.polyval(x,r)>0.],'guard_concurrent_threshold_ratio_stationary_point')
    return coeff,[(x,sorted(points[x])) for x in sorted(points)]


def _rows(plan):
    result={}
    for d in plan['demands']:
        parts=d['guard_cut_id'].split('|exact_threshold',1)
        if len(parts)!=2:raise ValueError('Exact threshold core identity required')
        key=(parts[1],d['kind'])
        if key in result:raise ValueError('Duplicate same-cut demand guard')
        result[key]=[d[k] for k in FIELDS]
    if not result:raise ValueError('At least one physical guard required')
    return result


def interval_points(provider,section):
    """provider(normalized station,trace) returns independently recovered guards."""
    samples=[_rows(provider(x,'minus' if x==1 else 'plus')) for x in (0.,.5,1.)]
    # An irrational interior witness cannot coincide with the fitting stations.
    at=1/math.sqrt(7);witness=_rows(provider(at,'plus'))
    keys=set(samples[0])
    if any(set(s)!=keys for s in samples[1:]+[witness]):raise ValueError('Guard inventory changes inside a declared force interval')
    points={};audits=[]
    for key in sorted(keys):
        coeff,locations=polynomial_points([s[key] for s in samples],section)
        pred=np.array([1,at,at*at])@coeff;actual=np.asarray(witness[key]);residual=pred-actual
        if np.any(abs(residual)>1e-8+1e-9*np.maximum(abs(pred),abs(actual))):raise ValueError('Guard force failed independent polynomial interval witness')
        for x,reasons in locations:points.setdefault(x,set()).update(reasons)
        audits.append(dict(guard_suffix=key[0],kind=key[1],coefficients=coeff.tolist(),independent_fraction=at,
            independent_residual=residual.tolist()))
    return [(x,sorted(points[x])) for x in sorted(points)],audits


def detail_boundaries(detail,clear_span_in,baseline_spacing_in):
    """All actual changes used by transverse_variants plus physical zone edges."""
    local=detail.get('splice_detail') or {};positions=local.get('hoop_stations_in') or []
    if not positions:return []
    zone=(local.get('transverse_detail') or {}).get('zone_in') or []
    values=[positions[0]-baseline_spacing_in,positions[0],positions[-1],positions[-1]+baseline_spacing_in,*zone]
    if any(not math.isfinite(float(x)) for x in values):raise ValueError('Finite actual torsion detail boundaries required')
    return sorted({float(x) for x in values if 0<float(x)<clear_span_in})
