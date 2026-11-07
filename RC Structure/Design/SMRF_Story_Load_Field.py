"""Exact fixed physical section-load polynomials for directional story LPs.

Each piece uses GLOBAL normalized member coordinate s. Compression P is
positive, M is sagging about the gross-depth midplane, V=dM/dx only when
there is no applied side couple. Distinct one-sided traces are preserved.
The fixed field is added to free LP end resultants, not counted as strength.
This algebra validates interpolation; physical trace equilibrium/refinement
and candidate/source binding must be established by the producing adapter.
"""
from __future__ import annotations
import math
import numpy as np

METHOD='piecewise_affine_axial_quadratic_moment_fixed_load_field_v1'


def validate(field):
    if field.get('method')!=METHOD or not field.get('source_identity'):
        raise ValueError('Identified exact physical section-load polynomial method required')
    pieces=field.get('pieces',[]);cursor=0.
    if not pieces:raise ValueError('Complete fixed section-load field required')
    for piece in pieces:
        a,b=float(piece['start_fraction']),float(piece['end_fraction'])
        if not math.isfinite(a+b) or abs(a-cursor)>1e-12 or not a<b<=1.:
            raise ValueError('Section-load pieces must partition the complete member without gaps/overlap')
        for key,size in (('P_offset_kip',2),('M_offset_kipin',3),('V_offset_kip',2)):
            values=np.asarray(piece[key],float)
            if values.shape!=(size,) or not np.isfinite(values).all():
                raise ValueError('Explicit finite affine P/V and quadratic M coefficients required')
        cursor=b
    if abs(cursor-1.)>1e-12:raise ValueError('Section-load field omits a member end')
    if abs(evaluate(field,'M',0.,side='right'))>1e-8 or abs(evaluate(field,'M',1.,side='left'))>1e-8:
        raise ValueError('Fixed moment offsets must vanish at both free LP end resultants')
    return field


def piece_at(field,s,*,side='right'):
    s=float(s)
    if not math.isfinite(s) or not -1e-12<=s<=1.+1e-12 or side not in ('left','right'):
        raise ValueError('Section field requires a fraction in[0,1] and explicit trace side')
    pieces=field['pieces']
    if s<=0.:return pieces[0]
    if s>=1.:return pieces[-1]
    for p in pieces:
        a,b=p['start_fraction'],p['end_fraction']
        if a<s<b or (side=='right' and abs(s-a)<=1e-12) or (side=='left' and abs(s-b)<=1e-12):return p
    raise ValueError('Section field has no physical trace at requested station')


def evaluate(field,component,s,*,side='right'):
    key={'P':'P_offset_kip','M':'M_offset_kipin','V':'V_offset_kip'}[component]
    values=piece_at(field,s,side=side)[key]
    return float(sum(v*float(s)**i for i,v in enumerate(values)))


def breaks(field,left=0.,right=1.):
    return sorted({float(left),float(right),*[float(p['end_fraction']) for p in field['pieces'] if left<p['end_fraction']<right]})


def extrema_samples(field,left,right,mi,mj,halfplanes):
    """All exact extrema of every affine-N/quadratic-M strength facet.

The end-variable contribution to M is (1-s)*Mi-s*Mj. Each facet a*P+b*M
is quadratic on a piece. Its unique stationary point plus BOTH one-sided
piece endpoints are exhaustive. Axial domain extrema are piece endpoints.
"""
    out={(s,side) for s in breaks(field,left,right) for side in
         (('right',) if s==left else ('left',) if s==right else ('left','right'))}
    for p in field['pieces']:
        lo=max(left,p['start_fraction']);hi=min(right,p['end_fraction'])
        if hi<=lo:continue
        p1=p['P_offset_kip'][1];_,m1,m2=p['M_offset_kipin']
        for a,b,_ in halfplanes:
            linear=a*p1+b*(m1-mi-mj);quadratic=b*m2
            if quadratic:
                root=-linear/(2*quadratic)
                if lo<root<hi:out.add((float(root),'right'))
    return sorted(out)


def maximum_violation(field,left,right,p,mi,mj,interaction,bound):
    planes=interaction[bound+'_halfplanes'];low,high=interaction['axial_domain_kip']
    worst=-math.inf;at=None
    for s,side in extrema_samples(field,left,right,mi,mj,planes):
        axial=p+evaluate(field,'P',s,side=side)
        moment=(1-s)*mi-s*mj+evaluate(field,'M',s,side=side)
        violation=max(low-axial,axial-high,*(a*axial+b*moment-c for a,b,c in planes))
        if violation>worst:worst,at=violation,(s,side)
    return float(worst),at
