"""Signed common cuts for slab, beam web and diaphragm action accounting.

Units: inch, kip. z is upward from the slab midsurface; compression N is
positive and sagging M=sum(N*z) is positive. Plate fields and the C0 line
increment have disjoint ownership. The line's hidden flange/web couple is
RECOVERED, never added again to the total moment. Load-component provenance
is part of the result so incompatible combinations cannot be superposed.

This module supplies equilibrium bookkeeping, not an applicability assertion
or a reinforcement/section-strength approval. ACI318-19 8.4.1.9 / 12.6.3
require the downstream concurrent demand and physical reinforcement checks.
"""
from __future__ import annotations
import hashlib
import json
import math

METHOD = "common_slab_web_diaphragm_section_cut_v1"


def _finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value=float(value)
    if positive and value<=0:raise ValueError(f"{name} must be positive")
    return value


def transport_moment(n_compression_kip, moment_kip_in, from_z_in, to_z_in):
    """Transport the same normal-force resultant to another section datum."""
    n,m,z0,z1=(_finite(v,k) for v,k in zip((n_compression_kip,moment_kip_in,from_z_in,to_z_in),('N','M','from_z','to_z')))
    return m+n*(z0-z1)


def recover_composite_line(*, n_compression_kip, m_slab_midplane_kip_in,
                           web_width_in, beam_depth_in, slab_thickness_in,
                           flange_width_in, web_E_ksi, slab_E_ksi):
    """Recover the C0 line's axial couple with independent web/flange moduli.

    C0 line = web local bending + web/flange axial coupling. The flange's
    local plate-bending stiffness/resultant is NOT in this line operator.
    [N,M0]=[[A,B],[B,D]]*[epsilon0,kappa], with compression strain positive.
    This exactly generalizes the recorded equal-E zero-total-N identity.
    It is not by itself proof of two-way shear lag or cracked compatibility.
    """
    n=_finite(n_compression_kip,'N');m=_finite(m_slab_midplane_kip_in,'M')
    b,h,t,bf,ew,ef=(_finite(v,k,True) for v,k in zip(
        (web_width_in,beam_depth_in,slab_thickness_in,flange_width_in,web_E_ksi,slab_E_ksi),
        ('b','h','t','bf','Ew','Ef')))
    if h<=t or bf<b:raise ValueError('A downstand web and flange at least as wide as the web are required')
    aw=b*(h-t);af=bf*t;iw=b*(h-t)**3/12;zw=-h/2
    a=ew*aw+ef*af;bb=ew*aw*zw;d=ew*iw+ew*aw*zw**2
    condensed=d-bb*bb/a
    kappa=(m-bb*n/a)/condensed;epsilon0=(n-bb*kappa)/a
    nf=ef*af*epsilon0;nw=ew*aw*(epsilon0+zw*kappa);mw=ew*iw*kappa
    couple=zw*nw
    nr=nf+nw-n;mr=mw+couple-m
    if abs(nr)>1e-10*max(abs(n),abs(nf),abs(nw),1.) or abs(mr)>1e-10*max(abs(m),abs(mw),abs(couple),1.):
        raise ValueError('Composite cut equilibrium residual exceeded tolerance')
    return dict(method=METHOD,n_flange_compression_kip=nf,n_web_compression_kip=nw,
                m_web_centroid_kip_in=mw,m_offset_couple_kip_in=couple,
                n_total_compression_kip=n,m_total_slab_midplane_kip_in=m,
                web_centroid_z_in=zw,epsilon_slab_midplane=epsilon0,curvature_per_in=kappa,
                line_condensed_EI_kip_in2=condensed,axial_residual_kip=nr,moment_residual_kip_in=mr,
                ownership='plate local bending excluded; hidden axial couple recovered once')


def compose_cut(components, *, combination_id, reference_z_in=0.):
    """Sum independently owned components of one concurrent physical cut.

    Each row requires unique component_id, common cut_id and combination_id,
    n_compression_kip, m_kip_in, reference_z_in. Duplicate ownership or a
    different combination/cut raises instead of producing plausible numbers.
    """
    if not isinstance(combination_id,str) or not combination_id:raise ValueError('Combination identity is required')
    if not components:raise ValueError('Explicit cut components are required, including zero actions')
    ids=set();cuts=set();n_total=m_total=0.;rows=[]
    for row in components:
        cid=row['component_id']
        if not isinstance(cid,str) or not cid or cid in ids:raise ValueError('Missing or duplicate physical component ownership')
        if row['combination_id']!=combination_id:raise ValueError('Nonconcurrent load combinations cannot be added')
        if not isinstance(row['cut_id'],str) or not row['cut_id']:raise ValueError('Physical cut identity is required')
        ids.add(cid);cuts.add(row['cut_id'])
        n=_finite(row['n_compression_kip'],'N')
        m=transport_moment(n,row['m_kip_in'],row['reference_z_in'],reference_z_in)
        rows.append({**row,'transported_m_kip_in':m});n_total+=n;m_total+=m
    if len(cuts)!=1:raise ValueError('Different physical cuts cannot be added')
    record=dict(method=METHOD,cut_id=next(iter(cuts)),combination_id=combination_id,
                reference_z_in=reference_z_in,n_compression_kip=n_total,m_kip_in=m_total,components=rows)
    record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return record


def layer_force_resultants(n_compression_kip_per_ft, m_sagging_kip_in_per_ft, top_z_in, bottom_z_in):
    """Exact two-plane force equilibrium, not a strength or strain assumption.

    The downstream section check must use actual steel/concrete geometry and
    Chapter22 phi/compatibility; division by fy alone does not establish that.
    """
    n,m,zt,zb=(_finite(v,k) for v,k in zip((n_compression_kip_per_ft,m_sagging_kip_in_per_ft,top_z_in,bottom_z_in),('N','M','zt','zb')))
    if zt<=zb:raise ValueError('Distinct ordered physical mat elevations required')
    ft=(m-n*zb)/(zt-zb);fb=n-ft
    return dict(top_compression_kip_per_ft=ft,bottom_compression_kip_per_ft=fb,
                top_tension_kip_per_ft=max(-ft,0.),bottom_tension_kip_per_ft=max(-fb,0.),
                force_residual=ft+fb-n,moment_residual=ft*zt+fb*zb-m)


def verify_reinforcement_ownership(rows):
    """Require each actual bar/mat segment to have one owner in a physical cut."""
    ids=set();areas={}
    for r in rows:
        identity=r['physical_segment_id'];owner=r['owner']
        if not isinstance(identity,str) or not identity or identity in ids:raise ValueError('Reinforcement was omitted or credited more than once')
        if owner not in ('slab_mat','beam_bar_in_flange','beam_bar_in_web','dedicated_collector'):
            raise ValueError('Unknown physical reinforcement owner')
        ids.add(identity);areas[owner]=areas.get(owner,0.)+_finite(r['area_in2'],'area',True)
    return dict(segments=len(ids),area_by_owner_in2=areas,total_area_in2=sum(areas.values()))
