"""Support free-body recovery of slab punching resultants; no qualification.

The support reaction and actual beam cuts fix the net slab V/Mx/My by static
equilibrium. This is distinct from the discontinuous raw shell stress trace.
It does not determine the local stress field or replace a mesh convergence
check. ACI318-19 8.4.4 uses the shear and unbalanced moment at the connection;
the subsequent gamma/Jc check remains in SMRF_Slab_Local_Shear.

The caller supplies every physical distributed load and the actual net joint
wrench ON the floor (including any concentrated load at that same joint).
No additional out-of-plane concentrated loads may be silently omitted.
"""
from __future__ import annotations

import math
import numpy as np
from Design.SMRF_Physical_Floor_Cuts import _bounds,web_section

METHOD='slab_support_free_body_resultants_v1_unqualified'


def _transport(force,moment,point,reference):
    f=np.asarray(force,float);m=np.asarray(moment,float)
    return np.r_[f,m+np.cross(np.asarray(point,float)-reference,f)]


def recover_support_free_body(response,perimeter,*,support_wrench_kip_kipin,
                              pressure_by_element,web_load_by_element,trace,
                              additional_point_wrenches=()):
    """Exact loaded rectangle balance with the actual disjoint beam webs.

    Both traces are explicit when a web cut coincides with its FE node. The
    pressure is piecewise uniform on the original slab cells; web loads are
    uniform on original clear-span elements. Slab footprint clipping is already
    present in the perimeter. Wrenches use global axes at the slab midplane.
    """
    if trace not in ('inside','outside'):raise ValueError('Explicit physical cut trace required')
    elements=response['elements'];slabs=[r for r in elements if r['kind']=='slab'];webs=[r for r in elements if r['kind']=='web']
    if len({r['id'] for r in elements})!=len(elements):raise ValueError('Unique physical element inventory required')
    if set(pressure_by_element)!={r['id'] for r in slabs} or set(web_load_by_element)!={r['id'] for r in webs}:
        raise ValueError('Every actual slab pressure and clear-web load required')
    if any(not math.isfinite(float(v)) for v in [*pressure_by_element.values(),*web_load_by_element.values()]):
        raise ValueError('Finite signed distributed loads required')
    x0,x1,y0,y1=perimeter['bounds_in'];cx,cy=perimeter['centroid_in'];reference=np.array([cx,cy,0.])
    sx,sy=perimeter['column_center_in'];joint=np.asarray(support_wrench_kip_kipin,float)
    if joint.shape!=(6,) or not np.isfinite(joint).all():raise ValueError('Actual finite six-component support wrench required')
    support=_transport(joint[:3],joint[3:],[sx,sy,0.],reference)
    body=np.zeros(6);area=0.;body_parts=[]
    for row in slabs:
        a,b,c,d=_bounds(row);lo,hi=max(x0,a),min(x1,b);low,high=max(y0,c),min(y1,d)
        if hi<=lo or high<=low:continue
        ar=(hi-lo)*(high-low);area+=ar;q=float(pressure_by_element[row['id']])
        wrench=_transport([0.,0.,q*ar],[0.,0.,0.],[(lo+hi)/2,(low+high)/2,0.],reference)
        body+=wrench;body_parts.append(dict(element_id=row['id'],area_in2=ar,wrench=wrench.tolist()))
    if not math.isclose(area,(x1-x0)*(y1-y0),rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Slab cells do not cover the free-body rectangle')
    for row in webs:
        axis=row['axis'];a=0 if axis=='x' else 1;p=np.asarray(row['positions_in'],float)
        lower,upper=(x0,x1) if axis=='x' else (y0,y1);crosslo,crosshi=(y0,y1) if axis=='x' else (x0,x1)
        if not crosslo+1e-9<p[0,1-a]<crosshi-1e-9:continue
        lo,hi=max(lower,p[0,a]),min(upper,p[1,a])
        if hi<=lo:continue
        point=p[0].copy();point[a]=(lo+hi)/2;q=float(web_load_by_element[row['id']])
        wrench=_transport([0.,0.,q*(hi-lo)],[0.,0.,0.],point,reference)
        body+=wrench;body_parts.append(dict(element_id=row['id'],length_in=hi-lo,wrench=wrench.tolist()))
    points=np.zeros(6)
    for row in additional_point_wrenches:
        p=np.asarray(row['point_in'],float);w=np.asarray(row['wrench_kip_kipin'],float)
        if p.shape!=(3,) or w.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(w).all():raise ValueError('Finite explicit point wrench required')
        if not x0<=p[0]<=x1 or not y0<=p[1]<=y1:raise ValueError('Point wrench lies outside the declared body')
        points+=_transport(w[:3],w[3:],p,reference)
    crossing=np.zeros(6);cuts=[]
    for side in perimeter['sides']:
        axis=side['axis'];a=0 if axis=='x' else 1;station=side['station_in'];lo,hi=side['interval_in'];normal=side['outward_normal'][a]
        direction=('minus' if normal>0 else 'plus') if trace=='inside' else ('plus' if normal>0 else 'minus')
        seen=set()
        for row in webs:
            if row['axis']!=axis:continue
            p=np.asarray(row['positions_in'],float);start,end=p[:,a]
            if not lo+1e-9<p[0,1-a]<hi-1e-9:continue
            selected=start+1e-9<station<end-1e-9 or abs(station-(end if direction=='minus' else start))<=1e-9
            if not selected:continue
            if row['member_id'] in seen:raise ValueError('Duplicate web crossing on one trace')
            seen.add(row['member_id']);cut=web_section(row,station,uniform_vertical_load_kip_in=web_load_by_element[row['id']])
            n,v,q,m,t,ms=(cut[k] for k in ('N_tension_kip','V_inplane_kip','Q_vertical_kip','M_centroid_kipin','T_about_positive_axis_kipin','Ms_transverse_first_moment_kipin'))
            force=[n,v,-q] if axis=='x' else [v,n,-q]
            moment=[t,m,-ms] if axis=='x' else [-m,t,ms]
            point=p[0].copy();point[a]=station
            wrench=normal*_transport(force,moment,point,reference);crossing+=wrench
            cuts.append(dict(side=side,element_id=row['id'],section=cut,wrench=wrench.tolist()))
    total=-(support+body+points);slab=total-crossing
    return dict(method=METHOD,trace=trace,reference_xyz_in=reference.tolist(),
        support_wrench=support.tolist(),distributed_load_wrench=body.tolist(),point_load_wrench=points.tolist(),
        complete_boundary_wrench=total.tolist(),web_boundary_wrench=crossing.tolist(),slab_boundary_wrench=slab.tolist(),
        slab_resultant_kip_kipin=slab[[2,3,4]].tolist(),body_parts=body_parts,web_cuts=cuts,
        balance_residual=(support+body+points+crossing+slab).tolist(),qualification_asserted=False)
