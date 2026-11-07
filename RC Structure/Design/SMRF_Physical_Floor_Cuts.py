"""Recover physical slab/web cuts without adding or reallocating actions.

Common reference is the slab midplane; N is tension positive and M=int(z*sigma).
ShellMITC4 raw [Nxx,Nyy,Nxy,Mxx,Myy,Mxy,Qx,Qy] uses this moment convention
(its raw moments are the negatives of physical sagging moments). The Gauss
order is (-,-),(+,-),(+,+),(-,+). Each adjacent element trace is retained at a
mesh line. A stress trace is not asserted to be equilibrated merely because
the finite element nodal residual is small; trace jumps/refinement are reported.

Web end forces are global resisting forces ON the element nodes. They include
element fixed-end loads. Interior force recovery uses the declared uniform
gravity load, so moments are quadratic and shear linear between mesh nodes.
There is no effective-flange capacity or reinforcement credit in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

METHOD='physical_slab_web_common_cut_recovery_v1'
_A=1/math.sqrt(3)
_GP=np.array([[-_A,-_A],[_A,-_A],[_A,_A],[-_A,_A]])
_V=np.column_stack((np.ones(4),_GP[:,0],_GP[:,1],_GP[:,0]*_GP[:,1]))
_VINV=np.linalg.inv(_V)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _bounds(row):
    p=np.asarray(row['positions_in'],float)
    if p.shape!=(4,3) or not np.isfinite(p).all():raise ValueError('Four finite physical slab corners required')
    x0,x1,y0,y1=p[0,0],p[1,0],p[0,1],p[3,1]
    if x1<=x0 or y1<=y0 or not np.allclose(p[:,:2],[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],rtol=0,atol=1e-9):
        raise ValueError('Aligned physical slab rectangle required')
    return x0,x1,y0,y1


def slab_tensor(row,x_in,y_in):
    """Exact bilinear interpolation/extrapolation of the element GP field.

Extrapolation is allowed only as far as this element's own physical boundary.
No smoothing across a support, material change or adjacent element is performed.
"""
    x0,x1,y0,y1=_bounds(row);x,y=float(x_in),float(y_in)
    if not(x0-1e-9<=x<=x1+1e-9 and y0-1e-9<=y<=y1+1e-9):
        raise ValueError('Requested field point lies outside its actual physical element')
    values=np.asarray(row['raw_resultants'],float)
    if values.shape!=(4,8) or not np.isfinite(values).all():raise ValueError('Four finite eight-resultant shell fields required')
    xi=2*(x-x0)/(x1-x0)-1;eta=2*(y-y0)/(y1-y0)-1
    return np.array([1.,xi,eta,xi*eta])@_VINV@values


def slab_line_integral(elements,*,axis,station_in,interval_in,trace):
    """Integrate all eight fields over one disjoint slab trace.

axis is the cut normal. trace='minus' selects elements to the lower-coordinate
side when station is a mesh line; 'plus' selects the other side. At a physical
outside face only the interior trace exists. Every requested inch must be
covered exactly once; unsupported or duplicate coverage is an input error.
"""
    if axis not in ('x','y') or trace not in ('minus','plus'):raise ValueError('Explicit axis and one-sided trace required')
    lo,hi=map(float,interval_in);s=float(station_in)
    if not all(math.isfinite(v) for v in (lo,hi,s)) or hi<=lo:raise ValueError('Positive finite cut interval required')
    pieces=[]
    for row in elements:
        if row['kind']!='slab':continue
        x0,x1,y0,y1=_bounds(row)
        a,b,c,d=(x0,x1,y0,y1) if axis=='x' else (y0,y1,x0,x1)
        inside=a+1e-9<s<b-1e-9
        edge=(abs(s-b)<=1e-9 if trace=='minus' else abs(s-a)<=1e-9)
        if not(inside or edge):continue
        u,v=max(lo,c),min(hi,d)
        if v-u<=1e-9:continue
        midpoint=(u+v)/2
        tensor=slab_tensor(row,s,midpoint) if axis=='x' else slab_tensor(row,midpoint,s)
        # The transverse first moment is quadratic even when the tensor is
        # bilinear. Two-point Gauss integrates it exactly, independently of
        # any subsequent effective-flange/section interpretation.
        first=np.zeros(8)
        for xi in (-_A,_A):
            at=midpoint+xi*(v-u)/2
            value=slab_tensor(row,s,at) if axis=='x' else slab_tensor(row,at,s)
            first+=at*value*(v-u)/2
        pieces.append({'element_id':row['id'],'interval_in':[u,v],
                       'integral':(tensor*(v-u)).tolist(),'transverse_first_moment':first.tolist()})
    pieces.sort(key=lambda p:p['interval_in'][0]);cursor=lo
    for p in pieces:
        a,b=p['interval_in']
        if abs(a-cursor)>1e-8:raise ValueError('Slab cut contains a gap or overlapping physical elements')
        cursor=b
    if abs(cursor-hi)>1e-8:raise ValueError('Slab cut does not cover its entire declared physical interval')
    return {'axis':axis,'station_in':s,'interval_in':[lo,hi],'trace':trace,'pieces':pieces,
            'integral':np.sum([p['integral'] for p in pieces],axis=0).tolist(),
            'transverse_first_moment':np.sum([p['transverse_first_moment'] for p in pieces],axis=0).tolist()}


def web_section(row,station_in,*,uniform_vertical_load_kip_in):
    """Exact line-element N/M/V/T about its actual web centroid.

station_in is global X/Y, not member fraction. No distributed axial load is
allowed by this physical-floor element contract. Both end axial forces and
the independent quadratic end-moment reconstruction must agree.
"""
    axis=row['axis'];p=np.asarray(row['positions_in'],float);f=np.asarray(row['global_resisting_force'],float)
    if axis not in ('x','y') or p.shape!=(2,3) or f.shape!=(12,) or not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError('Finite aligned physical web end state required')
    a=0 if axis=='x' else 1;start,end=p[:,a];length=end-start;t=float(station_in)-start
    if length<=0 or not -1e-9<=t<=length+1e-9:raise ValueError('Web cut lies outside its actual positive-axis segment')
    q=float(uniform_vertical_load_kip_in)
    if not math.isfinite(q):raise ValueError('Finite signed physical web gravity load required')
    # Beam X: Mxx=-My_i, +My_j. Beam Y: Myy=+Mx_i,-Mx_j.
    rot=4 if axis=='x' else 3;sign=-1. if axis=='x' else 1.
    n0,n1=-f[a],f[6+a]
    m0,m1=sign*f[rot],-sign*f[6+rot]
    m=m0-f[2]*t-q*t*t/2
    across=1-a;weak_sign=1. if axis=='x' else -1.
    ms0,ms1=weak_sign*f[5],-weak_sign*f[11]
    ms=ms0-f[across]*t
    end_error=np.array([n1-n0,m0-f[2]*length-q*length*length/2-m1,f[2]+f[8]+q*length,
                        ms0-f[across]*length-ms1,f[across]+f[6+across],f[3+a]+f[9+a]])
    scales=np.array([max(1.,abs(n0),abs(n1)),max(1.,abs(m0),abs(m1),abs(f[2]*length)),
                     max(1.,abs(f[2]),abs(f[8]),abs(q*length)),
                     max(1.,abs(ms0),abs(ms1),abs(f[across]*length)),max(1.,abs(f[across]),abs(f[6+across])),
                     max(1.,abs(f[3+a]),abs(f[9+a]))])
    if np.max(abs(end_error)/scales)>1e-8:raise ValueError('Web cut loads/end forces do not satisfy physical element equilibrium')
    return {'member_id':row['member_id'],'element_id':row['id'],'axis':axis,'station_in':float(station_in),
            'N_tension_kip':float(n0),'M_centroid_kipin':float(m),'Q_vertical_kip':float(f[2]+q*t),
            'Ms_transverse_first_moment_kipin':float(ms),'transverse_reference_in':float(p[0,across]),
            'V_inplane_kip':float(-f[across]),
            'T_about_positive_axis_kipin':float(-f[3+a]),'centroid_z_in':float(p[0,2]),
            'end_equilibrium_residual':end_error.tolist(),'end_equilibrium_scale':scales.tolist()}


def common_strip_cut(response,*,axis,station_in,interval_in,trace,web_id=None,
                     uniform_web_loads=None,state_identity,combination_id,location_id,
                     recovery_id):
    """Compose actual slab-strip and optional disjoint web about one reference.

The result contains computable equilibrium evidence. Spatial strain/strength
projection and complete floor coverage are left pending, never switched on by
this algebra routine. Callers must establish those through their actual model
and mesh witnesses before the shared-strength adapter can return pass.
"""
    elements=response['elements'];lo,hi=map(float,interval_in)
    slab=slab_line_integral(elements,axis=axis,station_in=station_in,interval_in=interval_in,trace=trace)
    resultants=np.array(slab['integral'][:6]);web=None
    transverse_reference=(lo+hi)/2
    j=0 if axis=='x' else 1
    ms=slab['transverse_first_moment'][j]-transverse_reference*slab['integral'][j]
    if web_id is not None:
        candidates=[]
        for e in elements:
            if e['kind']!='web' or str(e['member_id'])!=str(web_id):continue
            if e['axis']!=axis:raise ValueError('Common strip web must run along its section normal')
            p=np.asarray(e['positions_in']);a=0 if axis=='x' else 1;start,end=p[:,a]
            if start+1e-9<station_in<end-1e-9 or (abs(station_in-(end if trace=='minus' else start))<=1e-9):
                candidates.append(e)
        if len(candidates)!=1:raise ValueError('Common cut needs exactly one physical web trace outside its joint')
        e=candidates[0];a=0 if axis=='x' else 1;across=1-a;line=e['positions_in'][0][across]
        width=e['section']['b_in']
        if line-width/2<lo-1e-9 or line+width/2>hi+1e-9:raise ValueError('Declared common strip does not contain its entire physical web')
        if uniform_web_loads is None or e['id'] not in uniform_web_loads:raise ValueError('Actual web load identity required at every physical cut')
        web=web_section(e,station_in,uniform_vertical_load_kip_in=uniform_web_loads[e['id']])
        resultants[j]+=web['N_tension_kip']
        resultants[j+3]+=web['M_centroid_kipin']+web['centroid_z_in']*web['N_tension_kip']
        ms+=web['Ms_transverse_first_moment_kipin']+(web['transverse_reference_in']-transverse_reference)*web['N_tension_kip']
    ident=dict(state_identity=str(state_identity),combination_id=str(combination_id),
               location_id=str(location_id),recovery_id=str(recovery_id))
    if any(not v for v in ident.values()):raise ValueError('Actual common-cut identities required')
    residual=response.get('force_partition_relative_error')
    if residual is None or not math.isfinite(residual):raise ValueError('Actual physical nodal force partition evidence required')
    account=dict(method='disjoint_actual_fields',evidence_id=identity({'slab':slab,'web':web,'identity':ident}),
                 equilibrium_residual=[residual],equilibrium_tolerance=[1e-9],
                 common_width_strain_projection_verified=False,core_remote_coverage_verified=False)
    return dict(method=METHOD,identity=ident,source_identities=[ident],reference_z_in=0.,units='integrated_strip',
                cut_width_in=hi-lo,resultants=resultants.tolist(),slab=slab,web=web,accounting=account,
                transverse_reference_in=transverse_reference,Ms_transverse_first_moment_kipin=float(ms),
                transverse_shear_integrals=slab['integral'][6:],qualification_asserted=False)


def disjoint_transverse_partition(outer_interval,web_intervals):
    """Full physical coverage by declared common beam strips and remote slab."""
    lo,hi=map(float,outer_interval)
    if not(math.isfinite(lo) and math.isfinite(hi) and hi>lo):raise ValueError('Finite positive floor cut width required')
    result=[];cursor=lo;seen=set()
    for name,a,b in sorted(web_intervals,key=lambda r:r[1]):
        a,b=float(a),float(b)
        if name in seen or not(lo<=a<b<=hi) or a<cursor-1e-9:raise ValueError('Physical beam strips overlap, duplicate or exceed the slab')
        seen.add(name)
        if a>cursor+1e-9:result.append(dict(kind='remote_slab',interval_in=[cursor,a]))
        result.append(dict(kind='common_beam_strip',member_id=name,interval_in=[a,b]));cursor=b
    if cursor<hi-1e-9:result.append(dict(kind='remote_slab',interval_in=[cursor,hi]))
    if not result:result=[dict(kind='remote_slab',interval_in=[lo,hi])]
    return {'method':METHOD,'interval_in':[lo,hi],'pieces':result,
            'coverage_residual_in':sum(p['interval_in'][1]-p['interval_in'][0] for p in result)-(hi-lo)}


def bar_clear_interval(required_interval,available_interval,bar_families):
    """Smallest enlargement of a web band whose boundaries cut no mat bar.

Families contain actual center coordinates and db, all transverse to the cut.
Only a boundary is moved, past the far face of every bar it intersects. No bar
phase, field, web width or effective flange is altered. Final adjacent strip
overlap must still be checked by disjoint_transverse_partition.
"""
    low,high=map(float,required_interval);outer_low,outer_high=map(float,available_interval)
    if not outer_low<=low<high<=outer_high:raise ValueError('Physical web interval must lie within available slab')
    circles=[]
    for family in bar_families:
        r=float(family['db_in'])/2
        if not math.isfinite(r) or r<=0:raise ValueError('Positive physical bar radius required')
        circles.extend((float(c)-r,float(c)+r) for c in family['global_transverse_coordinates_in'])
    for _ in range(len(circles)+1):
        old=(low,high)
        for a,b in circles:
            if a+1e-9<low<b-1e-9:low=a-1e-8
            if a+1e-9<high<b-1e-9:high=b+1e-8
        if old==(low,high):break
    if low<outer_low-1e-9 or high>outer_high+1e-9:raise ValueError('A physical mat bar crosses the slab/accounting outer boundary')
    return [low,high]
