"""Static nominal story resistance with complete plane-frame equilibrium.

This module is a *strength* calculation, not an ELF demand procedure. A unit
generalized story force is a horizontal force pair at the bounding floors (at
the first story the lower force is a base reaction). All other floor forces
are unchanged. Nodal rotations and vertical motions remain free, so adjacent
stories, roof and base members can participate in the limiting mechanism.
Horizontal translations of nodes on each diaphragm are equal; no arbitrary
nodal distribution of the story force is imposed.

The static LP maximizes that force subject to member N-M strength surfaces,
end/midspan shear and bending limits, joint limits and nodal equilibrium.
Its dual is the corresponding compatible virtual mechanism. A returned
feasible static solution alone is not called an exact section strength: an
inner/outer interaction pair gives a rigorous numerical strength bracket.

Nominal strengths use specified fy and phi=1. Developed slab participation
belongs in the full section strengths. A concurrent-demand steel reservation
must NOT be subtracted from these mechanism or capacity-design strengths.

Physical basis: ASCE/SEI 7-22 Table 12.3-2 (story lateral strength); ACI 318-19
22.2 (nominal section equilibrium/strain compatibility); NISTIR 6867, section
3.16 (sum of column shears limited by controlling action); FEMA 352 section
5.9.6 (plastic mechanism analysis when a single column mechanism is not
established). The last two explain mechanism mechanics, not a code-edition
substitution or nonlinear-performance qualification.

Units: kip, inch. In the plane, x is right and z is up. Each member points
from i to j. P is compression-positive. End moments Mi/Mj act counterclockwise
ON the nodes. For a horizontal beam, section sagging is Mi at i and -Mj at j.
``transverse_load_kip_per_in`` is downward local-y gravity load; an optional
``axial_load_kip_per_in`` acts opposite the local member axis (column weight).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from Design import SMRF_Story_Load_Field as load_field

METHOD = "equilibrated_story_force_pair_nominal_inner_outer_physical_field_v2"
INTERACTION_METHOD = "finite_bar_nominal_lipschitz_inner_outer_v1"


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is not finite in its supported range")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _active_envelope_lines(slopes,intercepts,lower,upper):
    """Exactly redundant halfplanes only; validate at every envelope vertex."""
    ordered=sorted(range(len(slopes)),key=lambda i:(-slopes[i],intercepts[i]))
    hull=[];starts=[]
    for i in ordered:
        if hull and slopes[i]==slopes[hull[-1]]:continue
        while hull:
            cross=(intercepts[i]-intercepts[hull[-1]])/(slopes[hull[-1]]-slopes[i])
            if cross>starts[-1]:break
            hull.pop();starts.pop()
        start=-math.inf if not hull else cross
        hull.append(i);starts.append(start)
    ends=starts[1:]+[math.inf]
    keep=[i for i,a,b in zip(hull,starts,ends) if b>=lower and a<=upper]
    vertices=np.array(sorted({lower,upper,*[v for v in starts if lower<v<upper]}))
    original=np.min(np.asarray(slopes)[:,None]*vertices+np.asarray(intercepts)[:,None],axis=0)
    reduced=np.min(np.asarray(slopes)[keep,None]*vertices+np.asarray(intercepts)[keep,None],axis=0)
    # Both functions are linear between their union of envelope vertices.
    # A failed floating-point equivalence check retains all constraints.
    if np.max(np.abs(original-reduced),initial=0.)>1e-8:return list(range(len(slopes)))
    return keep


def lipschitz_interaction(axial, upper_moment, lower_moment, lipschitz_in, *, sample_error_kip_in=0.0):
    """Inner/outer convex halfplanes from certified Lipschitz section samples.

    ``upper_moment`` bounds +M; ``lower_moment`` bounds -M. The caller must
    supply a justified global Lipschitz constant, not just sampled slopes.
    For each interval, its secant minus L*deltaP/2 and the sample error is
    everywhere below the actual capacity. Intersecting all these halfplanes
    remains below the capacity even if the sampled curve is nonconcave.

    The sampled Lipschitz upper tents give an independently computed maximum
    gap D above that inner envelope. Adding D to every inner intercept makes
    an outer envelope. Therefore nonconcavity cannot silently become an
    unsafe convex hull: it enlarges the explicit bracket. A broad bracket
    must be refined or retained as an uncertainty, not relabelled exact.
    """
    p = np.asarray(axial, dtype=float)
    l = _number(lipschitz_in, "Lipschitz length", positive=True)
    err = _number(sample_error_kip_in, "sample error", nonnegative=True)
    if p.ndim != 1 or len(p) < 3 or not np.isfinite(p).all() or (np.diff(p) <= 0).any():
        raise ValueError("Strictly increasing finite axial samples are required")
    sides = []
    for sign, supplied in ((1, upper_moment), (-1, lower_moment)):
        f = np.asarray(supplied, dtype=float)
        if f.shape != p.shape or not np.isfinite(f).all():
            raise ValueError("Moment sample dimensions differ or contain nonfinite values")
        dp = np.diff(p)
        if (np.abs(np.diff(f)) > l * dp + 2 * err + 1e-7).any():
            raise ValueError("Samples violate the supplied global Lipschitz bound")
        slopes = np.diff(f) / dp
        intercepts = f[:-1] - slopes * p[:-1] - l * dp / 2 - err
        # On each interval the upper tent has only one corner. Its difference
        # from any inner line attains its maximum at an endpoint or that corner.
        cross = np.clip((np.diff(f) + l * dp) / (2 * l), 0., dp)
        witnesses = np.concatenate((p, p[:-1] + cross))
        idx = np.clip(np.searchsorted(p, witnesses, side="right") - 1, 0, len(dp)-1)
        upper = np.minimum(f[idx] + l * (witnesses-p[idx]),
                           f[idx+1] + l * (p[idx+1]-witnesses)) + err
        inner = np.min(slopes[:, None] * witnesses + intercepts[:, None], axis=0)
        gap = max(0., float(np.max(upper-inner)))
        keep=_active_envelope_lines(slopes,intercepts,float(p[0]),float(p[-1]))
        sides.append({"moment_sign": sign, "slopes": slopes[keep].tolist(), "intercepts": intercepts[keep].tolist(),
                      "maximum_moment_gap_kip_in": gap})
    inner_planes = []; outer_planes = []
    for side in sides:
        for slope, intercept in zip(side["slopes"], side["intercepts"]):
            inner_planes.append([-slope, side["moment_sign"], intercept])
            outer_planes.append([-slope, side["moment_sign"], intercept+side["maximum_moment_gap_kip_in"]])
    return {"method": INTERACTION_METHOD, "axial_domain_kip": [float(p[0]), float(p[-1])],
            "inner_halfplanes": inner_planes, "outer_halfplanes": outer_planes,
            "samples": {"axial_kip": p.tolist(), "upper_moment_kip_in": list(map(float, upper_moment)),
                        "lower_moment_kip_in": list(map(float, lower_moment))},
            "lipschitz_in": l, "sample_error_kip_in": err,
            "maximum_moment_gap_kip_in": max(s["maximum_moment_gap_kip_in"] for s in sides),
            "proof": "secant minus L*deltaP/2; global inner intersection; Lipschitz upper-tent gap",
            "nominal": True, "phi": 1.0}


def rectangular_interaction(section, *, axis="h", divisions=128):
    """Certified finite-bar nominal interaction for the actual symmetric cage.

    ``section`` is a SMRF_Biaxial_Section.BiaxialSection. Its returned phi is
    removed before solving nominal axial equilibrium. Every circular bar must
    be contained and disjoint. Then gross-block minus bar areas is precisely
    the area of actual concrete: as c increases every concrete/steel force
    increment is nonnegative. Thus |deltaM| <= depth/2*deltaP, including yield
    changes. This proves the global bound used above; sampled slopes alone
    would not prove it. Tied compression is limited to 0.8P0, without phi.
    """
    if axis not in ("h", "b") or isinstance(divisions, bool) or int(divisions) != divisions or divisions < 8:
        raise ValueError("Use axis h/b and at least eight integer axial intervals")
    bars = np.asarray(section.bars)
    if section.fy >= .003*section.es:
        raise ValueError('Nominal P0 convention requires steel yield strain below the concrete ultimate strain')
    radii = np.sqrt(bars[:, 2]/math.pi)
    if ((np.abs(bars[:, 0])+radii > section.b/2+1e-10).any()
            or (np.abs(bars[:, 1])+radii > section.h/2+1e-10).any()):
        raise ValueError("Finite circular bars must lie entirely inside the concrete section")
    for i in range(len(bars)):
        distances = np.hypot(bars[i+1:, 0]-bars[i, 0], bars[i+1:, 1]-bars[i, 1])
        if (distances < radii[i+1:]+radii[i]-1e-10).any():
            raise ValueError("Overlapping finite bars invalidate the nominal interpolation proof")
    depth = section.h if axis == "h" else section.b
    theta = math.pi/2 if axis == "h" else 0.
    p = np.linspace(-section.fy*section.ast, .8*section.p0, int(divisions)+1)
    lo = np.full(p.shape, depth*1e-12); hi = np.full(p.shape, depth*100.)
    for _ in range(75):
        c = (lo+hi)/2; state = section.state(c, theta)
        pn = state["p"]/state["phi"]
        hi = np.where(pn > p, c, hi); lo = np.where(pn <= p, c, lo)
    state = section.state((lo+hi)/2, theta)
    pn = state["p"]/state["phi"]
    mn = state["mh" if axis == "h" else "mb"]/state["phi"]
    force_error = float(np.max(np.abs(pn-p)))
    if force_error > 1e-8*section.force_scale:
        raise ValueError("Nominal axial curve did not equilibrate")
    result = lipschitz_interaction(p, mn, mn, depth/2, sample_error_kip_in=depth/2*force_error)
    result.update(section_geometry={"b_in": section.b, "h_in": section.h, "fc_ksi": section.fc,
                                    "fy_ksi": section.fy, "es_ksi": section.es, "bars": bars.tolist()},
                  axis=axis, finite_bars_contained_and_disjoint=True,
                  maximum_axial_equilibrium_error_kip=force_error,
                  material_increment_proof="Finite Whitney concrete region minus disjoint contained bar circles is nested as c grows; each EPP steel centroid force is nondecreasing; all moment arms <=depth/2.")
    result["sha256"] = _hash(result)
    return result


def reverse_moment_interaction(interaction):
    """Express a physical sagging-positive curve as the opposite end action."""
    result=dict(interaction)
    for key in ('inner_halfplanes','outer_halfplanes'):
        result[key]=[[a,-b,c] for a,b,c in interaction[key]]
    result.pop('sha256',None)
    result['moment_sign_transform']=-int(interaction.get('moment_sign_transform',1))
    result['sha256']=_hash(result)
    return result


def layered_interaction(section, *, divisions=128):
    """Finite-layer T/L beam curve with a checked global material bound.

    Net concrete force is monotone only if removed bar chords never exceed
    the gross concrete chord. The interval bound below checks that premise
    analytically, including every layer and concrete-region boundary.
    """
    if int(divisions)!=divisions or divisions<8:raise ValueError('At least eight axial intervals required')
    layers=np.asarray(section.layers);r=np.sqrt(layers[:,2]/math.pi)
    if (layers[:,3]>=.003*section.es).any():raise ValueError('Unsupported yield strain for nominal P0')
    breaks=sorted(set([0.,section.h,*section.regions[:,:2].ravel(),*(layers[:,0]-r),*(layers[:,0]+r)]))
    maximum_density=0.
    for left,right in zip(breaks,breaks[1:]):
        mid=(left+right)/2
        width=next(float(b) for y0,y1,b,fc in section.regions if y0<=mid<=y1)
        nearest=np.clip(layers[:,0],left,right)
        chords=2*np.sqrt(np.maximum(0.,r*r-(nearest-layers[:,0])**2))*layers[:,1]/layers[:,2]
        maximum_density=max(maximum_density,float(chords.sum()/width))
    if maximum_density>1.+1e-10:raise ValueError('Aggregate finite-bar chord bound cannot establish nested net concrete')
    p=np.linspace(-section.steel_force,.8*section.p0,int(divisions)+1)
    interval=section.moment_interval(p,phi_mode='nominal')
    err=max(float(np.max(np.abs(interval[s]['axial_residual_kip']))) for s in ('positive','negative'))*section.h/2
    result=lipschitz_interaction(p,interval['maximum'],-interval['minimum'],section.h/2,sample_error_kip_in=err)
    result.update(section_geometry={'depth_in':section.h,'concrete_regions':section.regions.tolist(),'layers':layers.tolist()},
                  maximum_removed_to_gross_chord_bound=maximum_density,
                  material_increment_proof='Every net concrete chord is nonnegative by interval maxima; EPP steel force is monotone; all force arms <=h/2.')
    result['sha256']=_hash(result)
    return result


def _point_loads(member):
    rows=[]
    for position,force in member.get('transverse_point_loads',[]):
        position=_number(position,'point load fraction');force=_number(force,'point transverse load')
        if not 0.<position<1.:raise ValueError('Point loads must lie strictly inside a member')
        rows.append((position,force))
    return rows


def _gravity_moment(member,length,s,*,side='right'):
    """Exact simple-support moment of the declared uniform and point loads."""
    if member.get('fixed_section_field') is not None:return load_field.evaluate(member['fixed_section_field'],'M',s,side=side)
    w=float(member.get('transverse_load_kip_per_in',0.))
    return w*length**2*s*(1-s)/2+sum(p*length*min(s,a)*(1-max(s,a)) for a,p in _point_loads(member))


def _gravity_shear(member,length,s,*,side='right'):
    """Physical sagging-moment derivative, including the selected jump side."""
    if member.get('fixed_section_field') is not None:return load_field.evaluate(member['fixed_section_field'],'V',s,side=side)
    w=float(member.get('transverse_load_kip_per_in',0.))
    return w*length*(.5-s)+sum(p*((1-a)-(1. if a<s or (a==s and side=='right') else 0.)) for a,p in _point_loads(member))


def _axial_shift(member,length,s,*,side='right'):
    if member.get('fixed_section_field') is not None:return load_field.evaluate(member['fixed_section_field'],'P',s,side=side)
    return float(member.get('axial_load_kip_per_in',0.))*length*(.5-s)


class _Rows:
    def __init__(self, n):
        self.n=n; self.rows=[]; self.rhs=[]; self.names=[]
    def add(self, coefficients, rhs, name):
        self.rows.append({i:float(v) for i,v in coefficients.items() if v != 0.})
        self.rhs.append(float(rhs)); self.names.append(name)
    def matrix(self):
        row=[]; col=[]; data=[]
        for j, values in enumerate(self.rows):
            for i,v in values.items():row.append(j); col.append(i); data.append(v)
        return coo_matrix((data,(row,col)),shape=(len(self.rows),self.n)).tocsr()


def _merge(target, source, factor=1.):
    for key,value in source.items():target[key]=target.get(key,0.)+factor*value


def _build(model, story, sway, bound, extra_points):
    nodes=model["nodes"]; members=model["members"]
    if not nodes or not members:raise ValueError("A nonempty actual frame is required")
    by_node={str(n["id"]):n for n in nodes}
    if len(by_node)!=len(nodes):raise ValueError("Duplicate node ids")
    by_member={str(m["id"]):m for m in members}
    if len(by_member)!=len(members):raise ValueError("Duplicate member ids")
    floors=sorted({int(n["floor"]) for n in nodes})
    if floors!=list(range(max(floors)+1)) or story not in floors or story==0:
        raise ValueError("Complete consecutive floor roster including floor 0 is required")
    zlevels={f:{_number(n['z_in'],'node z') for n in nodes if int(n['floor'])==f} for f in floors}
    if any(len(v)!=1 for v in zlevels.values()):raise ValueError('Each diaphragm floor must be horizontal')
    if any(next(iter(zlevels[f]))<=next(iter(zlevels[f-1])) for f in floors[1:]):raise ValueError('Floors must rise in order')
    ndof={}; names=[]
    for floor in floors[1:]:ndof[('floor',floor,'x')]=len(names); names.append(f'floor:{floor}:x')
    for n in nodes:
        node=str(n['id']); floor=int(n['floor'])
        if floor==0:
            if n.get('fixed_base') is not True:raise ValueError('Every base node needs explicit fixed_base=True')
            continue
        for comp in ('z','rotation'):ndof[(node,comp)]=len(names); names.append(f'node:{node}:{comp}')
    envelope_members=[str(m['id']) for m in members if m['kind']=='beam' and float(m.get('axial_load_kip_per_in',0.))==0. and m.get('fixed_section_field') is None]
    envelope_indices={mid:(3*len(members)+2*k,3*len(members)+2*k+1) for k,mid in enumerate(envelope_members)}
    nvar=3*len(members)+2*len(envelope_members)+1; lam=nvar-1
    eq=_Rows(nvar); inequalities=_Rows(nvar)
    dof_rows=[{} for _ in names]; loads=np.zeros(len(names))
    for n in nodes:
        node=str(n['id']); floor=int(n['floor'])
        if floor==0:continue
        vector=n.get('load',{})
        loads[ndof[('floor',floor,'x')]]+=_number(vector.get('x_kip',0.),'node x load')
        loads[ndof[(node,'z')]]+=_number(vector.get('z_kip',0.),'node z load')
        loads[ndof[(node,'rotation')]]+=_number(vector.get('moment_kip_in',0.),'node moment load')
    dof_rows[ndof[('floor',story,'x')]][lam]=float(sway)
    if story>1:dof_rows[ndof[('floor',story-1,'x')]][lam]=-float(sway)
    index={}; geometry={}; constants={}; bounds=[]; incident=defaultdict(list)
    for j,m in enumerate(members):
        mid=str(m['id']); ni,nj=by_node[str(m['i'])],by_node[str(m['j'])]
        dx=_number(nj['x_in'],'x')-_number(ni['x_in'],'x'); dz=_number(nj['z_in'],'z')-_number(ni['z_in'],'z')
        length=math.hypot(dx,dz)
        if length<=0:raise ValueError('Zero member length')
        cx,cz=dx/length,dz/length; sy,sz=-cz,cx
        if m['kind'] not in ('beam','column'):raise ValueError('Member kind must be beam or column')
        if (m['kind']=='beam' and (abs(cz)>1e-10 or cx<=0 or ni['floor']!=nj['floor'])) or (m['kind']=='column' and (abs(cx)>1e-10 or cz<=0 or int(nj['floor'])!=int(ni['floor'])+1)):
            raise ValueError('Members must run left-to-right within a floor or bottom-to-top through one story')
        ip,im,ij=3*j,3*j+1,3*j+2; index[mid]={'P':ip,'Mi':im,'Mj':ij}
        gravity=_number(m.get('transverse_load_kip_per_in',0.),'uniform transverse load')
        axial=_number(m.get('axial_load_kip_per_in',0.),'uniform axial load')
        field=m.get('fixed_section_field')
        if field is not None:
            load_field.validate(field)
            if m['kind']!='beam' or axial or gravity or _point_loads(m):
                raise ValueError('A fixed physical beam field must own all distributed actions exactly once')
        if m['kind']=='beam' and axial!=0.:raise ValueError('The supported beam has constant axial force; distributed axial loading needs its own extrema proof')
        if m['kind']=='column' and (gravity!=0. or _point_loads(m)):raise ValueError('Column transverse distributed loads are outside the end-controlled column model')
        shear=_number(m['nominal_shear_kip'],'nominal shear strength',positive=True)
        if bound=='outer' and 'outer_nominal_shear_kip' in m:
            shear=(math.inf if m['outer_nominal_shear_kip'] is None else _number(m['outer_nominal_shear_kip'],'outer nominal shear strength',positive=True))
        geometry[mid]=(length,cx,cz); constants[mid]=(gravity,axial)
        reference=m.get('capacity_reference')
        if reference not in ('joint_faces','centerlines'):
            raise ValueError('Explicit capacity_reference joint_faces/centerlines is required')
        offsets={end:_number(m.get('face_offset_'+end+'_in',0.) if reference=='centerlines' else m['face_offset_'+end+'_in'], 'capacity face offset',nonnegative=True) for end in ('i','j')}
        if sum(offsets.values())>=length:raise ValueError('Capacity faces leave no clear member length')
        end_interactions=[]
        for end,mi in [('i',im),('j',ij)]:
            s=offsets[end]/length
            position=s if end=='i' else 1-s;side='right' if end=='i' else 'left'
            pshift=_axial_shift(m,length,position,side=side)
            moment={im:(1-s if end=='i' else -s),ij:(-s if end=='i' else 1-s)}
            moment_fixed=_gravity_moment(m,length,position,side=side)*(1 if end=='i' else -1)
            interaction=m['interaction_'+end]
            if interaction.get('nominal') is not True or interaction.get('phi')!=1.:
                raise ValueError('Story resistance requires explicit nominal phi=1 interaction surfaces')
            domain=interaction['axial_domain_kip']
            if len(domain)!=2 or not all(math.isfinite(float(v)) for v in domain) or domain[0]>=domain[1]:
                raise ValueError('Invalid nominal axial domain')
            end_interactions.append((float(domain[0])-pshift,float(domain[1])-pshift))
            for k,(a,b,c) in enumerate(interaction[bound+'_halfplanes']):
                a,b,c=(_number(v,'interaction coefficient') for v in (a,b,c))
                if b==0:raise ValueError('Moment interaction halfplanes require nonzero M coefficient')
                inequalities.add({ip:a,im:b*moment[im],ij:b*moment[ij]},c-a*pshift-b*moment_fixed,f'{mid}:{end}:NM:{k}')
        low=max(v[0] for v in end_interactions); high=min(v[1] for v in end_interactions)
        if low>high:raise ValueError('End axial domains do not overlap after member weight correction')
        bounds.extend([(low,high),(None,None),(None,None)])
        shear_points={offsets['i']/length,1-offsets['j']/length,*[a for a,p in _point_loads(m) if offsets['i']/length<a<1-offsets['j']/length]}
        if field is not None:shear_points.update(load_field.breaks(field,offsets['i']/length,1-offsets['j']/length))
        for sign in (-1.,1.):
            for s in sorted(shear_points):
                for side in ('left','right'):
                    fixed=_gravity_shear(m,length,s,side=side)
                    if math.isfinite(shear):inequalities.add({im:-sign/length,ij:-sign/length},shear-sign*fixed,f'{mid}:shear:{s:.12g}:{side}:{sign:g}')
        if m['kind']=='beam':
            span=m.get('span_interaction')
            if span is None or span.get('nominal') is not True or span.get('phi')!=1.:
                raise ValueError('Every beam requires its physical midspan nominal phi=1 interaction')
            left,right=offsets['i']/length,1-offsets['j']/length
            if mid in envelope_indices:
                lower_m,upper_m=envelope_indices[mid]
                for k,(a,b,c) in enumerate(span[bound+'_halfplanes']):
                    inequalities.add({ip:a,(upper_m if b>0 else lower_m):b},c,f'{mid}:span_capacity:NM:{k}')
                span_low,span_high=span['axial_domain_kip']
                inequalities.add({ip:1.},span_high,f'{mid}:span:axial_max')
                inequalities.add({ip:-1.},-span_low,f'{mid}:span:axial_min')
            stations={left,(left+right)/2,right,*[a for a,p in _point_loads(m) if left<a<right],*extra_points.get(mid,())}
            if field is not None:stations.update(load_field.breaks(field,left,right))
            traces=[]
            for s in sorted(stations):
                sides=(('right',) if s==left else ('left',) if s==right else ('left','right')) if field is not None else ('right',)
                traces.extend((s,side) for side in sides)
            for x,side in traces:
                # M(s)=(1-s)Mi-sMj+wL^2*s(1-s)/2, sagging-positive.
                fixed=_gravity_moment(m,length,x,side=side)
                pshift=_axial_shift(m,length,x,side=side)
                name=f'{mid}:span:{x:.12g}'+(f':{side}' if field is not None else '')
                if mid in envelope_indices:
                    inequalities.add({im:1-x,ij:-x,upper_m:-1.},-fixed,name+':upper_envelope')
                    inequalities.add({im:x-1,ij:x,lower_m:1.},fixed,name+':lower_envelope')
                    continue
                for k,(a,b,c) in enumerate(span[bound+'_halfplanes']):
                    inequalities.add({ip:a,im:b*(1-x),ij:-b*x},c-b*fixed-a*pshift,name+f':NM:{k}')
                span_low,span_high=span['axial_domain_kip']
                inequalities.add({ip:1.},span_high-pshift,name+':axial_max')
                inequalities.add({ip:-1.},-span_low+pshift,name+':axial_min')
        # Member forces applied to nodes. Fixed distributed-load terms are
        # moved to the external load vector, preserving vertical equilibrium.
        for node,end,axsign,vs,mi in [(ni,'i',-1.,1.,im),(nj,'j',1.,-1.,ij)]:
            nid=str(node['id']);floor=int(node['floor']);incident[nid].append(mid)
            if floor==0:continue
            xrow={ip:axsign*cx,im:vs*sy/length,ij:vs*sy/length}
            zrow={ip:axsign*cz,im:vs*sz/length,ij:vs*sz/length}
            _merge(dof_rows[ndof[('floor',floor,'x')]],xrow)
            _merge(dof_rows[ndof[(nid,'z')]],zrow)
            dof_rows[ndof[(nid,'rotation')]][mi]=1.
            reaction=(_gravity_shear(m,length,0.,side='left') if end=='i' else -_gravity_shear(m,length,1.))
            axial_fixed=axsign*_axial_shift(m,length,0. if end=='i' else 1.,side='right' if end=='i' else 'left')
            loads[ndof[('floor',floor,'x')]]+=axial_fixed*cx-reaction*sy
            loads[ndof[(nid,'z')]]+=axial_fixed*cz-reaction*sz
    bounds.extend([(None,None)]*(2*len(envelope_members)))
    bounds.append((0.,None))
    covered=set()
    for constraint in model.get('joint_constraints',[]):
        if constraint.get('bound','both') not in ('both',bound):continue
        if not constraint.get('basis') or constraint.get('nominal') is not True:
            raise ValueError('Joint constraints require an explicit nominal physical basis')
        node=str(constraint['node']);sense=constraint.get('sense')
        if sense not in (-1,1):raise ValueError('Every joint constraint requires its positive/negative sense')
        covered.add((node,sense));coefficients={}
        for mid,values in constraint['terms'].items():
            if mid not in index or mid not in incident[node]:raise ValueError('Joint constraint names an absent or nonincident member')
            for name,value in values.items():
                if name not in index[mid]:raise ValueError('Joint term must be P, Mi or Mj')
                coefficients[index[mid][name]]=_number(value,'joint coefficient')
        if not any(coefficients.values()):raise ValueError('An empty joint constraint cannot establish coverage')
        limit=_number(constraint['limit'],'joint limit',positive=True)
        constant=_number(constraint.get('constant',0.),'joint constant')
        inequalities.add(coefficients,limit-constant,'joint:'+str(constraint['id']))
    needed={(nid,sense) for nid,mids in incident.items() if int(by_node[nid]['floor'])>0
            and any(by_member[mid]['kind']=='beam' for mid in mids)
            and any(by_member[mid]['kind']=='column' for mid in mids) for sense in (-1,1)}
    if covered!=needed:raise ValueError('Joint coverage mismatch: '+str({'missing':sorted(needed-covered),'unexpected':sorted(covered-needed)}))
    for row,rhs,name in zip(dof_rows,-loads,names):eq.add(row,rhs,name)
    objective=np.zeros(nvar);objective[lam]=-1.
    return dict(c=objective,A_eq=eq.matrix(),b_eq=np.array(eq.rhs),A_ub=inequalities.matrix(),
                b_ub=np.array(inequalities.rhs),bounds=bounds,index=index,geometry=geometry,
                constants=constants,envelope_indices=envelope_indices,eq_names=eq.names,limit_names=inequalities.names)


def _cached_build(model,story,sway,bound,extra_points,cache):
    """Reuse identical physical inequalities; only the force-pair column varies."""
    if cache is None:return _build(model,story,sway,bound,extra_points)
    key=(_hash(model),bound,_hash(extra_points))
    if key not in cache:cache[key]=_build(model,story,sway,bound,extra_points)
    template=cache[key];force=np.zeros(len(template['b_eq']))
    force[template['eq_names'].index(f'floor:{story}:x')]=float(sway)
    if story>1:force[template['eq_names'].index(f'floor:{story-1}:x')]=-float(sway)
    from scipy.sparse import hstack,csr_matrix
    return {**template,'A_eq':hstack((template['A_eq'][:,:-1],csr_matrix(force[:,None])),format='csr')}


def _solve(model, story, sway, bound, *, time_limit_seconds, tolerance):
    start=time.perf_counter();points={};passes=[];active_names=set();generation=[];built=None;groups=None
    def unresolved(message):
        return {'status':'unresolved','message':message,'story':story,'sway':sway,'bound':bound,
                'elapsed_seconds':time.perf_counter()-start,'constraint_generation':generation}
    for iteration in range(32):
        remaining=time_limit_seconds-(time.perf_counter()-start)
        if remaining<=0:return unresolved('Story resistance LP exceeded its declared time limit')
        # Exact constraint generation, not facet deletion: start with a small
        # strength subset, test EVERY full inequality, add violated facets,
        # and accept only after the complete primal/dual checks below.
        if built is None:
            built=_build(model,story,sway,bound,points)
            groups=defaultdict(list)
            for j,name in enumerate(built['limit_names']):
                if ':NM:' in name:groups[name.rsplit(':',1)[0]].append(j)
                else:active_names.add(name)
            for indices in groups.values():
                initial=(indices[:1]+indices[-1:] if ':span:' in built['limit_names'][indices[0]]
                         else indices[::max(1,len(indices)//8)]+indices[-1:])
                for j in initial:active_names.add(built['limit_names'][j])
        active=np.array([j for j,name in enumerate(built['limit_names']) if name in active_names],dtype=int)
        if len(built['limit_names'])<=100000:
            # The exact beam-envelope reformulation makes ordinary candidate
            # matrices small enough that complete facets cost less than
            # repeated constraint-generation solves.
            active=np.arange(len(built['limit_names']),dtype=int)
            active_names.update(built['limit_names'])
        remaining=time_limit_seconds-(time.perf_counter()-start)
        if remaining<=0:return unresolved('Story resistance LP exceeded its declared time limit')
        solved=linprog(built['c'],A_ub=built['A_ub'][active],b_ub=built['b_ub'][active],A_eq=built['A_eq'],b_eq=built['b_eq'],
                       bounds=built['bounds'],method='highs',options={'time_limit':remaining,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
        if not solved.success:
            return {'status':'infeasible' if solved.status==2 else 'unresolved','solver_status':solved.status,
                    'message':solved.message,'story':story,'sway':sway,'bound':bound,'elapsed_seconds':time.perf_counter()-start,
                    'constraint_generation':generation}
        x=solved.x;all_slack=built['A_ub']@x-built['b_ub'];cuts=[];span_worst={}
        for prefix,indices in groups.items():
            j=max(indices,key=lambda k:all_slack[k])
            if all_slack[j]<=tolerance:continue
            if ':span:' in prefix:
                member=prefix.split(':span:',1)[0]
                if member not in span_worst or all_slack[j]>span_worst[member][0]:span_worst[member]=(all_slack[j],indices)
            else:cuts.extend(indices)
        for violation,indices in span_worst.values():cuts.extend(indices)
        newcuts=[j for j in cuts if built['limit_names'][j] not in active_names]
        if cuts and not newcuts:return unresolved('The LP solver did not satisfy an already active full-strength constraint')
        active_names.update(built['limit_names'][j] for j in newcuts)
        generation.append({'iteration':iteration+1,'active_inequalities':len(active),'full_inequalities':len(all_slack),
                           'maximum_full_constraint_violation':float(np.max(all_slack,initial=0.)),'added_facets':len(newcuts)})
        if cuts:continue
        added=False;max_span=0.
        for member in model['members']:
            if member['kind']!='beam':continue
            mid=str(member['id']);idx=built['index'][mid];length,_,_=built['geometry'][mid];w,_=built['constants'][mid]
            left=member.get('face_offset_i_in',0.)/length;right=1-member.get('face_offset_j_in',0.)/length
            if member.get('fixed_section_field') is not None:
                violation,at=load_field.maximum_violation(member['fixed_section_field'],left,right,x[idx['P']],x[idx['Mi']],x[idx['Mj']],member['span_interaction'],bound)
                max_span=max(max_span,violation)
                if violation>tolerance:points.setdefault(mid,set()).add(at[0]);added=True
                continue
            breaks=sorted({left,right,*[a for a,p in _point_loads(member) if left<a<right]})
            if not w:continue # all extrema of piecewise-linear bending were already constrained
            for a,b in zip(breaks,breaks[1:]):
                shear=_gravity_shear(member,length,(a+b)/2)-(x[idx['Mi']]+x[idx['Mj']])/length
                s=(a+b)/2+shear/(w*length)
                if not a<s<b:continue
                p=x[idx['P']]+built['constants'][mid][1]*length*(.5-s)
                moment=(1-s)*x[idx['Mi']]-s*x[idx['Mj']]+_gravity_moment(member,length,s)
                violation=max(ca*p+cb*moment-cc for ca,cb,cc in member['span_interaction'][bound+'_halfplanes'])
                max_span=max(max_span,violation)
                if violation>tolerance:
                    points.setdefault(mid,set()).add(float(s));added=True
        passes.append({'iteration':iteration+1,'strength_kip':float(x[-1]),'maximum_midspan_violation':max_span})
        if not added:break
        built=None
    else:return unresolved('Exact strength-facet/interior-moment refinement did not converge')
    eqres=built['A_eq']@x-built['b_eq'];ineq=built['A_ub']@x-built['b_ub']
    # HiGHS minimization dual. Its objective and stationarity provide an
    # independently recomputed LP certificate, not just solver.success.
    y=solved.eqlin.marginals;z=np.zeros(len(built['b_ub']));z[active]=solved.ineqlin.marginals
    lower=solved.lower.marginals;upper=solved.upper.marginals
    dual=float(built['b_eq']@y+built['b_ub']@z)
    for j,(lo,hi) in enumerate(built['bounds']):
        if lo is not None:dual+=lo*lower[j]
        if hi is not None:dual+=hi*upper[j]
    station=built['c']-built['A_eq'].T@y-built['A_ub'].T@z-lower-upper
    certificate={'maximum_equilibrium_residual':float(np.max(np.abs(eqres),initial=0.)),
                 'maximum_strength_constraint_violation':float(np.max(np.maximum(ineq,0.),initial=0.)),
                 'dual_stationarity_residual':float(np.max(np.abs(station),initial=0.)),
                 'primal_dual_gap':abs(float(solved.fun)-dual),
                 'dual_nonpositive_inequalities':bool((z<=1e-8).all()),
                 'virtual_mechanism':dict(zip(built['eq_names'],map(float,y)))}
    certificate['dual_proof']={'equations':list(map(float,y)),
        'inequalities':[[int(i),float(v)] for i,v in enumerate(z) if v!=0.],
        'lower_bounds':[[int(i),float(v)] for i,v in enumerate(lower) if v!=0.],
        'upper_bounds':[[int(i),float(v)] for i,v in enumerate(upper) if v!=0.]}
    scale=max(1.,float(np.max(np.abs(x),initial=0.)))
    if (max(certificate[k] for k in ['maximum_equilibrium_residual','maximum_strength_constraint_violation','dual_stationarity_residual','primal_dual_gap'])>max(tolerance,1e-8*scale)
            or not certificate['dual_nonpositive_inequalities']):
        raise RuntimeError('Static/dual story resistance certificate failed')
    baseline_shear=sum(float(n.get('load',{}).get('x_kip',0.)) for n in model['nodes'] if int(n['floor'])>=story)
    contributions=defaultdict(float)
    for member in model['members']:
        if member['kind']=='column' and int(next(n['floor'] for n in model['nodes'] if str(n['id'])==str(member['j'])))==story:
            mid=str(member['id']);indices=built['index'][mid];length,_,_=built['geometry'][mid]
            contributions[str(member.get('line_id','unspecified'))]+=-sway*(x[indices['Mi']]+x[indices['Mj']])/length
    return {'status':'solved','story':story,'sway':sway,'bound':bound,'strength_kip':float(x[-1])+sway*baseline_shear,
            'model_sha256':_hash(model),'primal_variables':list(map(float,x)),
            'span_refinement_points':{mid:sorted(values) for mid,values in points.items()},
            'additional_story_force_kip':float(x[-1]),'baseline_story_shear_kip':baseline_shear,
            'resisting_shear_contributions_by_line':dict(contributions),
            'contribution_note':'One admissible force allocation, not unique independent line capacities or a torsion fraction.',
            'member_end_actions':{mid:{name:float(x[i]) for name,i in indices.items()} for mid,indices in built['index'].items()},
            'active_limits':[name for name,slack in zip(built['limit_names'],-ineq) if slack<=max(tolerance,1e-7*scale)],
            'certificate':certificate,'midspan_refinement':passes,'elapsed_seconds':time.perf_counter()-start,
            'constraint_generation':generation,
            'sizes':{'variables':len(x),'equations':built['A_eq'].shape[0],'inequalities':built['A_ub'].shape[0]}}


def analyze_frame(model, *, time_limit_seconds=30., tolerance=1e-5):
    """Both nominal strength bounds and sway senses at every story.

    See this module's protocol/tests for the explicit model schema. Joint
    linear constraints must be derived upstream from the actual nominal panel
    force equilibrium (member-force coefficients and known gravity constants).
    They are not guessed from a beam Mn/nominal depth ratio here. Missing
    joint/axial/beam-span inputs raise; infeasibility stays distinct from an
    unresolved solver. Source and combination identity are mandatory.
    """
    limit=_number(time_limit_seconds,'LP time limit',positive=True)
    tolerance=_number(tolerance,'absolute LP tolerance',positive=True)
    if not model.get('state_identity') or not model.get('combination_id') or not model.get('source_identity'):
        raise ValueError('Final state, load-combination and source identity are mandatory')
    start=time.perf_counter();rows=[]
    floors=sorted({int(n['floor']) for n in model['nodes']})
    for story in floors[1:]:
        senses=[]
        for sway in (1,-1):
            inner=_solve(model,story,sway,'inner',time_limit_seconds=limit,tolerance=tolerance)
            outer=_solve(model,story,sway,'outer',time_limit_seconds=limit,tolerance=tolerance)
            lower=inner.get('strength_kip') if inner['status']=='solved' else None
            upper=outer.get('strength_kip') if outer['status']=='solved' else None
            if lower is not None and upper is not None:
                if inner['strength_kip']>outer['strength_kip']+tolerance:
                    raise RuntimeError('Inner story strength exceeds its outer bound')
                status='bounded'
            else:status='lower_certified' if lower is not None else 'incomplete'
            senses.append({'sway':sway,'status':status,'lower_kip':lower,'upper_kip':upper,'inner':inner,'outer':outer})
        # The story measure is min(R+,R-). Both feasible lower fields are
        # necessary, but a certified upper bound on either sense bounds that
        # minimum from above. An unresolved opposite upper remains explicit.
        lower=min(s['lower_kip'] for s in senses) if all(s['lower_kip'] is not None for s in senses) else None
        uppers=[s['upper_kip'] for s in senses if s['upper_kip'] is not None]
        upper=min(uppers) if uppers else None
        rows.append({'story':story,'senses':senses,'status':'bounded' if lower is not None and upper is not None else 'incomplete',
                     'lower_kip':lower,'upper_kip':upper,
                     'upper_bound_sway_inventory':[s['sway'] for s in senses if s['upper_kip'] is not None]})
    return {'method':METHOD,'model_sha256':_hash(model),'state_identity':model['state_identity'],
            'source_identity':model['source_identity'],'combination_id':model['combination_id'],
            'status':'bounded' if all(r['status']=='bounded' for r in rows) else 'incomplete',
            'stories':rows,'elapsed_seconds':time.perf_counter()-start,
            'load_definition':'unit horizontal force pair at the target story bounding diaphragms; other floor lateral loads retained; no arbitrary nodal distribution',
            'scope':'static nominal plane-frame story resistance; no ELF replacement, stability/degradation or nonlinear-performance qualification'}


def transport_inner(reference_model,reference_solution,target_model,story,sway,*,tolerance=1e-5,build_cache=None):
    """Certified feasible lower strength for a different exact gravity case.

    A reference lateral force increment is added to the target's exact
    gravity force field. Scaling that increment along a convex feasible
    segment preserves equilibrium. Every full strength inequality and every
    exact beam moment extremum is checked. This is a feasible lower bound,
    not an optimized value or a claimed matching-dual solution.
    """
    start=time.perf_counter()
    if reference_solution.get('status')!='solved':raise ValueError('An optimized certified reference solution is required')
    if reference_model['state_identity']!=target_model['state_identity']:raise ValueError('Transport cannot change the physical candidate state')
    if set(reference_solution['member_end_actions'])!={str(m['id']) for m in target_model['members']}:raise ValueError('Transport member inventory differs')
    built=_cached_build(target_model,story,sway,'inner',{},build_cache)
    base=np.zeros(len(built['c']));increment=np.zeros_like(base);beam_data=[];field_beams=[]
    for m in target_model['members']:
        mid=str(m['id']);indices=built['index'][mid]
        for key,j in indices.items():
            base[j]=target_model['baseline_member_actions'][mid][key]
            increment[j]=reference_solution['member_end_actions'][mid][key]-reference_model['baseline_member_actions'][mid][key]
        if m['kind']=='beam':
            if m.get('fixed_section_field') is not None:
                length=built['geometry'][mid][0]
                field_beams.append((m,indices,m.get('face_offset_i_in',0.)/length,1-m.get('face_offset_j_in',0.)/length))
                continue
            if mid not in built['envelope_indices']:raise ValueError('Transport requires constant beam axial force')
            length=built['geometry'][mid][0];left=m.get('face_offset_i_in',0.)/length;right=1-m.get('face_offset_j_in',0.)/length
            points=_point_loads(m);w=float(m.get('transverse_load_kip_per_in',0.))
            fractions=np.array(sorted({left,right,*[a for a,p in points if left<a<right]}))
            fixed=np.array([_gravity_moment(m,length,s) for s in fractions]);mids=(fractions[:-1]+fractions[1:])/2
            shear=np.array([_gravity_shear(m,length,s) for s in mids])
            beam_data.append((mid,indices,length,fractions,fixed,mids,shear,w,points))
    increment[-1]=reference_solution['additional_story_force_kip']
    lows=np.array([-np.inf if a is None else a for a,b in built['bounds']]);highs=np.array([np.inf if b is None else b for a,b in built['bounds']])
    def vector(alpha):
        x=base+alpha*increment
        for mid,idx,length,s,fixed,mids,shear,w,points in beam_data:
            mi,mj=x[idx['Mi']],x[idx['Mj']];moments=(1-s)*mi-s*mj+fixed
            if w:
                roots=mids+(shear-(mi+mj)/length)/(w*length)
                valid=(roots>s[:-1])&(roots<s[1:]);roots=roots[valid]
                if len(roots):
                    gm=w*length**2*roots*(1-roots)/2
                    for a,p in points:gm+=p*length*np.minimum(roots,a)*(1-np.maximum(roots,a))
                    moments=np.concatenate((moments,(1-roots)*mi-roots*mj+gm))
            lo,hi=built['envelope_indices'][mid];x[lo]=float(moments.min());x[hi]=float(moments.max())
        return x
    def violation(x):
        return max(float(np.max(built['A_ub']@x-built['b_ub'],initial=0.)),
                   float(np.max(lows-x,initial=0.)),float(np.max(x-highs,initial=0.)),
                   *[load_field.maximum_violation(m['fixed_section_field'],left,right,x[idx['P']],x[idx['Mi']],x[idx['Mj']],m['span_interaction'],'inner')[0]
                     for m,idx,left,right in field_beams])
    baseline=vector(0.)
    base_violation=violation(baseline)
    if base_violation>tolerance:
        return {'status':'unavailable','reason':'The saved elastic gravity field is outside the conservative inner limits; redistribution may still be feasible.',
                'maximum_gravity_constraint_violation':base_violation,'elapsed_seconds':time.perf_counter()-start}
    endpoint=vector(1.);iterations=0
    if violation(endpoint)<=tolerance:alpha=1.;x=endpoint
    else:
        lower,upper=0.,1.
        for iterations in range(40):
            alpha=(lower+upper)/2;x=vector(alpha)
            if violation(x)<=tolerance:lower=alpha
            else:upper=alpha
        alpha=lower;x=vector(alpha)
    residual=float(np.max(np.abs(built['A_eq']@x-built['b_eq']),initial=0.));strength_violation=violation(x)
    if residual>max(tolerance,1e-8*max(1.,float(np.max(np.abs(x))))) or strength_violation>tolerance*1.01:
        raise RuntimeError('Transported static force field failed its independent full certificate')
    baseline_shear=sum(float(n.get('load',{}).get('x_kip',0.)) for n in target_model['nodes'] if int(n['floor'])>=story)
    return {'status':'feasible_lower_bound','story':story,'sway':sway,'bound':'inner',
            'model_sha256':_hash(target_model),'primal_variables':list(map(float,x)),'span_refinement_points':{},
            'strength_kip':float(x[-1])+sway*baseline_shear,'additional_story_force_kip':float(x[-1]),
            'baseline_story_shear_kip':baseline_shear,'transport_factor':alpha,
            'reference_combination_id':reference_model['combination_id'],'target_combination_id':target_model['combination_id'],
            'reference_model_sha256':_hash(reference_model),'target_model_sha256':_hash(target_model),
            'member_end_actions':{mid:{key:float(x[j]) for key,j in indices.items()} for mid,indices in built['index'].items()},
            'certificate':{'maximum_equilibrium_residual':residual,'maximum_strength_constraint_violation':strength_violation,
                           'every_exact_beam_span_extremum_checked':True,'full_inequalities_checked':len(built['b_ub']),
                           'bisection_iterations':iterations+1 if alpha<1 else 0,
                           'optimality_claim':False,'basis':'Equilibrated force transport plus complete inner-domain primal certificate'},
            'elapsed_seconds':time.perf_counter()-start}


def verify_solution(model,solution,*,tolerance=1e-5,build_cache=None):
    """Read back a saved primal/dual certificate without solving another LP."""
    if solution.get('status') not in ('solved','feasible_lower_bound'):raise ValueError('No completed strength certificate to verify')
    if solution.get('model_sha256')!=_hash(model):raise ValueError('Saved strength certificate model hash differs')
    story,sway,bound=solution['story'],solution['sway'],solution['bound']
    if bound=='outer' and solution['status']!='solved':raise ValueError('An outer bound requires an optimized dual certificate')
    built=_cached_build(model,story,sway,bound,solution.get('span_refinement_points',{}),build_cache)
    x=np.asarray(solution['primal_variables'],dtype=float)
    if x.shape!=built['c'].shape or not np.isfinite(x).all():raise ValueError('Saved primal vector shape/values invalid')
    eq=float(np.max(np.abs(built['A_eq']@x-built['b_eq']),initial=0.))
    strength=float(np.max(built['A_ub']@x-built['b_ub'],initial=0.))
    for j,(lo,hi) in enumerate(built['bounds']):
        if lo is not None:strength=max(strength,lo-x[j])
        if hi is not None:strength=max(strength,x[j]-hi)
    # Exact quadratic extrema between every point-load station, independently
    # of whatever refinement points the saved LP chose.
    for m in model['members']:
        mid=str(m['id']);idx=built['index'][mid]
        for key,j in idx.items():
            if not math.isclose(solution['member_end_actions'][mid][key],x[j],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('Saved member forces differ from primal vector')
        if m['kind']!='beam':continue
        length=built['geometry'][mid][0];w=float(m.get('transverse_load_kip_per_in',0.))
        left=m.get('face_offset_i_in',0.)/length;right=1-m.get('face_offset_j_in',0.)/length
        if m.get('fixed_section_field') is not None:
            strength=max(strength,load_field.maximum_violation(m['fixed_section_field'],left,right,x[idx['P']],x[idx['Mi']],x[idx['Mj']],m['span_interaction'],bound)[0])
            continue
        breaks=sorted({left,right,*[a for a,p in _point_loads(m) if left<a<right]})
        samples=list(breaks)
        if w:
            for a,b in zip(breaks,breaks[1:]):
                s=(a+b)/2+(_gravity_shear(m,length,(a+b)/2)-(x[idx['Mi']]+x[idx['Mj']])/length)/(w*length)
                if a<s<b:samples.append(s)
        for s in samples:
            moment=(1-s)*x[idx['Mi']]-s*x[idx['Mj']]+_gravity_moment(m,length,s)
            strength=max(strength,max(a*x[idx['P']]+b*moment-c for a,b,c in m['span_interaction'][bound+'_halfplanes']))
    baseline=sum(float(n.get('load',{}).get('x_kip',0.)) for n in model['nodes'] if int(n['floor'])>=story)
    stated=float(x[-1])+sway*baseline
    if not math.isclose(stated,solution['strength_kip'],rel_tol=1e-10,abs_tol=tolerance):raise ValueError('Saved strength does not match certified generalized force')
    report={'maximum_equilibrium_residual':eq,'maximum_strength_constraint_violation':max(0.,strength),'exact_span_extrema_checked':True}
    if solution['status']=='solved':
        proof=solution['certificate']['dual_proof'];y=np.asarray(proof['equations']);z=np.zeros(len(built['b_ub']));lo=np.zeros(len(x));hi=np.zeros(len(x))
        if y.shape!=built['b_eq'].shape or not np.isfinite(y).all():raise ValueError('Invalid saved dual equation multipliers')
        for vector,key in ((z,'inequalities'),(lo,'lower_bounds'),(hi,'upper_bounds')):
            seen=set()
            for j,v in proof[key]:
                if type(j) is not int or not 0<=j<len(vector) or j in seen or not math.isfinite(v):raise ValueError('Invalid or duplicate saved dual multiplier')
                vector[j]=v;seen.add(j)
        if (z>1e-8).any() or (lo< -1e-8).any() or (hi>1e-8).any():raise ValueError('Saved dual multiplier signs are invalid')
        objective=float(built['b_eq']@y+built['b_ub']@z)
        for j,(a,b) in enumerate(built['bounds']):
            if a is None and lo[j]!=0. or b is None and hi[j]!=0.:raise ValueError('Dual multiplier on an absent variable bound')
            if a is not None:objective+=a*lo[j]
            if b is not None:objective+=b*hi[j]
        report.update(dual_stationarity_residual=float(np.max(np.abs(built['c']-built['A_eq'].T@y-built['A_ub'].T@z-lo-hi),initial=0.)),
                      primal_dual_gap=abs(float(built['c']@x)-objective))
    scale=max(1.,float(np.max(np.abs(x),initial=0.)))
    if max(v for k,v in report.items() if k!='exact_span_extrema_checked')>max(tolerance*1.01,1e-8*scale):raise ValueError('Saved strength certificate failed independent numerical readback')
    return report


def adjacent_strength_bounds(stories):
    """Conservative ratio intervals, preserving unresolved threshold crossings."""
    rows=[]
    for lower,upper in zip(stories,stories[1:]):
        if lower.get('status')!='bounded' or upper.get('status')!='bounded' or min(lower['lower_kip'],upper['lower_kip'])<=0:
            rows.append({'story':lower['story'],'status':'incomplete'});continue
        lo=lower['lower_kip']/upper['upper_kip'];hi=lower['upper_kip']/upper['lower_kip']
        rows.append({'story':lower['story'],'status':'bounded','ratio_lower':lo,'ratio_upper':hi,
                     'definitely_below_1':hi<1.,'definitely_at_least_1':lo>=1.,
                     'definitely_below_0_65':hi<.65,'definitely_at_least_0_65':lo>=.65,
                     'definitely_at_least_0_8':lo>=.8,
                     'threshold_crossings':[t for t in (.65,.8,1.) if lo<t<=hi]})
    return rows


def analyze_gravity_family(models,*,time_limit_seconds=60.,tolerance=1e-5):
    """Minimum directional strength over every exact gravity state.

    The first state's optimized outer solution bounds the minimum over the
    family from above. Every state's feasible inner force field is required
    for the lower bound. Transport failures trigger a direct inner LP; they
    never become fabricated physical infeasibility or omitted cases.
    """
    if not models or len({m['combination_id'] for m in models})!=len(models):raise ValueError('Unique nonempty exact gravity family required')
    start=time.perf_counter();reference=models[0];optimized=analyze_frame(reference,time_limit_seconds=time_limit_seconds,tolerance=tolerance)
    results=[optimized];direct_inner=0;transported=0
    for model in models[1:]:
        rows=[];build_cache={}
        for ref_story in optimized['stories']:
            senses=[];story=ref_story['story']
            for ref_sense in ref_story['senses']:
                sway=ref_sense['sway'];ref=ref_sense['inner']
                if ref['status']=='solved':answer=transport_inner(reference,ref,model,story,sway,tolerance=tolerance,build_cache=build_cache)
                else:answer={'status':'unavailable','reason':'Reference inner solution is not available'}
                if answer['status']=='unavailable':
                    answer=_solve(model,story,sway,'inner',time_limit_seconds=time_limit_seconds,tolerance=tolerance);direct_inner+=1
                else:transported+=1
                valid=answer['status'] in ('solved','feasible_lower_bound')
                senses.append({'sway':sway,'status':'lower_certified' if valid else 'incomplete',
                               'lower_kip':answer.get('strength_kip') if valid else None,'upper_kip':None,'inner':answer,
                               'outer':{'status':'not_optimized','reason':'The reference outer bound already bounds the family minimum'}})
            complete=all(s['status']=='lower_certified' for s in senses)
            rows.append({'story':story,'senses':senses,'status':'lower_certified' if complete else 'incomplete',
                         'lower_kip':min(s['lower_kip'] for s in senses) if complete else None,'upper_kip':None})
        results.append({'method':METHOD,'combination_id':model['combination_id'],'model_sha256':_hash(model),
                        'state_identity':model['state_identity'],'source_identity':model['source_identity'],'stories':rows,
                        'status':'lower_certified' if all(r['status']=='lower_certified' for r in rows) else 'incomplete'})
    stories=[]
    for ref_story in optimized['stories']:
        story=ref_story['story'];entries=[next(s for s in r['stories'] if s['story']==story) for r in results]
        lower_complete=all(s['lower_kip'] is not None for s in entries)
        upper=ref_story['upper_kip'];lower=min(s['lower_kip'] for s in entries) if lower_complete else None
        if lower is not None and upper is not None and lower>upper+tolerance:raise RuntimeError('Family lower bound exceeds its optimized reference upper bound')
        stories.append({'story':story,'status':'bounded' if lower_complete and upper is not None else 'incomplete',
                        'lower_kip':lower,'upper_kip':upper,
                        'governing_lower_case':min(zip(results,entries),key=lambda pair:pair[1]['lower_kip'])[0]['combination_id'] if lower_complete else None})
    return {'method':METHOD,'stories':stories,'cases':results,'reference_case':reference['combination_id'],
            'status':'bounded' if all(r['status']=='bounded' for r in stories) else 'incomplete',
            'additional_direct_inner_solves':direct_inner,'transported_inner_certificates':transported,
            'elapsed_seconds':time.perf_counter()-start,
            'family_bound_proof':'Minimum of certified feasible lower strengths for every state; optimized outer strength of one state bounds the family minimum from above.'}
