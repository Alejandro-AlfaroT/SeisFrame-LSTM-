"""Interval integration adapter for the existing SpatialFloorSection geometry.

Uses its disjoint slab/web rectangles, nominal circular steel voids, actual
transverse rows and external void rectangles. No new constitutive law, steel
credit, phi or strain limit is introduced. A result bounds a specified strain
box only; it does not find compatible roots or establish continuous cut coverage.
"""
from __future__ import annotations
from decimal import Decimal
import hashlib
import json

from Design.SMRF_Interval_Material import (Interval,Gradient,UnsupportedBox,
    concrete_stress_bounds,concrete_stress_gradient,steel_stress_bounds,
    _principal_stress,_principal_stress_bounds)
from Design.SMRF_Interval_Quadrature import integrate_rectangle
from Design.SMRF_Interval_Taylor import Taylor

METHOD='actual_spatial_section_outward_integral_v2_gauss_remainder_unqualified'


def _constant(value, like):
    return (type(like).constant(value,len(like.derivatives)) if isinstance(like,(Gradient,Taylor))
            else value if isinstance(value,Interval) else Interval(value))


def _interval(value):
    return value.value if isinstance(value,(Gradient,Taylor)) else value


def _density(section,q,region,s,z):
    strain=[_constant(q[k],s)+z*q[k+3] for k in range(3)]
    strain[section.j]=strain[section.j]+(s-section.sref)*q[6]
    zero=_constant(0,s)
    if region['id']=='slab':
        method=concrete_stress_gradient if isinstance(s,(Gradient,Taylor)) else concrete_stress_bounds
        stress=method(*strain,fc_ksi=section.fc,peak_strain=section.peak)
    else:
        method=_principal_stress if isinstance(s,(Gradient,Taylor)) else _principal_stress_bounds
        normal=method(strain[section.j],zero,Interval(section.fc),Interval(section.peak))
        stress=(normal,zero,zero) if section.j==0 else (zero,normal,zero)
    return list(stress)+[v*z for v in stress]+[stress[section.j]*(s-section.sref)]


def _rectangle(section,q,region):
    def provider(s,z):return _density(section,q,region,s,z)
    return provider,provider,[region['s'],region['z']]


def _circle(section,q,region,bar):
    """Unit-square disk map with exact Jacobian; clip only to its owner region."""
    def provider(u,v):
        # s=sc+r*u, z=zc+r*sqrt(1-u^2)*v, Jacobian=r^2*sqrt(1-u^2).
        # No trigonometric or numerical circle-boundary approximation is used.
        root=(1-u.square()).sqrt()
        s=bar['s_in']+bar['radius']*u
        z=bar['z_in']+bar['radius']*root*v
        si,zi=_interval(s),_interval(z)
        sa,sb=map(Decimal,region['s']);za,zb=map(Decimal,region['z'])
        if si.hi<=sa or si.lo>=sb or zi.hi<=za or zi.lo>=zb:
            return [_constant(0,u)]*7
        inside=sa<=si.lo and si.hi<=sb and za<=zi.lo and zi.hi<=zb
        if not inside and isinstance(u,(Gradient,Taylor)):
            raise UnsupportedBox('circle_crosses_region_boundary')
        # A partial cell may include strain outside the concrete rectangle.
        # Intersect coordinate ranges for its value enclosure before evaluating.
        if not inside:
            s=Interval(max(sa,si.lo),min(sb,si.hi))
            z=Interval(max(za,zi.lo),min(zb,zi.hi))
        jacobian=root*Interval(bar['radius']).square()
        values=[value*jacobian for value in _density(section,q,region,s,z)]
        if not inside:
            values=[Interval(min(Decimal(0),v.lo),max(Decimal(0),v.hi)) for v in values]
        return values
    return provider,provider,[(-1,1),(-1,1)]


def _transverse(section,q,region,row):
    def provider(u,v):
        root=(1-u.square()).sqrt()
        a,b=row['run_interval_in']
        s=a+(v+1)*(Interval(b)-Interval(a))/2
        z=row['z_in']+row['radius']*u
        jacobian=(Interval(b)-Interval(a))*row['radius']*row['radius']*row['count_per_length_in']
        # Put the Gradient on the left when present.
        jacobian=root*jacobian
        return [value*jacobian for value in _density(section,q,region,s,z)]
    return provider,provider,[(-1,1),(-1,1)]


def section_resultant_enclosure(section,q_box,*,absolute_nominal_error,
                                maximum_cells_per_term=1024,maximum_terms=128,use_fourth_derivative=True):
    """Bound the nominal/design seven resultants of an actual supplied strain box.

    Every integral term gets an explicit share of the requested nominal error.
    Unresolved cells, material domains or work limits leave status unknown.
    Supplied strain fields are not asserted to balance any demand; this adapter
    never sets continuous_capacity_verified or code_qualified to true.
    """
    if len(q_box)!=7 or len(absolute_nominal_error)!=7:
        raise ValueError('Seven strain boxes and nominal error bounds required')
    if not isinstance(use_fourth_derivative,bool):raise ValueError('Explicit quadrature selection required')
    q=[Interval(*v) for v in q_box]
    tolerance=[Interval(v) for v in absolute_nominal_error]
    if any(v.lo<=0 for v in tolerance):raise ValueError('Positive nominal error bounds required')
    if not isinstance(maximum_terms,int) or isinstance(maximum_terms,bool) or not 1<=maximum_terms<=4096:
        raise ValueError('Explicit bounded term inventory required')
    specifications=[]
    for region in section.regions:
        specifications.append((1,region['id']+'/gross',_rectangle(section,q,region)))
        for bar in section.longitudinal:
            if ((Interval(bar['s_in'])+bar['radius']).hi<=Decimal(region['s'][0])
                or (Interval(bar['s_in'])-bar['radius']).lo>=Decimal(region['s'][1])
                or (Interval(bar['z_in'])+bar['radius']).hi<=Decimal(region['z'][0])
                or (Interval(bar['z_in'])-bar['radius']).lo>=Decimal(region['z'][1])):continue
            specifications.append((-1,region['id']+'/longitudinal/'+bar['id'],_circle(section,q,region,bar)))
        if region['id']=='slab':
            for row in section.transverse:
                specifications.append((-1,'slab/transverse/'+row['id'],_transverse(section,q,region,row)))
            for void in section.slab_voids:
                specifications.append((-1,'slab/external/'+void['physical_void_id'],_rectangle(section,q,void)))
    geometry=dict(regions=section.regions,longitudinal=section.longitudinal,transverse=section.transverse,
                  slab_voids=section.slab_voids,axis=section.axis,reference_s_in=section.sref,width_in=section.width,
                  fc_ksi=section.fc,peak_strain=section.peak,phi=section.phi,steel_strain_limit=section.steel_limit)
    geometry=json.loads(json.dumps(geometry,allow_nan=False))
    payload=dict(method=METHOD,geometry=geometry,q_box=[v.record() for v in q],
                 absolute_nominal_error=[str(v.lo) for v in tolerance],maximum_cells_per_term=maximum_cells_per_term,
                 maximum_terms=maximum_terms,continuous_capacity_verified=False,code_qualified=False,
                 use_fourth_derivative=use_fourth_derivative,
                 physical_failure_claimed=False,terms=[],status='unknown')
    if len(specifications)>maximum_terms:
        payload.update(reason='declared_term_budget_exhausted',required_terms=len(specifications))
    else:
        total=[Interval(0)]*7
        errors=[];steel=[]
        try:
            for sign,identity,providers in specifications:
                values,gradients,rectangle=providers
                # Error is in nominal resultants per unit strip width.
                allocation=[(t*section.width/len(specifications)).lo for t in tolerance]
                result=integrate_rectangle(values,gradients,rectangle,provider_identity=METHOD+'/'+identity,
                    absolute_error=allocation,maximum_cells=maximum_cells_per_term,
                    fourth_provider=gradients if use_fourth_derivative else None)
                payload['terms'].append(dict(sign=sign,identity=identity,integration=result))
                total=[a+sign*Interval(*b) for a,b in zip(total,result['integral'])]
                if not result['requested_error_met']:errors.append(identity+'/integration_error_not_enclosed')
            for bar in section.longitudinal:
                strain=q[section.j]+q[section.j+3]*bar['z_in']+q[6]*(bar['s_in']-section.sref)
                stress=steel_stress_bounds(strain,fy_ksi=bar['fy_ksi'],es_ksi=bar['es_ksi'],strain_limit=section.steel_limit)
                if not bar['strength_credited']:stress=Interval(0)
                force=bar['area_in2']*stress
                total[section.j]+=force;total[section.j+3]+=bar['z_in']*force
                total[6]+=(bar['s_in']-section.sref)*force
                steel.append(dict(id=bar['id'],strain=strain.record(),stress=stress.record()))
            for row in section.transverse:
                strain=q[section.other]+q[section.other+3]*row['z_in']
                stress=steel_stress_bounds(strain,fy_ksi=row['fy_ksi'],es_ksi=row['es_ksi'],strain_limit=section.steel_limit)
                run=row['strength_run_interval_in'];length=Interval(0) if run is None else Interval(run[1])-Interval(run[0])
                force=length*row['area_in2']*row['count_per_length_in']*stress
                total[section.other]+=force;total[section.other+3]+=row['z_in']*force
                steel.append(dict(id=row['id'],strain=strain.record(),stress=stress.record()))
            nominal=[v/section.width for v in total]
            payload.update(nominal_resultants=[v.record() for v in nominal],
                           design_resultants=[(section.phi*v).record() for v in nominal],steel=steel,
                           status='strain_box_resultants_enclosed' if not errors else 'unknown',errors=errors)
        except UnsupportedBox as exc:
            payload.update(reason=str(exc),errors=errors)
    payload['payload_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload


def replay_section_resultant_enclosure(section,record):
    """Recompute from the actual supplied section; no saved claim is trusted."""
    rebuilt=section_resultant_enclosure(section,record['q_box'],
        absolute_nominal_error=record['absolute_nominal_error'],
        maximum_cells_per_term=record['maximum_cells_per_term'],maximum_terms=record['maximum_terms'],
        use_fourth_derivative=record['use_fourth_derivative'])
    if rebuilt!=record:
        raise ValueError('Saved actual-section integral does not reconstruct')
    return rebuilt
