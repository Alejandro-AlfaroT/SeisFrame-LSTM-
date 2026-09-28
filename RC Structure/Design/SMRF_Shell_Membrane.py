"""Equilibrium kernel for orthogonal reinforcement in a concrete membrane.

This is the minimum-sum tensile-force solution of a compression-only concrete
stress field. It is a mechanics kernel, not an ACI strength or detailing check.
No strength-reduction factor, concrete softening law, minimum steel, strain
compatibility, shear capacity or shell-layer iteration is supplied here.

Basis: Colombo, Della Bella and Bittencourt (2014), section 2.2, cases I-IV,
https://doi.org/10.1590/S1983-41952014000100004 . Actions are per inch of cut.
"""
from __future__ import annotations
import math

METHOD_VERSION='orthogonal_membrane_equilibrium_v2_actual_steel_depths'


def _three(values,name):
    if not isinstance(values,(list,tuple)) or len(values)!=3:
        raise ValueError(f'{name} requires [xx, yy, xy]')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError(f'{name} requires finite numeric components')
    return tuple(float(v) for v in values)


def membrane_equilibrium(resultants_kip_per_in):
    """Return orthogonal steel tensions and the balancing concrete tensor.

    Nxx and Nyy are tension positive; Nxy is signed tensor shear (no factor 2).
    The concrete tensor must be negative semidefinite. Tension forces returned
    for the reinforcement are NOT required areas until a material/design basis
    and the actual reinforcement depths have been established.
    """
    nx,ny,nxy=_three(resultants_kip_per_in,'Membrane actions');q=abs(nxy)
    mean=(nx+ny)/2;radius=math.hypot((nx-ny)/2,nxy)
    if mean+radius<=0:
        tx=ty=0.;branch='biaxial_compression'
    elif q==0:
        tx,ty=max(nx,0.),max(ny,0.);branch='uncoupled_normals'
    elif nx+q>=0 and ny+q>=0:
        tx,ty=nx+q,ny+q;branch='both_directions'
    elif nx+q<0:
        tx,ty=0.,ny-nxy*nxy/nx;branch='y_only'
    else:
        tx,ty=nx-nxy*nxy/ny,0.;branch='x_only'
    cx,cy=nx-tx,ny-ty
    cm=(cx+cy)/2;cr=math.hypot((cx-cy)/2,nxy)
    principal=[cm-cr,cm+cr]
    scale=max(1.,abs(nx),abs(ny),q)
    if min(tx,ty)<-1e-12*scale or principal[1]>1e-12*scale:
        raise RuntimeError('Membrane equilibrium produced an inadmissible stress field')
    return dict(method=METHOD_VERSION,branch=branch,actions_kip_per_in=[nx,ny,nxy],
                steel_tension_kip_per_in=[tx,ty],concrete_tensor_kip_per_in=[cx,cy,nxy],
                concrete_principal_kip_per_in=principal,
                engineering_verified=False,capacity_pairing_authorized=False)


def shell_layer_resultants(normal_kip_per_in,moment_kip_in_per_in,*,top_center_in,bottom_center_in):
    """Split shell N and M between declared concrete membrane centroids.

    The reference is the shell midplane. Positive z points upward. Distances
    ht and hb are positive; positive M puts the bottom layer in tension:
    N=Nt+Nb and M=-ht*Nt+hb*Nb. Actual steel need not lie at these centroids;
    do not use these layer forces as final reinforcement demands without the
    steel-depth correction and compression-layer iteration.
    """
    n=_three(normal_kip_per_in,'Shell normal actions');m=_three(moment_kip_in_per_in,'Shell moments')
    for v in (top_center_in,bottom_center_in):
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError('Membrane centroid distances must be finite and positive')
    ht,hb=top_center_in,bottom_center_in;lever=ht+hb
    top=[(a*hb-b)/lever for a,b in zip(n,m)]
    bottom=[(a*ht+b)/lever for a,b in zip(n,m)]
    return dict(top_kip_per_in=top,bottom_kip_per_in=bottom,
                centroid_distances_in=[ht,hb],reinforcement_depth_corrected=False,
                engineering_verified=False,capacity_pairing_authorized=False)


def correct_steel_depths(top_kip_per_in,bottom_kip_per_in,*,top_center_in,bottom_center_in,
                        top_steel_centers_in,bottom_steel_centers_in,max_iterations=100):
    """Rebalance fixed concrete layers at the actual two orthogonal bar depths.

    Centroid distances are positive magnitudes from the shell midplane. When
    only one face needs steel, changing its lever arm requires changing the
    opposite concrete membrane and recomputing its reinforcement. Negative
    steel corrections trigger that same recalculation; they are not clipped
    while keeping the old concrete force. This fixed-layer operation does not
    select a concrete strength or iterate the compression-layer thickness.
    """
    nt=_three(top_kip_per_in,'Top membrane');nb=_three(bottom_kip_per_in,'Bottom membrane')
    ht,hb=top_center_in,bottom_center_in
    distances=[ht,hb]
    for depths in (top_steel_centers_in,bottom_steel_centers_in):
        if not isinstance(depths,(list,tuple)) or len(depths)!=2:raise ValueError('Two steel depths required per face')
        distances.extend(depths)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in distances):
        raise ValueError('Layer and steel centroid distances must be finite and positive')
    if type(max_iterations) is not int or not 1<=max_iterations<=1000:raise ValueError('Invalid iteration budget')
    dt=[0.,0.];db=[0.,0.];scale=max(1.,*(abs(v) for v in nt+nb))
    for iteration in range(1,max_iterations+1):
        top=membrane_equilibrium([nt[0]+dt[0],nt[1]+dt[1],nt[2]])
        bottom=membrane_equilibrium([nb[0]+db[0],nb[1]+db[1],nb[2]])
        new_dt=[0.,0.];new_db=[0.,0.];tt=[];tb=[]
        for axis,(t,b,st,sb) in enumerate(zip(top['steel_tension_kip_per_in'],bottom['steel_tension_kip_per_in'],
                                             top_steel_centers_in,bottom_steel_centers_in)):
            ta=(t*(ht+sb)+b*(sb-hb))/(st+sb);ba=t+b-ta
            if t<=0 or ta<0:
                ta=0.;ba=b*(ht+hb)/(ht+sb);new_dt[axis]=b-ba
            elif b<=0 or ba<0:
                ba=0.;ta=t*(ht+hb)/(st+hb);new_db[axis]=t-ta
            tt.append(ta);tb.append(ba)
        difference=max(abs(a-b) for a,b in zip(dt+db,new_dt+new_db))
        if difference<=1e-11*scale:break
        dt,db=new_dt,new_db
    else:raise RuntimeError('Actual steel-depth correction did not converge')
    ct=top['concrete_tensor_kip_per_in'];cb=bottom['concrete_tensor_kip_per_in']
    actual_n=[ct[k]+cb[k]+(tt[k]+tb[k] if k<2 else 0.) for k in range(3)]
    actual_m=[-ht*ct[k]+hb*cb[k]+(-top_steel_centers_in[k]*tt[k]+bottom_steel_centers_in[k]*tb[k] if k<2 else 0.) for k in range(3)]
    expected_n=[a+b for a,b in zip(nt,nb)];expected_m=[-ht*a+hb*b for a,b in zip(nt,nb)]
    residual=max([abs(a-b)/scale for a,b in zip(actual_n,expected_n)]+
                 [abs(a-b)/(scale*(ht+hb)) for a,b in zip(actual_m,expected_m)])
    if not math.isfinite(residual) or residual>1e-9 or min(tt+tb)<0:
        raise RuntimeError('Actual steel-depth correction failed shell equilibrium')
    return dict(method=METHOD_VERSION,top_steel_tension_kip_per_in=tt,bottom_steel_tension_kip_per_in=tb,
                concrete_centroid_distances_in=[ht,hb],top_steel_centers_in=list(top_steel_centers_in),
                bottom_steel_centers_in=list(bottom_steel_centers_in),
                top_concrete=top,bottom_concrete=bottom,iterations=iteration,
                top_membrane_correction_kip_per_in=dt,bottom_membrane_correction_kip_per_in=db,
                reconstructed_normal_kip_per_in=actual_n,reconstructed_moment_kip_in_per_in=actual_m,
                normalized_equilibrium_residual=residual,compression_layer_thickness_iterated=False,
                engineering_verified=False,capacity_pairing_authorized=False)
