"""Actual rectangular slab critical cuts and ACI318-19 local shear checks.

References: 8.4.4.1/.2 (printed107-108), 8.6.1.2 (111),
22.6.2/.3/.4 (412), Table22.6.5.2 and22.6.5.3 (416).
All forces are signed and contemporaneous; the same physical slab/web split
as the floor recovery is retained. Beam web forces are separately reported,
never relabeled as slab shear. A numerical field result alone does not assert
mesh qualification, development or adequate reinforcement.
"""
from __future__ import annotations

import math
import numpy as np
from Design.SMRF_Physical_Floor_Cuts import slab_line_integral,slab_tensor,_bounds

METHOD='actual_slab_critical_perimeter_aci31819_v2_code_field_and_peak_diagnostic'


def check_one_way_strength(*,fc_ksi,slab_thickness_in,cut_width_in,effective_depth_in,
                           developed_tension_area_in2,n_tension_kip,q_vertical_kip,
                           phi=.75,lambda_concrete=1.):
    """Table22.5.5.1(c), including the concurrent signed axial force.

    This is the actual slab part of a cut, without shear reinforcement. The
    supplied As must be developed longitudinal tension steel at this section;
    neither the beam web nor a nominal mat spacing supplies that evidence.
    N is tension positive here, opposite to the Code table's Nu convention.
    Retain both the axial-term cap and the total Vc cap, and floor Vc at zero.
    Analysis/mesh/steel-location/development qualification stays upstream.
    """
    fc,h,b,d,area,n,q,ph,lam=map(float,(fc_ksi,slab_thickness_in,cut_width_in,
        effective_depth_in,developed_tension_area_in2,n_tension_kip,q_vertical_kip,phi,lambda_concrete))
    if not all(math.isfinite(v) for v in (fc,h,b,d,area,n,q,ph,lam)) or min(fc,h,b,d,ph,lam)<=0 or area<0 or ph>1 or d>=h:
        raise ValueError('Finite physical slab dimensions, signed actions and nonnegative developed tension area required')
    if fc>10:raise ValueError('Unreinforced one-way slab shear requires fc<=10ksi under22.5.3.1')
    rho=area/(b*d);ls=min(1.,math.sqrt(2/(1+d/10)))
    root=math.sqrt(1000*fc)/1000
    axial_raw=-n/(6*b*h);axial=min(axial_raw,.05*fc)
    base=8*ls*lam*rho**(1/3)*root
    upper=5*lam*root;vc=max(0.,min(base+axial,upper));capacity=ph*vc*b*d
    return dict(method=METHOD+'/one_way_concurrent_N',status='pass' if abs(q)<=capacity else 'fail',
        basis='ACI318-19 Table22.5.5.1(c);22.5.3.1;22.5.5.1.1-.3',
        n_tension_kip=n,q_vertical_kip=q,rho_longitudinal=rho,lambda_s=ls,
        axial_stress_term_raw_ksi=axial_raw,axial_stress_term_ksi=axial,
        reinforcement_stress_term_ksi=base,vc_upper_limit_ksi=upper,vc_ksi=vc,
        phi=ph,phi_vc_kip=capacity,developed_tension_area_in2=area,
        utilization=abs(q)/capacity if capacity else (0. if q==0 else None),
        zero_shear_capacity=capacity==0.,qualification_asserted=False)


def critical_perimeter(footprint,*,column_x_in,column_y_in,column_h_x_in,column_b_y_in,
                       effective_depth_x_in,effective_depth_y_in):
    """Minimum rectangular/U/L trace at d/2, clipped at actual free edges.

Average orthogonal effective depth follows22.6.2.1. Every remaining concrete
side is counted; an edge coinciding with the physical free edge carries no
critical shear area. Openings/drop panels/shear caps require additional paths
and are outside this uniform-slab geometry helper.
"""
    x,y,h,b,dx,dy=map(float,(column_x_in,column_y_in,column_h_x_in,column_b_y_in,
                           effective_depth_x_in,effective_depth_y_in))
    if not all(math.isfinite(v) for v in (x,y,h,b,dx,dy)) or min(h,b,dx,dy)<=0:raise ValueError('Finite actual column and orthogonal effective depths required')
    d=(dx+dy)/2;offset=d/2
    fx0,fx1,fy0,fy1=[float(footprint[k]) for k in ('x_min_in','x_max_in','y_min_in','y_max_in')]
    if not fx0<=x<=fx1 or not fy0<=y<=fy1:raise ValueError('Column center lies outside the physical slab')
    raw=(x-h/2-offset,x+h/2+offset,y-b/2-offset,y+b/2+offset)
    x0,x1,y0,y1=max(fx0,raw[0]),min(fx1,raw[1]),max(fy0,raw[2]),min(fy1,raw[3])
    sides=[]
    for axis,station,interval,normal,free in (
            ('x',x0,[y0,y1],[-1.,0.],x0<=fx0+1e-9),
            ('x',x1,[y0,y1],[1.,0.],x1>=fx1-1e-9),
            ('y',y0,[x0,x1],[0.,-1.],y0<=fy0+1e-9),
            ('y',y1,[x0,x1],[0.,1.],y1>=fy1-1e-9)):
        if not free:sides.append(dict(axis=axis,station_in=station,interval_in=interval,outward_normal=normal))
    if len(sides) not in (2,3,4):raise ValueError('Only actual interior, edge or corner critical sections are supported')
    # Two surviving parallel faces would be a strip, not a corner connection.
    if len(sides)==2 and sides[0]['axis']==sides[1]['axis']:raise ValueError('Two parallel critical faces are outside the corner-column domain')
    bo=sum(s['interval_in'][1]-s['interval_in'][0] for s in sides)
    center=np.zeros(2)
    for side in sides:
        a,z=side['interval_in'];p=[side['station_in'],(a+z)/2] if side['axis']=='x' else [(a+z)/2,side['station_in']]
        center+=(z-a)*np.array(p)
    center/=bo;cov=np.zeros((2,2));torsion=np.zeros((2,2));vertices=[]
    for side in sides:
        a,z=side['interval_in'];length=z-a;axis=side['axis']
        tangent=np.array([0.,1.]) if axis=='x' else np.array([1.,0.])
        midpoint=np.array([side['station_in'],(a+z)/2] if axis=='x' else [(a+z)/2,side['station_in']])
        r=midpoint-center
        cov+=d*(length*np.outer(r,r)+length**3/12*np.outer(tangent,tangent))
        # ACI R8.4.4.2.3 Jc includes the finite vertical wall's d^3/12
        # contribution on sides parallel to the moment-gradient direction.
        torsion+=length*d**3/12*np.outer(tangent,tangent)
        vertices.extend([(midpoint-tangent*length/2).tolist(),(midpoint+tangent*length/2).tolist()])
    jc=cov+torsion
    if np.linalg.eigvalsh(jc).min()<=0:raise ValueError('Critical section has singular moment-transfer geometry')
    return dict(method=METHOD,sides=sides,bounds_in=[x0,x1,y0,y1],effective_depth_in=d,
                bo_in=bo,shear_area_in2=bo*d,centroid_in=center.tolist(),
                covariance_in4=cov.tolist(),wall_torsion_in4=torsion.tolist(),Jc_in4=jc.tolist(),
                vertices_in=vertices,column_dimensions_in=[h,b],column_center_in=[x,y],
                support_type={4:'interior',3:'edge',2:'corner'}[len(sides)],
                alpha_s={4:40.,3:30.,2:20.}[len(sides)])


def recover_slab_perimeter(response,perimeter,*,trace):
    """Exact bilinear slab stress integration around a physical critical cut.

trace='inside'/'outside' selects the indicated element trace at every face.
Return all signed shear and bending/twisting resultants. The physical moments
about the perimeter centroid include both Q*r and actual shell edge couples.
Linear fields have their exact absolute maximum at each clipped segment end.
"""
    if trace not in ('inside','outside'):raise ValueError('Explicit inside/outside physical perimeter trace required')
    elements=response['elements'];byid={r['id']:r for r in elements}
    if len(byid)!=len(elements):raise ValueError('Duplicate physical element identity')
    cx,cy=perimeter['centroid_in'];total=np.zeros(3);sides=[];maximum=0.;peaks=[]
    for side in perimeter['sides']:
        nx,ny=side['outward_normal'];axis=side['axis'];normal=nx or ny
        direction=('minus' if normal>0 else 'plus') if trace=='inside' else ('plus' if normal>0 else 'minus')
        cut=slab_line_integral(elements,axis=axis,station_in=side['station_in'],interval_in=side['interval_in'],trace=direction)
        r=np.array(cut['integral']);first=np.array(cut['transverse_first_moment'])
        q=nx*r[6]+ny*r[7]
        if axis=='x':
            xq=(side['station_in']-cx)*q;yq=nx*(first[6]-cy*r[6])
        else:
            xq=ny*(first[7]-cx*r[7]);yq=(side['station_in']-cy)*q
        mx=yq-(nx*r[5]+ny*r[4]);my=-xq+(nx*r[3]+ny*r[5])
        total+=np.array([q,mx,my])
        for p in cut['pieces']:
            element=byid[p['element_id']]
            for at in p['interval_in']:
                x,y=(side['station_in'],at) if axis=='x' else (at,side['station_in'])
                tensor=slab_tensor(element,x,y);value=(nx*tensor[6]+ny*tensor[7])/perimeter['effective_depth_in']
                peaks.append(dict(element_id=element['id'],xy_in=[x,y],signed_shear_stress_ksi=float(value)))
                maximum=max(maximum,abs(float(value)))
        sides.append(dict(**side,cut=cut,resultant_kip_kipin=[float(q),float(mx),float(my)]))
    return dict(method=METHOD,trace=trace,perimeter=perimeter,sides=sides,
                slab_resultant_kip_kipin=total.tolist(),maximum_direct_slab_shear_ksi=maximum,
                extrema=peaks,qualification_asserted=False)


def check_two_way_strength(perimeter,*,fc_ksi,slab_resultant_kip_kipin,
                           direct_shear_peak_ksi,phi=.75,lambda_concrete=1.):
    """ACI critical-section shear field, with the raw FE peak kept separately.

No gamma_f enhancement (Table8.4.2.2.4) is taken. The full biaxial Jc matrix
retains centroidal coupling at edge/corner supports. Strength is the least of
all three Table22.6.5.2 expressions, including the 100psi sqrt(fc) cap.
The returned code check uses the prescribed critical-section field. An
unsmoothed FE point peak is a refinement diagnostic, not an additional ACI
point-stress limit. Its old conservative maximum/ratio remains explicit.
The local check still requires model/mesh/development and gamma_f*Msc
flexural checks under the production contract.
"""
    fc,phi,lam,peak=map(float,(fc_ksi,phi,lambda_concrete,direct_shear_peak_ksi))
    force=np.asarray(slab_resultant_kip_kipin,float)
    if force.shape!=(3,) or not np.isfinite(force).all() or not all(math.isfinite(v) for v in (fc,phi,lam,peak)) or min(fc,phi,lam)<=0 or peak<0 or phi>1:
        raise ValueError('Finite actual signed slab V/Mx/My and material design factors required')
    d=perimeter['effective_depth_in'];bo=perimeter['bo_in'];h,b=perimeter['column_dimensions_in'];beta=max(h,b)/min(h,b)
    ls=min(1.,math.sqrt(2/(1+d/10)));root=min(math.sqrt(fc*1000),100.)/1000
    terms=[4*ls*lam*root,(2+4/beta)*ls*lam*root,(2+perimeter['alpha_s']*d/bo)*ls*lam*root]
    vc=min(terms);capacity=phi*vc
    x0,x1,y0,y1=perimeter['bounds_in'];bx,by=x1-x0,y1-y0
    gf_x=1/(1+2/3*math.sqrt(bx/by));gf_y=1/(1+2/3*math.sqrt(by/bx))
    # Qz at(x,y) gives physical Mx=y*Q, My=-x*Q. Gradients are solved
    # in global[x,y] coordinates with the actual unsymmetric section Jc.
    demand=np.array([-(1-gf_x)*force[2],(1-gf_y)*force[1]])
    gradient=np.linalg.solve(np.asarray(perimeter['Jc_in4']),demand)
    center=np.asarray(perimeter['centroid_in']);vuv=force[0]/(bo*d)
    stresses=[float(vuv+gradient@(np.array(p)-center)) for p in perimeter['vertices_in']]
    code_peak=max(map(abs,stresses));governing=max(peak,code_peak)
    return dict(method=METHOD,status='pass' if code_peak<=capacity else 'fail',phi=phi,
                concrete_strengths_ksi=terms,vc_ksi=vc,phi_vc_ksi=capacity,lambda_s=ls,
                code_peak_shear_ksi=code_peak,direct_peak_shear_ksi=peak,governing_shear_ksi=code_peak,
                utilization=code_peak/capacity,vuv_ksi=abs(float(vuv)),
                conservative_peak_envelope_ksi=governing,conservative_peak_utilization=governing/capacity,
                conservative_peak_status='pass' if governing<=capacity else 'fail',
                direct_peak_role='unsmoothed physical FE stress trace; numerical refinement diagnostic',
                gamma_f={'x':gf_x,'y':gf_y},flexural_transfer_required_kipin={'x':gf_x*force[2],'y':gf_y*force[1]},
                signed_vertex_shear_ksi=stresses,linear_shear_gradient_ksi_per_in=gradient.tolist(),
                moment_transfer_residual_kipin=(np.asarray(perimeter['Jc_in4'])@gradient-demand).tolist(),
                basis='ACI318-19 8.4.4.2;22.6.2.1;22.6.3.1;Table22.6.5.2',qualification_asserted=False)


def local_minimum_steel(perimeter,footprint,*,slab_thickness_in,fc_ksi,fy_ksi,vuv_ksi,
                        provided_developed_area_in2,phi=.75,lambda_concrete=1.):
    """Actual 8.6.1.2 trigger and integer developed steel in each local band.

provided_developed_area_in2 explicitly contains x_top/x_bottom/y_top/y_bottom.
The API checks both faces for load reversal; the production caller may mark a
face not required only with its complete signed concurrent local moment proof.
This function never assumes that a nominal spacing implies two integrity bars.
"""
    h,fc,fy,vuv=map(float,(slab_thickness_in,fc_ksi,fy_ksi,vuv_ksi))
    if not all(math.isfinite(v) for v in (h,fc,fy,vuv,phi,lambda_concrete)) or min(h,fc,fy,phi,lambda_concrete)<=0 or vuv<0 or phi>1:
        raise ValueError('Actual finite local slab/material/shear inputs required')
    d=perimeter['effective_depth_in'];ls=min(1.,math.sqrt(2/(1+d/10)))
    trigger=phi*2*ls*lambda_concrete*min(math.sqrt(fc*1000),100.)/1000
    cx,cy=perimeter['column_center_in'];hx,by=perimeter['column_dimensions_in']
    bands={'x':[max(footprint['y_min_in'],cy-by/2-1.5*h),min(footprint['y_max_in'],cy+by/2+1.5*h)],
           'y':[max(footprint['x_min_in'],cx-hx/2-1.5*h),min(footprint['x_max_in'],cx+hx/2+1.5*h)]}
    rows={}
    for axis in ('x','y'):
        bslab=bands[axis][1]-bands[axis][0]
        required=5*vuv*bslab*perimeter['bo_in']/(phi*perimeter['alpha_s']*fy) if vuv>trigger else 0.
        for face in ('top','bottom'):
            name=f'{axis}_{face}';provided=provided_developed_area_in2.get(name)
            if provided is None:status='pass' if required==0 else 'unknown'
            else:
                provided=float(provided)
                if not math.isfinite(provided) or provided<0:raise ValueError('Developed physical steel area must be nonnegative finite or unknown')
                status='pass' if provided>=required else 'fail'
            rows[name]=dict(status=status,band_in=bands[axis],bslab_in=bslab,required_in2=required,provided_in2=provided)
    return dict(method=METHOD,trigger_shear_ksi=trigger,vuv_ksi=vuv,triggered=vuv>trigger,checks=rows,
                status='unknown' if any(v['status']=='unknown' for v in rows.values()) else 'fail' if any(v['status']=='fail' for v in rows.values()) else 'pass',
                basis='ACI318-19 8.6.1.2;Table8.4.2.2.3;22.6.5.3',qualification_asserted=False)
