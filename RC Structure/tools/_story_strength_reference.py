"""Independent first-order plastic-limit reference with a kinematic dual.

No repository imports. Member nominal capacities are frozen observations, not
independently calibrated properties. Planar lines are directional checks of the
3D candidate. Column axial redistribution is recovered and screened explicitly.
"""
from pathlib import Path
import argparse, collections, gzip, hashlib, json, math, time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix, hstack, vstack

def column_interaction(r,axis,knots=33):
    """Nominal strain-compatible P-M curve with circular concrete replacement.

    An inscribed, convex piecewise-linear domain is used by the LP. Dense
    numerical checks quantify interpolation/refinement rather than silently
    treating the polygon as the exact section curve.
    """
    s=r['sections'];re=r['reinforcement'];mat=r['materials']
    b=s['b_col_in'];h=s['h_col_in'];off=re['col_longitudinal_centroid_offset_in']
    a=re['col_bar_area_in2'];nt=re['col_top_bars'];nb=re['col_bot_bars'];ns=re['col_side_bars']
    coords=[(y,z) for count,z in [(nt,h-off),(nb,off)] for y in np.linspace(off,b-off,count)]
    coords += [(y,z) for y in (off,b-off) for z in np.linspace(off,h-off,ns+2)[1:-1]]
    if axis=='y':b,h=h,b;depth=np.array([h/2-(y-h/2) for y,z in coords])
    else:depth=np.array([h-z for y,z in coords])
    fc=s['fc_col_ksi'];fy=mat['fy_ksi'];es=mat['es_ksi'];beta=max(.65,.85-.05*(fc-4));rad=math.sqrt(a/math.pi)
    maximum=max(h/beta,max(depth)/(1-fy/(.003*es)))*1.01
    cs=np.unique(np.r_[np.linspace(1e-9,maximum,8001),depth/(1+fy/(.003*es)),depth/(1-fy/(.003*es)),(depth-rad)/beta,(depth+rad)/beta])
    ad=np.minimum(beta*cs,h)[:,None]
    z=np.clip((ad-depth)/rad,-1,1);sq=np.sqrt(np.maximum(0,1-z*z))
    area=rad*rad*(np.arcsin(z)+math.pi/2+z*sq)
    first=depth*area-(2/3)*rad**3*sq**3
    steel=a*np.clip(.003*es*(1-depth/cs[:,None]),-fy,fy)
    n=.85*fc*b*ad[:,0]+np.sum(steel-.85*fc*area,axis=1)
    m=.85*fc*b*ad[:,0]*(h/2-ad[:,0]/2)+np.sum(steel*(h/2-depth)-.85*fc*(h/2*area-first),axis=1)
    assert np.max(np.abs(np.sum(a*(h/2-depth))))<1e-10
    assert min(np.diff(n))>-1e-7
    keep=np.r_[True,np.diff(n)>1e-7];n=n[keep];m=m[keep]
    n[0]=-fy*a*len(depth);m[0]=0.;m[-1]=0.
    # Use the tied-column nominal axial cap, 0.80 Po, without a phi factor.
    # This also avoids convexifying the nonconcave near-pure-compression tail
    # across the low-axial-load range. Report the cap and audit its activity.
    po=.85*fc*(b*h-a*len(depth))+fy*a*len(depth)
    pmax=.8*po;at_cap=float(np.interp(pmax,n,m));inside=n<pmax
    n=np.r_[n[inside],pmax];m=np.r_[m[inside],at_cap]
    x=np.linspace(n[0],n[-1],knots);y=np.interp(x,n,m)
    # Build the upper convex envelope first. Intersecting every local secant
    # instead would extrapolate nonconvex portions across unrelated axial states
    # and would not approach the section locus as the grid is refined.
    hull=ConvexHull(np.vstack([np.c_[x,y],np.c_[x,-y]]))
    lines=[(-an/am,-constant/am) for an,am,constant in hull.equations if am>1e-10]
    polygon=np.min([slope*n+intercept for slope,intercept in lines],axis=0)
    # The ultimate-strain section locus need not be strictly concave. Keep a
    # numerically inner convex approximation rather than its outer convex hull.
    mask=polygon>1e-5
    scale=min(1.,float(np.min(np.maximum(m[mask],0)/polygon[mask])))
    scale*=1-1e-10
    lines=[(slope*scale,intercept*scale) for slope,intercept in lines]
    polygon*=scale
    excess=float(np.max(polygon-m))
    # Dense samples are a numerical audit, not a formal proof between samples.
    assert excess<.05,excess
    return dict(axis=axis,knots=knots,method='tied_axial_cap_then_inner_convex_polygon_v3',N=list(x),M=list(y),lines=lines,
                dense_N=n,dense_M=m,inner_polygon_scale=scale,max_dense_polygon_excess_kip_in=excess,
                largest_dense_capacity_loss_kip_in=float(np.max(m-polygon)),
                steel_area_in2=a*len(depth),bar_count=len(depth),pure_compression_Po_kip=po,nominal_axial_cap_kip=pmax)

def end_map(q,c,s,ai,aj):
    """Local face end actions to global joint-center actions."""
    q=np.array(q,dtype=float).copy()
    q[2]+=ai*q[1];q[5]-=aj*q[4]
    return np.array([c*q[0]-s*q[1],s*q[0]+c*q[1],q[2],
                     c*q[3]-s*q[4],s*q[3]+c*q[4],q[5]])

def local_statics(L):
    return np.array([[-1,0,0],[0,1/L,1/L],[0,1,0],
                     [1,0,0],[0,-1/L,-1/L],[0,0,1]],float)

def solve_pair(A,H,h,b,f,dual=True):
    """max lambda: A q=b+f lambda, Hq<=h; independently solve dual."""
    n=A.shape[1];m=A.shape[0]
    primary=linprog(np.r_[np.zeros(n),-1.],
        A_ub=hstack([H,csr_matrix((len(h),1))]),b_ub=h,
        A_eq=hstack([A,csr_matrix(-f[:,None])]),b_eq=b,
        bounds=[(None,None)]*n+[(0,None)],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not primary.success:
        return dict(success=False,message=primary.message)
    q=primary.x[:-1];load=primary.x[-1]
    eq=np.asarray(A@q-b-load*f)
    violation=max(0.,float(np.max(H@q-h)))
    result=dict(success=True,capacity_kip=float(load),q=q,
        equilibrium_residual=float(np.max(np.abs(eq))),yield_violation=violation)
    if dual:
        # Min h*z-b*u, A.T*u-H.T*z=0, f*u=1, z>=0.
        dual_eq=vstack([hstack([A.T,-H.T]),csr_matrix(np.r_[f,np.zeros(len(h))][None,:])])
        secondary=linprog(np.r_[-b,h],A_eq=dual_eq,b_eq=np.r_[np.zeros(n),1.],
            bounds=[(None,None)]*m+[(0,None)]*len(h),method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert secondary.success,secondary.message
        u=secondary.x[:m];z=secondary.x[m:]
        result.update(dual_capacity_kip=float(secondary.fun),dual_displacements=u,dual_flows=z,
            primal_dual_gap_kip=float(secondary.fun-load),
            compatibility_residual=float(np.max(np.abs(A.T@u-H.T@z))),
            load_work=float(f@u),
            plastic_work=float(h@z),gravity_work=float(b@u),
            complementarity_residual=float(np.max(np.abs(z*(h-H@q)))))
        assert abs(secondary.fun-load)<1e-6*max(1.,load)
        assert abs(f@u-1)<1e-8
        assert result['compatibility_residual']<1e-7
    assert result['equilibrium_residual']<1e-6
    assert violation<1e-6
    return result

def benchmark():
    """A fixed-base portal: compatible virtual work has known capacity."""
    # Rigid columns/beam with four end rotations as the only yield freedoms.
    # Build the same generic matrix assembly as used for multi-story frames.
    tests=[]
    for height,span,mc,mb in [(120.,240.,1000.,500.),(156.,192.,600.,1200.)]:
        # Free DOFs: common sway, left/right vertical displacements, rotations.
        lookup={(1,0,0):0,(1,1,0):0,(1,0,1):1,(1,1,1):2,(1,0,2):3,(1,1,2):4}
        elements=[((0,0),(1,0),height,0.,1.,mc,mc),
                  ((0,1),(1,1),height,0.,1.,mc,mc),
                  ((1,0),(1,1),span,1.,0.,mb,mb)]
        A=np.zeros((5,9));H=[];h=[]
        for ie,(ni,nj,L,c,s,m1,m2) in enumerate(elements):
            local=local_statics(L)
            for k in range(3):
                gl=end_map(local[:,k],c,s,0.,0.)
                for node,part in [(ni,gl[:3]),(nj,gl[3:])]:
                    for dof,val in enumerate(part):
                        if node[0]:A[lookup[(*node,dof)],3*ie+k]+=val
            for end,limit in [(1,m1),(2,m2)]:
                row=np.zeros(9);row[3*ie+end]=1
                H.extend([row,-row]);h.extend([limit,limit])
        ans=solve_pair(csr_matrix(A),csr_matrix(H),np.array(h),np.zeros(5),np.array([1.,0,0,0,0]))
        expected=2*(mc+min(mc,mb))/height
        assert math.isclose(ans['capacity_kip'],expected,rel_tol=1e-10)
        tests.append(dict(height_in=height,span_in=span,column_Mn_kip_in=mc,beam_Mn_kip_in=mb,
                          analytical_capacity_kip=expected,static_capacity_kip=ans['capacity_kip'],
                          kinematic_capacity_kip=ans['dual_capacity_kip']))
    return tests

class Reference:
    def __init__(self,r,cap,axis,line,gravity,pm=None):
        self.r=r;self.axis=axis;self.line=line;self.gravity=gravity
        g=r['geometry'];s=r['sections'];self.nf=g['num_floor'];self.np=g['num_bay_'+axis]+1
        lk='grid_j' if axis=='x' else 'grid_i';pk='grid_i' if axis=='x' else 'grid_j'
        joints={(j['floor'],j[pk]):j for j in r['joint_evidence']['joints'] if j[lk]==line}
        self.nd=self.nf*(1+2*self.np)
        self.dof={}
        for floor in range(1,self.nf+1):
            for point in range(self.np):
                self.dof[(floor,point,0)]=floor-1
                self.dof[(floor,point,1)]=self.nf+(floor-1)*2*self.np+2*point
                self.dof[(floor,point,2)]=self.nf+(floor-1)*2*self.np+2*point+1
        combos=[c for c in r['design_actions']['combinations'] if c['id'].startswith('seismic_'+gravity+'_gravity_X_')]
        assert len(combos)==4
        self.combos=combos
        def mean_force(tag):
            return np.mean([c['members'][str(tag)]['local_force_kip_kipin'] for c in combos],axis=0)
        def observation(tag):return combos[0]['members'][str(tag)]
        elems=[]
        for floor in range(1,self.nf+1):
            for point in range(self.np):
                joint=joints[(floor,point)];tag=next(c['tag'] for c in joint['columns'] if c['position']=='below')
                act=observation(tag);ai,aj=act['joint_face_offsets_in'];L=g['story_h_in']-ai-aj
                force=mean_force(tag)
                if axis=='x':qcenter=force[[2,0,4,8,6,10]]
                else:qcenter=np.array([-force[1],force[0],force[5],-force[7],force[6],force[11]])
                wi=act['axial_line_load_kip_per_in'];q0=np.array([wi*L/2,0,0,wi*L/2,0,0])
                lower=cap[(axis,tag,'i')];upper=cap[(axis,tag,'j')]
                vs=r['capacity_design']['columns']['hoops']['by_direction'][axis]
                limit=min(vs['av_in2']*r['materials']['fyt_ksi']*vs['d_in']/r['capacity_design']['columns']['hoops']['spacing_in'],r['capacity_design']['columns']['vs_limit_by_direction_kip'][axis])
                elems.append(dict(kind='column',tag=tag,story=floor,point=point,ni=(floor-1,point),nj=(floor,point),
                    L=L,c=0.,s=1.,ai=ai,aj=aj,q0=q0,qcenter=qcenter,
                    arm_external=np.array([0.,-wi*ai,0.,0.,-wi*aj,0.]),
                    limits=[(-lower['Mn'],lower['Mn']),(-upper['Mn'],upper['Mn'])],
                    axial_domain=[lower,upper],shear_limit=limit))
        endpoints={}
        for node,j in joints.items():
            for member in j['beams_'+axis]:endpoints.setdefault(member['tag'],{})[member['end']]=node
        for tag,nodes in endpoints.items():
            force=mean_force(tag);qcenter=np.array([force[0],force[2],-force[4],force[6],force[8],-force[10]])
            span=observation(tag)['span_bending'];Lc=span['length_in'];ai=aj=s['h_col_in' if axis=='x' else 'b_col_in']/2
            L=Lc-ai-aj;w=span['uniform_z_kip_per_in']
            point_lists=[c['members'][str(tag)]['span_bending']['point_z_loads'] for c in combos]
            assert all([p[0] for p in a]==[p[0] for p in point_lists[0]] for a in point_lists)
            points=[(p[0]*Lc,float(np.mean([a[i][1] for a in point_lists]))) for i,p in enumerate(point_lists[0])]
            W=-w*L;Q=-w*L*L/2
            arms=np.array([0.,w*ai,w*ai*ai/2,0.,w*aj,-w*aj*aj/2])
            inside=[]
            for x,p in points:
                if x<=ai:
                    arms[1]+=p;arms[2]+=p*x
                elif x>=Lc-aj:
                    arms[4]+=p;arms[5]+=p*(x-Lc)
                else:W-=p;Q-=p*(x-ai);inside.append((x-ai,p))
            q0=np.array([0,W-Q/L,0,0,Q/L,0])
            assert abs(arms[1]+arms[4]-W-(w*Lc+sum(p for x,p in points)))<1e-8
            # Check total applied load moment about the left joint center.
            body_moment=-(Q+ai*W)+arms[2]+arms[5]+arms[4]*Lc
            original=w*Lc*Lc/2+sum(x*p for x,p in points)
            assert abs(body_moment-original)<1e-7
            bs=lambda end,sign:r['beam_slab_strengths'][f'{tag}/{end}/{sign}']['mn_composite_kip_in']
            hd=r['capacity_design']['beams']['hoops'];offsets=r['reinforcement']['beam_bar_stacking']['offsets_in'][axis]
            depth=s['h_beam_in']-max(offsets.values())
            limit=min(hd['av_in2']*r['materials']['fyt_ksi']*depth/hd['spacing_in'],8*math.sqrt(s['fc_beam_ksi']*1000)*s['b_beam_in']*depth/1000)
            elems.append(dict(kind='beam',tag=tag,story=nodes['i'][0],point=nodes['i'][1],ni=nodes['i'],nj=nodes['j'],
                L=L,c=1.,s=0.,ai=ai,aj=aj,q0=q0,qcenter=qcenter,arm_external=arms,
                limits=[(-bs('i','positive'),bs('i','negative')),(-bs('j','negative'),bs('j','positive'))],
                shear_limit=limit,uniform=w,point_loads=inside,
                span_capacity_positive=min(bs('i','positive'),bs('j','positive')),
                span_capacity_negative=min(bs('i','negative'),bs('j','negative'))))
        self.elements=elems;n=3*len(elems)
        A=np.zeros((self.nd,n));p=np.zeros(self.nd);known=np.zeros(self.nd);H=[];h=[];labels=[]
        def project(vector,ni,nj):
            out=np.zeros(self.nd)
            for node,part in [(ni,vector[:3]),(nj,vector[3:])]:
                if node[0]:
                    for dof,value in enumerate(part):out[self.dof[(*node,dof)]]+=value
            return out
        for ie,e in enumerate(elems):
            local=local_statics(e['L']);e['local_matrix']=local
            for k in range(3):A[:,3*ie+k]=project(end_map(local[:,k],e['c'],e['s'],e['ai'],e['aj']),e['ni'],e['nj'])
            p+=project(e['qcenter']+e['arm_external'],e['ni'],e['nj'])
            known+=project(end_map(e['q0'],e['c'],e['s'],e['ai'],e['aj']),e['ni'],e['nj'])
            for j,(lo,hi) in enumerate(e['limits'],1):
                if e['kind']=='column' and pm is not None:
                    # Compression P=-N + q0_i at bottom, -N-q0_j at top.
                    p0=e['q0'][0] if j==1 else -e['q0'][3]
                    for sign,limit in [(1,pm['N'][-1]),(-1,-pm['N'][0])]:
                        a=np.zeros(n);a[3*ie]=-sign
                        H.append(a);h.append(limit-sign*p0)
                        labels.append((e['kind'],e['tag'],'axial',j,'compression' if sign==1 else 'tension'))
                    for slope,intercept in pm['lines']:
                        for sign in (1,-1):
                            a=np.zeros(n);a[3*ie]=slope;a[3*ie+j]=sign
                            H.append(a);h.append(intercept+slope*p0)
                            labels.append((e['kind'],e['tag'],'P-M',j,'positive' if sign==1 else 'negative'))
                    e['pm']=pm
                else:
                    a=np.zeros(n);a[3*ie+j]=1
                    H.extend([a,-a]);h.extend([hi,-lo]);labels.extend([(e['kind'],e['tag'],'moment',j,'positive'),(e['kind'],e['tag'],'moment',j,'negative')])
            for component in (1,4):
                a=np.zeros(n);a[3*ie:3*ie+3]=local[component]
                H.extend([a,-a]);h.extend([e['shear_limit']-e['q0'][component],e['shear_limit']+e['q0'][component]])
                labels.extend([(e['kind'],e['tag'],'shear',component,'positive'),(e['kind'],e['tag'],'shear',component,'negative')])
        self.A=csr_matrix(A);self.H=csr_matrix(H);self.h=np.array(h);self.b=p-known;self.labels=labels
        span_points=[]
        for ie,e in enumerate(elems):
            if e['kind']=='beam':
                xs=sorted(set([0.,e['L']]+[a for a,p in e['point_loads']]+[e['L']/2]))
                for x in xs:self.add_span_cut(ie,x)
                for i,x in enumerate(xs):
                    width=max(x-xs[i-1] if i else 0.,xs[i+1]-x if i<len(xs)-1 else 0.)
                    delta=abs(e['uniform'])*width*width/8
                    span_points.append((ie,x,delta if e['uniform']<0 else 0.,delta if e['uniform']>0 else 0.))
        self.outer_H=self.H.copy();self.outer_h=self.h.copy()
        for ie,x,positive_margin,negative_margin in span_points:
            self.add_span_cut(ie,x,positive_margin,negative_margin)
        self.maximum_span_interpolation_margin=max((max(p,n) for ie,x,p,n in span_points),default=0.)
        self.input_equilibrium=[]
        # Map frozen centerline actions to the new faces by statics, not by
        # copying the centerline moments into face hinges. Verify assembled loads.
        qf=[]
        for e in elems:
            # Global center forces back to local for extracting axial and moments.
            q=e['qcenter'];c=e['c'];s=e['s']
            local=np.array([c*q[0]+s*q[1],-s*q[0]+c*q[1],q[2],c*q[3]+s*q[4],-s*q[3]+c*q[4],q[5]])
            if e['kind']=='column':
                wi=observation(e['tag'])['axial_line_load_kip_per_in']
                face_i=local[0]-wi*e['ai'];face_j=local[3]-wi*e['aj']
                nforce=(face_j-face_i)/2
                mi=local[2]-e['ai']*local[1];mj=local[5]+e['aj']*local[4]
            else:
                # Remove forces/couples of end-zone loads from centerline free body.
                ni=local[:3]+e['arm_external'][:3];nj=local[3:]+e['arm_external'][3:]
                nforce=(nj[0]-ni[0])/2
                mi=ni[2]-e['ai']*ni[1];mj=nj[2]+e['aj']*nj[1]
            qf.extend([nforce,mi,mj])
        residual=np.max(np.abs(self.A@np.array(qf)-self.b))
        self.frozen_action_reconstruction_residual=float(residual)
        assert residual<1e-6,residual

    def add_span_cut(self,ie,x,positive_margin=0.,negative_margin=0.):
        e=self.elements[ie];a=np.zeros(self.A.shape[1]);a[3*ie+1]=-1+x/e['L'];a[3*ie+2]=x/e['L']
        constant=e['q0'][1]*x+e['uniform']*x*x/2+sum(p*(x-y) for y,p in e['point_loads'] if y<x)
        self.H=vstack([self.H,csr_matrix(np.array([a,-a]))],format='csr')
        self.h=np.r_[self.h,e['span_capacity_positive']-constant-positive_margin,e['span_capacity_negative']+constant-negative_margin]
        self.labels.extend([('beam',e['tag'],'span',x,'positive'),('beam',e['tag'],'span',x,'negative')])

    def solve(self,story,sway=1,pattern='story_couple',dual=True,cut_iteration=0):
        f=np.zeros(self.nd)
        if pattern=='story_couple':
            f[story-1]=sway
            if story>1:f[story-2]=-sway
        elif pattern=='uniform':f[:self.nf]=sway/self.nf
        elif pattern=='height':f[:self.nf]=sway*np.arange(1,self.nf+1)/sum(range(1,self.nf+1))
        ans=solve_pair(self.A,self.H,self.h,self.b,f,dual)
        if not ans['success']:return ans
        outer=solve_pair(self.A,self.outer_H,self.outer_h,self.b,f,False)
        assert outer['success']
        ans['continuous_span_upper_bound_kip']=outer['capacity_kip']
        ans['continuous_span_bound_gap_relative']=(outer['capacity_kip']-ans['capacity_kip'])/ans['capacity_kip']
        ans['maximum_span_interpolation_margin_kip_in']=self.maximum_span_interpolation_margin
        q=ans.pop('q');flow=ans.pop('dual_flows',None);u=ans.pop('dual_displacements',None)
        actions=[];outside=0;max_excess=0.;max_span_ratio=0.;cuts=[];max_pm_ratio=0.
        for ie,e in enumerate(self.elements):
            local=e['local_matrix']@q[3*ie:3*ie+3]+e['q0']
            item=dict(kind=e['kind'],tag=e['tag'],story=e['story'],point=e['point'],
                axial_i_kip=float(-local[0]) if e['kind']=='beam' else float(local[0]),
                axial_j_kip=float(local[3]) if e['kind']=='beam' else float(-local[3]),
                moment_i_kip_in=float(local[2]),moment_j_kip_in=float(local[5]),
                shear_i_kip=float(local[1]),shear_j_kip=float(local[4]))
            if e['kind']=='column':
                item['axial_domain']=e['axial_domain']
                for key,domain in zip(('axial_i_kip','axial_j_kip'),e['axial_domain']):
                    excess=max(domain['Nmin']-item[key],item[key]-domain['Nmax'],0.)
                    outside+=int(excess>1e-6);max_excess=max(max_excess,excess)
                if 'pm' in e:
                    for end,mindex in [('i',2),('j',5)]:
                        pforce=item['axial_'+end+'_kip'];curve=e['pm']
                        assert curve['N'][0]-1e-5<=pforce<=curve['N'][-1]+1e-5
                        capacity=np.interp(pforce,curve['dense_N'],curve['dense_M'])
                        max_pm_ratio=max(max_pm_ratio,abs(local[mindex])/max(capacity,1e-9))
            else:
                # Check the full span including stationary moments between loads.
                boundaries=sorted(set([0.,e['L']]+[x for x,p in e['point_loads']]))
                check=list(boundaries)
                for left,right in zip(boundaries,boundaries[1:]):
                    v=local[1]+sum(p for x,p in e['point_loads'] if x<=left)+e['uniform']*left
                    if e['uniform']:
                        root=left-v/e['uniform']
                        if left<root<right:check.append(root)
                for x in check:
                    m=-local[2]+local[1]*x+e['uniform']*x*x/2+sum(p*(x-a) for a,p in e['point_loads'] if a<x)
                    ratio=m/e['span_capacity_positive'] if m>=0 else -m/e['span_capacity_negative']
                    max_span_ratio=max(max_span_ratio,ratio)
                    if ratio>1+1e-8:cuts.append((ie,x))
            actions.append(item)
        assert not cuts,('Continuous-span certificate failed',max_span_ratio)
        yielded=[]
        if flow is not None:
            for label,value,slack in zip(self.labels,flow,self.h-self.H@q):
                if value>1e-8:yielded.append(dict(kind=label[0],tag=label[1],mode=label[2],component=label[3],sign=label[4],flow=float(value),slack=float(slack)))
        ans.update(axis=self.axis,line=self.line,gravity=self.gravity,story=story,sway=sway,pattern=pattern,
            column_ends_outside_frozen_axial_domain=outside,maximum_axial_domain_excess_kip=max_excess,
            maximum_beam_span_ratio=float(max_span_ratio),maximum_column_dense_pm_ratio=float(max_pm_ratio),
            adaptive_span_cut_iterations=cut_iteration,mechanism=yielded,member_actions=actions,
            frozen_action_reconstruction_residual=self.frozen_action_reconstruction_residual,
            mechanism_displacements=list(u) if u is not None else None)
        return ans

