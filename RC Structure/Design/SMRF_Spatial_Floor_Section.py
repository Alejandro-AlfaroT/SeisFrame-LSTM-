"""Seven-resultant common slab/web section with actual transverse bar positions.

q=[ex,ey,gxy,kx,ky,kxy,ks]. Only the beam-axis normal strain additionally
contains ks*(s-sref). Its seventh conjugate resultant is int((s-sref)*sigma_b
dA)/strip_width. Other six conventions match SMRF_Layered_Slab_Section.
Slab and downstand occupy disjoint actual rectangles. Finite longitudinal
bar circles displace concrete once; transverse bar circles are homogenized
over an EXPLICIT longitudinal accounting length with their actual run segments.
No strong/weak capacity sums or duplicate flange steel credit are used.

This is a physical section constitutive/equilibrium kernel. Actual action
recovery, complete spatial coverage, development and transverse shear remain
separate. Inverse failure means unknown, never proved capacity exceedance.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares,minimize
from Design.SMRF_Layered_Slab_Section import concrete_response,design_concrete_peak_strain

METHOD='actual_spatial_common_slab_web_seven_resultants_v2_external_slab_voids'
SOLVER_METHOD='bounded_seven_resultant_inverse_v4_shear_active_warm_start'

@lru_cache(maxsize=16)
def _gauss(order):
    nodes,weights=leggauss(order);nodes.setflags(write=False);weights.setflags(write=False)
    return nodes,weights


def _positive(x,name):
    x=float(x)
    if not math.isfinite(x) or x<=0:raise ValueError(f'Positive finite {name} required')
    return x


def _interval(x,name):
    a=np.asarray(x,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all() or a[1]<=a[0]:raise ValueError(f'Positive finite {name} interval required')
    return (float(a[0]),float(a[1]))


def _positive_affine_rectangle(sa,sb,za,zb,constant,slope_s,slope_z):
    """Exact integral max(c+as+bz,0) by half-plane polygon clipping."""
    points=[(sa,za),(sb,za),(sb,zb),(sa,zb)];out=[]
    def value(p):return constant+slope_s*p[0]+slope_z*p[1]
    for a,b in zip(points,points[1:]+points[:1]):
        va,vb=value(a),value(b)
        if va>=0:out.append(a)
        if (va<0<=vb) or (vb<0<=va):
            ratio=va/(va-vb);out.append((a[0]+ratio*(b[0]-a[0]),a[1]+ratio*(b[1]-a[1])))
    if len(out)<3:return 0.
    area=ms=mz=0.
    for a,b in zip(out,out[1:]+out[:1]):
        cross=a[0]*b[1]-b[0]*a[1];area+=cross/2;ms+=(a[0]+b[0])*cross/6;mz+=(a[1]+b[1])*cross/6
    return max(0.,constant*area+slope_s*ms+slope_z*mz)


class SpatialFloorSection:
    def __init__(self,thickness_in,fc_ksi,*,ec_ksi,axis,slab_interval_in,
                 web,reference_s_in,longitudinal_bars,transverse_rows,
                 phi_requirements,concrete_peak_strain=None,steel_strain_limit=.01,
                 quadrature_order=12,slab_void_rectangles=()):
        self.h=_positive(thickness_in,'slab thickness');self.fc=_positive(fc_ksi,'fc')
        ec=_positive(ec_ksi,'Ec');self.sref=float(reference_s_in)
        if not math.isfinite(self.sref):raise ValueError('Finite transverse reference required')
        if axis not in ('x','y'):raise ValueError('Actual beam x or y axis required')
        self.axis=axis;self.j=0 if axis=='x' else 1;self.other=1-self.j
        self.slab=_interval(slab_interval_in,'physical slab');self.width=self.slab[1]-self.slab[0]
        self.regions=[dict(id='slab',s=self.slab,z=(-self.h/2,self.h/2))]
        if web is not None:
            ws=_interval(web['interval_in'],'actual web');wz=_interval(web['z_interval_in'],'web depth')
            if ws[0]<self.slab[0]-1e-10 or ws[1]>self.slab[1]+1e-10 or wz[1]>-self.h/2+1e-10:
                raise ValueError('Actual web must be below slab and within the physical strip')
            self.regions.append(dict(id=str(web.get('id','web')),s=ws,z=wz))
        if not phi_requirements or any(not 0<float(v)<=1 for v in phi_requirements.values()):
            raise ValueError('Explicit positive applicable phi factors required')
        self.phi=min(.65,*[float(v) for v in phi_requirements.values()]);self.phi_requirements=dict(phi_requirements)
        self.peak=(design_concrete_peak_strain(self.fc,ec)['peak_strain'] if concrete_peak_strain is None
                   else _positive(concrete_peak_strain,'source peak'))
        if self.peak>.003:raise ValueError('Declared peak cannot exceed.003')
        self.steel_limit=_positive(steel_strain_limit,'steel strain limit')
        if not isinstance(quadrature_order,int) or not 4<=quadrature_order<=32:raise ValueError('Quadrature order4..32required')
        self.order=quadrature_order;self.longitudinal=[];self.transverse=[];ids=set()
        def steel(raw):
            r=dict(raw)
            if not isinstance(r.get('id'),str) or not r['id'] or r['id'] in ids:raise ValueError('Unique actual physical bar/row IDs required')
            ids.add(r['id'])
            for key in ('area_in2','fy_ksi'):r[key]=_positive(r[key],key)
            r['es_ksi']=_positive(r.get('es_ksi',29000.),'Es');r['z_in']=float(r['z_in'])
            if not math.isfinite(r['z_in']):raise ValueError('Finite actual bar elevation required')
            r['radius']=math.sqrt(r['area_in2']/math.pi)
            r['physical_radius']=_positive(r.get('db_in',2*r['radius']),'actual physical bar diameter')/2
            return r
        for raw in longitudinal_bars:
            r=steel(raw);r['s_in']=float(r['s_in'])
            r['strength_credited']=raw.get('strength_credited',True)
            if not isinstance(r['strength_credited'],bool):raise ValueError('Explicit boolean longitudinal strength credit required')
            if not math.isfinite(r['s_in']):raise ValueError('Finite actual transverse bar coordinate required')
            radius=r['physical_radius'];slo,shi=r['s_in']-radius,r['s_in']+radius
            zlo,zhi=r['z_in']-radius,r['z_in']+radius
            contained=any(slo>=v['s'][0]-1e-10 and shi<=v['s'][1]+1e-10 and
                          zlo>=v['z'][0]-1e-10 and zhi<=v['z'][1]+1e-10 for v in self.regions)
            # A real longitudinal bar may straddle the disjoint slab/web
            # boundary; subtract its respective portions from each region.
            if not contained and len(self.regions)==2:
                w=self.regions[1]
                contained=(abs(w['z'][1]+self.h/2)<1e-10 and slo>=w['s'][0]-1e-10 and shi<=w['s'][1]+1e-10
                           and zlo>=w['z'][0]-1e-10 and zhi<=self.h/2+1e-10)
            if not contained:raise ValueError('Actual longitudinal bar circle lies outside the physical concrete union')
            for old in self.longitudinal:
                if math.hypot(r['s_in']-old['s_in'],r['z_in']-old['z_in'])<radius+old['physical_radius']-1e-10:
                    raise ValueError('Actual longitudinal bar circles overlap')
            self.longitudinal.append(r)
        for raw in transverse_rows:
            r=steel(raw);r['run_interval_in']=_interval(r['run_interval_in'],'actual transverse run')
            credited=raw.get('strength_run_interval_in',r['run_interval_in'])
            if credited is None:r['strength_run_interval_in']=None
            else:
                r['strength_run_interval_in']=_interval(credited,'fully developed transverse strength run')
                if credited[0]<r['run_interval_in'][0]-1e-10 or credited[1]>r['run_interval_in'][1]+1e-10:
                    raise ValueError('Credited strength run must lie within installed physical run')
            physical=r.get('physical_bar_ids')
            if not isinstance(physical,(list,tuple)) or not physical or len(set(physical))!=len(physical):
                raise ValueError('Transverse row requires actual integer physical bar IDs')
            if ids.intersection(physical):raise ValueError('Physical steel credited in two directions/rows')
            ids.update(physical)
            r['accounting_length_in']=_positive(r['accounting_length_in'],'actual longitudinal accounting length')
            r['count_per_length_in']=len(physical)/r['accounting_length_in']
            if r['run_interval_in'][0]<self.slab[0]-1e-10 or r['run_interval_in'][1]>self.slab[1]+1e-10:
                raise ValueError('Transverse physical run must be clipped to this actual strip')
            if r['z_in']-r['physical_radius'] < -self.h/2-1e-10 or r['z_in']+r['physical_radius']>self.h/2+1e-10:
                raise ValueError('Transverse mat circle extends beyond slab depth')
            if 2*r['physical_radius']*r['count_per_length_in']>=1:raise ValueError('Transverse bars exhaust longitudinal spacing')
            for old in self.longitudinal:
                if r['run_interval_in'][0]-old['physical_radius']<old['s_in']<r['run_interval_in'][1]+old['physical_radius'] and abs(r['z_in']-old['z_in'])<r['physical_radius']+old['physical_radius']-1e-10:
                    raise ValueError('Orthogonal actual bars intersect; actual stacked elevations required')
            self.transverse.append(r)
        if not self.longitudinal and not self.transverse:raise ValueError('Actual physical reinforcement required')
        self.slab_voids=[]
        for raw in slab_void_rectangles:
            ident=raw.get('id')
            if not isinstance(ident,str) or not ident or ident in ids:
                raise ValueError('Unique externally owned slab void identity required')
            ids.add(ident);ss=_interval(raw['interval_in'],'external slab void');zz=_interval(raw['z_interval_in'],'external slab void depth')
            if ss[0]<self.slab[0]-1e-10 or ss[1]>self.slab[1]+1e-10 or zz[0]<-self.h/2-1e-10 or zz[1]>self.h/2+1e-10:
                raise ValueError('External slab void must lie within the physical slab section')
            for bar in self.longitudinal:
                ds=max(ss[0]-bar['s_in'],0.,bar['s_in']-ss[1]);dz=max(zz[0]-bar['z_in'],0.,bar['z_in']-zz[1])
                if math.hypot(ds,dz)<bar['radius']-1e-10:
                    raise ValueError('Nominal external slab void overlaps a longitudinal bar void')
            for row in self.transverse:
                if min(ss[1],row['run_interval_in'][1])>max(ss[0],row['run_interval_in'][0])+1e-10 and min(zz[1],row['z_in']+row['radius'])>max(zz[0],row['z_in']-row['radius'])+1e-10:
                    raise ValueError('Nominal external slab void overlaps a transverse layer void')
            for prior in self.slab_voids:
                if min(ss[1],prior['s'][1])>max(ss[0],prior['s'][0])+1e-10 and min(zz[1],prior['z'][1])>max(zz[0],prior['z'][0])+1e-10:
                    raise ValueError('Nominal external slab void rectangles overlap')
            self.slab_voids.append(dict(id='slab',physical_void_id=ident,s=ss,z=zz))

    def component_capacity_upper_bounds(self):
        """Relaxed design bounds; no bar void subtraction or strain compatibility.

        Concrete normal stresses are compression-only and bounded by fc;
        |tau|<=fc/2. Gross concrete and independent +/-fy steel enlarge the
        possible resultant set. These loose bounds can prove an exceedance,
        never prove adequacy. They include the actual seventh first moment.
        """
        plus=np.zeros(7);minus=np.zeros(7)
        def absolute_integral(a,b,origin=0.):
            a-=origin;b-=origin
            return (b*abs(b)-a*abs(a))/2
        for region in self.regions:
            sa,sb=region['s'];za,zb=region['z'];area=(sb-sa)*(zb-za)
            axes=(0,1) if region['id']=='slab' else (self.j,)
            for j in axes:
                minus[j]+=self.fc*area/self.width
                value=self.fc*(sb-sa)*absolute_integral(za,zb)/self.width
                plus[j+3]+=value;minus[j+3]+=value
            if region['id']=='slab':
                plus[2]+=self.fc/2*area/self.width;minus[2]+=self.fc/2*area/self.width
                v=self.fc/2*(sb-sa)*absolute_integral(za,zb)/self.width
                plus[5]+=v;minus[5]+=v
            v=self.fc*(zb-za)*absolute_integral(sa,sb,self.sref)/self.width
            plus[6]+=v;minus[6]+=v
        for bar in self.longitudinal:
            if not bar['strength_credited']:continue
            f=bar['area_in2']*bar['fy_ksi']/self.width
            for j,v in ((self.j,f),(self.j+3,f*abs(bar['z_in'])),(6,f*abs(bar['s_in']-self.sref))):
                plus[j]+=v;minus[j]+=v
        for row in self.transverse:
            run=row['strength_run_interval_in']
            if run is None:continue
            f=row['area_in2']*row['count_per_length_in']*(run[1]-run[0])*row['fy_ksi']/self.width
            for j,v in ((self.other,f),(self.other+3,f*abs(row['z_in']))):plus[j]+=v;minus[j]+=v
        return dict(positive=(self.phi*plus).tolist(),negative_magnitude=(self.phi*minus).tolist(),
                    method='gross_concrete_independent_component_and_steel_relaxation')

    def support_upper_bound(self,direction):
        """Rigorous relaxed support of ALL seven simultaneous design actions.

        Enlarge the constitutive set to independent concrete components:
        -fc<=sigma_x,sigma_y<=0, |tau|<=fc/2, gross concrete without voids;
        actual developed steel may independently attain +/-fy. Compatibility,
        softening and compression/shear interaction are discarded ONLY to
        enlarge capacity. A target beyond this upper bound cannot be carried
        by the stricter section. The continuous linear field is integrated
        analytically; no numerical quadrature tolerance certifies this bound.
        """
        d=np.asarray(direction,float)
        if d.shape!=(7,) or not np.isfinite(d).all():raise ValueError('Seven finite dual coefficients required')
        value=0.
        for region in self.regions:
            sa,sb=region['s'];za,zb=region['z'];axes=(0,1) if region['id']=='slab' else (self.j,)
            for axis in axes:
                slope=d[6] if axis==self.j else 0.
                value+=self.fc*_positive_affine_rectangle(sa-self.sref,sb-self.sref,za,zb,-d[axis],-slope,-d[axis+3])
            if region['id']=='slab':
                value+=self.fc/2*sum(_positive_affine_rectangle(sa,sb,za,zb,sign*d[2],0.,sign*d[5]) for sign in (-1.,1.))
        for bar in self.longitudinal:
            if bar['strength_credited']:
                value+=bar['area_in2']*bar['fy_ksi']*abs(d[self.j]+bar['z_in']*d[self.j+3]+(bar['s_in']-self.sref)*d[6])
        for row in self.transverse:
            run=row['strength_run_interval_in']
            if run is not None:
                value+=row['area_in2']*row['count_per_length_in']*(run[1]-run[0])*row['fy_ksi']*abs(d[self.other]+row['z_in']*d[self.other+3])
        return float(self.phi*value/self.width)

    def capacity_separation_certificate(self,factored_resultants,*,maximum_iterations=80):
        """Search for a dual direction; independently recompute exact proof.

        Optimizer success is immaterial. No separating direction means no
        conclusion. A found direction is a mathematical upper-bound failure,
        unlike failure of the strain-compatible inverse solver.
        """
        target=np.asarray(factored_resultants,float)
        if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('Seven finite actual target resultants required')
        depth=max(self.h,max(abs(z) for r in self.regions for z in r['z']))
        scale=np.array([1.,1.,1.,1/depth,1/depth,1/depth,1/self.width])
        def objective(x):
            d=x*scale;return self.support_upper_bound(d)-float(d@target)
        normalized=target*scale;size=max(1.,float(max(abs(normalized))));seed=np.clip(normalized/size,-1,1)
        fit=minimize(objective,seed,method='L-BFGS-B',bounds=[(-1.,1.)]*7,
                     options={'maxiter':int(maximum_iterations),'ftol':1e-12,'gtol':1e-9,'maxls':30})
        candidates=[seed,np.asarray(fit.x,float)]
        direction=min(candidates,key=objective)*scale;upper=self.support_upper_bound(direction);demand=float(direction@target)
        tolerance=1e-8+1e-9*max(abs(upper),abs(demand));exceeded=demand>upper+tolerance
        return dict(method='continuous_gross_component_and_developed_steel_dual_support_upper_bound',
            capacity_exceeded=bool(exceeded),direction=direction.tolist(),demand=demand,proved_design_upper_bound=upper,
            floating_point_allowance=tolerance,excess=demand-upper,
            optimizer_success=bool(fit.success),optimizer_iterations=int(fit.nit),
            failure_inferred_from_optimizer=False,adequacy_inferred_when_no_separation=False)

    def _strain(self,q,s,z):
        s,z=np.broadcast_arrays(np.asarray(s,dtype=float),np.asarray(z,dtype=float))
        values=q[:3]+z[...,None]*q[3:6]
        values[...,self.j]+=q[6]*(s-self.sref)
        e=np.zeros((*s.shape,2,2));e[...,0,0]=values[...,0];e[...,1,1]=values[...,1]
        e[...,0,1]=e[...,1,0]=values[...,2]/2
        return e

    def _density(self,q,s,z,region):
        e=self._strain(q,s,z)
        if region['id']!='slab':
            axial=e[...,self.j,self.j].copy();e=np.zeros_like(e);e[...,self.j,self.j]=axial
        response=concrete_response(e,self.fc,peak_strain=self.peak,compression_strain_limit=self.peak,
                                   model='polak1993_thorenfeldt')
        stress=response['stress_ksi'][...,(0,1,0),(0,1,1)]
        values=np.concatenate((stress,stress*np.asarray(z)[...,None]),axis=-1)
        return np.concatenate((values,((np.asarray(s)-self.sref)*stress[...,self.j])[...,None]),axis=-1)

    def _z_breaks(self,q,region,lo,hi):
        breaks=[lo,hi]
        for sv in region['s']:
            a,b,c=q[:3].copy();d,e,f=q[3:6]
            if self.j==0:a+=q[6]*(sv-self.sref)
            else:b+=q[6]*(sv-self.sref)
            roots=[]
            # Exact spectral constitutive branch surfaces: concrete tension
            # cutoff, transverse-tension softening onset and prepeak limit.
            # Omitting the softening max(1,.8+.34*eps/peak) kink can prevent
            # even32/64 Gauss agreement despite a converged inverse solution.
            for threshold in (0.,(.2/.34)*self.peak,-self.peak):
                aa,bb=a-threshold,b-threshold
                scale=max(abs(aa),abs(bb),abs(c),abs(d)*self.h,abs(e)*self.h,abs(f)*self.h,1e-300)
                aa,bb,cc,dd,ee,ff=np.array([aa,bb,c,d,e,f])/scale
                polynomial=[dd*ee-ff*ff/4,aa*ee+bb*dd-cc*ff/2,aa*bb-cc*cc/4]
                nz=np.flatnonzero(np.asarray(polynomial)!=0)
                roots.extend(np.roots(polynomial[nz[0]:]) if len(nz) else [])
            for intercept,slope in ((a,d),(b,e),(a+b,d+e)):
                if abs(slope)>1e-24:roots.append(-intercept/slope)
            breaks.extend(float(np.real(v)) for v in roots if abs(np.imag(v))<1e-10 and lo<float(np.real(v))<hi)
        return sorted(set(breaks))

    def _s_breaks(self,q,z,lo,hi):
        breaks=[lo,hi]
        if abs(q[6])>1e-24:
            v=q[:3]+z*q[3:6];a,b=v[self.j],v[self.other];g=v[2]
            targets=[0.,-b]
            for threshold in (0.,(.2/.34)*self.peak,-self.peak):
                if abs(b-threshold)>1e-24:targets.append(threshold+g*g/(4*(b-threshold)))
            for value in targets:
                root=self.sref+(value-a)/q[6]
                if lo<root<hi:breaks.append(float(root))
        return sorted(set(breaks))

    def _integration_points(self,q,region,order,*,circle=None,transverse=None):
        # Direct rectangle/circle/actual-run integration, with separate roots
        # of the affine strain field. Cosine z coordinates regularize circle edges.
        lo,hi=region['z'];slo,shi=region['s']
        if circle is not None:
            lo=max(lo,circle['z_in']-circle['radius']);hi=min(hi,circle['z_in']+circle['radius'])
        if transverse is not None:
            lo=max(lo,transverse['z_in']-transverse['radius']);hi=min(hi,transverse['z_in']+transverse['radius'])
            slo=max(slo,transverse['run_interval_in'][0]);shi=min(shi,transverse['run_interval_in'][1])
        if hi<=lo or shi<=slo:return np.empty(0),np.empty(0),np.empty(0)
        gauss,weights=_gauss(order);zz=[];ss=[];ww=[]
        breaks=self._z_breaks(q,region,lo,hi)
        for za,zb in zip(breaks[:-1],breaks[1:]):
            theta=(gauss+1)*math.pi/2
            zvals=(za+zb)/2-(zb-za)*np.cos(theta)/2
            zweights=weights*(zb-za)*math.pi*np.sin(theta)/4
            for z,zw in zip(zvals,zweights):
                a,b=slo,shi;density=1.
                if circle is not None:
                    rad=math.sqrt(max(0.,circle['radius']**2-(z-circle['z_in'])**2))
                    a=max(a,circle['s_in']-rad);b=min(b,circle['s_in']+rad)
                if transverse is not None:
                    density=2*math.sqrt(max(0.,transverse['radius']**2-(z-transverse['z_in'])**2))*transverse['count_per_length_in']
                if b<=a:continue
                sbreaks=self._s_breaks(q,z,a,b)
                for sa,sb in zip(sbreaks[:-1],sbreaks[1:]):
                    values=(sa+sb)/2+(sb-sa)/2*gauss
                    ss.extend(values);zz.extend(np.full(order,z));ww.extend(weights*(sb-sa)/2*zw*density)
        return np.asarray(ss),np.asarray(zz),np.asarray(ww)

    def _integrate(self,q,region,order,*,circle=None,transverse=None,integrand=None,result_size=7):
        ss,zz,ww=self._integration_points(q,region,order,circle=circle,transverse=transverse)
        if not len(ww):return np.zeros(result_size)
        density=self._density(q,ss,zz,region) if integrand is None else integrand(ss,zz)
        return np.sum(density*ww[:,None],axis=0)

    def fixed_quadrature_resultants_batch(self,generalized_strains,*,order=12):
        """Fast auxiliary fixed-point integration for numerical Jacobians.

        No strain-root splits, domain certificate or acceptance is inferred.
        Ultimate acceptance always uses adaptive actual-region integration.
        All finite steel voids and developed strength lengths remain exact
        geometric owners. Different trial strain states are vectorized only.
        """
        qs=np.asarray(generalized_strains,float)
        if qs.ndim!=2 or qs.shape[1]!=7 or not np.isfinite(qs).all():raise ValueError('Finite batch of seven trial strains required')
        cache=getattr(self,'_fixed_points_cache',{})
        if order not in cache:
            groups=[]
            for region in self.regions:
                points=[self._integration_points(np.zeros(7),region,order)];signs=[1.]
                for bar in self.longitudinal:points.append(self._integration_points(np.zeros(7),region,order,circle=bar));signs.append(-1.)
                if region['id']=='slab':
                    for row in self.transverse:points.append(self._integration_points(np.zeros(7),region,order,transverse=row));signs.append(-1.)
                    for void in self.slab_voids:points.append(self._integration_points(np.zeros(7),void,order));signs.append(-1.)
                s=np.concatenate([p[0] for p in points]);z=np.concatenate([p[1] for p in points]);w=np.concatenate([p[2]*v for p,v in zip(points,signs)])
                groups.append((region,s,z,w))
            cache[order]=groups;self._fixed_points_cache=cache
        result=np.zeros((len(qs),7))
        for region,s,z,w in cache[order]:
            v=qs[:,None,:3]+z[None,:,None]*qs[:,None,3:6]
            v[:,:,self.j]+=(s-self.sref)[None,:]*qs[:,6,None]
            e=np.zeros((len(qs),len(s),2,2))
            if region['id']=='slab':e[:,:,0,0]=v[:,:,0];e[:,:,1,1]=v[:,:,1];e[:,:,0,1]=e[:,:,1,0]=v[:,:,2]/2
            else:e[:,:,self.j,self.j]=v[:,:,self.j]
            c=concrete_response(e,self.fc,peak_strain=self.peak,compression_strain_limit=self.peak,model='polak1993_thorenfeldt')
            stress=c['stress_ksi'][...,(0,1,0),(0,1,1)]
            values=np.concatenate([stress,stress*z[None,:,None],((s-self.sref)[None,:]*stress[:,:,self.j])[:,:,None]],axis=2)
            result+=np.einsum('npi,p->ni',values,w)
        for bar in self.longitudinal:
            if not bar['strength_credited']:continue
            eps=qs[:,self.j]+bar['z_in']*qs[:,self.j+3]+(bar['s_in']-self.sref)*qs[:,6]
            f=bar['area_in2']*np.clip(bar['es_ksi']*eps,-bar['fy_ksi'],bar['fy_ksi'])
            result[:,self.j]+=f;result[:,self.j+3]+=bar['z_in']*f;result[:,6]+=(bar['s_in']-self.sref)*f
        for row in self.transverse:
            run=row['strength_run_interval_in']
            if run is None:continue
            eps=qs[:,self.other]+row['z_in']*qs[:,self.other+3]
            f=row['area_in2']*row['count_per_length_in']*(run[1]-run[0])*np.clip(row['es_ksi']*eps,-row['fy_ksi'],row['fy_ksi'])
            result[:,self.other]+=f;result[:,self.other+3]+=row['z_in']*f
        return self.phi*result/self.width

    def web_nominal_secant(self,generalized_normal_strain,*,beam_bar_ids,slab_accounting=None):
        """Disjoint web-concrete + assigned actual beam steel3x3 nominal secant.

        Input strain=[epsilon_axis_at_sref_z0,kstrong,kweak]. Output FULL beam
        [N,int(z*sigma),int((s-sref)*sigma)] and its3x3 operator; no phi.
        Beam steel in the slab may be assigned to this line only with matching
        slab finite-void subtraction and no duplicate slab steel ownership.
        The explicit slab_accounting ledger has evidence_id,
        void_subtraction_ids and steel_owner_ids. Missing entries return unknown
        alongside the diagnostic matrix, never implicit ownership approval.
        No slab concrete, shear/torsion or stiffness floor is included.
        """
        p=np.asarray(generalized_normal_strain,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Three finite actual beam normal strains required')
        webs=[r for r in self.regions if r['id']!='slab']
        if len(webs)!=1:raise ValueError('Exactly one disjoint physical web required')
        wanted=list(beam_bar_ids)
        if not wanted or len(wanted)!=len(set(wanted)):raise ValueError('Explicit unique physical beam steel IDs required')
        byid={b['id']:b for b in self.longitudinal}
        if not set(wanted)<=set(byid):raise ValueError('Assigned beam steel is absent from actual common inventory')
        selected=[byid[i] for i in wanted];web=webs[0]
        q=np.zeros(7);q[self.j]=p[0];q[self.j+3]=p[1];q[6]=p[2]
        n=.8+self.fc*6.894757293168/17
        initial=self.fc*n/(self.peak*(n-1))
        def density(s,z):
            l=np.stack([np.ones_like(s),z,s-self.sref],axis=-1);eps=l@p
            e=np.zeros((len(s),2,2));e[:,0,0]=eps
            state=concrete_response(e,self.fc,peak_strain=self.peak,compression_strain_limit=self.peak,model='polak1993_thorenfeldt')
            stress=state['stress_ksi'][:,0,0]
            secant=np.divide(stress,eps,out=np.full(len(eps),initial),where=eps!=0)
            return np.concatenate(((secant[:,None,None]*l[:,:,None]*l[:,None,:]).reshape(-1,9),stress[:,None]*l),axis=1)
        def concrete(order):
            value=self._integrate(q,web,order,integrand=density,result_size=12)
            for bar in self.longitudinal:
                value-=self._integrate(q,web,order,circle=bar,integrand=density,result_size=12)
            return value[:9].reshape(3,3),value[9:]
        kc,rc=concrete(self.order);fine,rf=concrete(self.order*2)
        ks=np.zeros((3,3));rs=np.zeros(3);steel=[];issues=[]
        for bar in selected:
            l=np.array([1.,bar['z_in'],bar['s_in']-self.sref]);eps=float(l@p)
            stress=float(np.clip(bar['es_ksi']*eps,-bar['fy_ksi'],bar['fy_ksi']))
            secant=stress/eps if eps!=0 else bar['es_ksi']
            ks+=bar['area_in2']*secant*np.outer(l,l);rs+=bar['area_in2']*stress*l
            steel.append(dict(id=bar['id'],strain=eps,stress_ksi=stress,secant_ksi=secant))
            if abs(eps)>self.steel_limit+1e-12:issues.append('beam_steel_strain_exceeds_declared_limit')
        minimum=min(p[0]+p[1]*z+p[2]*(s-self.sref) for s in web['s'] for z in web['z'])
        if minimum < -self.peak-1e-12:issues.append('web_compression_exceeds_declared_prepeak_limit')
        K,R=kc+ks,rc+rs;residual=K@p-R
        converged=bool(np.all(abs(kc-fine)<=1e-5+1e-5*np.maximum(abs(kc),abs(fine))) and
                       np.all(abs(rc-rf)<=1e-7+1e-5*np.maximum(abs(rc),abs(rf))))
        if not converged:issues.append('web_secant_quadrature_not_verified')
        if not np.all(abs(residual)<=1e-7+1e-6*abs(R)):issues.append('web_secant_reconstruction_failed')
        scale=np.diag([1,1/max(self.h,abs(web['z'][0])),1/self.width])
        eigen=np.linalg.eigvalsh(scale@K@scale);threshold=1e-11*max(1.,max(abs(eigen)))
        if min(eigen)<-threshold:issues.append('web_secant_negative_energy')
        required_slab=[b['id'] for b in selected if b['z_in']+b['radius']>-self.h/2+1e-10]
        slab_accounting=slab_accounting or {}
        missing=sorted(set(required_slab)-set(slab_accounting.get('void_subtraction_ids',[])))
        duplicates=sorted(set(wanted)&set(slab_accounting.get('steel_owner_ids',[])))
        unassigned=sorted(b['id'] for b in self.longitudinal if b['z_in']-b['radius']<-self.h/2-1e-10 and b['id'] not in wanted)
        if missing:issues.append('beam_steel_slab_concrete_void_subtraction_not_established')
        if duplicates:issues.append('beam_steel_also_credited_to_slab_operator')
        if unassigned:issues.append('actual_web_steel_missing_from_line_operator')
        if required_slab and not slab_accounting.get('evidence_id'):issues.append('shared_slab_web_material_ledger_identity_missing')
        return dict(method='actual_spatial_web_nominal_secant_v1',ABD_normal=K.tolist(),
            web_concrete_ABD=kc.tolist(),beam_steel_ABD=ks.tolist(),nominal_resultants=R.tolist(),
            web_concrete_resultants=rc.tolist(),beam_steel_resultants=rs.tolist(),
            generalized_normal_strain=p.tolist(),secant_reconstruction_residual=residual.tolist(),
            material_state=steel,phi_applied=False,slab_concrete_included=False,
            required_slab_void_subtraction_ids=required_slab,missing_slab_void_ids=missing,
            duplicate_steel_owner_ids=duplicates,unassigned_web_steel_ids=unassigned,
            domain_and_accounting_errors=sorted(set(issues)),status='pass' if not issues else 'unknown',
            quadrature_verified=converged,rank=int(np.sum(eigen>threshold)),scaled_eigenvalues=eigen.tolist(),
            full_floor_equilibrium_converged=False)

    def state(self,generalized_strain,*,quadrature_order=None):
        q=np.asarray(generalized_strain,dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Seven finite compatible strains required')
        order=self.order if quadrature_order is None else int(quadrature_order)
        if not 4<=order<=64:raise ValueError('Quadrature order4..64required')
        concrete=np.zeros(7);regions=[];errors=[]
        for region in self.regions:
            force=self._integrate(q,region,order)
            for bar in self.longitudinal:force-=self._integrate(q,region,order,circle=bar)
            if region['id']=='slab':
                for row in self.transverse:force-=self._integrate(q,region,order,transverse=row)
                for void in self.slab_voids:force-=self._integrate(q,void,order)
            concrete+=force
            corners=[(s,z) for s in region['s'] for z in region['z']]
            e=self._strain(q,[s for s,z in corners],[z for s,z in corners])
            if region['id']!='slab':
                axial=e[:,self.j,self.j].copy();e[:]=0;e[:,self.j,self.j]=axial
            c=concrete_response(e,self.fc,peak_strain=self.peak,compression_strain_limit=self.peak,model='polak1993_thorenfeldt')
            if not c['prepeak_valid']:errors.append('concrete_compression_exceeds_declared_prepeak_limit')
            if not c['backbone_valid']:errors.append('concrete_backbone_n_not_above_one')
            regions.append(dict(id=region['id'],nominal_resultants_per_in=(force/self.width).tolist()))
        steel=np.zeros(7);bar_states=[]
        for bar in self.longitudinal:
            e=q[self.j]+bar['z_in']*q[self.j+3]+(bar['s_in']-self.sref)*q[6]
            stress=float(np.clip(bar['es_ksi']*e,-bar['fy_ksi'],bar['fy_ksi'])) if bar['strength_credited'] else 0.
            f=bar['area_in2']*stress
            steel[self.j]+=f;steel[self.j+3]+=bar['z_in']*f;steel[6]+=(bar['s_in']-self.sref)*f
            bar_states.append(dict(id=bar['id'],axis=self.axis,s_in=bar['s_in'],z_in=bar['z_in'],strain=float(e),
                stress_ksi=stress,nominal_stress_ksi=stress,area_in2=bar['area_in2'],fy_ksi=bar['fy_ksi'],force_kip=f,
                strength_credited=bar['strength_credited']))
        for row in self.transverse:
            e=q[self.other]+row['z_in']*q[self.other+3]
            stress=float(np.clip(row['es_ksi']*e,-row['fy_ksi'],row['fy_ksi']))
            run=row['strength_run_interval_in'];length=0. if run is None else run[1]-run[0]
            f=row['area_in2']*row['count_per_length_in']*length*stress
            steel[self.other]+=f;steel[self.other+3]+=row['z_in']*f
            bar_states.append(dict(id=row['id'],physical_bar_ids=row['physical_bar_ids'],axis='y' if self.axis=='x' else 'x',
                z_in=row['z_in'],strain=float(e),stress_ksi=stress,homogenized_force_kip=f,
                physical_run_interval_in=list(row['run_interval_in']),strength_run_interval_in=run))
        if max((abs(v['strain']) for v in bar_states),default=0)>self.steel_limit+1e-12:
            errors.append('reinforcement_strain_exceeds_declared_limit')
        nominal=(concrete+steel)/self.width
        return dict(method=METHOD,generalized_strain=q.tolist(),nominal_resultants=nominal.tolist(),
            design_resultants=(self.phi*nominal).tolist(),full_strip_nominal_resultants=(self.width*nominal).tolist(),
            concrete_nominal_resultants=(concrete/self.width).tolist(),steel_nominal_resultants=(steel/self.width).tolist(),
            concrete_regions=regions,steel=bar_states,physical_width_in=self.width,reference_s_in=self.sref,
            external_slab_void_rectangles=[dict(id=v['physical_void_id'],interval_in=list(v['s']),z_interval_in=list(v['z'])) for v in self.slab_voids],
            axis=self.axis,phi=self.phi,material_domain_valid=not errors,material_domain_errors=sorted(set(errors)),
            quadrature_order=order,transverse_first_moment_preserved=True,code_qualified=False)

    def verify_quadrature(self,q,*,relative_tolerance=1e-5,absolute_tolerance=1e-7):
        a=np.array(self.state(q)['nominal_resultants']);b=np.array(self.state(q,quadrature_order=2*self.order)['nominal_resultants'])
        err=abs(a-b);passed=bool(np.all(err<=absolute_tolerance+relative_tolerance*np.maximum(abs(a),abs(b))))
        return dict(passed=passed,absolute_errors=err.tolist(),orders=[self.order,2*self.order])

    def _inactive_shear_seed(self,q,target):
        """One domain-valid trial off a nearly inactive slab-shear branch.

        Orthogonal reinforcement supplies no Nxy/Mxy. At a strictly tensile
        concrete iterate those residual derivatives vanish. A very thin active
        region can likewise stall the inverse. Activate principal
        compression with a signed shear trial; this is only an initial guess,
        never an engineering stress field or a capacity/equilibrium assertion.
        """
        state=self.state(q)
        slab=next(r for r in state['concrete_regions'] if r['id']=='slab')
        if abs(target[2])<=1e-6:return None
        ratio=abs(self.phi*slab['nominal_resultants_per_in'][2]/target[2])
        # Work-selection trigger only; neither an acceptance nor strength limit.
        if ratio>1e-3:return None
        trial=np.asarray(q,float).copy();e=self._strain(trial,self.sref,0.)
        trial[2]=np.sign(target[2])*(2*np.sqrt(max(0.,e[0,0])*max(0.,e[1,1]))+.1*self.peak)
        trial[5]=12*target[5]/(target[2]*self.h**2)*trial[2]
        if not np.isfinite(trial).all():return None
        candidate=self.state(trial)
        steel=max((abs(b['strain'])/self.steel_limit for b in candidate['steel']),default=0.)
        compression=0.
        for region in self.regions:
            for s in region['s']:
                for z in region['z']:
                    e=self._strain(trial,s,z)
                    value=e[self.j,self.j] if region['id']!='slab' else np.linalg.eigvalsh(e)[0]
                    compression=max(compression,-float(value)/self.peak)
        contraction=min(1.,.99/max(1.,steel,compression));trial*=contraction
        if not self.state(trial)['material_domain_valid']:return None
        return trial,dict(method=SOLVER_METHOD,initialization_only=True,
            old_slab_concrete_resultants=slab['nominal_resultants_per_in'],
            slab_shear_to_target_ratio=ratio,activation_ratio_limit=1e-3,
            trial_strain=trial.tolist(),radial_contraction=contraction)

    def solve(self,factored_resultants,*,initial_strain=None,max_evaluations=250,use_batched_tangent=True):
        target=np.asarray(factored_resultants,dtype=float)
        if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('Seven finite concurrent factored resultants required')
        if not isinstance(max_evaluations,int) or not 1<=max_evaluations<=1000:raise ValueError('Bounded positive solve count required')
        factors=1e-3*np.array([1,1,1,2/self.h,2/self.h,2/self.h,2/self.width])
        scale=np.array([self.fc*self.h]*3+[self.fc*self.h**2/4]*3+[self.fc*self.h*self.width/4])
        q0=np.zeros(7) if initial_strain is None else np.asarray(initial_strain,dtype=float)
        if q0.shape!=(7,) or not np.isfinite(q0).all():raise ValueError('Seven finite initial strains required')
        attempts=[];used=0;order=self.order
        def fit_at(wanted,start,limit,active_order,label,batched=None):
            nonlocal used
            batched=use_batched_tangent if batched is None else batched
            def tangent(x):
                steps=1e-7*np.maximum(1.,abs(x));trial=np.repeat(x[None,:],8,axis=0)
                trial[1:]+=np.diag(steps)
                r=self.fixed_quadrature_resultants_batch(trial*factors,order=12)
                return ((r[1:]-r[0])/steps[:,None]).T/scale[:,None]
            fit=least_squares(lambda x:(np.array(self.state(x*factors,quadrature_order=active_order)['design_resultants'])-wanted)/scale,
                start/factors,jac=tangent if batched else '2-point',max_nfev=limit,
                ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
            q=fit.x*factors;state=self.state(q,quadrature_order=active_order);error=np.array(state['design_resultants'])-wanted
            balanced=bool(np.all(abs(error)<=1e-6+1e-6*abs(wanted)));used+=fit.nfev
            attempts.append(dict(stage=label,order=active_order,evaluations=fit.nfev,equilibrium_verified=balanced,
                material_domain_valid=state['material_domain_valid'],residual=error.tolist(),message=str(fit.message),
                jacobian='auxiliary_vectorized_fixed_quadrature' if batched else 'actual_adaptive_quadrature_finite_difference'))
            return fit,q,state,error,balanced
        initial_seed=self._inactive_shear_seed(q0,target) if initial_strain is not None else None
        if initial_seed is not None:
            fit,q,out,error,balanced=fit_at(target,initial_seed[0],min(100,max_evaluations),order,'initial_slab_shear_activation')
            attempts[-1]['initialization']=initial_seed[1]
        else:
            fit,q,out,error,balanced=fit_at(target,q0,min(100,max_evaluations),order,'direct')
        if use_batched_tangent and not balanced and max_evaluations-used>=20:
            fit,q,out,error,balanced=fit_at(target,q,min(50,max_evaluations-used),order,'exact_jacobian_polish',batched=False)
        if not balanced and max_evaluations-used>=20:
            seed=self._inactive_shear_seed(q,target)
            if seed is not None:
                trial=fit_at(target,seed[0],min(50,max_evaluations-used),order,'inactive_slab_shear_activation')
                attempts[-1]['initialization']=seed[1]
                if trial[4] or np.linalg.norm(trial[3]/scale)<np.linalg.norm(error/scale):
                    fit,q,out,error,balanced=trial
        # A yielded, all-tension flat Jacobian is a numerical stationary point,
        # not a capacity proof. A bounded proportional path supplies a distinct
        # physically meaningful initial state while keeping the total budget.
        if not balanced and max_evaluations-used>=20:
            pathq=np.zeros(7)
            fractions=(.1,.25,.5,.75,1.)
            for k,fraction in enumerate(fractions):
                remaining=max_evaluations-used
                if remaining<1:break
                limit=max(1,remaining//(len(fractions)-k))
                trial=fit_at(fraction*target,pathq,limit,order,f'continuation_{fraction:g}')
                tf,tq,ts,te,tb=trial
                if not tb:break
                pathq=tq
                if fraction==1.:fit,q,out,error,balanced=trial
        quad_attempts=[];quad=None
        # Refinement changes only integration accuracy; original tolerances
        # and force equilibrium are retained at the refined solution.
        while True:
            fine=min(64,2*order)
            refined=self.state(q,quadrature_order=fine)
            coarse=np.asarray(out['nominal_resultants']);fine_r=np.asarray(refined['nominal_resultants'])
            err=abs(coarse-fine_r);quadpass=bool(np.all(err<=1e-7+1e-5*np.maximum(abs(coarse),abs(fine_r))))
            refined_error=np.asarray(refined['design_resultants'])-target
            refined_balance=bool(np.all(abs(refined_error)<=1e-6+1e-6*abs(target)))
            quad_attempts.append(dict(orders=[order,fine],passed=quadpass,absolute_errors=err.tolist(),refined_equilibrium_verified=refined_balance))
            quad=dict(quad_attempts[-1],attempts=list(quad_attempts))
            if quadpass and (not balanced or refined_balance):
                out=refined;error=refined_error;balanced=refined_balance;break
            if not balanced or order>=32 or max_evaluations-used<2:break
            order=min(32,2*order)
            fit,q,out,error,balanced=fit_at(target,q,max_evaluations-used,order,'quadrature_refined_inverse')
        passed=balanced and quad['passed'] and out['material_domain_valid']
        out.update(factored_resultants=target.tolist(),equilibrium_residual=error.tolist(),equilibrium_verified=balanced,
            quadrature=quad,restricted_model_state_verified=passed,status='pass' if passed else 'unknown',
            numerical_solver_method=SOLVER_METHOD,
            numerical_solver_success=bool(fit.success),evaluations=used,solve_attempts=attempts,
            numerical_solver_message=str(fit.message),reason=None if passed else 'Numerical/domain failure is not proof of capacity exceedance')
        return out
