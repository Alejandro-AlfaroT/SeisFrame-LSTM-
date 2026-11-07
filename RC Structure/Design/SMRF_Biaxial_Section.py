"""Biaxial strength of a doubly symmetric rectangular reinforced-concrete section.

ACI 318-19 22.2 strain compatibility and equilibrium, with the same Whitney
block and EPP steel as the existing uniaxial routine. The neutral axis rotates;
phi follows the actual extreme tension strain of that rotated section (21.2.2).
The section is not assigned a fixed empirical load-contour exponent.

Coordinates: x across b, y across h, centered at the gross centroid. Mh is
sum(F*y), Mb is sum(F*x). Positive compression is P. Bars displace concrete
inside the block exactly once. Supports only actual doubly symmetric cages;
unsupported inputs or an unconverged equilibrium raise, never imply acceptance.
"""
from __future__ import annotations
import math
import hashlib
import numpy as np

METHOD = "rotated_whitney_finite_bar_displacement_biaxial_v1"


class BiaxialSection:
    def __init__(self, b, h, fc, fy, bars, es=29000.0):
        self.b,self.h,self.fc,self.fy,self.es=map(float,(b,h,fc,fy,es))
        self.bars=np.asarray(bars,dtype=float)
        if any(not math.isfinite(v) or v<=0 for v in (self.b,self.h,self.fc,self.fy,self.es)):
            raise ValueError("Biaxial section dimensions and materials must be finite and positive.")
        if self.bars.ndim!=2 or self.bars.shape[1]!=3 or not len(self.bars) or not np.isfinite(self.bars).all():
            raise ValueError("Biaxial bars must be finite [x, y, area] rows.")
        if (np.abs(self.bars[:,0])>=self.b/2).any() or (np.abs(self.bars[:,1])>=self.h/2).any() or (self.bars[:,2]<=0).any():
            raise ValueError("Biaxial bars must have positive area inside the section.")
        points=sorted(tuple(round(v,9) for v in row) for row in self.bars)
        for axis in (0,1):
            mirror=self.bars.copy();mirror[:,axis]*=-1
            if points!=sorted(tuple(round(v,9) for v in row) for row in mirror):
                raise ValueError("The biaxial section solver requires a doubly symmetric actual cage.")
        self.beta=max(.65,min(.85,.85-.05*(self.fc-4)))
        self.ast=float(self.bars[:,2].sum())
        # Equivalent circular areas avoid the artificial force jump created by
        # deleting a whole bar's concrete as the block passes its centroid.
        # Steel strain remains evaluated at each longitudinal bar centroid.
        self.radii=np.sqrt(self.bars[:,2]/math.pi)
        self.p0=.85*self.fc*(self.b*self.h-self.ast)+self.fy*self.ast
        self.compression_cap=.65*.8*self.p0
        self.tension_cap=-.9*self.fy*self.ast
        self.force_scale=max(self.p0,self.fy*self.ast,1.)
        self._ray_cache={}

    @staticmethod
    def _block(b,h,u,v,a):
        """Area and moments of a clipped corner, analytic piecewise integration.

        Integrate H(x)=clip((a-u*x)/v,0,h). Swap axes so division is by
        the larger normal component. Using the local trapezoid width avoids
        subtracting nearly equal powers near a uniaxial direction.
        """
        swap=u>v
        B=np.where(swap,h,b);H=np.where(swap,b,h)
        U=np.where(swap,v,u);V=np.where(swap,u,v)
        safe=np.maximum(U,1e-30)
        full=np.clip((a-V*H)/safe,0.,B)
        stop=np.clip(a/safe,0.,B)
        width=np.maximum(0.,stop-full)
        height=np.clip((a-U*full)/V,0.,H)
        slope=U/V
        strip=height*width-.5*slope*width**2
        area=H*full+strip
        qx=.5*H*full**2+full*strip+.5*height*width**2-slope*width**3/3
        qy=.5*H**2*full+.5*(height**2*width-height*slope*width**2+slope**2*width**3/3)
        axial=U<1e-14
        height0=np.clip(a/V,0.,H)
        area=np.where(axial,B*height0,area)
        qx=np.where(axial,.5*B**2*height0,qx)
        qy=np.where(axial,.5*B*height0**2,qy)
        # Restore global axes and convert corner distances to centroid coordinates.
        Qx=np.where(swap,qy,qx);Qy=np.where(swap,qx,qy)
        return area,b/2*area-Qx,h/2*area-Qy

    def state(self,c,theta):
        c,theta=np.broadcast_arrays(np.asarray(c,dtype=float),np.asarray(theta,dtype=float))
        u,v=np.cos(theta),np.sin(theta)
        a=self.beta*c
        area,qx,qy=self._block(self.b,self.h,u,v,a)
        distance=(self.b*u+self.h*v)[...,None]/2-u[...,None]*self.bars[:,0]-v[...,None]*self.bars[:,1]
        strain=.003*(1-distance/c[...,None])
        stress=np.clip(self.es*strain,-self.fy,self.fy)
        steel=stress*self.bars[:,2]
        cut=np.clip(a[...,None]-distance,-self.radii,self.radii)
        root=np.sqrt(np.maximum(0.,self.radii**2-cut**2))
        removed_area=self.radii**2*(np.arcsin(cut/self.radii)+math.pi/2)+cut*root
        removed_first_normal=2./3*root**3
        net=steel-.85*self.fc*removed_area
        pn=.85*self.fc*area+net.sum(axis=-1)
        mh=.85*self.fc*qy+(net*self.bars[:,1]-.85*self.fc*removed_first_normal*v[...,None]).sum(axis=-1)
        mb=.85*self.fc*qx+(net*self.bars[:,0]-.85*self.fc*removed_first_normal*u[...,None]).sum(axis=-1)
        et=-strain.min(axis=-1)
        phi=.65+.25*np.clip((et-self.fy/self.es)/.003,0.,1.)
        return dict(p=phi*pn,mh=phi*mh,mb=phi*mb,phi=phi,c=c,theta=theta,eps_t=et)

    def at_axial(self,p,theta):
        p,theta=np.broadcast_arrays(np.asarray(p,dtype=float),np.asarray(theta,dtype=float))
        if not np.isfinite(p).all() or (p<self.tension_cap-1e-8).any() or (p>self.compression_cap+1e-8).any():
            raise ValueError("Axial demand is outside the tied-column strength domain.")
        lo=np.full(p.shape,1e-9);hi=np.full(p.shape,100*max(self.b,self.h))
        extent=self.b*np.cos(theta)+self.h*np.sin(theta)
        c=extent*(.2+.8*(p-self.tension_cap)/(self.compression_cap-self.tension_cap))
        tolerance=1e-10*self.force_scale
        # Safeguarded Newton only accelerates the bracketed equilibrium solve.
        # Every accepted point still meets the same force residual test.
        for _ in range(18):
            value=self.state(c,theta)['p'];error=value-p
            done=np.abs(error)<=tolerance
            if done.all():break
            hi=np.where((error>0)&~done,c,hi);lo=np.where((error<=0)&~done,c,lo)
            step=np.maximum(1e-7,c*1e-5)
            derivative=(self.state(c+step,theta)['p']-value)/step
            new=c-error/np.where(derivative>1e-12,derivative,1.)
            safe=(derivative>1e-12)&(new>lo)&(new<hi)&np.isfinite(new)
            c=np.where(done,c,np.where(safe,new,(lo+hi)/2))
        else:
            # Retain completed rows and bisect only the unresolved rows.
            done=np.abs(self.state(c,theta)['p']-p)<=tolerance
            for _ in range(45):
                mid=(lo+hi)/2;value=self.state(mid,theta)['p']
                hi=np.where((value>p)&~done,mid,hi);lo=np.where((value<=p)&~done,mid,lo)
            c=np.where(done,c,(lo+hi)/2)
        result=self.state(c,theta)
        if np.max(np.abs(result['p']-p),initial=0.)>1e-8*self.force_scale:
            raise ValueError("Biaxial axial equilibrium did not converge.")
        return result

    def ray_capacity(self,p,mh,mb):
        """Factored moment magnitude along each demand ray at its factored P.

        Reflection symmetry permits absolute moments. A positive Mh requires
        rotation toward the h compression face. Axial equilibrium is solved
        afresh at each angle; phi is never borrowed from a uniaxial diagram.
        """
        p,mh,mb=np.broadcast_arrays(np.asarray(p,dtype=float),np.abs(mh),np.abs(mb))
        magnitude=np.hypot(mh,mb)
        if not np.isfinite(magnitude).all() or (magnitude<=0).any():
            raise ValueError("A biaxial moment ray must be finite and nonzero.")
        target=np.arctan2(mh,mb)
        key=(p.shape,hashlib.sha256(p.tobytes()+target.tobytes()).digest())
        if key in self._ray_cache:
            result={k:v.copy() for k,v in self._ray_cache[key].items()}
            result['dcr']=magnitude/result['capacity']
            return result
        lo=np.zeros(p.shape);hi=np.full(p.shape,math.pi/2)
        angle=target.copy()
        done=(mh==0.)|(mb==0.)
        for _ in range(15):
            result=self.at_axial(p,angle)
            actual=np.arctan2(np.maximum(result['mh'],0.),np.maximum(result['mb'],0.))
            error=actual-target
            done=done|(np.abs(error)<1e-9)
            if done.all():break
            hi=np.where((error>0)&~done,angle,hi);lo=np.where((error<=0)&~done,angle,lo)
            offset=np.where(angle+1e-5<math.pi/2,1e-5,-1e-5)
            perturbed=self.at_axial(p,angle+offset)
            other=np.arctan2(np.maximum(perturbed['mh'],0.),np.maximum(perturbed['mb'],0.))
            derivative=(other-actual)/offset
            new=angle-error/np.where(derivative>1e-12,derivative,1.)
            safe=(derivative>1e-12)&(new>lo)&(new<hi)&np.isfinite(new)
            angle=np.where(done,angle,np.where(safe,new,(lo+hi)/2))
        else:
            for _ in range(27):
                mid=(lo+hi)/2;value=self.at_axial(p,mid)
                actual=np.arctan2(np.maximum(value['mh'],0.),np.maximum(value['mb'],0.))
                hi=np.where((actual>target)&~done,mid,hi);lo=np.where((actual<=target)&~done,mid,lo)
            angle=np.where(done,angle,(lo+hi)/2)
        angle=np.where(mh==0.,0.,np.where(mb==0.,math.pi/2,angle))
        result=self.at_axial(p,angle)
        cap=np.hypot(result['mh'],result['mb'])
        error=np.abs(np.arctan2(np.maximum(result['mh'],0.),np.maximum(result['mb'],0.))-target)
        if (cap<=1e-8).any() or np.max(error,initial=0.)>1e-6:
            raise ValueError("Biaxial moment-direction equilibrium did not converge.")
        result.update(capacity=cap,dcr=magnitude/cap,angle_error_rad=error)
        if len(self._ray_cache)>=8:self._ray_cache.pop(next(iter(self._ray_cache)))
        self._ray_cache[key]={k:np.asarray(v).copy() for k,v in result.items() if k!='dcr'}
        return result
