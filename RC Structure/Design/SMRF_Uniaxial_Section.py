"""Layered rectangular/T/L section N-M equilibrium with actual strain phi.

The concrete outline is a stack of nonoverlapping horizontal rectangles.
Finite circular reinforcement area displaces concrete once; steel strain is
evaluated at its layer centroid, as in the existing section methods. A layer
has total area and individual bar area (fractional equivalent distributed
mat counts are permitted). Demand and nominal/probable uses are separate:
phi_mode='auto' is Chapter21; phi_mode='nominal' is phi=1, NEVER reserved steel.
Coordinates y downward from section top. Positive N compression, M sagging
about the gross-depth midplane. All strengths are kip / kip-in.
"""
from __future__ import annotations
import math
import numpy as np

METHOD='finite_bar_layered_section_axial_flexure_v1'


class LayeredSection:
    def __init__(self,depth_in,concrete_regions,layers,es_ksi=29000.):
        self.h=float(depth_in);self.es=float(es_ksi)
        self.regions=np.asarray(concrete_regions,dtype=float) # y0,y1,width,fc
        self.layers=np.asarray(layers,dtype=float) # y,total As,bar As,fy
        if not math.isfinite(self.h) or self.h<=0 or not math.isfinite(self.es) or self.es<=0:raise ValueError('Positive finite section dimensions/materials required')
        if self.regions.ndim!=2 or self.regions.shape[1]!=4 or not len(self.regions) or not np.isfinite(self.regions).all():raise ValueError('Concrete regions require y0,y1,width,fc')
        if self.layers.ndim!=2 or self.layers.shape[1]!=4 or not len(self.layers) or not np.isfinite(self.layers).all():raise ValueError('Steel layers require y,total area,individual bar area,fy')
        previous=0.
        for y0,y1,b,fc in self.regions:
            if abs(y0-previous)>1e-9 or y1<=y0 or b<=0 or fc<=0:raise ValueError('Concrete regions must partition the full depth without overlap')
            previous=y1
        if abs(previous-self.h)>1e-9:raise ValueError('Concrete outline does not cover full section depth')
        if (self.layers[:,1:]<=0).any():raise ValueError('Positive steel layer areas and grades required')
        self.radius=np.sqrt(self.layers[:,2]/math.pi)
        if (self.layers[:,0]-self.radius<0).any() or (self.layers[:,0]+self.radius>self.h).any():raise ValueError('Bar area extends outside section depth')
        # Current pilot beam and slab concrete use the same grade. Mixed grades
        # need per-material stress block and are refused, not homogenized silently.
        if np.ptp(self.regions[:,3])>1e-10:raise ValueError('Mixed concrete grades are outside this section method')
        self.fc=float(self.regions[0,3]);self.beta=max(.65,min(.85,.85-.05*(self.fc-4)))
        self.ag=float(np.sum((self.regions[:,1]-self.regions[:,0])*self.regions[:,2]))
        self.ast=float(np.sum(self.layers[:,1]));self.steel_force=float(np.sum(self.layers[:,1]*self.layers[:,3]))
        self.p0=.85*self.fc*(self.ag-self.ast)+self.steel_force
        if self.ast>=self.ag:raise ValueError('Steel displaces the entire section')

    def state(self,c,compression_face='top',phi_mode='auto',fy_factor=1.):
        c=np.asarray(c,dtype=float)
        if (c<=0).any() or not np.isfinite(c).all():raise ValueError('Neutral-axis depth must be positive and finite')
        if compression_face not in ('top','bottom') or phi_mode not in ('auto','nominal'):raise ValueError('Unknown face or phi mode')
        if not math.isfinite(fy_factor) or fy_factor<1:raise ValueError('Strength fy factor must be finite and at least one')
        if phi_mode=='auto' and fy_factor!=1.:raise ValueError('Probable steel strength is not a factored demand capacity')
        a=self.beta*c;cc=np.zeros(c.shape);qc=np.zeros(c.shape)
        for y0,y1,b,fc in self.regions:
            lo,hi=(y0,y1) if compression_face=='top' else (self.h-y1,self.h-y0)
            height=np.clip(a-lo,0.,hi-lo)
            force=.85*fc*b*height;cc+=force;qc+=force*(lo+height/2)
        depth=self.layers[:,0] if compression_face=='top' else self.h-self.layers[:,0]
        strain=.003*(1-depth/c[...,None]);fy=self.layers[:,3]*fy_factor
        steel=np.clip(self.es*strain,-fy,fy)*self.layers[:,1]
        cut=np.clip(a[...,None]-depth,-self.radius,self.radius)
        root=np.sqrt(np.maximum(0.,self.radius**2-cut**2))
        count=self.layers[:,1]/self.layers[:,2]
        removed=(self.radius**2*(np.arcsin(cut/self.radius)+math.pi/2)+cut*root)*count
        # Portion above block edge: first moment about bar centroid is negative
        # in compression-face depth coordinates.
        removed_q=(removed*depth-2/3*root**3*count)
        pn=cc+np.sum(steel-.85*self.fc*removed,axis=-1)
        q=qc+np.sum(steel*depth-.85*self.fc*removed_q,axis=-1)
        mn=pn*self.h/2-q
        if compression_face=='bottom':mn=-mn
        et=-strain.min(axis=-1)
        phi=np.ones(c.shape) if phi_mode=='nominal' else .65+.25*np.clip((et-max(self.layers[:,3])/self.es)/.003,0.,1.)
        return dict(p=phi*pn,m=phi*mn,pn=pn,mn=mn,phi=phi,c=c,eps_t=et,
                    steel_forces_nominal_kip=steel,concrete_force_nominal_kip=cc-.85*self.fc*removed.sum(axis=-1),
                    compression_face=compression_face,phi_mode=phi_mode,method=METHOD)

    def _axial_only(self,c,compression_face,phi_mode,fy_factor):
        """phi Pn of state() at a neutral axis depth, the same arithmetic without the moment and force records."""
        a=self.beta*c;cc=np.zeros(c.shape)
        for y0,y1,b,fc in self.regions:
            lo,hi=(y0,y1) if compression_face=='top' else (self.h-y1,self.h-y0)
            height=np.clip(a-lo,0.,hi-lo)
            cc+=.85*fc*b*height
        depth=self.layers[:,0] if compression_face=='top' else self.h-self.layers[:,0]
        strain=.003*(1-depth/c[...,None]);fy=self.layers[:,3]*fy_factor
        steel=np.clip(self.es*strain,-fy,fy)*self.layers[:,1]
        cut=np.clip(a[...,None]-depth,-self.radius,self.radius)
        root=np.sqrt(np.maximum(0.,self.radius**2-cut**2))
        count=self.layers[:,1]/self.layers[:,2]
        removed=(self.radius**2*(np.arcsin(cut/self.radius)+math.pi/2)+cut*root)*count
        pn=cc+np.sum(steel-.85*self.fc*removed,axis=-1)
        et=-strain.min(axis=-1)
        phi=np.ones(c.shape) if phi_mode=='nominal' else .65+.25*np.clip((et-max(self.layers[:,3])/self.es)/.003,0.,1.)
        return phi*pn

    def at_axial(self,p,compression_face='top',phi_mode='auto',fy_factor=1.):
        p=np.asarray(p,dtype=float)
        lower=-self.steel_force*fy_factor*(.9 if phi_mode=='auto' else 1.)
        upper=(.65 if phi_mode=='auto' else 1.)*.8*(self.p0+(fy_factor-1)*self.steel_force)
        if not np.isfinite(p).all() or (p<lower-1e-8).any() or (p>upper+1e-8).any():raise ValueError('Axial demand outside supported section domain')
        lo=np.full(p.shape,1e-10);hi=np.full(p.shape,self.h*1000)
        for _ in range(64):
            mid=(lo+hi)/2;value=self._axial_only(mid,compression_face,phi_mode,fy_factor)
            hi=np.where(value>p,mid,hi);lo=np.where(value<=p,mid,lo)
        result=self.state((lo+hi)/2,compression_face,phi_mode,fy_factor)
        residual=np.abs(result['p']-p)
        if np.max(residual,initial=0.)>1e-8*max(self.p0,self.steel_force,1.):raise ValueError('Section axial equilibrium did not converge')
        result['axial_residual_kip']=result['p']-p
        return result

    def moment_interval(self,p,phi_mode='auto',fy_factor=1.):
        positive=self.at_axial(p,'top',phi_mode,fy_factor);negative=self.at_axial(p,'bottom',phi_mode,fy_factor)
        return dict(minimum=negative['m'],maximum=positive['m'],positive=positive,negative=negative,method=METHOD)
