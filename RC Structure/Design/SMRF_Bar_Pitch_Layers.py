"""Mesh-independent actual-mat layer mapping, pending engineering qualification.

Polak/Vecchio1993 pp3441-3442 Fig2 and Irawan/Maekawa1997 pp123-124
apply layered reinforcement at in-plane integration points. A mesh cell's
interception of discrete cylinders is a different spatial material model.
This module defines fixed symmetric strips from actual neighboring mat bars,
before any FE mesh exists. Each strip preserves its bar's steel volume,
all three first moments and depth second moment. It intentionally changes
the second transverse plan moment, which is recorded, not asserted exact.

Only straight slab-owned mats are mapped. Beam steel, hook/bend ownership,
development, strength acceptance and physical field qualification are separate.
No existing strength/accounting gate is changed by importing this module.
"""
from __future__ import annotations
import math
import numpy as np
from Design.SMRF_Physical_Floor_Materials import _hash,ClippedBar

METHOD='actual_symmetric_bar_pitch_layers_v1_unqualified'
POINT_METHOD='actual_fixed_pitch_point_layers_and_cylinder_voids_v1_unqualified'


class BarPitchLayer:
    def __init__(self,bar,pitch_interval,*,footprint,slab_thickness_in,inventory_identity):
        self.bar=dict(bar);self.h=float(slab_thickness_in);self.source=inventory_identity
        if not isinstance(self.source,str) or not self.source or not bar.get('id') or bar.get('operator_owner')!='slab' or bar.get('axis') not in ('x','y'):
            raise ValueError('Actual slab-owned bar and bound inventory required')
        if any(not math.isfinite(float(bar[k])) or float(bar[k])<=0 for k in ('fy_ksi','es_ksi')):
            raise ValueError('Actual positive finite steel strengths and modulus required')
        self.axis=bar['axis'];self.c=float(bar['s_in']);self.z=float(bar['z_in'])
        self.area=float(bar['area_in2']);self.db=float(bar['db_in']);self.r=math.sqrt(self.area/math.pi) if self.area>0 else 0.
        self.pitch=tuple(map(float,pitch_interval));self.run=tuple(map(float,bar['straight_run_interval_in']))
        if (len(self.pitch)!=2 or len(self.run)!=2 or not all(math.isfinite(v) for v in (*self.pitch,*self.run,self.c,self.z,self.h,self.area,self.db))
                or self.pitch[1]<=self.pitch[0] or self.run[1]<=self.run[0] or min(self.h,self.area,self.db)<=0):
            raise ValueError('Finite positive actual layer geometry required')
        self.width=self.pitch[1]-self.pitch[0]
        if abs(sum(self.pitch)/2-self.c)>1e-10:raise ValueError('Symmetric fixed strip must preserve the actual bar centroid')
        normal='y' if self.axis=='x' else 'x'
        if (self.pitch[0]<footprint[normal+'_min_in']-1e-10 or self.pitch[1]>footprint[normal+'_max_in']+1e-10
                or self.run[0]<footprint[self.axis+'_min_in']-1e-10 or self.run[1]>footprint[self.axis+'_max_in']+1e-10):
            raise ValueError('Actual straight layer must lie within the physical footprint')
        if self.width<2*self.r or abs(self.z)+self.r>self.h/2:
            raise ValueError('Nominal finite bar volume does not fit the declared layer')
        self.identity=_hash(dict(method=METHOD,bar=self.bar,pitch=self.pitch,footprint=footprint,h=self.h,source=self.source))

    def cell(self,bounds):
        x0,x1,y0,y1=map(float,bounds)
        if not all(math.isfinite(v) for v in (x0,x1,y0,y1)) or x1<=x0 or y1<=y0:
            raise ValueError('Positive finite cell required')
        along=(x0,x1) if self.axis=='x' else (y0,y1);across=(y0,y1) if self.axis=='x' else (x0,x1)
        run=[max(along[0],self.run[0]),min(along[1],self.run[1])]
        band=[max(across[0],self.pitch[0]),min(across[1],self.pitch[1])]
        area=(x1-x0)*(y1-y0);length=max(0.,run[1]-run[0]);width=max(0.,band[1]-band[0])
        coefficient=length*width/self.width
        moments=coefficient*self.area*np.array([1.,self.z,self.z*self.z+self.r*self.r/4])
        along_first=moments[0]*sum(run)/2;across_first=moments[0]*sum(band)/2
        mx,my=(along_first,across_first) if self.axis=='x' else (across_first,along_first)
        record=dict(id=self.bar['id'],axis=self.axis,operator_owner='slab',method=METHOD,
            layer_identity=self.identity,inventory_identity=self.source,bounds_in=[x0,x1,y0,y1],
            fixed_pitch_interval_in=list(self.pitch),clipped_run_interval_in=run,
            straight_volume_in3=float(moments[0]),first_x_in4=float(mx),first_y_in4=float(my),
            first_z_in4=float(moments[1]),second_z_in5=float(moments[2]),
            layer_area_in2_per_in=float(moments[0]/area),
            whole_strip_transverse_second_moment_increment_per_unit_volume_in2=float(self.width*self.width/12-self.r*self.r/4),
            engineering_qualification_inferred=False)
        return _CellLayer(self,coefficient/area,record)


class _CellLayer:
    def __init__(self,layer,density_coefficient,record):
        self.layer=layer;self.coefficient=density_coefficient;self.record=record
    def occupied(self,z):
        z=np.asarray(z,float);r=self.layer.r
        return self.coefficient*2*np.sqrt(np.maximum(0.,r*r-(z-self.layer.z)**2))
    def steel_layer(self):
        b=self.layer.bar
        return dict(id=b['id'],axis=b['axis'],z_in=b['z_in'],area_in2_per_in=self.record['layer_area_in2_per_in'],
            bar_area_in2=b['area_in2'],db_in=b['db_in'],fy_ksi=b['fy_ksi'],es_ksi=b['es_ksi'])


def installed_mat_layers(inventory,level):
    """Freeze each actual family before meshing; never infer a bar or its pitch.

    Width is the lesser adjacent center spacing, limited symmetrically by the
    actual footprint. Uniform mats retain their actual spacing. Unequal end
    cover or irregular spacing can leave explicitly unreinforced gaps. These
    are exposed in the record, not filled by invented fractional bars.
    """
    material=inventory['floors'][str(level)];fp=material['footprint'];h=inventory['slab_thickness_in']
    source=inventory.get('identity_sha256');families={};ids=set()
    for bar in material['bars']:
        if bar['operator_owner']!='slab':continue
        if bar['id'] in ids:raise ValueError('Duplicate actual mat bar identity')
        ids.add(bar['id'])
        if not bar.get('mat_name'):raise ValueError('Authoritative actual mat family required')
        families.setdefault((bar['axis'],bar['mat_name']),[]).append(bar)
    layers=[];audits=[]
    for (axis,name),bars in sorted(families.items()):
        bars.sort(key=lambda b:float(b['s_in']));positions=np.array([float(b['s_in']) for b in bars])
        if len(bars)<2 or not np.isfinite(positions).all() or np.any(np.diff(positions)<=0):
            raise ValueError('At least two distinct ordered actual mat bars define a pitch')
        cross='y' if axis=='x' else 'x';lo,hi=float(fp[cross+'_min_in']),float(fp[cross+'_max_in']);bands=[]
        for i,bar in enumerate(bars):
            s=positions[i];distances=[2*(s-lo),2*(hi-s)]
            if i:distances.append(s-positions[i-1])
            if i+1<len(bars):distances.append(positions[i+1]-s)
            width=min(distances);band=[float(s-width/2),float(s+width/2)]
            layer=BarPitchLayer(bar,band,footprint=fp,slab_thickness_in=h,inventory_identity=source)
            if bands and band[0]<bands[-1][1]-1e-10:raise ValueError('Actual fixed mat strips overlap')
            bands.append(band);layers.append(layer)
        gaps=[];cursor=lo
        for a,b in bands:
            if a>cursor+1e-10:gaps.append([cursor,a])
            cursor=b
        if cursor<hi-1e-10:gaps.append([cursor,hi])
        audits.append(dict(axis=axis,mat_name=name,actual_bar_count=len(bars),fixed_strips=bands,
            uncovered_intervals_in=gaps,adjacent_center_spacings_in=np.diff(positions).tolist()))
    return layers,dict(method=METHOD,inventory_identity=source,level=int(level),families=audits,
        production_analysis_or_strength_activated=False,engineering_qualified=False)


class FixedPitchFloorMaterial:
    """One bound whole-floor mapping; cell queries cannot redefine its pitch."""
    def __init__(self,inventory,level):
        self.layers,self.audit=installed_mat_layers(inventory,level)
        self.state=inventory['state_identity'];self.source=inventory['identity_sha256']
        self.h=float(inventory['slab_thickness_in'])
        self.footprint=dict(inventory['floors'][str(level)]['footprint'])
        bars=inventory['floors'][str(level)]['bars']
        if len({b['id'] for b in bars})!=len(bars) or any(b['operator_owner'] not in ('slab','web_line') for b in bars):
            raise ValueError('Unique complete slab/web ownership required')
        self.beam_bars=[dict(b) for b in bars if b['operator_owner']=='web_line']
        # A point query may miss an invalid cylinder entirely (e.g. NaN fails
        # every intersection comparison). Validate its physical input once,
        # independently of which cells/points the caller happens to sample.
        fp=self.footprint;whole=(fp['x_min_in'],fp['x_max_in'],fp['y_min_in'],fp['y_max_in'])
        if not math.isfinite(self.h) or self.h<=0:raise ValueError('Positive finite physical slab thickness required')
        for bar in self.beam_bars:
            ClippedBar(bar,whole,self.h)
            if not math.isfinite(float(bar['db_in'])) or float(bar['db_in'])<=0:
                raise ValueError('Positive finite actual beam bar diameter required')
        self.identity=_hash(dict(mapping=self.audit,layer_ids=[p.identity for p in self.layers],beam_bars=self.beam_bars,state=self.state))

    def profile(self,bounds):return _PitchCellProfile(self,bounds)

    def point_profile(self,x_in,y_in,*,trace=('plus','plus')):
        """Local field before cell averaging; one-sided at material boundaries.

        Mat steel occupies its fixed physical pitch/run strip. Beam-owned
        cylinders remove their exact local concrete chord but give no slab
        resistance credit. This query is independent of any FE mesh. The
        existing cell profile is its area average, not its value at each GP.
        No reference-builder or strength gate is changed by this method.
        """
        return _PitchPointProfile(self,x_in,y_in,trace)


def _inside_trace(value,interval,trace):
    if trace=='plus':return interval[0]<=value<interval[1]
    return interval[0]<value<=interval[1]


class _PitchPointProfile:
    def __init__(self,field,x_in,y_in,trace):
        self.x,self.y=float(x_in),float(y_in);self.h=field.h
        if (not math.isfinite(self.x) or not math.isfinite(self.y) or
                not isinstance(trace,(tuple,list)) or len(trace)!=2 or any(t not in ('plus','minus') for t in trace)):
            raise ValueError('Finite physical point and two explicit one-sided traces required')
        self.trace=tuple(trace);fp=field.footprint
        if not (fp['x_min_in']<=self.x<=fp['x_max_in'] and fp['y_min_in']<=self.y<=fp['y_max_in']):
            raise ValueError('Material point lies outside the physical floor')
        self.mats=[];self.beam=[];self.records=[]
        for p in field.layers:
            a,c,ta,tc=(self.x,self.y,trace[0],trace[1]) if p.axis=='x' else (self.y,self.x,trace[1],trace[0])
            if not (_inside_trace(a,p.run,ta) and _inside_trace(c,p.pitch,tc)):continue
            moments=p.area/p.width*np.array([1.,p.z,p.z*p.z+p.r*p.r/4])
            self.mats.append(p)
            self.records.append(dict(id=p.bar['id'],axis=p.axis,operator_owner='slab',
                layer_identity=p.identity,depth_moments_per_unit_area=moments.tolist()))
        for bar in field.beam_bars:
            a,c,ta=(self.x,self.y,trace[0]) if bar['axis']=='x' else (self.y,self.x,trace[1])
            if not _inside_trace(a,bar['straight_run_interval_in'],ta):continue
            radius=math.sqrt(float(bar['area_in2'])/math.pi);distance=c-float(bar['s_in'])
            if abs(distance)>=radius:continue
            half=math.sqrt((radius-abs(distance))*(radius+abs(distance)))
            lo=max(-self.h/2,float(bar['z_in'])-half);hi=min(self.h/2,float(bar['z_in'])+half)
            if hi<=lo:continue
            moments=np.array([hi-lo,(hi*hi-lo*lo)/2,(hi**3-lo**3)/3])
            self.beam.append((dict(bar),lo,hi))
            self.records.append(dict(id=bar['id'],axis=bar['axis'],operator_owner='web_line',
                depth_interval_in=[lo,hi],depth_moments_per_unit_area=moments.tolist()))
        self.void_only_ids=sorted(b['id'] for b,_,_ in self.beam)
        self.breaks=sorted({z for p in self.mats for z in (p.z-p.r,p.z+p.r)}|
                           {z for _,a,b in self.beam for z in (a,b)})
        self.exact_depth_moments_per_unit_area=sum((np.array(r['depth_moments_per_unit_area']) for r in self.records),np.zeros(3))
        self.identity=_hash(dict(method=POINT_METHOD,field_identity=field.identity,point=[self.x,self.y],
                                 trace=self.trace,records=self.records))

    def occupied(self,z):
        z=np.asarray(z,float);out=np.zeros_like(z)
        for p in self.mats:out+=2*np.sqrt(np.maximum(0.,p.r*p.r-(z-p.z)**2))/p.width
        for _,lo,hi in self.beam:out+=((z>lo)&(z<hi)).astype(float)
        return out

    def slab_steel_layers(self):
        return [dict(id=p.bar['id'],axis=p.axis,z_in=p.z,area_in2_per_in=p.area/p.width,
                     bar_area_in2=p.area,db_in=p.db,fy_ksi=p.bar['fy_ksi'],es_ksi=p.bar['es_ksi']) for p in self.mats]


class _PitchCellProfile:
    def __init__(self,field,bounds):
        self.bounds=tuple(map(float,bounds));self.h=field.h
        if len(self.bounds)!=4 or not all(math.isfinite(v) for v in self.bounds) or self.bounds[1]<=self.bounds[0] or self.bounds[3]<=self.bounds[2]:
            raise ValueError('Positive finite cell bounds required')
        self.area=(self.bounds[1]-self.bounds[0])*(self.bounds[3]-self.bounds[2])
        mats=[p.cell(self.bounds) for p in field.layers]
        self.mats=[p for p in mats if p.record['straight_volume_in3']>1e-14]
        beam=[ClippedBar(b,self.bounds,self.h) for b in field.beam_bars]
        self.beam=[p for p in beam if p.moments[0]>1e-14]
        self.records=[p.record for p in self.mats]+[p.record() for p in self.beam]
        self.void_only_ids=sorted(p.bar['id'] for p in self.beam)
        self.breaks=sorted({v for p in self.mats for v in (p.layer.z-p.layer.r,p.layer.z+p.layer.r)}|
                           {v for p in self.beam for v in p.breaks})
        self.exact_depth_moments=np.array([sum(r[k] for r in self.records) for k in ('straight_volume_in3','first_z_in4','second_z_in5')])
        self.identity=_hash(dict(method=METHOD,field_identity=field.identity,bounds=self.bounds,records=self.records))
    def occupied(self,z):
        return sum((p.occupied(z) for p in [*self.mats,*self.beam]),np.zeros_like(np.asarray(z,float)))
    def slab_steel_layers(self):return [p.steel_layer() for p in self.mats]
