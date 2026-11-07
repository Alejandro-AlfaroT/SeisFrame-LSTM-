"""Physical outer-face diaphragm adapter for signed conventional ASCE forces.

This maps a frozen physical floor and source-bound frame influence record to
the already verified membrane action engine. It does not claim strength,
mesh convergence, stiffness qualification or a new nonlinear load response.
The verified zero-inertia second-order remainder remains a whole transfer
free body; code factors multiply forces, not an invented nonlinear response.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from Model.Floor_Footprint import from_perimeter_beams,column_overlap
from Design.SMRF_Coupled_Floor_Design import validate_definition,physical_grid
from Design import SMRF_Diaphragm_Actions as actions
from Design import SMRF_Floor_Influences as influences_api

METHOD='physical_outer_face_consistent_affine_joint_inertia_adapter_v3'


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def installed_grid_masses(definition):
    """Freeze the complete actual frame mass inventory, in weight units kip."""
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Model.Build_Model import installed_node_seismic_mass,node_tag
    validate_definition(definition)
    if mg.active() is None or mg.active().identity()!=definition['state_identity']:
        raise ValueError('Frozen floor and installed mass state differ')
    g=definition['geometry']
    if (g['nx'],g['ny'],g['nf'],g['lx'],g['ly'])!=(sp.NUM_BAY_X,sp.NUM_BAY_Y,sp.NUM_FLOOR,sp.BAY_X,sp.BAY_Y):
        raise ValueError('Frozen floor and installed mass geometry differ')
    return {str(k):[{'node':node_tag(k,i,j),'xy_in':[i*g['lx'],j*g['ly']],
                     'weight_kip':installed_node_seismic_mass(k,i,j)*sp.G}
                    for j in range(g['ny']+1) for i in range(g['nx']+1)] for k in range(1,g['nf']+1)}


def rectangle_weights(x,y,bounds):
    """Exact nonnegative Q4 shape integrals over a clipped physical rectangle."""
    x=np.asarray(x,float);y=np.asarray(y,float);x0,x1,y0,y1=map(float,bounds)
    if not(x[0]<=x0<x1<=x[-1] and y[0]<=y0<y1<=y[-1]):
        raise ValueError('Positive physical traction rectangle must lie inside the slab mesh')
    out=np.zeros(len(x)*len(y))
    for j,(ya,yb) in enumerate(zip(y[:-1],y[1:])):
        c,d=max(ya,y0),min(yb,y1)
        if d<=c:continue
        upper=(d-c)*((c+d)/2-ya)/(yb-ya);wy=[d-c-upper,upper]
        for i,(xa,xb) in enumerate(zip(x[:-1],x[1:])):
            a,b=max(xa,x0),min(xb,x1)
            if b<=a:continue
            right=(b-a)*((a+b)/2-xa)/(xb-xa);wx=[b-a-right,right]
            for dj in (0,1):
                for di in (0,1):out[(j+dj)*len(x)+i+di]+=wx[di]*wy[dj]
    return out


def _point_weights(x,y,point):
    px,py=map(float,point)
    if not(x[0]<=px<=x[-1] and y[0]<=py<=y[-1]):
        raise ValueError('Actual residual member mass centroid lies outside physical slab; positive projection unsupported')
    i=min(len(x)-2,max(0,int(np.searchsorted(x,px,side='right'))-1))
    j=min(len(y)-2,max(0,int(np.searchsorted(y,py,side='right'))-1))
    a,b=(px-x[i])/(x[i+1]-x[i]),(py-y[j])/(y[j+1]-y[j])
    result=np.zeros(len(x)*len(y))
    for di,wx in ((0,1-a),(1,a)):
        for dj,wy in ((0,1-b),(1,b)):result[(j+dj)*len(x)+i+di]+=wx*wy
    return result


def _empty_measure(count):
    return {'weights':np.zeros(count),'first_x':np.zeros(count),'first_y':np.zeros(count),'second':0.}


def _add_measure(target,measure,factor):
    for key in ('weights','first_x','first_y'):target[key]+=factor*measure[key]
    target['second']+=factor*measure['second']


def point_shape_measure(x,y,point):
    """Consistent positive barycentric projection of a physical point mass."""
    w=_point_weights(x,y,point);px,py=map(float,point)
    return {'weights':w,'first_x':w*px,'first_y':w*py,'second':px*px+py*py}


def rectangle_shape_measure(x,y,bounds):
    """Exact integrals Ni,Ni*x,Ni*y and x²+y² of one clipped rectangle.

    Two Gauss points per coordinate integrate the Q4/affine-load products
    exactly, including cells intersected only partly by a column footprint.
    Nodal radius times integral(Ni) is not substituted for integral(Ni*r).
    """
    x=np.asarray(x,float);y=np.asarray(y,float);x0,x1,y0,y1=map(float,bounds)
    if not(x[0]<=x0<x1<=x[-1] and y[0]<=y0<y1<=y[-1]):raise ValueError('Positive rectangle inside actual physical mesh required')
    result=_empty_measure(len(x)*len(y));gauss=(-1/math.sqrt(3),1/math.sqrt(3))
    for j,(ya,yb) in enumerate(zip(y[:-1],y[1:])):
        c,d=max(ya,y0),min(yb,y1)
        if d<=c:continue
        for i,(xa,xb) in enumerate(zip(x[:-1],x[1:])):
            a,b=max(xa,x0),min(xb,x1)
            if b<=a:continue
            for xi in gauss:
                px=(a+b)/2+xi*(b-a)/2;wx=[(xb-px)/(xb-xa),(px-xa)/(xb-xa)]
                for eta in gauss:
                    py=(c+d)/2+eta*(d-c)/2;wy=[(yb-py)/(yb-ya),(py-ya)/(yb-ya)];weight=(b-a)*(d-c)/4
                    result['second']+=weight*(px*px+py*py)
                    for dj in (0,1):
                        for di in (0,1):
                            index=(j+dj)*len(x)+i+di;v=weight*wx[di]*wy[dj]
                            result['weights'][index]+=v;result['first_x'][index]+=v*px;result['first_y'][index]+=v*py
    return result


def line_shape_measure(x,y,axis,line,start,end):
    """Exact clear-web line mass moments; positive two-point Gauss projection."""
    if axis not in ('x','y') or end<=start:raise ValueError('Actual positive aligned clear web line required')
    grid=np.asarray(x if axis=='x' else y,float)
    cuts=[start]+[float(v) for v in grid if start<v<end]+[end];result=_empty_measure(len(x)*len(y))
    for a,b in zip(cuts,cuts[1:]):
        for sign in (-1.,1.):
            along=(a+b)/2+sign*(b-a)/(2*math.sqrt(3));point=[along,line] if axis=='x' else [line,along]
            _add_measure(result,point_shape_measure(x,y,point),(b-a)/2)
    return result


def _measure_properties(measure):
    total=float(np.sum(measure['weights']));first=np.array([np.sum(measure['first_x']),np.sum(measure['first_y'])])
    if total<=0:raise ValueError('Positive physical inertia/traction measure required')
    center=first/total;polar=float(measure['second']-total*np.dot(center,center))
    return total,first,center,polar


def consistent_affine_forces(measure,force_kip,torque_at_centroid=0.):
    """Consistent nodal forces for ONE fixed physical affine load function."""
    total,first,center,polar=_measure_properties(measure);force=np.asarray(force_kip,float);torque=float(torque_at_centroid)
    if force.shape!=(2,) or not np.isfinite(force).all() or not math.isfinite(torque):raise ValueError('Finite physical force/couple required')
    w=measure['weights'];out=w[:,None]*force/total
    if torque:
        if polar<=0:raise ValueError('A physical couple requires a finite polar measure')
        out+=torque/polar*np.c_[-(measure['first_y']-center[1]*w),measure['first_x']-center[0]*w]
    return out


def affine_submeasure_wrench(total_measure,submeasure,force_kip,torque_at_centroid=0.,origin_xy_in=(0.,0.)):
    """Exact force and moment of that same continuum load on a fixed subset."""
    total,first,center,polar=_measure_properties(total_measure)
    weight=float(np.sum(submeasure['weights']));subfirst=np.array([np.sum(submeasure['first_x']),np.sum(submeasure['first_y'])])
    force=np.asarray(force_kip,float);origin=np.asarray(origin_xy_in,float);torque=float(torque_at_centroid)
    answer=weight*force/total;armfirst=subfirst-weight*origin
    moment=(armfirst[0]*force[1]-armfirst[1]*force[0])/total
    if torque:
        if polar<=0:raise ValueError('A physical couple requires a finite polar measure')
        answer+=torque/polar*np.array([-(subfirst[1]-center[1]*weight),subfirst[0]-center[0]*weight])
        moment+=torque/polar*(submeasure['second']-np.dot(origin+center,subfirst)+np.dot(origin,center)*weight)
    return np.r_[answer,moment]


def force_through_joint(measure,force_kip,application_xy_in,moment_at_application=0.):
    """Deliver a joint wrench through its actual finite traction footprint.

    This changes only delivery within the rigid column core. It preserves
    the point-lumped inertia force and moment, adding no member polar inertia.
    An offset mass centroid retains its real r cross F at the joint center.
    """
    center=_measure_properties(measure)[2];force=np.asarray(force_kip,float)
    arm=np.asarray(application_xy_in,float)-center
    torque=float(moment_at_application)+arm[0]*force[1]-arm[1]*force[0]
    return consistent_affine_forces(measure,force,torque)


def centered_target_axis(axis,targets,*,previous_axis=(),radius_fraction=.4):
    """Add actual section-centered cells, preserving every geometry/grid line.

    No stress averaging or point movement is involved. Previous axis lines
    can be retained for a nested refinement; the new target cell is strictly
    inside the old one. Targets coincident with required lines need explicit
    one-sided analysis instead and are refused here.
    """
    base=np.unique(np.r_[np.asarray(axis,float),np.asarray(previous_axis,float)])
    targets=np.unique(np.asarray(targets,float))
    if len(base)<2 or not np.isfinite(base).all() or not np.isfinite(targets).all() or not 0<radius_fraction<.5:
        raise ValueError('Finite increasing physical axis and radius fraction in(0,.5) required')
    additions=[];cells=[]
    for target in targets:
        if not base[0]<target<base[-1] or np.min(np.abs(base-target))<1e-8:
            raise ValueError('Centered target must lie strictly inside a physical cell, away from required mesh lines')
        others=targets[np.abs(targets-target)>1e-8]
        clearance=min(float(np.min(np.abs(base-target))),float(np.min(np.abs(others-target)))/2 if len(others) else float('inf'))
        radius=radius_fraction*clearance
        if radius<1e-7:raise ValueError('Centered physical sections are too close for a resolved positive cell')
        additions.extend([target-radius,target+radius]);cells.append({'target_in':float(target),'from_in':float(target-radius),'to_in':float(target+radius)})
    return np.unique(np.r_[base,additions]).tolist(),cells


class PhysicalDiaphragm:
    def __init__(self,definition,floor,*,grid_mass_by_floor,seismic_live_load_fraction,
                 source_identity,membrane_stiffness_factor=1.,web_axial_stiffness_factor=1.,
                 subdivisions_per_bay=2,mesh_axes=None,maximum_nodes=10000):
        validate_definition(definition)
        if not source_identity:raise ValueError('Physical diaphragm source identity is required')
        self.definition=definition;self.floor=int(floor);self.source_identity=str(source_identity)
        g=definition['geometry'];self.geometry=g
        if type(floor) is not int or not 1<=floor<=g['nf']:raise ValueError('Actual elevated target floor required')
        if not 0<=float(seismic_live_load_fraction)<=1:raise ValueError('Explicit seismic live-load fraction must be in[0,1]')
        if not 0<float(web_axial_stiffness_factor)<=1 or not 0<float(membrane_stiffness_factor)<=1:
            raise ValueError('Explicit membrane/web stiffness factors must be in(0,1]')
        self.description=next(f for f in definition['floors'] if f['level']==floor)
        self.footprint=from_perimeter_beams(g,self.description['beams'])
        x,y=physical_grid(self.description,g,subdivisions_per_bay=subdivisions_per_bay) if mesh_axes is None else mesh_axes
        x,y=np.asarray(x,float),np.asarray(y,float)
        if (len(x)*len(y)>maximum_nodes or len(x)<2 or len(y)<2 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0)):
            raise ValueError('Finite increasing physical diaphragm grid within declared node bound required')
        expected=[self.footprint[k] for k in ('x_min_in','x_max_in','y_min_in','y_max_in')]
        if [x[0],x[-1],y[0],y[-1]]!=expected:raise ValueError('Membrane mesh must cover the exact physical slab footprint')
        self.x,self.y=x,y
        supports={(s['x'],s['y']):s for s in self.description['supports']}
        for s in supports.values():
            if s['x'] not in x or s['y'] not in y:raise ValueError('Every actual joint center must be a physical mesh node')
        h=definition['slab_thickness_in'];gamma=definition['gamma_kip_in3'];beams=[]
        self.clear_beams=[]
        for b in self.description['beams']:
            axis=b['axis'];start=b['start'];end=b['end'];line=b['line']
            si=supports[(start,line) if axis=='x' else (line,start)]
            sj=supports[(end,line) if axis=='x' else (line,end)]
            ci=start+si['h_in' if axis=='x' else 'b_in']/2
            cj=end-sj['h_in' if axis=='x' else 'b_in']/2
            grid=x if axis=='x' else y;perp=y if axis=='x' else x
            if ci not in grid or cj not in grid or line not in perp or cj<=ci or b['h_in']<=h:
                raise ValueError('Exact actual beam faces and positive disjoint web dimensions required')
            clear=dict(b,start=ci,end=cj);self.clear_beams.append(clear)
            fixed=int(np.where(perp==line)[0][0]);first=int(np.where(grid==ci)[0][0]);last=int(np.where(grid==cj)[0][0])
            for station in range(first,last):
                ni=fixed*len(x)+station if axis=='x' else station*len(x)+fixed
                nj=ni+1 if axis=='x' else ni+len(x)
                beams.append(dict(id=f'{b["id"]}:external_axial:{station-first}',member_id=str(b['id']),i=ni,j=nj,
                    area_web_in2=b['b_in']*(b['h_in']-h),E_ksi=b['E_ksi'],stiffness_factor=web_axial_stiffness_factor,
                    axis=axis,start_in=float(grid[station]),end_in=float(grid[station+1])))
        self.mesh=actions.rectangular_mesh(x,y,thickness_in=h,elastic_modulus_ksi=self.description['E_ksi'],
                    membrane_stiffness_factor=membrane_stiffness_factor,beams=beams)
        self.column_footprints={};self.column_measures={};self.traction_metadata={}
        for c in definition['columns']:
            if c['story'] not in (floor,floor+1):continue
            cx,cy=c['grid_i']*g['lx'],c['grid_j']*g['ly']
            overlap=column_overlap(self.footprint,cx,cy,c['h_in'],c['b_in'])
            weights=rectangle_weights(x,y,[overlap[k] for k in ('x_min_in','x_max_in','y_min_in','y_max_in')])
            self.column_footprints[str(c['id'])]={i:float(v) for i,v in enumerate(weights) if v>0}
            self.column_measures[str(c['id'])]=rectangle_shape_measure(x,y,expected_bounds(overlap))
            self.traction_metadata[str(c['id'])]={'side':'below' if c['story']==floor else 'above',
                    'application_xy_in':[cx,cy],'physical_overlap':overlap,'reference_transport':True}
        self.grid_mass_by_floor=grid_mass_by_floor
        if set(grid_mass_by_floor)!=set(map(str,range(1,g['nf']+1))):raise ValueError('Complete building grid mass snapshot required')
        for desc in definition['floors']:
            rows=grid_mass_by_floor[str(desc['level'])];expected={s['id']:(s['x'],s['y']) for s in desc['supports']}
            if len(rows)!=len(expected) or {r['node'] for r in rows}!=set(expected):raise ValueError('Grid masses omit or duplicate actual joints')
            for r in rows:
                if tuple(r['xy_in'])!=expected[r['node']] or not math.isfinite(float(r['weight_kip'])) or r['weight_kip']<=0:
                    raise ValueError('Actual frame grid mass position/weight is invalid')
        q=(definition['floor_dead_load_ksf']+seismic_live_load_fraction*definition['floor_live_load_ksf'])/144.
        self.mass_measure=_empty_measure(len(x)*len(y))
        self.member_inertia_sources=[]
        self.mass_pieces=[('rectangle',expected_bounds(self.footprint),q)]
        _add_measure(self.mass_measure,rectangle_shape_measure(x,y,expected_bounds(self.footprint)),q)
        for b in self.clear_beams:
            _add_measure(self.mass_measure,line_shape_measure(x,y,b['axis'],b['line'],b['start'],b['end']),gamma*b['b_in']*(b['h_in']-h))
            self.mass_pieces.append(('line',[b['axis'],b['line'],b['start'],b['end']],gamma*b['b_in']*(b['h_in']-h)))
        for c in (c for c in definition['columns'] if c['story']==floor):
            cx,cy=c['grid_i']*g['lx'],c['grid_j']*g['ly'];full=c['b_in']*c['h_in']*g['story_h']
            overlap=column_overlap(self.footprint,cx,cy,c['h_in'],c['b_in']);removed=overlap['area_in2']*h
            volume=full-removed
            if volume<=0:raise ValueError('Nonpositive actual residual column volume')
            centroid=(full*np.array([cx,cy])-removed*np.array(overlap['centroid_in']))/volume
            _add_measure(self.mass_measure,point_shape_measure(x,y,centroid),gamma*volume)
            self.mass_pieces.append(('point',centroid.tolist(),gamma*volume))
            self.member_inertia_sources.append({'column_id':str(c['id']),'kind':'residual_column',
                'application_xy_in':centroid.tolist(),'weight_kip':gamma*volume})
            roof=float(c.get('roof_extension_weight_kip',0.))
            if roof<0 or (roof and floor!=g['nf']):raise ValueError('Roof extension weight must be nonnegative and roof-only')
            _add_measure(self.mass_measure,point_shape_measure(x,y,[cx,cy]),roof)
            if roof:
                self.mass_pieces.append(('point',[cx,cy],roof))
                self.member_inertia_sources.append({'column_id':str(c['id']),'kind':'roof_extension',
                    'application_xy_in':[cx,cy],'weight_kip':roof})
        self.mass_weights=self.mass_measure['weights']
        rows=grid_mass_by_floor[str(floor)];grid_w=np.array([r['weight_kip'] for r in rows]);grid_xy=np.array([r['xy_in'] for r in rows])
        xy=np.asarray(self.mesh['nodes']);expected_wrench=np.r_[grid_w.sum(),grid_w@grid_xy]
        actual_wrench=np.r_[self.mass_weights.sum(),self.mass_weights@xy]
        error=float(np.max(np.abs(actual_wrench-expected_wrench)/np.maximum(1.,np.abs(expected_wrench))))
        if min(self.mass_weights)<0 or error>1e-10:raise ValueError('Physical mass does not reproduce actual installed frame total/CM')
        self.mass_audit={'actual_grid_total_first_moments':expected_wrench.tolist(),'physical_total_first_moments':actual_wrench.tolist(),
             'maximum_relative_error':error,'center_xy_in':(actual_wrench[1:]/actual_wrench[0]).tolist(),
             'physical_polar_kip_in2':_measure_properties(self.mass_measure)[3],
             'basis':'physical slab area, clear web line mass, exact residual column centroid and roof extension; positive barycentric mass projection',
             'member_inertia_delivery':'same actual finite joint footprint as column traction; original point-lumped acceleration/rotation and wrench unchanged',
             'member_inertia_sources':self.member_inertia_sources}
        self.context={'method':METHOD,'definition_sha256':definition['identity_sha256'],'state_identity':definition['state_identity'],
            'source_identity':self.source_identity,'nodal_mass_sha256':identity(grid_mass_by_floor),'floor':floor,
            'mesh_sha256':identity(self.mesh),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.context['identity_sha256']=identity(self.context)

    def _left_measure(self,pieces,axis,coordinate):
        if axis not in ('x','y'):raise ValueError('Physical subset axis must be x or y')
        index=0 if axis=='x' else 1;result=_empty_measure(len(self.mesh['nodes']))
        for kind,geometry,density in pieces:
            if kind=='rectangle':
                bounds=list(geometry);pair=0 if axis=='x' else 2
                bounds[pair+1]=min(bounds[pair+1],coordinate)
                if bounds[pair+1]<=bounds[pair]:continue
                measure=rectangle_shape_measure(self.x,self.y,bounds)
            elif kind=='line':
                along,line,start,end=geometry
                if along==axis:end=min(end,coordinate)
                elif line>=coordinate:continue
                if end<=start:continue
                measure=line_shape_measure(self.x,self.y,along,line,start,end)
            else:
                if abs(geometry[index]-coordinate)<1e-9:raise ValueError('Cut through point inertia needs explicit side ownership')
                if geometry[index]>=coordinate:continue
                measure=point_shape_measure(self.x,self.y,geometry)
            _add_measure(result,measure,density)
        return result

    def component_wrench_on_left(self,row,axis,coordinate_in,*,origin_xy_in=(0.,0.)):
        """Exact applied continuum wrench left of a fixed physical plane.

        This integrates the same affine body/footprint load functions used
        for consistent nodal forces. It is independent of where mesh nodes
        happen to lie and must not be confused with a nodal step partition.
        """
        load=np.asarray(row['applied_load'],float);center=np.asarray(self.mass_audit['center_xy_in'])
        arm=np.asarray(row['master_xy_in'])-center;torque=load[2]+arm[0]*load[1]-arm[1]*load[0]
        distributed_pieces=[p for p in self.mass_pieces if p[0]!='point']
        result=affine_submeasure_wrench(self.mass_measure,self._left_measure(distributed_pieces,axis,coordinate_in),
                                        load[:2],torque,origin_xy_in)
        for source,force in self.member_inertia_forces(load,row['master_xy_in']):
            point=np.asarray(source['application_xy_in']);tag=source['column_id'];measure=self.column_measures[tag]
            centroid=_measure_properties(measure)[2];arm=point-centroid;couple=arm[0]*force[1]-arm[1]*force[0]
            subset=self._left_measure([('rectangle',expected_bounds(self.traction_metadata[tag]['physical_overlap']),1.)],axis,coordinate_in)
            result+=affine_submeasure_wrench(measure,subset,force,couple,origin_xy_in)
        for field in ('columns_above_on_floor','columns_below_on_floor'):
            for c in row[field]:
                tag=str(c['column_tag']);meta=self.traction_metadata[tag];rect=meta['physical_overlap'];force=np.array([c['Fx_kip'],c['Fy_kip']])
                arm=np.asarray(c['xy_in'])-np.array(rect['centroid_in']);torque=c['Mz_kip_in']+arm[0]*force[1]-arm[1]*force[0]
                subset=self._left_measure([('rectangle',expected_bounds(rect),1.)],axis,coordinate_in)
                result+=affine_submeasure_wrench(self.column_measures[tag],subset,force,torque,origin_xy_in)
        return result

    def member_inertia_forces(self,applied_load,master_xy_in):
        """Actual point-lumped forces before their finite joint delivery."""
        load=np.asarray(applied_load,float);total,_,center,polar=_measure_properties(self.mass_measure)
        arm=np.asarray(master_xy_in,float)-center;torque=load[2]+arm[0]*load[1]-arm[1]*load[0]
        out=[]
        for source in self.member_inertia_sources:
            delta=np.asarray(source['application_xy_in'])-center;weight=source['weight_kip']
            force=weight*load[:2]/total+weight*torque/polar*np.array([-delta[1],delta[0]])
            out.append((source,force))
        return out

    def external_inertia_forces(self,applied_load,reference_xy_in):
        """External floor inertia only, with the requested wrench at reference.

        The affine acceleration is integrated on the physical mass measure.
        Column/roof point masses keep their original acceleration and wrench,
        delivered through their actual rigid joint; no resisting column
        traction or membrane response is included.
        """
        load=np.asarray(applied_load,float);reference=np.asarray(reference_xy_in,float)
        if load.shape!=(3,) or reference.shape!=(2,) or not np.isfinite(load).all() or not np.isfinite(reference).all():
            raise ValueError('Finite [Fx,Fy,Mz] and explicit planar wrench reference required')
        center=np.asarray(self.mass_audit['center_xy_in']);arm=reference-center
        torque=load[2]+arm[0]*load[1]-arm[1]*load[0]
        inertia=consistent_affine_forces(self.mass_measure,load[:2],torque)
        for source,force in self.member_inertia_forces(load,reference):
            point=source['application_xy_in'];measure=self.column_measures[source['column_id']]
            inertia-=_point_weights(self.x,self.y,point)[:,None]*force
            inertia+=force_through_joint(measure,force,point)
        return inertia

    def components(self,floor_components):
        xy=np.asarray(self.mesh['nodes']);center=np.array(self.mass_audit['center_xy_in']);out=[]
        for row in floor_components:
            mapped={k:row[k] for k in ('id','role','provenance')}
            mapped['action_origin']=row.get('action_origin',row['role'])
            inertia=self.external_inertia_forces(row['applied_load'],row['master_xy_in'])
            mapped['inertia']=inertia.tolist()
            for field in ('columns_above_on_floor','columns_below_on_floor'):
                total=np.zeros_like(xy);seen=set();side='above' if field=='columns_above_on_floor' else 'below'
                for c in row[field]:
                    tag=str(c['column_tag'])
                    if tag in seen or tag not in self.column_footprints:raise ValueError('Unknown or duplicate actual column traction')
                    seen.add(tag);meta=self.traction_metadata[tag]
                    if meta['side']!=side or list(c['xy_in'])!=meta['application_xy_in']:raise ValueError('Column traction side/reference differs from physical member')
                    force=np.array([c['Fx_kip'],c['Fy_kip']])
                    total+=force_through_joint(self.column_measures[tag],force,c['xy_in'],c['Mz_kip_in'])
                expected={tag for tag,meta in self.traction_metadata.items() if meta['side']==side}
                if seen!=expected:raise ValueError('A physical above/below column is omitted from the floor free body')
                mapped[field]=total.tolist()
            out.append(mapped)
        return out

    def prepare_load_cases(self,influences,witness_id,*,conventional_forces,include_residual_sign_sensitivity=False):
        """Verify and map the complete signed P5 ledger without a membrane solve."""
        validate_definition(self.definition)
        if (self.definition['identity_sha256']!=self.context['definition_sha256'] or
            identity(self.mesh)!=self.context['mesh_sha256'] or identity(self.grid_mass_by_floor)!=self.context['nodal_mass_sha256']):
            raise ValueError('Physical diaphragm inputs changed after construction')
        for key,value in (('physical_definition_sha256',self.context['definition_sha256']),
                          ('state_identity',self.context['state_identity']),('nodal_mass_sha256',self.context['nodal_mass_sha256'])):
            if influences.get(key)!=value:raise ValueError(f'Influence record lacks matching computed physical binding: {key}')
        if influences.get('source_identity')!=self.source_identity:raise ValueError('Influence and physical source identities differ')
        self._verify_topology(influences)
        linearizations=self._verify_linearizations(influences)
        witness=next(w for w in influences['witnesses'] if w['id']==witness_id)
        conventional_forces=dict(conventional_forces)
        strength_factor=float(conventional_forces.pop('witness_strength_factor',1.))
        if not math.isfinite(strength_factor) or strength_factor<1.:
            raise ValueError('Finite witness strength factor at least one required')
        actual_elf=max(abs(float(v)) for v in witness['loads'][self.floor-1][:2])
        if not math.isclose(float(conventional_forces['elf_floor_force_kip'])*strength_factor,actual_elf,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Conventional ELF denominator differs from the actual principal signed floor force')
        decomposed=influences_api.decompose_with_second_order_residual(influences,witness_id,self.floor)
        components=self.components(decomposed['components'])
        direct=influences_api.floor_free_bodies(witness['direct_increment'],witness['loads'])[self.floor-1]
        mapped_direct=self.components([{**direct,'id':'original_direct','role':'local_inertia','provenance':'exact original direct witness'}])[0]
        reconstruction=max(float(np.max(np.abs(sum(np.asarray(c[key]) for c in components)-mapped_direct[key])))
                           for key in ('inertia','columns_above_on_floor','columns_below_on_floor'))
        if reconstruction>1e-7:raise ValueError('Physical traction mapping does not reconstruct original direct nodal fields')
        original_components=list(components)
        base_decomposed=None
        if strength_factor!=1.:
            normalized=witness.get('conventional_base_witness')
            if not normalized or normalized.get('strength_factor')!=strength_factor:
                raise ValueError('Actual base-ELF direct response required; nonlinear forces cannot be divided by redundancy')
            if not np.allclose(np.asarray(normalized['loads'])*strength_factor,witness['loads'],rtol=1e-12,atol=1e-12):
                raise ValueError('Base-ELF witness loads do not reproduce the complete original signed wrench')
            base_decomposed=influences_api.decompose_with_second_order_residual(
                {**influences,'witnesses':[{**normalized,'id':witness_id}]},witness_id,self.floor)
            # Distinct complete force ledgers: the five conventional cases use
            # a fresh base-ELF nonlinear solve. The original rho-scaled direct
            # response remains two additional guards, never a scaled solution.
            raw_base=[{**row,'id':'base_elf/'+row['id']} for row in base_decomposed['components']]
            base_components=self.components(raw_base)
            direct_base=influences_api.floor_free_bodies(normalized['direct_increment'],normalized['loads'])[self.floor-1]
            mapped_base=self.components([{**direct_base,'id':'base_direct','role':'local_inertia','provenance':'actual base ELF direct solve'}])[0]
            base_error=max(float(np.max(np.abs(sum(np.asarray(c[key]) for c in base_components)-mapped_base[key])))
                           for key in ('inertia','columns_above_on_floor','columns_below_on_floor'))
            if base_error>1e-7:raise ValueError('Base-ELF physical traction mapping does not reconstruct its actual direct response')
            components+=base_components
        else:
            raw_base=decomposed['components'];base_components=components;base_error=reconstruction
        local=[r['id'] for r in base_components if r['role']=='local_inertia'];transfer=[r['id'] for r in base_components if r['role']=='transfer']
        cases=actions.conventional_design_cases(local_component_ids=local,transfer_component_ids=transfer,**conventional_forces)
        if base_decomposed is not None:
            original_transfer={r['id'] for r in original_components if r['role']=='transfer'}
            for case in cases:
                if case['id']=='diaphragm_structural_analysis':
                    case['factors']={key.removeprefix('base_elf/'):value for key,value in case['factors'].items()}
                elif case['id'] in ('diaphragm_fpx','collector_minimum'):
                    case['factors']={(key.removeprefix('base_elf/') if key.removeprefix('base_elf/') in original_transfer else key):value
                                     for key,value in case['factors'].items()}
            # 12.3.4.1 permits rho=1 for Fpx and overstrength cases, not
            # ordinary structural/transfer effects. Retain their original
            # nonlinear force ledger, including signed redistribution.
        cases.append({'id':'structural_original_direct','scope':'diaphragm','basis':'retained exact original direct structural force ledger',
                      'factors':{r['id']:1. for r in original_components}})
        # Explicit exact Ω times original direct branch is retained even if
        # a separate Type4 tf exceeds Ω in the conventional collector case.
        cases.append({'id':'collector_original_direct_omega','scope':'collector','basis':'retained Omega0 times exact direct original force ledger',
                      'factors':{r['id']:float(conventional_forces['omega0']) for r in original_components}})
        if include_residual_sign_sensitivity:
            for case in list(cases):
                if case['id'] in ('diaphragm_fpx','collector_eq_omega','collector_minimum'):
                    residual=next(r['id'] for r in components if r['action_origin']=='second_order_redistribution' and r['id'] in case['factors'])
                    cases.append({**case,'id':case['id']+'_opposite_residual_sensitivity',
                                  'factors':{**case['factors'],residual:-case['factors'][residual]},
                                  'basis':case['basis']+'; finite opposite residual sensitivity, not nonlinear envelope'})
        equilibrium=[actions._component(self.mesh,row,1e-8)[1] for row in components]
        return {'adapter':self.context,'mesh':self.mesh,'influences_sha256':identity(influences),'witness_id':witness_id,
                'mass_audit':self.mass_audit,'traction_footprints':self.traction_metadata,
                'direct_reconstruction_max_abs_kip_kipin':decomposed['direct_reconstruction_max_abs_kip_kipin'],
                'physical_nodal_reconstruction_max_abs_kip':reconstruction,'linearization_replay':linearizations,
                'force_case_definitions':cases,'components':components,
                'raw_components':decomposed['components']+(raw_base if base_decomposed is not None else []),
                'conventional_force_basis':dict(witness_strength_factor=strength_factor,
                    separately_solved_base_elf=base_decomposed is not None,
                    redundancy_basis='ASCE7-22 12.3.4.1(5-7): base Fpx/overstrength; original structural and ordinary transfer effects retained',
                    base_physical_reconstruction_max_abs_kip=base_error,
                    base_direct_reconstruction_max_abs_kip_kipin=(base_decomposed or decomposed)['direct_reconstruction_max_abs_kip_kipin']),
                'component_equilibrium':equilibrium,
                'residual_method':'whole zero-inertia transfer free body; conventional signed FORCE amplification',
                'nonlinear_branch_resolve_asserted':False,'qualification_asserted':False,'official_pilot_started':False}

    def solve(self,influences,witness_id,*,conventional_forces,cuts=(),include_residual_sign_sensitivity=False,
              operator=None,retain_slab_samples=True):
        prepared=self.prepare_load_cases(influences,witness_id,conventional_forces=conventional_forces,
            include_residual_sign_sensitivity=include_residual_sign_sensitivity)
        result=actions.solve_diaphragm_actions(self.mesh,prepared['components'],prepared['force_case_definitions'],cuts=cuts,
            operator=operator,retain_slab_samples=retain_slab_samples)
        continuum_by_cut=[{row['id']:self.component_wrench_on_left(row,cut['axis'],cut['coordinate_in'])
                           for row in prepared['raw_components']} for cut in cuts]
        bysegment={b['id']:b for b in self.mesh['beams']}
        for case in result['cases']:
            members={}
            for b in case['beam_axial']:
                source=bysegment[b['id']]
                members.setdefault(b['member_id'],[]).append({'segment_id':b['id'],'axis':source['axis'],
                    'start_in':source['start_in'],'end_in':source['end_in'],'N_kip':b['N_kip'],'sign':'tension_positive'})
            case['member_external_axial']=members
            case['continuum_cut_equilibrium']=[]
            for cut,raw in zip(cuts,continuum_by_cut):
                applied=sum(case['factors'].get(key,0.)*value for key,value in raw.items())
                required=-applied;along=self.x if cut['axis']=='x' else self.y
                side=cut.get('side');has_trace=side is not None or np.min(np.abs(along-cut['coordinate_in']))>=1e-9
                trace=physical_membrane_cut(result,case['id'],cut['axis'],cut['coordinate_in'],side=side) if has_trace else None
                case['continuum_cut_equilibrium'].append({**cut,'origin_xy_in':[0.,0.],
                    'applied_continuum_left_wrench_kip_kipin':applied.tolist(),'required_total_cut_wrench_kip_kipin':required.tolist(),
                    'raw_trace':trace,'raw_trace_minus_required_kip_kipin':(np.array(trace['total_wrench_kip_kipin'])-required).tolist() if trace else None,
                    'basis':'exact full-width free-body resultant; distinct from raw stress and algebraic nodal partition',
                    'web_slab_allocation_verified':False,'qualification_asserted':False})
        return {**result,**prepared}

    def solve_batch(self,requests,*,operator=None,retain_slab_samples=False):
        """Many complete signed witnesses on one frozen physical floor/LU.

        Each request supplies id, influences, witness_id, conventional_forces,
        and optional cuts/include_residual_sign_sensitivity. No force-case
        response is reused between witnesses. Omitted GP dictionaries can be
        recovered exactly from saved displacements at actual physical cuts;
        this avoids multiplying large point-cloud records by every load case.
        """
        ids=[r['id'] for r in requests]
        if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError('Unique nonempty signed witness batch identities required')
        operator=actions.PreparedDiaphragmOperator(self.mesh) if operator is None else operator
        operator.validate(self.mesh);results={}
        for row in requests:
            results[row['id']]=self.solve(row['influences'],row['witness_id'],conventional_forces=row['conventional_forces'],
                cuts=row.get('cuts',()),include_residual_sign_sensitivity=row.get('include_residual_sign_sensitivity',False),
                operator=operator,retain_slab_samples=retain_slab_samples)
        return dict(method='same_actual_floor_signed_witness_batch_v1',adapter=self.context,results=results,
            prepared_operator=operator.record(),qualification_asserted=False,official_pilot_started=False)

    def _verify_topology(self,influences):
        if influences['num_floor']!=self.geometry['nf']:raise ValueError('Influence floor count differs')
        nodes={n['id']:n['xyz'] for n in self.definition['nodes']};columns={str(c['id']):c for c in self.definition['columns']}
        responses=[r['response'] for r in influences['basis']]+[w['direct_increment'] for w in influences['witnesses']]
        responses += [w['conventional_base_witness']['direct_increment'] for w in influences['witnesses'] if 'conventional_base_witness' in w]
        for response in responses:
            expected_masters=[[self.geometry['nx']*self.geometry['lx']/2,self.geometry['ny']*self.geometry['ly']/2,
                               k*self.geometry['story_h']] for k in range(1,self.geometry['nf']+1)]
            if [m['xyz'] for m in response['floor_masters']]!=expected_masters:
                raise ValueError('Influence master force references differ from actual frame grid centers')
            rows=response['columns']
            if len(rows)!=len(columns) or {str(c['tag']) for c in rows}!=set(columns):raise ValueError('Influence columns differ from actual physical topology')
            for c in rows:
                actual=columns[str(c['tag'])]
                if (c['i_node'],c['j_node'],c['i_level'],c['j_level'])!=(actual['i'],actual['j'],actual['story']-1,actual['story']):
                    raise ValueError('Influence column endpoint/story identity differs')
                if c['i_xyz']!=nodes[actual['i']] or c['j_xyz']!=nodes[actual['j']]:raise ValueError('Influence column coordinates differ')

    def _verify_linearizations(self,influences):
        rows=influences['basis'];nf=self.geometry['nf']
        if {(r['source_floor'],r['dof']) for r in rows}!={(k,d) for k in range(1,nf+1) for d in ('x','y','rz')} or len(rows)!=3*nf:
            raise ValueError('Complete unique three-coordinate central floor basis required')
        checks=[]
        for row in rows:
            saved=row.get('linearization_verification') or {};tol=saved.get('tolerances') or {}
            if 'coarse_response' not in saved:raise ValueError('Missing actual central step-halving response')
            load=np.zeros((nf,3));load[row['source_floor']-1,('x','y','rz').index(row['dof'])]=row['amplitude']
            kwargs={k:tol[v] for k,v in (('force_atol','force_kip'),('moment_atol','moment_kip_in'),
                    ('displacement_atol','displacement_in_rad'),('relative_tolerance','relative'))}
            trial={'num_floor':nf,'basis':[{**row,'response':saved['coarse_response']}]}
            check=influences_api.verify_superposition(trial,load,row['response'],**kwargs)
            if not check['passed']:raise ValueError('Central floor linearization does not reproduce its saved tolerance')
            checks.append({'floor':row['source_floor'],'dof':row['dof'],**check})
        return checks


def expected_bounds(footprint):
    return [footprint[k] for k in ('x_min_in','x_max_in','y_min_in','y_max_in')]


def conventional_force_inputs(elf,floor,*,sds,importance_factor,omega0,transfer_factor=1.,irregularity_factor=1.):
    """ASCE7-22 Eq12.10-1/2/3 from full SDS and the exact current ELF ledger."""
    forces=np.asarray(elf['story_forces_kip'],float);weights=np.asarray(elf['story_weights_kip'],float)
    if (forces.shape!=weights.shape or forces.ndim!=1 or not 1<=floor<=len(weights)
        or not np.isfinite(forces).all() or not np.isfinite(weights).all() or np.any(forces<=0) or np.any(weights<=0)):
        raise ValueError('Positive actual ELF force and seismic weight at every floor required')
    if min(sds,importance_factor,omega0)<=0:raise ValueError('Full positive SDS, importance and overstrength required')
    index=floor-1;eq=float(forces[index:].sum()/weights[index:].sum()*weights[index])
    return {'elf_floor_force_kip':float(forces[index]),'fpx_eq_kip':eq,
            'fpx_min_kip':.2*float(sds)*float(importance_factor)*weights[index],
            'fpx_max_kip':.4*float(sds)*float(importance_factor)*weights[index],
            'omega0':float(omega0),'transfer_factor':float(transfer_factor),'irregularity_factor':float(irregularity_factor)}


def physical_membrane_cut(result,case_id,axis,coordinate_in,*,origin_xy_in=(0.,0.),side=None):
    """Integrate actual Q4 stress trace and crossing disjoint web axial forces.

    A cut on a mesh line requires explicit ``minus`` or ``plus`` ownership;
    the two traces are never averaged. This is a physical
    stress trace used for refinement; it is distinct from the engine's exact
    algebraic nodal partition equilibrium cut. Return [Fx,Fy,Mz] on +normal.
    """
    mesh=result['mesh'];normal=0 if axis=='x' else 1 if axis=='y' else None
    if normal is None:raise ValueError('Physical membrane cut axis must be x or y')
    at=float(coordinate_in);along=np.array(mesh['x_in'] if axis=='x' else mesh['y_in'])
    across=np.array(mesh['y_in'] if axis=='x' else mesh['x_in'])
    if side not in (None,'minus','plus'):raise ValueError('Stress trace side must be minus or plus')
    on_line=np.min(np.abs(along-at))<1e-9
    if not along[0]<at<along[-1] or (on_line and side is None):raise ValueError('Stress trace requires explicit inside-cell cut ownership')
    owned=int(np.searchsorted(along,at,side='left' if side=='minus' else 'right')-1)
    origin=np.asarray(origin_xy_in,float);slab=np.zeros(3);web=np.zeros(3)
    case=next(c for c in result['cases'] if c['id']==case_id);xy=np.array(mesh['nodes'])
    for transverse,(a,b) in enumerate(zip(across[:-1],across[1:])):
        ix,iy=(owned,transverse) if axis=='x' else (transverse,owned)
        cell=mesh['cells'][iy*(len(mesh['x_in'])-1)+ix]
        displacement=np.asarray(case['displacements_in'])[cell].ravel()
        for sign in (-1.,1.):
            q=(a+b)/2+sign*(b-a)/(2*math.sqrt(3.));point=[at,q] if axis=='x' else [q,at]
            xi=2*(point[0]-mesh['x_in'][ix])/(mesh['x_in'][ix+1]-mesh['x_in'][ix])-1
            eta=2*(point[1]-mesh['y_in'][iy])/(mesh['y_in'][iy+1]-mesh['y_in'][iy])-1
            gradient,_,_=actions._q4(xy[cell],xi,eta)
            n=mesh['thickness_in']*actions._constitutive(mesh)@gradient@displacement
            traction=np.array([n[0],n[2]] if axis=='x' else [n[2],n[1]])*(b-a)/2
            r=np.asarray(point)-origin;slab+=np.r_[traction,r[0]*traction[1]-r[1]*traction[0]]
    axial={b['id']:b['N_kip'] for b in case['beam_axial']}
    for b in mesh['beams']:
        p,q=xy[b['i']],xy[b['j']]
        low,high=sorted((p[normal],q[normal]))
        crosses=(low<at<=high) if side=='minus' else (low<=at<high) if side=='plus' else (low<at<high)
        if b['axis']!=axis or not crosses:continue
        point=p+(at-p[normal])/(q[normal]-p[normal])*(q-p)
        force=np.zeros(2);force[normal]=axial[b['id']];r=point-origin
        web+=np.r_[force,r[0]*force[1]-r[1]*force[0]]
    return {'axis':axis,'coordinate_in':at,'origin_xy_in':origin.tolist(),'mesh_line_side':side,
            'slab_wrench_kip_kipin':slab.tolist(),'web_wrench_kip_kipin':web.tolist(),
            'total_wrench_kip_kipin':(slab+web).tolist(),
            'basis':'integrated physical Q4 stress trace plus disjoint axial webs; not algebraic nodal cut'}


def physical_membrane_strip_cut(result,case_id,*,axis,station_in,interval_in,trace,
                                transverse_reference_in,web_widths_in):
    """Exact declared-width planar N and transverse first moment, no smoothing.

    Return actual affine slab line fields suitable for the common-action
    adapter, plus each whole disjoint web N at the slab load plane z=0.
    This is a prescribed in-plane DESIGN FORCE state, not eccentric-web
    out-of-plane displacement compatibility. It inserts no strong-axis M.
    Raw trace/refinement qualification is explicitly left to its witnesses.
    """
    mesh=result['mesh'];normal={'x':0,'y':1}.get(axis)
    if normal is None or trace not in ('minus','plus'):raise ValueError('Explicit actual normal and one-sided physical trace required')
    at=float(station_in);lo,hi=map(float,interval_in);sref=float(transverse_reference_in)
    along=np.asarray(mesh['x_in'] if axis=='x' else mesh['y_in']);across=np.asarray(mesh['y_in'] if axis=='x' else mesh['x_in'])
    if not np.isfinite([at,lo,hi,sref]).all() or not across[0]<=lo<hi<=across[-1]:raise ValueError('Physical strip must lie within the actual slab')
    owned=int(np.searchsorted(along,at,side='left' if trace=='minus' else 'right')-1)
    if not 0<=owned<len(along)-1 or not along[owned]-1e-9<=at<=along[owned+1]+1e-9:raise ValueError('Requested physical one-sided trace is outside the slab')
    cases=[c for c in result['cases'] if c['id']==case_id]
    if len(cases)!=1:raise ValueError('Unique actual signed force case required')
    case=cases[0];xy=np.asarray(mesh['nodes']);u=np.asarray(case['displacements_in']);d=actions._constitutive(mesh)
    integrated=np.zeros(3);first=np.zeros(3);pieces=[]
    for transverse,(left,right) in enumerate(zip(across[:-1],across[1:])):
        a,b=max(lo,left),min(hi,right)
        if b<=a:continue
        ix,iy=(owned,transverse) if axis=='x' else (transverse,owned)
        cellid=iy*(len(mesh['x_in'])-1)+ix;cell=mesh['cells'][cellid];displacement=u[cell].ravel()
        def value(q):
            point=[at,q] if axis=='x' else [q,at]
            xi=2*(point[0]-mesh['x_in'][ix])/(mesh['x_in'][ix+1]-mesh['x_in'][ix])-1
            eta=2*(point[1]-mesh['y_in'][iy])/(mesh['y_in'][iy+1]-mesh['y_in'][iy])-1
            gradient,_,_=actions._q4(xy[cell],xi,eta)
            return mesh['thickness_in']*d@gradient@displacement
        na,nb=value(a),value(b);slope=(nb-na)/(b-a);mid=(a+b)/2;nm=value(mid)
        error=nm-(na+nb)/2
        if max(abs(error))>1e-10*max(1.,max(abs(na)),max(abs(nb))):raise ValueError('Actual Q4 cut is not reconstructed by its affine source field')
        force=(na+nb)*(b-a)/2;moment=mid*force+slope*(b-a)**3/12
        integrated+=force;first+=moment
        coefficients=np.zeros((2,8));coefficients[0,:3]=na;coefficients[1,:3]=slope
        samples=np.zeros((3,8));samples[:,:3]=[na,nm,nb]
        pieces.append(dict(cell=cellid,interval_in=[a,b],origin_in=a,coefficients=coefficients.tolist(),
            sample_coordinates_in=[a,mid,b],sample_tensors=samples.tolist(),integrated_N_kip=force.tolist(),
            transverse_first_moment_kipin=moment.tolist(),affine_transfer_error=error.tolist()))
    axial={r['id']:r['N_kip'] for r in case['beam_axial']};webs=[];total=integrated.copy();totalfirst=first.copy()
    for row in mesh['beams']:
        p,q=xy[row['i']],xy[row['j']]
        if row.get('axis')!=axis:continue
        low,high=sorted((p[normal],q[normal]));crosses=low<at<=high if trace=='minus' else low<=at<high
        if not crosses:continue
        tag=str(row.get('member_id',row['id']))
        if tag not in web_widths_in:raise ValueError('Every physical cut web needs its actual width, including outside-strip webs')
        width=float(web_widths_in[tag]);line=float(p[1-normal]);wa,wb=line-width/2,line+width/2
        if not math.isfinite(width) or width<=0:raise ValueError('Positive actual physical web width required')
        overlap=min(hi,wb)-max(lo,wa)
        if overlap<=1e-9:continue
        if wa<lo-1e-9 or wb>hi+1e-9:raise ValueError('Physical strip boundary cuts a web; only whole disjoint webs are supported')
        n=float(axial[row['id']]);total[normal]+=n;totalfirst[normal]+=line*n
        webs.append(dict(segment_id=row['id'],member_id=tag,interval_in=[wa,wb],reference_s_in=line,reference_z_in=0.,
            n_tension_kip=n,m_z_kip_in=0.,m_s_kip_in=0.,source_scope='prescribed planar force at slab plane; not physical eccentric strain recovery'))
    report=dict(method='exact_planar_Q4_web_declared_width_cut_v1',case_id=case_id,axis=axis,station_in=at,
        interval_in=[lo,hi],trace=trace,transverse_reference_in=sref,reference_z_in=0.,slab_fields=pieces,webs=webs,
        slab_integrated_N_kip=integrated.tolist(),slab_transverse_first_moment_kipin=first.tolist(),
        total_integrated_N_kip=total.tolist(),Ms_transverse_first_moment_kipin=float(totalfirst[normal]-sref*total[normal]),
        source_displacements_sha256=identity(case['displacements_in']),source_mesh_sha256=identity(mesh),
        adapter_context=result.get('adapter'),influences_sha256=result.get('influences_sha256'),
        physical_out_of_plane_compatibility_claimed=False,mesh_qualification_inferred=False)
    report['identity_sha256']=identity(report);return report


def rigid_diaphragm_permission(definition,*,horizontal_irregularity_checks):
    """Computed ASCE7-22 12.3.1.2 sufficient domain, never a universal waiver.

    Concrete slabs, span/depth<=3 and no horizontal Types2--5. A shortened
    span is credited only between complete straight SFRS beam lines spanning
    the whole physical grid depth. Partial core-ring sides are not promoted
    to complete support lines. Perimeter overhangs remain explicit zones.
    Irregularity rows must carry completed numerical/geometric evidence; a
    missing/unknown Type3 diaphragm-stiffness check cannot be called clear.
    """
    validate_definition(definition);g=definition['geometry'];zones=[];gaps=[]
    if definition['slab_thickness_in']<=0:gaps.append('Concrete slab absent')
    for kind in ('2','3','4','5'):
        row=horizontal_irregularity_checks.get(kind)
        if not isinstance(row,dict) or row.get('status')!='complete' or not row.get('evidence_sha256'):
            gaps.append(f'Horizontal Type{kind} classification not established')
        elif row.get('present') is not False:gaps.append(f'Horizontal Type{kind} is present or unresolved')
    for floor in definition['floors']:
        fp=from_perimeter_beams(g,floor['beams'])
        for axis in ('x','y'):
            count,length=(g['nx'],g['lx']) if axis=='x' else (g['ny'],g['ly'])
            fullspans={(i*length,(i+1)*length) for i in range(count)}
            byline={}
            for b in floor['beams']:
                if b['axis']==axis and b.get('system')=='sfrs':byline.setdefault(b['line'],set()).add((b['start'],b['end']))
            lines=sorted(line for line,spans in byline.items() if spans==fullspans)
            transverse_count,transverse_span=(g['ny'],g['ly']) if axis=='x' else (g['nx'],g['lx'])
            if not lines or lines[0]!=0 or lines[-1]!=transverse_count*transverse_span:
                gaps.append(f'Floor{floor["level"]}/{axis} lacks complete bounding SFRS lines');continue
            low,high=(fp['y_min_in'],fp['y_max_in']) if axis=='x' else (fp['x_min_in'],fp['x_max_in'])
            depth=fp['x_max_in']-fp['x_min_in'] if axis=='x' else fp['y_max_in']-fp['y_min_in']
            points=[low]+lines+[high]
            for i,(a,b) in enumerate(zip(points,points[1:])):
                if b<=a:continue
                ratio=(b-a)/depth
                zones.append({'floor':floor['level'],'loading_axis':axis,'span_from_in':a,'span_to_in':b,
                              'span_in':b-a,'depth_in':depth,'span_to_depth':ratio,'limit':3.,
                              'type':'perimeter_overhang' if i in (0,len(points)-2) else 'between_full_sfrs_lines',
                              'permitted_ratio':ratio<=3.})
                if ratio>3:gaps.append(f'Floor{floor["level"]}/{axis} span/depth {ratio:.12g} exceeds3')
    result={'basis':'ASCE7-22 12.3.1.2, printed118/PDF179; concrete slab sufficient condition',
            'definition_sha256':definition['identity_sha256'],'irregularity_evidence_sha256':identity(horizontal_irregularity_checks),
            'zones':zones,'technical_gaps':gaps,'permitted':bool(zones) and not gaps,
            'scope':'rigid diaphragm idealization permission only; in-plane force, compatibility and strength checks remain'}
    result['identity_sha256']=identity(result)
    return result
