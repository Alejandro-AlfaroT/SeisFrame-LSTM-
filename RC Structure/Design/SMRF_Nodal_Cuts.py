"""Native nodal cut actions with an explicit continuous line projection.

Each face retains native resisting actions minus its allocated applied loads.
The continuous linear trace is the consistent-mass (L2 dual) representation
of those nodal actions. Integration conserves the full force and transported
moment exactly; local regions still require convergence and independent checks.
This is a declared recovery operator, not an empirical equilibrium correction.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_banded
from .SMRF_Composite_Sections import recover_planar_cut,transport_wrench,_vector
from .SMRF_Cut_Regions import effective_flange_regions
from .SMRF_Floor_Mesh import coupled_floor_mesh

METHOD_VERSION='consistent_nodal_line_cut_projection_v1'


def _validate_inventory(response,geometry,sections,slab_perimeter):
    """Require the complete declared shell and web geometry, including unloaded cells."""
    meta=response.get('mesh_metadata') or {}
    spec=dict(x_offsets_in=meta.get('x_offsets_in'),y_offsets_in=meta.get('y_offsets_in'),max_shells=meta.get('shell_budget'))
    nx,ny,lx,ly=(geometry[k] for k in ('num_bay_x','num_bay_y','bay_x_in','bay_y_in'))
    grid=coupled_floor_mesh(nx,ny,lx,ly,2,mesh_spec=spec,beam_width_in=sections['b_beam_in'],slab_perimeter=slab_perimeter)
    if meta.get('coordinate_sha256')!=grid['coordinate_sha256']:
        raise ValueError('Response mesh does not match the declared geometry and slab perimeter')
    xs=dict(enumerate(grid['x_coordinates_in'],start=grid['index_start']))
    ys=dict(enumerate(grid['y_coordinates_in'],start=grid['index_start']))
    expected={(i,j) for i in list(xs)[:-1] for j in list(ys)[:-1]}
    shells=response['shells']
    if len(shells)!=len(expected) or {(s['cell_i'],s['cell_j']) for s in shells}!=expected:
        raise ValueError('Incomplete or duplicate shell inventory')
    for shell in shells:
        i,j=shell['cell_i'],shell['cell_j']
        want=[[xs[i],ys[j],0.],[xs[i+1],ys[j],0.],[xs[i+1],ys[j+1],0.],[xs[i],ys[j+1],0.]]
        actual=shell['node_positions_in']
        if len(actual)!=4 or any(abs(a-b)>1e-8 for p,q in zip(actual,want) for a,b in zip(_vector(p,3,'shell node'),q)):
            raise ValueError('Shell coordinates do not match the declared complete local floor mesh')
    mx,my=grid['subdivisions_x_per_bay'],grid['subdivisions_y_per_bay']
    expected_webs={}
    for axis,lines,spans,count in (('x',ny+1,nx,mx),('y',nx+1,ny,my)):
        for line in range(lines):
            for span in range(spans):
                for segment in range(count):
                    k=span*count+segment;i,j=(k,line*my) if axis=='x' else (line*mx,k)
                    expected_webs[axis,line,span,segment]=[[xs[i],ys[j],-sections['h_beam_in']/2],
                        [xs[i+int(axis=='x')],ys[j+int(axis=='y')],-sections['h_beam_in']/2]]
    webs=response['webs'];seen=set()
    for web in webs:
        key=tuple(web[k] for k in ('axis','line_index','span_index','segment_index'))
        if key not in expected_webs or key in seen:raise ValueError('Unexpected or duplicate web inventory')
        seen.add(key);actual=web['node_positions_in']
        if len(actual)!=2 or any(abs(a-b)>1e-8 for p,q in zip(actual,expected_webs[key]) for a,b in zip(_vector(p,3,'web node'),q)):
            raise ValueError('Web coordinates do not match the declared physical offset and mesh')
    if seen!=set(expected_webs):raise ValueError('Incomplete web inventory')


def line_projection(coordinates,nodal_actions):
    """Solve integral Ni Nj * tj = nodal action i for all six components."""
    y=np.asarray(coordinates,dtype=float);f=np.asarray(nodal_actions,dtype=float)
    if y.ndim!=1 or len(y)<2 or f.shape!=(len(y),6) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(f)):
        raise ValueError('Finite ordered coordinates and six nodal action components are required')
    h=np.diff(y)
    if np.any(h<=1e-8):raise ValueError('Line coordinates must increase strictly')
    ab=np.zeros((3,len(y)));ab[1,:-1]+=h/3;ab[1,1:]+=h/3
    ab[0,1:]=h/6;ab[2,:-1]=h/6
    t=solve_banded((1,1),ab,f)
    check=ab[1,:,None]*t
    check[:-1]+=h[:,None]*t[1:]/6;check[1:]+=h[:,None]*t[:-1]/6
    relative=np.max(np.abs(check-f))/max(1.,np.max(np.abs(f)))
    if relative>1e-12:raise RuntimeError('Line projection failed its nodal virtual-work check')
    return dict(coordinates_in=y.tolist(),wrench_density=t.tolist(),
                nodal_actions=f.tolist(),projection_relative_residual=float(relative))


def integrate_line(trace,*,axis,cut_in,z_in,interval_in,reference_in):
    if axis not in ('x','y'):raise ValueError('Axis must be x or y')
    ref=_vector(reference_in,3,'reference');lo,hi=_vector(interval_in,2,'interval')
    ys=trace['coordinates_in'];values=trace['wrench_density'];n=0 if axis=='x' else 1;t=1-n
    if hi<=lo or lo<ys[0]-1e-8 or hi>ys[-1]+1e-8:raise ValueError('Interval must lie inside the complete cut')
    if abs(ref[n]-cut_in)>1e-8:raise ValueError('Reference must lie on the cut')
    result=[]
    for i,(a,b) in enumerate(zip(ys,ys[1:])):
        start,end=max(a,lo),min(b,hi)
        if end<=start:continue
        for g in (-1/math.sqrt(3),1/math.sqrt(3)):
            y=(start+end)/2+g*(end-start)/2;alpha=(y-a)/(b-a)
            f=[((1-alpha)*u+alpha*v)*(end-start)/2 for u,v in zip(values[i],values[i+1])]
            p=[0.,0.,z_in];p[n]=cut_in;p[t]=y
            result.append(transport_wrench(f,p,ref))
    return [math.fsum(f[k] for f in result) for k in range(6)]


def recover_shell_trace(shells,*,axis,cut_in,side):
    if axis not in ('x','y') or side not in ('left','right'):raise ValueError('Explicit axis and face required')
    n=0 if axis=='x' else 1;t=1-n;nodes={};z=None
    for shell in shells:
        p=shell['node_positions_in'];low=min(v[n] for v in p);high=max(v[n] for v in p)
        owns=abs((high if side=='left' else low)-cut_in)<1e-8
        if not owns:continue
        f=_vector(shell['global_nodal_force_kip_kip_in'],24,'shell resisting actions')
        loads=_vector(shell['applied_nodal_force_kip_kip_in'],24,'shell applied actions')
        for i,point in enumerate(p):
            if abs(point[n]-cut_in)>1e-8:continue
            if z is not None and abs(z-point[2])>1e-8:raise ValueError('Shell cut is not coplanar')
            z=point[2]
            nodes.setdefault(point[t],np.zeros(6))
            nodes[point[t]]+=np.array(f[6*i:6*i+6])-np.array(loads[6*i:6*i+6])
    if len(nodes)<2:raise ValueError('Cut must lie on a shell mesh boundary with both faces present')
    coordinates=sorted(nodes)
    trace=line_projection(coordinates,[nodes[y] for y in coordinates])
    return dict(trace,z_in=z,side=side)


def recover_nodal_regions(response,geometry,sections,slab,*,axis,cut_in,slab_perimeter):
    """Recover one local-z floor response, with its actual boundary reactions."""
    _validate_inventory(response,geometry,sections,slab_perimeter)
    shells,webs=response['shells'],response['webs']
    keys=[(i,j) for j in range(geometry['num_bay_y']+1) for i in range(geometry['num_bay_x']+1)]
    f=_vector(response['boundary_force'],6*len(keys),'complete floor boundary actions')
    z=shells[0]['node_positions_in'][0][2]
    external=[dict(position_in=[i*geometry['bay_x_in'],j*geometry['bay_y_in'],z],force_moment=f[6*k:6*k+6])
              for k,(i,j) in enumerate(keys)]
    n=0 if axis=='x' else 1;t=1-n
    common=[geometry['num_bay_x']*geometry['bay_x_in']/2,geometry['num_bay_y']*geometry['bay_y_in']/2,z];common[n]=cut_in
    native=recover_planar_cut(shells,webs,external,axis=axis,position_in=cut_in,reference_in=common)
    if not native['numerical_balance_passed']:raise ValueError('Native cut does not satisfy the floor free body')
    partition=effective_flange_regions(geometry,sections,slab,axis=axis,slab_perimeter=slab_perimeter)
    traces={side:recover_shell_trace(shells,axis=axis,cut_in=cut_in,side=side) for side in ('left','right')}
    for trace in traces.values():
        if any(abs(a-b)>1e-8 for a,b in zip([trace['coordinates_in'][0],trace['coordinates_in'][-1]],partition['domain_interval_in'])):
            raise ValueError('Recovered trace does not cover the declared floor domain')
    totals={side:np.zeros(6) for side in traces};rows=[]
    for region in partition['regions']:
        ref=list(common);ref[t]=region['reference_transverse_in'];sides={}
        for side,trace in traces.items():
            shell=integrate_line(trace,axis=axis,cut_in=cut_in,z_in=z,interval_in=region['interval_in'],reference_in=ref)
            web=np.zeros(6);count=0
            if region['kind']=='beam_flange':
                for item in webs:
                    if item['axis']!=axis or item['line_index']!=region['line_index']:continue
                    k=1 if side=='left' else 0
                    if abs(item['node_positions_in'][k][n]-cut_in)>1e-8:continue
                    web+=transport_wrench(item['global_force_kip_kip_in'][6*k:6*k+6],item['node_positions_in'][k],ref);count+=1
                if count!=1:raise ValueError('Exactly one web cut end is required')
            total=np.array(shell)+web
            totals[side]+=transport_wrench(total.tolist(),ref,common)
            sign=1 if side=='left' else -1
            sides[side]=dict(shell_wrench=shell,web_wrench=web.tolist(),total_wrench=total.tolist(),
                axial_tension_kip=float(sign*total[n]),moment_sagging_kip_in=float(sign*(-total[4] if axis=='x' else total[3])))
        rows.append(dict(region,reference_in=ref,sides=sides,
            face_sum_residual=(np.array(sides['left']['total_wrench'])+sides['right']['total_wrench']).tolist()))
    gaps={side:(total-native['sides'][side]['total_force_moment']).tolist() for side,total in totals.items()}
    return dict(method=METHOD_VERSION,axis=axis,cut_in=cut_in,partition=partition,rows=rows,
                native_reassembly_residual=gaps,native_whole_cut=native,
                native_reassembly_passed=max(abs(v) for gap in gaps.values() for v in gap)<1e-7,
                field_basis='Consistent linear line projection of native nodal cut actions after allocated pressure loads',
                section_reference_depth_from_slab_top_in=slab['thickness_in']/2,
                action_convention='Global Fx,Fy,Fz,Mx,My,Mz at each region reference; outward on each face. N tension positive; flexural M sagging positive.',
                capacity_pairing_authorized=False,engineering_verified=False)
