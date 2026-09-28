"""Spatial shell/web cut diagnostics with explicit effective-flange regions.

Integrates an interpolated Gauss-resultant field, not consistent nodal forces.
It retains both recovery faces, unallocated slab regions and all six actions.
Exact partition closure does NOT establish physical equilibrium: both faces
are compared separately with the independently balanced native nodal cut.
No averaging of the faces or undocumented balancing correction is performed.
"""
from __future__ import annotations

import math
from .SMRF_Composite_Sections import recover_floor_cut, transport_wrench, _vector

METHOD_VERSION='spatial_shell_web_cut_regions_v1'
_GAUSS=((-1,-1),(1,-1),(1,1),(-1,1))
_TOL=1e-8


def _number(v,name):
    return _vector([v],1,name)[0]


def _sum(values):
    return [math.fsum(v[i] for v in values) for i in range(6)]


def effective_flange_regions(geometry, sections, slab, *, axis, slab_perimeter='centerlines'):
    """Partition the declared shell domain, exposing any flange clipping.

    Requested edge sections include the full beam width plus the interior
    overhang. If the model lacks the outer half-width, mark it incomplete;
    never silently call the clipped interval the strength section's flange.
    Regions describe a cut in one direction only, not 2D slab steel ownership.
    """
    from Structure_Parameters import effective_flange_width_in
    if axis not in ('x','y'):raise ValueError('Axis must be x or y')
    if slab_perimeter not in ('centerlines','beam_outer_faces'):raise ValueError('Unknown slab perimeter')
    transverse='y' if axis=='x' else 'x'
    count=geometry[f'num_bay_{transverse}']
    if type(count) is not int or count<1:raise ValueError('Transverse bay count must be a positive integer')
    spacing=_number(geometry[f'bay_{transverse}_in'],'bay spacing')
    span=_number(geometry[f'bay_{axis}_in'],'span')
    bw=_number(sections['b_beam_in'],'beam width')
    h=_number(sections['h_beam_in'],'beam depth')
    t=_number(slab['thickness_in'],'slab thickness')
    column=_number(sections['h_col_in' if axis=='x' else 'b_col_in'],'column depth')
    if min(bw,t,spacing-bw,span-column)<=0 or t>=h:raise ValueError('Invalid beam/flange dimensions')
    extent=count*spacing;beams=[]
    extension=bw/2 if slab_perimeter=='beam_outer_faces' else 0.
    domain=[-extension,extent+extension]
    for line in range(count+1):
        sides=1 if line in (0,count) else 2
        bf,overhang=effective_flange_width_in(bw,t,span-column,spacing-bw,sides)
        center=line*spacing
        requested=([center-bw/2,center+bw/2+overhang] if line==0 else
                   [center-bw/2-overhang,center+bw/2] if line==count else
                   [center-bw/2-overhang,center+bw/2+overhang])
        actual=[max(domain[0],requested[0]),min(domain[1],requested[1])]
        beams.append(dict(id=f'beam_{axis}_line_{line}',kind='beam_flange',line_index=line,
                          reference_transverse_in=center,requested_interval_in=requested,interval_in=actual,
                          requested_flange_width_in=bf,represented_flange_width_in=actual[1]-actual[0],
                          missing_flange_width_in=bf-(actual[1]-actual[0]),
                          strength_geometry_matches=all(abs(a-b)<_TOL for a,b in zip(requested,actual))))
    regions=[];last=domain[0]
    for beam in beams:
        lo,hi=beam['interval_in']
        if lo<last-_TOL:raise ValueError('Effective flange regions overlap; explicit shared-region treatment required')
        if lo>last+_TOL:
            regions.append(dict(id=f'slab_{axis}_gap_{len(regions)}',kind='complementary_slab',
                                interval_in=[last,lo],reference_transverse_in=(last+lo)/2))
        regions.append(beam);last=hi
    if abs(last-domain[1])>_TOL:raise ValueError('Region partition does not cover the shell domain')
    return dict(axis=axis,domain_interval_in=domain,slab_perimeter=slab_perimeter,regions=regions,
                geometry_matches_all_strength_sections=all(b['strength_geometry_matches'] for b in beams),
                reinforcement_ownership_assigned=False,
                basis='Existing effective flange helper; clipped model coverage recorded separately')


def _shell(shell):
    p=[_vector(v,3,'shell node') for v in shell['node_positions_in']]
    if len(p)!=4:raise ValueError('Four shell nodes required')
    x0,y0,z=p[0];x1,y1,z1=p[2]
    expected=[[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]]
    if x1<=x0 or y1<=y0 or any(abs(a-b)>_TOL for v,w in zip(p,expected) for a,b in zip(v,w)):
        raise ValueError('Only CCW rectangular XY shells with local normal +Z are supported')
    raw=_vector(shell['gauss_resultants_raw'],32,'Gauss resultants')
    return [x0,y0,z],[x1,y1,z], [raw[8*k:8*k+8] for k in range(4)]


def _field(raw,xi,eta):
    weights=[(1+math.sqrt(3)*sx*xi)*(1+math.sqrt(3)*sy*eta)/4 for sx,sy in _GAUSS]
    return [math.fsum(weights[k]*raw[k][j] for k in range(4)) for j in range(8)]


def _traction(raw,axis):
    # m11,m22,m12 = first moments of membrane stress about the +Z normal.
    # A positive-X cut has r_z cross F = (-m12,+m11,0), and a positive-Y
    # cut has (-m22,+m12,0). Native shell drilling is not a physical m33.
    n1,n2,n12,m1,m2,m12,q1,q2=raw
    return [n1,n12,q1,-m12,m1,0.] if axis=='x' else [n12,n2,q2,-m2,m12,0.]


def integrate_shell_interval(shells, *, axis, cut_in, interval_in, reference_in, side):
    """Integrate one physical interval using a declared one-sided field.

    At a mesh line, left selects cells on the negative-coordinate side and
    right the positive side, returning outward actions on each free body.
    At an interior cell cut, the same field is used with opposite normals.
    Linear recovery in a cell and wrench transport are integrated exactly
    with two Gauss points per intersected transverse interval.
    """
    if axis not in ('x','y') or side not in ('left','right'):raise ValueError('Explicit axis and left/right side required')
    n=0 if axis=='x' else 1;t=1-n;sign=1 if side=='left' else -1
    cut=_number(cut_in,'cut');lo,hi=_vector(interval_in,2,'interval');ref=_vector(reference_in,3,'reference')
    if hi<=lo or abs(ref[n]-cut)>_TOL:raise ValueError('Interval must be positive and reference must lie on cut plane')
    pieces=[];coverage=[];ids=[]
    for shell in shells:
        low,high,raw=_shell(shell)
        inside=low[n]+_TOL<cut<high[n]-_TOL
        face=abs((high if side=='left' else low)[n]-cut)<=_TOL
        if not (inside or face):continue
        start,end=max(lo,low[t]),min(hi,high[t])
        if end-start<=_TOL:continue
        coverage.append((start,end));ids.append(shell.get('element',shell.get('tag')))
        for g in (-1/math.sqrt(3),1/math.sqrt(3)):
            point=[0.,0.,low[2]];point[n]=cut;point[t]=(start+end)/2+g*(end-start)/2
            xi=2*(point[0]-low[0])/(high[0]-low[0])-1
            eta=2*(point[1]-low[1])/(high[1]-low[1])-1
            f=_traction(_field(raw,xi,eta),axis)
            weighted=[sign*(end-start)/2*v for v in f]
            pieces.append(transport_wrench(weighted,point,ref))
    coverage.sort();cursor=lo
    for a,b in coverage:
        if abs(a-cursor)>_TOL:raise ValueError('Gap, overlap or duplicate cell in shell cut coverage')
        cursor=b
    if abs(cursor-hi)>_TOL:raise ValueError('Incomplete shell cut coverage')
    return dict(global_wrench=_sum(pieces),reference_in=ref,interval_in=[lo,hi],side=side,
                crossed_elements=ids,interpolation='Bilinear recovery of four native Gauss blocks; no equilibration')


def recover_floor_regions(result, floor, axis, cut_in):
    """Diagnostic partition, native whole-cut comparison and geometry audit.

    Use completed finite-membrane native gravity records on the current
    uniform interior grid and declared perimeter. A native cut is required as independent
    reference, so the section must lie on a mesh line between column lines.
    """
    if (result.get('inputs',{}).get('inplane_restraint','finite_membrane')!='finite_membrane'
            or result.get('section_action_schema') not in ('native_global_actions_and_applied_loads_v1',
                                                          'native_global_actions_applied_loads_and_constraint_actions_v2')):
        raise ValueError('This recovery requires the finite-membrane native action schema')
    native=recover_floor_cut(result,floor,axis,cut_in)
    if not native['numerical_balance_passed']:raise ValueError('Native whole-floor free body does not balance')
    g,s,slab=(result['inputs'][k] for k in ('geometry','sections','slab'))
    partition=effective_flange_regions(g,s,slab,axis=axis,
                                      slab_perimeter=result['inputs'].get('slab_perimeter','centerlines'))
    n=0 if axis=='x' else 1;t=1-n;cut=float(cut_in);common=native['reference_in']
    shells=[v for v in result['shell_resultants'] if v['floor']==floor]
    webs=[v for v in result['web_segment_actions'] if v['floor']==floor and v['axis']==axis]
    rows=[];assembled={'left':[],'right':[]}
    for region in partition['regions']:
        ref=list(common);ref[t]=region['reference_transverse_in'];sides={}
        for side in ('left','right'):
            sh=integrate_shell_interval(shells,axis=axis,cut_in=cut,interval_in=region['interval_in'],reference_in=ref,side=side)
            web_actions=[];web_ids=[]
            if region['kind']=='beam_flange':
                for web in webs:
                    if web['line_index']!=region['line_index']:continue
                    positions=web['node_positions_in'];i=1 if side=='left' else 0
                    if abs(positions[i][n]-cut)>_TOL:continue
                    f=_vector(web['global_force_kip_kip_in'],12,'web forces')[6*i:6*i+6]
                    web_actions.append(transport_wrench(f,positions[i],ref));web_ids.append(web.get('tag'))
                if len(web_actions)!=1:raise ValueError('Exactly one cut-end web is required for each beam and recovery face')
            web_total=_sum(web_actions);total=_sum([sh['global_wrench'],web_total])
            sides[side]=dict(shell_wrench=sh['global_wrench'],web_wrench=web_total,total_wrench=total,
                             web_tags=web_ids,axial_tension_kip=(1 if side=='left' else -1)*total[n],
                             moment_sagging_kip_in=(1 if side=='left' else -1)*(-total[4] if axis=='x' else total[3]))
            assembled[side].append(transport_wrench(total,ref,common))
        rows.append(dict(region,reference_in=ref,sides=sides,
                         face_sum_residual=_sum([sides[k]['total_wrench'] for k in ('left','right')])))
    comparison={}
    for side in ('left','right'):
        total=_sum(assembled[side])
        full=integrate_shell_interval(shells,axis=axis,cut_in=cut,interval_in=partition['domain_interval_in'],reference_in=common,side=side)
        nc=native['sides'][side]['components']
        direct=_sum([full['global_wrench']]+[nc[k] for k in ('web_intrinsic','web_eccentricity','web_inplane_transport')])
        closure=[a-b for a,b in zip(total,direct)]
        gap=[a-b for a,b in zip(total,native['sides'][side]['total_force_moment'])]
        comparison[side]=dict(recovered_wrench=total,native_wrench=native['sides'][side]['total_force_moment'],
                              region_reassembly_residual=closure,native_equilibrium_gap=gap,
                              native_shell_cut_load_correction=nc['shell_boundary_load_correction'],
                              force_gap_over_floor_scale=max(abs(v) for v in gap[:3])/native['force_scale_kip'],
                              moment_gap_over_floor_scale=max(abs(v) for v in gap[3:])/native['moment_scale_kip_in'])
    return dict(method=METHOD_VERSION,floor=floor,axis=axis,cut_in=cut,reference_in=common,partition=partition,rows=rows,
                comparisons=comparison,whole_cut_face_sum=_sum([comparison[k]['recovered_wrench'] for k in ('left','right')]),
                native_whole_cut_balanced=True,
                numerical_partition_passed=all(max(abs(v) for v in c['region_reassembly_residual'])<1e-7 for c in comparison.values()),
                equilibrium_correction_applied=False,engineering_verified=False,production_enabled=False,
                capacity_pairing_authorized=False,
                scope='Spatial resultant integration only; inspect geometry, face imbalance, native equilibrium and mesh convergence before strength use')
