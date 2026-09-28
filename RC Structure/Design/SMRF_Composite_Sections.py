"""Equilibrium-complete cuts through a planar shell/eccentric-web assembly.

Cuts must follow a mesh line between column lines. Native element nodal
actions are transported to one explicitly stated reference using M+r×F.
Shell loads allocated to nodes on the cut are subtracted from its native
resisting actions, because those loads belong to the cut-side free body.
Beam element body loads are retained in the free-body load ledger.

This recovers whole-floor sections, not an effective flange or a qualified
individual beam design moment. It does not partition shared slab steel.
"""
from __future__ import annotations

import math

METHOD_VERSION = "native_shell_web_composite_section_v1"


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} requires {size} finite numeric components")
    return [float(v) for v in value]


def _sum(vectors):
    return [math.fsum(v[i] for v in vectors) for i in range(6)]


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def transport_wrench(force_moment, position, reference):
    f = _vector(force_moment, 6, 'force_moment')
    p, r = _vector(position, 3, 'position'), _vector(reference, 3, 'reference')
    moment = _cross([a-b for a,b in zip(p,r)], f[:3])
    return f[:3] + [f[i+3]+moment[i] for i in range(3)]


def recover_planar_cut(shells, webs, external_actions, *, axis, position_in, reference_in,
                       numerical_tolerance=1e-8):
    """Recover both outward faces and two independent free-body balances.

    Inputs are explicit full-assembly inventories. ``external_actions`` are
    forces/couples ON the floor (column reactions or benchmark boundary loads).
    Shell entries use CCW rectangular nodes and separate native/applied nodal
    actions. Web entries use global end actions and a uniform body-load resultant.
    Entire shell widths are included; a cut cannot pass through a cell, a web
    element, a parallel web or an external point load/support.
    """
    if axis not in ('x', 'y'):
        raise ValueError('Section axis must be x or y')
    normal = 0 if axis == 'x' else 1
    cut = _vector([position_in], 1, 'cut position')[0]
    reference = _vector(reference_in, 3, 'section reference')
    if abs(reference[normal] - cut) > 1e-8:
        raise ValueError('Section reference must lie on the cut plane')
    if (isinstance(numerical_tolerance, bool) or not isinstance(numerical_tolerance, (int, float))
            or not math.isfinite(numerical_tolerance) or not 0 < numerical_tolerance < 1):
        raise ValueError('Provide a finite positive numerical tolerance below one')
    if not isinstance(shells, list) or not shells:
        raise ValueError('A nonempty shell inventory is required')
    pieces = {side: {key: [] for key in ('shell_native', 'shell_boundary_load_correction',
                                        'web_intrinsic', 'web_eccentricity', 'web_inplane_transport', 'external')}
              for side in ('left', 'right')}
    cells, crossed = set(), {'left': [], 'right': []}
    selected_webs = {'left': [], 'right': []}
    all_external = []

    def side_for(coordinates):
        lo, hi = min(coordinates), max(coordinates)
        if lo < cut-1e-8 and hi > cut+1e-8:
            raise ValueError('Cut must follow element boundaries; it crosses an element interior')
        if abs(lo-cut) <= 1e-8 and abs(hi-cut) <= 1e-8:
            raise ValueError('Cut cannot contain a parallel web or an external point action')
        return 'left' if (lo+hi)/2 < cut else 'right'

    for shell in shells:
        positions = [_vector(p, 3, 'shell node position') for p in shell['node_positions_in']]
        if len(positions) != 4:
            raise ValueError('A shell requires four CCW node positions')
        x0, y0, z0 = positions[0]
        x1, y1 = positions[2][:2]
        expected = [[x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0]]
        if x1 <= x0 or y1 <= y0 or any(abs(a-b)>1e-8 for p,e in zip(positions,expected) for a,b in zip(p,e)):
            raise ValueError('Only CCW axis-aligned rectangular coplanar shell cells are supported')
        bounds = (x0, x1, y0, y1, z0)
        if bounds in cells:
            raise ValueError('Duplicate shell cell in section inventory')
        cells.add(bounds)
        forces = _vector(shell['global_nodal_force_kip_kip_in'], 24, 'shell native forces')
        loads = _vector(shell['applied_nodal_force_kip_kip_in'], 24, 'shell applied nodal loads')
        side = side_for([p[normal] for p in positions])
        for i, p in enumerate(positions):
            applied = transport_wrench(loads[6*i:6*i+6], p, reference)
            pieces[side]['external'].append(applied)
            all_external.append(applied)
            if abs(p[normal]-cut) <= 1e-8:
                pieces[side]['shell_native'].append(transport_wrench(forces[6*i:6*i+6], p, reference))
                pieces[side]['shell_boundary_load_correction'].append([-v for v in applied])
        if any(abs(p[normal]-cut)<=1e-8 for p in positions):
            tangent = 1-normal
            crossed[side].append((min(p[tangent] for p in positions), max(p[tangent] for p in positions)))
    if not crossed['left'] or sorted(crossed['left']) != sorted(crossed['right']):
        raise ValueError('Both sides of the cut must have matching, complete shell intervals')
    intervals = sorted(crossed['left'])
    for a, b in zip(intervals, intervals[1:]):
        if abs(a[1]-b[0]) > 1e-8:
            raise ValueError('Gap or overlap in cut-face shell coverage')
    for web in webs:
        positions = [_vector(p, 3, 'web node position') for p in web['node_positions_in']]
        if len(positions) != 2 or positions[0] == positions[1]:
            raise ValueError('A web requires two distinct node positions')
        forces = _vector(web['global_force_kip_kip_in'], 12, 'web native forces')
        body = _vector(web['body_load_force_kip'], 3, 'web body load')
        center = _vector(web['body_load_position_in'], 3, 'web load centroid')
        if any(abs(c-(a+b)/2)>1e-8 for c,a,b in zip(center,*positions)):
            raise ValueError('Only a uniform web body load at the element midpoint is supported')
        side = side_for([p[normal] for p in positions])
        body_wrench = transport_wrench(body+[0.,0.,0.], center, reference)
        pieces[side]['external'].append(body_wrench)
        all_external.append(body_wrench)
        for i, p in enumerate(positions):
            if abs(p[normal]-cut) <= 1e-8:
                f = forces[6*i:6*i+6]
                dz = p[2]-reference[2]
                eccentricity = [0.,0.,0.] + _cross([0.,0.,dz], f[:3])
                transverse = [0.,0.,0.] + _cross([p[0]-reference[0],p[1]-reference[1],0.], f[:3])
                pieces[side]['web_intrinsic'].append(f)
                pieces[side]['web_eccentricity'].append(eccentricity)
                pieces[side]['web_inplane_transport'].append(transverse)
                selected_webs[side].append({'tag': web.get('tag'), 'position_in': p, 'native_force_moment': f,
                                            'eccentricity_moment': eccentricity[3:]})
    for action in external_actions:
        p = _vector(action['position_in'], 3, 'external action position')
        side = side_for([p[normal]])
        f = transport_wrench(action['force_moment'], p, reference)
        pieces[side]['external'].append(f)
        all_external.append(f)
    force_scale = max(1., math.fsum(math.hypot(*w[:3]) for w in all_external))
    moment_scale = max(1., math.fsum(math.hypot(*w[3:]) for w in all_external))
    sides = {}
    for side, parts in pieces.items():
        totals = {key: _sum(values) for key, values in parts.items()}
        total = _sum([v for k,v in totals.items() if k != 'external'])
        balance = _sum([total, totals['external']])
        sides[side] = dict(components=totals, total_force_moment=total, free_body_residual=balance,
                           force_relative_error=max(abs(v) for v in balance[:3])/force_scale,
                           moment_relative_error=max(abs(v) for v in balance[3:])/moment_scale,
                           web_contributions=selected_webs[side])
    continuity = _sum([sides['left']['total_force_moment'], sides['right']['total_force_moment']])
    force_continuity = max(abs(v) for v in continuity[:3])/force_scale
    moment_continuity = max(abs(v) for v in continuity[3:])/moment_scale
    errors = [force_continuity, moment_continuity] + [sides[s][k] for s in sides for k in
                                                     ('force_relative_error','moment_relative_error')]
    return dict(method=METHOD_VERSION, axis=axis, position_in=cut, reference_in=reference,
                convention='Global Fx,Fy,Fz,Mx,My,Mz; each side acts outward on its own free body',
                units=['kip']*3+['kip-in']*3, sides=sides, shell_intervals_in=intervals,
                continuity_residual=continuity, force_continuity_relative_error=force_continuity,
                moment_continuity_relative_error=moment_continuity, force_scale_kip=force_scale,
                moment_scale_kip_in=moment_scale, numerical_tolerance=numerical_tolerance,
                numerical_balance_passed=max(errors)<=numerical_tolerance, engineering_verified=False,
                scope='Whole assembly width; not a beam effective-flange allocation or member design demand')


ACCEPTED_SCHEMAS = ('native_global_actions_and_applied_loads_v1',
                    'native_global_actions_applied_loads_and_constraint_actions_v2')


def recover_floor_cut(result, floor, axis, position_in, *, reference_in=None, numerical_tolerance=1e-8):
    """Recover one whole-floor section from a coupled native-action record.

    A record made under a declared rigid in-plane restraint carries the
    diaphragm constraint forces on its slab nodes (schema v2); they enter the
    free bodies as external actions, and a cut through a constrained node is
    rejected because a point action on the cut belongs to neither side.
    """
    if result.get('status') != 'diagnostic_complete' or result.get('section_action_schema') not in ACCEPTED_SCHEMAS:
        raise ValueError('A completed coupled solve with native section-action records is required')
    geometry, mesh = result['inputs']['geometry'], result['mesh_per_bay']
    if isinstance(floor, bool) or not isinstance(floor,int) or not 1<=floor<=geometry['num_floor']:
        raise ValueError('Floor must identify an existing positive integer story')
    if axis not in ('x','y'):
        raise ValueError('Section axis must be x or y')
    length = geometry['bay_x_in' if axis=='x' else 'bay_y_in']
    cut = _vector([position_in],1,'section position')[0]
    if not 0<cut<length*geometry['num_bay_x' if axis=='x' else 'num_bay_y'] or abs(cut/length-round(cut/length))<1e-9:
        raise ValueError('Choose an interior cut between column lines')
    shells = [s for s in result['shell_resultants'] if s['floor']==floor]
    from .SMRF_Floor_Mesh import coupled_floor_mesh
    grid = coupled_floor_mesh(geometry['num_bay_x'],geometry['num_bay_y'],geometry['bay_x_in'],geometry['bay_y_in'],mesh,
        beam_width_in=result['inputs']['sections']['b_beam_in'],
        slab_perimeter=result['inputs'].get('slab_perimeter','centerlines'), mesh_spec=result['inputs'].get('mesh_spec'))
    xs = dict(enumerate(grid['x_coordinates_in'],start=grid['index_start']))
    ys = dict(enumerate(grid['y_coordinates_in'],start=grid['index_start']))
    expected = {(i,j) for i in list(xs)[:-1] for j in list(ys)[:-1]}
    if len(shells)!=len(expected) or {(s['cell_i'],s['cell_j']) for s in shells}!=expected:
        raise ValueError('Incomplete or duplicate floor shell inventory')
    for shell in shells:
        i,j=shell['cell_i'],shell['cell_j'];z=floor*geometry['story_h_in']
        expected_nodes=[[xs[i],ys[j],z],[xs[i+1],ys[j],z],[xs[i+1],ys[j+1],z],[xs[i],ys[j+1],z]]
        actual=shell['node_positions_in']
        if len(actual)!=4 or any(max(abs(a-b) for a,b in zip(_vector(p,3,'shell position'),q))>1e-8
                                for p,q in zip(actual,expected_nodes)):
            raise ValueError('Shell coordinates do not match the declared perimeter and cell inventory')
    webs = [b for b in result['web_segment_actions'] if b['floor']==floor]
    expected_webs = {(a,line,span,k) for a,lines,spans in
                    (('x',geometry['num_bay_y']+1,geometry['num_bay_x']),
                     ('y',geometry['num_bay_x']+1,geometry['num_bay_y']))
                    for line in range(lines) for span in range(spans) for k in range(grid['subdivisions_x_per_bay' if a=='x' else 'subdivisions_y_per_bay'])}
    if len(webs)!=len(expected_webs) or {(b['axis'],b['line_index'],b['span_index'],b['segment_index']) for b in webs}!=expected_webs:
        raise ValueError('Incomplete or duplicate floor web inventory')
    external = [a for a in result['floor_boundary_actions'] if a['floor']==floor]
    expected_count = (geometry['num_bay_x']+1)*(geometry['num_bay_y']+1)*(1 if floor==geometry['num_floor'] else 2)
    if len(external)!=expected_count or len({(a['source_column'],a['end']) for a in external})!=expected_count:
        raise ValueError('Incomplete or duplicate column boundary-action inventory')
    if result['section_action_schema'] != 'native_global_actions_and_applied_loads_v1':
        assembly = result.get('assembly') or {}
        expected_tied = assembly.get('constrained_node_count_per_floor')
        tied = [a for a in result.get('diaphragm_constraint_actions', []) if a['floor']==floor]
        if (not isinstance(expected_tied,int) or isinstance(expected_tied,bool) or len(tied)!=expected_tied
                or len({a['node'] for a in tied})!=expected_tied
                or (expected_tied>0) != (assembly.get('inplane_restraint') in ('rigid_joints','rigid_floor'))):
            raise ValueError('Incomplete or duplicate diaphragm constraint-action inventory')
        external = external+[dict(a, source='diaphragm_constraint') for a in tied]
    if reference_in is None:
        reference_in = [geometry['num_bay_x']*geometry['bay_x_in']/2,
                        geometry['num_bay_y']*geometry['bay_y_in']/2, floor*geometry['story_h_in']]
        reference_in[0 if axis=='x' else 1] = cut
    answer = recover_planar_cut(shells,webs,external,axis=axis,position_in=cut,
                                reference_in=reference_in,numerical_tolerance=numerical_tolerance)
    return dict(answer,floor=floor)
