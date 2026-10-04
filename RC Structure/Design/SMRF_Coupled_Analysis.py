"""Monolithic elastic slab/web/column gravity diagnostic, NOT design approval.

Every floor has shared ShellMITC4 nodes at its slab midsurface. Downstand
beam webs (only Hbeam-hslab, not a second slab flange) are rigid-linked at
their actual centroid offset to those nodes. Full-height elastic columns
connect the floor grid intersections to a fixed base. Nothing is pinned at
an elevated column node and no in-plane shell DOFs are artificially fixed.

The auxiliary shell membrane supplies finite in-plane stiffness: this is
NOT the production rigid-diaphragm NTHA model. No seismic loads, P-delta,
creep/cracking iteration, solid joint region or independent engineering
qualification are claimed. The existing project's Iy+Iz torsion proxy is
retained explicitly for sensitivity work, not asserted as Saint-Venant J.

In-plane restraint is a declared assembly option (``inplane_restraint``):
``finite_membrane`` (the default; nothing constrained), ``rigid_joints``
(one rigidDiaphragm per floor over the column-joint slab nodes, the
production frame's assumption placed in this assembly) or ``rigid_floor``
(every slab node of the floor tied, the rigid-membrane limit). A diaphragm
constrains slab nodes that retain web nodes, a constraint chain the
Transformation handler does not support, so those variants require the
``Lagrange`` (exact) or ``Penalty`` handler; the handler is a declared
option too, so a finite-membrane run under Lagrange can be compared with
the Transformation baseline before a restrained run is read. The constraint
forces the diaphragm exerts on the slab nodes are recovered from node
equilibrium and exported as ``diaphragm_constraint_actions`` so whole-floor
free bodies still close. None of this is a design or NTHA model choice.

No files or global Structure_Parameters are changed. The caller must provide
an empty OpenSees domain. The diagnostic clears only its own scratch domain.

References: OpenSees ShellMITC4, elasticBeamColumn, rigidLink, rigidDiaphragm
and the constraint handlers:
https://opensees.berkeley.edu/OpenSees/manuals/usermanual/640.htm
https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html
https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidDiaphragm.html
https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/TransformationMethod.html
https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/LagrangeMultipliers.html
"""
from __future__ import annotations

import copy
import math

import openseespy.opensees as ops

from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Floor_Analysis import _inputs, _integer, _number, MAX_SHELLS
from Design.SMRF_Floor_Mesh import coupled_floor_mesh

METHOD_VERSION = "smrf_monolithic_eccentric_shell_web_column_gravity_v5_explicit_mesh"
# Sections of a grouped design (2026-10-02): every floor described line by line, in ascending floor order.
# The beams of floor k and the columns of story k (the supports of floor k: the column below it) are that
# floor's; one column concrete grade. {"schema", "floors": [by-line floor, ...], "fc_col_ksi", modifiers}.
BY_MEMBER_SCHEMA = "smrf_coupled_sections_by_member_v1"
SECTION_ACTION_SCHEMA = "native_global_actions_applied_loads_and_constraint_actions_v2"
INPLANE_RESTRAINTS = ("finite_membrane", "rigid_joints", "rigid_floor")
CONSTRAINT_HANDLERS = ("Transformation", "Lagrange", "Penalty")


def _rectangle(b, h, fc, modifier, nu):
    e = 57.0 * math.sqrt(fc * 1000.)
    iy, iz = b*h**3/12., h*b**3/12.
    return {"area_in2": b*h, "ec_ksi": e, "g_ksi": e/(2*(1+nu)),
            "iy_in4": modifier*iy, "iz_in4": modifier*iz, "j_in4": modifier*(iy+iz),
            "bending_torsion_modifier": modifier, "axial_modifier": 1.,
            "torsion_basis": "Inherited project polar-inertia proxy Iy+Iz; NOT verified Saint-Venant torsion."}


def _frame_element(tag, ni, nj, section, transform):
    ops.element("elasticBeamColumn", tag, ni, nj, section["area_in2"], section["ec_ksi"],
                section["g_ksi"], section["j_in4"], section["iy_in4"], section["iz_in4"], transform)


def _vector(values, length, label):
    values = list(values)
    if len(values) != length or any(not math.isfinite(v) for v in values):
        raise RuntimeError(f"Invalid {label}: expected {length} finite components.")
    return values


def _transport(force_moment, position, reference):
    """Same wrench at ``reference``: M_ref = M + (p - ref) x F."""
    f = force_moment
    r = [a-b for a, b in zip(position, reference)]
    return [f[0], f[1], f[2],
            f[3] + r[1]*f[2] - r[2]*f[1],
            f[4] + r[2]*f[0] - r[0]*f[2],
            f[5] + r[0]*f[1] - r[1]*f[0]]


def _assembly_options(inplane_restraint, constraint_handler, penalty_alpha):
    if inplane_restraint not in INPLANE_RESTRAINTS:
        raise ValueError(f"inplane_restraint must be one of {INPLANE_RESTRAINTS}.")
    if constraint_handler not in CONSTRAINT_HANDLERS:
        raise ValueError(f"constraint_handler must be one of {CONSTRAINT_HANDLERS}.")
    if inplane_restraint != "finite_membrane" and constraint_handler == "Transformation":
        raise ValueError("A rigid diaphragm constrains slab nodes that retain web nodes; the Transformation "
                         "handler does not support that constraint chain. Use Lagrange or Penalty.")
    if constraint_handler == "Penalty":
        if (isinstance(penalty_alpha, bool) or not isinstance(penalty_alpha, (int, float))
                or not math.isfinite(penalty_alpha) or penalty_alpha <= 0):
            raise ValueError("Penalty handler requires a finite positive penalty_alpha.")
        penalty_alpha = float(penalty_alpha)
    else:
        penalty_alpha = None
    return {"inplane_restraint": inplane_restraint, "constraint_handler": constraint_handler,
            "penalty_alpha": penalty_alpha}


def analyze_coupled_gravity(slab_record, geometry, sections, floor_loadcases,
                            mesh_per_bay=4, include_member_weight=True, *,
                            inplane_restraint="finite_membrane",
                            constraint_handler="Transformation", penalty_alpha=1.0e9,
                            slab_perimeter="centerlines", mesh_spec=None):
    """Solve one explicit gravity combination on all floors simultaneously.

    ``floor_loadcases`` is a list of length num_floor in ascending floor order,
    using the existing floor-analysis D/L/panel-pattern schema. Each floor's
    dead factor also scales that level's beam and underlying column weight.
    ``include_member_weight=False`` is a bounded diagnostic comparison only.
    Optional beam/column_stiffness_modifier default to 1 (gross); these scale
    bending/torsion, not axial stiffness. Maximum 8192 shells across ALL floors.

    ``inplane_restraint`` and ``constraint_handler`` are declared assembly
    options (module docstring); they are recorded in ``inputs`` and
    ``assembly``. The defaults reproduce the previous method exactly.
    ``slab_perimeter='beam_outer_faces'`` includes the slab out to the full
    perimeter beam width; added strips inherit adjacent panel pressures.
    """
    options = _assembly_options(inplane_restraint, constraint_handler, penalty_alpha)
    nf = _integer(geometry.get("num_floor"), "num_floor")
    sh = _number(geometry.get("story_h_in"), "story_h_in")
    if not isinstance(include_member_weight, bool):
        raise ValueError("include_member_weight must be Boolean.")
    if not isinstance(floor_loadcases, list) or len(floor_loadcases) != nf:
        raise ValueError("Provide one explicit load case per floor, in ascending order.")
    parsed = [_inputs(slab_record, geometry, sections, case, mesh_per_bay, allow_zero=True, mesh_spec=mesh_spec) for case in floor_loadcases]
    nx, ny, lx, ly, hs, fc, nu, mesh, _ = parsed[0]
    by_member = isinstance(sections, dict) and sections.get("schema") == BY_MEMBER_SCHEMA
    fcc = _number(sections.get("fc_col_ksi"), "fc_col_ksi")
    bm = _number(sections.get("beam_stiffness_modifier", 1.), "beam_stiffness_modifier")
    cm = _number(sections.get("column_stiffness_modifier", 1.), "column_stiffness_modifier")
    if by_member:
        floors = sections.get("floors")
        if not isinstance(floors, list) or len(floors) != nf:
            raise ValueError("By-member coupled sections need one by-line floor per elevated floor, in ascending order.")
        if slab_perimeter != "centerlines":
            raise ValueError("The perimeter-face slab extension assumes one beam width; a grouped design uses 'centerlines'.")
        for floor in floors:
            floor_sections.validate(floor, nx, ny)
            for beam in floor["beam_lines"]["x"] + floor["beam_lines"]["y"]:
                if not hs < beam["h_in"] < sh or beam["b_in"] >= min(lx, ly):
                    raise ValueError("Coupled geometry must leave a downstand web, clear column height and clear beam spans.")
            for row in floor["supports"]:
                for col in row:
                    if col["h_in"] >= lx or col["b_in"] >= ly:
                        raise ValueError("Coupled geometry must leave a downstand web, clear column height and clear beam spans.")
        b = min(beam["b_in"] for floor in floors for beam in floor["beam_lines"]["x"] + floor["beam_lines"]["y"])
        web = column = offset = None

        def web_of(k, axis, line):
            beam = floors[k-1]["beam_lines"][axis][line]
            return _rectangle(beam["b_in"], beam["h_in"]-hs, beam["fc_ksi"], bm, nu), -beam["h_in"]/2., beam

        def column_of(k, i, j):
            col = floors[k-1]["supports"][j][i]
            return _rectangle(col["b_in"], col["h_in"], fcc, cm, nu), col
    else:
        b = _number(sections.get("b_beam_in"), "b_beam_in")
        hb = _number(sections.get("h_beam_in"), "h_beam_in")
        bc = _number(sections.get("b_col_in"), "b_col_in")
        hc = _number(sections.get("h_col_in"), "h_col_in")
        fcb = _number(sections.get("fc_beam_ksi"), "fc_beam_ksi")
        if not hs < hb < sh or hc >= lx or bc >= ly or b >= min(lx, ly):
            raise ValueError("Coupled geometry must leave a downstand web, clear column height and clear beam spans.")
    grid = coupled_floor_mesh(nx, ny, lx, ly, mesh, beam_width_in=b, slab_perimeter=slab_perimeter, mesh_spec=mesh_spec)
    if nf*grid['shell_count'] > MAX_SHELLS:
        raise ValueError(f"Coupled diagnostic exceeds {MAX_SHELLS} total shells across floors, including perimeter.")
    xs = dict(enumerate(grid['x_coordinates_in'], start=grid['index_start']))
    ys = dict(enumerate(grid['y_coordinates_in'], start=grid['index_start']))
    cell_x, cell_y = list(xs)[:-1], list(ys)[:-1]
    if not by_member:
        web = _rectangle(b, hb-hs, fcb, bm, nu)
        column = _rectangle(bc, hc, fcc, cm, nu)
        # From slab mid-plane to downstand-web centroid: -(hs/2+(hb-hs)/2).
        offset = -hb/2.
    link_offset = {}
    gamma = _number(slab_record["concrete_unit_weight_kcf"], "concrete_unit_weight_kcf") / 1728.
    if ops.getNodeTags() or ops.getEleTags():
        raise RuntimeError("Coupled diagnostic requires an empty OpenSees domain; existing model was preserved.")
    mx, my = grid['subdivisions_x_per_bay'], grid['subdivisions_y_per_bay']
    ex, ey = nx*mx, ny*my
    floor_nodes, web_nodes, base_nodes = {}, {}, {}
    positions, applied_nodes, shells, beams, columns, links = {}, {}, [], [], [], []
    masters, constrained = {}, {}
    weight_ledger = {"slab_area_load_kip": 0., "beam_drop_weight_kip": 0., "column_weight_kip": 0.}
    # Global applied force and moment about (0,0,0); all gravity forces vertical.
    applied_force = [0., 0., 0.]
    applied_moment = [0., 0., 0.]
    restraint_basis = {
        "finite_membrane": "No in-plane constraint; the ElasticMembranePlateSection membrane carries in-plane action.",
        "rigid_joints": "One rigidDiaphragm per floor over the column-joint slab nodes only (ux, uy, rz tied to an "
                        "unloaded master at the plan centroid), the production frame's diaphragm placed in this assembly; "
                        "slab nodes between joints keep the finite membrane.",
        "rigid_floor": "One rigidDiaphragm per floor over every slab node (rigid-membrane limit); the slab carries "
                       "no in-plane strain."}
    result = {"method_version": METHOD_VERSION, "verified": False, "applied_to_design": False,
              "status": "not_analyzed", "inputs": {"slab": copy.deepcopy(slab_record),
              "geometry": copy.deepcopy(geometry), "sections": copy.deepcopy(sections),
              "floor_loadcases": copy.deepcopy(floor_loadcases), "include_member_weight": include_member_weight,
              "slab_perimeter": slab_perimeter, "mesh_spec": copy.deepcopy(mesh_spec), **options},
              "assembly": {**options, "basis": restraint_basis[inplane_restraint],
                           "diaphragm_masters": [], "constrained_node_count_per_floor": 0,
                           "constraint_action_convention": "Force and couple the diaphragm constraint exerts ON the "
                           "slab node, from node equilibrium (element resisting forces of the node and its rigid-linked "
                           "web node, transported, minus the applied load); zero without a diaphragm."},
              "mesh_per_bay": mesh, "mesh_metadata": grid, "web_section": web, "column_section": column,
              "slab_plan_inventory": {"area_in2_per_floor": grid['represented_area_in2'],
                  "added_area_in2_per_floor": grid['added_area_in2'],
                  "added_volume_in3_per_floor": grid['added_area_in2']*hs,
                  "added_self_weight_kip_per_floor": grid['added_area_in2']*hs*gamma,
                  "mass_assigned": False,
                  "scope": "Slab-only plan correction; no independent 3D joint-overlap volume verification"},
              "shell_response_convention": {"gauss_order": ["p11", "p22", "p12", "m11", "m22", "m12", "q1", "q2"],
                                            "axes": "local 1=X, local 2=Y, normal=+Z; CCW node ordering",
                                            "units": "membrane/shear kip/in; moments kip-in/in",
                                            "sagging_moments": "negative raw m11 and m22",
                                            "location": "four Gauss points per cell; no face extrapolation"},
              "web_centroid_offset_from_slab_midplane_in": offset,
              "limitations": ["Static linear gravity diagnostic only, not an approved SMRF design or NTHA.",
                              "Finite elastic membrane in this auxiliary model unless a rigid in-plane restraint is declared; "
                              "production rigid diaphragm unchanged.",
                              "Gross shell stiffness; optional web/column modifiers are sensitivity assumptions, not a verified composite cracking model.",
                              "Beam web excludes slab flange, but centerline joint stiffness overlaps remain unmodeled in 3D solid detail.",
                              "Member weight uses clear beam spans and slab-excluded column volumes, smeared on centerlines.",
                              "Iy+Iz torsion proxy retained; verify actual torsional stiffness, joint-face actions, cracking, creep and support-face resultants.",
                              "Web segment moments alone are NOT composite beam/slab design moments.",
                              "A declared rigid in-plane restraint is a comparison assumption, not a validated floor model."]}

    def node(x, y, z):
        tag = len(positions)+1
        positions[tag] = (x, y, z)
        ops.node(tag, x, y, z)
        return tag

    def gravity_resultant(p, x, y):
        applied_force[2] -= p
        applied_moment[0] -= p*y
        applied_moment[1] += p*x

    try:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.section("ElasticMembranePlateSection", 1, 57.*math.sqrt(fc*1000.), nu, hs, 0.)
        ops.geomTransf("Linear", 1, 1., 0., 0.)  # column
        ops.geomTransf("Linear", 2, 0., 0., 1.)  # both beam directions
        for j in range(ny+1):
            for i in range(nx+1):
                n = node(i*lx, j*ly, 0.)
                base_nodes[i, j] = n
                ops.fix(n, 1, 1, 1, 1, 1, 1)
        for k in range(1, nf+1):
            for j, y in ys.items():
                for i, x in xs.items():
                    n = node(x, y, k*sh)
                    floor_nodes[k, i, j] = n
                    if 0 <= i <= ex and 0 <= j <= ey and (i % mx == 0 or j % my == 0):
                        if by_member:
                            # One web node for each beam line through this point, at that line's own centroid
                            # depth; where an X and a Y line cross, two web nodes hang from the one slab node.
                            for axis, on_line, line in (("x", j % my == 0, j // my), ("y", i % mx == 0, i // mx)):
                                if on_line:
                                    depth = web_of(k, axis, line)[1]
                                    nw = node(x, y, k*sh+depth)
                                    web_nodes[k, i, j, axis] = nw
                                    ops.rigidLink("beam", n, nw)
                                    links.append((n, nw))
                                    link_offset[nw] = depth
                            continue
                        nw = node(x, y, k*sh+offset)
                        web_nodes[k, i, j] = nw
                        # No constraint chains: shell node is retained only;
                        # the web node is constrained only, in one rigid link.
                        ops.rigidLink("beam", n, nw)
                        links.append((n, nw))
                        link_offset[nw] = offset
        if inplane_restraint != "finite_membrane":
            for k in range(1, nf+1):
                # Unloaded bookkeeping master at the plan centroid; its out-of-plane
                # DOFs are fixed because nothing else gives them stiffness. Declaring
                # it after the links makes the chain explicit: the constrained slab
                # nodes here are the retained nodes of the web links above.
                master = node(nx*lx/2., ny*ly/2., k*sh)
                ops.fix(master, 0, 0, 1, 1, 1, 0)
                if inplane_restraint == "rigid_joints":
                    tied = [floor_nodes[k, i*mx, j*my] for j in range(ny+1) for i in range(nx+1)]
                else:
                    tied = [floor_nodes[k, i, j] for j in ys for i in xs]
                ops.rigidDiaphragm(3, master, *tied)
                masters[k] = master
                constrained[k] = tied
                result["assembly"]["diaphragm_masters"].append(
                    {"floor": k, "node": master, "position_in": positions[master],
                     "constrained_nodes": len(tied)})
            result["assembly"]["constrained_node_count_per_floor"] = len(constrained[1])
        tag = 0
        for k in range(1, nf+1):
            for j in cell_y:
                for i in cell_x:
                    tag += 1
                    nodes = [floor_nodes[k, i, j], floor_nodes[k, i+1, j],
                             floor_nodes[k, i+1, j+1], floor_nodes[k, i, j+1]]
                    ops.element("ShellMITC4", tag, *nodes, 1)
                    pi, pj = max(0, min(nx-1, i//mx)), max(0, min(ny-1, j//my))
                    pressure = parsed[k-1][-1][pi, pj] / 144.
                    p = pressure*(xs[i+1]-xs[i])*(ys[j+1]-ys[j])
                    weight_ledger["slab_area_load_kip"] += p
                    gravity_resultant(p, (xs[i+1]+xs[i])/2, (ys[j+1]+ys[j])/2)
                    for n in nodes:
                        applied_nodes[n] = applied_nodes.get(n, 0.)+p/4
                    shells.append({"tag": tag, "floor": k, "i": i, "j": j, "nodes": nodes,
                                   "panel_i": pi, "panel_j": pj, "P": p})
            for j in range(ny+1):
                for i in range(nx+1):
                    ni = base_nodes[i, j] if k == 1 else floor_nodes[k-1, i*mx, j*my]
                    nj = floor_nodes[k, i*mx, j*my]
                    tag += 1
                    _frame_element(tag, ni, nj, column_of(k, i, j)[0] if by_member else column, 1)
                    columns.append({"tag": tag, "story": k, "grid_i": i, "grid_j": j, "nodes": [ni, nj]})
            for axis, lines, intervals in (("x", ny+1, ex), ("y", nx+1, ey)):
                for line in range(lines):
                    for t in range(intervals):
                        i, j = (t, line*my) if axis == "x" else (line*mx, t)
                        if by_member:
                            ni = web_nodes[k, i, j, axis]
                            nj = web_nodes[k, i+int(axis == "x"), j+int(axis == "y"), axis]
                        else:
                            ni = web_nodes[k, i, j]
                            nj = web_nodes[k, i+int(axis == "x"), j+int(axis == "y")]
                        tag += 1
                        _frame_element(tag, ni, nj, web_of(k, axis, line)[0] if by_member else web, 2)
                        beams.append({"tag": tag, "floor": k, "axis": axis, "line_index": line,
                                      "span_index": t//(mx if axis == "x" else my), "segment_index": t % (mx if axis == "x" else my), "nodes": [ni, nj],
                                      "body_load_force_kip": [0., 0., 0.]})
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for n, p in applied_nodes.items():
            ops.load(n, 0., 0., -p, 0., 0., 0.)
        slab_force, slab_moment = applied_force.copy(), applied_moment.copy()
        if include_member_weight:
            for item in columns:
                factor = floor_loadcases[item["story"]-1]["dead_factor"]
                if by_member:
                    col = column_of(item["story"], item["grid_i"], item["grid_j"])[1]
                    p = factor*col["b_in"]*col["h_in"]*(sh-hs)*gamma
                else:
                    p = factor*bc*hc*(sh-hs)*gamma
                ops.eleLoad("-ele", item["tag"], "-type", "-beamUniform", 0., 0., -p/sh)
                weight_ledger["column_weight_kip"] += p
                gravity_resultant(p, item["grid_i"]*lx, item["grid_j"]*ly)
            for item in beams:
                axis = item["axis"]
                factor = floor_loadcases[item["floor"]-1]["dead_factor"]
                if by_member:
                    # The drop of this line's own web, between the faces of the columns below the floor at its two ends.
                    floor, line, span = floors[item["floor"]-1], item["line_index"], item["span_index"]
                    beam = floor["beam_lines"][axis][line]
                    if axis == "x":
                        length = lx
                        clear = lx-0.5*floor["supports"][line][span]["h_in"]-0.5*floor["supports"][line][span+1]["h_in"]
                    else:
                        length = ly
                        clear = ly-0.5*floor["supports"][span][line]["b_in"]-0.5*floor["supports"][span+1][line]["b_in"]
                    w = factor*beam["b_in"]*(beam["h_in"]-hs)*gamma*clear/length
                else:
                    length, clear = (lx, lx-hc) if axis == "x" else (ly, ly-bc)
                    w = factor*b*(hb-hs)*gamma*clear/length
                ops.eleLoad("-ele", item["tag"], "-type", "-beamUniform", 0., -w, 0.)
                p = w*math.dist(positions[item["nodes"][0]], positions[item["nodes"][1]])
                item["body_load_force_kip"] = [0., 0., -p]
                ni, nj = item["nodes"]
                weight_ledger["beam_drop_weight_kip"] += p
                gravity_resultant(p, (positions[ni][0]+positions[nj][0])/2., (positions[ni][1]+positions[nj][1])/2.)
        if constraint_handler == "Penalty":
            ops.constraints("Penalty", options["penalty_alpha"], options["penalty_alpha"])
        else:
            ops.constraints(constraint_handler)
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.)
        ops.analysis("Static")
        code = ops.analyze(1)
        result["analysis_return_code"] = code
        if code != 0:
            result["status"] = "analysis_failed"
            return result
        ops.reactions()
        reactions, rf, rm = [], [0., 0., 0.], [0., 0., 0.]
        for (i, j), n in base_nodes.items():
            r = _vector(ops.nodeReaction(n), 6, "base reaction")
            x, y, z = positions[n]
            for d in range(3):
                rf[d] += r[d]
            rm[0] += r[3]+y*r[2]-z*r[1]
            rm[1] += r[4]+z*r[0]-x*r[2]
            rm[2] += r[5]+x*r[1]-y*r[0]
            reactions.append({"node": n, "grid_i": i, "grid_j": j, "force_moment": r})
        force_scale = max(1., abs(applied_force[2]))
        moment_scale = force_scale*max(nx*lx, ny*ly, nf*sh)
        force_error = max(abs(a+b) for a, b in zip(rf, applied_force))/force_scale
        moment_error = max(abs(a+b) for a, b in zip(rm, applied_moment))/moment_scale
        disps = {n: _vector(ops.nodeDisp(n), 6, "node displacement") for n in positions}
        link_error = 0.
        for retained, constrained_node in links:
            u = disps[retained]
            depth = link_offset[constrained_node]
            expected = [u[0]+depth*u[4], u[1]-depth*u[3], u[2], *u[3:]]
            link_error = max(link_error, *(abs(a-b) for a, b in zip(disps[constrained_node], expected)))
        # Diaphragm kinematics: every tied node moves with its master's rigid
        # in-plane field. Exact under Lagrange; a Penalty run reports its slip.
        diaphragm_error = 0.
        for k, master in masters.items():
            um, (cx, cy, _) = disps[master], positions[master]
            for n in constrained[k]:
                x, y, _ = positions[n]
                u = disps[n]
                expected = [um[0]-um[5]*(y-cy), um[1]+um[5]*(x-cx), um[5]]
                diaphragm_error = max(diaphragm_error, abs(u[0]-expected[0]), abs(u[1]-expected[1]),
                                      abs(u[5]-expected[2]))
        # Shell -> frame actions at retained shell coordinates, all six DOFs.
        # These are an equilibrium partition at THIS coupled displacement,
        # not reusable fixed loads for a different bare frame stiffness.
        shell_forces = {n: [0.]*6 for n in floor_nodes.values()}
        panels = []
        for item in shells:
            f = _vector(ops.eleForce(item["tag"]), 24, "shell nodal force")
            for index, n in enumerate(item["nodes"]):
                for d in range(6):
                    shell_forces[n][d] += f[index*6+d]
            stresses = _vector(ops.eleResponse(item["tag"], "stresses"), 32, "shell Gauss resultants")
            panels.append({"floor": item["floor"], "panel_i": item["panel_i"],
                           "panel_j": item["panel_j"], "element": item["tag"],
                           "cell_i": item["i"], "cell_j": item["j"],
                           "bounds_xy_in": [xs[item['i']], xs[item['i']+1], ys[item['j']], ys[item['j']+1]],
                           "gauss_resultants_raw": stresses,
                           "node_positions_in": [positions[n] for n in item["nodes"]],
                           "global_nodal_force_kip_kip_in": f,
                           "applied_nodal_force_kip_kip_in": [0., 0., -item['P']/4., 0., 0., 0.]*4})
        for group in (beams, columns):
            for item in group:
                item["local_force_kip_kip_in"] = _vector(ops.eleResponse(item["tag"], "localForce"), 12, "member local force")
                item["global_force_kip_kip_in"] = _vector(ops.eleForce(item["tag"]), 12, "member global force")
                item["node_positions_in"] = [positions[n] for n in item["nodes"]]
                if group is beams:
                    item["body_load_position_in"] = [(a+b)/2. for a,b in zip(*item["node_positions_in"])]
        # Element resisting forces gathered per node, with each web node's
        # forces transported to the slab node its rigid link retains. The rigid
        # link's own force pair cancels in that sum (equal and opposite wrench),
        # so what remains against the applied load is the diaphragm force.
        element_at_node = {n: list(shell_forces[n]) for n in floor_nodes.values()}
        for col in columns:
            for end, n in enumerate(col["nodes"]):
                if n in element_at_node:
                    for d in range(6):
                        element_at_node[n][d] += col["global_force_kip_kip_in"][6*end+d]
        shell_of_web = {nw: n for n, nw in links}
        for beam in beams:
            for end, nw in enumerate(beam["nodes"]):
                n = shell_of_web[nw]
                transported = _transport(beam["global_force_kip_kip_in"][6*end:6*end+6], positions[nw], positions[n])
                for d in range(6):
                    element_at_node[n][d] += transported[d]
        constraint_actions, free_node_residual = [], 0.
        constrained_set = {n for tied in constrained.values() for n in tied}
        grid_of = {n: (i//mx, j//my) if i % mx == 0 and j % my == 0 else None
                   for (k, i, j), n in floor_nodes.items()}
        floor_of = {n: k for (k, i, j), n in floor_nodes.items()}
        for n, f in element_at_node.items():
            action = [f[0], f[1], f[2]+applied_nodes.get(n, 0.), f[3], f[4], f[5]]
            if n in constrained_set:
                grid = grid_of[n]
                constraint_actions.append({"floor": floor_of[n], "node": n, "position_in": positions[n],
                                           "grid_i": None if grid is None else grid[0],
                                           "grid_j": None if grid is None else grid[1],
                                           "force_moment": action})
            else:
                free_node_residual = max(free_node_residual, *(abs(v) for v in action))
        diaphragm_balance = []
        for k, master in masters.items():
            total = [0.]*6
            for item in constraint_actions:
                if item["floor"] == k:
                    w = _transport(item["force_moment"], item["position_in"], positions[master])
                    for d in range(6):
                        total[d] += w[d]
            out_of_plane = max(abs(item["force_moment"][d]) for item in constraint_actions
                               if item["floor"] == k for d in (2, 3, 4))
            diaphragm_balance.append({"floor": k, "sum_about_master": total,
                                      "force_relative_error": max(abs(v) for v in total[:3])/force_scale,
                                      "moment_relative_error": max(abs(v) for v in total[3:])/moment_scale,
                                      "max_out_of_plane_component": out_of_plane})
        constraint_errors = [free_node_residual/force_scale] + [
            max(item["force_relative_error"], item["moment_relative_error"],
                item["max_out_of_plane_component"]/force_scale) for item in diaphragm_balance]
        interface = []
        # Each retained slab node once (a crossing of unlike beam lines retains two web nodes).
        interface_nodes = list(dict.fromkeys(retained for retained, _ in links))
        if inplane_restraint == "rigid_floor":
            # Interior slab nodes carry in-plane diaphragm forces too; the
            # global shell-to-frame balance has to see them.
            line_nodes = set(interface_nodes)
            interface_nodes += [n for n in floor_nodes.values() if n not in line_nodes]
        for retained in interface_nodes:
            loads = [-v for v in shell_forces[retained]]
            loads[2] -= applied_nodes.get(retained, 0.)
            interface.append({"node": retained, "position_in": positions[retained], "force_moment": loads})
        interface_force, interface_moment = [0.]*3, [0.]*3
        for item in interface:
            f, (x, y, z) = item["force_moment"], item["position_in"]
            for d in range(3):
                interface_force[d] += f[d]
            interface_moment[0] += f[3]+y*f[2]-z*f[1]
            interface_moment[1] += f[4]+z*f[0]-x*f[2]
            interface_moment[2] += f[5]+x*f[1]-y*f[0]
        interface_f_error = max(abs(a-b) for a, b in zip(interface_force, slab_force))/force_scale
        interface_m_error = max(abs(a-b) for a, b in zip(interface_moment, slab_moment))/moment_scale
        floor_boundary_actions = []
        for col in columns:
            # Column actions ON the floor are opposite the column's native
            # nodal resisting forces. Include both adjacent stories separately.
            for floor, end in ((col["story"]-1, 0), (col["story"], 1)):
                if floor > 0:
                    floor_boundary_actions.append({"floor": floor, "source_column": col["tag"],
                        "end": "i" if end == 0 else "j", "position_in": col["node_positions_in"][end],
                        "force_moment": [-v for v in col["global_force_kip_kip_in"][6*end:6*end+6]]})
        floors = []
        for k in range(1, nf+1):
            nodes = [n for (kk, i, j), n in floor_nodes.items() if kk == k]
            floors.append({"floor": k, "maximum_downward_displacement_in": max(0., -min(disps[n][2] for n in nodes)),
                           "column_joint_displacements": [{"grid_i": i, "grid_j": j,
                                "displacement_rotation": disps[floor_nodes[k, i*mx, j*my]]}
                                for j in range(ny+1) for i in range(nx+1)],
                           "diaphragm_master_displacement_rotation": disps[masters[k]] if k in masters else None})
        result.update({"weight_ledger": weight_ledger, "base_reactions": reactions,
                       "equilibrium": {"applied_force_kip": applied_force, "applied_moment_kip_in": applied_moment,
                                       "base_force_kip": rf, "base_moment_kip_in": rm,
                                       "force_relative_error": force_error, "moment_relative_error": moment_error,
                                       "numerical_balance_passed": max(force_error, moment_error) < 1e-8},
                       "rigid_offset_max_residual": link_error, "diaphragm_max_residual": diaphragm_error,
                       "constraint_action_check": {"free_node_max_residual_kip": free_node_residual,
                                                   "per_floor_diaphragm_balance": diaphragm_balance,
                                                   "numerical_balance_passed": max(constraint_errors) < 1e-8},
                       "shell_count": len(shells),
                       "beam_segment_count": len(beams), "column_count": len(columns),
                       "node_count": len(positions), "floors": floors, "shell_resultants": panels,
                       "web_segment_actions": beams, "column_actions": columns,
                       "floor_boundary_actions": floor_boundary_actions,
                       "diaphragm_constraint_actions": constraint_actions,
                       "section_action_schema": SECTION_ACTION_SCHEMA,
                       "shell_to_frame_nodal_actions": interface,
                       "shell_to_frame_equilibrium": {
                           "interface_force_kip": interface_force, "interface_moment_kip_in": interface_moment,
                           "slab_applied_force_kip": slab_force, "slab_applied_moment_kip_in": slab_moment,
                           "force_relative_error": interface_f_error, "moment_relative_error": interface_m_error,
                           "numerical_balance_passed": max(interface_f_error, interface_m_error) < 1e-8,
                           "basis": "All six interface actions at retained slab-node coordinates (plus interior slab "
                                    "nodes under a rigid floor); member self-weight excluded."}})
        result["status"] = ("diagnostic_complete"
                            if max(force_error, moment_error, interface_f_error, interface_m_error, *constraint_errors) < 1e-8
                            and link_error < 1e-8 and diaphragm_error < 1e-8 else "equilibrium_failed")
        return result
    finally:
        ops.wipe()
