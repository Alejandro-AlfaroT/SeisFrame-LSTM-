"""Independent elastic two-way floor diagnostic, NOT qualified slab demands.

ShellMITC4 panels share nodes across every bay. Two support models exist:

``rigid_lines`` (diagnostic): vertical supports lie on every beam centerline;
their rotations remain free and no elastic supporting beams are included.
This idealization is NOT a conservative bound for all slab or frame
quantities. It diagnoses continuity and load paths only.

``flexible_beams`` (load transfer): elastic beam segments run along every
beam centerline on the slab mid-plane and share the shell nodes; only column
intersections are held vertically. Each beam carries the gross ACI 318-19
8.4.1.8 T/L-section inertia that the slab thickness screen uses for alpha_f
(SMRF_Slab._beam_inertia), less the flange's own mid-plane term, which the
shells already provide. Gross stiffness on both slab and beam is the same
basis ACI uses to decide how much of a two-way slab the beams support. The
slab-to-beam load path is read from the beam segment shears, so each frame
beam receives the discrete node loads the slab actually delivers to it and
each column the part that bypasses the beams. Beam moments from this model
are informational only (columns are rigid here); the frame model computes
the design moments from the transferred loads.

Neither model modifies the production rigid-diaphragm/NTHA model. Both
require an empty OpenSees domain.

Primary implementation references (accessed 2026-09-12):
https://opensees.berkeley.edu/OpenSees/manuals/usermanual/640.htm
https://opensees.berkeley.edu/OpenSees/manuals/usermanual/231.htm
https://github.com/OpenSees/OpenSees/blob/master/SRC/element/shell/ShellMITC4.cpp
https://github.com/OpenSees/OpenSees/blob/master/SRC/material/section/ElasticMembranePlateSection.cpp

The element returns four Gauss-point blocks ordered p11,p22,p12,m11,m22,m12,
q1,q2. Moments are section resultants (kip-in per inch), NOT moments in kip-in.
For the imposed CCW XY mesh, local 1=X, 2=Y, and positive normal=+Z. Sagging
moments here equal negative raw m11/m22 (confirmed against a simply supported
downward-loaded plate). Twisting resultants are saved but not converted into
reinforcement design moments: top/bottom envelopes alone are not design loads.
"""
from __future__ import annotations

import copy
import math

import openseespy.opensees as ops
from Design import SMRF_Floor_Sections as floor_sections
from Design.SMRF_Floor_Mesh import floor_mesh, span_offsets


METHOD_VERSION = "shellmitc4_rigid_beam_line_floor_diagnostic_v1"
METHOD_VERSION_FLEXIBLE = "shellmitc4_flexible_beam_line_floor_transfer_v2_equilibrium_audit"
SUPPORT_MODELS = ("rigid_lines", "flexible_beams")
MAX_SHELLS = 8192


def transfer_mesh_per_bay(num_bay_x, num_bay_y, preferred=16, minimum=8):
    """Finest even subdivision, from ``preferred`` down, within MAX_SHELLS.

    Point supports leak slab load straight into the column node at coarse
    meshes (60% at 4/bay, 3% at 16/bay on a 20x17 ft bay), so the transfer
    model wants the finest mesh the shell cap allows.
    """
    for mesh in range(preferred, minimum - 1, -2):
        if num_bay_x * num_bay_y * mesh * mesh <= MAX_SHELLS:
            return mesh
    raise ValueError(f"No transfer mesh of at least {minimum}/bay fits {MAX_SHELLS} shells for this floor.")
SOURCE_URLS = [
    "https://opensees.berkeley.edu/OpenSees/manuals/usermanual/640.htm",
    "https://opensees.berkeley.edu/OpenSees/manuals/usermanual/231.htm",
    "https://github.com/OpenSees/OpenSees/blob/master/SRC/element/shell/ShellMITC4.cpp",
    "https://github.com/OpenSees/OpenSees/blob/master/SRC/material/section/ElasticMembranePlateSection.cpp",
]


def _number(value, name, *, zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return result


def _integer(value, name):
    result = _number(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(result)


def _inputs(slab_record, geometry, sections, loadcase, mesh_per_bay, *, allow_zero=False, mesh_spec=None):
    if not all(isinstance(x, dict) for x in (slab_record, geometry, sections, loadcase)):
        raise ValueError("Floor inputs must be dictionaries.")
    try:
        nx = _integer(geometry["num_bay_x"], "num_bay_x")
        ny = _integer(geometry["num_bay_y"], "num_bay_y")
        lx = _number(geometry["bay_x_in"], "bay_x_in")
        ly = _number(geometry["bay_y_in"], "bay_y_in")
        h = _number(slab_record["thickness_in"], "thickness_in")
        fc = _number(slab_record["concrete_fc_ksi"], "concrete_fc_ksi")
        gamma = _number(slab_record["concrete_unit_weight_kcf"], "concrete_unit_weight_kcf")
        sdl = _number(slab_record["superimposed_dead_load_ksf"], "superimposed_dead_load_ksf", zero=True)
        case_id = loadcase["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("loadcase.id must be a nonempty string.")
        dead_factor = _number(loadcase["dead_factor"], "dead_factor", zero=True)
        live_factor = _number(loadcase["live_factor"], "live_factor", zero=True)
        live_load = _number(loadcase["live_load_ksf"], "live_load_ksf", zero=True)
        pattern = loadcase["live_pattern"]
    except KeyError as exc:
        raise ValueError(f"Missing floor-analysis input: {exc.args[0]}.") from exc
    mesh = _integer(mesh_per_bay, "mesh_per_bay")
    if mesh_spec is None and (not 2 <= mesh <= 24 or nx * ny * mesh * mesh > MAX_SHELLS):
        raise ValueError(f"Floor analysis requires 2-24 subdivisions per bay and at most {MAX_SHELLS} shells.")
    nu = _number(sections.get("slab_poisson_ratio", 0.2), "slab_poisson_ratio", zero=True)
    if nu >= 0.5 or h >= min(lx, ly):
        raise ValueError("Require 0 <= slab Poisson ratio < .5 and slab thickness < either bay.")
    if isinstance(pattern, str):
        if pattern not in {"all", "none"}:
            raise ValueError("live_pattern must be 'all', 'none', or explicit zero-based panel indices.")
        active = {(i, j) for j in range(ny) for i in range(nx)} if pattern == "all" else set()
    elif isinstance(pattern, (list, tuple)):
        active = set()
        for panel in pattern:
            if (not isinstance(panel, (list, tuple)) or len(panel) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in panel)):
                raise ValueError("Each live-pattern panel must be [zero-based integer i, j].")
            i, j = panel
            if not 0 <= i < nx or not 0 <= j < ny or (i, j) in active:
                raise ValueError("Live-pattern panels must be unique and inside the floor grid.")
            active.add((i, j))
    else:
        raise ValueError("live_pattern must explicitly identify loaded panels.")
    dead = gamma * h / 12.0 + sdl
    pressures = {(i, j): dead_factor * dead + (live_factor * live_load if (i, j) in active else 0.0)
                 for j in range(ny) for i in range(nx)}
    if max(pressures.values()) <= 0 and not allow_zero:
        raise ValueError("Floor load case must apply nonzero downward pressure.")
    return nx, ny, lx, ly, h, fc, nu, mesh, pressures


def _beam_inputs(sections, support_model, slab_h, lx, ly):
    """Frame beam sections for the flexible support model; None for rigid lines.

    Bending inertia per line position follows SMRF_Slab._beam_inertia (ACI
    318-19 8.4.1.8 gross T/L-section: one flange on perimeter lines, two on
    interior lines, projection capped by the transverse bay). The flange's own
    mid-plane term b_f*h^3/12 is removed because the shell mesh already
    bends that slab strip. ``beam_stiffness_modifier`` (default 1.0, gross)
    and ``beam_torsion_modifier`` (default 1.0 on the polar approximation
    the frame builders use) scale those values.
    """
    if support_model not in SUPPORT_MODELS:
        raise ValueError(f"support_model must be one of {SUPPORT_MODELS}.")
    if support_model == "rigid_lines":
        return None
    from Design.SMRF_Slab import _beam_inertia
    if floor_sections.is_by_line(sections):
        return _beam_inputs_by_line(sections, slab_h, lx, ly)
    try:
        b = _number(sections["b_beam_in"], "b_beam_in")
        h = _number(sections["h_beam_in"], "h_beam_in")
        fc = _number(sections["fc_beam_ksi"], "fc_beam_ksi")
    except KeyError as exc:
        raise ValueError(f"Flexible beam support requires sections.{exc.args[0]}.") from exc
    if h <= slab_h:
        raise ValueError("Flexible beam support requires a beam deeper than the slab.")
    modifier = _number(sections.get("beam_stiffness_modifier", 1.0), "beam_stiffness_modifier")
    torsion = _number(sections.get("beam_torsion_modifier", 1.0), "beam_torsion_modifier")
    footprint = None
    if "b_col_in" in sections or "h_col_in" in sections:
        footprint = (_number(sections["h_col_in"], "h_col_in") * _number(sections["b_col_in"], "b_col_in"))
    ec = 57.0 * math.sqrt(fc * 1000.0)
    inertia = {}
    for axis, transverse in (("x", ly), ("y", lx)):
        for position, flanges in (("edge", 1), ("interior", 2)):
            gross, projection, flange_width = _beam_inertia(b, h, slab_h, flanges, transverse)
            inertia[f"{axis}_{position}"] = {
                "t_section_gross_in4": gross, "flange_width_in": flange_width,
                "flange_projection_in": projection,
                "flange_midplane_term_in4": flange_width * slab_h**3 / 12.0,
                "iy_in4": modifier * (gross - flange_width * slab_h**3 / 12.0),
            }
    return {"b_in": b, "h_in": h, "fc_ksi": fc, "ec_ksi": ec, "g_ksi": 0.4 * ec,
            "area_in2": b * h, "iz_in4": modifier * h * b**3 / 12.0,
            "j_in4": torsion * (b * h**3 + h * b**3) / 12.0,
            "stiffness_modifier": modifier, "torsion_modifier": torsion,
            "line_inertia": inertia, "column_footprint_in2": footprint,
            "basis": ("ACI 318-19 8.4.1.8 gross T/L-section per beam line (SMRF_Slab._beam_inertia), "
                      "less the flange mid-plane term carried by the shells; centroidal element on the "
                      "slab mid-plane, no beam/slab centroid offset.")}


def _beam_inputs_by_line(sections, slab_h, lx, ly):
    """The same beam model for a floor described line by line: every line its own section and stiffness."""
    from Design.SMRF_Slab import _beam_inertia
    nx, ny = len(sections["beam_lines"]["y"]) - 1, len(sections["beam_lines"]["x"]) - 1
    floor_sections.validate(sections, nx, ny)
    modifier = _number(sections.get("beam_stiffness_modifier", 1.0), "beam_stiffness_modifier")
    torsion = _number(sections.get("beam_torsion_modifier", 1.0), "beam_torsion_modifier")

    def line(beam, axis, index, last, transverse):
        b, h, fc = (_number(beam[key], key) for key in ("b_in", "h_in", "fc_ksi"))
        if h <= slab_h:
            raise ValueError("Flexible beam support requires a beam deeper than the slab.")
        position = "edge" if index in (0, last) else "interior"
        gross, projection, flange_width = _beam_inertia(b, h, slab_h, 1 if position == "edge" else 2, transverse)
        ec = 57.0 * math.sqrt(fc * 1000.0)
        return {"axis": axis, "line_index": index, "position": position, "b_in": b, "h_in": h, "fc_ksi": fc,
                "ec_ksi": ec, "g_ksi": 0.4 * ec, "area_in2": b * h, "iz_in4": modifier * h * b**3 / 12.0,
                "j_in4": torsion * (b * h**3 + h * b**3) / 12.0,
                "t_section_gross_in4": gross, "flange_width_in": flange_width, "flange_projection_in": projection,
                "flange_midplane_term_in4": flange_width * slab_h**3 / 12.0,
                "iy_in4": modifier * (gross - flange_width * slab_h**3 / 12.0)}

    lines = {"x": [line(beam, "x", j, ny, ly) for j, beam in enumerate(sections["beam_lines"]["x"])],
             "y": [line(beam, "y", i, nx, lx) for i, beam in enumerate(sections["beam_lines"]["y"])]}
    footprints = [[_number(column["h_in"], "h_in") * _number(column["b_in"], "b_in") for column in row]
                  for row in sections["supports"]]
    return {"by_line": True, "lines": lines, "stiffness_modifier": modifier, "torsion_modifier": torsion,
            "column_footprint_in2": footprints, "floor_sections_sha256": floor_sections.signature(sections),
            "basis": ("ACI 318-19 8.4.1.8 gross T/L-section of every beam line on its own web (SMRF_Slab._beam_inertia), "
                      "less the flange mid-plane term carried by the shells; centroidal element on the "
                      "slab mid-plane, no beam/slab centroid offset.")}


def _line_properties(beam, axis, line, last):
    """(A, E, G, J, Iy, Iz) of the beam segments on one line."""
    if beam.get("by_line"):
        p = beam["lines"][axis][line]
        return p["area_in2"], p["ec_ksi"], p["g_ksi"], p["j_in4"], p["iy_in4"], p["iz_in4"]
    iy = beam["line_inertia"][f"{axis}_edge" if line in (0, last) else f"{axis}_interior"]["iy_in4"]
    return beam["area_in2"], beam["ec_ksi"], beam["g_ksi"], beam["j_in4"], iy, beam["iz_in4"]


def _constrain_plate_node(node, held):
    """Install the existing zero plate constraints in the active constant pattern.

    The fresh floor builder visits each node once. Using pattern SPs avoids
    the costly domain-level fix insertion path on large meshes. These are
    exactly the same homogeneous restraints; rotations 4/5 stay free.
    """
    for dof in (1, 2, 6):
        ops.sp(node, dof, 0.0)
    if held:
        ops.sp(node, 3, 0.0)


def _extrema(values):
    """Keep extrema with their original element/Gauss-point locations."""
    return {"minimum": min(values, key=lambda item: item["value"]),
            "maximum": max(values, key=lambda item: item["value"])}


def analyze_floor(slab_record, geometry, sections, loadcase, mesh_per_bay=4,
                  support_model="rigid_lines", *, mesh_spec=None):
    """Analyze one common floor using explicit D/L factors and live pattern.

    Load case keys: ``id``, ``dead_factor``, ``live_factor``, ``live_load_ksf``,
    ``live_pattern``. Pattern is 'all', 'none', or a list of zero-based [i,j]
    panels. Only slab concrete + SDL + specified live pressure is applied;
    beam/column weight belongs to the frame and is not added to slab pressure.

    ``support_model`` selects rigid beam-line supports (diagnostic) or
    flexible frame beams (slab-to-frame transfer); see the module docstring.

    No files are written. An existing OpenSees model is refused, not erased.
    Numerical convergence/equilibrium does not set ``verified`` to True.
    """
    nx, ny, lx, ly, h, fc, nu, mesh, pressures = _inputs(
        slab_record, geometry, sections, loadcase, mesh_per_bay, mesh_spec=mesh_spec)
    grid = floor_mesh(nx, ny, lx, ly, mesh, mesh_spec=mesh_spec, uniform_shell_limit=MAX_SHELLS)
    xs, ys = grid["x_coordinates_in"], grid["y_coordinates_in"]
    mx, my = grid["subdivisions_x_per_bay"], grid["subdivisions_y_per_bay"]
    beam = _beam_inputs(sections, support_model, h, lx, ly)
    flexible = beam is not None
    if ops.getNodeTags() or ops.getEleTags():
        raise RuntimeError("Floor diagnostic requires an empty OpenSees domain; preserve/extract the frame first.")
    ex, ey = len(xs) - 1, len(ys) - 1
    ec = 57.0 * math.sqrt(fc * 1000.0)
    nodal_loads = {}
    element_grid = []
    tag = lambda i, j: 1 + i + (ex + 1) * j
    node_positions = {}
    supported = {}
    result = {
        "method_version": METHOD_VERSION_FLEXIBLE if flexible else METHOD_VERSION,
        "verified": False,
        "status": "not_analyzed", "loadcase": copy.deepcopy(loadcase),
        "support_model": ("elastic_frame_beams_on_slab_midplane_vertical_fixed_column_intersections"
                          if flexible else
                          "vertical_fixed_beam_centerlines_free_rotations_no_beam_flexibility"),
        "beam_model": copy.deepcopy(beam),
        "inputs": {"slab": copy.deepcopy(slab_record), "geometry": copy.deepcopy(geometry),
                   "sections": copy.deepcopy(sections)},
        "mesh": grid,
        "constraint_storage": "homogeneous_SPs_in_separate_constant_pattern",
        "material": {"ec_ksi": ec, "poisson_ratio": nu, "thickness_in": h,
                     "basis": "linear elastic gross isotropic concrete, no cracking or creep"},
        "response_convention": {
            "raw_gauss_order": ["p11", "p22", "p12", "m11", "m22", "m12", "q1", "q2"],
            "local_axes": "1=global X, 2=global Y, normal=global +Z; nodes CCW",
            "sagging_mx_my": "negative raw m11/m22; positive values indicate bottom-face tension",
            "moment_units": "kip-in/in", "shear_units": "kip/in",
            "evaluation_location": "Gauss points; no face/corner extrapolation or strip averaging",
        },
        "demand_evidence": {"verified": False, "engineering_support_model_verified": False,
                            "all_live_patterns_enveloped": False, "mesh_refinement_verified": False,
                            "twisting_design_processed": False, "reinforcement_design_ready": False},
        "limitations": ([
            "Frame beams are centroidal ACI 8.4.1.8 gross T/L-sections on the slab mid-plane; the beam/slab centroid offset and in-plane (membrane) composite action are omitted.",
            "Column intersections are held vertically; column shortening and joint rotation belong to the frame model, so beam moments here are informational only.",
        ] if flexible else [
            "Beam-centerline vertical supports are rigid; actual beam bending/torsion, column deflection and joint stiffness are omitted.",
            "This support idealization is not a conservative bound for all quantities and is not approved design demand evidence.",
        ]) + [
            "One explicit load pattern is analyzed, not a complete load-combination/pattern envelope.",
            "Raw twisting moments are retained but no Wood-Armer or other torsion-to-reinforcement design transformation is applied.",
            "Gauss-point envelopes exclude support-face extrapolation, design-strip integration, cracking, creep and long-term deflection.",
        ] + ([] if flexible else [
            "Beam-line reactions exclude line-intersection reactions, which cannot be assigned to a unique beam by this model.",
        ]) + [
            "No slab openings, drops, cantilevers, lateral inertial forces or membrane design are included.",
        ],
        "sources": SOURCE_URLS,
    }
    try:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.section("ElasticMembranePlateSection", 1, ec, nu, h, 0.0)
        # Separate from pressure pattern 1; zero constraints persist at every load factor.
        ops.timeSeries("Constant", 2)
        ops.pattern("Plain", 2, 2)
        for j in range(ey + 1):
            for i in range(ex + 1):
                node = tag(i, j)
                node_positions[node] = (xs[i], ys[j])
                ops.node(node, xs[i], ys[j], 0.0)
                intersection = i % mx == 0 and j % my == 0
                line_support = i % mx == 0 or j % my == 0
                held = intersection if flexible else line_support
                # In-plane DOFs and drilling are suppressed in this pure
                # linear plate problem; bending rotations remain free.
                _constrain_plate_node(node, held)
                if held:
                    supported[node] = (i, j)
        for j in range(ey):
            for i in range(ex):
                element = 1 + i + ex * j
                nodes = [tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1)]
                ops.element("ShellMITC4", element, *nodes, 1)
                panel = (i // mx, j // my)
                dx, dy = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
                pressure = pressures[panel] / 144.0
                for node in nodes:
                    nodal_loads[node] = nodal_loads.get(node, 0.0) + pressure * dx * dy / 4.0
                element_grid.append((element, i, j, panel))
        beam_segments = []
        if flexible:
            # Both transforms use vecxz=(0,0,1): local z is global Z, so the
            # vertical shear is localForce[2]/[8] and vertical bending is
            # localForce[4]/[10], exactly as in the frame builders.
            ops.geomTransf("Linear", 1, 0, 0, 1)
            segment = ex * ey
            for line in range(ny + 1):
                row = line * my
                area, e_mod, g_mod, j_tor, iy, iz = _line_properties(beam, "x", line, ny)
                for i in range(ex):
                    segment += 1
                    ops.element("elasticBeamColumn", segment, tag(i, row), tag(i + 1, row),
                                area, e_mod, g_mod, j_tor, iy, iz, 1)
                    beam_segments.append((segment, "x", line, i // mx, i % mx, tag(i, row), tag(i + 1, row)))
            for line in range(nx + 1):
                col = line * mx
                area, e_mod, g_mod, j_tor, iy, iz = _line_properties(beam, "y", line, nx)
                for j in range(ey):
                    segment += 1
                    ops.element("elasticBeamColumn", segment, tag(col, j), tag(col, j + 1),
                                area, e_mod, g_mod, j_tor, iy, iz, 1)
                    beam_segments.append((segment, "y", line, j // my, j % my, tag(col, j), tag(col, j + 1)))
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for node, load in nodal_loads.items():
            ops.load(node, 0, 0, -load, 0, 0, 0)
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system(grid["solver"])
        ops.test("NormDispIncr", 1e-12, 20)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        code = ops.analyze(1)
        result["analysis_return_code"] = code
        if code != 0:
            result["status"] = "analysis_failed"
            return result
        ops.reactions()
        reactions = []
        for node, (i, j) in supported.items():
            rz = float(ops.nodeReaction(node, 3))
            if not math.isfinite(rz):
                raise RuntimeError("Nonfinite floor support reaction.")
            reactions.append({"node": node, "x_in": xs[i], "y_in": ys[j],
                              "upward_reaction_kip": rz,
                              "support_kind": "intersection" if i % mx == 0 and j % my == 0 else "beam_line"})
        applied = sum(nodal_loads.values())
        reaction_sum = sum(r["upward_reaction_kip"] for r in reactions)
        applied_x = sum(p * node_positions[n][0] for n, p in nodal_loads.items())
        applied_y = sum(p * node_positions[n][1] for n, p in nodal_loads.items())
        reaction_x = sum(r["upward_reaction_kip"] * r["x_in"] for r in reactions)
        reaction_y = sum(r["upward_reaction_kip"] * r["y_in"] for r in reactions)
        relative_errors = [abs(reaction_sum - applied) / applied,
                           abs(reaction_x - applied_x) / max(abs(applied_x), applied * lx),
                           abs(reaction_y - applied_y) / max(abs(applied_y), applied * ly)]
        result["equilibrium"] = {
            "applied_downward_kip": applied, "support_upward_kip": reaction_sum,
            "vertical_relative_error": relative_errors[0],
            "x_first_moment_relative_error": relative_errors[1],
            "y_first_moment_relative_error": relative_errors[2],
            "numerical_balance_passed": max(relative_errors) < 1e-8,
        }
        result["support_node_reactions"] = reactions
        result["intersection_reactions"] = [r for r in reactions if r["support_kind"] == "intersection"]
        reaction_by_node = {r["node"]: r["upward_reaction_kip"] for r in reactions}
        if flexible:
            _extract_beam_transfer(result, beam_segments, reaction_by_node, mesh,
                                   nx, ny, lx, ly, applied, pressures, beam["column_footprint_in2"],
                                   grid=grid)
        else:
            line_reactions = []
            for axis in ("x", "y"):
                lines, spans = (ny + 1, nx) if axis == "x" else (nx + 1, ny)
                for line in range(lines):
                    for span in range(spans):
                        nodes = [tag(span * mx + k, line * my) if axis == "x"
                                 else tag(line * mx, span * my + k) for k in range(1, mx if axis == "x" else my)]
                        line_reactions.append({"axis": axis, "line_index": line, "span_index": span,
                                               "node_tags_excluding_intersections": nodes,
                                               "downward_load_to_line_kip": sum(reaction_by_node[n] for n in nodes),
                                               "intersection_loads_allocated": False})
            result["beam_line_reactions"] = line_reactions
        displacements = [{"node": n, "x_in": xy[0], "y_in": xy[1],
                          "uz_in": float(ops.nodeDisp(n, 3))} for n, xy in node_positions.items()]
        if any(not math.isfinite(v["uz_in"]) for v in displacements):
            raise RuntimeError("Nonfinite floor displacement.")
        result["vertical_displacements"] = displacements
        result["maximum_downward_displacement_in"] = max(0.0, -min(v["uz_in"] for v in displacements))
        panel_points = {p: [] for p in pressures}
        gauss = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        max_membrane = 0.0
        for element, i, j, panel in element_grid:
            dx, dy = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
            values = list(ops.eleResponse(element, "stresses"))
            if len(values) != 32 or not all(math.isfinite(v) for v in values):
                raise RuntimeError("ShellMITC4 response is not four finite 8-resultant Gauss blocks.")
            for k, (gx, gy) in enumerate(gauss):
                raw = values[8 * k:8 * k + 8]
                max_membrane = max(max_membrane, abs(raw[0]), abs(raw[1]), abs(raw[2]))
                panel_points[panel].append({"element": element, "gauss_point": k + 1,
                    "x_in": xs[i] + (.5 + gx / (2 * math.sqrt(3))) * dx,
                    "y_in": ys[j] + (.5 + gy / (2 * math.sqrt(3))) * dy,
                    "mx": -raw[3], "my": -raw[4], "mxy_raw": raw[5],
                    "qx_raw": raw[6], "qy_raw": raw[7]})
        panels = []
        for (i, j), points in panel_points.items():
            extrema = {key: _extrema([{"value": p[key], "element": p["element"],
                                      "gauss_point": p["gauss_point"], "x_in": p["x_in"], "y_in": p["y_in"]}
                                     for p in points])
                       for key in ("mx", "my", "mxy_raw", "qx_raw", "qy_raw")}
            panels.append({"panel_id": f"panel_x{i + 1}_y{j + 1}", "i": i, "j": j,
                           "pressure_ksf": pressures[(i, j)], "resultant_extrema": extrema,
                           "gauss_point_resultants": points,
                           "direct_bending_face_envelopes_kip_in_per_in": {
                               axis: {"bottom": max(0.0, extrema["m" + axis]["maximum"]["value"]),
                                      "top": max(0.0, -extrema["m" + axis]["minimum"]["value"])}
                               for axis in ("x", "y")},
                           "design_moments_including_twisting": None})
        result["panels"] = panels
        # In-plane DOFs are fixed, so membrane resultants must vanish; the
        # slab strength routine requires that to be shown, not assumed.
        result["max_abs_membrane_kip_per_in"] = max_membrane
        balanced = result["equilibrium"]["numerical_balance_passed"]
        if flexible:
            balanced = balanced and result["transfer_equilibrium"]["numerical_balance_passed"]
        result["status"] = ("transfer_complete" if flexible else "diagnostic_complete") if balanced else "equilibrium_failed"
        return result
    finally:
        ops.wipe()


def _extract_beam_transfer(result, beam_segments, reaction_by_node, mesh, nx, ny, lx, ly,
                           applied, pressures, footprint_in2, *, grid=None):
    """Exact slab-to-frame interface: node forces AND couples, from the solved segments.

    localForce components are the forces the nodes exert on the element. At an
    interior beam-line node the slab delivers (external on the node) the sum of
    what the node exerts on the two adjoining segments: a downward force
    -(Vz_j of the left + Vz_i of the right), a couple about the beam axis
    (Tx_j + Tx_i, the slab's edge moment as beam torsion) and a couple about the
    bending axis (My_j + My_i, the plate's share of bending handed to the
    beam). At a column node the reaction balances the beam end shears plus the
    slab load that reaches the column directly, and the beam end couples are
    the couples the slab applies there.

    Exporting all three actions per node is a statically exact partition of the
    floor model's applied load: force and both global first moments balance to
    solver precision for any pattern (checked below, with the couples). Nothing
    is redistributed: the column node keeps whatever this point-support model
    delivers to it directly (recorded as ``column_direct_fraction``; the
    finite-footprint share is recorded beside it for information only).
    Gravity_Loads applies the couples to the bare frame as statically
    equivalent nodal moments at the beam ends.
    """
    grid = grid or floor_mesh(nx, ny, lx, ly, mesh)
    mesh_x, mesh_y = grid["subdivisions_x_per_bay"], grid["subdivisions_y_per_bay"]
    forces = {}
    by_span = {}
    for segment, axis, line, span, k, node_i, node_j in beam_segments:
        values = list(ops.eleResponse(segment, "localForce"))
        if len(values) != 12 or not all(math.isfinite(v) for v in values):
            raise RuntimeError("Beam segment localForce is not a finite 12-component vector.")
        forces[segment] = values
        by_span.setdefault((axis, line, span), []).append((k, segment, node_i, node_j))
    end_shear_at_node = {}
    end_couple_at_node = {}

    def global_couple(axis, torsion, bending):
        # X beams: local x=X, y=Y. Y beams: local x=Y, y=-X.
        return (torsion, bending) if axis == "x" else (-bending, torsion)

    beam_transfer = []
    for (axis, line, span), segments in sorted(by_span.items()):
        segments.sort()
        length = lx if axis == "x" else ly
        node_loads, node_couples = [], []
        offsets = span_offsets(grid, axis, span)
        for k in range(1, len(offsets) - 1):
            left, right = forces[segments[k - 1][1]], forces[segments[k][1]]
            node_loads.append({"x_fraction": offsets[k] / length, "load_kip": -(left[8] + right[2])})
            node_couples.append({"x_fraction": offsets[k] / length,
                                 "local_x_kip_in": left[9] + right[3],      # torsion about the beam axis
                                 "local_y_kip_in": left[10] + right[4]})    # bending about local y
        first, last = forces[segments[0][1]], forces[segments[-1][1]]
        node_first, node_last = segments[0][2], segments[-1][3]
        end_shear_at_node[node_first] = end_shear_at_node.get(node_first, 0.0) + first[2]
        end_shear_at_node[node_last] = end_shear_at_node.get(node_last, 0.0) + last[8]
        for node, vector, start in ((node_first, first, 0), (node_last, last, 6)):
            mx, my = global_couple(axis, vector[start + 3], vector[start + 4])
            old_x, old_y = end_couple_at_node.get(node, (0., 0.))
            end_couple_at_node[node] = (old_x + mx, old_y + my)
        total = sum(item["load_kip"] for item in node_loads)
        beam_transfer.append({
            "axis": axis, "line_index": line, "span_index": span,
            "node_loads": node_loads, "node_couples": node_couples, "total_kip": total,
            "equivalent_uniform_kip_per_in": total / length,
            "end_shears_kip": [first[2], last[8]],
            "end_moments_kip_in_informational": [first[4], last[10]],
        })
    ex = nx * mesh_x
    column_direct = []
    for node, reaction in reaction_by_node.items():
        index = node - 1
        gi, gj = (index % (ex + 1)) // mesh_x, (index // (ex + 1)) // mesh_y
        # One footprint for every support, or (a floor described line by line) each support's own column.
        quadrant = (footprint_in2[gj][gi] if isinstance(footprint_in2, list) else (footprint_in2 or 0.0)) / 4.0
        share = sum(pressures[(pi, pj)] / 144.0 * quadrant
                    for pi in (gi - 1, gi) for pj in (gj - 1, gj) if (pi, pj) in pressures)
        direct = reaction - end_shear_at_node.get(node, 0.0)
        mx, my = end_couple_at_node.get(node, (0., 0.))
        column_direct.append({"node": node, "grid_i": gi, "grid_j": gj, "reaction_kip": reaction,
                              "direct_load_kip": direct, "footprint_share_kip_informational": share,
                              "couple_global_mx_kip_in": mx, "couple_global_my_kip_in": my})
    column_direct.sort(key=lambda item: (item["grid_j"], item["grid_i"]))
    beam_total = math.fsum(item["total_kip"] for item in beam_transfer)
    direct_total = math.fsum(item["direct_load_kip"] for item in column_direct)
    error = abs(beam_total + direct_total - applied) / applied
    # Global balance of everything exported: for downward P at (x, y) the
    # external moments are Mx = -P*y + couple_x and My = P*x + couple_y.
    load_x = load_y = couple_x = couple_y = 0.0
    for item in beam_transfer:
        for load, couple in zip(item["node_loads"], item["node_couples"]):
            fraction = load["x_fraction"]
            x = (item["span_index"] + fraction) * lx if item["axis"] == "x" else item["line_index"] * lx
            y = item["line_index"] * ly if item["axis"] == "x" else (item["span_index"] + fraction) * ly
            load_x += load["load_kip"] * x
            load_y += load["load_kip"] * y
            gx, gy = global_couple(item["axis"], couple["local_x_kip_in"], couple["local_y_kip_in"])
            couple_x += gx
            couple_y += gy
    for c in column_direct:
        load_x += c["direct_load_kip"] * c["grid_i"] * lx
        load_y += c["direct_load_kip"] * c["grid_j"] * ly
        couple_x += c["couple_global_mx_kip_in"]
        couple_y += c["couple_global_my_kip_in"]
    applied_x = sum(q * lx * ly / 144.0 * (i + .5) * lx for (i, j), q in pressures.items())
    applied_y = sum(q * lx * ly / 144.0 * (j + .5) * ly for (i, j), q in pressures.items())
    error_x = abs(load_x + couple_y - applied_x) / max(abs(applied_x), applied * lx)
    error_y = abs(load_y - couple_x - applied_y) / max(abs(applied_y), applied * ly)
    result["beam_transfer"] = beam_transfer
    result["column_direct_loads"] = column_direct
    result["transfer_equilibrium"] = {
        "applied_downward_kip": applied,
        "beam_node_loads_kip": beam_total,
        "column_direct_loads_kip": direct_total,
        "column_direct_fraction": direct_total / applied,
        "column_footprint_in2": footprint_in2,
        "relative_error": error,
        "applied_x_first_moment_kip_in": applied_x,
        "applied_y_first_moment_kip_in": applied_y,
        "vertical_load_x_first_moment_kip_in": load_x,
        "vertical_load_y_first_moment_kip_in": load_y,
        "global_couple_x_kip_in": couple_x,
        "global_couple_y_kip_in": couple_y,
        "x_first_moment_relative_error": error_x,
        "y_first_moment_relative_error": error_y,
        "vertical_load_only_x_first_moment_relative_error": abs(load_x - applied_x) / max(abs(applied_x), applied * lx),
        "vertical_load_only_y_first_moment_relative_error": abs(load_y - applied_y) / max(abs(applied_y), applied * ly),
        "vertical_force_balance_passed": error < 1e-8,
        "numerical_balance_passed": max(error, error_x, error_y) < 1e-8,
        "basis": ("exported node forces and couples (torsion and bending) at beam-line nodes plus column-node "
                  "forces and couples; no redistribution; balance includes the couples"),
    }
