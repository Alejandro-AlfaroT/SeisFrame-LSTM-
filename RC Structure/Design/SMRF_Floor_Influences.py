"""Actual frame floor-force influences and auditable inertia/transfer separation.

No OpenSees import or solve occurs on module import. The optional production
runner rebuilds the installed production design frame for every requested basis
and witness, and needs an explicit protocol identity and bounded solve budget.
It does not launch a design search. Pure functions operate on saved global end
forces. All increments retain above and below columns separately.

The frame uses PDelta transforms: a superposition assertion is NEVER implicit.
The direct full-load witnesses must reconstruct from the basis within declared
tolerances. Failure is recorded and prevents validated decomposition, even if
the elastic-material label would otherwise suggest a linear response.
"""
from __future__ import annotations

import copy
import math
import time
import numpy as np

METHOD_VERSION = "actual_frame_floor_influences_v2_central_residual"
DOFS = ("x", "y", "rz")


def _loads(loads, count):
    a = np.asarray(loads, float)
    if a.shape != (count, 3) or not np.isfinite(a).all():
        raise ValueError("floor loads must be finite [Fx_kip,Fy_kip,Mz_kip_in] per floor")
    return a


def _column_dict(response):
    rows = {str(c["tag"]): c for c in response["columns"]}
    if len(rows) != len(response["columns"]):
        raise ValueError("duplicate column tags")
    for row in rows.values():
        force = np.asarray(row["global_force"], float)
        if force.shape != (12,) or not np.isfinite(force).all():
            raise ValueError("column global_force must have twelve finite entries")
    return rows


def subtract_response(response, baseline):
    """Remove exactly the same gravity state, preserving geometry and force sign."""
    rows, base = _column_dict(response), _column_dict(baseline)
    if set(rows) != set(base) or response["floor_masters"] != baseline["floor_masters"]:
        raise ValueError("response/baseline model topology differs")
    out = copy.deepcopy(response)
    for c in out["columns"]:
        b = base[str(c["tag"])]
        for key in ("i_level", "j_level", "i_node", "j_node", "i_xyz", "j_xyz"):
            if c[key] != b[key]:
                raise ValueError("response/baseline physical column geometry differs")
        c["global_force"] = (np.asarray(c["global_force"])-np.asarray(b["global_force"])).tolist()
    displ, db = np.asarray(response["master_displacements"]), np.asarray(baseline["master_displacements"])
    if displ.shape != db.shape or not np.isfinite(displ).all() or not np.isfinite(db).all():
        raise ValueError("master displacement arrays differ or are nonfinite")
    out["master_displacements"] = (displ-db).tolist()
    return out


def sum_responses(weighted_responses):
    """Linear combination of saved force increments, including every column."""
    weighted_responses = list(weighted_responses)
    if not weighted_responses:
        raise ValueError("no influence responses supplied")
    template = weighted_responses[0][1]
    out = copy.deepcopy(template)
    columns = _column_dict(out)
    for c in columns.values():
        c["global_force"] = np.zeros(12)
    disp = np.zeros_like(np.asarray(template["master_displacements"], float))
    for factor, response in weighted_responses:
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError("nonfinite influence multiplier")
        rows = _column_dict(response)
        if set(rows) != set(columns) or response["floor_masters"] != template["floor_masters"]:
            raise ValueError("influence topology changed")
        for key, c in columns.items():
            r = rows[key]
            if any(c[k] != r[k] for k in ("i_level", "j_level", "i_node", "j_node", "i_xyz", "j_xyz")):
                raise ValueError("influence physical column geometry changed")
            c["global_force"] += factor*np.asarray(r["global_force"])
        disp += factor*np.asarray(response["master_displacements"], float)
    for c in columns.values():
        c["global_force"] = c["global_force"].tolist()
    out["master_displacements"] = disp.tolist()
    return out


def floor_free_bodies(response, applied_floor_loads, *, tolerance=1e-7):
    """Force ON floor = minus column resisting globalForce at its incident end.

    At each elevated level, a column below contributes its j-end and a column
    above contributes its i-end. Column torsion Mz remains an applied floor
    couple, separate from the r-cross-F moment of column horizontal forces.
    All columns, including gravity/leaning columns, are retained.
    """
    masters = response["floor_masters"]
    loads = _loads(applied_floor_loads, len(masters))
    rows = []
    for level, master in enumerate(masters, 1):
        origin = np.asarray(master["xyz"][:2], float)
        above, below = [], []
        balance = loads[level-1].copy()
        magnitude, length = np.abs(balance).copy(), 1.0
        for c in _column_dict(response).values():
            if c["i_level"] == level:
                off, bucket, node, coord = 0, above, c["i_node"], c["i_xyz"]
            elif c["j_level"] == level:
                off, bucket, node, coord = 6, below, c["j_node"], c["j_xyz"]
            else:
                continue
            f = -np.asarray(c["global_force"], float)[off:off+6]
            r = np.asarray(coord[:2])-origin
            mz = f[5]+r[0]*f[1]-r[1]*f[0]
            term = np.array([f[0], f[1], mz])
            balance += term
            magnitude += np.abs(term)
            length = max(length, float(np.linalg.norm(r)))
            bucket.append({"column_tag": str(c["tag"]), "node": node, "xy_in": list(coord[:2]),
                           "Fx_kip": float(f[0]), "Fy_kip": float(f[1]), "Mz_kip_in": float(f[5])})
        scales = np.array([max(magnitude[0], 1.0), max(magnitude[1], 1.0),
                           max(magnitude[2], length)])
        err = float(np.max(np.abs(balance)/scales))
        rows.append({"level": level, "master_xy_in": origin.tolist(),
                     "applied_load": loads[level-1].tolist(), "columns_above_on_floor": above,
                     "columns_below_on_floor": below, "balance_error_kip_kipin": balance.tolist(),
                     "normalized_balance_error": err, "equilibrium_passed": err <= tolerance})
    return rows


def reconstruct_from_basis(influences, loads, *, source_floor=None):
    """Optional source_floor selects that floor's entire force+torsion influence."""
    n = influences["num_floor"]
    loads = _loads(loads, n)
    terms, included = [], set()
    for row in influences["basis"]:
        floor, direction = row["source_floor"], row["dof"]
        if (floor, direction) in included or direction not in DOFS or not 1 <= floor <= n:
            raise ValueError("duplicate/invalid floor influence coordinate")
        included.add((floor, direction))
        if not math.isfinite(float(row["amplitude"])) or row["amplitude"] == 0:
            raise ValueError("influence amplitude must be nonzero and finite")
        if source_floor is None or floor == source_floor:
            terms.append((loads[floor-1, DOFS.index(direction)]/row["amplitude"], row["response"]))
    for floor in range(1, n+1):
        if source_floor is None or floor == source_floor:
            for d, direction in enumerate(DOFS):
                if loads[floor-1, d] != 0 and (floor, direction) not in included:
                    raise ValueError("nonzero applied load has no corresponding unit influence")
    return sum_responses(terms)


def verify_superposition(influences, loads, direct_increment, *, force_atol=1e-6,
                         moment_atol=1e-4, displacement_atol=1e-8, relative_tolerance=1e-6):
    """Compare all global column end forces and master planar displacements.

    These numerical tolerances test affine response, not a physical model error
    allowance. Tolerances are recorded and must be fixed in the fresh protocol.
    """
    expected = reconstruct_from_basis(influences, loads)
    actual = _column_dict(direct_increment)
    if min(force_atol, moment_atol, displacement_atol, relative_tolerance) <= 0:
        raise ValueError("superposition tolerances must be positive")
    if set(actual) != set(_column_dict(expected)):
        raise ValueError("direct witness column topology differs")
    rows, passed, maximum = [], True, 0.0
    for key, c in _column_dict(expected).items():
        a, b = np.asarray(c["global_force"]), np.asarray(actual[key]["global_force"])
        atol = np.array([force_atol]*3+[moment_atol]*3)*1.0
        allowed = np.tile(atol, 2)+relative_tolerance*np.maximum(np.abs(a), np.abs(b))
        ratio = float(np.max(np.abs(a-b)/allowed))
        maximum = max(maximum, ratio)
        ok = ratio <= 1
        passed &= ok
        rows.append({"column_tag": key, "maximum_error_over_tolerance": ratio, "passed": ok})
    a, b = np.asarray(expected["master_displacements"]), np.asarray(direct_increment["master_displacements"])
    allowed = displacement_atol+relative_tolerance*np.maximum(np.abs(a), np.abs(b))
    displ_ratio = float(np.max(np.abs(a-b)/allowed))
    passed &= displ_ratio <= 1
    bodies = floor_free_bodies(direct_increment, loads)
    passed &= all(f["equilibrium_passed"] for f in bodies)
    return {"passed": bool(passed), "maximum_force_error_over_tolerance": maximum,
            "maximum_displacement_error_over_tolerance": displ_ratio, "column_checks": rows,
            "direct_free_bodies": bodies, "tolerances": {"force_kip": force_atol,
            "moment_kip_in": moment_atol, "displacement_in_rad": displacement_atol,
            "relative": relative_tolerance}}


def decompose_floor_influences(influences, loads, target_floor, *, require_verified=True):
    """Separate target floor inertia from all other-floor transfer influences.

    Pass the actual signed force/torsion vector used in the verified full case.
    Witness validation is matched to that load vector, not an unrelated case.
    """
    n = influences["num_floor"]
    loads = _loads(loads, n)
    if not 1 <= int(target_floor) <= n:
        raise ValueError("target floor outside building")
    if require_verified:
        matches = [w for w in influences.get("witnesses", []) if
                   np.array_equal(np.asarray(w["loads"]), loads) and w["verification"]["passed"]]
        if not matches:
            raise ValueError("this complete signed load vector lacks a passing direct superposition witness")
    local = reconstruct_from_basis(influences, loads, source_floor=target_floor)
    complete = reconstruct_from_basis(influences, loads)
    transfer = subtract_response(complete, local)
    local_load = np.zeros_like(loads)
    local_load[target_floor-1] = loads[target_floor-1]
    transfer_load = loads-local_load
    out = []
    for role, response, applied in (("local_inertia", local, local_load), ("transfer", transfer, transfer_load)):
        floor = floor_free_bodies(response, applied)[target_floor-1]
        if not floor["equilibrium_passed"]:
            raise ValueError(f"{role} target-floor component does not balance")
        out.append({**floor, "id": f"floor_{target_floor}_{role}", "role": role,
                    "provenance": f"{METHOD_VERSION}; source={influences['source_identity']}; target floor {target_floor}"})
    return out


def membrane_components(mesh, floor_components, *, column_footprints, nodal_mass_weights):
    """Map physical column force/couple footprints to membrane solver arrays.

    column_footprints maps column_tag to its own node-weight footprint. Actual
    above/below dimensions may differ. Nodal masses must preserve the production
    seismic weight distribution. The inertia couple is transformed between the
    frame master and mass centroid once, preserving total planar moment.
    """
    from Design.SMRF_Diaphragm_Actions import footprint_traction, inertial_nodal_forces
    xy = np.asarray(mesh["nodes"])
    weights = np.asarray(nodal_mass_weights, float)
    if weights.shape != (len(xy),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("finite nonnegative actual nodal mass weights required")
    center = weights @ xy / weights.sum()
    out = []
    for row in floor_components:
        f = np.asarray(row["applied_load"], float)
        arm = np.asarray(row["master_xy_in"])-center
        torque_at_center = f[2]+arm[0]*f[1]-arm[1]*f[0]
        component = {"id": row["id"], "role": row["role"], "provenance": row["provenance"],
                     "inertia": inertial_nodal_forces(mesh, weights, f[:2], torque_at_center)}
        for field in ("columns_above_on_floor", "columns_below_on_floor"):
            tractions = np.zeros_like(xy)
            for column in row[field]:
                tag = column["column_tag"]
                if tag not in column_footprints:
                    raise ValueError(f"no physical traction footprint for column {tag}")
                tractions += np.asarray(footprint_traction(mesh, column_footprints[tag],
                       [column["Fx_kip"], column["Fy_kip"]], application_xy_in=column["xy_in"],
                       moment_kip_in=column["Mz_kip_in"]))
            component[field] = tractions.tolist()
        out.append(component)
    return out


def decompose_with_second_order_residual(influences, witness_id, target_floor):
    """Exact action ledger using a verified gravity-state linearization.

    The residual has zero applied floor inertia and is kept as a distinct
    self-equilibrated second-order redistribution. It is not silently folded
    into an inertial scale or discarded after a failed superposition test.
    This ledger does not by itself qualify a diaphragm design procedure.
    """
    if not influences.get('all_linearizations_converged'):
        raise ValueError('Converged central-difference gravity-state linearization required')
    witnesses = [w for w in influences.get('witnesses', []) if w['id'] == witness_id]
    if len(witnesses) != 1:
        raise ValueError('Unique direct witness required')
    witness = witnesses[0]; n = influences['num_floor']
    loads = _loads(witness['loads'], n)
    if not 1 <= target_floor <= n:
        raise ValueError('Target floor outside model')
    local = reconstruct_from_basis(influences, loads, source_floor=target_floor)
    linear = reconstruct_from_basis(influences, loads)
    transfer = subtract_response(linear, local)
    residual = subtract_response(witness['direct_increment'], linear)
    local_load = np.zeros_like(loads); local_load[target_floor-1] = loads[target_floor-1]
    components = []
    for label, role, response, applied in (
            ('local_inertia', 'local_inertia', local, local_load),
            ('other_floor_transfer', 'transfer', transfer, loads-local_load),
            ('second_order_redistribution', 'transfer', residual, np.zeros_like(loads))):
        floor = floor_free_bodies(response, applied)[target_floor-1]
        if not floor['equilibrium_passed']:
            raise ValueError(f'{label} physical floor free body failed')
        components.append({**floor, 'id': f'floor_{target_floor}_{label}', 'role': role,
            'action_origin': label, 'provenance': f'{METHOD_VERSION}; source={influences["source_identity"]}; direct={witness_id}'})
    reconstructed = sum_responses([(1., local), (1., transfer), (1., residual)])
    direct = _column_dict(witness['direct_increment'])
    maximum = max(float(np.max(np.abs(np.asarray(c['global_force'])-np.asarray(direct[tag]['global_force']))))
                  for tag, c in _column_dict(reconstructed).items())
    if maximum > 1e-7:
        raise ValueError('Second-order action ledger does not reconstruct the direct response')
    return {'components': components, 'direct_reconstruction_max_abs_kip_kipin': maximum,
            'method_qualification': 'not_asserted', 'residual_inertial_scaling_permitted': False,
            'scope': 'exact original witness action ledger; amplified design method requires separate evidence'}


def validate_physical_binding(definition,grid_mass_by_floor,*,physical_definition_sha256,state_identity,nodal_mass_sha256):
    """Recompute all producer bindings before any model build or solve."""
    from Design.SMRF_Coupled_Floor_Design import validate_definition
    from Design.SMRF_Physical_Diaphragm import identity
    validate_definition(definition)
    actual={'physical_definition_sha256':definition['identity_sha256'],
            'state_identity':definition['state_identity'],'nodal_mass_sha256':identity(grid_mass_by_floor)}
    expected={'physical_definition_sha256':physical_definition_sha256,
              'state_identity':state_identity,'nodal_mass_sha256':nodal_mass_sha256}
    if any(not isinstance(v,str) or not v for v in expected.values()) or actual!=expected:
        raise ValueError('Declared physical definition/state/mass identity differs from computed installed inputs')
    return actual


def run_production_floor_influences(*, protocol_id, source_identity, gravity_combination,
                                   physical_definition_sha256,state_identity,nodal_mass_sha256,
                                   witness_cases, max_model_solves, max_wall_seconds,
                                   force_amplitude_kip=1.0, torque_amplitude_kip_in=1.0,
                                   verification_tolerances=None, central_step_halving=False,
                                   linearization_tolerances=None, analysis_displacement_tolerance=1e-10):
    """Explicit bounded runner on the installed production design model.

    The installed grouped state is used, including actual layout, member grades,
    sizes and release policy. Geometry/member state is not changed. The OpenSees
    domain is rebuilt and left at the last witness; use in an isolated worker.
    Every solve includes identical factored gravity plus the selected master
    loads. Material elasticity is not used to waive the PDelta affine witness.
    witness_cases rows require id and loads (nfloor by3). Budget includes one
    gravity baseline,3*nfloor bases, and every direct witness. This function is
    not called by existing design code; integration must pin the fresh protocol.
    """
    if not protocol_id or not source_identity or not witness_cases:
        raise ValueError("protocol/source identity and direct witness cases are required")
    for value in (force_amplitude_kip, torque_amplitude_kip_in, max_wall_seconds):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("positive finite amplitudes and wall budget required")
    if type(max_model_solves) is not int or max_model_solves < 1:
        raise ValueError("max_model_solves must be a positive integer")
    import openseespy.opensees as ops
    import Structure_Parameters as sp
    from Analysis.Gravity import run_gravity_analysis
    from Design import Design_Driver as driver
    from Design import Perimeter_Systems as systems
    from Design.SMRF_Elastic import physical_members
    from Model.Build_Model import floor_master_node
    from Model import Member_Groups as mg
    from Design.SMRF_Coupled_Floor_Design import installed_definition
    from Design.SMRF_Physical_Diaphragm import installed_grid_masses
    physical_definition=installed_definition()
    grid_masses=installed_grid_masses(physical_definition)
    physical_binding=validate_physical_binding(physical_definition,grid_masses,
        physical_definition_sha256=physical_definition_sha256,state_identity=state_identity,nodal_mass_sha256=nodal_mass_sha256)
    n = sp.NUM_FLOOR
    precision = min(float(sp.GRAVITY_TOL), float(analysis_displacement_tolerance))
    if not math.isfinite(precision) or precision <= 0:
        raise ValueError('Positive finite static displacement tolerance required')
    expected = 1+(12 if central_step_halving else 3)*n+len(witness_cases)
    if expected > max_model_solves:
        raise ValueError(f"protocol budget {max_model_solves} is below required {expected} solves")
    gravity = {"id": "floor_influence_gravity", "dead": float(gravity_combination["dead"]),
               "live": float(gravity_combination["live"]), "ex": 0.0, "ey": 0.0,
               "live_pattern": gravity_combination.get("live_pattern", "all")}
    model = systems.design_model_for()
    start, solves = time.monotonic(), 0
    def run(loads):
        nonlocal solves
        if solves >= max_model_solves or time.monotonic()-start > max_wall_seconds:
            raise RuntimeError("floor influence protocol budget exhausted")
        loads = _loads(loads, n)
        with systems.lateral_model(model):
            driver._apply_combination_loads(gravity, None, None)
            ops.timeSeries("Linear", 951)
            ops.pattern("Plain", 951, 951)
            for level, f in enumerate(loads, 1):
                ops.load(floor_master_node(level), f[0], f[1], 0., 0., 0., f[2])
            original_tolerance = sp.GRAVITY_TOL
            try:
                # Resolving a small self-equilibrated residual requires more
                # accuracy than routine total-force design analyses. Tighten,
                # never loosen, the declared production convergence criterion.
                sp.GRAVITY_TOL = precision
                with driver._quiet():
                    run_gravity_analysis(constraint_handler='Transformation')
            finally:
                sp.GRAVITY_TOL = original_tolerance
        solves += 1
        columns = []
        for tag, ni, nj, kind in physical_members():
            if kind != "column":
                continue
            ci, cj = list(ops.nodeCoord(ni)), list(ops.nodeCoord(nj))
            columns.append({"tag": str(tag), "i_node": ni, "j_node": nj, "i_xyz": ci, "j_xyz": cj,
                            "i_level": int(round(ci[2]/sp.STORY_H)),
                            "j_level": int(round(cj[2]/sp.STORY_H)),
                            "global_force": list(ops.eleResponse(tag, "globalForce"))})
        masters = [{"node": floor_master_node(k), "xyz": list(ops.nodeCoord(floor_master_node(k)))}
                   for k in range(1, n+1)]
        displ = [[ops.nodeDisp(row["node"], dof) for dof in (1, 2, 6)] for row in masters]
        return {"columns": columns, "floor_masters": masters, "master_displacements": displ}
    baseline = run(np.zeros((n, 3)))
    out = {"method": METHOD_VERSION, "protocol_id": str(protocol_id), "source_identity": str(source_identity),
           **physical_binding,'grid_mass_by_floor':grid_masses,
           "lateral_design_model": model, "active_layout": mg.active_layout(), "num_floor": n,
           'analysis_displacement_tolerance': precision,
           'constraint_handler': 'Transformation; actual elastic frame has no IMK constraint chains',
           "gravity_combination": gravity, "baseline": baseline, "basis": [], "witnesses": []}
    for floor in range(1, n+1):
        for dof in DOFS:
            amplitude = torque_amplitude_kip_in if dof == "rz" else force_amplitude_kip
            loads = np.zeros((n, 3)); loads[floor-1, DOFS.index(dof)] = amplitude
            linearization = None
            if central_step_halving:
                plus, minus = run(loads), run(-loads)
                coarse = sum_responses([(.5, subtract_response(plus, minus))])
                # Both rows are normalized to the FULL amplitude, so they
                # compare the same tangent-force derivative times amplitude.
                half_plus, half_minus = run(loads/2), run(-loads/2)
                response = subtract_response(half_plus, half_minus)
                trial = {'num_floor': n, 'basis': [{'source_floor': floor, 'dof': dof,
                          'amplitude': amplitude, 'response': coarse}]}
                linearization = verify_superposition(trial, loads, response, **(linearization_tolerances or {}))
                linearization['coarse_response'] = coarse
                linearization['definition'] = 'central differences at +/-h and +/-h/2 about identical gravity; stored response normalized to h'
            else:
                response = subtract_response(run(loads), baseline)
            bodies = floor_free_bodies(response, loads)
            if not all(f["equilibrium_passed"] for f in bodies):
                raise ValueError("unit floor influence does not satisfy physical floor equilibrium")
            out["basis"].append({"source_floor": floor, "dof": dof, "amplitude": amplitude,
                                  "response": response, "free_bodies": bodies,
                                  'linearization_verification': linearization})
    for witness in witness_cases:
        loads = _loads(witness["loads"], n)
        direct = subtract_response(run(loads), baseline)
        check = verify_superposition(out, loads, direct, **(verification_tolerances or {}))
        out["witnesses"].append({"id": str(witness["id"]), "loads": loads.tolist(),
                                  "direct_increment": direct, "verification": check})
    out.update(model_solves=solves, wall_seconds=time.monotonic()-start,
               all_linearizations_converged=bool(central_step_halving and all(
                   row['linearization_verification']['passed'] for row in out['basis'])),
               all_witnesses_passed=all(w["verification"]["passed"] for w in out["witnesses"]),
               scope="only matching signed witness vectors; no silent PDelta linearity assumption")
    return out
