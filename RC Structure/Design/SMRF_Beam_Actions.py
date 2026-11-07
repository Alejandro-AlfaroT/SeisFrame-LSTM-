"""Recover vertical beam bending from signed static equilibrium, not sampling.

Scope: the project's straight, horizontal 3D frame beams, local z upward,
Linear transformations, full-length uniform loads and point forces. No
distributed couples, partial/trapezoidal loads, or distributed inertial
loading. Unknown elemental loads are rejected rather than silently omitted.

OpenSees references (2026-09-13):
https://openseespydoc.readthedocs.io/en/latest/src/eleload.html
https://github.com/OpenSees/OpenSees/blob/master/SRC/interpreter/OpenSeesOutputCommands.cpp
getEleLoadData returns reference data; getLoadFactor supplies the current
pattern factor, including frozen gravity after loadConst('-time', 0).

With the existing beam transform, sagging M(x) = My_i + Vz_i*x +
wz*x*x/2 + sum(Pz*(x-a)) for loads to the left of x. Downward wz/Pz are
negative. At the right end, M(L)=-My_j and V(L-)=-Vz_j. These identities
are checked before accepting an envelope. Local-y bending only; this is not
a biaxial or beam-axial strength check or nonlinear hinge qualification.
"""
from __future__ import annotations

import math

METHOD_VERSION = "smrf_beam_static_span_equilibrium_v1"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Beam actions {name} must be finite numeric data.")
    return float(value)


def recover_beam_bending(length_in, local_force, uniform_z_kip_per_in=0.0,
                         point_z_loads=(), face_offsets_in=(0.0, 0.0)):
    """Exact piecewise-quadratic envelope under signed vertical static loads.

    Point rows are [fraction_of_length, signed_local_z_force_kip]. Duplicate
    locations are combined. Both full-centerline and optional clear-face
    intervals are reported; no moment reduction is inferred from column size.
    """
    length = _finite(length_in, "length")
    if length <= 0 or len(local_force) != 12:
        raise ValueError("Beam actions require a positive length and 12 local forces.")
    f = [_finite(v, "local force") for v in local_force]
    w = _finite(uniform_z_kip_per_in, "uniform load")
    points = {}
    for row in point_z_loads:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("Beam point loads must be [fraction, signed force] pairs.")
        fraction, p = _finite(row[0], "point position"), _finite(row[1], "point load")
        if not 0 < fraction < 1:
            raise ValueError("Beam point loads must be strictly inside the element.")
        points[fraction * length] = points.get(fraction * length, 0.0) + p
    points = sorted(points.items())
    if len(face_offsets_in) != 2:
        raise ValueError("Provide two nonnegative face offsets.")
    offset_i, offset_j = [_finite(v, "face offset") for v in face_offsets_in]
    if min(offset_i, offset_j) < 0 or offset_i + offset_j >= length:
        raise ValueError("Beam face offsets must leave a positive clear span.")

    def moment(x):
        return f[4] + f[2] * x + w * x*x/2 + math.fsum(p * (x-a) for a, p in points if a < x)

    total = math.fsum(p for a, p in points)
    end_m, end_v = moment(length), f[2] + w * length + total
    force_scale = max(1.0, abs(f[2]), abs(f[8]), abs(w)*length + math.fsum(abs(p) for a, p in points))
    moment_scale = max(1.0, abs(f[4]), abs(f[10]), force_scale * length)
    vm_error, mm_error = abs(end_v + f[8]), abs(end_m + f[10])
    if vm_error > 1e-8 * force_scale or mm_error > 1e-8 * moment_scale:
        raise ValueError("Beam end-force/load equilibrium failed; loads, local axes or analysis scope do not match. "
                         f"Shear residual={vm_error:.9g} kip, moment residual={mm_error:.9g} kip-in.")
    candidates = {0.0, length, offset_i, length-offset_j, *(a for a, p in points)}
    breaks = [0.0, *(a for a, p in points), length]
    for left, right in zip(breaks, breaks[1:]):
        if w != 0:
            root = -(f[2] + math.fsum(p for a, p in points if a <= left)) / w
            if left < root < right:
                candidates.add(root)
    samples = [{"x_in": x, "moment_kip_in": moment(x)} for x in sorted(candidates)]

    def envelope(start, stop):
        rows = [p for p in samples if start <= p["x_in"] <= stop]
        low = min(rows, key=lambda p: p["moment_kip_in"])
        high = max(rows, key=lambda p: p["moment_kip_in"])
        return {"mu_positive_kip_in": max(0.0, high["moment_kip_in"]),
                "mu_negative_kip_in": max(0.0, -low["moment_kip_in"]),
                "positive_x_in": high["x_in"], "negative_x_in": low["x_in"],
                "interval_in": [start, stop]}
    return {"method_version": METHOD_VERSION, "length_in": length,
            "uniform_z_kip_per_in": w, "point_z_loads": [[a/length, p] for a, p in points],
            "face_offsets_in": [offset_i, offset_j],
            "full_span": envelope(0., length), "clear_span": envelope(offset_i, length-offset_j),
            "critical_sections": samples,
            "equilibrium": {"right_shear_residual_kip": vm_error,
                            "right_moment_residual_kip_in": mm_error, "passed": True},
            "scope": "vertical static local-y bending; full uniform and interior point forces; no member P-delta or distributed inertia"}


def applied_element_loads():
    """Read fresh pattern-by-pattern loads; never infer factors from domain time.

    Class tags 5/6 and data lengths/order are pinned by OpenSees integration
    tests. Do not reuse these loads after changing the domain or load factors.
    """
    import openseespy.opensees as ops
    result = {}
    for pattern in ops.getPatterns():
        tags = list(ops.getEleLoadTags(pattern))
        classes = list(ops.getEleLoadClassTags(pattern))
        raw = list(ops.getEleLoadData(pattern))
        factor = _finite(ops.getLoadFactor(pattern), "pattern factor")
        if len(tags) != len(classes):
            raise ValueError("OpenSees element-load tag/class inventories disagree.")
        cursor = 0
        for tag, kind in zip(tags, classes):
            count = {5: 3, 6: 4}.get(kind)
            if count is None:
                raise ValueError(f"Unsupported OpenSees element load class {kind}; cannot recover a complete beam envelope.")
            data = raw[cursor:cursor+count]
            if len(data) != count:
                raise ValueError("Truncated OpenSees element-load data.")
            data = [_finite(v, "reference load") for v in data]
            cursor += count
            entry = result.setdefault(tag, {"uniform_z_kip_per_in": 0., "point_z_loads": []})
            if kind == 5:  # reference [wy, wz, wx]
                entry["uniform_z_kip_per_in"] += factor * data[1]
            else:          # reference [Py, Pz, Px, fraction]
                entry["point_z_loads"].append([data[3], factor * data[1]])
        if cursor != len(raw):
            raise ValueError("Unconsumed OpenSees element-load data.")
    return result


def current_beam_bending(beam_tags):
    """Batch-recover current project beam envelopes without mutating OpenSees."""
    import openseespy.opensees as ops
    beam_tags = list(beam_tags)
    if not beam_tags:
        return {}
    loads = applied_element_loads()
    result = {}
    for tag in beam_tags:
        nodes = ops.eleNodes(tag)
        if len(nodes) != 2:
            raise ValueError("Beam recovery requires two-node frame elements.")
        a, b = [ops.nodeCoord(n) for n in nodes]
        if len(a) != 3 or len(b) != 3 or abs(a[2]-b[2]) > 1e-9:
            raise ValueError("Beam recovery is restricted to horizontal 3D project beams.")
        length = math.dist(a, b)
        force = list(ops.eleResponse(tag, "localForce"))
        item = loads.get(tag, {"uniform_z_kip_per_in": 0., "point_z_loads": []})
        result[tag] = recover_beam_bending(length, force, **item)
    return result


def _matches(saved, recomputed):
    if isinstance(recomputed, dict):
        return (isinstance(saved, dict) and set(saved) == set(recomputed)
                and all(_matches(saved[k], v) for k, v in recomputed.items()))
    if isinstance(recomputed, list):
        return (isinstance(saved, list) and len(saved) == len(recomputed)
                and all(_matches(a, b) for a, b in zip(saved, recomputed)))
    if isinstance(recomputed, bool):
        return saved is recomputed
    if isinstance(recomputed, (int, float)):
        return (not isinstance(saved, bool) and isinstance(saved, (int, float))
                and math.isfinite(saved) and math.isclose(saved, recomputed, rel_tol=1e-10, abs_tol=1e-8))
    return saved == recomputed


def evaluate_saved_beam_bending(record):
    """Recompute saved beam envelopes for every physical beam and solved case.

    This closes the bounded interior-envelope software check, not independent
    validation of the frame demand basis or its interface mechanics.
    """
    from Design.SMRF_Common import make_check, not_evaluated
    from Design.SMRF_Design_Evidence import analysis_input_signature
    check_id, clause = "beam.interior_flexure_envelope", "Static loaded-beam span equilibrium and exact extrema"
    actions = record.get("design_actions") or {}
    if not actions.get("combinations"):
        return not_evaluated(check_id, clause, "No solved beam-action inventory.")
    try:
        if actions.get("analysis_input_sha256") != analysis_input_signature(record):
            raise ValueError("Beam actions do not match the current frame and load inputs.")
        g = record["geometry"]
        nx, ny, nf = (g[k] for k in ("num_bay_x", "num_bay_y", "num_floor"))
        nc, nbx = nf*(nx+1)*(ny+1), nf*nx*(ny+1)
        expected = {str(t): ("beam_x", g["bay_x_in"]) for t in range(nc+1, nc+nbx+1)}
        expected.update({str(t): ("beam_y", g["bay_y_in"])
                         for t in range(nc+nbx+1, nc+nbx+nf*ny*(nx+1)+1)})
        count, case_ids = 0, set()
        for case in actions["combinations"]:
            if case["id"] in case_ids or case.get("analysis_succeeded") is not True:
                raise ValueError("Duplicated or unsuccessful solved combination.")
            case_ids.add(case["id"])
            members = case["members"]
            beam_keys = {k for k, m in members.items() if m.get("member_type") in ("beam_x", "beam_y")}
            if beam_keys != set(expected):
                raise ValueError("Saved beam-action inventory is incomplete or unexpected.")
            for tag, (kind, length) in expected.items():
                member, span = members[tag], members[tag].get("span_bending")
                if not isinstance(span, dict):
                    return not_evaluated(check_id, clause, "Saved member actions predate span-interior recovery; rerun design-only in a new root.")
                if member["member_type"] != kind or not _matches(span.get("length_in"), length):
                    raise ValueError(f"Beam {tag} geometry differs from its recovered diagram.")
                if str(span.get('method_version','')).startswith('physical_piecewise_section_force_extrema_'):
                    from Design.SMRF_Physical_Member_Actions import validated_member_span
                    recomputed=validated_member_span(member)
                else:
                    recomputed = recover_beam_bending(length, member["local_force_kip_kipin"],
                                                      span["uniform_z_kip_per_in"], span["point_z_loads"],
                                                      span["face_offsets_in"])
                if not _matches(span, recomputed):
                    raise ValueError(f"Beam {tag} saved extrema differ from recomputed equilibrium.")
                count += 1
        return make_check(check_id, clause, 1, 1, "==",
                          details={"method_version": METHOD_VERSION, "beam_case_count": count,
                                   "basis": "Full-centerline vertical bending; joint-face capacity design and independent engineering verification remain separate."})
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        return make_check(check_id, clause, 0, 1, "==", details={"error": str(exc)})
