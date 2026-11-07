"""Equilibrated diaphragm action recovery, independent of strength qualification.

Units are inch, kip, ksi. Membrane Nxx/Nyy/Nxy are kip/in, tension positive;
beam N is kip, tension positive. Q4 plane-stress slab elements and axial-only
beam WEB trusses occupy disjoint concrete. The slab is counted over the whole
plan; beam area MUST exclude its overlap with the slab. Forces supplied here
are forces ON the floor, including separate columns above/below. This free-body
solver does not invent vertical-system reactions or redistribute them to two
exterior supports. Three gauges remove rigid motion; their reactions must vanish.

ACI 318-19 12.4.2.3/12.4.2.4(d) permit a consistent cracked-stiffness FE model.
12.5.1.3(c) separately requires Chapter 22 strength; this module provides NO
strength certification. 12.6.3 requires in-plane steel in addition to steel
required for other concurrent effects (except shrinkage/temperature steel).
ASCE 7-22 conventional 12.10.1.1/12.10.2.1 require separate inertial and transfer
components. A validated unit-floor-load influence decomposition is necessary:
scaling a cancelling total above-minus-below reaction is not an alternative.

Scope: full rectangular floors on an aligned rectilinear mesh, finite prescribed
column traction footprints, no openings, no in-plane bending stiffness in beam
webs. Stiffness sensitivity, physical footprint and mesh witnesses remain the
caller's responsibilities. Point-load peaks are not converged design stresses.
"""
from __future__ import annotations

import math
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

METHOD_VERSION = "q4_membrane_web_axial_balanced_components_v1"
_GAUSS = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


def _finite(value, name, positive=False):
    out = float(value)
    if not math.isfinite(out) or (positive and out <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return out


def _vector(values, n, name):
    out = np.asarray(values, dtype=float)
    if out.shape != (n, 2) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {n} finite [Fx,Fy] vectors")
    return out


def rectangular_mesh(x_in, y_in, *, thickness_in, elastic_modulus_ksi,
                     poisson=0.2, membrane_stiffness_factor=1.0, beams=()):
    """Create the complete rectangular floor. Beam rows: id,i,j,area_web_in2,E_ksi.

    i/j are mesh node indices, row-major in y then x. Each truss must connect
    adjacent nodes horizontally/vertically. Split actual beams at mesh nodes,
    retaining a member_id on every segment for force envelope reconstruction.
    Stiffness factors are explicit inputs, not hidden code-prescribed values.
    """
    x, y = np.asarray(x_in, dtype=float), np.asarray(y_in, dtype=float)
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or
           np.any(np.diff(a) <= 0) for a in (x, y)):
        raise ValueError("x_in and y_in must be strictly increasing finite coordinates")
    t = _finite(thickness_in, "thickness_in", True)
    e = _finite(elastic_modulus_ksi, "elastic_modulus_ksi", True)
    nu = _finite(poisson, "poisson")
    sf = _finite(membrane_stiffness_factor, "membrane_stiffness_factor", True)
    if not -1 < nu < 0.5:
        raise ValueError("poisson must be between -1 and 0.5")
    nodes = np.array([(a, b) for b in y for a in x])
    nx = len(x)
    cells = [[j*nx+i, j*nx+i+1, (j+1)*nx+i+1, (j+1)*nx+i]
             for j in range(len(y)-1) for i in range(nx-1)]
    seen, web = set(), []
    for b in beams:
        row = dict(b)
        ident = str(row["id"])
        i, j = int(row["i"]), int(row["j"])
        if ident in seen or not (0 <= i < len(nodes) and 0 <= j < len(nodes)):
            raise ValueError("beam IDs must be unique and node indices in range")
        seen.add(ident)
        dx = nodes[j] - nodes[i]
        adjacent = (abs(i-j) == 1 and i//nx == j//nx) or abs(i-j) == nx
        if not adjacent or np.count_nonzero(np.abs(dx) > 1e-12) != 1:
            raise ValueError("each beam web must join adjacent aligned mesh nodes")
        row.update(id=ident, i=i, j=j,
                   area_web_in2=_finite(row["area_web_in2"], "area_web_in2", True),
                   E_ksi=_finite(row.get("E_ksi", e), "beam E_ksi", True),
                   stiffness_factor=_finite(row.get("stiffness_factor", 1.0), "beam stiffness_factor", True))
        web.append(row)
    return {"nodes": nodes.tolist(), "cells": cells, "x_in": x.tolist(), "y_in": y.tolist(),
            "thickness_in": t, "E_ksi": e, "poisson": nu,
            "membrane_stiffness_factor": sf, "beams": web}


def _q4(nodes, xi, eta):
    deriv = np.array([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
                      [(1+eta), (1+xi)], [-(1+eta), (1-xi)]]) / 4.0
    jac = nodes.T @ deriv
    det = np.linalg.det(jac)
    if det <= 0:
        raise ValueError("Q4 element has nonpositive Jacobian")
    grad = deriv @ np.linalg.inv(jac)
    b = np.zeros((3, 8))
    b[0, 0::2], b[1, 1::2] = grad[:, 0], grad[:, 1]
    b[2, 0::2], b[2, 1::2] = grad[:, 1], grad[:, 0]
    shape = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                      (1+xi)*(1+eta), (1-xi)*(1+eta)]) / 4.0
    return b, det, shape


def _constitutive(mesh):
    e, nu = mesh["E_ksi"] * mesh["membrane_stiffness_factor"], mesh["poisson"]
    return e / (1-nu*nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])


def _assemble(mesh):
    nodes, d = np.asarray(mesh["nodes"]), _constitutive(mesh)
    rows, cols, vals, elements = [], [], [], []
    for cell in mesh["cells"]:
        xy = nodes[cell]
        ke = np.zeros((8, 8))
        for xi in _GAUSS:
            for eta in _GAUSS:
                b, det, _ = _q4(xy, xi, eta)
                ke += b.T @ d @ b * mesh["thickness_in"] * det
        dofs = np.array([[2*i, 2*i+1] for i in cell]).ravel()
        elements.append((dofs, ke, "slab", cell))
    for row in mesh["beams"]:
        i, j = row["i"], row["j"]
        delta = nodes[j] - nodes[i]
        length = np.linalg.norm(delta)
        direction = delta / length
        a = np.r_[-direction, direction]
        ke = row["E_ksi"] * row["stiffness_factor"] * row["area_web_in2"] / length * np.outer(a, a)
        elements.append((np.array([2*i, 2*i+1, 2*j, 2*j+1]), ke, "beam", row))
    for dofs, ke, _, _ in elements:
        rows.extend(np.repeat(dofs, len(dofs)))
        cols.extend(np.tile(dofs, len(dofs)))
        vals.extend(ke.ravel())
    k = coo_matrix((vals, (rows, cols)), shape=(2*len(nodes), 2*len(nodes))).tocsr()
    return k, elements


def resultant(nodes, forces, origin=(0.0, 0.0)):
    xy, f = np.asarray(nodes, float) - np.asarray(origin, float), np.asarray(forces, float)
    return np.array([f[:, 0].sum(), f[:, 1].sum(),
                     (xy[:, 0]*f[:, 1] - xy[:, 1]*f[:, 0]).sum()])


def inertial_nodal_forces(mesh, weights, force_kip=(0.0, 0.0), torsion_kip_in=0.0):
    """Lumped inertia with its exact resultant at the weight centroid.

    A polar-weighted tangential couple gives prescribed torsion, with zero net
    translation. Weights must be actual tributary seismic mass/weight shares.
    Torsion is an independent input from the final ASCE torsion analysis.
    """
    xy = np.asarray(mesh["nodes"])
    w = np.asarray(weights, float)
    if w.shape != (len(xy),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be nonnegative finite nodal weights with positive sum")
    w = w / w.sum()
    f0 = np.asarray(force_kip, float)
    if f0.shape != (2,) or not np.isfinite(f0).all():
        raise ValueError("force_kip must be a finite two-vector")
    center = w @ xy
    r = xy - center
    polar = np.sum(w * np.sum(r*r, axis=1))
    torque = _finite(torsion_kip_in, "torsion_kip_in")
    if polar <= 0 and torque:
        raise ValueError("nonzero torsion requires a finite mass footprint")
    f = w[:, None] * f0
    if torque:
        f += torque / polar * w[:, None] * np.c_[-r[:, 1], r[:, 0]]
    return f.tolist()


def footprint_traction(mesh, node_weights, force_kip, *, application_xy_in,
                       moment_kip_in=0.0, centroid_tolerance_in=1e-8,
                       allow_reference_transport=False):
    """Spread one column-on-floor force over a declared physical footprint.

    node_weights maps node index to nonnegative tributary fractions. The weighted
    centroid must equal the actual column application point by default.
    ``allow_reference_transport=True`` explicitly permits an asymmetric edge/
    corner footprint: transport the supplied wrench to its actual centroid,
    M_centroid=M_application+(application-centroid) cross F. The resulting
    affine traction may change sign; it is not a uniform pressure assertion.
    A supplied column torsional moment is spread as a zero-resultant polar
    couple. Physical footprint size and transport policy are caller evidence.
    """
    w = np.zeros(len(mesh["nodes"]))
    for index, value in node_weights.items():
        i = int(index)
        if not 0 <= i < len(w):
            raise ValueError("footprint node outside mesh")
        w[i] += _finite(value, "footprint weight")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("footprint weights must be nonnegative with positive sum")
    actual = w @ np.asarray(mesh["nodes"]) / w.sum()
    application = np.asarray(application_xy_in, float)
    if application.shape != (2,) or not np.isfinite(application).all():
        raise ValueError('physical force application point must contain two finite coordinates')
    if np.linalg.norm(actual - application) > centroid_tolerance_in and not allow_reference_transport:
        raise ValueError("traction footprint centroid differs from physical application point")
    force = np.asarray(force_kip, float)
    if force.shape != (2,) or not np.isfinite(force).all():
        raise ValueError('physical force must contain two finite components')
    arm = application-actual
    transported = moment_kip_in + arm[0]*force[1]-arm[1]*force[0]
    return inertial_nodal_forces(mesh, w, force, transported)


def _component(mesh, component, tolerance):
    n = len(mesh["nodes"])
    required = ("inertia", "columns_above_on_floor", "columns_below_on_floor")
    arrays = {key: _vector(component[key], n, key) for key in required}
    role = component.get("role")
    if role not in ("local_inertia", "transfer") or not component.get("provenance"):
        raise ValueError("component needs role local_inertia/transfer and decomposition provenance")
    if role == "transfer" and np.max(np.abs(arrays["inertia"])) > tolerance:
        raise ValueError("transfer influence must have no local floor inertia")
    total = sum(arrays.values())
    xy = np.asarray(mesh["nodes"])
    length = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    scale = max(sum(np.abs(a).sum() for a in arrays.values()), 1.0)
    balance = resultant(xy, total, xy.mean(axis=0))
    normalized = max(abs(balance[0]), abs(balance[1]), abs(balance[2])/length) / scale
    if normalized > tolerance:
        raise ValueError(f"component {component.get('id')!r} is not an equilibrated floor free body: {balance.tolist()}")
    return total, {"id": str(component["id"]), "role": role,
                   "resultant_error": balance.tolist(), "normalized_balance_error": float(normalized),
                   "equilibrium_tolerance": float(tolerance), "force_normalization_kip": float(scale),
                   "moment_normalization_length_in": float(length),
                   "resultant_reference_xy_in": xy.mean(axis=0).tolist(),
                   "provenance": str(component["provenance"])}


def conventional_design_cases(*, local_component_ids, transfer_component_ids,
                              elf_floor_force_kip, fpx_eq_kip, fpx_min_kip,
                              fpx_max_kip, omega0, transfer_factor=1.0,
                              irregularity_factor=1.0):
    """Separate ASCE 7-22 conventional diaphragm/collector seismic action cases.

    local components are summed unit-floor-inertia influence responses scaled
    to the actual ELF floor force (including its torsion). Transfer components
    are the other-floor influence responses, whose LOCAL inertia is zero.
    Pass independently identified components, NEVER net total nodal reactions.
    Returned cases still need the governing gravity/orthogonal combinations.

    Collector 12.10.2.1(b) uses Eq.(12.10-1), not its optional (12.10-3) cap;
    (c) uses the (12.10-2) minimum without Omega0. Keep all signed cases before
    taking capacities/DCR envelopes, because transfer can oppose local inertia.
    transfer_factor accounts for 12.10.1.1 horizontal Type4 requirements;
    irregularity_factor accounts for applicable 12.3.3.5 (caller supplies basis).
    No alternative 12.10.3 factor or strength-limited exception is assumed.
    """
    local, transfer = list(local_component_ids), list(transfer_component_ids)
    if not local or len(set(local+transfer)) != len(local+transfer):
        raise ValueError("distinct local and transfer component identities are required")
    elf = _finite(elf_floor_force_kip, "elf_floor_force_kip", True)
    eq = _finite(fpx_eq_kip, "fpx_eq_kip", True)
    low = _finite(fpx_min_kip, "fpx_min_kip", True)
    high = _finite(fpx_max_kip, "fpx_max_kip", True)
    om, tf, ir = (_finite(v, key, True) for v, key in
                  ((omega0, "omega0"), (transfer_factor, "transfer_factor"),
                   (irregularity_factor, "irregularity_factor")))
    if high < low or min(om, tf, ir) < 1:
        raise ValueError("force bounds and amplification factors are invalid")
    fpx = min(max(eq, low), high)
    def case(name, inertial_scale, transfer_scale, scope, clause):
        return {"id": name, "scope": scope, "basis": clause,
                "factors": {**{i: inertial_scale for i in local},
                            **{i: transfer_scale for i in transfer}}}
    return [case("diaphragm_structural_analysis", ir, ir*tf, "diaphragm", "12.10.1.1 structural analysis"),
            case("diaphragm_fpx", ir*fpx/elf, ir*tf, "diaphragm", "12.10.1.1 Eq.12.10-1/2/3 plus transfer"),
            case("collector_elf_omega", om, max(om, tf), "collector", "12.10.2.1(a)"),
            case("collector_eq_omega", om*eq/elf, max(om, tf), "collector", "12.10.2.1(b)"),
            case("collector_minimum", ir*low/elf, ir*tf, "collector", "12.10.2.1(c); 12.3.3.5 non-overstrength collector amplification")]


def _identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class PreparedDiaphragmOperator:
    """One actual mesh/material operator and LU for many signed load families.

    No global cache or cross-candidate lookup. The caller owns the lifetime.
    Exact mesh/material/source hashes are rechecked before every solve. Cached
    Q4 strain maps are geometric operators, not reused response/force fields.
    """
    def __init__(self,mesh):
        self.mesh=copy.deepcopy(mesh);self.mesh_sha256=_identity(mesh)
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.k,self.elements=_assemble(self.mesh);xy=np.asarray(self.mesh['nodes'],float);n=len(xy)
        centered=xy-xy.mean(axis=0);length=max(np.ptp(xy[:,0]),np.ptp(xy[:,1]),1.)
        self.g=np.zeros((2*n,3));self.g[0::2,0]=self.g[1::2,1]=1/math.sqrt(n)
        self.g[0::2,2],self.g[1::2,2]=-centered[:,1]/length,centered[:,0]/length
        self.g[:,2]/=np.linalg.norm(self.g[:,2]);self.scale=float(np.max(self.k.diagonal()))
        self.aug=bmat([[self.k/self.scale,csr_matrix(self.g)],[csr_matrix(self.g.T),csr_matrix((3,3))]],format='csc')
        try:self.factorization=splu(self.aug)
        except RuntimeError as error:raise ValueError('membrane stiffness is singular') from error
        self.d=_constitutive(self.mesh);self.local_points=[(a,b) for a in _GAUSS for b in _GAUSS]+[(0.,0.)]
        self.cell_dofs=np.array([[[2*i,2*i+1] for i in cell] for cell in self.mesh['cells']]).reshape(-1,8)
        self.sample_b=[];self.sample_xy=[]
        for cell in self.mesh['cells']:
            maps=[];points=[]
            for xi,eta in self.local_points:
                b,_,shape=_q4(xy[cell],xi,eta);maps.append(b);points.append(shape@xy[cell])
            self.sample_b.append(maps);self.sample_xy.append(points)
        self.sample_b=np.asarray(self.sample_b);self.sample_xy=np.asarray(self.sample_xy)
        self.solve_calls=0;self.case_rhs_solved=0

    def validate(self,mesh):
        if (_identity(mesh)!=self.mesh_sha256 or _identity(self.mesh)!=self.mesh_sha256 or
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=self.source_sha256):
            raise ValueError('Prepared diaphragm operator mesh/material/source is stale')

    def record(self):
        return dict(method='explicit_frozen_Q4_web_operator_reuse_v1',mesh_sha256=self.mesh_sha256,source_sha256=self.source_sha256,
            assemblies=1,factorizations=1,solve_calls=self.solve_calls,case_rhs_solved=self.case_rhs_solved)

    def solve(self,components,cases,*,cuts=(),equilibrium_tolerance=1e-8,retain_slab_samples=True):
        return solve_diaphragm_actions(self.mesh,components,cases,cuts=cuts,equilibrium_tolerance=equilibrium_tolerance,
            operator=self,retain_slab_samples=retain_slab_samples)


def solve_diaphragm_actions(mesh, components, cases, *, cuts=(), equilibrium_tolerance=1e-8,
                            operator=None,retain_slab_samples=True):
    """Solve signed components/cases; return action records and equilibrium proof.

    Each component has id, role, provenance and all three explicit (nnode,2)
    arrays described above. Case factors map component IDs to signed multipliers.
    Three rigid-motion gauge constraints carry no physical force. No output flag
    here claims mesh convergence or code strength: use compare_mesh_actions at
    common physical samples and separately check member/support cut resultants.
    """
    if operator is None:operator=PreparedDiaphragmOperator(mesh)
    if not isinstance(operator,PreparedDiaphragmOperator):raise ValueError('Explicit prepared physical diaphragm operator required')
    operator.validate(mesh);k,elements=operator.k,operator.elements
    xy = np.asarray(mesh["nodes"], float)
    n = len(xy)
    vectors, audits = {}, []
    for row in components:
        ident = str(row["id"])
        if ident in vectors:
            raise ValueError("duplicate component identity")
        vectors[ident], audit = _component(mesh, row, equilibrium_tolerance)
        audits.append(audit)
    if not vectors:
        raise ValueError("at least one independently balanced component is required")
    g,scale,aug,factorization,d=operator.g,operator.scale,operator.aug,operator.factorization,operator.d
    outputs, used_ids = [], set()
    for case in cases:
        ident = str(case["id"])
        if ident in used_ids or not case.get("factors"):
            raise ValueError("cases require unique IDs and nonempty component factors")
        used_ids.add(ident)
        force = np.zeros((n, 2))
        for key, factor in case["factors"].items():
            if key not in vectors:
                raise ValueError(f"unknown component {key}")
            force += _finite(factor, "case factor") * vectors[key]
        rhs = np.r_[force.ravel()/scale, [0, 0, 0]]
        answer = factorization.solve(rhs)
        operator.case_rhs_solved+=1
        if not np.isfinite(answer).all():
            raise ValueError("membrane solve is singular or nonfinite")
        u = answer[:2*n]
        residual = k @ u - force.ravel()
        residual_rel = np.max(np.abs(residual)) / max(np.max(np.abs(force)), 1.0)
        initial_residual_rel = float(residual_rel)
        corrections = 0
        # Highly graded physical section meshes can expose LU roundoff even
        # when the scaled augmented residual is near machine precision.
        # Correct against the SAME matrix and factor; the original physical
        # equilibrium criterion remains the acceptance guard below.
        while residual_rel > equilibrium_tolerance and corrections < 3:
            answer += factorization.solve(rhs - aug @ answer)
            corrections += 1
            if not np.isfinite(answer).all():
                raise ValueError("membrane refinement produced a nonfinite solution")
            u = answer[:2*n]
            residual = k @ u - force.ravel()
            residual_rel = np.max(np.abs(residual)) / max(np.max(np.abs(force)), 1.0)
        if residual_rel > equilibrium_tolerance:
            raise ValueError("gauge reactions/non-equilibrium exceed tolerance")
        samples, beam_results = [], []
        if retain_slab_samples:
            strain=np.einsum('cpij,cj->cpi',operator.sample_b,u[operator.cell_dofs])
            sample_actions=mesh['thickness_in']*np.einsum('ij,cpj->cpi',d,strain)
            for cellid in range(len(mesh['cells'])):
                for local,(xi,eta) in enumerate(operator.local_points):
                    action=sample_actions[cellid,local]
                    samples.append({'cell':cellid,'xy_in':operator.sample_xy[cellid,local].tolist(),
                        'local':[xi,eta],'Nxx_kip_per_in':float(action[0]),'Nyy_kip_per_in':float(action[1]),'Nxy_kip_per_in':float(action[2])})
        for row in mesh["beams"]:
            i, j = row["i"], row["j"]
            delta = xy[j]-xy[i]
            span = np.linalg.norm(delta)
            axial = row["E_ksi"]*row["stiffness_factor"]*row["area_web_in2"]/span * np.dot(delta/span, u[2*j:2*j+2]-u[2*i:2*i+2])
            beam_results.append({"id": row["id"], "member_id": row.get("member_id", row["id"]),
                                 "i": i, "j": j, "N_kip": float(axial)})
        cut_results = []
        for cut in cuts:
            axis = {"x": 0, "y": 1}.get(cut["axis"])
            if axis is None:
                raise ValueError("cut axis must be x or y")
            at = float(cut["coordinate_in"])
            if not xy[:, axis].min() < at < xy[:, axis].max():
                raise ValueError("cut must be inside the floor")
            left = xy[:, axis] < at
            crossing = np.zeros((n, 2))
            for dofs, ke, _, _ in elements:
                idx = dofs[0::2]//2
                mask = left[idx]
                if np.any(mask) and not np.all(mask):
                    internal = (ke @ u[dofs]).reshape(-1, 2)
                    np.add.at(crossing, idx[mask], -internal[mask])
            ext = resultant(xy[left], force[left])
            traction = resultant(xy, crossing)
            cut_results.append({"axis": cut["axis"], "coordinate_in": at,
                                "kind": "algebraic_nodal_partition_not_pointwise_stress_cut",
                                "applied_left_resultant": ext.tolist(),
                                "interface_resultant_on_left": traction.tolist(),
                                "balance_error": (ext+traction).tolist()})
        outputs.append({"id": ident, "scope": case.get("scope"), "factors": dict(case["factors"]),
                        "displacements_in": u.reshape(n, 2).tolist(), "slab_samples": samples,
                        "slab_samples_omitted":not retain_slab_samples,
                        "beam_axial": beam_results, "cuts": cut_results,
                        "force_moment_balance": resultant(xy, force).tolist(),
                        "force_moment_tolerance": [
                            sum(abs(case['factors'].get(a['id'], 0.))*a['equilibrium_tolerance']*a['force_normalization_kip']
                                * (1. if j < 2 else a['moment_normalization_length_in']
                                   + abs(a['resultant_reference_xy_in'][0])+abs(a['resultant_reference_xy_in'][1]))
                                for a in audits) for j in range(3)],
                        "force_moment_tolerance_basis": "Triangle bound of original signed component equilibrium guards, transported from mesh mean to global origin",
                        "maximum_nodal_residual_kip": float(np.max(np.abs(residual))),
                        "normalized_nodal_residual": float(residual_rel),
                        "initial_normalized_nodal_residual": initial_residual_rel,
                        "iterative_refinement_corrections": corrections,
                        "gauge_reaction_norm_kip": float(np.linalg.norm(scale*g @ answer[2*n:]))})
    if not outputs:
        raise ValueError("at least one load case is required")
    operator.solve_calls+=1
    return {"method": METHOD_VERSION, "mesh": mesh, "component_audit": audits,
            "prepared_operator":operator.record(),
            "cases": outputs, "strength_verified": False, "mesh_convergence_verified": False,
            "scope": "prescribed physical frame tractions; Q4 membrane plus disjoint axial web stiffness"}


def sample_membrane_action(result, case_id, x_in, y_in):
    """Evaluate signed N at a fixed physical point; reject mesh-line ambiguity."""
    mesh = result["mesh"]
    x, y = np.asarray(mesh["x_in"]), np.asarray(mesh["y_in"])
    px, py = float(x_in), float(y_in)
    if not x[0] < px < x[-1] or not y[0] < py < y[-1]:
        raise ValueError("sample point must lie strictly inside the floor")
    if min(np.abs(x-px)) < 1e-9 or min(np.abs(y-py)) < 1e-9:
        raise ValueError("sample point on mesh interface needs explicit side ownership")
    i, j = np.searchsorted(x, px)-1, np.searchsorted(y, py)-1
    cell = mesh["cells"][j*(len(x)-1)+i]
    xi, eta = 2*(px-x[i])/(x[i+1]-x[i])-1, 2*(py-y[j])/(y[j+1]-y[j])-1
    b, _, _ = _q4(np.asarray(mesh["nodes"])[cell], xi, eta)
    case = next(c for c in result["cases"] if c["id"] == case_id)
    u = np.asarray(case["displacements_in"])[cell].ravel()
    return (mesh["thickness_in"]*_constitutive(mesh) @ b @ u).tolist()


def compare_mesh_actions(coarse, fine, samples, *, relative_tolerance=0.05,
                         absolute_tolerance_kip_per_in=1e-5):
    """Compare signed components at common physical samples away from singularities.

    This witness checks the specified samples only, not every floor/candidate.
    samples rows have case_id, x_in,y_in and optional id. At least one is required.
    It does not replace physical face/cut/beam axial and footprint witnesses.
    """
    if not samples:
        raise ValueError("mesh comparison needs common physical samples")
    rt = _finite(relative_tolerance, "relative_tolerance", True)
    at = _finite(absolute_tolerance_kip_per_in, "absolute_tolerance", True)
    rows = []
    for p in samples:
        a = np.array(sample_membrane_action(coarse, p["case_id"], p["x_in"], p["y_in"]))
        b = np.array(sample_membrane_action(fine, p["case_id"], p["x_in"], p["y_in"]))
        allowed = at + rt*np.maximum(np.abs(a), np.abs(b))
        rows.append({"id": p.get("id"), "coarse": a.tolist(), "fine": b.tolist(),
                     "delta": (b-a).tolist(), "allowed": allowed.tolist(),
                     "passed": bool(np.all(np.abs(b-a) <= allowed))})
    return {"criterion": "componentwise absolute+relative at identical physical positions",
            "relative_tolerance": rt, "absolute_tolerance_kip_per_in": at,
            "samples": rows, "passed": all(r["passed"] for r in rows),
            "scope": "specified samples only; does not assert domain-wide applicability"}
