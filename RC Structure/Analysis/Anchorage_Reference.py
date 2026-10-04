"""Section-equilibrium anchorage reference for a beam face (joint-spring review 2026-09-29; repaired
after the anchorage follow-up review of 2026-09-30).

The review amended the calibration target to "a section-equilibrium anchorage reference, followed by a
reduced interface fitted jointly with the member". This module builds the reference and stops there:
nothing is fitted and nothing is installed in the frame.

For one beam section and one bending direction (hogging: top bars in tension; sagging: bottom bars in
tension) it derives, at every moment level of a curvature-controlled section analysis:
  1. the section state with every bar layer kept by a stable identity: neutral-axis depth c, each layer's
     strain and stress from the chosen steel law, the extreme concrete strain;
  2. for each tension layer, the anchorage state of its bars through the joint: the bond lengths needed on
     the near side and on the far side, where the far-face stress of the SAME layer comes from the opposite
     face's section at a declared companion moment; a zero-stress zone when both fit inside the column
     depth, a fully mobilised two-face tension profile when the far face is also in tension, and an
     explicit "capacity exceeded" state when a compressive far face cannot be developed (the near-face slip
     is then a screening estimate, marked invalid);
  3. each layer's slip at the face from bond compatibility with two strain-profile treatments reported side
     by side: Sezen and Setzler's bilinear approximation over the yielded length and the direct integration
     of the inverse steel law along the uniform-bond stress gradient (the plateau strains are skipped by the
     inverse law, as its authors' hardening branch implies); the Sezen form is the declared default;
  4. one face rotation from compatibility: the area-weighted least-squares plane rotation about the neutral
     axis that best fits the tension layers' slips, with each layer's own implied rotation and residual.

Events are found by an individual layer's strain crossing (yield strain, hardening strain) with linear
interpolation inside the curvature grid interval; they are reported separately as first-layer yield,
all-tension-layers yielded, hardening onset, the ACI nominal strength Mn (0.85 f'c block, eps_cu 0.003, no
hardening) and the end of the curve with its limit; an absent event is None, never the last row.

Declared limits (also written into the output): the bond model is a development-length screen with uniform
bond, not a solved through-bar boundary-value problem; the companion rule is an assumption, not joint
equilibrium; concrete is unconfined; slab bars, compression-bar slip and cyclic bond evolution are absent;
`frame_section` is the bare x-beam of Structure_Parameters, not the slab-aware per-end member My. Units are
whatever the inputs use, consistently; the Sezen bond values are converted from their psi form.

usage: python Analysis/Anchorage_Reference.py --out <dir> [--root <design root> --case <case>] [--unit1]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

VERSION = "anchorage_reference_v3.1_provisional"
STATUSES = ("valid", "invalid", "unsupported", "not_applicable")      # 'valid' (bool) is False only for invalid and unsupported
PSI_PER_KSI, PSI_PER_MPA = 1000.0, 145.0377
STRAIN_TOL = 1e-9
ASSUMPTIONS = {
    "anchorage_model": "uniform bond (Sezen and Setzler 2008): u_e while the bar is elastic, u_y over its yielded length; a development-length "
                       "screen against the column depth, not a solved through-bar boundary-value problem; the near-face slip does not depend on "
                       "the available length unless both faces are in tension",
    "two_face_interaction": "the same layer's stress at the far face comes from the opposite direction's section at the companion moment; a "
                            "compressive far face needs its own elastic length; a tensile far face whose zero-stress zone does not fit makes a fully "
                            "mobilised profile with a minimum stress inside the joint; a compressive far face that cannot be developed within the "
                            "column depth is 'capacity exceeded' and its row is marked invalid",
    "two_face_domain": "the fully mobilised profile (yielded bond over the near yielded length, elastic bond on both sides of the interior minimum) "
                       "is accepted only inside its domain: both segments nonnegative and summing to the available length, the yielded length within "
                       "the near segment, the interior minimum between zero and both face stresses; a far-face tension too small to be reached by "
                       "bond within the joint (including a zero far stress) is 'capacity exceeded' (status invalid, required-length screen retained); "
                       "a far face yielded in tension, or a far face carrying more tension than the near face, is 'unsupported' (no profile derived); "
                       "'anchorage_ratio' is 1 by construction in the fully mobilised state and 'required_length_ratio' (zero-stress-zone requirement "
                       "over the available length) is reported beside it; states outside the domain are never clamped into acceptance",
    "companion_rule": "the opposite face carries the same fraction of its own first-layer-yield moment as the near face, capped at its curve; an "
                      "assumption, not joint equilibrium; the far-unstressed alternative is reported alongside",
    "strain_profile": "reported both ways: 'sezen_bilinear' (average of the face strain and the yield strain over the yielded length; declared "
                      "default) and 'inverse_law' (strain from the inverse of the hardening branch integrated along the stress gradient; plateau "
                      "strains between eps_y and eps_sh are skipped, as the hardening branch implies)",
    "face_rotation": "area-weighted least-squares plane rotation about the neutral axis fitted to the tension layers' slips; each layer's implied "
                     "rotation and residual are reported; compression-side slip is neglected",
    "section": "cracked section, unconfined concrete law, no slab bars; frame_section() is the bare x-beam of Structure_Parameters",
    "status_contract": "every layer result carries the same record (stresses, three lengths, anchorage_ratio, required_length_ratio, both slips, state, "
                       "status, valid) in every branch; status is 'valid', 'invalid', 'unsupported' or 'not_applicable'; 'not_applicable' is a layer of "
                       "the nominal tension group whose near face carries no tension: zero anchorage demand and zero slip by definition, which is "
                       "bookkeeping and not a compression-bar slip model; such a layer does not invalidate its row and is not in the rotation fit",
    "not_covered": "cyclic rule, bond deterioration, compression-bar slip, confinement, the installed hinge's per-end slab-aware My",
}


@dataclass
class Steel:
    fy: float
    es: float
    fu: float
    eps_sh: float
    esh: float
    eps_su: float
    status: str = "provisional"

    @property
    def eps_y(self):
        return self.fy / self.es

    @property
    def p(self):
        return self.esh * (self.eps_su - self.eps_sh) / max(self.fu - self.fy, 1e-9)

    def stress(self, eps):
        """Elastic - plateau - Chang and Mander hardening to (eps_su, fu); symmetric in compression."""
        s = np.sign(eps); e = np.abs(eps)
        out = np.where(e <= self.eps_y, self.es * e, self.fy)
        hard = self.fu + (self.fy - self.fu) * np.clip((self.eps_su - e) / (self.eps_su - self.eps_sh), 0.0, 1.0) ** self.p
        out = np.where(e > self.eps_sh, hard, out)
        return s * out

    def strain_on_hardening_branch(self, sigma):
        """Inverse of the hardening branch for fy < sigma <= fu (eps between eps_sh and eps_su)."""
        sigma = np.clip(sigma, self.fy, self.fu)
        return self.eps_su - (self.eps_su - self.eps_sh) * ((self.fu - sigma) / (self.fu - self.fy)) ** (1.0 / self.p)


@dataclass
class Concrete:
    fc: float
    eps0: float = 0.002
    eps_cu: float = 0.004
    residual: float = 0.2
    status: str = "unconfined Concrete01-type law (parabola to eps0, linear to residual x f'c at eps_cu); confinement not modelled"

    def stress(self, eps):
        """Compression positive here; tension carries nothing (cracked section)."""
        e = np.asarray(eps, dtype=float)
        out = np.zeros_like(e)
        rising = (e > 0) & (e <= self.eps0)
        out[rising] = self.fc * (2 * e[rising] / self.eps0 - (e[rising] / self.eps0) ** 2)
        falling = (e > self.eps0) & (e <= self.eps_cu)
        out[falling] = self.fc * (1.0 - (1.0 - self.residual) * (e[falling] - self.eps0) / (self.eps_cu - self.eps0))
        out[e > self.eps_cu] = self.fc * self.residual
        return out


@dataclass
class Bond:
    """Uniform bond stresses of the Sezen and Setzler (2008) model in the stress unit of the section."""
    u_elastic: float
    u_yielded: float
    basis: str

    @classmethod
    def sezen(cls, fc_anchorage, stress_unit):
        fc_psi = fc_anchorage * (PSI_PER_KSI if stress_unit == "ksi" else PSI_PER_MPA if stress_unit == "MPa" else 1.0)
        to_unit = (1.0 / PSI_PER_KSI) if stress_unit == "ksi" else (1.0 / PSI_PER_MPA) if stress_unit == "MPa" else 1.0
        return cls(12.0 * math.sqrt(fc_psi) * to_unit, 6.0 * math.sqrt(fc_psi) * to_unit,
                   "Sezen and Setzler 2008: u_e = 12 sqrt(f'c) psi (elastic bar), u_y = 6 sqrt(f'c) psi (yielded bar); provisional")


@dataclass
class Section:
    b: float
    h: float
    layers: list                      # [{"id", "depth_from_top", "area", "bars", "db", "group": "top" | "bottom"}]
    concrete: Concrete
    steel: Steel
    anchorage_length: float           # the through-bar's length inside the joint (column depth)
    fc_anchorage: float
    label: str = ""
    units: dict = field(default_factory=lambda: {"force": "kip", "length": "in", "stress": "ksi"})

    def __post_init__(self):
        # stable identities from geometry, independent of the order the layers were listed in
        for group in ("top", "bottom"):
            members = sorted((ly for ly in self.layers if ly["group"] == group), key=lambda ly: ly["depth_from_top"] if group == "top" else -ly["depth_from_top"])
            for k, ly in enumerate(members, start=1):
                ly.setdefault("id", f"{group}{k}")
        self.layers = sorted(self.layers, key=lambda ly: ly["depth_from_top"])

    def fc_psi_beam(self):
        return _fc_psi(self.concrete.fc, self.units["stress"])


def _fc_psi(fc, stress_unit):
    return fc * (PSI_PER_KSI if stress_unit == "ksi" else PSI_PER_MPA if stress_unit == "MPa" else 1.0)


# ---- section analysis -------------------------------------------------------------------------------
def moment_curvature(sec, direction, n_strips=200, phi_max=None):
    """Curvature-controlled analysis of the cracked section under pure bending. ``direction`` "hogging"
    puts the top of the section in tension (analysed upside down). Each row keeps every layer's state under
    its stable id. The curvature sequence is geometric: fine well below first yield, continuing until the
    concrete or steel limit."""
    flip = direction == "hogging"
    layers = [{**ly, "y": (sec.h - ly["depth_from_top"]) if flip else ly["depth_from_top"]} for ly in sec.layers]   # y from the compression face
    tension_group = "top" if flip else "bottom"
    t_area = sum(ly["area"] for ly in layers if ly["group"] == tension_group)
    d_t = sum(ly["y"] * ly["area"] for ly in layers if ly["group"] == tension_group) / t_area
    ys = (np.arange(n_strips) + 0.5) / n_strips * sec.h
    dA = sec.b * sec.h / n_strips
    phi_0 = 0.02 * sec.steel.eps_y / max(sec.h - d_t, 1e-9)
    phi_cap = phi_max or 2.5 * sec.steel.eps_su / max(sec.h - d_t, 1e-9)
    phis = phi_0 * (1.015 ** np.arange(0, 2000))
    phis = phis[phis <= phi_cap]
    rows, limit = [], None
    for phi in phis:
        def resultant(c_try):
            eps_c = phi * (c_try - ys)
            fc = sec.concrete.stress(np.clip(eps_c, 0.0, None)).sum() * dA
            fs = sum(float(sec.steel.stress(phi * (ly["y"] - c_try))) * ly["area"] for ly in layers)
            return fc - fs
        lo, hi = 1e-6, sec.h
        f_lo, f_hi = resultant(lo), resultant(hi)
        if f_lo * f_hi > 0:
            limit = "no equilibrium root"
            break
        for _ in range(60):
            mid = 0.5 * (lo + hi); f_mid = resultant(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        c = 0.5 * (lo + hi)
        eps_c = phi * (c - ys)
        fconc = sec.concrete.stress(np.clip(eps_c, 0.0, None)) * dA
        m = float((fconc * (c - ys)).sum())
        states = {}
        for ly in layers:
            eps_s = float(phi * (ly["y"] - c)); sig = float(sec.steel.stress(eps_s))
            m += sig * ly["area"] * (ly["y"] - c)
            states[ly["id"]] = {"id": ly["id"], "group": ly["group"], "y": ly["y"], "depth_from_top": ly["depth_from_top"], "area": ly["area"],
                                "bars": ly["bars"], "db": ly["db"], "eps": eps_s, "sigma": sig, "force": sig * ly["area"]}
        eps_top = float(phi * c)
        eps_t_max = max(st["eps"] for st in states.values() if st["group"] == tension_group)
        rows.append({"phi": float(phi), "c": float(c), "M": m, "eps_c_top": eps_top, "eps_t_max": eps_t_max, "d_t": float(d_t), "layers": states})
        if eps_top >= sec.concrete.eps_cu:
            limit = "concrete eps_cu"; break
        if eps_t_max >= sec.steel.eps_su:
            limit = "steel eps_su"; break
    return rows, tension_group, d_t, limit or "curvature cap"


def aci_nominal(sec, direction):
    """ACI nominal strength: 0.85 f'c block, eps_cu 0.003, elastic-perfectly-plastic steel, iteration on c."""
    flip = direction == "hogging"
    layers = [{**ly, "y": (sec.h - ly["depth_from_top"]) if flip else ly["depth_from_top"]} for ly in sec.layers]
    beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (sec.fc_psi_beam() - 4000.0) / 1000.0))
    fy, es = sec.steel.fy, sec.steel.es

    def forces(c):
        cc = 0.85 * sec.concrete.fc * sec.b * beta1 * c
        fs = [max(-fy, min(fy, es * 0.003 * (ly["y"] - c) / c)) * ly["area"] for ly in layers]
        return cc, fs
    lo, hi = 1e-6, sec.h
    for _ in range(80):
        mid = 0.5 * (lo + hi); cc, fs = forces(mid)
        if cc - sum(fs) > 0:
            hi = mid
        else:
            lo = mid
    c = 0.5 * (lo + hi); cc, fs = forces(c)
    m = cc * (c - beta1 * c / 2.0) + sum(f * (ly["y"] - c) for f, ly in zip(fs, layers))
    return {"Mn": float(m), "c": float(c), "beta1": beta1, "basis": "0.85 f'c block, eps_cu 0.003, no hardening, phi = 1 (a nominal-strength check, not an experimental quantity)"}


# ---- events ---------------------------------------------------------------------------------------
def _interpolate_row(sec, rows, i, layer_id, target_eps):
    """The section state where layer ``layer_id`` reaches ``target_eps``, linear in curvature inside the grid
    interval (rows[i-1], rows[i]); every layer's strain is interpolated and its stress re-evaluated."""
    r, prev = rows[i], rows[max(i - 1, 0)]
    e1, e0 = r["layers"][layer_id]["eps"], prev["layers"][layer_id]["eps"]
    frac = (target_eps - e0) / (e1 - e0) if e1 != e0 else 1.0
    frac = min(max(frac, 0.0), 1.0)
    out = {"phi": prev["phi"] + frac * (r["phi"] - prev["phi"]), "c": prev["c"] + frac * (r["c"] - prev["c"]),
           "M": prev["M"] + frac * (r["M"] - prev["M"]), "eps_c_top": prev["eps_c_top"] + frac * (r["eps_c_top"] - prev["eps_c_top"]),
           "d_t": r["d_t"], "row_index": i, "interpolation_fraction": frac, "layers": {}}
    for lid, st in r["layers"].items():
        eps = prev["layers"][lid]["eps"] + frac * (st["eps"] - prev["layers"][lid]["eps"])
        sig = float(sec.steel.stress(eps))
        out["layers"][lid] = {**st, "eps": float(eps), "sigma": sig, "force": sig * st["area"]}
    out["eps_t_max"] = max(st["eps"] for st in out["layers"].values() if st["group"] == r["layers"][layer_id]["group"])
    return out


def find_events(sec, rows, tension_group):
    """First-layer yield, all-tension-layers yielded and hardening onset by individual layer strain crossings
    with interpolation; None when a crossing never happens."""
    ey, esh = sec.steel.eps_y, sec.steel.eps_sh
    t_ids = [lid for lid, st in rows[0]["layers"].items() if st["group"] == tension_group]
    events = {"first_layer_yield": None, "all_tension_layers_yielded": None, "hardening_onset": None}
    for i, r in enumerate(rows):
        crossing = [lid for lid in t_ids if r["layers"][lid]["eps"] >= ey * (1.0 - STRAIN_TOL)]
        if crossing and events["first_layer_yield"] is None:
            lid = max(crossing, key=lambda l: r["layers"][l]["eps"])
            events["first_layer_yield"] = {"layer": lid, **_interpolate_row(sec, rows, i, lid, ey)}
        if len(crossing) == len(t_ids) and events["all_tension_layers_yielded"] is None:
            lid = min(t_ids, key=lambda l: r["layers"][l]["eps"])          # the last layer to arrive
            events["all_tension_layers_yielded"] = {"layer": lid, **_interpolate_row(sec, rows, i, lid, ey)}
        hardening = [lid for lid in t_ids if r["layers"][lid]["eps"] >= esh * (1.0 - STRAIN_TOL)]
        if hardening and events["hardening_onset"] is None:
            lid = max(hardening, key=lambda l: r["layers"][l]["eps"])
            events["hardening_onset"] = {"layer": lid, **_interpolate_row(sec, rows, i, lid, esh)}
        if all(events.values()):
            break
    return events


# ---- anchorage of one layer ---------------------------------------------------------------------------
def _elastic_elongation(sigma_a, sigma_b, db, u, es):
    """Elongation of an elastic bar segment whose stress falls linearly from sigma_a to sigma_b at 4 u / db."""
    return (sigma_a ** 2 - sigma_b ** 2) * db / (8.0 * u * es)


def _yielded_elongation(steel, sigma_near, eps_near, db, u_y, profile):
    """Elongation over the yielded length (stress from sigma_near down to fy at 4 u_y / db)."""
    if sigma_near <= steel.fy:
        return 0.0, 0.0
    l_y = (sigma_near - steel.fy) * db / (4.0 * u_y)
    if profile == "sezen_bilinear":
        return 0.5 * (eps_near + steel.eps_y) * l_y, l_y
    sig = np.linspace(sigma_near, steel.fy, 401)
    eps = steel.strain_on_hardening_branch(sig)
    return float(np.trapezoid(eps, dx=l_y / 400.0)), l_y


def layer_anchorage(steel, bond, db, sigma_near, eps_near, sigma_far_signed, anchorage_length):
    """Bond compatibility of one layer's bars through the joint. ``sigma_far_signed`` is the same layer's
    stress at the far face (tension positive). Every branch returns the same record: the two face stresses and
    the near strain, L_near / L_yielded / L_far, anchorage_ratio, required_length_ratio, the near-face slip under
    both strain profiles, state, status (one of STATUSES) and valid (False only for 'invalid' and 'unsupported')."""
    fy, es = steel.fy, steel.es
    out = {"sigma_near": float(sigma_near), "eps_near": float(eps_near), "sigma_far": float(sigma_far_signed)}
    if sigma_near <= 0.0:
        # no tension to anchor at this face: a complete record with zero demand. Bookkeeping only; the zero slip is not a
        # compression-bar slip model (compression slip is not covered by this reference)
        out.update(L_near=0.0, L_yielded=0.0, L_far=0.0, slip_sezen_bilinear=0.0, slip_inverse_law=0.0, anchorage_ratio=0.0, required_length_ratio=0.0,
                   state="near face not in tension (no tension anchorage demand; compression-bar slip not modelled)", status="not_applicable", valid=True)
        return out
    slips = {}
    for profile in ("sezen_bilinear", "inverse_law"):
        s_y, l_y = _yielded_elongation(steel, sigma_near, eps_near, db, bond.u_yielded, profile)
        top = min(sigma_near, fy)
        s_e = _elastic_elongation(top, 0.0, db, bond.u_elastic, es)
        slips[profile] = {"s_total_to_zero": s_y + s_e, "s_y": s_y, "l_y": l_y, "l_e": top * db / (4.0 * bond.u_elastic)}
    l_y, l_e = slips["sezen_bilinear"]["l_y"], slips["sezen_bilinear"]["l_e"]
    l_near = l_y + l_e
    top = min(sigma_near, fy)
    s_zero = {p: slips[p]["s_total_to_zero"] for p in slips}      # near-face slip when the stress reaches zero inside the joint
    far_c = -sigma_far_signed                           # compression positive at the far face
    if far_c > 0.0:
        far_top = min(far_c, fy)
        l_far = far_top * db / (4.0 * bond.u_elastic) + max(far_c - fy, 0.0) * db / (4.0 * bond.u_yielded)
        ratio = (l_near + l_far) / anchorage_length
        ok = ratio <= 1.0
        out.update(L_near=l_near, L_yielded=l_y, L_far=l_far, anchorage_ratio=ratio, required_length_ratio=ratio,
                   state="zero-stress zone inside the joint" if ok else "capacity exceeded: the compressive far face cannot be developed",
                   status="valid" if ok else "invalid", valid=bool(ok),
                   slip_sezen_bilinear=s_zero["sezen_bilinear"], slip_inverse_law=s_zero["inverse_law"])
        return out
    # far face in tension or unstressed
    far_t = max(sigma_far_signed, 0.0)
    if far_t > fy:
        # the two-face formula treats the far segment as elastic; a yielded far face is outside it (no profile derived)
        l_far = fy * db / (4.0 * bond.u_elastic) + (far_t - fy) * db / (4.0 * bond.u_yielded)
        ratio = (l_near + l_far) / anchorage_length
        out.update(L_near=l_near, L_yielded=l_y, L_far=l_far, anchorage_ratio=ratio, required_length_ratio=ratio,
                   state="unsupported: far face yielded in tension (required-length screen only; no two-face profile derived)",
                   status="unsupported", valid=False, slip_sezen_bilinear=s_zero["sezen_bilinear"], slip_inverse_law=s_zero["inverse_law"])
        return out
    l_far = far_t * db / (4.0 * bond.u_elastic)
    required = (l_near + l_far) / anchorage_length      # zero-stress-zone requirement over the available length
    base = dict(L_near=l_near, L_yielded=l_y, L_far=l_far, anchorage_ratio=required, required_length_ratio=required,
                slip_sezen_bilinear=s_zero["sezen_bilinear"], slip_inverse_law=s_zero["inverse_law"])
    if required <= 1.0:
        out.update(base, state="zero-stress zone inside the joint" if far_t > 0.0 else "far face unstressed", status="valid", valid=True)
        return out
    # the zero-stress zone does not fit: the fully mobilised two-face profile (yielded bond over l_y at the near face, then elastic
    # bond at 4 u_e / db down to an interior minimum sigma_min and back up to far_t) is accepted only inside its domain
    remaining = anchorage_length - l_y
    if remaining < 0.0:
        out.update(base, state="capacity exceeded: the yielded length alone exceeds the available anchorage", status="invalid", valid=False)
        return out
    sigma_min = 0.5 * (top + far_t - 4.0 * bond.u_elastic * remaining / db)
    tol = STRAIN_TOL * fy
    if sigma_min > far_t + tol:
        # the far segment would be negative: bond cannot reduce the near-face stress to the far-face stress within the joint
        # (a zero far stress always lands here once required > 1); the required-length screen and its over-limit status are retained
        out.update(base, sigma_far_required=float(top - 4.0 * bond.u_elastic * remaining / db),
                   state="capacity exceeded: the near-face stress cannot be reduced to the far-face stress within the joint",
                   status="invalid", valid=False)
        return out
    if sigma_min > top + tol:
        # the near elastic segment would be negative: the far face carries more tension than the near face and the profile is monotone
        # toward the far face; the near face is not the loaded face this reference assumes
        out.update(base, state="unsupported: far-face tension exceeds the near face's (profile monotone toward the far face; no profile derived)",
                   status="unsupported", valid=False)
        return out
    sigma_min = min(max(sigma_min, 0.0), top, far_t)   # trims only the tolerance band; required > 1 already implies sigma_min > 0
    x_near = l_y + (top - sigma_min) * db / (4.0 * bond.u_elastic)
    x_far = (far_t - sigma_min) * db / (4.0 * bond.u_elastic)
    if not (x_near >= l_y and x_far >= 0.0 and abs(x_near + x_far - anchorage_length) <= 1e-9 * anchorage_length):
        out.update(base, state="unsupported: two-face profile outside its domain", status="unsupported", valid=False)
        return out
    out.update(L_near=x_near, L_yielded=l_y, L_far=x_far, anchorage_ratio=1.0, required_length_ratio=required, sigma_min_inside_joint=float(sigma_min),
               state="bond fully mobilised: tension at both faces", status="valid", valid=True,
               slip_sezen_bilinear=slips["sezen_bilinear"]["s_y"] + _elastic_elongation(top, sigma_min, db, bond.u_elastic, es),
               slip_inverse_law=slips["inverse_law"]["s_y"] + _elastic_elongation(top, sigma_min, db, bond.u_elastic, es))
    return out


def face_rotation(layer_results, c, profile="sezen_bilinear"):
    """One compatible face rotation: the area-weighted least-squares plane rotation about the neutral axis
    through the tension layers' slips; each layer's implied rotation and residual are kept."""
    key = f"slip_{profile}"
    pts = [(lr["area"], lr["y"] - c, lr[key]) for lr in layer_results.values() if lr["in_tension"] and (lr["y"] - c) > 0]
    if not pts:
        return {"theta_slip": None, "profile": profile, "per_layer": {}}
    num = sum(a * s * z for a, z, s in pts); den = sum(a * z * z for a, z, s in pts)
    theta = num / den if den > 0 else None
    per_layer = {}
    for lid, lr in layer_results.items():
        if lr["in_tension"] and (lr["y"] - c) > 0:
            implied = lr[key] / (lr["y"] - c)
            per_layer[lid] = {"implied_rotation": implied, "residual_slip": lr[key] - (theta or 0.0) * (lr["y"] - c)}
    return {"theta_slip": theta, "profile": profile, "per_layer": per_layer,
            "rule": "area-weighted least squares of s_k = theta (y_k - c) over the tension layers"}


# ---- the reference for one direction ----------------------------------------------------------------
def reference(sec, bond, direction, companion="same_fraction_of_first_layer_yield"):
    rows, t_group, d_t, limit = moment_curvature(sec, direction)
    opp_rows, _, _, _ = moment_curvature(sec, "sagging" if direction == "hogging" else "hogging")
    events = find_events(sec, rows, t_group)
    opp_events = find_events(sec, opp_rows, t_group if False else ("bottom" if t_group == "top" else "top"))
    my_here = events["first_layer_yield"]["M"] if events["first_layer_yield"] else None
    my_opp = opp_events["first_layer_yield"]["M"] if opp_events["first_layer_yield"] else None
    opp_m = np.array([r["M"] for r in opp_rows])

    def anchorage_at(state):
        """Per-layer anchorage and the face rotation for one section state (a row or an interpolated event)."""
        if my_here and my_opp and companion == "same_fraction_of_first_layer_yield":
            m_comp = min(state["M"] / my_here * my_opp, float(opp_m.max()))
        else:
            m_comp = 0.0
        k = int(np.argmin(np.abs(opp_m - m_comp)))
        far = opp_rows[k]["layers"]
        per_layer = {}
        for lid, st in state["layers"].items():
            in_tension = st["group"] == t_group
            res = {"id": lid, "group": st["group"], "y": st["y"], "depth_from_top": st["depth_from_top"], "area": st["area"], "bars": st["bars"], "db": st["db"],
                   "eps": st["eps"], "sigma": st["sigma"], "in_tension": in_tension}
            if in_tension:
                res.update(layer_anchorage(sec.steel, bond, st["db"], st["sigma"], st["eps"], far[lid]["sigma"], sec.anchorage_length))
                alt = layer_anchorage(sec.steel, bond, st["db"], st["sigma"], st["eps"], 0.0, sec.anchorage_length)      # the far-unstressed alternative
                res.update({f"{k2}_far_unstressed": alt[k2] for k2 in ("anchorage_ratio", "required_length_ratio", "slip_sezen_bilinear", "slip_inverse_law", "state", "status", "valid")})
                res["far_face_eps"] = far[lid]["eps"]; res["far_face_sigma"] = far[lid]["sigma"]
            per_layer[lid] = res
        rot = {p: face_rotation(per_layer, state["c"], p) for p in ("sezen_bilinear", "inverse_law")}
        tension = [lr for lr in per_layer.values() if lr["in_tension"]]
        valid = all(lr["valid"] for lr in tension)
        unsupported = any(lr["status"] == "unsupported" for lr in tension)
        return {"companion_M_opposite_face": float(opp_m[k]), "layers": per_layer, "face_rotation": rot,
                "theta_slip": rot["sezen_bilinear"]["theta_slip"], "theta_slip_inverse_law": rot["inverse_law"]["theta_slip"],
                "valid": valid, "unsupported": unsupported,
                "anchorage_ratio_max": max((lr["anchorage_ratio"] for lr in tension), default=0.0),
                "required_length_ratio_max": max((lr["required_length_ratio"] for lr in tension), default=0.0),
                "valid_far_unstressed": all(lr["valid_far_unstressed"] for lr in tension)}
    out_rows = [{**{k2: v for k2, v in r.items() if k2 != "layers"}, **anchorage_at(r)} for r in rows]
    ev = {}
    for name, state in events.items():
        ev[name] = None if state is None else {**{k2: v for k2, v in state.items() if k2 != "layers"}, **anchorage_at(state)}
    aci = aci_nominal(sec, direction)
    at_mn = next((r for r in out_rows if r["M"] >= aci["Mn"]), None)
    ev["aci_nominal_Mn"] = {**aci, "reached_on_curve": at_mn is not None,
                            "state_at_Mn": None if at_mn is None else {k2: at_mn[k2] for k2 in ("phi", "c", "eps_t_max", "theta_slip", "anchorage_ratio_max", "required_length_ratio_max", "valid", "unsupported", "valid_far_unstressed")}}
    ev["end_of_curve"] = {"limit": limit, **{k2: out_rows[-1][k2] for k2 in ("M", "phi", "c", "eps_c_top", "eps_t_max", "theta_slip", "anchorage_ratio_max", "required_length_ratio_max", "valid", "unsupported", "valid_far_unstressed")}}
    if ev["first_layer_yield"] and ev["first_layer_yield"]["theta_slip"]:
        f = ev["first_layer_yield"]
        ev["secant_to_first_layer_yield"] = {"M_over_theta_slip": f["M"] / f["theta_slip"],
                                             "note": "a reduced-model quantity if the interface yield were set at first-layer yield; not fitted, not approved"}
    return {"direction": direction, "tension_group": t_group, "d_t": d_t, "rows": out_rows, "events": ev, "companion_rule": companion,
            "bond": asdict(bond), "curve_limit": limit, "n_rows": len(out_rows),
            "invalid_rows": int(sum(1 for r in out_rows if not r["valid"])),
            "unsupported_rows": int(sum(1 for r in out_rows if r["unsupported"])),
            "far_unstressed_invalid_rows": int(sum(1 for r in out_rows if not r["valid_far_unstressed"])),
            "layer_state_counts": {s: int(n) for s, n in sorted(Counter(lr["state"] for r in out_rows for lr in r["layers"].values() if lr["in_tension"]).items())},
            "far_unstressed_state_counts": {s: int(n) for s, n in sorted(Counter(lr["state_far_unstressed"] for r in out_rows for lr in r["layers"].values() if lr["in_tension"]).items())}}


# ---- sections -----------------------------------------------------------------------------------------
def frame_section(label="frame_default"):
    """The project's bare x-beam section from Structure_Parameters (after any installed design record)."""
    import Structure_Parameters as sp
    rows = sp.beam_bar_layers()["x"]
    db = sp.rebar_diameter(sp.BEAM_BAR_SIZE)
    area = math.pi * db ** 2 / 4.0
    layers = []
    for group in ("top", "bottom"):
        for n, off in zip(rows[group]["per_layer"], rows[group]["offsets_in"]):
            layers.append({"depth_from_top": off if group == "top" else sp.H_BEAM - off, "area": n * area, "bars": n, "db": db, "group": group})
    steel = Steel(fy=sp.FY_KSI, es=sp.ES_KSI, fu=90.0, eps_sh=0.010, esh=1000.0, eps_su=0.12,
                  status="provisional A706 Grade 60 hardening (fu 90 ksi, eps_sh 0.010, Esh 1000 ksi, eps_su 0.12): declared, not measured")
    return Section(b=sp.B_BEAM, h=sp.H_BEAM, layers=layers, concrete=Concrete(fc=sp.FC_BEAM_KSI), steel=steel,
                   anchorage_length=sp.H_COL, fc_anchorage=sp.FC_COL_KSI, label=label)


def unit1_section():
    """Park and Ruitong Unit 1 (N, mm, MPa): 457 x 229 beam, 5 D16 top in two layers, 2 D16 bottom, through a 406 mm column."""
    a16 = math.pi * 16.0 ** 2 / 4.0
    layers = [{"depth_from_top": 42.0, "area": 3 * a16, "bars": 3, "db": 16.0, "group": "top"},
              {"depth_from_top": 75.0, "area": 2 * a16, "bars": 2, "db": 16.0, "group": "top"},
              {"depth_from_top": 457.0 - 42.0, "area": 2 * a16, "bars": 2, "db": 16.0, "group": "bottom"}]
    steel = Steel(fy=294.0, es=210400.0, fu=434.0, eps_sh=0.0255, esh=3580.0, eps_su=0.20, status="measured (paper Table 2(b)); eps_su bounded 0.15-0.25")
    return Section(b=229.0, h=457.0, layers=layers, concrete=Concrete(fc=45.9), steel=steel, anchorage_length=406.0, fc_anchorage=45.9,
                   label="unit1", units={"force": "N", "length": "mm", "stress": "MPa"})


# ---- export -------------------------------------------------------------------------------------------
LAYER_COLUMNS = ("eps", "sigma", "sigma_far", "L_near", "L_yielded", "L_far", "anchorage_ratio", "required_length_ratio", "sigma_min_inside_joint",
                 "slip_sezen_bilinear", "slip_inverse_law", "state", "status", "valid",
                 "anchorage_ratio_far_unstressed", "required_length_ratio_far_unstressed", "slip_sezen_bilinear_far_unstressed",
                 "slip_inverse_law_far_unstressed", "state_far_unstressed", "status_far_unstressed", "valid_far_unstressed")


def export_rows(ref, path):
    layer_ids = list(ref["rows"][0]["layers"].keys())
    base = ["phi", "c", "M", "eps_c_top", "eps_t_max", "d_t", "companion_M_opposite_face", "theta_slip", "theta_slip_inverse_law",
            "anchorage_ratio_max", "required_length_ratio_max", "valid", "unsupported", "valid_far_unstressed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(base + [f"{lid}_{col}" for lid in layer_ids for col in LAYER_COLUMNS])
        for r in ref["rows"]:
            cells = [r[k] for k in base]
            for lid in layer_ids:
                lr = r["layers"][lid]
                cells += [lr.get(col, "") for col in LAYER_COLUMNS]
            w.writerow(cells)


def run(sec, out, label=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    label = label or sec.label
    bond = Bond.sezen(sec.fc_anchorage, sec.units["stress"])
    result = {"version": VERSION, "label": label, "units": sec.units, "assumptions": ASSUMPTIONS,
              "section": {**asdict(sec), "concrete": asdict(sec.concrete), "steel": asdict(sec.steel)}, "bond": asdict(bond), "directions": {},
              "status": "reference only: nothing fitted, nothing installed; bond and hardening inputs provisional; nominal-strength agreement is a section check, not experimental validation"}
    plots = {}
    for direction in ("hogging", "sagging"):
        ref = reference(sec, bond, direction)
        export_rows(ref, out / f"{label}_{direction}.csv")
        result["directions"][direction] = {k: v for k, v in ref.items() if k != "rows"}
        plots[direction] = ref["rows"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for direction, style in (("hogging", "-"), ("sagging", "--")):
        rows = plots[direction]; ev = result["directions"][direction]["events"]
        M = np.array([r["M"] for r in rows]); th = np.array([r["theta_slip"] if r["theta_slip"] is not None else np.nan for r in rows])
        th2 = np.array([r["theta_slip_inverse_law"] if r["theta_slip_inverse_law"] is not None else np.nan for r in rows])
        valid = np.array([r["valid"] for r in rows]); ar = np.array([r["anchorage_ratio_max"] for r in rows])
        unsup = np.array([r["unsupported"] for r in rows]); invalid = ~valid & ~unsup
        axes[0].plot(np.where(valid, th, np.nan), M, style, color="black", lw=1.2, label=f"{direction} (Sezen bilinear)")
        axes[0].plot(np.where(valid, th2, np.nan), M, style, color="0.55", lw=0.9, label=f"{direction} (inverse law)")
        axes[0].plot(np.where(invalid, th, np.nan), M, style, color="red", lw=0.9, label=f"{direction} invalid (capacity exceeded)" if invalid.any() else None)
        axes[0].plot(np.where(unsup, th, np.nan), M, style, color="orange", lw=0.9, label=f"{direction} unsupported (no profile derived)" if unsup.any() else None)
        colour = "red" if direction == "hogging" else "blue"
        for name, marker in (("first_layer_yield", "o"), ("all_tension_layers_yielded", "^"), ("hardening_onset", "s")):
            if ev.get(name) and ev[name]["theta_slip"] is not None:
                axes[0].plot(ev[name]["theta_slip"], ev[name]["M"], marker, color=colour, ms=6, label=f"{direction} {name}")
        axes[0].axhline(ev["aci_nominal_Mn"]["Mn"], color="0.6", ls=":", lw=0.8)
        for lid in rows[0]["layers"]:
            if rows[0]["layers"][lid]["in_tension"]:
                axes[1].plot(M, [r["layers"][lid]["sigma"] for r in rows], style, lw=1.0, label=f"{direction} {lid}")
        req = np.array([r["required_length_ratio_max"] for r in rows])
        axes[2].plot(M, req, style, color="black", lw=1.2, label=f"{direction}: requirement / available")
        imposed = np.abs(req - ar) > 1e-12                  # only the fully mobilised branch reports a ratio different from the requirement
        if imposed.any():
            axes[2].plot(M, np.where(imposed, ar, np.nan), style, color="0.55", lw=2.2,
                         label=f"{direction}: fully mobilised, ratio fixed at 1")
    fu = sec.units
    axes[0].set_xlabel("face slip rotation theta_slip (rad)"); axes[0].set_ylabel(f"moment ({fu['force']} {fu['length']})"); axes[0].set_title("moment against compatible slip rotation (dotted: ACI Mn)", fontsize=10)
    axes[1].set_xlabel(f"moment ({fu['force']} {fu['length']})"); axes[1].set_ylabel(f"tension layer stress ({fu['stress']})"); axes[1].set_title("stress of each tension layer", fontsize=10)
    axes[2].set_xlabel(f"moment ({fu['force']} {fu['length']})"); axes[2].set_ylabel("zero-stress-zone required / available anchorage (worst layer)"); axes[2].axhline(1.0, color="red", ls=":", lw=0.8)
    axes[2].set_title("through-bar anchorage screen, companion far face\n(grey: ratio fixed at 1 by the fully mobilised branch, not evidence of adequacy)", fontsize=9)
    for ax in axes:
        ax.grid(True, lw=0.3, color="0.9"); ax.legend(fontsize=6, loc="lower right" if ax is axes[2] else "best")
    fig.suptitle(f"Section-equilibrium anchorage reference v3.1: {label} (provisional inputs; nothing fitted)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / f"{label}_anchorage_reference.png", dpi=130); plt.close(fig)
    (out / f"{label}_anchorage_reference.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    for direction in ("hogging", "sagging"):
        ev = result["directions"][direction]["events"]; f = ev["first_layer_yield"]; a = ev["all_tension_layers_yielded"]
        print(f"[{label} {direction}] first-layer yield ({f['layer']}) M {f['M']:.4g} theta_slip {f['theta_slip']:.3e} (inverse law {f['theta_slip_inverse_law']:.3e}) "
              f"ratio {f['anchorage_ratio_max']:.2f}; all layers yielded M {a['M']:.4g}" if f and a else f"[{label} {direction}] events missing",
              f"| ACI Mn {ev['aci_nominal_Mn']['Mn']:.4g} | end M {ev['end_of_curve']['M']:.4g} ({ev['end_of_curve']['limit']}) ratio {ev['end_of_curve']['anchorage_ratio_max']:.2f}, "
              f"invalid rows {result['directions'][direction]['invalid_rows']}, unsupported {result['directions'][direction]['unsupported_rows']}, "
              f"far-unstressed invalid {result['directions'][direction]['far_unstressed_invalid_rows']}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=None, help="design root holding <case>/ (Fixed_Design_Diagnostics.read_record)")
    parser.add_argument("--case", default=None)
    parser.add_argument("--unit1", action="store_true", help="also run Park and Ruitong Unit 1's section as a cross-check")
    args = parser.parse_args(argv)
    label = "frame_default"
    if args.root and args.case:
        from Analysis.Fixed_Design_Diagnostics import install_record, read_record
        _, _, record = read_record(Path(args.root), args.case)
        install_record(record, args.case)
        label = f"frame_{args.case}"
    results = {label: run(frame_section(label), args.out)}
    if args.unit1:
        results["unit1"] = run(unit1_section(), args.out)
    return results


if __name__ == "__main__":
    main()
