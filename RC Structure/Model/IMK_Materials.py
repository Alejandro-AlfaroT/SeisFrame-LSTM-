"""Explicit modern OpenSees IMK commands for moment-rotation springs.

Units: kip-in, rad, and kip-in/rad. Lamda is a command-level deformation
parameter: E_ref = Lamda * My. It is NEVER multiplied by yield rotation here.
Paper-normalized coefficients must be translated and documented upstream.
This adapter validates inputs; it does not supply experimental calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
import math

import openseespy.opensees as ops

ENERGY_CONVENTION = "opensees_ref_energy_equals_lamda_times_fy"
SCHEMA_VERSION = "rotational_imk_v2_explicit_energy_metadata"
MATERIAL_TYPES = ("IMKBilin", "IMKPeakOriented", "IMKPinching")
MAPPING_VERSION = "explicit_reference_energy_v1"
# Provisional research anchor (pre-generation review, 2026-10-04): the reference energy of every active mode
# is the base Lamda times the SMALLER of the two physical yield moments of the spring. It does not depend on
# which physical direction is the positive input branch. It is a modelling convention for a declared research
# batch: not an experimental calibration, not an ACI requirement and not a proven bound on response.
PROVISIONAL_ANCHOR_VERSION = "provisional_min_yield_anchor_v1"
PROVISIONAL_ANCHOR_STATUS = "provisional_research_not_experimentally_calibrated"


def active_energy_modes(material_type):
    if material_type not in MATERIAL_TYPES:
        raise ValueError(f"Unsupported IMK material {material_type!r}")
    return ("S", "C", "K") if material_type == "IMKBilin" else ("S", "C", "A", "K")


def validate_energy_calibration(calibration, material_type, *, verification_only=False):
    """Validate an explicit profile; this checks provenance, not experimental truth.

    Structural callers require reviewed experimental support. Synthetic profiles
    are accepted only through an explicit verification-only call, never inferred
    from a missing profile, legacy Lambda, or a selected yield moment.
    """
    if not isinstance(calibration, dict):
        raise ValueError("Explicit experimentally supported IMK energy calibration is required")
    expected = "verification_only" if verification_only else "experimentally_supported"
    if calibration.get("status") != expected:
        raise ValueError(f"IMK energy calibration status must be {expected}")
    for key in ("calibration_id", "deformation_scope", "derivation", "applicability_basis"):
        if not isinstance(calibration.get(key), str) or not calibration[key].strip():
            raise ValueError(f"IMK energy calibration requires {key}")
    if calibration.get("material_type") != material_type:
        raise ValueError("Energy calibration material type does not match the installed material")
    if calibration.get("units") != "kip-in*rad":
        raise ValueError("Rotational reference energies must be explicitly supplied in kip-in*rad")
    for key in ("source_refs", "specimen_ids"):
        values = calibration.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"IMK energy calibration requires nonempty {key}")
    if not verification_only:
        for key in ("reviewed_by", "review_basis"):
            if not isinstance(calibration.get(key), str) or not calibration[key].strip():
                raise ValueError(f"Experimental energy calibration requires {key}")
        try:
            if date.fromisoformat(calibration["review_date"]).isoformat() != calibration["review_date"]:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Experimental energy calibration requires review_date YYYY-MM-DD") from exc
    energies = calibration.get("energies_kip_in_rad")
    modes = active_energy_modes(material_type)
    if not isinstance(energies, dict) or set(energies) != set(modes):
        raise ValueError(f"Explicit energy capacities must contain exactly {modes}")
    for mode, value in energies.items():
        _positive(f"E_ref_{mode}", value)
    return json.loads(json.dumps(calibration, allow_nan=False))


def _positive(name, value):
    if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be an explicit positive finite value; got {value!r}")


@dataclass(frozen=True)
class RotationalBackbone:
    dp: float
    dpc: float
    du: float
    fy: float
    fmax_fy: float
    fres_fy: float

    def arguments(self):
        for name in ("dp", "dpc", "du", "fy", "fmax_fy"):
            _positive(name, getattr(self, name))
        if self.fmax_fy < 1.0:
            raise ValueError("fmax_fy must be at least 1 (a strength ratio, not a hardening slope)")
        if not math.isfinite(self.fres_fy) or not 0 <= self.fres_fy <= 1:
            raise ValueError("fres_fy must lie in [0, 1]")
        return (self.dp, self.dpc, self.du, self.fy, self.fmax_fy, self.fres_fy)


@dataclass(frozen=True)
class CyclicParameters:
    lamda_s: float
    lamda_c: float
    lamda_k: float
    c_s: float
    c_c: float
    c_k: float
    d_pos: float
    d_neg: float
    lamda_a: float | None = None
    c_a: float | None = None
    kappa_f: float | None = None
    kappa_d: float | None = None
    energy_convention: str = ENERGY_CONVENTION

    def arguments(self, material_type):
        if material_type not in MATERIAL_TYPES:
            raise ValueError(f"Unsupported IMK material {material_type!r}")
        if self.energy_convention != ENERGY_CONVENTION:
            raise ValueError("IMK inputs must use modern OpenSees E_ref = Lamda * Fy; no implicit conversion")
        for name in ("lamda_s", "lamda_c", "lamda_k", "c_s", "c_c", "c_k", "d_pos", "d_neg"):
            _positive(name, getattr(self, name))
        if self.d_pos > 1 or self.d_neg > 1:
            raise ValueError("d_pos and d_neg must lie in (0, 1]")
        if material_type == "IMKBilin":
            if self.lamda_a is not None or self.c_a is not None:
                raise ValueError("IMKBilin has no accelerated-reloading A arguments")
            result = (self.lamda_s, self.lamda_c, self.lamda_k, self.c_s, self.c_c, self.c_k,
                      self.d_pos, self.d_neg)
        else:
            _positive("lamda_a", self.lamda_a)
            _positive("c_a", self.c_a)
            result = (self.lamda_s, self.lamda_c, self.lamda_a, self.lamda_k,
                      self.c_s, self.c_c, self.c_a, self.c_k, self.d_pos, self.d_neg)
        if material_type == "IMKPinching":
            for name in ("kappa_f", "kappa_d"):
                _positive(name, getattr(self, name))
                if getattr(self, name) >= 1:
                    raise ValueError(f"{name} must lie in (0, 1)")
            result += (self.kappa_f, self.kappa_d)
        elif self.kappa_f is not None or self.kappa_d is not None:
            raise ValueError(f"{material_type} has no pinching arguments")
        return result


def define_rotational_imk(material_type, mat_tag, ke, positive, negative, cyclic, *, provenance):
    """Install one material and return its exact command and calibration identity.

    `provenance` must identify the parameter set, its status and the deformation
    it represents. Status is recorded, not promoted to production acceptance.
    The two branches use positive magnitudes; Ke is shared by both directions.
    """
    _positive("mat_tag", mat_tag)
    if int(mat_tag) != mat_tag:
        raise ValueError("mat_tag must be an integer")
    _positive("ke", ke)
    for name in ("calibration_id", "status", "deformation_scope"):
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ValueError(f"Calibration provenance requires {name}")
    args = (material_type, int(mat_tag), ke, *positive.arguments(), *negative.arguments(),
            *cyclic.arguments(material_type))
    for branch in (positive, negative):
        if branch.du <= branch.fy / ke + branch.dp:
            raise ValueError("Ultimate rotation must exceed the capping rotation "
                             "(the spring's own elastic yield rotation plus the plastic rotation)")
    identity = {"schema_version": SCHEMA_VERSION, "material_type": material_type,
                "units": {"moment": "kip-in", "rotation": "rad", "stiffness": "kip-in/rad", "lamda": "rad"},
                "ke": ke, "positive": asdict(positive), "negative": asdict(negative),
                "cyclic": asdict(cyclic), "provenance": provenance,
                "reference_energies_kip_in_rad": {
                    mode: getattr(cyclic, f"lamda_{mode.lower()}") * positive.fy
                    for mode in active_energy_modes(material_type)},
                "reference_energy_equation": "E_ref,m = Lamda_m * Fy_positive"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    # Snapshot caller dictionaries so later configuration changes cannot rewrite
    # the recorded identity of a material that is already installed.
    metadata = json.loads(encoded)
    metadata.update(parameter_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                    material_tag=int(mat_tag), opensees_version=ops.version(),
                    command=list(args))
    ops.uniaxialMaterial(*args)
    return metadata


def _map_physical_branches(material_type, positive, negative, cyclic, reverse, physical_directions):
    """Validate two physical branches and place them, with their directional D, on the input coordinates."""
    if not isinstance(reverse, bool):
        raise ValueError("reverse must be a boolean coordinate mapping")
    if (len(physical_directions) != 2 or len(set(physical_directions)) != 2
            or any(not isinstance(v, str) or not v.strip() for v in physical_directions)):
        raise ValueError("Provide two distinct physical direction labels")
    positive.arguments()
    negative.arguments()
    cyclic.arguments(material_type)
    mapped_pos, mapped_neg = (negative, positive) if reverse else (positive, negative)
    labels = tuple(reversed(physical_directions)) if reverse else tuple(physical_directions)
    directional = {"d_pos": cyclic.d_neg if reverse else cyclic.d_pos,
                   "d_neg": cyclic.d_pos if reverse else cyclic.d_neg}
    record = {"coordinate_reversed": reverse,
              "positive_input_direction": labels[0], "negative_input_direction": labels[1],
              "physical_positive": asdict(positive), "physical_negative": asdict(negative),
              "physical_d_pos": cyclic.d_pos, "physical_d_neg": cyclic.d_neg}
    return mapped_pos, mapped_neg, directional, record


def define_mapped_rotational_imk(material_type, mat_tag, ke, positive, negative, cyclic, *,
                                calibration, reverse=False, physical_directions=("positive", "negative"),
                                provenance, verification_only=False):
    """Map complete physical branches, directional D and mode energies together."""
    profile = validate_energy_calibration(calibration, material_type, verification_only=verification_only)
    mapped_pos, mapped_neg, directional, record = _map_physical_branches(
        material_type, positive, negative, cyclic, reverse, physical_directions)
    mapped_cyclic = replace(cyclic,
        **{f"lamda_{mode.lower()}": value / mapped_pos.fy
           for mode, value in profile["energies_kip_in_rad"].items()},
        **directional)
    mapped_provenance = {**provenance, "energy_calibration": profile,
        "energy_mapping": {"version": MAPPING_VERSION, **record, "verification_only": verification_only}}
    return define_rotational_imk(material_type, mat_tag, ke, mapped_pos, mapped_neg, mapped_cyclic,
                                 provenance=mapped_provenance)


def define_anchored_rotational_imk(material_type, mat_tag, ke, positive, negative, cyclic, *,
                                  reverse=False, physical_directions=("positive", "negative"), provenance):
    """Install a spring whose mode energies are anchored to its smaller physical yield moment.

    ``positive`` and ``negative`` are the PHYSICAL branches and ``cyclic`` carries the base Lamda of
    every mode (already the OpenSees deformation parameter; never multiplied by a rotation here). For
    each active mode, the suppressed large-capacity modes included,

        M_ref = min(My_physical_positive, My_physical_negative)
        E_ref = Lamda_base * M_ref
        Lamda_installed = Lamda_base * (M_ref / My_positive_input)

    so E_ref is the same whichever physical branch is the positive input, and equal strengths install
    the base Lamda unchanged. The record carries the policy and its status; nothing here is, or is
    labelled as, an experimental calibration.
    """
    mapped_pos, mapped_neg, directional, record = _map_physical_branches(
        material_type, positive, negative, cyclic, reverse, physical_directions)
    modes = active_energy_modes(material_type)
    base = {mode: getattr(cyclic, f"lamda_{mode.lower()}") for mode in modes}
    reference_moment = min(positive.fy, negative.fy)
    ratio = reference_moment / mapped_pos.fy
    mapped_cyclic = replace(cyclic, **{f"lamda_{mode.lower()}": base[mode] * ratio for mode in modes}, **directional)
    mapped_provenance = {**provenance, "energy_mapping_status": PROVISIONAL_ANCHOR_VERSION,
        "energy_mapping": {"version": PROVISIONAL_ANCHOR_VERSION, "status": PROVISIONAL_ANCHOR_STATUS,
                           "anchor_policy": "reference moment is the smaller physical yield moment of the spring",
                           "reference_moment_kip_in": reference_moment,
                           "base_lamda_rad": base,
                           "reference_energies_kip_in_rad": {mode: base[mode] * reference_moment for mode in modes},
                           "installed_lamda_rad": {mode: base[mode] * ratio for mode in modes},
                           "equations": ["M_ref = min(My_physical_positive, My_physical_negative)",
                                         "E_ref,m = Lamda_base,m * M_ref",
                                         "Lamda_installed,m = E_ref,m / My_positive_input"],
                           **record,
                           "limits": ("modelling convention for a declared research batch; not an experimental "
                                      "calibration, not an ACI requirement, not a proven bound on response")}}
    return define_rotational_imk(material_type, mat_tag, ke, mapped_pos, mapped_neg, mapped_cyclic,
                                 provenance=mapped_provenance)
