"""Member-to-native calibration of the IMK member-end springs: the frozen SeisFrame V2 calibration (2026-10-29).

The member regressions (Model/IMK_Calibration: Haselton et al. 2008 theta_p, theta_pc and lambda) describe a MEMBER:
plastic and post-capping rotation intervals of a cantilever of shear span Ls, and a deterioration capacity in units of
its yield energy. The model carries that member as an elastic element between two stiff zeroLength springs, so the
spring's own (native) command values are not the member intervals. Until 2026-10-29 the member intervals were
installed directly as native dp/dpc and the energy used a nominal 0.005/0.004 rad yield rotation. The frozen
calibration converts them on a nominal half-member reference instead.

Reference (per spring; the member's centerline length L and existing effective EI0):

    Ls  = L / 2                      K0 = 3 EI0 / Ls = 6 EI0 / L = Kh / n
    n   = 20                         Kh = n K0  (the installed spring stiffness)
    EIe = 1.05 EI0                   Kb = 3 EIe / Ls = 1.05 K0  (the elastic element between the springs)

Each physical branch (positive magnitudes; My is that branch's adopted yield moment):

    Mc            = 1.10 My
    theta_y_ref   = My / K0
    theta_cap_ref = theta_y_ref + theta_p,member
    dp_native     = theta_cap_ref - Mc / Kb - My / Kh
    dpc_native    = theta_pc,member + Mc / Kb
    du_native     = theta_p,member + theta_pc,member + legacy padding (0.005 beam / 0.004 column), unchanged
    theta_reference_chord = theta_native_spring + M / Kb

The member theta_pc carries the published 0.10 ceiling before conversion; dpc_native may exceed 0.10 and is not
clamped back. With the retained split the assembled initial stiffness is 420/421 K0; the cap point and the post-capping
slope of the reference are reproduced exactly.

Deterioration energy (S and C active and equal; A and K suppressed with large finite capacities):

    theta_y_ref,E = My_canonical / K0
    Eref_S = Eref_C = lambda_H * My_canonical * theta_y_ref,E
    Eref_A = Eref_K = 1e12 * My_canonical
    Lambda_native,m = Eref_m / My_native_positive

My_canonical is the canonical PHYSICAL direction of Model/IMK_Materials (hogging for a beam's vertical bending, the
section's positive branch for the symmetric springs), so reversing the native coordinate keeps every physical energy.

Status: approved on 2026-10-29 by the principal researcher, with the RA's agreement, as the normal V2 member-hinge
calibration; numerically equivalent to the research profile v2_imk_half_member_reference_research_20261028 of the
twenty-run MCER/DBE study. It is NOT an experimental validation: approval freezes a modelling choice for V2, it does
not establish universal applicability or certify a dataset release, and the retained du is a provisional numerical
cutoff, not an observed collapse rotation.

``LEGACY_REPLAY`` reproduces the pre-freeze rule of commit 66f964c1 (member intervals installed as native values,
nominal-yield-rotation energy) for explicit historical replay only.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

FROZEN_V2 = "seisframe_imk_v2_frozen_20261029"
RESEARCH_EQUIVALENT = "v2_imk_half_member_reference_research_20261028"
LEGACY_REPLAY = "legacy_direct_interval_nominal_energy_66f964c1"
CALIBRATIONS = (FROZEN_V2, LEGACY_REPLAY)

FROZEN_STATUS = "frozen_v2_modelling_choice_not_experimentally_validated"
LEGACY_STATUS = "historical_replay_of_the_pre_freeze_rule"

# The frozen constants. They are part of the calibration identity: a build that configures other values is refused,
# never mixed (configuration_problems).
SPLIT_N = 20.0                  # Kh = n K0 (IMK_HINGE_STIFFNESS_FACTOR)
INTERIOR_FACTOR = 1.05          # EIe = 1.05 EI0 = (n + 1)/n EI0 (Model/IMK_Hinges.imk_elastic_inertia_factor)
MC_OVER_MY = 1.10
MR_OVER_MY = 0.20
SUPPRESSED_MODE_LAMDA = 1.0e12  # Eref_A = Eref_K = 1e12 My_canonical
LEGACY_CUTOFF_PADDING_RAD = {"beam": 0.005, "column": 0.004}
ASSEMBLED_INITIAL_STIFFNESS_OVER_K0 = 1.0 / (1.0 / SPLIT_N + 1.0 / INTERIOR_FACTOR)   # 420/421

SCIENTIFIC_BASELINE_COMMIT = "66f964c10d6a9ce915aba5b1ec5db40ca3d72822"
APPROVAL = {
    "calibration_id": FROZEN_V2,
    "approved_on": "2026-10-29",
    "approved_by": "principal researcher, reporting agreement with the RA",
    "authorization": "freeze and implement the reviewed calibration as the normal SeisFrame V2 member-hinge calibration",
    "numerically_equivalent_to": RESEARCH_EQUIVALENT,
    "evidence": "twenty-run MCER/DBE study of ten saved designs (RSN864 pair, average-intensity scaling)",
    "scientific_baseline_commit": SCIENTIFIC_BASELINE_COMMIT,
    "experimental_validation": False,
    "production_generation_authorized": False,
    "statement": ("Not experimentally validated. The approval freezes a modelling choice for V2; it is not proof of "
                  "universal applicability or a dataset-release certification. The retained native du is a provisional "
                  "numerical cutoff, not an experimentally established collapse rotation."),
}
RETAINED_ASSUMPTIONS = {
    "effective_stiffness": "existing ACI 318-19 Table 6.6.3.1.1(a) modifiers on EI0 (Structure_Parameters)",
    "reference_length": "nominal half-member: Ls = L/2 of the model centerline length (no clear span, no inferred inflection point)",
    "split": "n = 20, interior factor 1.05; assembled initial stiffness 420/421 K0, retained, not corrected",
    "strength_ratios": "Mc/My = 1.10, Mr/My = 0.20 (residual and tail unchanged)",
    "regressions": "Haselton et al. (2008) Eq. 3.10, 3.16 (0.10 ceiling on the member interval), 3.20, with their existing inputs, floors and clamps",
    "deterioration_modes": "S and C active with equal energies; A and K suppressed by 1e12 My_canonical; c = 1, D = 1",
    "ultimate_cutoff": "native du = theta_p + theta_pc + legacy padding (0.005 beam / 0.004 column), provisional numerical cutoff",
    "bond_slip": "existing per-end ownership and a_sl policy (Model/Deformation_Ownership)",
}
# The scope the approval was given for; a build outside it is recorded (applicability), not refused.
APPROVED_SCOPE = {"JOINT_MODEL": "rigid_centerline", "BEAM_STIFFNESS_MODIFIER": 0.35, "COLUMN_STIFFNESS_MODIFIER": 0.70,
                  "CRACKED_SECTION_ANALYSIS": True}
DU_POLICY = ("retained native cutoff: member theta_p + member theta_pc + legacy padding, installed unchanged; independent of "
             "the energy yield rotation; provisional numerical cutoff, not a collapse rotation")


def active_calibration():
    import Structure_Parameters as sp
    return getattr(sp, "IMK_MEMBER_SPRING_CALIBRATION", FROZEN_V2)


def is_frozen(calibration=None):
    calibration = calibration or active_calibration()
    if calibration not in CALIBRATIONS:
        raise ValueError(f"Unknown IMK member spring calibration {calibration!r}; known: {list(CALIBRATIONS)}")
    return calibration == FROZEN_V2


def legacy_cutoff_padding(member_type, calibration=None):
    """The additive padding of the retained native du (0.005 beam / 0.004 column).

    Under the frozen calibration it is the frozen constant, decoupled from the energy yield rotation. Under the legacy
    replay it is the nominal IMK_*_THETA_Y the pre-freeze rule added (same numbers by default), read as before.
    """
    kind = "column" if member_type == "column" else "beam"
    if is_frozen(calibration):
        return LEGACY_CUTOFF_PADDING_RAD[kind]
    import Structure_Parameters as sp
    return sp.IMK_COLUMN_THETA_Y if kind == "column" else sp.IMK_BEAM_THETA_Y


def configuration_problems(calibration=None):
    """Why the configured state cannot install ``calibration`` (default: the active one); empty when it can.

    The frozen calibration fixes the mapping, the energy rule and their constants. Settings that would mix it with
    another policy (another material, stiffness rule, deterioration or energy mapping, strength ratio, c or D) are
    problems; the replay has none.
    """
    import Structure_Parameters as sp
    from Model.IMK_Materials import CANONICAL_BASELINE_VERSION, ENERGY_CONVENTION
    calibration = calibration or active_calibration()
    if calibration not in CALIBRATIONS:
        return [f"IMK_MEMBER_SPRING_CALIBRATION {calibration!r} is not one of {list(CALIBRATIONS)}"]
    if calibration != FROZEN_V2:
        return []
    required = {"IMK_MATERIAL_TYPE": "IMKPeakOriented", "IMK_USE_CALIBRATED_BACKBONE": True,
                "IMK_DETERIORATION_MODE": "haselton_2008", "IMK_ENERGY_MAPPING_MODE": CANONICAL_BASELINE_VERSION,
                "IMK_ENERGY_CONVENTION": ENERGY_CONVENTION, "IMK_HINGE_STIFFNESS_MODE": "member_stiffness_factor",
                "IMK_HINGE_STIFFNESS_FACTOR": SPLIT_N,
                "IMK_FMAXFY_POS": MC_OVER_MY, "IMK_FMAXFY_NEG": MC_OVER_MY,
                "IMK_FRESFY_POS": MR_OVER_MY, "IMK_FRESFY_NEG": MR_OVER_MY,
                "IMK_C_S": 1.0, "IMK_C_C": 1.0, "IMK_C_A": 1.0, "IMK_C_K": 1.0, "IMK_D_POS": 1.0, "IMK_D_NEG": 1.0}
    problems = [f"{key} is {getattr(sp, key, None)!r}; {FROZEN_V2} requires {want!r}"
                for key, want in required.items() if getattr(sp, key, None) != want]
    if getattr(sp, "IMK_MEMBER_ENERGY_CALIBRATIONS", None):
        problems.append(f"IMK_MEMBER_ENERGY_CALIBRATIONS supplies explicit member energies; {FROZEN_V2} derives its own")
    return problems


def require_configuration(calibration=None):
    problems = configuration_problems(calibration)
    if problems:
        raise ValueError(f"The configured IMK settings cannot install member spring calibration "
                         f"{calibration or active_calibration()}: " + "; ".join(problems))


def scope_record():
    """Whether the build is inside the scope the frozen calibration was approved for. Recorded, never enforced."""
    import Structure_Parameters as sp
    from Model import Deformation_Ownership as ownership
    outside = [f"{key} is {getattr(sp, key, None)!r} (approved study: {want!r})"
               for key, want in APPROVED_SCOPE.items() if getattr(sp, key, None) != want]
    if ownership.slip_interfaces():
        outside.append("face slip interfaces are registered (approved study: none; a_sl = 1 at every end)")
    return {"approved_scope": dict(APPROVED_SCOPE, slip_interfaces="none"), "within_approved_scope": not outside,
            "outside_scope": outside}


def _context_label(context):
    context = context or {}
    axis = {5: "y", 6: "z"}.get(context.get("spring_local_direction"), context.get("spring_local_direction"))
    return (f"{context.get('member_type', 'member')} {context.get('physical_member_tag', '?')} end "
            f"{context.get('end', '?')} axis {axis}")


def _finite_positive(name, value, context):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0.0:
        raise ValueError(f"{_context_label(context)}: {name} must be a positive finite value; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class MemberBranch:
    """One PHYSICAL branch of a member hinge in member terms: the regression intervals, not native command values.

    A distinct type from Model.IMK_Materials.RotationalBackbone (native values) so a native branch can never be
    converted a second time: native_branch accepts only this type.
    """
    physical_direction: str
    my: float
    theta_p: float
    theta_pc: float
    du: float              # retained native cutoff, installed unchanged
    fmax_fy: float
    fres_fy: float


@dataclass(frozen=True)
class HalfMemberReference:
    kh: float
    k0: float
    kb: float
    n: float
    interior_factor: float

    def record(self, length_in=None, k0_from_section=None):
        return {"definition": "nominal half-member: Ls = L/2 of the model centerline length",
                "length_in": length_in, "shear_span_in": None if length_in is None else 0.5 * length_in,
                "K0_kip_in_per_rad": self.k0, "Kh_kip_in_per_rad": self.kh, "Kb_kip_in_per_rad": self.kb,
                "n": self.n, "interior_factor": self.interior_factor,
                "K0_from_section_6EI_over_L": k0_from_section,
                "assembled_initial_stiffness_over_K0": ASSEMBLED_INITIAL_STIFFNESS_OVER_K0,
                "equations": ["K0 = 3 EI0 / Ls = 6 EI0 / L = Kh / n", "Kh = n K0", "Kb = 3 EIe / Ls = 1.05 K0"]}


def half_member_reference(kh, *, context=None, k0_from_section=None, element_factor=None):
    """K0 = Kh/n and Kb = 1.05 K0 from the installed spring stiffness (the approved adapter's arithmetic).

    ``k0_from_section`` (6 EI0/L of the member's own section) and ``element_factor`` (the inertia factor the elastic
    element was given) are checked against the reference when supplied: the spring and the element must describe
    the same member, and the interior correction is applied once.
    """
    kh = _finite_positive("Kh", kh, context)
    k0 = kh / SPLIT_N
    kb = INTERIOR_FACTOR * k0
    if k0_from_section is not None and abs(k0_from_section / k0 - 1.0) > 1e-12:
        raise ValueError(f"{_context_label(context)}: K0 = Kh/n = {k0!r} disagrees with 6 EI0/L = {k0_from_section!r}")
    if element_factor is not None and element_factor != INTERIOR_FACTOR:
        raise ValueError(f"{_context_label(context)}: the elastic element carries inertia factor {element_factor!r}; "
                         f"{FROZEN_V2} requires {INTERIOR_FACTOR!r} exactly once")
    return HalfMemberReference(kh=kh, k0=k0, kb=kb, n=SPLIT_N, interior_factor=INTERIOR_FACTOR)


def native_branch(branch, reference, *, input_sign, context=None):
    """(native RotationalBackbone, record) for one physical member branch on the half-member reference."""
    from Model.IMK_Materials import RotationalBackbone
    if not isinstance(branch, MemberBranch):
        raise TypeError(f"{_context_label(context)}: native_branch converts a MemberBranch (member intervals) only; got "
                        f"{type(branch).__name__}. A native branch must not be converted again.")
    label = f"{branch.physical_direction} branch ({input_sign} input)"
    for name in ("my", "theta_p", "theta_pc", "du", "fmax_fy"):
        _finite_positive(f"{label} {name}", getattr(branch, name), context)
    if branch.fmax_fy != MC_OVER_MY or branch.fres_fy != MR_OVER_MY:
        raise ValueError(f"{_context_label(context)}: {label} strength ratios {branch.fmax_fy!r}/{branch.fres_fy!r} are "
                         f"not the frozen Mc/My = {MC_OVER_MY} and Mr/My = {MR_OVER_MY}")
    fy, k0, kb, kh = branch.my, reference.k0, reference.kb, reference.kh
    theta_y_ref = fy / k0
    cap = fy / k0 + branch.theta_p
    mc = fy * branch.fmax_fy
    dp = cap - mc / kb - fy / kh
    dpc = branch.theta_pc + mc / kb
    native = RotationalBackbone(dp=dp, dpc=dpc, du=branch.du, fy=fy, fmax_fy=branch.fmax_fy, fres_fy=branch.fres_fy)
    if not (math.isfinite(dp) and dp > 0.0):
        raise ValueError(f"{_context_label(context)}: {label} maps to native dp = {dp!r} (theta_cap_ref {cap!r} - Mc/Kb "
                         f"{mc / kb!r} - My/Kh {fy / kh!r}); the frozen calibration does not repair it")
    if not (math.isfinite(dpc) and dpc > 0.0):
        raise ValueError(f"{_context_label(context)}: {label} maps to native dpc = {dpc!r}")
    native_cap = fy / kh + dp
    if not branch.du > native_cap:
        raise ValueError(f"{_context_label(context)}: {label} retained cutoff du = {branch.du!r} is at or before the "
                         f"mapped native capping rotation My/Kh + dp = {native_cap!r}; the frozen calibration neither "
                         "enlarges the cutoff nor floors the intervals")
    record = {"physical_direction": branch.physical_direction, "input_sign": input_sign,
              "My_kip_in": fy, "Mc_kip_in": mc, "Mr_kip_in": fy * branch.fres_fy,
              "fmax_fy": branch.fmax_fy, "fres_fy": branch.fres_fy,
              "member_theta_p_rad": branch.theta_p, "member_theta_pc_rad": branch.theta_pc,
              "member_theta_y_reference_rad": theta_y_ref, "member_theta_cap_reference_rad": cap,
              "native_theta_y_rad": fy / kh, "native_dp_rad": dp, "native_dpc_rad": dpc,
              "native_theta_cap_rad": native_cap, "native_du_rad": branch.du,
              "native_spring": asdict(native)}
    return native, record


def frozen_energies(lambda_haselton, my_canonical, reference, modes, *, context=None):
    """Physical reference energies {mode: Eref} of the frozen rule, with their record."""
    lam = _finite_positive("lambda_H", lambda_haselton, context)
    mref = _finite_positive("My_canonical", my_canonical, context)
    theta_ref = mref / reference.k0
    active = lam * mref * theta_ref
    energies = {}
    for mode in modes:
        energies[mode] = active if mode in ("S", "C") else SUPPRESSED_MODE_LAMDA * mref
    record = {"lambda_haselton_dimensionless": lam, "My_canonical_kip_in": mref,
              "energy_reference_theta_y_rad": theta_ref,
              "reference_energies_kip_in_rad": dict(energies),
              "active_modes": [m for m in modes if m in ("S", "C")],
              "suppressed_modes": [m for m in modes if m not in ("S", "C")],
              "equations": ["theta_y_ref,E = My_canonical / K0",
                            "Eref_S = Eref_C = lambda_H * My_canonical * theta_y_ref,E",
                            "Eref_A = Eref_K = 1e12 * My_canonical",
                            "Lambda_native,m = Eref_m / My_native_positive"]}
    return energies, record


def reference_chord_rotation(theta_native_spring, moment, kb):
    """theta_reference_chord = theta_native_spring + M / Kb (member-reference rotation of one end)."""
    return theta_native_spring + moment / kb


SPRING_POINTS_VERSION = "imk_spring_points_v1_native_and_member_reference"


def spring_points(material):
    """The rotations an INSTALLED member-end spring is read against, per native input sign.

    ``material`` is one hinge-registry record (installed_materials[end][axis]). Native values come from the installed
    command: yield My/Kh, capping My/Kh + dp, cutoff du (a provisional numerical cutoff, not a collapse rotation).
    Under the frozen calibration the member-reference values of the half-member are added (K0, Kb, My/K0, the
    member cap and intervals); under the legacy replay the member intervals ARE the native ones and no reference
    exists. Returns None for a released (elastic) end. Consumers read capping, yield and cutoff from here, never from
    the global IMK_THETA_* constants or the member intervals alone.
    """
    if material is None or material.get("released") or material.get("material_type") == "Elastic":
        return None
    ke = float(material["ke"])
    provenance = material.get("provenance") or {}
    mapping = provenance.get("member_spring_calibration")
    by_sign = {}
    for sign in ("positive", "negative"):
        branch = material[sign]
        fy = float(branch["fy"])
        by_sign[sign] = {"My_kip_in": fy, "Mc_kip_in": fy * branch["fmax_fy"], "Mr_kip_in": fy * branch["fres_fy"],
                         "native_theta_y_rad": fy / ke, "native_dp_rad": branch["dp"], "native_dpc_rad": branch["dpc"],
                         "native_theta_cap_rad": fy / ke + branch["dp"], "native_du_rad": branch["du"]}
    result = {"version": SPRING_POINTS_VERSION, "material_tag": material.get("material_tag"), "Kh_kip_in_per_rad": ke,
              "calibration_id": mapping["calibration_id"] if mapping else provenance.get("calibration_id"),
              "member_reference": None, "positive": by_sign["positive"], "negative": by_sign["negative"]}
    if mapping:
        reference = mapping["reference"]
        result["member_reference"] = {key: reference[key] for key in
                                      ("K0_kip_in_per_rad", "Kb_kip_in_per_rad", "length_in", "shear_span_in")}
        result["energy"] = {key: mapping["energy"][key] for key in
                            ("lambda_haselton_dimensionless", "My_canonical_kip_in", "energy_reference_theta_y_rad",
                             "reference_energies_kip_in_rad")}
        for record in mapping["branches"].values():
            by_sign[record["input_sign"]].update(
                physical_direction=record["physical_direction"],
                member_theta_p_rad=record["member_theta_p_rad"], member_theta_pc_rad=record["member_theta_pc_rad"],
                member_theta_y_reference_rad=record["member_theta_y_reference_rad"],
                member_theta_cap_reference_rad=record["member_theta_cap_reference_rad"])
    else:
        mapped = provenance.get("energy_mapping") or {}
        for sign in ("positive", "negative"):
            by_sign[sign].update(physical_direction=mapped.get(f"{sign}_input_direction"),
                                 member_theta_p_rad=by_sign[sign]["native_dp_rad"],
                                 member_theta_pc_rad=by_sign[sign]["native_dpc_rad"],
                                 member_theta_y_reference_rad=None, member_theta_cap_reference_rad=None)
    return result


DEMAND_VERSION = "imk_native_spring_demand_v1"
DEMAND_DEFINITIONS = {
    "rotation": "native zeroLength spring rotation (hinge node minus joint), rad; not a story drift and not a chord rotation",
    "plastic_rotation": "max(0, peak native rotation of the sign - native yield My/Kh of that sign)",
    "capping_ratio": "plastic_rotation / native dp of the same sign: 1 at the native capping point My/Kh + dp",
    "past_capping": "peak native rotation reached the native capping rotation My/Kh + dp of its sign",
    "past_native_cutoff": ("peak native rotation reached the retained native du of its sign: a provisional numerical "
                           "cutoff, not an observed collapse or material failure"),
}


def rotation_demand(points, rotation_max, rotation_min):
    """One installed spring's peak rotations against its OWN native points, sign by sign.

    ``points`` is spring_points(material); ``rotation_max`` and ``rotation_min`` are the extreme native spring
    rotations reached (signed, rad). Each native sign is measured on its own branch (beam ends are asymmetric), and
    the governing sign is the larger capping ratio. Returns None for a released end.
    """
    if points is None:
        return None
    signs = {}
    for sign, demand in (("positive", max(0.0, float(rotation_max))), ("negative", max(0.0, -float(rotation_min)))):
        branch = points[sign]
        plastic = max(0.0, demand - branch["native_theta_y_rad"])
        signs[sign] = {"peak_rotation_rad": demand, "plastic_rotation_rad": plastic,
                       "capping_ratio": plastic / branch["native_dp_rad"],
                       "yielded": demand > branch["native_theta_y_rad"],
                       "past_capping": demand >= branch["native_theta_cap_rad"],
                       "past_native_cutoff": demand >= branch["native_du_rad"]}
    governing = max(signs, key=lambda s: (signs[s]["capping_ratio"], signs[s]["peak_rotation_rad"]))
    return {"version": DEMAND_VERSION, "governing_sign": governing, **{k: signs[governing][k] for k in
            ("peak_rotation_rad", "plastic_rotation_rad", "capping_ratio")},
            "yielded": any(s["yielded"] for s in signs.values()),
            "past_capping": any(s["past_capping"] for s in signs.values()),
            "past_native_cutoff": any(s["past_native_cutoff"] for s in signs.values()),
            "by_sign": signs, "points": {"native_theta_y_rad": points[governing]["native_theta_y_rad"],
                                         "native_theta_cap_rad": points[governing]["native_theta_cap_rad"],
                                         "native_du_rad": points[governing]["native_du_rad"],
                                         "member_theta_y_reference_rad": points[governing]["member_theta_y_reference_rad"],
                                         "member_theta_p_rad": points[governing]["member_theta_p_rad"],
                                         "physical_direction": points[governing].get("physical_direction")}}


def hinge_end_demand(points_by_axis, peaks_by_axis):
    """The governing spring of one member end (both bending axes): the larger capping ratio of rotation_demand.

    ``points_by_axis`` is {"y": spring_points, "z": spring_points}; ``peaks_by_axis`` is {"y": (max, min), "z": ...}.
    The axes are never mixed: each rotation is measured against its own spring.
    """
    demands = {axis: rotation_demand(points_by_axis.get(axis), *peaks_by_axis[axis])
               for axis in ("y", "z") if axis in peaks_by_axis}
    demands = {axis: d for axis, d in demands.items() if d is not None}
    if not demands:
        return None
    axis = max(demands, key=lambda a: (demands[a]["capping_ratio"], demands[a]["peak_rotation_rad"]))
    return {"governing_axis": axis, **demands[axis], "yielded": any(d["yielded"] for d in demands.values()),
            "past_capping": any(d["past_capping"] for d in demands.values()),
            "past_native_cutoff": any(d["past_native_cutoff"] for d in demands.values()), "by_axis": demands}


def define_frozen_member_imk(material_type, mat_tag, kh, positive, negative, *, lambda_haselton, c_s, c_c, c_a, c_k,
                             d_pos, d_neg, reverse, physical_directions, provenance, context=None, length_in=None,
                             k0_from_section=None, element_factor=None):
    """Install one member-end spring on the frozen calibration.

    ``positive`` and ``negative`` are the PHYSICAL member branches (MemberBranch), ``positive`` the canonical one;
    ``reverse`` places the physical negative branch on the native positive input (a beam's end j), exactly as the
    canonical baseline does. Every branch is converted once, the physical energies are computed from the canonical
    branch, and the native command is installed through Model.IMK_Materials' physical-energy mapping.
    """
    from Model import IMK_Materials as mm
    if getattr(mm, "_research_half_member_installed", False):
        raise RuntimeError("A process-local research half-member adapter is installed; it would map the frozen native "
                           "parameters a second time. Build in a clean process.")
    if material_type != "IMKPeakOriented":
        raise ValueError(f"{_context_label(context)}: {FROZEN_V2} installs IMKPeakOriented only, not {material_type!r}")
    if "member_spring_calibration" in provenance:
        raise ValueError(f"{_context_label(context)}: the provenance already records a member-to-native mapping")
    reference = half_member_reference(kh, context=context, k0_from_section=k0_from_section,
                                      element_factor=element_factor)
    native_pos, record_pos = native_branch(positive, reference, input_sign="negative" if reverse else "positive",
                                           context=context)
    native_neg, record_neg = native_branch(negative, reference, input_sign="positive" if reverse else "negative",
                                           context=context)
    modes = mm.active_energy_modes(material_type)
    energies, energy_record = frozen_energies(lambda_haselton, positive.my, reference, modes, context=context)
    # Lamda relative to the canonical branch (lambda_H theta_y_ref,E for S/C, the suppression value for A/K);
    # _install_physical_energies re-derives the native Lamda_m = Eref_m / My of whichever branch is the native
    # positive input from the energies themselves.
    base = {mode: (energy_record["lambda_haselton_dimensionless"] * energy_record["energy_reference_theta_y_rad"]
                   if mode in ("S", "C") else SUPPRESSED_MODE_LAMDA) for mode in modes}
    cyclic = mm.CyclicParameters(lamda_s=base["S"], lamda_c=base["C"], lamda_k=base["K"], lamda_a=base.get("A"),
                                 c_s=c_s, c_c=c_c, c_k=c_k, c_a=c_a, d_pos=d_pos, d_neg=d_neg)
    mapping_record = {
        "calibration_id": FROZEN_V2, "status": FROZEN_STATUS, "approval": APPROVAL,
        "numerically_equivalent_to": RESEARCH_EQUIVALENT,
        "reference": reference.record(length_in, k0_from_section),
        "branches": {"physical_positive": record_pos, "physical_negative": record_neg},
        "energy": energy_record,
        "du_policy": DU_POLICY,
        "retained_assumptions": RETAINED_ASSUMPTIONS,
        "scope": scope_record(),
    }
    mapping = {"version": FROZEN_V2, "status": FROZEN_STATUS,
               "orientation_convention": mm.CANONICAL_BASELINE_VERSION,
               "reference_direction": mm.CANONICAL_REFERENCE_DIRECTION,
               "canonical_physical_direction": physical_directions[0],
               "reference_moment_kip_in": positive.my, "base_lamda_rad": base,
               "reference_energies_kip_in_rad": energies,
               "equations": energy_record["equations"],
               "limits": ("frozen V2 modelling choice (2026-10-29), numerically equivalent to the approved research "
                          "profile; not an experimental calibration of the energy magnitude")}
    installed = mm._install_physical_energies(
        material_type, mat_tag, reference.kh, native_pos, native_neg, cyclic, energies, reverse=reverse,
        physical_directions=physical_directions,
        provenance={**provenance, "calibration_id": FROZEN_V2, "status": FROZEN_STATUS,
                    "energy_mapping_status": FROZEN_V2, "member_spring_calibration": mapping_record},
        mapping=mapping)
    return installed
