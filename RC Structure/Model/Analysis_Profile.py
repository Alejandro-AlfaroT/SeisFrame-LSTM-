"""Named nonlinear-model profiles: one explicit set of model settings, applied in one place, with an
identity that travels into the design request, and a check of the INSTALLED OpenSees domain (2026-10-01).

A profile is a small table of Structure_Parameters settings plus the state it requires and what it
declares about itself. ``apply_profile`` sets the settings (and nothing else), refuses a required state
that does not hold, and stamps ``Structure_Parameters.ANALYSIS_PROFILE_ID``. ``profile_identity`` is
what ``Design_Driver.design_request_identity`` stores under ``policy.model_profile``, so a design made
under one profile is refused under another (Verification_Integrity.validate_record), and a parent
process, its worker, the saved record and a later nonlinear build can be shown to agree.
``verify_installed_domain`` reads the live domain and the per-build registries rather than the
configuration text: member hinges present on every member, no joint spring, no face slip interface.

Profile ``v2_nonlinear_flexure_screening_v1`` (user decision 2026-10-01, "v2 we will carry over with the
nonlinear-flexure model for now, so lets just shelf imkpinching for now"): IMKPeakOriented hinges at
both ends of every beam and column, rigid centerline joints, no IMKPinching joint spring and no explicit
face slip interface. IMKPinching and its joint/interface calibration are deferred, not deleted: their
sources, fixtures and reference work stay in the repository and are simply not installed here.

What the profile does NOT do: it does not create clear spans or rigid end offsets (``rigid_centerline``
is a centerline idealisation: beams and columns meet at one node), it does not defer joint DESIGN (joint
shear, confinement, anchorage, bar fit and strong-column/weak-beam checks stay in the design loop), it
does not certify the member energy mapping (``legacy_unmapped``, provisional) and it does not release
generation. The hinge rotation it produces is the total relative spring rotation of a member calibration
that still contains Haselton's bar-slip share (a_sl = 1, Model/Deformation_Ownership); it is not
experimentally isolated plastic flexure.
"""
from __future__ import annotations

import hashlib
import json

import Structure_Parameters as sp

UNPROFILED = "unprofiled_repository_defaults"
V2_FLEXURE = "v2_nonlinear_flexure_screening_v1"
V2_FLEXURE_ANCHORED = "v2_nonlinear_flexure_anchored_energy_research_v1"

# The Structure_Parameters names every profile governs; the unprofiled identity reports the same names.
GOVERNED_KEYS = ("ELEMENT_FORMULATION", "IMK_MATERIAL_TYPE", "IMK_APPLY_TO_BEAMS", "IMK_APPLY_TO_COLUMNS",
                 "JOINT_MODEL", "NTHA_HINGE_HISTORY_STRIDE")
# State a profile is declared under; reported in the identity and checked, never set, by apply_profile.
REPORTED_STATE_KEYS = ("JOINT_DEFORMATION_SCOPE", "IMK_ENERGY_MAPPING_MODE", "IMK_CYCLIC_CALIBRATION_ID",
                       "IMK_CYCLIC_CALIBRATION_STATUS", "IMK_DETERIORATION_MODE", "IMK_USE_CALIBRATED_BACKBONE",
                       "ASCE_RISK_CATEGORY", "ASCE_IE")

PROFILES = {
    V2_FLEXURE: {
        "settings": {
            "ELEMENT_FORMULATION": "imk",
            "IMK_MATERIAL_TYPE": "IMKPeakOriented",
            "IMK_APPLY_TO_BEAMS": True,
            "IMK_APPLY_TO_COLUMNS": True,
            "JOINT_MODEL": "rigid_centerline",
            # Full-rate hinge histories for the first screening runs (screening decision 2026-10-01).
            "NTHA_HINGE_HISTORY_STRIDE": 1,
        },
        "required_state": {
            # The corrected member energy mapping stays disabled until experimentally supported
            # energies exist (user decision 2026-09-24); this profile neither enables nor certifies it.
            "IMK_ENERGY_MAPPING_MODE": "legacy_unmapped",
            "ASCE_RISK_CATEGORY": "III",
        },
        "explicit_face_slip_interfaces": False,
        "joint_springs": False,
        "declarations": {
            "scope": "nonlinear member-flexure baseline for V2 screening; diagnostic, not production acceptance",
            "joint_idealisation": ("rigid_centerline: beams and columns meet at one node; no joint spring, no clear spans, "
                                   "no rigid end offsets, no elastic panel model"),
            "joint_design": ("joint shear, confinement, anchorage, bar fit and strong-column/weak-beam checks remain active "
                             "in the design loop; deferring nonlinear joint response does not defer joint design"),
            "deferred": "IMKPinching joint springs and explicit face slip interfaces, with their calibration",
            "slip": ("the member hinge keeps Haselton's bar-slip share (a_sl = 1) because no face interface owns it; the "
                     "recorded quantity is total relative spring rotation, not isolated plastic flexure"),
            "member_energy": ("legacy_unmapped mapping, provisional cyclic calibration; coordinate-orientation discrepancies "
                              "of the member energy anchor are unresolved and not suppressed by this profile"),
            "fidelity": ("the benefit of calibrated pinching has not been established by a matched comparison; deferral does "
                         "not establish that its effect is negligible"),
            "release": "GENERATION_RELEASE_READY stays False; a profile is not a qualification",
        },
    },
}

# The same model with the provisional member energy anchor (pre-generation review, 2026-10-04). It is a
# separate profile so that a design or a run made under the legacy mapping is never read as this one.
PROFILES[V2_FLEXURE_ANCHORED] = {
    "settings": {**PROFILES[V2_FLEXURE]["settings"], "IMK_ENERGY_MAPPING_MODE": "provisional_min_yield_anchor_v1"},
    "required_state": {"ASCE_RISK_CATEGORY": "III"},
    "explicit_face_slip_interfaces": False,
    "joint_springs": False,
    "declarations": {
        **PROFILES[V2_FLEXURE]["declarations"],
        "scope": ("nonlinear member-flexure model for a declared diagnostic research batch; not production acceptance, "
                  "and a surrogate trained on it emulates this specified model, not physical RC structures"),
        "member_energy": ("provisional_min_yield_anchor_v1: every mode's reference energy is the base Lamda times the "
                          "smaller physical yield moment of the spring, so it is independent of the spring's sign "
                          "convention; a modelling convention, not an experimental calibration, an ACI requirement or a "
                          "proven bound on response; cyclic calibration stays provisional"),
        "columns": ("column hinges use a fixed strength at the gravity axial load on two independent bending axes; "
                    "changing axial load and coupled P-M-M response are not represented"),
    },
}


def _value(key):
    if key == "NTHA_HINGE_HISTORY_STRIDE":
        return int(getattr(sp, key, 8))
    return getattr(sp, key, None)


def active_profile_id():
    return getattr(sp, "ANALYSIS_PROFILE_ID", UNPROFILED) or UNPROFILED


def _require_known(profile_id):
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown analysis profile {profile_id!r}; known profiles: {sorted(PROFILES)}.")
    return PROFILES[profile_id]


def configuration_problems(profile_id=None):
    """Why the current Structure_Parameters state is not the named (default: active) profile; empty when it is."""
    profile_id = profile_id or active_profile_id()
    if profile_id == UNPROFILED:
        return []
    profile = _require_known(profile_id)
    problems = [f"{key} is {_value(key)!r}, the profile sets {want!r}"
                for key, want in profile["settings"].items() if _value(key) != want]
    problems += [f"{key} is {_value(key)!r}, the profile requires {want!r}"
                 for key, want in profile["required_state"].items() if _value(key) != want]
    if active_profile_id() != profile_id:
        problems.append(f"ANALYSIS_PROFILE_ID is {active_profile_id()!r}, not {profile_id!r}")
    return problems


def apply_profile(profile_id):
    """Set the profile's settings on Structure_Parameters, check its required state, stamp its id.

    Call it after the geometry, site and saved design are applied and before a model is built, in every
    process that builds or identifies a model (launcher, worker, later diagnostic). Returns the identity.
    """
    profile = _require_known(profile_id)
    unmet = [f"{key} is {_value(key)!r}, the profile requires {want!r}"
             for key, want in profile["required_state"].items() if _value(key) != want]
    if unmet:
        raise ValueError(f"Profile {profile_id} cannot be applied: " + "; ".join(unmet))
    for key, value in profile["settings"].items():
        setattr(sp, key, value)
    sp.ANALYSIS_PROFILE_ID = profile_id
    problems = configuration_problems(profile_id)
    if problems:                                           # cannot happen unless a setter was shadowed
        raise RuntimeError(f"Profile {profile_id} was applied but the state disagrees: " + "; ".join(problems))
    return profile_identity(profile_id)


def profile_identity(profile_id=None):
    """Identity of the active (or named) profile from the CURRENT Structure_Parameters state.

    A named profile whose state does not hold raises: an identity is a statement about what is
    configured, not about what was asked for. The unprofiled identity reports the same governed and
    reported names under UNPROFILED, so a change of any of them still changes the design identity.
    """
    profile_id = profile_id or active_profile_id()
    if profile_id != UNPROFILED:
        problems = configuration_problems(profile_id)
        if problems:
            raise ValueError(f"Profile {profile_id} is not the configured state: " + "; ".join(problems))
    payload = {"id": profile_id,
               "settings": {key: _value(key) for key in GOVERNED_KEYS},
               "state": {key: _value(key) for key in REPORTED_STATE_KEYS},
               "explicit_face_slip_interfaces": (PROFILES[profile_id]["explicit_face_slip_interfaces"]
                                                 if profile_id in PROFILES else None),
               "joint_springs": (PROFILES[profile_id]["joint_springs"] if profile_id in PROFILES
                                 else _value("JOINT_MODEL") == "imk_pinching_scissors"),
               "declarations": PROFILES[profile_id]["declarations"] if profile_id in PROFILES else {}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def expected_member_count():
    n_col = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * (sp.NUM_BAY_Y + 1)
    n_beam_x = sp.NUM_FLOOR * sp.NUM_BAY_X * (sp.NUM_BAY_Y + 1)
    n_beam_y = sp.NUM_FLOOR * (sp.NUM_BAY_X + 1) * sp.NUM_BAY_Y
    return {"column": n_col, "beam_x": n_beam_x, "beam_y": n_beam_y}


def installed_topology():
    """What the live domain and the per-build registries actually contain. Reads only; no judgement."""
    import openseespy.opensees as ops
    from Model.IMK_Hinges import hinge_registry, hinge_element_tag
    from Model.Joint_Springs import joint_registry
    from Model import Deformation_Ownership as own

    try:
        tags = set(int(t) for t in ops.getEleTags())
    except Exception:                                      # noqa: BLE001 -- no domain is an observation, not an error here
        tags = set()
    hinge_base = sp.IMK_HINGE_ELEMENT_TAG_BASE
    joint_base = getattr(sp, "JOINT_ELEMENT_TAG_BASE", None)
    registry = hinge_registry()
    member_types = {}
    material_types = {}
    missing_springs = []
    for ele_tag, entry in registry.items():
        member_types[entry.get("member_type")] = member_types.get(entry.get("member_type"), 0) + 1
        material_types[entry.get("material_type")] = material_types.get(entry.get("material_type"), 0) + 1
        for end_id in (1, 2):
            if hinge_element_tag(int(ele_tag), end_id) not in tags:
                missing_springs.append([int(ele_tag), "i" if end_id == 1 else "j"])
    hinge_elements = sorted(t for t in tags if t >= hinge_base)
    expected_hinge_tags = {hinge_element_tag(int(e), k) for e in registry for k in (1, 2)}
    joint_elements = sorted(t for t in tags if joint_base is not None and joint_base <= t < hinge_base)

    def tally(table, key):
        table[str(key)] = table.get(str(key), 0) + 1

    slip_owners, panel_owners, slip_indicators, installed_material_types, energy_modes = {}, {}, {}, {}, {}
    members_framing_into_a_beam_core = 0
    for entry in registry.values():
        tally(energy_modes, entry.get("energy_mapping_mode"))
        if entry.get("end_node_i") != entry.get("node_i") or entry.get("end_node_j") != entry.get("node_j"):
            members_framing_into_a_beam_core += 1
        for end in ("i", "j"):
            owned = (entry.get("deformation_ownership") or {}).get(end) or {}
            tally(slip_owners, owned.get("bar_slip")); tally(panel_owners, owned.get("panel_shear"))
            tally(slip_indicators, owned.get("bond_slip_indicator"))
            for material in ((entry.get("installed_materials") or {}).get(end) or {}).values():
                material = material or {}
                tally(installed_material_types, material.get("material_type") or (material.get("command") or [None])[0])
    core_base = getattr(sp, "JOINT_BEAM_CORE_TAG_BASE", None)
    try:
        beam_core_nodes = sorted(int(t) for t in ops.getNodeTags()
                                 if core_base is not None and core_base <= int(t) < sp.IMK_HINGE_NODE_TAG_BASE)
    except Exception:                                      # noqa: BLE001
        beam_core_nodes = []
    return {
        "installed_spring_material_types": installed_material_types,
        "member_energy_mapping_modes": energy_modes,
        "bar_slip_owner_by_end": slip_owners,
        "panel_shear_owner_by_end": panel_owners,
        "bond_slip_indicator_by_end": slip_indicators,
        "members_framing_into_a_beam_core": members_framing_into_a_beam_core,
        "joint_beam_core_nodes_in_domain": len(beam_core_nodes),
        "domain_element_count": len(tags),
        "members_with_hinges": len(registry),
        "members_with_hinges_by_type": member_types,
        "member_material_types": material_types,
        "hinge_spring_elements_in_domain": len(hinge_elements),
        "hinge_spring_elements_expected": len(expected_hinge_tags),
        "hinge_springs_missing_from_domain": missing_springs[:20],
        "hinge_range_elements_not_in_registry": sorted(set(hinge_elements) - expected_hinge_tags)[:20],
        "joint_spring_elements_in_domain": len(joint_elements),
        "joint_spring_element_tags": joint_elements[:20],
        "joint_registry_entries": len(joint_registry()),
        "face_slip_interfaces_registered": len(own.slip_interfaces()),
        "slip_domain": own.slip_domain(),
        "joint_model_configured": getattr(sp, "JOINT_MODEL", None),
        "joint_springs_installed": bool(joint_elements) or bool(joint_registry()) or bool(beam_core_nodes),
    }


def verify_installed_domain(profile_id=None, *, raise_on_failure=False):
    """Check the installed model against the profile: member hinges on every member, no joint spring,
    no face slip interface, the profile's member material. Returns the topology with ``problems``."""
    profile_id = profile_id or active_profile_id()
    topology = installed_topology()
    problems = list(configuration_problems(profile_id))
    if profile_id != UNPROFILED:
        profile = _require_known(profile_id)
        expected = expected_member_count()
        by_type = topology["members_with_hinges_by_type"]
        wanted = {"column": expected["column"] if profile["settings"]["IMK_APPLY_TO_COLUMNS"] else 0,
                  "beam_x": expected["beam_x"] if profile["settings"]["IMK_APPLY_TO_BEAMS"] else 0,
                  "beam_y": expected["beam_y"] if profile["settings"]["IMK_APPLY_TO_BEAMS"] else 0}
        for member_type, count in wanted.items():
            if by_type.get(member_type, 0) != count:
                problems.append(f"{by_type.get(member_type, 0)} {member_type} members carry hinges, the geometry has {count}")
        if topology["hinge_spring_elements_in_domain"] != topology["hinge_spring_elements_expected"] or topology["hinge_springs_missing_from_domain"]:
            problems.append(f"{topology['hinge_spring_elements_in_domain']} hinge springs are in the domain, the registry expects "
                            f"{topology['hinge_spring_elements_expected']}")
        if topology["hinge_range_elements_not_in_registry"]:
            problems.append(f"elements in the hinge tag range are not member hinges: {topology['hinge_range_elements_not_in_registry']}")
        wanted_material = profile["settings"]["IMK_MATERIAL_TYPE"]
        wrong_material = {k: v for k, v in {**topology["member_material_types"], **topology["installed_spring_material_types"]}.items()
                          if k != wanted_material}
        if wrong_material:
            problems.append(f"member hinges of another material are installed: {wrong_material}")
        if not profile["joint_springs"]:
            # No spring owns panel shear and every member end keeps its own bar-slip share (a_sl = 1).
            if set(topology["panel_shear_owner_by_end"]) - {"None"}:
                problems.append(f"member ends report a panel-shear owner: {topology['panel_shear_owner_by_end']}")
            if topology["members_framing_into_a_beam_core"] or topology["joint_beam_core_nodes_in_domain"]:
                problems.append(f"{topology['members_framing_into_a_beam_core']} members frame into a joint beam core "
                                f"({topology['joint_beam_core_nodes_in_domain']} core nodes in the domain)")
        if not profile["explicit_face_slip_interfaces"]:
            if set(topology["bar_slip_owner_by_end"]) - {"member_hinge"}:
                problems.append(f"bar slip is not owned by the member hinge at every end: {topology['bar_slip_owner_by_end']}")
            if set(topology["bond_slip_indicator_by_end"]) - {"1.0"}:
                problems.append(f"the member bond-slip indicator is not 1 at every end: {topology['bond_slip_indicator_by_end']}")
        required_mode = {**profile["settings"], **profile["required_state"]}.get("IMK_ENERGY_MAPPING_MODE")
        if required_mode and set(topology["member_energy_mapping_modes"]) - {required_mode}:
            problems.append(f"member energy mapping modes installed: {topology['member_energy_mapping_modes']}")
        if not profile["joint_springs"] and topology["joint_springs_installed"]:
            problems.append(f"{topology['joint_spring_elements_in_domain']} joint spring elements / "
                            f"{topology['joint_registry_entries']} joint registry entries are installed; the profile installs none")
        if not profile["explicit_face_slip_interfaces"] and topology["face_slip_interfaces_registered"]:
            problems.append(f"{topology['face_slip_interfaces_registered']} face slip interfaces are registered; the profile has none")
    result = {"profile_id": profile_id, "consistent": not problems, "problems": problems, **topology,
              "basis": "live OpenSees domain element tags and the per-build hinge, joint and slip registries; not the configuration text"}
    if problems and raise_on_failure:
        raise RuntimeError(f"Installed model does not match profile {profile_id}: " + "; ".join(problems))
    return result
