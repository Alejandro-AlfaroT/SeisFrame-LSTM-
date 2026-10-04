"""Per-end ownership of the deformation mechanisms at member ends (2026-09-27).

The slip-calibration resolution replaced the global slip-removal switch with accounting at
each physical member end: every mechanism (flexure, bar/anchorage slip, panel shear) has exactly
one owner at each end, a declared joint scope is not an owner, and a column base that has no
replacement interface never loses its slip contribution silently.

Policy (POLICY_ID): the member hinge keeps Haselton's included slip scope (a_sl = 1, the legacy
member calibration) unless a face slip interface is REGISTERED for that end, in which case the
member calibration excludes slip (a_sl = 0) and the interface owns it. A registered interface that
is only provisional makes the decomposition "diagnostic_unresolved"; it is recorded, not certified.
The panel spring (Model/Joint_Springs) owns panel shear only, whatever its declared scope says.

Nothing here supplies calibration: registering an interface is what the diagnostic subassembly does
in-process; the production frame registers none, so every production hinge keeps a_sl = 1.

Domain scope (Unit 1 review, 2026-09-28, commit blocker 1): the registry belongs to one OpenSees
domain. Every builder calls begin_domain() right after ops.wipe() and before any member is created, which
discards (and counts) registrations left from a previous domain; a registration names the element that
realises the interface, and it is honoured only while that element is in the current domain. A builder
that installs no face interfaces (the production frame) validates that nothing is registered once its
members exist. Registrations are never cleared in the middle of installing an interface assembly.
"""
from __future__ import annotations

import Structure_Parameters as sp

POLICY_ID = "member_hinge_unless_registered_face_interface_v1"
MECHANISMS = ("flexure", "bar_slip", "panel_shear")
END_LOCATIONS = ("column_base", "floor_joint", "roof_joint", "diagnostic_fixture")
INTERFACE_STATUSES = ("calibrated", "provisional_diagnostic_only")
MEMBER_SCOPES = ("flexure_and_slip", "flexure_only")
DEFAULT_MEMBER_INDICATOR = 1.0            # a_sl of the legacy member calibration (slip included)
LEGACY_UNOWNED_OVERRIDE = "legacy_unowned_slip_removed"   # diagnostic reproduction of the 2026-09-27 frame state

_INTERFACES: dict = {}
_DOMAIN: dict = {"generation": 0, "owner": None, "stale_discarded": 0, "stale_records": []}


# ---- registry of face slip interfaces --------------------------------------------------------------
def begin_domain(owner):
    """Open the registry's scope for a new OpenSees domain: called by every builder right after
    ops.wipe() and before any member is created. Registrations from the previous domain are discarded,
    counted and listed by identity in the domain record, so a builder that registered before the wipe
    is visible rather than silently forgotten."""
    stale = [{k: r[k] for k in ("member_tag", "end", "axis", "calibration_id", "generation")} for r in _INTERFACES.values()]
    _INTERFACES.clear()
    _DOMAIN.update({"generation": _DOMAIN["generation"] + 1, "owner": str(owner), "stale_discarded": len(stale),
                    "stale_records": stale})
    return slip_domain()


def slip_domain():
    return {"generation": _DOMAIN["generation"], "owner": _DOMAIN["owner"], "stale_discarded": _DOMAIN["stale_discarded"],
            "stale_records": [dict(r) for r in _DOMAIN["stale_records"]]}


def reset_slip_interfaces():
    """Clear the registry without opening a new domain (test fixtures; the end of a diagnostic run)."""
    _INTERFACES.clear()


def slip_interfaces():
    return dict(_INTERFACES)


def validate_installed(*, expect_none=False):
    """Every registration must name an element of the current OpenSees domain. A builder that installs
    no face interfaces passes expect_none=True: any registration is then a stale or misplaced claim and
    is refused. Returns the number of registered interfaces."""
    import openseespy.opensees as ops
    tags = {int(t) for t in ops.getEleTags()}
    problems = []
    for record in _INTERFACES.values():
        where = f"member {record['member_tag']} end {record['end']} ({record['calibration_id']})"
        if expect_none:
            problems.append(f"{where} is registered in a build that installs no face interfaces")
        elif record["element_tag"] is None:
            problems.append(f"{where} names no interface element")
        elif int(record["element_tag"]) not in tags:
            problems.append(f"{where} claims element {record['element_tag']}, absent from the domain")
    if problems:
        raise ValueError("slip interface registry does not match the OpenSees domain: " + "; ".join(problems))
    return len(_INTERFACES)


def _key(member_tag, end, axis):
    if end not in ("i", "j"):
        raise ValueError(f"end must be 'i' or 'j', got {end!r}")
    if axis not in ("rot_y", "rot_z", "both"):
        raise ValueError(f"axis must be 'rot_y', 'rot_z' or 'both', got {axis!r}")
    return (int(member_tag), end, axis)


def register_slip_interface(member_tag, end, axis="both", *, calibration_id, status, scope="member_face_bar_slip_rotation",
                            evidence=None, element_tag=None):
    """Declare that a face slip interface at (member, end, axis) owns bar slip there.

    ``status`` is "calibrated" only for a reviewed calibration; the subassembly registers
    "provisional_diagnostic_only". ``element_tag`` is the OpenSees element that realises the interface:
    a registration is honoured only while that element is in the current domain (a builder registers
    and installs within the same build). Registering the same end twice, or one bending axis while the
    member hinge shares a backbone across both axes, is refused rather than resolved silently.
    """
    if status not in INTERFACE_STATUSES:
        raise ValueError(f"interface status must be one of {INTERFACE_STATUSES}, got {status!r}")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("a slip interface needs a calibration_id")
    if axis != "both":
        raise NotImplementedError("a member hinge shares one backbone across rot_y and rot_z; per-axis "
                                  "interfaces need per-axis backbones before they can own slip on one axis")
    key = _key(member_tag, end, axis)
    if key in _INTERFACES:
        raise ValueError(f"duplicate slip interface at member {member_tag} end {end}: {_INTERFACES[key]['calibration_id']}")
    record = {"member_tag": int(member_tag), "end": end, "axis": axis, "calibration_id": calibration_id,
              "status": status, "scope": scope, "evidence": evidence,
              "element_tag": None if element_tag is None else int(element_tag), "generation": _DOMAIN["generation"]}
    _INTERFACES[key] = record
    return dict(record)


def registered_slip_interface(member_tag, end, axis="both"):
    """The registration at this end, if it belongs to the current domain and its element is installed."""
    record = _INTERFACES.get(_key(member_tag, end, axis))
    if record is None:
        return None
    if record["generation"] != _DOMAIN["generation"]:
        raise ValueError(f"slip interface at member {member_tag} end {end} was registered in domain generation "
                         f"{record['generation']}, the current domain is {_DOMAIN['generation']}")
    if record["element_tag"] is not None:
        import openseespy.opensees as ops
        if int(record["element_tag"]) not in {int(t) for t in ops.getEleTags()}:
            raise ValueError(f"slip interface at member {member_tag} end {end} claims element {record['element_tag']}, "
                             "which is not in the current OpenSees domain")
    return record


# ---- member ends -----------------------------------------------------------------------------------
def floor_index_of_node(node_tag):
    """Floor index (0 = base) of a frame node from its elevation."""
    import openseespy.opensees as ops
    z = ops.nodeCoord(int(node_tag))[2]
    return int(round(z / sp.STORY_H)) if sp.STORY_H > 0 else 0


def end_location(member_type, floor_index):
    """Physical location of a member end that frames into the joint at ``floor_index``.

    Only a column end at the base level is a column base. The frame never puts a beam at the base
    level or a member above the roof; isolated test fixtures do (a cantilever at z = 0), and those
    ends are classed as floor joints rather than refused, since nothing at that end is a footing.
    """
    if floor_index <= 0:
        return "column_base" if member_type == "column" else "floor_joint"
    if floor_index >= sp.NUM_FLOOR:
        return "roof_joint"
    return "floor_joint"


def member_end(member_tag, member_type, end, joint_node=None, *, floor_index=None, location=None, axis="both"):
    """Descriptor of one physical member end; the location comes from the joint node unless given."""
    if location is None:
        if floor_index is None:
            if joint_node is None:
                raise ValueError("a member end needs its joint node, a floor index or an explicit location")
            floor_index = floor_index_of_node(joint_node)
        location = end_location(member_type, floor_index)
    if location not in END_LOCATIONS:
        raise ValueError(f"unknown end location {location!r}")
    if end not in ("i", "j"):
        raise ValueError(f"end must be 'i' or 'j', got {end!r}")
    return {"member_tag": int(member_tag), "member_type": member_type, "end": end,
            "joint_node": None if joint_node is None else int(joint_node), "floor_index": floor_index,
            "location": location, "axis": axis}


# ---- ownership -------------------------------------------------------------------------------------
def slip_owner(end):
    """'face_interface' when an interface is registered at this end, else 'member_hinge'."""
    return "face_interface" if registered_slip_interface(end["member_tag"], end["end"], end["axis"]) else "member_hinge"


def bond_slip_indicator_for_end(end, *, member_scope=None):
    """Haselton's a_sl for this end under the policy; an explicit member scope that claims slip an
    interface already owns is a duplicate and is refused."""
    owner = slip_owner(end)
    if member_scope == LEGACY_UNOWNED_OVERRIDE:
        if owner == "face_interface":
            raise ValueError("the legacy unowned override cannot coexist with a registered interface")
        return 0.0
    if member_scope is not None and member_scope not in MEMBER_SCOPES:
        raise ValueError(f"member scope must be one of {MEMBER_SCOPES}, got {member_scope!r}")
    if member_scope == "flexure_and_slip" and owner == "face_interface":
        raise ValueError(f"duplicate slip scope at member {end['member_tag']} end {end['end']}: the member hinge and a "
                         "registered face interface both claim bar slip")
    if member_scope == "flexure_only" and owner == "member_hinge":
        raise ValueError(f"unowned slip at member {end['member_tag']} end {end['end']}: the member hinge excludes bar "
                         "slip but no face interface is registered")
    return 0.0 if owner == "face_interface" else DEFAULT_MEMBER_INDICATOR


def ownership(end, *, joint_spring_present=None, member_scope=None):
    """The complete ownership record for one end: one owner per mechanism, the member and interface
    scopes, the calibration identity of the interface and the status of the decomposition."""
    if joint_spring_present is None:
        joint_spring_present = (getattr(sp, "JOINT_MODEL", "rigid_centerline") == "imk_pinching_scissors"
                                and end["location"] in ("floor_joint", "roof_joint"))
    interface = registered_slip_interface(end["member_tag"], end["end"], end["axis"])
    indicator = bond_slip_indicator_for_end(end, member_scope=member_scope)
    owner = "face_interface" if interface else ("none" if indicator == 0.0 else "member_hinge")
    if interface is None:
        status = "legacy_included_slip_scope" if indicator == 1.0 else "slip_removed_without_owner_legacy_error"
    elif interface["status"] == "calibrated":
        status = "calibrated_separate_interface"
    else:
        status = "diagnostic_unresolved"
    return {"policy": POLICY_ID, "member_tag": end["member_tag"], "member_type": end["member_type"], "end": end["end"],
            "location": end["location"], "floor_index": end["floor_index"], "axis": end["axis"],
            "flexure": "member_hinge", "bar_slip": owner,
            "panel_shear": "joint_spring" if joint_spring_present else None,
            "member_deformation_scope": "flexure_and_slip" if indicator == 1.0 else "flexure_only",
            "interface_present": interface is not None,
            "interface_scope": None if interface is None else interface["scope"],
            "interface_calibration_id": None if interface is None else interface["calibration_id"],
            "interface_status": None if interface is None else interface["status"],
            "bond_slip_indicator": indicator, "decomposition_status": status,
            "evidence": None if interface is None else interface.get("evidence")}


def validate_joint_scope():
    """A declared joint scope never owns slip; the retired 'joint_shear_and_slip' value is refused."""
    scope = getattr(sp, "JOINT_DEFORMATION_SCOPE", "joint_shear_only")
    if scope == "joint_shear_and_slip":
        raise ValueError("JOINT_DEFORMATION_SCOPE 'joint_shear_and_slip' was retired on 2026-09-27: a declared scope "
                         "is not an owner of bar slip. Use 'joint_shear_only' and register calibrated face interfaces "
                         "(Model/Deformation_Ownership) for the ends whose member hinges are to drop it.")
    if scope != "joint_shear_only":
        raise ValueError(f"unknown JOINT_DEFORMATION_SCOPE {scope!r}")
    return scope


INVENTORY_COLUMNS = ("member_tag", "member_type", "end", "location", "floor_index", "axis", "flexure", "bar_slip",
                     "panel_shear", "member_deformation_scope", "interface_present", "interface_scope",
                     "interface_calibration_id", "interface_status", "bond_slip_indicator", "decomposition_status",
                     "theta_p", "policy")


def end_inventory(hinge_registry):
    """One row per member end from the hinge registry's recorded ownership (Model/IMK_Hinges)."""
    rows = []
    for tag in sorted(hinge_registry):
        entry = hinge_registry[tag]
        for end in ("i", "j"):
            record = (entry.get("deformation_ownership") or {}).get(end)
            if record is None:
                continue
            theta_p = ((entry.get("backbone_by_end") or {}).get(end) or {}).get("theta_p", entry.get("theta_p"))
            rows.append({**{column: record.get(column) for column in INVENTORY_COLUMNS if column != "theta_p"},
                         "theta_p": theta_p})
    return rows
