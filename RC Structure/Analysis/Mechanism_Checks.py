import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Data_Generation.Graph_Exporter import collect_element_rows
from Design.ACI_Checks import build_pm_diagram, check_column_pm
from Design.Config import DesignConfig
from Model.Build_Model import floor_master_node
from Model import IMK_Spring_Mapping as spring_mapping
from Model.IMK_Hinges import hinge_backbone, hinge_element_tag, imk_hinge_thresholds, installed_spring_points


ROOF_DRIFT_TARGETS = (0.045, 0.099)

# v2 (frozen IMK V2 calibration, 2026-10-29): every spring is measured against its own installed native points for
# the sign of the current rotation (yield My/Kh, capping My/Kh + dp, retained du), replacing one member-level
# theta_y + flat end-i theta_p and a flat theta_u for both signs and ends; the du event ultimate_exceeded is renamed
# native_cutoff_du_exceeded (and the summary keys with it); released (elastic) ends are listed as untracked; outside
# the IMK formulation no spring is tracked, and under the frozen calibration an IMK member missing from the hinge
# registry is an error, not an untracked spring.
MECHANISM_DIAGNOSTICS_VERSION = "mechanism_diagnostics_v2_installed_native_spring_points"
NATIVE_SIGNS = ("positive", "negative")
# Each hinge event and the per-sign native threshold it is measured against (Model/IMK_Spring_Mapping.spring_points).
# The retained du is a provisional numerical cutoff, not a collapse rotation or a material failure.
HINGE_EVENT_THRESHOLDS = (
    ("yielded", "native_theta_y_rad"),
    ("capped", "native_theta_cap_rad"),
    ("native_cutoff_du_exceeded", "native_du_rad"),
)
_SIGN_POINT_KEYS = (
    "physical_direction",
    "native_theta_y_rad",
    "native_dp_rad",
    "native_theta_cap_rad",
    "native_du_rad",
    "member_theta_p_rad",
    "member_theta_pc_rad",
    "member_theta_y_reference_rad",
    "member_theta_cap_reference_rad",
)


def _safe_ele_response(ele_tag, *args):
    try:
        response = ops.eleResponse(ele_tag, *args)
    except Exception:
        return None

    if response is None:
        return None

    return list(response)


def _floor_ux_values():
    return {
        floor: ops.nodeDisp(floor_master_node(floor), 1)
        for floor in range(1, sp.NUM_FLOOR + 1)
    }


def collect_story_drift_snapshot(step, roof_disp, base_shear):
    floor_ux = _floor_ux_values()
    story_rows = []
    lower_ux = 0.0

    for floor in range(1, sp.NUM_FLOOR + 1):
        upper_ux = floor_ux[floor]
        drift = (upper_ux - lower_ux) / sp.STORY_H
        story_rows.append(
            {
                "story": floor,
                "lower_floor": floor - 1,
                "upper_floor": floor,
                "floor_pair": f"{floor - 1}-{floor}",
                "lower_ux_in": lower_ux,
                "upper_ux_in": upper_ux,
                "story_drift_ratio": drift,
                "abs_story_drift_ratio": abs(drift),
            }
        )
        lower_ux = upper_ux

    max_story = max(story_rows, key=lambda row: row["abs_story_drift_ratio"])

    return {
        "step": step,
        "roof_disp_in": roof_disp,
        "roof_drift_ratio": roof_disp / (sp.NUM_FLOOR * sp.STORY_H),
        "base_shear_kip": base_shear,
        "max_story": max_story["story"],
        "max_floor_pair": max_story["floor_pair"],
        "max_lower_floor": max_story["lower_floor"],
        "max_upper_floor": max_story["upper_floor"],
        "max_story_drift_ratio": max_story["story_drift_ratio"],
        "max_abs_story_drift_ratio": max_story["abs_story_drift_ratio"],
        "story_drifts": story_rows,
        "story_drift_ranking": sorted(
            story_rows,
            key=lambda row: row["abs_story_drift_ratio"],
            reverse=True,
        ),
    }


def _installed_thresholds(points):
    """The installed spring's own native points per native sign (frozen V2 and legacy replay alike)."""
    reference = points.get("member_reference") or {}
    return {
        "threshold_source": "installed_spring_points",
        "spring_points_version": points["version"],
        "Kh_kip_in_per_rad": points["Kh_kip_in_per_rad"],
        "K0_kip_in_per_rad": reference.get("K0_kip_in_per_rad"),
        "Kb_kip_in_per_rad": reference.get("Kb_kip_in_per_rad"),
        "thresholds_by_sign": {
            sign: {key: points[sign].get(key) for key in _SIGN_POINT_KEYS}
            for sign in NATIVE_SIGNS
        },
    }


def _legacy_global_thresholds(element_type, rot_dir, length):
    """The global IMK_THETA_* thresholds, one value for both signs (legacy replay only)."""
    thresholds = imk_hinge_thresholds(element_type, rot_dir, length)
    theta_p = thresholds["theta_cap"] - thresholds["theta_y"]
    by_sign = {key: None for key in _SIGN_POINT_KEYS}
    # Under the legacy replay the member interval is installed as the native one.
    by_sign.update(
        native_theta_y_rad=thresholds["theta_y"],
        native_dp_rad=theta_p,
        native_theta_cap_rad=thresholds["theta_cap"],
        native_du_rad=thresholds["theta_u"],
        member_theta_p_rad=theta_p,
    )
    return {
        "threshold_source": "legacy_global_imk_hinge_thresholds",
        "spring_points_version": None,
        "Kh_kip_in_per_rad": thresholds["selected_stiffness"],
        "K0_kip_in_per_rad": None,
        "Kb_kip_in_per_rad": None,
        "thresholds_by_sign": {sign: dict(by_sign) for sign in NATIVE_SIGNS},
    }


def _imk_hinge_member_rows(element_rows, untracked=None):
    """One row per member-end spring (end x bending axis), with the thresholds of its own installed material.

    The thresholds are the installed spring's native points per native sign
    (Model/IMK_Hinges.installed_spring_points): yield My/Kh, capping
    My/Kh + dp and the retained cutoff du. They differ by end, axis and sign
    (asymmetric beam ends), and under the frozen V2 calibration (2026-10-29)
    the native dp is converted from the member theta_p and differs from it.
    A released (elastic) gravity-beam end has no IMK spring and is not
    tracked. Outside the IMK formulation (ELEMENT_FORMULATION != "imk") the
    members carry no zeroLength springs at all and no row is returned. The
    global IMK_THETA_* thresholds (imk_hinge_thresholds) serve only a member
    the registry never recorded, and only under the legacy replay; under the
    frozen calibration an IMK build registers every member it gives hinges
    (Model/Build_Model), so a missing entry is a stale or reset registry and
    raises. Untracked springs are appended to ``untracked`` with their reason
    when a list is passed.
    """
    rows = []
    untracked = [] if untracked is None else untracked
    if sp.ELEMENT_FORMULATION != "imk":
        return rows
    frozen = spring_mapping.is_frozen()

    for element in element_rows:
        element_type = element["element_type"]

        if element_type == "column" and not sp.IMK_APPLY_TO_COLUMNS:
            continue

        if element_type in {"beam_x", "beam_y"} and not sp.IMK_APPLY_TO_BEAMS:
            continue

        installed = hinge_backbone(element["ele_tag"])
        calibration = (installed or {}).get("member_spring_calibration") or spring_mapping.active_calibration()
        for end_id, end_name, node_key in ((1, "i", "node_i"), (2, "j", "node_j")):
            hinge_tag = hinge_element_tag(element["ele_tag"], end_id)
            for response_index, rot_dir in ((0, "rot_y"), (1, "rot_z")):
                axis = "y" if rot_dir == "rot_y" else "z"
                reason = None
                if installed is not None:
                    points = installed_spring_points(element["ele_tag"], end_name, axis)
                    if points is None:
                        reason = "released_or_elastic_end"
                    else:
                        thresholds = _installed_thresholds(points)
                elif frozen:
                    raise ValueError(
                        f"Element {element['ele_tag']} ({element_type}) has no hinge-registry entry under "
                        f"{spring_mapping.active_calibration()}: an IMK build registers every member it gives hinges, "
                        "so the registry is stale or was reset; the mechanism checks read the installed spring with "
                        "Model.IMK_Hinges.installed_spring_points(ele_tag, end, axis)"
                    )
                else:
                    thresholds = _legacy_global_thresholds(element_type, rot_dir, element["length_in"])
                if reason is not None:
                    untracked.append(
                        {
                            "physical_ele_tag": element["ele_tag"],
                            "hinge_ele_tag": hinge_tag,
                            "end": end_name,
                            "spring_axis": axis,
                            "element_type": element_type,
                            "reason": reason,
                        }
                    )
                    continue
                rows.append(
                    {
                        "physical_ele_tag": element["ele_tag"],
                        "hinge_ele_tag": hinge_tag,
                        "node_tag": element[node_key],
                        "end": end_name,
                        "rot_dir": rot_dir,
                        "spring_axis": axis,
                        "response_index": response_index,
                        "element_type": element_type,
                        "member_spring_calibration": calibration,
                        **thresholds,
                    }
                )

    return rows


def _column_rows(element_rows):
    return [row for row in element_rows if row["element_type"] == "column"]


class MechanismTracker:
    def __init__(self):
        self.element_rows = collect_element_rows()
        self.untracked_hinge_springs = []
        self.hinge_rows = _imk_hinge_member_rows(self.element_rows, self.untracked_hinge_springs)
        self.column_rows = _column_rows(self.element_rows)
        self.design_cfg = DesignConfig()
        self.column_pm_diagram = build_pm_diagram(self.design_cfg)
        self.story_drift_snapshots = {}
        self.hinge_state = {}
        self.hinge_events = []
        self.column_yield_events = []
        self.max_hinge_rotation = None
        self.max_column_ratio = None
        self.max_interstory_drift = None
        self.interstory_drift_envelope_by_story = {}

    def _record_story_targets(self, snapshot):
        roof_drift = snapshot["roof_drift_ratio"]

        for target in ROOF_DRIFT_TARGETS:
            key = f"{100.0 * target:.1f}%"
            if key in self.story_drift_snapshots:
                continue

            if roof_drift >= target:
                self.story_drift_snapshots[key] = snapshot

    def _record_interstory_drift_envelope(self, snapshot):
        max_story = snapshot["story_drift_ranking"][0]
        record = {
            "step": snapshot["step"],
            "roof_disp_in": snapshot["roof_disp_in"],
            "roof_drift_ratio": snapshot["roof_drift_ratio"],
            "base_shear_kip": snapshot["base_shear_kip"],
            **max_story,
        }

        if (
            self.max_interstory_drift is None
            or record["abs_story_drift_ratio"]
            > self.max_interstory_drift["abs_story_drift_ratio"]
        ):
            self.max_interstory_drift = record

        for story in snapshot["story_drifts"]:
            story_key = str(story["story"])
            story_record = {
                "step": snapshot["step"],
                "roof_disp_in": snapshot["roof_disp_in"],
                "roof_drift_ratio": snapshot["roof_drift_ratio"],
                "base_shear_kip": snapshot["base_shear_kip"],
                **story,
            }
            existing = self.interstory_drift_envelope_by_story.get(story_key)
            if (
                existing is None
                or story_record["abs_story_drift_ratio"]
                > existing["abs_story_drift_ratio"]
            ):
                self.interstory_drift_envelope_by_story[story_key] = story_record

    def _record_hinge_response(self, step, roof_disp, base_shear):
        for hinge in self.hinge_rows:
            deformation = _safe_ele_response(hinge["hinge_ele_tag"], "deformation")
            if deformation is None or len(deformation) <= hinge["response_index"]:
                continue

            # Native spring rotation (hinge node minus joint), measured against
            # the installed points of its own native sign: beam ends are
            # asymmetric, so one value for both signs would misplace the events.
            theta = deformation[hinge["response_index"]]
            abs_theta = abs(theta)
            sign = "positive" if theta >= 0.0 else "negative"
            thresholds = hinge["thresholds_by_sign"][sign]
            state_key = (
                hinge["hinge_ele_tag"],
                hinge["rot_dir"],
            )
            state = self.hinge_state.setdefault(
                state_key,
                {state_name: False for state_name, _ in HINGE_EVENT_THRESHOLDS},
            )

            if self.max_hinge_rotation is None or abs_theta > self.max_hinge_rotation["abs_rotation"]:
                self.max_hinge_rotation = {
                    **hinge,
                    "step": step,
                    "roof_disp_in": roof_disp,
                    "roof_drift_ratio": roof_disp / (sp.NUM_FLOOR * sp.STORY_H),
                    "base_shear_kip": base_shear,
                    "rotation_rad": theta,
                    "abs_rotation": abs_theta,
                    "native_sign": sign,
                }

            for state_name, threshold_name in HINGE_EVENT_THRESHOLDS:
                if state[state_name]:
                    continue

                threshold = thresholds[threshold_name]
                if threshold is not None and abs_theta >= threshold:
                    state[state_name] = True
                    self.hinge_events.append(
                        {
                            **hinge,
                            "event": state_name,
                            "native_sign": sign,
                            "threshold_name": threshold_name,
                            "threshold": threshold,
                            "step": step,
                            "roof_disp_in": roof_disp,
                            "roof_drift_ratio": roof_disp / (sp.NUM_FLOOR * sp.STORY_H),
                            "base_shear_kip": base_shear,
                            "rotation_rad": theta,
                            "abs_rotation": abs_theta,
                        }
                    )

    def _record_column_yield(self, step, roof_disp, base_shear):
        for column in self.column_rows:
            forces = _safe_ele_response(column["ele_tag"], "localForce")
            if forces is None or len(forces) != 12:
                continue

            axial = forces[0]
            for end_name, node_key, offset in (
                ("i", "node_i", 0),
                ("j", "node_j", 6),
            ):
                my = forces[offset + 4]
                mz = forces[offset + 5]
                resultant_moment = math.sqrt(my * my + mz * mz)
                pm_result = check_column_pm(
                    axial,
                    mz,
                    my,
                    self.column_pm_diagram,
                    self.design_cfg,
                )
                ratio = pm_result.dcr

                column_record = {
                    "ele_tag": column["ele_tag"],
                    "node_tag": column[node_key],
                    "end": end_name,
                    "step": step,
                    "roof_disp_in": roof_disp,
                    "roof_drift_ratio": roof_disp / (sp.NUM_FLOOR * sp.STORY_H),
                    "base_shear_kip": base_shear,
                    "axial_kip": axial,
                    "moment_y_kip_in": my,
                    "moment_z_kip_in": mz,
                    "resultant_moment_kip_in": resultant_moment,
                    "pm_capacity_kip_in": pm_result.capacity,
                    "pm_dcr": ratio,
                    "ratio_y": None,
                    "ratio_z": None,
                    "max_ratio": ratio,
                }

                if self.max_column_ratio is None or ratio > self.max_column_ratio["max_ratio"]:
                    self.max_column_ratio = column_record

                yielded_key = (column["ele_tag"], end_name)
                already_yielded = any(
                    event["ele_tag"] == yielded_key[0] and event["end"] == yielded_key[1]
                    for event in self.column_yield_events
                )

                if not already_yielded and ratio >= 1.0:
                    self.column_yield_events.append(column_record)

    def record_step(self, step, roof_disp, base_shear):
        n_hinge_events_before = len(self.hinge_events)
        n_column_yield_before = len(self.column_yield_events)
        story_snapshot = collect_story_drift_snapshot(step, roof_disp, base_shear)
        self._record_story_targets(story_snapshot)
        self._record_interstory_drift_envelope(story_snapshot)
        self._record_hinge_response(step, roof_disp, base_shear)
        self._record_column_yield(step, roof_disp, base_shear)
        new_hinge_events = self.hinge_events[n_hinge_events_before:]
        new_column_yield_events = self.column_yield_events[n_column_yield_before:]

        return {
            "story_snapshot": story_snapshot,
            "max_abs_story_drift_ratio": story_snapshot["max_abs_story_drift_ratio"],
            "new_hinge_events": new_hinge_events,
            "new_hinge_yield_events": [
                event for event in new_hinge_events if event["event"] == "yielded"
            ],
            "new_hinge_cap_events": [
                event for event in new_hinge_events if event["event"] == "capped"
            ],
            "new_hinge_native_cutoff_du_events": [
                event for event in new_hinge_events if event["event"] == "native_cutoff_du_exceeded"
            ],
            "new_column_yield_events": new_column_yield_events,
        }

    def summary(self):
        yielded_events = [event for event in self.hinge_events if event["event"] == "yielded"]
        capped_events = [event for event in self.hinge_events if event["event"] == "capped"]
        cutoff_events = [
            event for event in self.hinge_events if event["event"] == "native_cutoff_du_exceeded"
        ]
        requested_story_drift_targets = [
            f"{100.0 * target:.1f}%" for target in ROOF_DRIFT_TARGETS
        ]
        unreached_story_drift_targets = [
            target
            for target in requested_story_drift_targets
            if target not in self.story_drift_snapshots
        ]

        first_yield = yielded_events[0] if yielded_events else None
        first_cap = capped_events[0] if capped_events else None
        first_cutoff = cutoff_events[0] if cutoff_events else None
        first_yield_group = self._first_event_group(yielded_events)
        first_cap_group = self._first_event_group(capped_events)
        first_cutoff_group = self._first_event_group(cutoff_events)
        untracked_by_reason = {}
        for spring in self.untracked_hinge_springs:
            untracked_by_reason[spring["reason"]] = untracked_by_reason.get(spring["reason"], 0) + 1
        # Rows on the legacy global fallback (one value for both signs) are counted apart from the installed points.
        tracked_by_source = {}
        for row in self.hinge_rows:
            tracked_by_source[row["threshold_source"]] = tracked_by_source.get(row["threshold_source"], 0) + 1
        first_column_yield = (
            self.column_yield_events[0] if self.column_yield_events else None
        )

        if first_cap and first_column_yield:
            beam_sway_before_column = (
                first_cap["roof_drift_ratio"] < first_column_yield["roof_drift_ratio"]
            )
        elif first_cap and not first_column_yield:
            beam_sway_before_column = True
        else:
            beam_sway_before_column = None

        return {
            "version": MECHANISM_DIAGNOSTICS_VERSION,
            "member_spring_calibration": spring_mapping.active_calibration(),
            "installed_member_spring_calibrations": sorted(
                {str(row["member_spring_calibration"]) for row in self.hinge_rows}
            ),
            "element_formulation": sp.ELEMENT_FORMULATION,
            "num_tracked_hinge_springs": len(self.hinge_rows),
            "tracked_hinge_springs_by_threshold_source": tracked_by_source,
            "untracked_hinge_springs_by_reason": untracked_by_reason,
            "story_drift_targets": self.story_drift_snapshots,
            "requested_story_drift_targets": requested_story_drift_targets,
            "unreached_story_drift_targets": unreached_story_drift_targets,
            "first_hinge_yield": first_yield,
            "first_hinge_yield_group": first_yield_group,
            "first_hinge_cap": first_cap,
            "first_hinge_cap_group": first_cap_group,
            "first_hinge_native_cutoff_du_exceedance": first_cutoff,
            "first_hinge_native_cutoff_du_exceedance_group": first_cutoff_group,
            "num_hinge_yield_events": len(yielded_events),
            "num_hinge_cap_events": len(capped_events),
            "num_hinge_native_cutoff_du_events": len(cutoff_events),
            "first_column_nominal_yield": first_column_yield,
            "num_column_nominal_yield_events": len(self.column_yield_events),
            "max_interstory_drift_overall": self.max_interstory_drift,
            "interstory_drift_envelope_by_story": self.interstory_drift_envelope_by_story,
            "max_column_demand_ratio": self.max_column_ratio,
            "max_hinge_rotation": self.max_hinge_rotation,
            "beam_sway_before_column_mechanism_indicator": beam_sway_before_column,
            "notes": [
                "Hinge rotation is the native zeroLength spring rotation (hinge node minus joint, rad): not a story drift and not a member chord rotation (chord = native rotation + M/Kb).",
                "Each spring is measured against its own installed points for the native sign of the current rotation (Model/IMK_Spring_Mapping.spring_points; threshold_source installed_spring_points), never a global or member-level value for both signs, except a member missing from the hinge registry under the legacy replay, whose rows carry threshold_source legacy_global_imk_hinge_thresholds and one global IMK_THETA_* value for both signs (counted in tracked_hinge_springs_by_threshold_source).",
                "Hinge yield is flagged at abs(rotation) >= native My/Kh of that sign.",
                "Hinge cap is flagged at abs(rotation) >= the native capping rotation My/Kh + dp of that sign.",
                "native_cutoff_du_exceeded is flagged at abs(rotation) >= the retained native du of that sign (member theta_p + theta_pc + legacy padding for installed points; the global IMK_THETA_U for legacy_global_imk_hinge_thresholds rows): a provisional numerical cutoff, not an observed collapse rotation or a material failure.",
                "Released (elastic) gravity-beam ends carry no IMK spring and are counted in untracked_hinge_springs_by_reason, not measured; outside the IMK formulation (element_formulation) no member has a spring and none is tracked.",
                "Column yield is flagged when the live column end P-M demand/capacity ratio reaches 1.0; it is not a column IMK hinge state.",
            ],
        }

    @staticmethod
    def _first_event_group(events):
        if not events:
            return []

        first_step = events[0]["step"]
        return [event for event in events if event["step"] == first_step]

    def results(self):
        return {
            "version": MECHANISM_DIAGNOSTICS_VERSION,
            "summary": self.summary(),
            "hinge_events": self.hinge_events,
            "column_yield_events": self.column_yield_events,
            "untracked_hinge_springs": self.untracked_hinge_springs,
        }
