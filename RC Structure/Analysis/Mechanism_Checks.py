import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Data_Generation.Graph_Exporter import collect_element_rows
from Design.ACI_Checks import build_pm_diagram, check_column_pm
from Design.Config import DesignConfig
from Model.Build_Model import floor_master_node
from Model.IMK_Hinges import hinge_backbone, hinge_element_tag, imk_hinge_thresholds


ROOF_DRIFT_TARGETS = (0.045, 0.099)


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


def _imk_hinge_member_rows(element_rows):
    rows = []

    for element in element_rows:
        element_type = element["element_type"]

        if element_type == "column" and not sp.IMK_APPLY_TO_COLUMNS:
            continue

        if element_type in {"beam_x", "beam_y"} and not sp.IMK_APPLY_TO_BEAMS:
            continue

        for end_id, end_name, node_key in ((1, "i", "node_i"), (2, "j", "node_j")):
            hinge_tag = hinge_element_tag(element["ele_tag"], end_id)
            # Prefer the backbone actually installed on this member. With
            # per-member calibration the rotation capacities vary with axial
            # load, so imk_hinge_thresholds' global IMK_THETA_* constants no
            # longer describe the hinge being measured and would report
            # capping far too early. Fall back to them only for a member the
            # registry never recorded (an uncalibrated or older build).
            installed = hinge_backbone(element["ele_tag"])
            for response_index, rot_dir in ((0, "rot_y"), (1, "rot_z")):
                thresholds = imk_hinge_thresholds(
                    element_type,
                    rot_dir,
                    element["length_in"],
                )
                if installed:
                    thresholds = dict(thresholds)
                    # The spring's own elastic limit and stiffness as installed
                    # (the member's own family/axial properties), when recorded.
                    axis = "y" if rot_dir == "rot_y" else "z"
                    if installed.get(f"ke_{axis}_kip_in_per_rad"):
                        thresholds["selected_stiffness"] = installed[f"ke_{axis}_kip_in_per_rad"]
                        thresholds["theta_y"] = installed[f"theta_y_spring_{axis}"]
                    theta_y = thresholds["theta_y"]
                    thresholds["theta_cap"] = theta_y + installed["theta_p"]
                    thresholds["theta_u"] = installed["theta_u"]
                rows.append(
                    {
                        "physical_ele_tag": element["ele_tag"],
                        "hinge_ele_tag": hinge_tag,
                        "node_tag": element[node_key],
                        "end": end_name,
                        "rot_dir": rot_dir,
                        "response_index": response_index,
                        "element_type": element_type,
                        "theta_y": thresholds["theta_y"],
                        "theta_cap": thresholds["theta_cap"],
                        "theta_u": thresholds["theta_u"],
                    }
                )

    return rows


def _column_rows(element_rows):
    return [row for row in element_rows if row["element_type"] == "column"]


class MechanismTracker:
    def __init__(self):
        self.element_rows = collect_element_rows()
        self.hinge_rows = _imk_hinge_member_rows(self.element_rows)
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

            theta = deformation[hinge["response_index"]]
            abs_theta = abs(theta)
            state_key = (
                hinge["hinge_ele_tag"],
                hinge["rot_dir"],
            )
            state = self.hinge_state.setdefault(
                state_key,
                {
                    "yielded": False,
                    "capped": False,
                    "ultimate_exceeded": False,
                },
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
                }

            for threshold_name, state_name in (
                ("theta_y", "yielded"),
                ("theta_cap", "capped"),
                ("theta_u", "ultimate_exceeded"),
            ):
                if state[state_name]:
                    continue

                if abs_theta >= hinge[threshold_name]:
                    state[state_name] = True
                    self.hinge_events.append(
                        {
                            **hinge,
                            "event": state_name,
                            "threshold": hinge[threshold_name],
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
            "new_hinge_ultimate_events": [
                event for event in new_hinge_events if event["event"] == "ultimate_exceeded"
            ],
            "new_column_yield_events": new_column_yield_events,
        }

    def summary(self):
        yielded_events = [event for event in self.hinge_events if event["event"] == "yielded"]
        capped_events = [event for event in self.hinge_events if event["event"] == "capped"]
        ultimate_events = [
            event for event in self.hinge_events if event["event"] == "ultimate_exceeded"
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
        first_ultimate = ultimate_events[0] if ultimate_events else None
        first_yield_group = self._first_event_group(yielded_events)
        first_cap_group = self._first_event_group(capped_events)
        first_ultimate_group = self._first_event_group(ultimate_events)
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
            "story_drift_targets": self.story_drift_snapshots,
            "requested_story_drift_targets": requested_story_drift_targets,
            "unreached_story_drift_targets": unreached_story_drift_targets,
            "first_hinge_yield": first_yield,
            "first_hinge_yield_group": first_yield_group,
            "first_hinge_cap": first_cap,
            "first_hinge_cap_group": first_cap_group,
            "first_hinge_ultimate_exceedance": first_ultimate,
            "first_hinge_ultimate_exceedance_group": first_ultimate_group,
            "num_hinge_yield_events": len(yielded_events),
            "num_hinge_cap_events": len(capped_events),
            "num_hinge_ultimate_exceedance_events": len(ultimate_events),
            "first_column_nominal_yield": first_column_yield,
            "num_column_nominal_yield_events": len(self.column_yield_events),
            "max_interstory_drift_overall": self.max_interstory_drift,
            "interstory_drift_envelope_by_story": self.interstory_drift_envelope_by_story,
            "max_column_demand_ratio": self.max_column_ratio,
            "max_hinge_rotation": self.max_hinge_rotation,
            "beam_sway_before_column_mechanism_indicator": beam_sway_before_column,
            "notes": [
                "Hinge cap is flagged at abs(rotation) >= theta_y + theta_p.",
                "Hinge ultimate is flagged at abs(rotation) >= theta_u.",
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
            "summary": self.summary(),
            "hinge_events": self.hinge_events,
            "column_yield_events": self.column_yield_events,
        }
