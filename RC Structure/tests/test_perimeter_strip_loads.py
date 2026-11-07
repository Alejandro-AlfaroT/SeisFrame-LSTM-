"""The perimeter strip of the physical floor on the slab-transfer route (load audit of 2026-11-05).

A floor transfer is solved for the centreline plate; weights and masses use the physical footprint. These
tests work the difference by hand on the grouped fixture (2 x 3 bays of 240 x 216 in, 18 in beams, 24 in
edge Y beams on floors 1-2, 7 in slab, 0.1375 ksf dead and 0.04 ksf live) and check that

  * every perimeter beam carries the strip of its own side (width and line load by hand), interior beams none;
  * a floor's strip is its footprint load minus the plate load, in force and in both plan first moments;
  * a live pattern loads the strip pieces beside its own panels only;
  * the validator accepts a transfer of the centreline plate, refuses the footprint load as the plate's and
    still refuses an edited transfer; plate + strip must be the area the weights use;
  * the frame loads on the transfer route are the footprint loads, grouped and uniform, and the strip can be
    left out for a comparison against the plate alone;
  * the beam gravity shears taken straight from a transfer carry the same strip (grouped and uniform routes),
    their recorded basis names the strip method, and a capacity build makes one footprint per floor;
  * the wiring (2026-11-06): the uniform beam shear design hands the strip to its edge families, the grouped
    evaluation validates its transfers against the plate load, and the coupled comparison loads its frame
    with the centreline plate alone; the coupled model takes its column weight on the same physical floor as
    the frame, so the two hold one member weight;
  * nothing is read from the strip without a floor transfer or without a slab.

No structural solve: the transfer is a statically exact stand-in (every loaded panel sends a quarter of its
load to the midspan of each of its four beams) and the frame loads go to recording sinks.
"""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import Structure_Parameters as sp                                               # noqa: E402
from Design import Group_Capacity as capacity                                    # noqa: E402
from Design import SMRF_Capacity_Design as cd                                    # noqa: E402
from Design import SMRF_Floor_Sections as floor_sections                         # noqa: E402
from Design import SMRF_Floor_Transfer as ft                                     # noqa: E402
from Loads import Gravity_Loads as loads                                         # noqa: E402
from Loads import Perimeter_Strip as strip                                       # noqa: E402
from Model import Floor_Footprint as fp                                          # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402
from Model.Build_Model import node_tag                                           # noqa: E402

NX, NY, NZ, LX, LY = gf.NX, gf.NY, gf.NZ, gf.BAY_X, gf.BAY_Y
SDL, LIVE, UNIT_WEIGHT = 0.05, 0.04, 0.15
DEAD = SDL + UNIT_WEIGHT * gf.SLAB / 12.0                       # 0.1375 ksf
LOADS = {"FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF": SDL, "FLOOR_LIVE_LOAD_KSF": LIVE, "CONCRETE_UNIT_WEIGHT_KCF": UNIT_WEIGHT,
         "CONCRETE_UNIT_WEIGHT_KCI": UNIT_WEIGHT / 1728.0, "ROOF_COLUMN_EXTENSION": False}
GEOMETRY = {"num_bay_x": NX, "num_bay_y": NY, "num_floor": NZ, "bay_x_in": LX, "bay_y_in": LY}
PLATE_IN2 = NX * LX * NY * LY                                   # 311 040 in2
# Outstand of the footprint beyond the grid lines, by hand: half the perimeter beam width of each side.
# Changed state: 24 in edge Y beams on floors 1-2 (X sides 12 in), 18 in beams elsewhere (9 in).
OUTSTAND_X = {1: 12.0, 2: 12.0, 3: 9.0, 4: 9.0}
OUTSTAND_Y = 9.0
STRIP_IN2 = {1: 24624.0, 2: 24624.0, 3: 20628.0, 4: 20628.0}    # 504 x 666 and 498 x 666, less the plate
PATTERNS = [{"id": "two_panels", "rule": "test", "panels": [[0, 0], [1, 2]]},
            {"id": "all_panels", "rule": "test", "panels": [[i, j] for i in range(NX) for j in range(NY)]}]
BEAM_KEYS = [("x", j, i) for j in range(NY + 1) for i in range(NX)] + [("y", i, j) for i in range(NX + 1) for j in range(NY)]


def synthetic_transfer(sections, patterns=PATTERNS):
    """A statically exact transfer of the centreline plate, in the saved v3 form: every loaded panel sends a
    quarter of its load to the midspan of each of its four beams; the columns take nothing directly."""
    panel_sqft = LX * LY / 144.0

    def case(pressure, panels, descriptor):
        totals = {}
        for i, j in panels:
            for key in (("x", j, i), ("x", j + 1, i), ("y", i, j), ("y", i + 1, j)):
                totals[key] = totals.get(key, 0.0) + pressure * panel_sqft / 4.0
        beams = [{"axis": axis, "line_index": line, "span_index": span,
                  "node_loads": [[0.5, totals.get((axis, line, span), 0.0)]], "node_couples": [[0.5, 0.0, 0.0]],
                  "total_kip": totals.get((axis, line, span), 0.0),
                  "equivalent_uniform_kip_per_in": totals.get((axis, line, span), 0.0) / (LX if axis == "x" else LY)}
                 for axis, line, span in BEAM_KEYS]
        columns = [{"grid_i": i, "grid_j": j, "direct_load_kip": 0.0, "couple_global_mx_kip_in": 0.0,
                    "couple_global_my_kip_in": 0.0} for j in range(NY + 1) for i in range(NX + 1)]
        applied = pressure * panel_sqft * len(panels)
        return {"loadcase": descriptor, "applied_kip": applied, "beam_kip": sum(b["total_kip"] for b in beams),
                "column_direct_kip": 0.0, "equilibrium": {"numerical_balance_passed": True},
                "beams": beams, "columns": columns}

    everything = [[i, j] for i in range(NX) for j in range(NY)]
    cases = {"dead": case(DEAD, everything, {"dead_factor": 1.0, "live_factor": 0.0, "live_load_ksf": LIVE, "live_pattern": "none"}),
             "live": case(LIVE, everything, {"dead_factor": 0.0, "live_factor": 1.0, "live_load_ksf": LIVE, "live_pattern": "all"})}
    for pattern in patterns:
        cases[f"live_pattern_{pattern['id']}"] = case(LIVE, pattern["panels"], {
            "dead_factor": 0.0, "live_factor": 1.0, "live_load_ksf": LIVE, "live_pattern": [list(p) for p in pattern["panels"]]})
    by_line = floor_sections.is_by_line(sections)
    return {"schema": ft.TRANSFER_SCHEMA, "method_version": ft.METHOD_VERSION_FLEXIBLE, "load_model": ft.LOAD_MODEL_NAME,
            "mesh_per_bay": 2, "geometry": {key: GEOMETRY[key] for key in ("num_bay_x", "num_bay_y", "bay_x_in", "bay_y_in")},
            "sections": copy.deepcopy(sections),
            **({"floor_sections_sha256": floor_sections.signature(sections)} if by_line else {}),
            "slab_thickness_in": gf.SLAB, "dead_pressure_ksf": DEAD, "live_pressure_ksf": LIVE,
            "live_patterns": copy.deepcopy(list(patterns)), "unit_cases": cases}


def synthetic_container(description):
    """The by-floor container of the installed grouped design: one stand-in transfer per distinct floor."""
    return {"schema": ft.BY_FLOOR_SCHEMA, "method_version": ft.METHOD_VERSION_FLEXIBLE, "load_model": ft.LOAD_MODEL_NAME,
            "mesh_per_bay": 2, "floors": dict(description["floors"]),
            "transfers": {sha: synthetic_transfer(sections) for sha, sections, _floors in floor_sections.distinct_floors(description)}}


def uniform_sections():
    return {"b_beam_in": sp.B_BEAM, "h_beam_in": sp.H_BEAM, "fc_beam_ksi": sp.FC_BEAM_KSI, "b_col_in": sp.B_COL, "h_col_in": sp.H_COL}


def applied(**arguments):
    """apply_gravity_loads into recording sinks: (downward force, first moment about X = sum P x, about Y = sum P y,
    the element and nodal call lists). Couples enter the first moments as the offsets they stand for."""
    with mock.patch.object(loads.ops, "timeSeries"), mock.patch.object(loads.ops, "pattern"), \
         mock.patch.object(loads.ops, "load") as nodal, mock.patch.object(loads.ops, "eleLoad") as element:
        loads.apply_gravity_loads(**arguments)
    positions = {node_tag(k, i, j): (i * LX, j * LY) for k in range(1, NZ + 1) for j in range(NY + 1) for i in range(NX + 1)}
    force = first_x = first_y = 0.0
    for call in nodal.call_args_list:
        tag, _fx, _fy, fz, mx, my, _mz = call.args
        x, y = positions[tag]
        force -= fz
        first_x += my - x * fz              # a downward P at dx: My = P dx
        first_y += -mx - y * fz             # a downward P at dy: Mx = -P dy
    for call in element.call_args_list:
        args = call.args
        member = mg.resolve(args[1])
        length = mp.member_length_in(member)
        x, y = member.grid_i * LX, member.grid_j * LY
        if args[3] == "-beamPoint":
            load, along = -args[5], args[6] * length
        else:
            load, along = -(args[5] + args[6]) * length, 0.5 * length
        if member.member_type == "beam_x":
            x += along
        if member.member_type == "beam_y":
            y += along
        force += load
        first_x += load * x
        first_y += load * y
    return force, first_x, first_y, element.call_args_list, nodal.call_args_list


class StripGeometryAndLoads(unittest.TestCase):
    def setUp(self):
        self.stack = gf.frame(**LOADS)
        self.stack.__enter__()
        mg.install(gf.changed_state())
        self.description = floor_sections.by_floor()
        self.container = synthetic_container(self.description)
        self.transfers = {k: self.container["transfers"][self.container["floors"][str(k)]] for k in range(1, NZ + 1)}

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_strip_width_and_line_load_of_every_perimeter_beam_by_hand(self):
        self.assertAlmostEqual(sp.floor_dead_load_ksf(), 0.1375, places=12)
        for k in range(1, NZ + 1):
            pieces = strip.floor_pieces(k)
            self.assertEqual(len(pieces["beams"]), 2 * NX + 2 * NY)
            for (axis, line, span), piece in pieces["beams"].items():
                width = OUTSTAND_Y if axis == "x" else OUTSTAND_X[k]
                self.assertEqual(piece["width_in"], width, (k, axis, line))
                self.assertEqual(piece["length_in"], LX if axis == "x" else LY)
                self.assertEqual(piece["offset_in"], (-1.0 if line == 0 else 1.0) * width / 2.0)
                key = (axis, line, span)
                dead = strip.floor_line_load_kip_per_in(k, self.transfers[k], key, "dead")
                live = strip.floor_line_load_kip_per_in(k, self.transfers[k], key, "live")
                self.assertAlmostEqual(dead, 0.1375 / 144.0 * width, places=15)
                self.assertAlmostEqual(live, 0.04 / 144.0 * width, places=15)
            for joint, piece in pieces["corners"].items():
                self.assertEqual(piece["area_in2"], OUTSTAND_X[k] * OUTSTAND_Y)
            # Interior beams carry no strip.
            for key in (("x", 1, 0), ("x", 2, 1), ("y", 1, 0), ("y", 1, 2)):
                self.assertEqual(strip.floor_line_load_kip_per_in(k, self.transfers[k], key, "dead"), 0.0)
        # Two numbers written out: floor 1, the X beam on line 0 and the 24 in Y beam on line 0.
        self.assertAlmostEqual(strip.floor_line_load_kip_per_in(1, self.transfers[1], ("x", 0, 0), "dead"), 0.00859375, places=12)
        self.assertAlmostEqual(strip.floor_line_load_kip_per_in(1, self.transfers[1], ("y", 0, 0), "dead"), 0.0114583333333, places=12)
        self.assertAlmostEqual(strip.floor_line_load_kip_per_in(3, self.transfers[3], ("y", 0, 0), "dead"), 0.00859375, places=12)

    def test_floor_total_is_footprint_minus_centreline_plate(self):
        for k in range(1, NZ + 1):
            pieces = strip.floor_pieces(k)
            self.assertAlmostEqual(pieces["area_in2"], STRIP_IN2[k], places=9)
            self.assertAlmostEqual(pieces["area_in2"], mp.floor_area_in2(k) - PLATE_IN2, places=6)
            for dead_factor, live_factor in ((1.0, 0.0), (0.0, 1.0), (1.2, 1.6)):
                result = strip.floor_loads(k, self.transfers[k], dead_factor, live_factor)
                pressure = (dead_factor * 0.1375 + live_factor * 0.04) / 144.0
                self.assertAlmostEqual(result["total_kip"], pressure * STRIP_IN2[k], places=10)
                listed = (sum(w * (LX if key[0] == "x" else LY) for key, w in result["beams"].items())
                          + sum(row[0] for row in result["joints"].values()))
                self.assertAlmostEqual(listed, result["total_kip"], places=10)
        # 1.2D + 1.6L on floor 1, by hand: (0.165 + 0.064) / 144 x 24 624 in2 = 39.159 kip.
        self.assertAlmostEqual(strip.floor_loads(1, self.transfers[1], 1.2, 1.6)["total_kip"], 39.159, places=3)

    def test_strip_first_moments_are_the_footprint_minus_the_plate_with_unequal_faces(self):
        # Unequal outside faces on all four sides: nothing cancels by symmetry.
        footprint = fp.rectangle(-7.0, 492.0, -8.0, 662.0)
        with mock.patch.object(fp, "physical_floor", return_value=footprint):
            pieces = strip.floor_pieces(1)
            self.assertEqual(pieces["outstand_in"], {"x_minus": 7.0, "x_plus": 12.0, "y_minus": 8.0, "y_plus": 14.0})
            self.assertEqual(pieces["beams"][("x", NY, 0)]["offset_in"], 7.0)
            self.assertEqual(pieces["beams"][("y", 0, 1)]["offset_in"], -3.5)
            self.assertEqual(pieces["corners"][(NX, 0)], {"area_in2": 96.0, "offset_in": (6.0, -4.0), "panel": (NX - 1, 0)})
            result = strip.floor_loads(1, self.transfers[1], 1.0, 0.0)
        q = 0.1375 / 144.0
        force = first_x = first_y = 0.0
        for (axis, line, span), w in result["beams"].items():
            load = w * (LX if axis == "x" else LY)
            force += load
            first_x += load * ((span + 0.5) * LX if axis == "x" else line * LX)
            first_y += load * (line * LY if axis == "x" else (span + 0.5) * LY)
        for (i, j), (load, mx, my) in result["joints"].items():
            force += load
            first_x += load * i * LX + my
            first_y += load * j * LY - mx
        plate = fp.rectangle(0.0, NX * LX, 0.0, NY * LY)
        self.assertAlmostEqual(force, q * (footprint["area_in2"] - plate["area_in2"]), places=10)
        self.assertAlmostEqual(first_x, q * (footprint["first_moment_x_in3"] - plate["first_moment_x_in3"]), places=7)
        self.assertAlmostEqual(first_y, q * (footprint["first_moment_y_in3"] - plate["first_moment_y_in3"]), places=7)
        # One couple by hand: the X beam on line 0, span 0 carries q x 8 x 240 at 4 in outside its axis (dy = -4):
        # Mx = +P x 4, half at each end joint; joint (1, 0) also takes half of the next span's.
        p = q * 8.0 * 240.0
        self.assertAlmostEqual(result["joints"][(1, 0)][1], p * 4.0, places=12)
        self.assertEqual(result["joints"][(1, 0)][0], 0.0)

    def test_a_live_pattern_loads_the_strip_beside_its_own_panels_only(self):
        transfer, w = self.transfers[1], 0.04 / 144.0
        name = "live_pattern_two_panels"                                    # panels (0, 0) and (1, 2)
        expected = {("x", 0, 0): 9.0, ("x", 0, 1): 0.0, ("x", NY, 0): 0.0, ("x", NY, 1): 9.0,
                    ("y", 0, 0): 12.0, ("y", 0, 1): 0.0, ("y", 0, 2): 0.0,
                    ("y", NX, 0): 0.0, ("y", NX, 1): 0.0, ("y", NX, 2): 12.0}
        for key, width in expected.items():
            self.assertAlmostEqual(strip.floor_line_load_kip_per_in(1, transfer, key, name), w * width, places=15, msg=key)
            # Dead and the all-panel live case are not patterned.
            self.assertGreater(strip.floor_line_load_kip_per_in(1, transfer, key, "dead"), 0.0)
            self.assertGreater(strip.floor_line_load_kip_per_in(1, transfer, key, "live"), 0.0)
        result = strip.floor_loads(1, transfer, 0.0, 1.0, "two_panels")
        self.assertEqual(set(result["beams"]), {key for key, width in expected.items() if width})
        corners = {joint: row[0] for joint, row in result["joints"].items() if row[0]}
        self.assertEqual(set(corners), {(0, 0), (NX, NY)})                 # the corner squares of the loaded corner panels
        self.assertAlmostEqual(corners[(0, 0)], w * 12.0 * 9.0, places=15)
        self.assertAlmostEqual(result["total_kip"], w * (2 * 9.0 * LX + 2 * 12.0 * LY + 2 * 108.0), places=12)
        full = strip.floor_loads(1, transfer, 0.0, 1.0, "all_panels")
        self.assertAlmostEqual(full["total_kip"], strip.floor_loads(1, transfer, 0.0, 1.0)["total_kip"], places=12)
        with self.assertRaisesRegex(ValueError, "no live pattern"):
            strip.floor_loads(1, transfer, 0.0, 1.0, "not_saved")

    def test_validator_takes_the_plate_load_and_refuses_the_footprint_load_and_an_edited_transfer(self):
        dead_kip, live_kip = strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF)
        self.assertAlmostEqual(dead_kip, 0.1375 * 2160.0, places=10)        # 311 040 in2 = 2160 ft2 -> 297 kip
        self.assertAlmostEqual(live_kip, 0.04 * 2160.0, places=10)
        transfers = ft.validate_floor_transfers_by_floor(self.container, GEOMETRY, self.description, gf.SLAB, dead_kip, live_kip)
        self.assertEqual(sorted(transfers), [1, 2, 3, 4])
        # Plate + strip is the load of the area the weights use, floor by floor.
        for k in range(1, NZ + 1):
            both = dead_kip + strip.floor_loads(k, transfers[k], 1.0, 0.0)["total_kip"]
            self.assertAlmostEqual(both, sp.floor_dead_load_ksf() * mp.floor_area_in2(k) / 144.0, places=9)
        # The footprint load is not the load of the plate the transfer was solved for (the 2026-11-05 refusal).
        footprint_dead = {k: sp.floor_dead_load_ksf() * mp.floor_area_in2(k) / 144.0 for k in range(1, NZ + 1)}
        footprint_live = {k: sp.FLOOR_LIVE_LOAD_KSF * mp.floor_area_in2(k) / 144.0 for k in range(1, NZ + 1)}
        with self.assertRaisesRegex(ValueError, "dead pressure carries 0.1375"):
            ft.validate_floor_transfers_by_floor(self.container, GEOMETRY, self.description, gf.SLAB, footprint_dead, footprint_live)
        with self.assertRaisesRegex(ValueError, "dead pressure"):
            ft.validate_floor_transfers_by_floor(self.container, GEOMETRY, self.description, gf.SLAB, 1.01 * dead_kip, live_kip)
        edited = copy.deepcopy(self.container)
        sha = edited["floors"]["1"]
        edited["transfers"][sha]["unit_cases"]["dead"]["beams"][0]["node_loads"][0][1] += 0.5
        with self.assertRaises(ValueError):
            ft.validate_floor_transfers_by_floor(edited, GEOMETRY, self.description, gf.SLAB, dead_kip, live_kip)
        # Plate + strip must be the area the weights use, or no plate load is given out.
        with mock.patch.object(mp, "floor_area_in2", return_value=PLATE_IN2):
            with self.assertRaisesRegex(ValueError, "the weights use"):
                strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF)

    def test_grouped_frame_loads_on_the_transfer_route_are_the_footprint_loads(self):
        sp.FLOOR_TRANSFER = self.container
        self.assertEqual(sp.effective_gravity_load_model(), "slab_transfer")
        footprints = {k: fp.physical_floor(k) for k in range(1, NZ + 1)}
        q = (1.2 * 0.1375 + 1.6 * 0.04) / 144.0
        factors = dict(floor_factor=1.0, self_weight_factor=0.0, dead_factor=1.2, live_factor=1.6, roof_extension=False)
        force, first_x, first_y, element, _nodal = applied(**factors)
        self.assertAlmostEqual(force, q * sum(PLATE_IN2 + STRIP_IN2[k] for k in footprints), places=8)
        self.assertAlmostEqual(force, q * sum(f["area_in2"] for f in footprints.values()), places=8)
        self.assertAlmostEqual(first_x, q * sum(f["first_moment_x_in3"] for f in footprints.values()), places=5)
        self.assertAlmostEqual(first_y, q * sum(f["first_moment_y_in3"] for f in footprints.values()), places=5)
        # The strip is one uniform line load on each perimeter beam of each floor: floor 1, X beam on line 0.
        tag = loads._beam_element_tag(1, "x", 0, 0)
        uniform = [c.args for c in element if c.args[1] == tag and c.args[3] == "-beamUniform" and c.args[5] != 0.0]
        self.assertEqual(len(uniform), 1)
        self.assertAlmostEqual(-uniform[0][5], q * 9.0, places=14)
        interior = loads._beam_element_tag(1, "x", 1, 0)
        self.assertFalse([c for c in element if c.args[1] == interior and c.args[3] == "-beamUniform" and c.args[5] != 0.0])
        # Left out for a comparison against the plate alone: the transfer's own loads, nothing else.
        plate_force, plate_x, plate_y, _, _ = applied(perimeter_strip=False, **factors)
        self.assertAlmostEqual(plate_force, q * NZ * PLATE_IN2, places=8)
        self.assertAlmostEqual(plate_x, q * NZ * PLATE_IN2 * NX * LX / 2.0, places=5)
        self.assertAlmostEqual(plate_y, q * NZ * PLATE_IN2 * NY * LY / 2.0, places=5)
        # A pattern: dead on the footprint, live on the pattern's panels and the strip beside them.
        force, _, _, _, _ = applied(live_pattern="two_panels", **factors)
        live_in2 = 2 * LX * LY + sum(2 * OUTSTAND_Y * LX + 2 * OUTSTAND_X[k] * LY + 2 * OUTSTAND_X[k] * OUTSTAND_Y for k in footprints) / NZ
        self.assertAlmostEqual(force, 1.2 * 0.1375 / 144.0 * sum(f["area_in2"] for f in footprints.values())
                               + 1.6 * 0.04 / 144.0 * NZ * live_in2, places=8)

    def test_grouped_beam_gravity_shear_carries_the_strip_on_perimeter_beams(self):
        validated = ft.validate_floor_transfers_by_floor(
            self.container, GEOMETRY, self.description, gf.SLAB, *strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF))
        panel_dead, panel_live = 0.1375 * LX * LY / 144.0, 0.04 * LX * LY / 144.0    # 49.5 and 14.4 kip
        for kind, i, j, key, width in (("beam_x", 0, 0, ("x", 0, 0), 9.0), ("beam_y", 0, 1, ("y", 0, 1), 12.0),
                                       ("beam_x", 1, 2, ("x", 2, 1), 0.0), ("beam_y", 1, 0, ("y", 1, 0), 0.0)):
            member = mg.beam_at(kind, 1, i, j)
            face_i, face_j = mp.beam_face_distances_in(member)
            clear = mp.beam_span_in(member) - face_i - face_j
            sides = 1 if width else 2                                          # panels beside the beam
            # Midspan transfer loads on the clear span, the beam's drop weight and the strip, at 1.2D + 1.0L.
            hand = (0.5 * sides * (1.2 * panel_dead + 1.0 * panel_live) / 4.0
                    + 1.2 * mp.beam_drop_weight_kip_per_in(member) * clear / 2.0
                    + (1.2 * 0.1375 + 1.0 * 0.04) / 144.0 * width * clear / 2.0)
            reactions, basis = capacity.member_face_reactions(validated[1], member, 1.2, 1.0)
            self.assertAlmostEqual(reactions[0], hand, places=10, msg=key)
            self.assertAlmostEqual(reactions[1], hand, places=10, msg=key)
            self.assertEqual("perimeter strip" in basis, bool(width), key)
            self.assertEqual(strip.METHOD in basis, bool(width), key)          # the record names the strip method
            # Dead alone (the low-gravity family has no live load).
            reactions, _ = capacity.member_face_reactions(validated[1], member, 0.7, 0.0)
            self.assertAlmostEqual(reactions[0], 0.7 * (0.5 * sides * panel_dead / 4.0 + mp.beam_drop_weight_kip_per_in(member) * clear / 2.0
                                                        + 0.1375 / 144.0 * width * clear / 2.0), places=10, msg=key)
        # The shear and the frame load read one helper: the same line load on the same beam.
        sp.FLOOR_TRANSFER = self.container
        _, _, _, element, _ = applied(floor_factor=1.0, self_weight_factor=0.0, dead_factor=1.0, live_factor=0.0, roof_extension=False)
        tag = loads._beam_element_tag(1, "y", 0, 1)
        frame_w = [-c.args[5] for c in element if c.args[1] == tag and c.args[3] == "-beamUniform" and c.args[5] != 0.0]
        self.assertEqual(frame_w, [strip.floor_line_load_kip_per_in(1, validated[1], ("y", 0, 1), "dead")])

    def test_one_capacity_build_makes_one_footprint_per_floor(self):
        validated = ft.validate_floor_transfers_by_floor(
            self.container, GEOMETRY, self.description, gf.SLAB, *strip.plate_loads_kip(sp.floor_dead_load_ksf(), sp.FLOOR_LIVE_LOAD_KSF))
        state = mg.active()
        groups = [gid for gid, group in sorted(state.groups.items()) if group["member_type"] != "column"]
        self.assertEqual(len(groups), 8)                                              # two bands x two axes x edge, interior
        store = {}
        with mock.patch.object(strip, "floor_pieces", wraps=strip.floor_pieces) as built:
            rows = [row for gid in groups for row in capacity.beam_group_capacity(gid, state, validated, 1.0, store)["members"]]
        self.assertEqual(built.call_count, NZ)                                        # 4 floors, not 40 perimeter beams x 2 families
        self.assertEqual(sorted(store), [1, 2, 3, 4])
        self.assertEqual(len(rows), 2 * NZ * (NX * (NY + 1) + NY * (NX + 1)))         # 68 beams, two gravity families
        for row in rows:
            member = state.member(row["member_tag"])
            perimeter = member.grid_j in (0, NY) if member.member_type == "beam_x" else member.grid_i in (0, NX)
            # The shared store changes no number: each row is the member's own free body, and its basis names the strip.
            reactions, basis = capacity.member_face_reactions(validated[member.story_or_floor], member, row["dead_factor"], row["live_factor"])
            self.assertEqual(row["gravity_reactions_kip"], reactions)
            self.assertEqual(row["gravity_reaction_basis"], basis)
            self.assertEqual(strip.SHEAR_BASIS_CLAUSE in basis, perimeter, row["id"])
        # A group on its own builds its own floors once: two for an edge group, none for an interior one.
        with mock.patch.object(strip, "floor_pieces", wraps=strip.floor_pieces) as built:
            capacity.beam_group_capacity("s01_02__beam_y__edge", state, validated, 1.0)
            self.assertEqual(built.call_count, 2)
            capacity.beam_group_capacity("s01_02__beam_y__interior", state, validated, 1.0)
            self.assertEqual(built.call_count, 2)
        # Without a store one call builds its own floor once, whatever the number of unit cases; an interior beam none.
        with mock.patch.object(strip, "floor_pieces", wraps=strip.floor_pieces) as built:
            capacity.member_face_reactions(validated[1], mg.beam_at("beam_x", 1, 0, 0), 1.2, 1.0)
            self.assertEqual(built.call_count, 1)
            capacity.member_face_reactions(validated[1], mg.beam_at("beam_x", 1, 0, 1), 1.2, 1.0)
            self.assertEqual(built.call_count, 1)

    def test_grouped_evaluation_validates_its_transfers_against_the_plate_load(self):
        from Design import Grouped_Design as gd
        from Design.Config import DesignConfig

        class FrameSolveReached(Exception):
            pass

        def slab_stage(cfg, context):
            description = floor_sections.by_floor()
            return {"slab": {"thickness_in": gf.SLAB}, "attempts": [], "description": description, "reinforcement": None,
                    "evidence": None, "transfer": synthetic_container(description)}

        # The evaluation up to its first frame solve: the slab stage's transfers must pass the validator on the way.
        with mock.patch.object(gd, "slab_stage", slab_stage), mock.patch.object(gd, "_require_family_grades"), \
             mock.patch.object(gd.bar_layers, "row_audit", return_value={"valid": True}), \
             mock.patch.object(gd.bar_layers, "raise_arrangement_errors"), \
             mock.patch.object(ft, "validate_floor_transfers_by_floor", wraps=ft.validate_floor_transfers_by_floor) as validator, \
             mock.patch.object(gd.systems, "model_period", side_effect=FrameSolveReached):
            with self.assertRaises(FrameSolveReached):
                gd.evaluate_candidate(gf.changed_state(), DesignConfig(), {})
        self.assertEqual(validator.call_count, 1)
        dead_kip, live_kip = validator.call_args.args[4:6]
        self.assertAlmostEqual(dead_kip, 297.0, places=10)          # 0.1375 ksf x 2160 ft2: the plate, the same on every floor
        self.assertAlmostEqual(live_kip, 86.4, places=10)           # 0.04 ksf x 2160 ft2

    def test_coupled_comparison_loads_its_frame_with_the_centreline_plate_alone(self):
        from Design import SMRF_Coupled_Comparison as comparison
        sp.FLOOR_TRANSFER = self.container
        recorded = []
        with mock.patch.object(comparison, "build_design_model"), mock.patch.object(comparison, "run_gravity_analysis"), \
             mock.patch.object(comparison.ops, "wipe"), mock.patch.object(comparison.ops, "reactions"), \
             mock.patch.object(comparison.ops, "nodeReaction", return_value=0.0), \
             mock.patch.object(comparison, "apply_gravity_loads", side_effect=lambda **kw: recorded.append(applied(**kw))):
            comparison._frame_base_reactions()
        self.assertEqual(len(recorded), 1)
        # Member weight alone at the comparison's 1.2 (no area pressure, so no strip either way); the rest is the floor load.
        members = applied(floor_factor=1.0, self_weight_factor=1.2, dead_factor=0.0, live_factor=0.0, roof_extension=False)
        force, first_x, first_y = (recorded[0][n] - members[n] for n in range(3))
        # 1.2D + 1.6L = 0.229 ksf on four plates of 2160 ft2: 1978.56 kip, centred on the plate (240, 324) in.
        # With the strips (2 x 24 624 + 2 x 20 628 in2 more) it would be 2122.486 kip.
        self.assertAlmostEqual(force, 1978.56, places=7)
        self.assertAlmostEqual(first_x, 1978.56 * 240.0, places=4)
        self.assertAlmostEqual(first_y, 1978.56 * 324.0, places=4)

    def test_coupled_model_reads_the_physical_floor_of_the_frame_weights_for_its_columns(self):
        from Design import SMRF_Coupled_Analysis as coupled
        from Design.SMRF_Coupled_Comparison import coupled_sections_by_member
        physical = coupled._physical_floors(coupled_sections_by_member()["floors"], NZ, NX, NY, LX, LY, None)
        self.assertEqual(len(physical), NZ)
        for k in range(1, NZ + 1):
            for key in ("x_min_in", "x_max_in", "y_min_in", "y_max_in", "area_in2"):
                self.assertEqual(physical[k - 1][key], fp.physical_floor(k)[key], (k, key))
            self.assertEqual(physical[k - 1]["area_in2"], PLATE_IN2 + STRIP_IN2[k])
        # A 26 in perimeter column stands 13 in beyond its grid line, the floor 12 in (floors 1-2, X sides) or 9 in:
        # the column area outside the floor, by hand, is what keeps its concrete through the slab depth.
        for (k, i, j), hand in {(1, 0, 0): 676.0 - 25.0 * 22.0, (1, 0, 1): 676.0 - 25.0 * 26.0, (1, 1, 0): 676.0 - 26.0 * 22.0,
                                (3, 0, 0): 676.0 - 22.0 * 22.0, (3, NX, 2): 676.0 - 22.0 * 26.0, (4, 1, NY): 676.0 - 26.0 * 22.0,
                                (1, 1, 1): 0.0, (3, 1, 2): 0.0}.items():
            member = next(m for m in mg.all_members("column") if (m.story_or_floor, m.grid_i, m.grid_j) == (k, i, j))
            design = member.design
            outside = design.b_in * design.h_in - fp.column_overlap_area_in2(physical[k - 1], i * LX, j * LY, design.h_in, design.b_in)
            self.assertAlmostEqual(outside, hand, places=9, msg=(k, i, j))
            # The frame's column weight is on that clipped volume: gamma x (b h H - overlap x slab) per column.
            frame = mp.column_self_weight_kip_per_in(member) * sp.STORY_H
            self.assertAlmostEqual(frame, UNIT_WEIGHT / 1728.0 * (design.b_in * design.h_in * (sp.STORY_H - gf.SLAB) + outside * gf.SLAB),
                                   places=10, msg=(k, i, j))
        # One beam width for the whole building (the one-section form): 9 in beyond every side.
        uniform = coupled._physical_floors(None, 2, NX, NY, LX, LY, 18.0)
        self.assertEqual([floor["area_in2"] for floor in uniform], [498.0 * 666.0] * 2)

    def test_nothing_is_read_from_the_strip_without_a_floor_transfer(self):
        refused = AssertionError("the perimeter strip was read without a floor transfer")
        self.assertIsNone(sp.FLOOR_TRANSFER)
        for mode in ("nodal", "beam_uniform"):
            with self.subTest(mode=mode), mock.patch.object(sp, "GRAVITY_LOAD_MODEL", mode):
                baseline = applied(floor_factor=1.3, self_weight_factor=1.2, roof_extension=False)
                with mock.patch.object(strip, "floor_pieces", side_effect=refused), \
                     mock.patch.object(strip, "floor_loads", side_effect=refused), \
                     mock.patch.object(strip, "plate_loads_kip", side_effect=refused):
                    for flag in (True, False):
                        again = applied(floor_factor=1.3, self_weight_factor=1.2, roof_extension=False, perimeter_strip=flag)
                        self.assertEqual(again[3], baseline[3])
                        self.assertEqual(again[4], baseline[4])
        member = mg.beam_at("beam_x", 1, 0, 0)
        with mock.patch.object(strip, "floor_pieces", side_effect=refused), \
             mock.patch.object(strip, "floor_line_load_kip_per_in", side_effect=refused):
            # No transfer: the beam's own weight on its clear span, as before.
            reactions, basis = capacity.member_face_reactions(None, member, 1.2, 1.0)
            clear = mp.beam_clear_span_in(member)
            self.assertEqual(reactions, [1.2 * mp.beam_drop_weight_kip_per_in(member) * clear / 2.0] * 2)
            self.assertEqual(basis, "member self-weight only; no slab transfer supplied")
            # The computed common-action route hands its physical gravity actions on and is answered by them alone.
            physical = {"physical_gravity_member_actions": [], "expected_physical_gravity_case_ids": []}
            with mock.patch("Design.SMRF_Physical_Member_Actions.physical_face_reactions", return_value=([1.0, 2.0], "physical")) as route:
                self.assertEqual(capacity.member_face_reactions(physical, member, 1.2, 1.0), ([1.0, 2.0], "physical"))
            route.assert_called_once_with(physical, member, 1.2, 1.0)

    def test_without_a_slab_the_footprint_is_the_plate_and_there_is_no_strip(self):
        transfer = self.transfers[1]
        mg.clear()
        with mock.patch.object(sp, "SLAB_THICKNESS_IN", None):
            self.assertIsNone(strip.floor_pieces(1))
            self.assertEqual(strip.floor_line_load_kip_per_in(1, transfer, ("x", 0, 0), "dead"), 0.0)
            result = strip.floor_loads(1, transfer, 1.2, 1.6)
            self.assertEqual((result["beams"], result["joints"], result["total_kip"]), ({}, {}, 0.0))
            dead_kip, live_kip = strip.plate_loads_kip(0.1, 0.04)
            self.assertAlmostEqual(dead_kip, 216.0, places=10)
            self.assertAlmostEqual(live_kip, 86.4, places=10)


class UniformRoute(unittest.TestCase):
    """The one-beam design: 18 in beams on every side, 9 in strips, one transfer for every floor."""

    def setUp(self):
        self.stack = gf.frame(**LOADS)
        self.stack.__enter__()
        self.transfer = synthetic_transfer(uniform_sections())

    def tearDown(self):
        self.stack.__exit__(None, None, None)

    def test_uniform_frame_loads_on_the_transfer_route_are_the_footprint_loads(self):
        sp.FLOOR_TRANSFER = self.transfer
        footprint = fp.physical_floor(1)
        self.assertEqual(footprint["area_in2"], 498.0 * 666.0)
        self.assertEqual(mp.floor_area_in2(), 498.0 * 666.0)
        q = (1.2 * 0.1375 + 1.6 * 0.04) / 144.0
        factors = dict(floor_factor=1.0, self_weight_factor=0.0, dead_factor=1.2, live_factor=1.6, roof_extension=False)
        force, first_x, first_y, _, _ = applied(**factors)
        self.assertAlmostEqual(force, q * NZ * 498.0 * 666.0, places=8)
        self.assertAlmostEqual(first_x, q * NZ * footprint["first_moment_x_in3"], places=5)
        self.assertAlmostEqual(first_y, q * NZ * footprint["first_moment_y_in3"], places=5)
        plate_force, _, _, _, _ = applied(perimeter_strip=False, **factors)
        self.assertAlmostEqual(plate_force, q * NZ * PLATE_IN2, places=8)
        # The validator is given the plate: an edited transfer is still refused when the loads are applied.
        edited = copy.deepcopy(self.transfer)
        edited["dead_pressure_ksf"] = 0.1375 * 498.0 * 666.0 / PLATE_IN2
        sp.FLOOR_TRANSFER = edited
        with self.assertRaisesRegex(ValueError, "dead pressure"):
            applied(**factors)

    def test_uniform_strip_pieces_from_a_record_are_those_of_the_installed_design(self):
        from_record = strip.uniform_pieces(GEOMETRY, sp.B_BEAM)
        self.assertEqual(from_record, strip.floor_pieces(1))
        self.assertEqual(from_record["outstand_in"], {"x_minus": 9.0, "x_plus": 9.0, "y_minus": 9.0, "y_plus": 9.0})
        self.assertAlmostEqual(from_record["area_in2"], 498.0 * 666.0 - PLATE_IN2, places=9)

    def test_uniform_edge_family_gravity_shear_carries_the_strip(self):
        pieces = strip.uniform_pieces(GEOMETRY, sp.B_BEAM)
        depth, panel_dead, panel_live = sp.H_COL, 0.1375 * LX * LY / 144.0, 0.04 * LX * LY / 144.0
        clear = LX - depth
        for position, sides, width in (("edge", 1, 9.0), ("interior", 2, 0.0)):
            with_strip, _ = cd._face_reactions(self.transfer, "x", position, GEOMETRY, 1.2, 1.0, 0.0, column_depth_in=depth, strip=pieces)
            without, _ = cd._face_reactions(self.transfer, "x", position, GEOMETRY, 1.2, 1.0, 0.0, column_depth_in=depth)
            plate = 0.5 * sides * (1.2 * panel_dead + 1.0 * panel_live) / 4.0
            self.assertAlmostEqual(without[0], plate, places=10)
            self.assertAlmostEqual(with_strip[0], plate + (1.2 * 0.1375 + 1.0 * 0.04) / 144.0 * width * clear / 2.0, places=10)
            self.assertAlmostEqual(with_strip[1], with_strip[0], places=12)
        # The same line load as the grouped helper and the frame read.
        self.assertEqual(strip.line_load_kip_per_in(pieces, self.transfer, ("x", 0, 1), "dead"),
                         strip.floor_line_load_kip_per_in(1, self.transfer, ("x", 0, 1), "dead"))

    def test_uniform_design_beam_shear_hands_the_strip_to_its_edge_families(self):
        # The capacity-design fixture: 3 x 3 bays of 120 in, 26 in columns (94 in clear), 10 in beams (5 in strips),
        # Sds = 1, a 10 kip dead point load at every midspan, 0.01 kip/in smeared beam weight, no live case.
        import test_smrf_capacity_design as fixture
        transfer = fixture.single_load_transfer(10.0, 0.5)
        transfer["dead_pressure_ksf"], transfer["live_pressure_ksf"] = 0.144, 0.072          # 0.001 kip/in2 dead
        state = fixture.state(transfer=transfer)
        strengths = {f"{axis}_{position}": cd.probable_beam_strengths(state, position, axis)
                     for axis in ("x", "y") for position in ("edge", "interior")}
        families = cd.design_beam_shear(state, strengths, transfer)["families"]
        for key, entries in families.items():
            edge = key.endswith("edge")
            for entry, factor in zip(entries, (1.4, 0.7)):                                   # 1.2 + 0.2 Sds, 0.9 - 0.2 Sds
                self.assertEqual(entry["dead_factor"], factor)
                # Point load 5 kip per face, beam weight 0.01 x 120 / 2 = 0.6 kip, strip 0.001 x 5 x 94 / 2 = 0.235 kip.
                hand = factor * (5.0 + 0.6 + (0.235 if edge else 0.0))
                self.assertAlmostEqual(entry["gravity_reactions_kip"][0], hand, places=10, msg=key)
                self.assertAlmostEqual(entry["gravity_reactions_kip"][1], hand, places=10, msg=key)
                self.assertEqual(strip.SHEAR_BASIS_CLAUSE in entry["gravity_reaction_basis"], edge, key)
                self.assertEqual(strip.METHOD in entry["gravity_reaction_basis"], edge, key)
        self.assertAlmostEqual(families["x_edge"][0]["gravity_reactions_kip"][0], 8.169, places=10)     # 7.84 + 0.329
        self.assertAlmostEqual(families["y_edge"][1]["gravity_reactions_kip"][0], 4.0845, places=10)    # 3.92 + 0.1645
        self.assertAlmostEqual(families["x_interior"][0]["gravity_reactions_kip"][0], 7.84, places=10)

    def test_pushover_gravity_inventory_counts_the_strip(self):
        from Analysis import Pushover_Diagnostic as pushover
        sp.FLOOR_TRANSFER = self.transfer
        settings = SimpleNamespace(gravity_self_weight_factor=0.0, gravity_dead_factor=1.2, gravity_live_factor=1.6, live_pattern="all")
        total, basis = pushover.expected_gravity_total_kip(settings)
        self.assertAlmostEqual(total, (1.2 * 0.1375 + 1.6 * 0.04) / 144.0 * NZ * 498.0 * 666.0, places=8)
        self.assertIn("perimeter strip", basis)
        force, _, _, _, _ = applied(floor_factor=1.0, self_weight_factor=0.0, dead_factor=1.2, live_factor=1.6, roof_extension=False)
        self.assertAlmostEqual(force, total, places=8)


if __name__ == "__main__":
    unittest.main()
