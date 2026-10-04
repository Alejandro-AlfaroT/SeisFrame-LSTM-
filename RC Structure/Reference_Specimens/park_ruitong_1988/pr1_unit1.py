"""Park and Ruitong (1988) Unit 1: OpenSeesPy port of the OpenSees PR1 example, its checks, the fixed
experimental-history mode and the untuned input-correction cases (2026-09-27; revised 2026-09-29 after
the Unit 1 review of 2026-09-28).

Isolated reference-specimen diagnostic. Nothing here touches the project frame. The model is the official
OpenSees example PR1.tcl (N. Mitra, 16 Feb 2003, opensees.berkeley.edu/OpenSees/manuals/usermanual/1178.htm)
with its helper procedures procMKPC.tcl, procUniaxialPinching.tcl and procRC.tcl (pages 1179-1181) ported
line for line; every input is listed in UNIT1_INPUT_MANIFEST.md with its source and status. Units: N, mm,
MPa (the example's). Forces are reported in kN.

Loading protocols:
  example     procRC: displacement cycles 0 -> +y -> -y -> 0 at 0.1, 10, 10, 30, 30, ... 105, 105 mm (frozen control)
  paper       one load-controlled cycle to +-0.75 V2, Delta_y = 4/3 x the mean displacement reached, then two
              cycles at each of mu = 2 .. 7 x that model-derived Delta_y: SELF-SCALED, a diagnostic only, not an
              experiment-matched comparison (each model reaches a different demand)
  experiment  the load-controlled +-0.75 V2 = +-54.45 kN cycle, then FIXED displacement amplitudes reconstructed
              from the test (EXPERIMENT_AMPLITUDES_MM); the model's own yield displacement is reported, never used
              to rescale the history; --amplitude-shift applies the +-3 mm reading bound coherently to every peak

Input cases (review: no bond, pinching, damage or stiffness parameter is fitted in this matrix):
  example_inputs             the example's Steel02 beam steel and its confinement inputs
  measured_beam_steel_only   ReinforcingSteel with the measured plateau and hardening (paper Table 2(b), Fig. 5)
  verified_confinement_only  beam stirrup stations from Fig. 7 with R6(B) fy = 366 MPa (Table 2(a)); column unchanged
  both_corrections           both

Every run exports its history; a run that fails exports the completed states, the failing phase and target,
the solver attempts at the failing step and the exception under a *_FAILED label before the error propagates.

usage: python pr1_unit1.py --output-root <dir> [--protocol example|paper|experiment] [--case <name>]
       [--step 0.01] [--amplitude-shift 0.0] [--mirror] [--label name] [--checks] [--tolerance 1e-8]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------
# inputs (see UNIT1_INPUT_MANIFEST.md for sources and status)
# ---------------------------------------------------------------------------------------------------
STRFACTOR, LENFACTOR = 145.0, 1.0 / 25.4              # MPa -> psi, mm -> in (procMKPC works in psi/in)
COL_Y, COL_Z = 406.0, 305.0                            # column depth (in-plane) and width, mm (paper Fig. 3)
BM_Y, BM_Z = 457.0, 229.0                              # beam depth and width, mm (paper Fig. 3)
COL_COV, BM_COV1, BM_COV2 = 43.0, 42.0, 33.0           # covers to bar centre, mm (paper Fig. 7 sections 2-2, 3-3)
BEAM_LENGTH_IN, BEAM_LENGTH_OUT, COLUMN_LENGTH_CLEAR = 645.0, 1271.0, 1008.0   # mm (paper Figs. 3, 11, 12)
JOINT_WIDTH, JOINT_HEIGHT, JOINT_DEPTH = COL_Y, BM_Y, COL_Z
JOINT_VOLUME = JOINT_WIDTH * JOINT_HEIGHT * JOINT_DEPTH
H_TOTAL = 2.0 * COLUMN_LENGTH_CLEAR + JOINT_HEIGHT     # 2473 mm, column pin to pin (paper Fig. 3)
L_BEAM = BEAM_LENGTH_OUT + BEAM_LENGTH_IN + JOINT_WIDTH / 2.0   # 2119 mm, column centreline to beam pin
Y_BEAM = COLUMN_LENGTH_CLEAR + JOINT_HEIGHT / 2.0      # 1236.5 mm, beam centreline above the base pin
P_GRAVITY = 55000.0                                    # N per beam (paper section 2.6, Table 4)
X_GRAVITY = BEAM_LENGTH_IN + JOINT_WIDTH / 2.0         # 848 mm from the column centreline (paper Fig. 11)
V1_KN, V2_KN = 54.2, 72.6                              # theoretical first-hinge and both-hinge loads (paper Table 4)
LOAD_CONTROL_TARGET_KN = 0.75 * V2_KN                  # 54.45 kN, runs 1 and 2 of the test (paper section 3.2)

C_UNCONF_FC, C_UNCONF_EC = -45.9, -0.002               # column concrete (paper Table 1: 45.9 MPa at test)
C_TS_SPACE, C_TS_LENGTH, C_TS_FY, C_TS_AREA = 60.0, 1853.53, 282.0, 28.3   # column ties for confinement (example = Fig. 7 station next to the joint)
C_FY, C_ES, C_S_HRATIO, C_AS = 498.0, 196600.0, 0.004216, 201.06            # HD16 (paper Table 2; b = example)
B_UNCONF_FC, B_UNCONF_EC = -45.9, -0.002
B_TS_SPACE, B_TS_LENGTH, B_TS_FY, B_TS_AREA = 80.0, 1036.0, 282.0, 28.3     # beam stirrups for confinement (example)
B_FY, B_ES, B_AS, B_S_HRATIO = 294.0, 210400.0, 201.06, 0.002322             # D16 (paper Table 2; b = example)
BS_FSU, CS_FSU, BAR_DB = 434.0, 660.0, 16.0            # fsu D16 / HD16 (paper Table 2), bar diameter
BS_ESH = B_S_HRATIO * B_ES                              # 488.5 MPa: the example's bilinear hardening (NOT the measured 3,580)
CS_ESH = C_S_HRATIO * C_ES                              # 828.9 MPa
BST_NBARS, BSB_NBARS, CS_NBARS = 5, 2, 3                # bars anchored per bar-slip spring (paper Fig. 7)
PANEL_P = (2.1932, 4.0872, 4.4862, 4.4862e-3)           # MPa, shear-panel envelope stresses (example; PEER 2003/10 MCFT)
PANEL_STRAIN = (0.0002, 0.004465, 0.0131, 0.0269)
PANEL_RDISP, PANEL_RFORCE, PANEL_UFORCE = (0.25, 0.25), (0.15, 0.15), (0.0, 0.0)
PANEL_GAMMA_K = (1.13364492409642, 0.0, 0.10111033064469, 0.0, 0.91652498468618)
PANEL_GAMMA_D = (0.12, 0.0, 0.23, 0.0, 0.95)
PANEL_GAMMA_F = (1.11, 0.0, 0.319, 0.0, 0.125)
PANEL_GAMMA_E = 10.0
INTERFACE_MATERIAL = None                                # filled by interface_shear_material(); see manifest (tag 1 in the example)
PEAKPTS_MM = (0.1, 10, 10, 30, 30, 45, 45, 60, 60, 75, 75, 90, 90, 105, 105)
INCREMENT = 10                                          # procRC: dU = peakpts[0] / increment = 0.01 mm
PAPER_MU = (2, 3, 4, 5, 6, 7)
PAPER_CYCLES_PER_MU = 2
SOLVER = {"tolerance": 1e-8, "iterations": 150, "equilibrium_limit": 1e-6}   # the example's NormDispIncr test; --tolerance overrides
LOOSE_TOLERANCE_FLOOR = 1e-6                             # procRC's fallback tolerance
EQUILIBRIUM_SCALES = {"force_N": 80.3e3, "moment_Nmm": 80.3e3 * (2.0 * 1008.0 + 457.0)}   # characteristic: the observed maximum load and its base moment
POLISH_TOLERANCE = 1.0                                   # NormUnbalance (N, N mm mixed) for the zero-increment re-solve of an over-limit step
POLISH_ENABLED = False                                   # 2026-09-29: the re-solve fails at exactly the states that exceed the limit and the failed
                                                         # attempts break the next step (measured-steel case at 68.78 mm); rows over the limit are
                                                         # counted and reported instead (equilibrium_rows_over_limit)

# measured beam steel (paper Table 2(b), Fig. 5) for the measured-beam-steel case; the strain at fsu is NOT
# measured: Fig. 5 ends at 0.055 and eps_sf = 0.288 is the fracture strain, so eps_su carries a bounded uncertainty
BS_ESH_MEASURED, BS_EPS_SH = 3580.0, 0.0255
BS_EPS_SU, BS_EPS_SU_BOUNDS = 0.20, (0.15, 0.25)
FIG5_D16_STRESS_AT_0_05 = (370.0, 10.0)                  # Fig. 5 read at 0.05 strain (MPa, reading half-width)
R6A_FY, R6B_FY = 282.0, 366.0                            # plain round ties, stress at 0.005 strain (paper Table 2(a))

# transverse-steel stations from Fig. 7 (distances from the column face, mm): the beam's first 640 mm carries
# 9 R6(B) at 80 crs, then (after a 110 gap) 4 R6(B) at 110, (160 gap) 4 R6(A) at 180, (90 gap) 5 R6(A) at 90, 16
# to the end (640 + 110 + 330 + 160 + 540 + 90 + 360 + 16 = 2246). The column's "5 sets of R12 and R8 ties at
# 75 crs" sit inside the joint core (75 + 4 x 75 + 20 + 20 + 42 = 457 = the joint height); the column next to
# the joint carries 8 R6(A) at 60 crs = 420, which is what the example's column confinement input already is
# (rectangle 1078 + diamond 775.5 = 1853.5 mm of R6 per set). Joint hoops are never smeared into member sections.
BEAM_STATIONS = (
    {"station": "B1", "from_mm": 0.0, "to_mm": 640.0, "ties": "9 R6(B) at 80 crs", "spacing_mm": 80.0, "fy_MPa": R6B_FY,
     "hoop_length_mm": 1036.0, "area_mm2": 28.3, "elements": ("L_in", "R_in"), "note": "column face to the gravity load point (645)"},
    {"station": "B2", "from_mm": 640.0, "to_mm": 1080.0, "ties": "4 R6(B) at 110 crs (first 110 from the last B1 stirrup)", "spacing_mm": 110.0,
     "fy_MPa": R6B_FY, "hoop_length_mm": 1036.0, "area_mm2": 28.3, "elements": ("L_out", "R_out"),
     "note": "the outer elements' section; their integration point at the gravity load point sits on the B1/B2 boundary"},
    {"station": "B3", "from_mm": 1080.0, "to_mm": 1780.0, "ties": "4 R6(A) at 180 crs (after a 160 gap)", "spacing_mm": 180.0, "fy_MPa": R6A_FY,
     "hoop_length_mm": 1036.0, "area_mm2": 28.3, "elements": (), "note": "not used: the outer elements carry one section (B2)"},
    {"station": "B4", "from_mm": 1780.0, "to_mm": 2230.0, "ties": "5 R6(A) at 90 crs (after a 90 gap)", "spacing_mm": 90.0, "fy_MPa": R6A_FY,
     "hoop_length_mm": 1036.0, "area_mm2": 28.3, "elements": (), "note": "not used; low moment next to the pin"},
)
COLUMN_STATIONS = (
    {"station": "J", "where": "joint core, within the beam depth", "ties": "5 sets of R12 rectangular + R8 diamond at 75 crs",
     "used_for_member_confinement": False, "note": "joint hoops (Table 3: Vsh = 470 kN); they belong to the panel, not to a column section"},
    {"station": "C1", "where": "column next to the joint, both sides", "ties": "8 R6(A) at 60 crs = 420", "spacing_mm": 60.0, "fy_MPa": R6A_FY,
     "hoop_length_mm": 1853.53, "area_mm2": 28.3, "used_for_member_confinement": True, "matches_example_input": True},
    {"station": "C2", "where": "column toward the pins", "ties": "2 R6(B) at 120, 3 R6(B) at 50 crs at the end", "used_for_member_confinement": False,
     "note": "low-moment region next to the pins; the columns stayed elastic (paper section 4.1)"},
)
CASES = {
    "example_inputs": {"beam_steel": "example", "confinement": "example"},
    "measured_beam_steel_only": {"beam_steel": "measured", "confinement": "example"},
    "verified_confinement_only": {"beam_steel": "example", "confinement": "drawing"},
    "both_corrections": {"beam_steel": "measured", "confinement": "drawing"},
}
EXPERIMENT_AMPLITUDES_MM = (30.0, 30.0, 45.0, 45.0, 60.0, 60.0, 75.0, 75.0, 90.0, 90.0, 105.0, 105.0)
EXPERIMENT_AMPLITUDE_BASIS = ("provisional reconstruction of runs 3-26: mu x 15 mm at mu = 2 .. 7, two cycles each, from the run positions on "
                              "Fig. 16(a); audited against the pixel-read peak positions in unit1_observed_digitized_v2.json; the +-3 mm reading "
                              "bound is applied as a coherent shift of every amplitude (--amplitude-shift), never as independent shifts of single peaks")

# node and element tags (the example's)
N = {"base": 1, "col_bot": 2, "beamL_far": 3, "beamL_P": 4, "beamL_face": 5, "beamR_face": 6, "beamR_P": 7, "beamR_far": 8, "col_top": 9, "tip": 10}
E_COL = {"lower": 1, "upper": 2}
E_BEAM = {"L_out": 3, "L_in": 4, "R_in": 5, "R_out": 6}
E_JOINT = 7
MAT = {"beam_unconf": 10, "beam_conf": 20, "beam_conf_outer": 23, "beam_steel": 30, "col_unconf": 40, "col_conf": 50, "col_steel": 60,
       "bs_beamL_bot": 21, "bs_beamR_bot": 22, "bs_beamL_top": 31, "bs_beamR_top": 32,
       "bs_colB_L": 41, "bs_colB_R": 42, "bs_colT_L": 43, "bs_colT_R": 44, "panel": 5, "interface": 1}
SEC = {"column": 1, "beam": 2, "beam_outer": 3}
TRANSF = {"beam": 1, "column": 2}
JOINT_RESPONSES = ("node1BarSlipL", "node1BarSlipR", "node2BarSlipB", "node2BarSlipT", "node3BarSlipL", "node3BarSlipR",
                   "node4BarSlipB", "node4BarSlipT", "shearpanel")
Y_TOP_BAR, Y_BOT_BAR = BM_Y / 2.0 - BM_COV1, -(BM_Y / 2.0 - BM_COV1)     # first top layer and the bottom layer (mm from the beam axis)


def mkpc(unconf_fc, unconf_ec, y, z, cov, ts_space, ts_length, ts_fy, ts_area, strfactor=STRFACTOR, lenfactor=LENFACTOR):
    """procMKPC.tcl ported line for line: modified Kent-Park confined concrete from the unconfined
    strength, the section, the cover and the transverse steel; returns the example's 8-list
    [unconf fc, unconf ec, unconf fcu, unconf ecu, conf fc, conf ec, conf fcu, conf ecu] in MPa."""
    unconf_ecu = -0.004
    sec_wid, sec_dep, cover = lenfactor * z, lenfactor * y, lenfactor * cov
    ufc, ue0, uecu = -strfactor * unconf_fc, -unconf_ec, -unconf_ecu
    hoop_spc, hoop_lngth = lenfactor * ts_space, lenfactor * ts_length
    hoop_fy, hoop_area = strfactor * ts_fy, ts_area * lenfactor * lenfactor
    rho_s = (hoop_lngth * hoop_area) / ((sec_wid - 2 * cover) * (sec_dep - 2 * cover) * hoop_spc)
    b = sec_wid - 2 * cover
    temp = b / hoop_spc
    e50u = (3 + 0.002 * ufc) / (ufc - 1000)
    e50h = 3 * rho_s * math.pow(temp, 0.5) / 4
    zm = 0.5 * (ufc - 1000) / (3 + 0.002 * ufc)
    zz = 0.5 / (e50u + e50h - ue0)
    k = 1 + rho_s * hoop_fy / ufc
    ufcu = -ufc * (1 - zm * (uecu - ue0)) / strfactor
    ce0 = -k * ue0
    cfc = -k * ufc / strfactor
    cfcu = 0.2 * cfc
    cecu = -(0.8 / zz - ce0)
    return [unconf_fc, unconf_ec, ufcu, unconf_ecu, cfc, ce0, cfcu, cecu], {"rho_s": rho_s, "K": k, "Zm": zm, "Z": zz, "e50u": e50u, "e50h": e50h}


def interface_shear_material():
    """The example gives the four interface-shear springs material tag 1, defined as
    ``uniaxialMaterial Elastic 1 10000000000.0`` (N/mm): effectively rigid, the PEER report's
    assumption for this specimen (section 5, p. 49: interface-shear components elastic with the
    closed-crack stiffness, the test frame's axial restraint being unknown)."""
    return {"type": "Elastic", "args": (1.0e10,), "status": "example line: uniaxialMaterial Elastic 1 10000000000.0"}


def beam_steel_material(kind):
    """The beam longitudinal steel of a case. ``example``: the example's Steel02 (bilinear b = 0.002322, no
    plateau). ``measured``: ReinforcingSteel with the measured fy, fsu, Es, Esh and eps_sh (paper Table 2(b));
    eps_su is the centre of its declared bounds, not a measurement. The BarSlip springs keep the example's
    bilinear Eh in every case (BarSlip has no plateau-end input); that discrepancy stays explicit."""
    if kind == "example":
        return {"type": "Steel02", "args": (B_FY, B_ES, B_S_HRATIO, 18.5, 0.925, 0.15, 0.0, 0.4, 0.0, 0.5),
                "status": "example line: Steel02 30 294 210400 0.002322 18.5 0.925 0.15 0 0.4 0 0.5 (b E = 488.5 MPa, no plateau)"}
    if kind == "measured":
        return {"type": "ReinforcingSteel", "args": (B_FY, BS_FSU, B_ES, BS_ESH_MEASURED, BS_EPS_SH, BS_EPS_SU),
                "status": (f"measured fy {B_FY}, fsu {BS_FSU}, Es {B_ES}, Esh {BS_ESH_MEASURED}, eps_sh {BS_EPS_SH} (paper Table 2(b)); "
                           f"eps_su {BS_EPS_SU} is the centre of the declared bounds {BS_EPS_SU_BOUNDS} (Fig. 5 ends at 0.055; 0.288 is the fracture strain); "
                           "coupon check in steel_coupon_check.py"),
                "eps_su_bounds": BS_EPS_SU_BOUNDS}
    raise ValueError(f"beam steel must be 'example' or 'measured', got {kind!r}")


def beam_confinement(kind):
    """Confined-concrete inputs to procMKPC for the inner (station B1) and outer (station B2) beam elements.
    ``example``: the example's single input (80 crs, fy 282, 1036 mm of R6) for both. ``drawing``: Fig. 7
    stations with R6(B) fy = 366 MPa. The column's input is the drawing's station C1 in every case."""
    if kind == "example":
        st = {"spacing_mm": B_TS_SPACE, "fy_MPa": B_TS_FY, "hoop_length_mm": B_TS_LENGTH, "area_mm2": B_TS_AREA, "station": "example (uniform)"}
        return {"inner": st, "outer": dict(st)}
    if kind == "drawing":
        return {"inner": dict(BEAM_STATIONS[0]), "outer": dict(BEAM_STATIONS[1])}
    raise ValueError(f"confinement must be 'example' or 'drawing', got {kind!r}")


def _beam_section(tag, conf_mat):
    y, z = BM_Y / 2.0, BM_Z / 2.0
    ops.section("Fiber", tag)
    ops.patch("rect", conf_mat, 8, 1, BM_COV1 - y, BM_COV1 - z, y - BM_COV1, z - BM_COV1)
    ops.patch("rect", MAT["beam_unconf"], 2, 1, -y, BM_COV1 - z, BM_COV1 - y, z - BM_COV1)
    ops.patch("rect", MAT["beam_unconf"], 2, 1, y - BM_COV1, BM_COV1 - z, y, z - BM_COV1)
    ops.patch("rect", MAT["beam_unconf"], 8, 1, -y, -z, y, BM_COV1 - z)
    ops.patch("rect", MAT["beam_unconf"], 8, 1, -y, z - BM_COV1, y, z)
    ops.layer("straight", MAT["beam_steel"], 3, B_AS, y - BM_COV1, BM_COV1 - z, y - BM_COV1, z - BM_COV1)            # top layer, 3 D16
    ops.layer("straight", MAT["beam_steel"], 2, B_AS, y - BM_COV1 - BM_COV2, BM_COV1 - z, y - BM_COV1 - BM_COV2, z - BM_COV1)   # second top layer, 2 D16
    ops.layer("straight", MAT["beam_steel"], 2, B_AS, BM_COV1 - y, BM_COV1 - z, BM_COV1 - y, z - BM_COV1)            # bottom, 2 D16


def build(mirror=False, interface=None, beam_steel="example", confinement="example"):
    """The example's model (case ``example_inputs``) or one of the input-correction cases. ``mirror``
    reflects the geometry (x -> -x) keeping the joint element's node order anticlockwise, for the
    orientation-mapping check. The element and section topology is the same in every case: two beam
    sections (inner and outer elements) whose materials coincide under the example's confinement."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    s = -1.0 if mirror else 1.0
    yb = Y_BEAM
    coords = {N["base"]: (0.0, 0.0), N["col_bot"]: (0.0, COLUMN_LENGTH_CLEAR),
              N["beamL_far"]: (s * -(BEAM_LENGTH_OUT + BEAM_LENGTH_IN + JOINT_WIDTH / 2.0), yb),
              N["beamL_P"]: (s * -(BEAM_LENGTH_IN + JOINT_WIDTH / 2.0), yb), N["beamL_face"]: (s * -JOINT_WIDTH / 2.0, yb),
              N["beamR_face"]: (s * JOINT_WIDTH / 2.0, yb), N["beamR_P"]: (s * (BEAM_LENGTH_IN + JOINT_WIDTH / 2.0), yb),
              N["beamR_far"]: (s * (BEAM_LENGTH_OUT + BEAM_LENGTH_IN + JOINT_WIDTH / 2.0), yb),
              N["col_top"]: (0.0, COLUMN_LENGTH_CLEAR + JOINT_HEIGHT), N["tip"]: (0.0, 2.0 * COLUMN_LENGTH_CLEAR + JOINT_HEIGHT)}
    for tag, (x, y) in coords.items():
        ops.node(tag, x, y)
    ops.fix(N["base"], 1, 1, 0)
    ops.fix(N["beamL_far"], 0, 1, 0)
    ops.fix(N["beamR_far"], 0, 1, 0)

    # materials: concrete (procMKPC), steel, bar slip, shear panel, interface shear
    conf = beam_confinement(confinement)
    beam_conc_in, beam_kp_in = mkpc(B_UNCONF_FC, B_UNCONF_EC, BM_Y, BM_Z, BM_COV1, conf["inner"]["spacing_mm"], conf["inner"]["hoop_length_mm"],
                                    conf["inner"]["fy_MPa"], conf["inner"]["area_mm2"])
    beam_conc_out, beam_kp_out = mkpc(B_UNCONF_FC, B_UNCONF_EC, BM_Y, BM_Z, BM_COV1, conf["outer"]["spacing_mm"], conf["outer"]["hoop_length_mm"],
                                      conf["outer"]["fy_MPa"], conf["outer"]["area_mm2"])
    col_conc, col_kp = mkpc(C_UNCONF_FC, C_UNCONF_EC, COL_Y, COL_Z, COL_COV, C_TS_SPACE, C_TS_LENGTH, C_TS_FY, C_TS_AREA)
    steel = beam_steel_material(beam_steel)
    ops.uniaxialMaterial("Concrete01", MAT["beam_unconf"], beam_conc_in[0], beam_conc_in[1], beam_conc_in[2], beam_conc_in[3])
    ops.uniaxialMaterial("Concrete01", MAT["beam_conf"], beam_conc_in[4], beam_conc_in[5], beam_conc_in[6], beam_conc_in[7])
    ops.uniaxialMaterial("Concrete01", MAT["beam_conf_outer"], beam_conc_out[4], beam_conc_out[5], beam_conc_out[6], beam_conc_out[7])
    ops.uniaxialMaterial(steel["type"], MAT["beam_steel"], *steel["args"])
    ops.uniaxialMaterial("Concrete01", MAT["col_unconf"], col_conc[0], col_conc[1], col_conc[2], col_conc[3])
    ops.uniaxialMaterial("Concrete01", MAT["col_conf"], col_conc[4], col_conc[5], col_conc[6], col_conc[7])
    ops.uniaxialMaterial("Steel02", MAT["col_steel"], C_FY, C_ES, C_S_HRATIO, 18.5, 0.925, 0.15, 0.0, 0.4, 0.0, 0.5)
    bs_fc = -B_UNCONF_FC
    for tag in (MAT["bs_beamL_bot"], MAT["bs_beamR_bot"]):
        ops.uniaxialMaterial("BarSlip", tag, bs_fc, B_FY, B_ES, BS_FSU, BS_ESH, BAR_DB, JOINT_WIDTH, BSB_NBARS, COL_Z, BM_Y, "strong", "beamBot")
    for tag in (MAT["bs_beamL_top"], MAT["bs_beamR_top"]):
        ops.uniaxialMaterial("BarSlip", tag, bs_fc, B_FY, B_ES, BS_FSU, BS_ESH, BAR_DB, JOINT_WIDTH, BST_NBARS, COL_Z, BM_Y, "strong", "beamTop")
    cs_fc = -C_UNCONF_FC
    for tag in (MAT["bs_colB_L"], MAT["bs_colB_R"], MAT["bs_colT_L"], MAT["bs_colT_R"]):
        ops.uniaxialMaterial("BarSlip", tag, cs_fc, C_FY, C_ES, CS_FSU, CS_ESH, BAR_DB, JOINT_HEIGHT, CS_NBARS, COL_Z, COL_Y, "strong", "column")
    p = [v * JOINT_VOLUME for v in PANEL_P]
    n = [-v for v in p]
    ns = [-v for v in PANEL_STRAIN]
    ops.uniaxialMaterial("Pinching4", MAT["panel"], p[0], PANEL_STRAIN[0], p[1], PANEL_STRAIN[1], p[2], PANEL_STRAIN[2], p[3], PANEL_STRAIN[3],
                         n[0], ns[0], n[1], ns[1], n[2], ns[2], n[3], ns[3],
                         PANEL_RDISP[0], PANEL_RFORCE[0], PANEL_UFORCE[0], PANEL_RDISP[1], PANEL_RFORCE[1], PANEL_UFORCE[1],
                         *PANEL_GAMMA_K, *PANEL_GAMMA_D, *PANEL_GAMMA_F, PANEL_GAMMA_E, "energy")
    interface = interface or interface_shear_material()
    ops.uniaxialMaterial(interface["type"], MAT["interface"], *interface["args"])

    # fibre sections (the example's patch/layer layout; local y is the in-plane depth)
    y, z = COL_Y / 2.0, COL_Z / 2.0
    ops.section("Fiber", SEC["column"])
    ops.patch("rect", MAT["col_conf"], 8, 1, COL_COV - y, COL_COV - z, y - COL_COV, z - COL_COV)
    ops.patch("rect", MAT["col_unconf"], 2, 1, -y, COL_COV - z, COL_COV - y, z - COL_COV)
    ops.patch("rect", MAT["col_unconf"], 2, 1, y - COL_COV, COL_COV - z, y, z - COL_COV)
    ops.patch("rect", MAT["col_unconf"], 8, 1, -y, -z, y, COL_COV - z)
    ops.patch("rect", MAT["col_unconf"], 8, 1, -y, z - COL_COV, y, z)
    ops.layer("straight", MAT["col_steel"], 3, C_AS, y - COL_COV, COL_COV - z, y - COL_COV, z - COL_COV)
    ops.layer("straight", MAT["col_steel"], 2, C_AS, 0.0, COL_COV - z, 0.0, z - COL_COV)
    ops.layer("straight", MAT["col_steel"], 3, C_AS, COL_COV - y, COL_COV - z, COL_COV - y, z - COL_COV)
    _beam_section(SEC["beam"], MAT["beam_conf"])
    _beam_section(SEC["beam_outer"], MAT["beam_conf_outer"])

    ops.geomTransf("Linear", TRANSF["beam"])
    ops.geomTransf("Linear", TRANSF["column"])
    ops.element("nonlinearBeamColumn", E_COL["lower"], N["base"], N["col_bot"], 5, SEC["column"], TRANSF["column"])
    ops.element("nonlinearBeamColumn", E_COL["upper"], N["col_top"], N["tip"], 5, SEC["column"], TRANSF["column"])
    # every beam element runs toward +x so that the section's local y (its "top") points up; in the
    # mirrored geometry the node order is therefore reversed, which is what keeps the asymmetric
    # section (5 D16 top, 2 D16 bottom) the right way up
    def beam(tag, n_i, n_j, npts, sec):
        if mirror:
            n_i, n_j = n_j, n_i
        ops.element("nonlinearBeamColumn", tag, n_i, n_j, npts, sec, TRANSF["beam"])
    beam(E_BEAM["L_out"], N["beamL_far"], N["beamL_P"], 3, SEC["beam_outer"])
    beam(E_BEAM["L_in"], N["beamL_P"], N["beamL_face"], 2, SEC["beam"])
    beam(E_BEAM["R_in"], N["beamR_face"], N["beamR_P"], 2, SEC["beam"])
    beam(E_BEAM["R_out"], N["beamR_P"], N["beamR_far"], 3, SEC["beam_outer"])
    # the joint: nodes anticlockwise bottom, right, top, left; 13 materials = (L, R, interface) at node 1,
    # (B, T, interface) at node 2, (L, R, interface) at node 3, (B, T, interface) at node 4, then the panel
    right, left = (N["beamL_face"], N["beamR_face"]) if mirror else (N["beamR_face"], N["beamL_face"])
    right_mats, left_mats = ((MAT["bs_beamL_bot"], MAT["bs_beamL_top"]), (MAT["bs_beamR_bot"], MAT["bs_beamR_top"])) if mirror else \
                            ((MAT["bs_beamR_bot"], MAT["bs_beamR_top"]), (MAT["bs_beamL_bot"], MAT["bs_beamL_top"]))
    ops.element("beamColumnJoint", E_JOINT, N["col_bot"], right, N["col_top"], left,
                MAT["bs_colB_L"], MAT["bs_colB_R"], MAT["interface"], right_mats[0], right_mats[1], MAT["interface"],
                MAT["bs_colT_L"], MAT["bs_colT_R"], MAT["interface"], left_mats[0], left_mats[1], MAT["interface"], MAT["panel"])
    # gravity: the example's "load ... -const" (a constant nodal load, not scaled by the pattern's load
    # factor) under "integrator LoadControl 0"; OpenSeesPy's load command has no -const flag, so the
    # same thing is a Constant time series pattern followed by one zero-increment static step
    ops.timeSeries("Constant", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(N["beamL_P"], 0.0, -P_GRAVITY, 0.0)
    ops.load(N["beamR_P"], 0.0, -P_GRAVITY, 0.0)
    ops.system("ProfileSPD")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.0)
    ops.test("NormDispIncr", 1e-8, 150, 0)
    ops.algorithm("Newton")
    ops.numberer("RCM")
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("gravity step failed")
    ops.loadConst("-time", 0.0)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(N["tip"], 1.0, 0.0, 0.0)
    case = {"beam_steel": beam_steel, "confinement": confinement, "beam_steel_material": steel,
            "beam_confinement": {"inner": {**conf["inner"], "kent_park": beam_kp_in, "concrete01": beam_conc_in},
                                 "outer": {**conf["outer"], "kent_park": beam_kp_out, "concrete01": beam_conc_out}},
            "column_confinement": {"input": "example = Fig. 7 station C1 (8 R6(A) at 60 crs; rectangle 1078 + diamond 775.5 = 1853.5 mm of R6 per set)",
                                   "kent_park": col_kp, "concrete01": col_conc},
            "bar_slip_steel_Eh_MPa": BS_ESH, "bar_slip_note": "BarSlip's bilinear Eh is the example's 488.5 MPa in every case (no plateau-end input)"}
    return {"mirror": mirror, "case": case, "beam_concrete": beam_conc_in, "beam_kent_park": beam_kp_in, "column_concrete": col_conc,
            "column_kent_park": col_kp, "interface": interface, "coords": coords, "gravity_reactions_after_step": _reactions()}


def _reactions():
    ops.reactions()
    return {"base_x": ops.nodeReaction(N["base"], 1), "base_y": ops.nodeReaction(N["base"], 2),
            "beamL_far_y": ops.nodeReaction(N["beamL_far"], 2), "beamR_far_y": ops.nodeReaction(N["beamR_far"], 2)}


# ---------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------
def example_targets(peakpts=PEAKPTS_MM, sign=1.0):
    """procRC: for each peak y, 0 -> +y -> -y -> 0 (its first entry also sets the step)."""
    targets = []
    for y in peakpts:
        targets += [("disp", sign * y), ("disp", -sign * y), ("disp", 0.0)]
    return targets


class AnalysisFailure(RuntimeError):
    """A step that no attempt of the recovery chain converged; carries where and how it failed."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class Recorder:
    """One row per accepted analysis step (and per accepted sub-step), in the physical sign convention of
    the unmirrored specimen, with the solver settings that produced it."""

    ATTEMPT_CODES = {("Newton", "requested"): 0, ("KrylovNewton", "requested"): 1, ("Newton", "loose"): 2, ("KrylovNewton", "loose"): 3}

    def __init__(self, mirror):
        self.sign = -1.0 if mirror else 1.0
        self.mirror = mirror
        self.rows = []
        self.failures, self.retries = 0, 0
        self.recoveries = {}
        self.accepted = []                        # per accepted step: {"code", "algorithm", "tolerance"}
        self.last_attempt = None
        self.failed_attempts = None
        self.current = {"phase": "gravity", "target": None, "control": None}
        self._fiber_with_mat = None

    def _fiber(self, tag, sec, y, mat):
        if self._fiber_with_mat is not False:
            try:
                out = ops.eleResponse(tag, "section", sec, "fiber", y, 0.0, mat, "stressStrain")
                if len(out) >= 2:
                    self._fiber_with_mat = True
                    return out[0], out[1]
            except Exception:
                pass
            self._fiber_with_mat = False
        out = ops.eleResponse(tag, "section", sec, "fiber", y, 0.0, "stressStrain")
        return out[0], out[1]

    def record(self, phase, *, target=None, substep=0, step_complete=1):
        s = self.sign
        ops.reactions()
        row = {"u_tip": s * ops.nodeDisp(N["tip"], 1), "V": s * ops.getLoadFactor(1), "phase": phase,
               "u_col_bot": s * ops.nodeDisp(N["col_bot"], 1), "u_col_top": s * ops.nodeDisp(N["col_top"], 1),
               "r_base_x": s * ops.nodeReaction(N["base"], 1), "r_base_y": ops.nodeReaction(N["base"], 2),
               "r_beamL_y": ops.nodeReaction(N["beamL_far"], 2), "r_beamR_y": ops.nodeReaction(N["beamR_far"], 2),
               "rot_col_bot": s * ops.nodeDisp(N["col_bot"], 3), "rot_col_top": s * ops.nodeDisp(N["col_top"], 3),
               "rot_beamL_face": s * ops.nodeDisp(N["beamL_face"], 3), "rot_beamR_face": s * ops.nodeDisp(N["beamR_face"], 3)}
        for name in JOINT_RESPONSES:
            stress, strain = ops.eleResponse(E_JOINT, name, "stressStrain")
            # a reflection keeps bar tension as tension (bar-slip pairs unchanged) and flips shear (panel pair)
            flip = s if name == "shearpanel" else 1.0
            row[f"{name}_F"], row[f"{name}_d"] = flip * stress, flip * strain
        d = ops.eleResponse(E_JOINT, "deformation")
        # the element's deformation output: its bar-slip and interface components are invariant under a reflection
        # of the geometry (they follow the element's own left/right convention), its panel component flips like
        # the shear strain; recorded raw, mapped by reflection_map() in the comparison
        row["joint_def_barslip"], row["joint_def_interface"], row["joint_def_panel"], row["joint_def_total"] = d
        for label, tag in (("colL", E_COL["lower"]), ("colU", E_COL["upper"]), ("bLo", E_BEAM["L_out"]), ("bLi", E_BEAM["L_in"]),
                           ("bRi", E_BEAM["R_in"]), ("bRo", E_BEAM["R_out"])):
            bd = ops.eleResponse(tag, "basicDeformation")
            row[f"{label}_eps"], row[f"{label}_thi"], row[f"{label}_thj"] = bd[0], s * bd[1], s * bd[2]
        # curvature at the beam sections at the column faces and at the column sections at the joint
        # (the face section is the beam element's last or first integration point; reversed node order in the mirror)
        face_L, face_R = (1, 2) if self.mirror else (2, 1)
        grav_L, grav_R = (1, 3) if self.mirror else (3, 1)
        for label, tag, sec in (("kap_beamL_face", E_BEAM["L_in"], face_L), ("kap_beamR_face", E_BEAM["R_in"], face_R),
                                ("kap_col_bot", E_COL["lower"], 5), ("kap_col_top", E_COL["upper"], 1)):
            sd = ops.eleResponse(tag, "section", sec, "deformation")
            row[label] = sd[1]          # raw section curvature (its reflection rule is per member, see reflection_map)
        # section and bar histories at the beam sections at the column faces and at the gravity load points:
        # section moment (raw sign) and the stress/strain of the first top layer and of the bottom layer
        for label, tag, sec in (("bLface", E_BEAM["L_in"], face_L), ("bRface", E_BEAM["R_in"], face_R),
                                ("bLgrav", E_BEAM["L_out"], grav_L), ("bRgrav", E_BEAM["R_out"], grav_R)):
            sf = ops.eleResponse(tag, "section", sec, "force")
            row[f"{label}_M"] = sf[1]
            row[f"{label}_sig_top"], row[f"{label}_eps_top"] = self._fiber(tag, sec, Y_TOP_BAR, MAT["beam_steel"])
            row[f"{label}_sig_bot"], row[f"{label}_eps_bot"] = self._fiber(tag, sec, Y_BOT_BAR, MAT["beam_steel"])
        a = self.last_attempt or {"code": -1, "tolerance": 0.0}
        row["solver_code"], row["solver_tol"], row["substep"], row["step_complete"] = a["code"], a["tolerance"], substep, step_complete
        # the residual in the physical (unmirrored) convention, exactly as evaluate() forms it: the tip load and the
        # base shear carry the reflection sign, the beam reactions keep their nodes (a reflected model has its
        # "left" beam at +x, which the sign on V absorbs)
        row["equilibrium_residual"] = _residual_from(row["V"], row["r_base_x"], row["r_base_y"], row["r_beamL_y"], row["r_beamR_y"])
        self.rows.append(row)


def _residual_from(v, rx, ry, rl, rr):
    """Normalized force and moment equilibrium residual of a state from its reactions (support forces on the
    structure) against the applied tip load and the two gravity loads; characteristic scales: the observed
    maximum load and its base moment, the gravity total for the vertical sum."""
    fx = abs(rx + v) / EQUILIBRIUM_SCALES["force_N"]
    fy = abs(ry + rl + rr - 2.0 * P_GRAVITY) / (2.0 * P_GRAVITY)
    mz = abs(-v * H_TOTAL + (rr - rl) * L_BEAM) / EQUILIBRIUM_SCALES["moment_Nmm"]
    return max(fx, fy, mz)


def _polish(rec, integrator_args):
    """Zero-increment re-solve at the current (committed) state with a force-unbalance test, for a recorded
    step whose equilibrium residual exceeds the limit. The displacement-increment test the example uses does
    not bound the unbalanced force, and a committed step cannot be reverted in OpenSeesPy."""
    if integrator_args[0] == "DisplacementControl":
        ops.integrator("DisplacementControl", integrator_args[1], integrator_args[2], 0.0)
    else:
        ops.integrator("LoadControl", 0.0)
    ops.test("NormUnbalance", POLISH_TOLERANCE, SOLVER["iterations"], 0)
    for algorithm in ("Newton", "KrylovNewton"):
        ops.algorithm(algorithm)
        if ops.analyze(1) == 0:
            rec.recoveries["equilibrium_polish_" + algorithm] = rec.recoveries.get("equilibrium_polish_" + algorithm, 0) + 1
            break
    else:
        rec.recoveries["equilibrium_polish_failed"] = rec.recoveries.get("equilibrium_polish_failed", 0) + 1
    ops.test("NormDispIncr", SOLVER["tolerance"], SOLVER["iterations"], 0)
    ops.algorithm("Newton")


def _record_step(rec, phase, integrator_args, **kw):
    """Record the accepted step; if its equilibrium residual exceeds the limit, polish the state and record
    it again in place (the row carries the residual the state finally has)."""
    rec.record(phase, **kw)
    if rec.rows[-1]["equilibrium_residual"] > SOLVER["equilibrium_limit"]:
        rec.recoveries["over_equilibrium_limit"] = rec.recoveries.get("over_equilibrium_limit", 0) + 1
        if not POLISH_ENABLED:
            return
        _polish(rec, integrator_args)
        rec.rows.pop()
        rec.record(phase, **kw)
        if rec.rows[-1]["equilibrium_residual"] > SOLVER["equilibrium_limit"]:
            rec.recoveries["over_equilibrium_limit_after_polish"] = rec.recoveries.get("over_equilibrium_limit_after_polish", 0) + 1


def _analyze_step(rec, integrator_args, allow_loose=True):
    """One step through the recovery chain: Newton at the requested tolerance, then KrylovNewton, then (only
    when ``allow_loose``) Newton and KrylovNewton at the loose tolerance (procRC's 1e-6 fallback, or 100 x the
    requested one). Every attempt is recorded with its algorithm and tolerance; the accepted one is what the
    row carries. The equilibrium residual of the accepted state is evaluated by the recorder (review 2026-09-28:
    normalized residuals no greater than 1e-6) and an over-limit row is polished there."""
    ops.integrator(*integrator_args)
    loose = max(LOOSE_TOLERANCE_FLOOR, SOLVER["tolerance"] * 100.0)
    attempts = []
    for (algorithm, kind), code in Recorder.ATTEMPT_CODES.items():
        if kind == "loose" and not allow_loose:
            break
        tol = SOLVER["tolerance"] if kind == "requested" else loose
        ops.test("NormDispIncr", tol, SOLVER["iterations"], 0)
        ops.algorithm(algorithm)
        ok = ops.analyze(1) == 0
        attempts.append({"algorithm": algorithm, "tolerance": tol, "iterations": SOLVER["iterations"], "converged": ok})
        if code == 1 and not attempts[0]["converged"] and len(attempts) == 2:
            rec.retries += 1
        if ok:
            if code > 0:
                name = algorithm + ("" if kind == "requested" else "_loose_tol")
                rec.recoveries[name] = rec.recoveries.get(name, 0) + 1
            rec.last_attempt = {"code": code, "algorithm": algorithm, "tolerance": tol}
            rec.accepted.append({"code": code, "algorithm": algorithm, "tolerance": tol})
            ops.test("NormDispIncr", SOLVER["tolerance"], SOLVER["iterations"], 0)
            ops.algorithm("Newton")
            return True
    ops.test("NormDispIncr", SOLVER["tolerance"], SOLVER["iterations"], 0)
    ops.algorithm("Newton")
    if allow_loose:
        rec.failures += 1
    rec.last_attempt = None
    rec.failed_attempts = attempts
    return False


def _failure_record(rec, exc):
    try:
        u, lf = ops.nodeDisp(N["tip"], 1), ops.getLoadFactor(1)
    except Exception:
        u, lf = None, None
    return {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
            "phase": dict(rec.current), "u_tip_mm_at_failure": None if u is None else rec.sign * u,
            "V_kN_at_failure": None if lf is None else rec.sign * lf / 1000.0,
            "attempts_at_failing_step": exc.record.get("attempts") if isinstance(exc, AnalysisFailure) else rec.failed_attempts,
            "recoveries_so_far": dict(rec.recoveries), "completed_rows": len(rec.rows),
            "last_completed_phase": rec.rows[-1]["phase"] if rec.rows else None,
            "solver_requested": dict(SOLVER), "loose_tolerance": max(LOOSE_TOLERANCE_FLOOR, SOLVER["tolerance"] * 100.0)}


def run_displacement_cycles(rec, targets, step, phase_prefix):
    """Displacement control at the tip through a list of ("disp", target) entries, ``step`` mm per step.
    A step that fails the recovery chain is retried as ten sub-steps, each recorded on acceptance; a
    sub-step that fails raises AnalysisFailure with the phase, target, displacement and attempts."""
    current = ops.nodeDisp(N["tip"], 1)
    for k, (_, target) in enumerate(targets):
        phase = f"{phase_prefix}{k}"
        rec.current = {"phase": phase, "target": target, "control": "displacement"}
        n_steps = max(1, int(round(abs(target - current) / step)))
        du = (target - current) / n_steps
        for _ in range(n_steps):
            # the example's chain (requested tolerance, KrylovNewton, loose tolerance), then ten sub-steps with
            # the same chain; a sub-step at the requested tolerance can hit a material update failure where the
            # loose full step passes (seen at 50 mm in the example case), so the loose fallback stays first
            if _analyze_step(rec, ("DisplacementControl", N["tip"], 1, du)):
                _record_step(rec, phase, ("DisplacementControl", N["tip"], 1, du), target=target)
                continue
            rec.recoveries["substep_x10"] = rec.recoveries.get("substep_x10", 0) + 1
            for sub in range(1, 11):
                if not _analyze_step(rec, ("DisplacementControl", N["tip"], 1, du / 10.0)):
                    u = ops.nodeDisp(N["tip"], 1)
                    raise AnalysisFailure(f"step failed at u = {u:.3f} mm (phase {phase}, target {target}, sub-step {sub} of 10); "
                                          f"recoveries so far {rec.recoveries}",
                                          {"phase": phase, "target": target, "substep": sub, "u_tip_mm": u, "attempts": rec.failed_attempts})
                _record_step(rec, phase, ("DisplacementControl", N["tip"], 1, du / 10.0), target=target, substep=sub, step_complete=1 if sub == 10 else 0)
        current = ops.nodeDisp(N["tip"], 1)


def _load_control_cycle(rec, sign, force_step_n=500.0, target_n=LOAD_CONTROL_TARGET_KN * 1000.0):
    """Runs 1 and 2 of the test: load control to +target, -target, then back to zero load. Returns the
    signed displacements reached at the two targets."""
    reached = {}
    for label, goal in (("pos", sign * target_n), ("neg", -sign * target_n), ("zero", 0.0)):
        phase = f"lc_{label}"
        rec.current = {"phase": phase, "target": goal, "control": "load"}
        current = ops.getLoadFactor(1)
        n_steps = max(1, int(round(abs(goal - current) / force_step_n)))
        dl = (goal - current) / n_steps
        for _ in range(n_steps):
            if not _analyze_step(rec, ("LoadControl", dl)):
                lf = ops.getLoadFactor(1)
                raise AnalysisFailure(f"load control failed at V = {lf / 1000:.2f} kN (phase {phase}, target {goal / 1000:.2f} kN)",
                                      {"phase": phase, "target": goal, "substep": 0, "V_N": lf, "attempts": rec.failed_attempts})
            _record_step(rec, phase, ("LoadControl", dl), target=goal)
        reached[label] = sign * ops.nodeDisp(N["tip"], 1)
    return reached


def run_example_protocol(rec, step=None, sign=1.0):
    step = step or PEAKPTS_MM[0] / INCREMENT
    run_displacement_cycles(rec, example_targets(sign=sign), step, "ex")
    return {"protocol": "example", "step_mm": step, "peakpts_mm": PEAKPTS_MM,
            "protocol_status": "the example's history (frozen control); its 10 mm cycle stands in for the load-controlled runs 1-2 and its "
                               "amplitudes assume Delta_y = 15 mm"}


def run_paper_protocol(rec, step=0.01, sign=1.0, force_step_n=500.0):
    """Section 3.2 read literally: load control to +-0.75 V2, Delta_y from the two displacements the MODEL
    reaches, then displacement cycles at mu x that Delta_y. Self-scaled: a diagnostic, not an
    experiment-matched comparison (review 2026-09-28)."""
    reached = _load_control_cycle(rec, sign, force_step_n)
    delta_y = (4.0 / 3.0) * 0.5 * (abs(reached["pos"]) + abs(reached["neg"]))
    targets = []
    for mu in PAPER_MU:
        for _ in range(PAPER_CYCLES_PER_MU):
            targets += [("disp", sign * mu * delta_y), ("disp", -sign * mu * delta_y), ("disp", 0.0)]
    run_displacement_cycles(rec, targets, step, "mu")
    return {"protocol": "paper", "step_mm": step, "load_control_target_kN": LOAD_CONTROL_TARGET_KN,
            "delta_y1_mm": abs(reached["pos"]), "delta_y2_mm": abs(reached["neg"]), "delta_y_mm": delta_y,
            "mu_levels": PAPER_MU, "cycles_per_level": PAPER_CYCLES_PER_MU,
            "protocol_status": "SELF-SCALED: the mu amplitudes follow the model's own Delta_y, so the demand differs from the experiment's; "
                               "archived diagnostic, not a fit comparison"}


def run_experiment_protocol(rec, step=0.01, sign=1.0, amplitude_shift_mm=0.0, force_step_n=500.0):
    """The experimental history: the load-controlled +-0.75 V2 cycle (its displacements are the initial-
    compliance observables), then the FIXED reconstructed amplitudes of runs 3-26. The model-derived
    Delta_y is reported and never used to rescale the history."""
    reached = _load_control_cycle(rec, sign, force_step_n)
    amplitudes = [a + amplitude_shift_mm for a in EXPERIMENT_AMPLITUDES_MM]
    targets = []
    for a in amplitudes:
        targets += [("disp", sign * a), ("disp", -sign * a), ("disp", 0.0)]
    run_displacement_cycles(rec, targets, step, "mu")
    return {"protocol": "experiment", "step_mm": step, "load_control_target_kN": LOAD_CONTROL_TARGET_KN,
            "delta_at_pos_target_mm": reached["pos"], "delta_at_neg_target_mm": reached["neg"],
            "delta_y_model_derived_mm": (4.0 / 3.0) * 0.5 * (abs(reached["pos"]) + abs(reached["neg"])),
            "delta_y_model_derived_status": "reported separately; not used for the amplitudes",
            "amplitudes_mm": amplitudes, "amplitude_shift_mm": amplitude_shift_mm, "amplitude_basis": EXPERIMENT_AMPLITUDE_BASIS,
            "protocol_status": "fixed experimental history: load-controlled runs 1-2 then reconstructed displacement amplitudes"}


# ---------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------
def _cycles_from_phases(rows, prefix):
    """Group the recorded rows into loading cycles (three phases each: +peak, -peak, back to zero)."""
    phases = [r["phase"] for r in rows]
    indices = sorted({int(p[len(prefix):]) for p in phases if p.startswith(prefix)})
    cycles = []
    for c in range(0, len(indices), 3):
        block = {f"{prefix}{indices[c + j]}" for j in range(3) if c + j < len(indices)}
        idx = [i for i, p in enumerate(phases) if p in block]
        if idx:
            cycles.append((idx[0], idx[-1]))
    return cycles


def _interp_crossing(x, y, i):
    """Linear interpolation of x at the zero of y between points i and i + 1."""
    t = y[i] / (y[i] - y[i + 1])
    return float(x[i] + t * (x[i + 1] - x[i]))


def _branch_metrics(us, vs, i_peak):
    """The branch after the reversal at i_peak (fit criteria, 2026-09-28): the first zero-force
    crossing (residual displacement, linear interpolation between recorder points), the first
    zero-displacement crossing on that same branch with its force sign retained, and the unloading chord
    between the first 80 % and 20 % force levels measured toward zero force, with its endpoints."""
    v_peak, u_peak = float(vs[i_peak]), float(us[i_peak])
    sgn = 1.0 if v_peak >= 0.0 else -1.0
    out = {"peak_kN": v_peak, "u_peak_mm": u_peak, "residual_displacement_mm": None, "residual_index": None,
           "force_at_zero_displacement_kN": None, "zero_displacement_index": None, "unloading_chord_80_20": None}
    for i in range(i_peak, len(vs) - 1):
        if vs[i] * sgn > 0.0 >= vs[i + 1] * sgn:
            out["residual_displacement_mm"], out["residual_index"] = _interp_crossing(us, vs, i), i
            break
    for i in range(i_peak, len(us) - 1):
        if us[i] * sgn > 0.0 >= us[i + 1] * sgn:
            out["force_at_zero_displacement_kN"], out["zero_displacement_index"] = _interp_crossing(vs, us, i), i
            break

    def level(frac):
        target = frac * v_peak
        for i in range(i_peak, len(vs) - 1):
            if (vs[i] - target) * sgn >= 0.0 > (vs[i + 1] - target) * sgn:
                t = (vs[i] - target) / (vs[i] - vs[i + 1])
                return float(us[i] + t * (us[i + 1] - us[i])), float(target)
        return None
    p80, p20 = level(0.8), level(0.2)
    if p80 and p20 and p80[0] != p20[0]:
        out["unloading_chord_80_20"] = {"u_80_mm": p80[0], "v_80_kN": p80[1], "u_20_mm": p20[0], "v_20_kN": p20[1],
                                        "k_kN_per_mm": (p80[1] - p20[1]) / (p80[0] - p20[0]), "du_mm": abs(p80[0] - p20[0])}
    return out


def evaluate(rows, meta):
    u = np.array([r["u_tip"] for r in rows]); v = np.array([r["V"] for r in rows]) / 1000.0
    h = H_TOTAL
    # equilibrium: sum Fx, sum Fy, moment about the base pin (reactions vs applied)
    rx = np.array([r["r_base_x"] for r in rows]); ry = np.array([r["r_base_y"] for r in rows])
    rl = np.array([r["r_beamL_y"] for r in rows]); rr = np.array([r["r_beamR_y"] for r in rows])
    fx = rx + v * 1000.0
    fy = ry + rl + rr - 2.0 * P_GRAVITY
    # moment about the base pin, M = sum(x Fy - y Fx): the tip load gives -H V, the beam support forces
    # +-L Ry, the two gravity loads cancel (+-x_P); reactions are the support forces on the structure
    mz = -v * 1000.0 * h + (rr - rl) * L_BEAM
    # one declared denominator for every equilibrium statement (review 2026-09-30): the fixed characteristic
    # scales of EQUILIBRIUM_SCALES, the same ones the per-row residual uses
    scale_v = EQUILIBRIUM_SCALES["force_N"]
    scale_fy = 2.0 * P_GRAVITY
    scale_m = EQUILIBRIUM_SCALES["moment_Nmm"]
    rel = (float(np.max(np.abs(fx)) / scale_v), float(np.max(np.abs(fy)) / scale_fy), float(np.max(np.abs(mz)) / scale_m))
    equilibrium = {"sum_fx_max_abs_N": float(np.max(np.abs(fx))), "sum_fy_max_abs_N": float(np.max(np.abs(fy))),
                   "moment_about_base_max_abs_Nmm": float(np.max(np.abs(mz))),
                   "sum_fx_relative": rel[0], "sum_fy_relative": rel[1], "moment_relative": rel[2],
                   "scales": {"force_N": scale_v, "vertical_force_N": scale_fy, "moment_Nmm": scale_m,
                              "basis": "fixed characteristic scales (observed maximum load 80.3 kN, its base moment, the gravity total); identical to the per-row gate"},
                   "limit_relative": 1e-6, "passes_1e-6": bool(max(rel) <= 1e-6),
                   "basis": "reactions (base pin, beam rollers) against the applied tip load and the two gravity loads; V acts at the tip, "
                            "beam reactions at +-2119 mm; characteristic scales: 80.3 kN, 2 P, 80.3 kN x H (one declared denominator)"}
    prefix = "ex" if meta["protocol"] == "example" else "mu"
    cycles = []
    spans = _cycles_from_phases(rows, prefix)
    for k, (a, b) in enumerate(spans):
        seg = slice(max(a - 1, 0), b + 1)
        us, vs = u[seg], v[seg]
        seg_rows = rows[seg]
        i_max, i_min = int(np.argmax(vs)), int(np.argmin(vs))
        work = float(np.sum(0.5 * (vs[1:] + vs[:-1]) * np.diff(us)))
        r_pos = seg_rows[i_max]
        comp = {"barslip": r_pos["joint_def_barslip"], "interface": r_pos["joint_def_interface"], "panel": r_pos["joint_def_panel"],
                "total": r_pos["joint_def_total"]}
        cycles.append({"cycle": k + 1, "peak_pos_kN": float(vs[i_max]), "u_at_peak_pos_mm": float(us[i_max]),
                       "peak_neg_kN": float(vs[i_min]), "u_at_peak_neg_mm": float(us[i_min]),
                       "amplitude_mm": float(max(abs(us[i_max]), abs(us[i_min]))),
                       "signed_work_kN_mm": work,
                       "after_pos_peak": _branch_metrics(us, vs, i_max), "after_neg_peak": _branch_metrics(us, vs, i_min),
                       "joint_def_components_at_pos_peak": comp,
                       "joint_component_times_height_over_tip": ({key: float(abs(val) * h / abs(r_pos["u_tip"])) for key, val in comp.items()}
                                                                 if r_pos["u_tip"] else None),
                       "joint_component_times_height_status": "each element deformation component taken as a rigid rotation over the full column "
                                                              "height: NOT the measured joint share (that needs the diagonal-gauge kinematics)",
                       "panel_strain_at_pos_peak": r_pos["shearpanel_d"],
                       "bar_strain_max_face": {lab: float(np.max(np.abs([r[f"{lab}_eps_top"] for r in seg_rows]))) for lab in ("bLface", "bRface")}})
    # first TENSION yield of the beam bars at the joint faces: the bar-slip spring force reaching fy x As
    # (positive = tension; the compression branch carries the concrete resultant too and is not a yield check)
    first, tension_ratio = {}, {}
    for name, nbars in (("node2BarSlipB", BSB_NBARS), ("node2BarSlipT", BST_NBARS), ("node4BarSlipB", BSB_NBARS), ("node4BarSlipT", BST_NBARS)):
        fy_force = B_FY * B_AS * nbars
        forces = np.array([r[f"{name}_F"] for r in rows])
        hit = np.where(forces >= 0.99 * fy_force)[0]
        first[name] = {"step": int(hit[0]), "V_kN": float(v[hit[0]]), "u_mm": float(u[hit[0]])} if len(hit) else None
        tension_ratio[name] = {"max_tension_over_fy_As": float(np.max(forces) / fy_force), "fy_As_kN": fy_force / 1000.0,
                               "max_compression_kN": float(np.min(forces) / 1000.0)}
    # first yield of the beam steel in the fibre sections (strain at the first top layer / bottom layer reaching eps_y)
    eps_y = B_FY / B_ES
    fibre_yield = {}
    for lab in ("bLface", "bRface", "bLgrav", "bRgrav"):
        for layer in ("top", "bot"):
            eps = np.array([r[f"{lab}_eps_{layer}"] for r in rows])
            hit = np.where(eps >= eps_y)[0]
            fibre_yield[f"{lab}_{layer}"] = {"step": int(hit[0]), "V_kN": float(v[hit[0]]), "u_mm": float(u[hit[0]])} if len(hit) else None
    strength = {"peak_pos_kN": float(np.max(v)), "peak_neg_kN": float(np.min(v)),
                "observed_max_kN": 80.3, "V1_theoretical_kN": V1_KN, "V2_theoretical_kN": V2_KN,
                "peak_over_observed_max": float(np.max(np.abs(v)) / 80.3)}
    panel = np.array([r["shearpanel_d"] for r in rows])
    solver_codes = np.array([r["solver_code"] for r in rows])
    return {"equilibrium": equilibrium, "cycles": cycles, "first_bar_yield_at_joint_face": first, "beam_bar_slip_spring_forces": tension_ratio,
            "first_fibre_yield": fibre_yield, "strength": strength,
            "panel_strain_max": float(np.max(np.abs(panel))),
            "bar_slip_max_mm": {name: float(np.max(np.abs([r[f"{name}_d"] for r in rows]))) for name in JOINT_RESPONSES if name != "shearpanel"},
            "beam_bar_strain_max": {f"{lab}_{layer}": float(np.max(np.abs([r[f"{lab}_eps_{layer}"] for r in rows])))
                                    for lab in ("bLface", "bRface", "bLgrav", "bRgrav") for layer in ("top", "bot")},
            "total_signed_work_kN_mm": float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(u))),
            "all_finite": bool(all(np.all(np.isfinite([r[k] for r in rows])) for k in rows[0] if k not in ("phase", "solver_code", "solver_tol", "substep", "step_complete"))),
            "equilibrium_rows_over_limit": int(np.sum(np.array([r["equilibrium_residual"] for r in rows]) > SOLVER["equilibrium_limit"])),
            "solver_rows_by_code": {int(c): int(np.sum(solver_codes == c)) for c in np.unique(solver_codes)},
            "substep_rows": int(sum(1 for r in rows if r["substep"]))}


def observed_file():
    v2 = HERE / "unit1_observed_digitized_v2.json"
    return v2 if v2.exists() else HERE / "unit1_observed_digitized.json"


def export(label, rows, meta, ev, out, failed=False):
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with (out / f"{label}_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step"] + keys)
            for k, r in enumerate(rows):
                writer.writerow([k] + [r[key] for key in keys])
        np.savez_compressed(out / f"{label}_history.npz", **{key: np.array([r[key] for r in rows]) for key in keys if key != "phase"},
                            phase=np.array([r["phase"] for r in rows]))
    (out / f"{label}_result.json").write_text(json.dumps({"meta": meta, "evaluation": ev}, indent=1, default=float), encoding="utf-8")
    if failed or not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    observed = json.loads(observed_file().read_text(encoding="utf-8"))
    u = np.array([r["u_tip"] for r in rows]); v = np.array([r["V"] for r in rows]) / 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0][0]
    case = meta.get("case", {})
    ax.plot(u, v, color="black", lw=0.6, label=f"simulated ({case.get('beam_steel', '?')} steel, {case.get('confinement', '?')} confinement)")
    pk = [p for p in observed["peaks"] if not p.get("load_control")]
    ax.errorbar([p["delta_mm"] for p in pk], [p["V_kN"] for p in pk], xerr=observed["digitization"]["uncertainty_mm"],
                yerr=observed["digitization"]["uncertainty_kN"], fmt="o", color="red", ms=4, capsize=2, label="observed peaks, Fig. 16(a)")
    for val, name in ((V1_KN, "V1"), (V2_KN, "V2"), (80.3, "Vmax observed")):
        ax.axhline(val, color="0.6", ls=":", lw=0.8); ax.axhline(-val, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("horizontal deflection at the column tip (mm)"); ax.set_ylabel("horizontal load V (kN)")
    ax.set_title(f"{label}: load-deflection; peak {ev['strength']['peak_pos_kN']:.1f} / {ev['strength']['peak_neg_kN']:.1f} kN", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, color="0.9")
    ax = axes[0][1]
    for name, style in (("node2BarSlipT", "-"), ("node2BarSlipB", "--"), ("node4BarSlipT", "-."), ("node4BarSlipB", ":")):
        ax.plot([r[f"{name}_d"] for r in rows], [r[f"{name}_F"] / 1000.0 for r in rows], lw=0.6, ls=style, label=name)
    ax.set_xlabel("bar slip (mm)"); ax.set_ylabel("spring force (kN)"); ax.set_title("beam bar-slip springs at the joint faces", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, color="0.9")
    ax = axes[1][0]
    for lab, style in (("bLface", "-"), ("bRface", "--")):
        ax.plot([r[f"{lab}_eps_top"] for r in rows], [r[f"{lab}_sig_top"] for r in rows], lw=0.6, ls=style, label=f"{lab} first top layer")
    ax.set_xlabel("bar strain"); ax.set_ylabel("bar stress (MPa)"); ax.set_title("beam steel at the column-face sections", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, lw=0.3, color="0.9")
    ax = axes[1][1]
    ax.plot([r["shearpanel_d"] for r in rows], [r["shearpanel_F"] / 1.0e6 for r in rows], color="black", lw=0.6)
    ax.set_xlabel("panel shear strain (rad)"); ax.set_ylabel("panel moment (kN m)"); ax.set_title("shear panel", fontsize=10); ax.grid(True, lw=0.3, color="0.9")
    fig.suptitle(f"Park and Ruitong Unit 1, OpenSees PR1 example ported to OpenSeesPy: {label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / f"{label}_loops.png", dpi=130); plt.close(fig)


def run(label, out, protocol="example", step=None, mirror=False, interface=None, beam_steel="example", confinement="example",
        amplitude_shift_mm=0.0):
    """Build, load and export one run. A failure anywhere after the build exports the completed states and
    the failure record under ``<label>_FAILED`` and then propagates."""
    t0 = time.time()
    meta = {"label": label, "protocol": protocol, "mirror": mirror, "opensees_version": ops.version(), "solver_requested": dict(SOLVER),
            "loose_tolerance": max(LOOSE_TOLERANCE_FLOOR, SOLVER["tolerance"] * 100.0), "status": "running"}
    try:
        info = build(mirror=mirror, interface=interface, beam_steel=beam_steel, confinement=confinement)
    except Exception as exc:
        meta.update({"status": "failed", "failure": {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                                                     "phase": {"phase": "build", "target": None, "control": None}},
                     "elapsed_sec": time.time() - t0})
        export(label + "_FAILED", [], meta, None, out, failed=True)
        raise
    meta["case"] = info["case"]
    meta["build"] = {k: v for k, v in info.items() if k != "coords"}
    rec = Recorder(mirror)
    rec.record("gravity")
    sign = -1.0 if mirror else 1.0
    try:
        if protocol == "example":
            pmeta = run_example_protocol(rec, step=step, sign=sign)
        elif protocol == "paper":
            pmeta = run_paper_protocol(rec, step=step or 0.01, sign=sign)
        elif protocol == "experiment":
            pmeta = run_experiment_protocol(rec, step=step or 0.01, sign=sign, amplitude_shift_mm=amplitude_shift_mm)
        else:
            raise ValueError(f"unknown protocol {protocol!r}")
    except Exception as exc:
        meta.update({"status": "failed", "failure": _failure_record(rec, exc), "elapsed_sec": time.time() - t0,
                     "solver_retries": rec.retries, "solver_failures": rec.failures, "solver_recoveries_by_kind": rec.recoveries,
                     "steps": len(rec.rows)})
        export(label + "_FAILED", rec.rows, meta, None, out, failed=True)
        print(f"[{label}] FAILED in phase {meta['failure']['phase']}: {exc}; {len(rec.rows)} completed states exported", flush=True)
        raise
    meta.update(pmeta)
    meta.update({"status": "completed", "elapsed_sec": time.time() - t0, "solver": dict(SOLVER),
                 "solver_retries": rec.retries, "solver_failures": rec.failures, "solver_recoveries_by_kind": rec.recoveries,
                 "accepted_attempts_by_code": {code: sum(1 for a in rec.accepted if a["code"] == code) for code in range(4)},
                 "attempt_codes": {v: f"{k[0]} at the {k[1]} tolerance" for k, v in Recorder.ATTEMPT_CODES.items()},
                 "steps": len(rec.rows)})
    ev = evaluate(rec.rows, meta)
    export(label, rec.rows, meta, ev, out)
    print(f"[{label}] steps {len(rec.rows)}, peak V {ev['strength']['peak_pos_kN']:.1f} / {ev['strength']['peak_neg_kN']:.1f} kN "
          f"(observed max 80.3), equilibrium Fx {ev['equilibrium']['sum_fx_relative']:.1e} M {ev['equilibrium']['moment_relative']:.1e}, "
          f"retries {rec.retries}, failures {rec.failures}, {meta['elapsed_sec']:.0f} s", flush=True)
    return rec.rows, meta, ev


def reflection_map(key):
    """For a quantity recorded under key in the original run, the (key, sign) of the same physical quantity in
    the reflected run. Node-tag quantities keep their names (node 3 is the image of node 3), the recorder has
    already sign-corrected displacements, rotations and the panel pair, bar slip stays bar slip. What the
    reflection does swap: the joint element's left/right springs (its node 2 is "right" in both models, so the
    image of the original's node-2 springs are the reflected model's node-4 springs, and the column bars' L and
    R at nodes 1 and 3), the beam elements' end i and end j (their node order was reversed to keep the section
    the right way up), the element's own bar-slip and interface deformation components (invariant, raw) against
    its panel component (flips, raw), and the column curvature (its local y points to -x in both models, so
    the reported sign flips) against the beam curvature (invariant, local y up in both). Section moments and
    bar stresses/strains at the beam sections are invariant (the section is the right way up in both)."""
    joint = {"node1BarSlipL": "node1BarSlipR", "node1BarSlipR": "node1BarSlipL", "node3BarSlipL": "node3BarSlipR", "node3BarSlipR": "node3BarSlipL",
             "node2BarSlipB": "node4BarSlipB", "node2BarSlipT": "node4BarSlipT", "node4BarSlipB": "node2BarSlipB", "node4BarSlipT": "node2BarSlipT"}
    stem, _, suffix = key.rpartition("_")
    if stem in joint:
        return joint[stem] + "_" + suffix, 1.0
    if key.startswith("b") and (key.endswith("_thi") or key.endswith("_thj")):
        return key[:-1] + ("j" if key.endswith("i") else "i"), 1.0
    if key in ("joint_def_panel",):
        return key, -1.0
    if key in ("joint_def_barslip", "joint_def_interface"):
        return key, 1.0
    if key == "joint_def_total":
        return key, None          # mixed conventions inside the element's sum; compared through its components
    if key in ("kap_col_bot", "kap_col_top"):
        return key, -1.0
    if key in ("solver_code", "solver_tol", "substep", "step_complete", "equilibrium_residual"):
        return key, None          # bookkeeping, not a physical quantity
    return key, 1.0


def compare(rows_a, rows_b, keys):
    """Maximum discrepancy per key between the original and the reflected run through reflection_map, on
    the rows that complete a full step, matched by physical phase: two histories are compared only over the
    phases both completed, and a phase mismatch is reported instead of silently truncating."""
    ra = [r for r in rows_a if r.get("step_complete", 1)]
    rb = [r for r in rows_b if r.get("step_complete", 1)]
    pa, pb = [r["phase"] for r in ra], [r["phase"] for r in rb]
    n = min(len(ra), len(rb))
    first_mismatch = next((i for i in range(n) if pa[i] != pb[i]), None)
    matched = first_mismatch is None and len(ra) == len(rb)
    n_compared = n if first_mismatch is None else first_mismatch
    out = {"matching_phases": matched, "rows_original": len(ra), "rows_reflected": len(rb), "rows_compared": n_compared,
           "first_phase_mismatch": None if first_mismatch is None else {"index": first_mismatch, "original": pa[first_mismatch], "reflected": pb[first_mismatch]},
           "relative": {}, "absolute": {}}
    for key in keys:
        key_b, sign = reflection_map(key)
        if sign is None or key == "phase":
            continue
        a = np.array([r[key] for r in ra[:n_compared]], dtype=float); b = sign * np.array([r[key_b] for r in rb[:n_compared]], dtype=float)
        scale = max(float(np.max(np.abs(a))), 1e-12) if len(a) else 1.0
        diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
        out["relative"][key] = diff / scale
        out["absolute"][key] = diff
    out["max"] = max(out["relative"].values()) if out["relative"] else 0.0
    out["global_budget"] = {"V_kN": out["absolute"].get("V", 0.0) / 1000.0, "u_tip_mm": out["absolute"].get("u_tip", 0.0),
                            "limit_kN": 0.3, "limit_mm": 0.3,
                            "passes": bool(out["absolute"].get("V", 0.0) / 1000.0 <= 0.3 and out["absolute"].get("u_tip", 0.0) <= 0.3),
                            "basis": "review choice: 10 % of the digitization resolution (review 2026-09-28)"}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", default="example", choices=("example", "paper", "experiment"))
    parser.add_argument("--case", default="example_inputs", choices=tuple(CASES))
    parser.add_argument("--step", type=float, default=None, help="tip displacement step, mm (example default 0.01)")
    parser.add_argument("--amplitude-shift", type=float, default=0.0, help="experiment protocol: coherent shift of every amplitude, mm")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--checks", action="store_true", help="orientation (mirrored geometry) and step-size checks after the baseline")
    parser.add_argument("--tolerance", type=float, default=None, help="NormDispIncr tolerance (default the example's 1e-8)")
    args = parser.parse_args(argv)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    if args.tolerance is not None:
        SOLVER["tolerance"] = args.tolerance
    case = CASES[args.case]
    label = args.label or f"unit1_{args.protocol}_{args.case}" + ("_mirror" if args.mirror else "") + \
        (f"_shift{args.amplitude_shift:+g}" if args.amplitude_shift else "")
    kw = dict(protocol=args.protocol, step=args.step, beam_steel=case["beam_steel"], confinement=case["confinement"],
              amplitude_shift_mm=args.amplitude_shift)
    rows, meta, ev = run(label, out, mirror=args.mirror, **kw)
    summary = {"baseline": {"label": label, "case": args.case, "meta": meta, "evaluation": ev}, "checks": {}}
    if args.checks:
        rows_m, meta_m, ev_m = run(label + "_mirror", out, mirror=True, **kw)
        keys = [k for k in rows[0] if k != "phase"]
        summary["checks"]["orientation_mirror"] = compare(rows, rows_m, keys)
        summary["checks"]["orientation_mirror"]["basis"] = ("geometry reflected, beam elements kept running toward +x (section top up), joint node order anticlockwise, "
                                                            "loading reversed; every recorded quantity mapped by reflection_map(); rows matched by phase")
        summary["checks"]["orientation_mirror"]["solver"] = {"original": {"retries": meta["solver_retries"], "recoveries": meta["solver_recoveries_by_kind"]},
                                                             "mirror": {"retries": meta_m["solver_retries"], "recoveries": meta_m["solver_recoveries_by_kind"]},
                                                             "steps": meta["steps"], "requested": dict(SOLVER)}
        om = summary["checks"]["orientation_mirror"]
        print(f"[orientation] mirrored geometry: max relative discrepancy {om['max']:.2e}; V {om['global_budget']['V_kN']:.2e} kN, "
              f"u {om['global_budget']['u_tip_mm']:.2e} mm (budget 0.3 / 0.3); phases matched {om['matching_phases']}", flush=True)
        step0 = meta["step_mm"]
        for factor in (2.0, 5.0):
            rows_s, meta_s, ev_s = run(f"{label}_step_x{factor:g}", out, mirror=False, **{**kw, "step": step0 * factor})
            summary["checks"][f"step_x{factor:g}"] = {"step_mm": step0 * factor, "peak_pos_kN": ev_s["strength"]["peak_pos_kN"],
                                                       "peak_neg_kN": ev_s["strength"]["peak_neg_kN"],
                                                       "total_signed_work_kN_mm": ev_s["total_signed_work_kN_mm"],
                                                       "peak_change_kN": abs(ev_s["strength"]["peak_pos_kN"] - ev["strength"]["peak_pos_kN"]),
                                                       "peak_change_relative": abs(ev_s["strength"]["peak_pos_kN"] - ev["strength"]["peak_pos_kN"]) / abs(ev["strength"]["peak_pos_kN"]),
                                                       "work_change_relative": abs(ev_s["total_signed_work_kN_mm"] - ev["total_signed_work_kN_mm"]) / max(abs(ev["total_signed_work_kN_mm"]), 1e-9),
                                                       "within_0_3_kN": bool(abs(ev_s["strength"]["peak_pos_kN"] - ev["strength"]["peak_pos_kN"]) <= 0.3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    ops.wipe()
    return summary


if __name__ == "__main__":
    main()
