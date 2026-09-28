"""
Design/Export_SAP2000.py
========================

Emit a SAP2000 text-import (.$2k) model of a designed structure, for
independent verification of the design model in a second solver.

Why
---
Several qualification items in Design/Config.IndependentVerification attest
that a person checked things the design code cannot check about itself: that
the floor load transfer is right, that the strength model is right, that the
slab/frame idealization is acceptable. Hand calculation is not realistic for
a 3D multi-story frame with a shell floor transfer. A second finite-element
program modelling the same structure from the same record is the practical
form of independent verification, and SAP2000 also runs its own ACI 318-19
concrete design, which is a direct check on ACI_Checks.py.

Two variants are written, because the OpenSees design model is two things:

  frame   Frame + rigid diaphragm, gravity applied as the exact per-member
          loads the shell floor transfer produced (stored in the record).
          This is a comparison model, not a validated solver-equivalent copy:
          see the exported limitations, particularly P-Delta and mass mapping.

  slab    Frame + shell slab + area pressures. SAP performs its own floor
          transfer. Column axials and beam moments against the record's
          transfer totals check floor.independent_hand_verification; a
          frame-only vs with-slab comparison speaks to
          floor.compatibility_idealization_reviewed.

Labels match OpenSees exactly: joints are node_tag(k, i, j), frames are the
element tags in create_elements order, so results can be compared by id.

Units: Kip, in, F throughout, matching the OpenSees model. SAP honours the
CurrUnits declared in PROGRAM CONTROL on import.

Usage
-----
    python Design/Export_SAP2000.py <case_dir or design.json> --output-dir <dir>
    python Design/Export_SAP2000.py ... --variant frame|slab|both   (default both)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

RC_DIR = Path(os.environ.get("RC_STRUCTURE_DIR", Path(__file__).resolve().parents[1])).resolve()
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import Structure_Parameters as sp  # noqa: E402
from Geometry_Overrides import apply_geometry_overrides  # noqa: E402
from Loads.Seismic_ELF import elf_story_forces  # noqa: E402
from Design.SMRF_Floor_Transfer import global_couple  # noqa: E402

SAP_VERSION = "26.3.0"
G_IN_PER_SEC2 = 386.4
CONCRETE_UNIT_WEIGHT_KCI = 0.150 / 1728.0   # kip/in^3
STEEL_UNIT_WEIGHT_KCI = 0.490 / 1728.0

REBAR = {
    3: (0.11, 0.375), 4: (0.20, 0.500), 5: (0.31, 0.625), 6: (0.44, 0.750),
    7: (0.60, 0.875), 8: (0.79, 1.000), 9: (1.00, 1.128), 10: (1.27, 1.270),
    11: (1.56, 1.410), 14: (2.25, 1.693), 18: (4.00, 2.257),
}


# ---------------------------------------------------------------------------
# Record access
# ---------------------------------------------------------------------------

def load_record(path):
    path = Path(path)
    if path.is_dir():
        path = path / "design.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def node_tag(nx, ny, k, i, j):
    """Identical to Model.Build_Model.node_tag so joint labels match OpenSees."""
    return k * ((nx + 1) * (ny + 1)) + j * (nx + 1) + i + 1


def _beam_stiffness_ratios(record, frame):
    """I(T/L on the 6.3.2 flange) / I(rectangular web) per beam line, from the record's own geometry."""
    import Structure_Parameters as sp
    g, s = record["geometry"], record["sections"]
    b, h = float(s["b_beam_in"]), float(s["h_beam_in"])
    t = record.get("slab", {}).get("thickness_in")
    rect = sp.rect_iy(b, h)
    if t is None:
        return {(axis, position): 1.0 for axis in ("x", "y") for position in ("edge", "interior")}
    ratios = {}
    for axis in ("x", "y"):
        if axis == "x":
            clear_span, clear_web = float(g["bay_x_in"]) - float(s["h_col_in"]), float(g["bay_y_in"]) - b
        else:
            clear_span, clear_web = float(g["bay_y_in"]) - float(s["b_col_in"]), float(g["bay_x_in"]) - b
        for position, sides in (("edge", 1), ("interior", 2)):
            bf, _ = sp.effective_flange_width_in(b, float(t), clear_span, clear_web, sides)
            ratios[(axis, position)] = sp.t_section_inertia_in4(b, h, float(t), bf) / rect
    return ratios


class Frame:
    """Geometry and numbering, mirroring Model/Build_Model.py exactly."""

    def __init__(self, record):
        g = record["geometry"]
        self.nx, self.ny, self.nf = int(g["num_bay_x"]), int(g["num_bay_y"]), int(g["num_floor"])
        self.bx, self.by, self.sh = float(g["bay_x_in"]), float(g["bay_y_in"]), float(g["story_h_in"])
        self.columns, self.beams_x, self.beams_y = [], [], []
        tag = 1
        for k in range(self.nf):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    self.columns.append((tag, self.node(k, i, j), self.node(k + 1, i, j), k + 1, i, j))
                    tag += 1
        for k in range(1, self.nf + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    self.beams_x.append((tag, self.node(k, i, j), self.node(k, i + 1, j), k, i, j))
                    tag += 1
        for k in range(1, self.nf + 1):
            for j in range(self.ny):
                for i in range(self.nx + 1):
                    self.beams_y.append((tag, self.node(k, i, j), self.node(k, i, j + 1), k, i, j))
                    tag += 1

    def node(self, k, i, j):
        return node_tag(self.nx, self.ny, k, i, j)

    def coords(self, k, i, j):
        return i * self.bx, j * self.by, k * self.sh

    def floor_nodes(self, k):
        return [self.node(k, i, j) for j in range(self.ny + 1) for i in range(self.nx + 1)]

    def tributary_area_in2(self, i, j):
        tx = self.bx / 2.0 if i in (0, self.nx) else self.bx
        ty = self.by / 2.0 if j in (0, self.ny) else self.by
        return tx * ty

    @property
    def plan_x(self):
        return self.nx * self.bx

    @property
    def plan_y(self):
        return self.ny * self.by


def concrete_e_ksi(fc_ksi):
    return 57.0 * math.sqrt(fc_ksi * 1000.0)


def rect_props(b, h):
    """t2 = width b, t3 = depth h; I33 is bending about local 3 (strong for a beam)."""
    area = b * h
    i33 = b * h**3 / 12.0
    i22 = h * b**3 / 12.0
    ratio = min(b, h) / max(b, h)
    beta = (1.0 / 3.0) * (1.0 - 0.63 * ratio * (1.0 - ratio**4 / 12.0))
    torsion = beta * max(b, h) * min(b, h) ** 3
    return dict(Area=area, I33=i33, I22=i22, J=torsion, AS2=area * 5.0 / 6.0, AS3=area * 5.0 / 6.0,
                S33=i33 / (h / 2.0), S22=i22 / (b / 2.0), Z33=b * h**2 / 4.0, Z22=h * b**2 / 4.0,
                R33=math.sqrt(i33 / area), R22=math.sqrt(i22 / area))


# ---------------------------------------------------------------------------
# $2k writer
# ---------------------------------------------------------------------------

class S2K:
    def __init__(self):
        self.tables = []

    def table(self, name, rows):
        self.tables.append((name, rows))

    @staticmethod
    def fmt(value):
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return repr(float(value)) if abs(value) < 1e15 else f"{value:.6g}"
        text = str(value)
        return f'"{text}' + '"' if (" " in text or "/" in text or "," in text) else text

    def render(self):
        out = ["File written by StructGNN Design/Export_SAP2000.py", ""]
        for name, rows in self.tables:
            out.append(f'TABLE:  "{name}"')
            for row in rows:
                out.append("   " + "   ".join(f"{key}={self.fmt(value)}" for key, value in row.items()))
            out.append("")
        out.append("END TABLE DATA")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def build(record, variant):
    if variant not in ("frame", "slab"):
        raise ValueError("variant must be frame or slab")
    frame = Frame(record)
    sec = record["sections"]
    reinf = record["reinforcement"]
    mat = record.get("materials") or {}
    seismic = record.get("seismic") or {}
    floor = record.get("floor_loads") or {}
    slab = record.get("slab") or {}
    torsion = (record.get("demand_basis") or {}).get("torsion") or {}
    transfer = record.get("floor_transfer") or {}
    demand = record.get("demand") or {}
    drift_assumptions = {**((record.get("drift_screen") or {}).get("assumptions") or {}),
                         **((record.get("demand_basis") or {}).get("drift") or {})}

    fy = float(mat.get("fy_ksi", 60.0))
    fc_col, fc_beam = float(sec["fc_col_ksi"]), float(sec["fc_beam_ksi"])
    fc_slab = float(slab.get("concrete_fc_ksi") or fc_beam)
    b_col, h_col = float(sec["b_col_in"]), float(sec["h_col_in"])
    b_beam, h_beam = float(sec["b_beam_in"]), float(sec["h_beam_in"])
    col_mod = float(drift_assumptions["column_stiffness_modifier"])
    beam_mod = float(drift_assumptions["beam_stiffness_modifier"])
    combinations = (record.get("design_actions") or {}).get("combinations") or []
    if not combinations:
        raise ValueError("Saved design load combinations are required; no guessed combinations are exported.")
    patterns = {p["id"]: p for p in record["demand_basis"]["live_load_patterns"]}
    live_patterns = {"all": "LIVE", **{pid: "LIVE_" + pid for pid in patterns}}
    for combo in combinations:
        if combo.get("live_pattern", "all") not in live_patterns:
            raise ValueError("Saved combination refers to a missing live-load pattern.")

    # --- ELF story forces, recomputed on the record's period and hazard -----
    apply_geometry_overrides({"NUM_BAY_X": frame.nx, "NUM_BAY_Y": frame.ny, "NUM_FLOOR": frame.nf,
                              "BAY_X": frame.bx, "BAY_Y": frame.by, "STORY_H": frame.sh}, emit=False)
    if seismic.get("site_label"):
        sp.apply_seismic_site(seismic["site_label"])
    for key, value in (record.get("request_identity") or {}).get("inputs", {}).items():
        if hasattr(sp, key):
            setattr(sp, key, value)
    # The seismic weight depends on the sections (member self-weight) and on
    # the slab state the design selected. Mirror Design_Driver exactly, or the
    # ELF is computed on the legacy load model and comes out ~2% high.
    sp.B_COL, sp.H_COL, sp.B_BEAM, sp.H_BEAM = b_col, h_col, b_beam, h_beam
    sp.SLAB_THICKNESS_IN = None
    if floor.get("slab_thickness_in") is not None:
        sp.SLAB_THICKNESS_IN = float(floor["slab_thickness_in"])
        sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF = float(floor.get("floor_superimposed_dead_load_ksf",
                                                              sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF))
        sp.SEISMIC_LIVE_LOAD_FRACTION = float(floor.get("seismic_live_load_fraction",
                                                        sp.SEISMIC_LIVE_LOAD_FRACTION))
    elf = elf_story_forces(demand.get("model_period_sec"))
    record_weight = float(floor.get("total_floor_seismic_weight_kip") or 0.0) * frame.nf
    if record_weight and abs(elf["seismic_weight_kip"] - record_weight) > 1e-3 * record_weight:
        raise RuntimeError(
            f"Seismic weight mismatch: exporter {elf['seismic_weight_kip']:.2f} kip vs record "
            f"{record_weight:.2f} kip. The load model was not reproduced; refusing to emit "
            "an ELF that differs from the one the design was checked against.")
    ecc_ratio = float(torsion.get("ratio", 0.05)) * float(torsion.get("amplification", 1.0))
    # Keep section stiffness intact, but account for concrete only once.
    gamma = float(slab["concrete_unit_weight_kcf"]) / 1728.0
    weight_mod = {
        "column": sp.col_self_weight_kip_per_in() / (gamma * b_col * h_col),
        "x": sp.beam_self_weight_kip_per_in("x") / (gamma * b_beam * h_beam),
        "y": sp.beam_self_weight_kip_per_in("y") / (gamma * b_beam * h_beam),
    }

    s = S2K()
    s.table("PROGRAM CONTROL", [dict(
        ProgramName="SAP2000", Version=SAP_VERSION, CurrUnits="Kip, in, F",
        SteelCode="AISC 360-16", ConcCode="ACI 318-19", AlumCode="AA 2015",
        ColdCode="AISI-16", ConcSCode="Eurocode 2-2004", RegenHinge="Yes")])
    s.table("ACTIVE DEGREES OF FREEDOM", [dict(UX=True, UY=True, UZ=True, RX=True, RY=True, RZ=True)])

    # --- materials -----------------------------------------------------------
    materials = [("CONC_COL", fc_col), ("CONC_BEAM", fc_beam)]
    if variant == "slab":
        materials.append(("CONC_SLAB", fc_slab))
    s.table("MATERIAL PROPERTIES 01 - GENERAL",
            [dict(Material=name, Type="Concrete", SymType="Isotropic", TempDepend=False, Color="Blue")
             for name, _ in materials]
            + [dict(Material="A706Gr60", Type="Rebar", SymType="Uniaxial", TempDepend=False, Color="Gray8Dark")])
    s.table("MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES",
            [dict(Material=name, UnitWeight=gamma,
                  UnitMass=gamma / G_IN_PER_SEC2,
                  E1=concrete_e_ksi(fc), G12=0.4 * concrete_e_ksi(fc), U12=0.2, A1=5.5e-6)
             for name, fc in materials]
            + [dict(Material="A706Gr60", UnitWeight=STEEL_UNIT_WEIGHT_KCI,
                    UnitMass=STEEL_UNIT_WEIGHT_KCI / G_IN_PER_SEC2, E1=29000.0, G12=11153.8, U12=0.3, A1=6.5e-6)])
    s.table("MATERIAL PROPERTIES 03B - CONCRETE DATA",
            [dict(Material=name, Fc=fc, eFc=fc, LtWtConc=False, SSCurveOpt="Mander", SSHysType="Takeda",
                  SFc=0.00221914, SCap=0.005, FinalSlope=-0.1, FAngle=0, DAngle=0)
             for name, fc in materials])
    s.table("MATERIAL PROPERTIES 03E - REBAR DATA",
            [dict(Material="A706Gr60", Fy=fy, Fu=1.25 * fy + 5.0, eFy=1.1 * fy, eFu=1.25 * fy + 15.0,
                  SSCurveOpt="Simple", SSHysType="Kinematic", SHard=0.01, SCap=0.09, FinalSlope=-0.1, UseCTDef=False)])
    s.table("REBAR SIZES", [dict(RebarID=f"#{size}", Area=area, Diameter=dia)
                            for size, (area, dia) in sorted(REBAR.items())])

    # --- frame sections ------------------------------------------------------
    col_props, beam_props = rect_props(b_col, h_col), rect_props(b_beam, h_beam)

    def section_row(name, material, b, h, props):
        return dict(SectionName=name, Material=material, Shape="Rectangular", t3=h, t2=b,
                    Area=props["Area"], TorsConst=props["J"], I33=props["I33"], I22=props["I22"], I23=0.0,
                    AS2=props["AS2"], AS3=props["AS3"], S33Top=props["S33"], S33Bot=props["S33"],
                    S22Left=props["S22"], S22Right=props["S22"], Z33=props["Z33"], Z22=props["Z22"],
                    R33=props["R33"], R22=props["R22"], ConcCol=(name == "COL"), ConcBeam=(name == "BEAM"),
                    Color="Green", TotalWt=0.0, TotalMass=0.0, FromFile=False, AMod=1, A2Mod=1, A3Mod=1,
                    JMod=1, I2Mod=1, I3Mod=1, MMod=1, WMod=1)

    s.table("FRAME SECTION PROPERTIES 01 - GENERAL",
            [section_row("COL", "CONC_COL", b_col, h_col, col_props),
             section_row("BEAM", "CONC_BEAM", b_beam, h_beam, beam_props)])

    # Column cage: top/bottom bars along the faces parallel to local 3, side
    # bars per face plus the shared corners along the faces parallel to local 2.
    bars_3 = int(reinf["col_top_bars"])
    bars_2 = int(reinf["col_side_bars"]) + 2
    # Tie legs per direction: those crossing the b faces run along local 2
    # (parallel to t3 = h), those crossing the h faces along local 3.
    legs_by_direction = reinf.get("col_stirrup_legs_by_direction") or {
        "across_b_face": reinf["col_stirrup_legs"], "across_h_face": reinf["col_stirrup_legs"]}
    s.table("FRAME SECTION PROPERTIES 05 - CONCRETE COLUMN", [dict(
        SectionName="COL", RebarMatL="A706Gr60", RebarMatC="A706Gr60", ReinfConfig="Rectangular",
        LatReinf="Ties", Cover=float(reinf.get("col_clear_cover_in", 1.5)),
        NumBars3Dir=bars_3, NumBars2Dir=bars_2, BarSizeL=f"#{int(reinf['col_bar_size'])}",
        BarSizeC=f"#{int(reinf['col_stirrup_bar_size'])}", SpacingC=float(reinf["col_stirrup_spacing_in"]),
        NumCBars2=int(legs_by_direction["across_b_face"]), NumCBars3=int(legs_by_direction["across_h_face"]),
        ReinfType="Check")])
    beam_top = int(reinf["beam_top_bars"]) * REBAR[int(reinf["beam_bar_size"])][0]
    beam_bot = int(reinf["beam_bot_bars"]) * REBAR[int(reinf["beam_bar_size"])][0]
    cover_beam = float(reinf.get("beam_longitudinal_centroid_offset_in", 2.25))
    s.table("FRAME SECTION PROPERTIES 06 - CONCRETE BEAM", [dict(
        SectionName="BEAM", RebarMatL="A706Gr60", RebarMatC="A706Gr60",
        TopCover=cover_beam, BotCover=cover_beam,
        TopLeftArea=beam_top, TopRghtArea=beam_top, BotLeftArea=beam_bot, BotRghtArea=beam_bot)])

    # --- joints, frames, restraints, diaphragms -----------------------------
    joints = []
    for k in range(frame.nf + 1):
        for j in range(frame.ny + 1):
            for i in range(frame.nx + 1):
                x, y, z = frame.coords(k, i, j)
                joints.append(dict(Joint=frame.node(k, i, j), CoordSys="GLOBAL", CoordType="Cartesian",
                                   XorR=x, Y=y, Z=z, SpecialJt=False))
    s.table("JOINT COORDINATES", joints)

    connectivity = [dict(Frame=tag, JointI=ni, JointJ=nj, IsCurved=False)
                    for tag, ni, nj, *_ in frame.columns + frame.beams_x + frame.beams_y]
    s.table("CONNECTIVITY - FRAME", connectivity)
    s.table("JOINT RESTRAINT ASSIGNMENTS",
            [dict(Joint=frame.node(0, i, j), U1=True, U2=True, U3=True, R1=True, R2=True, R3=True)
             for j in range(frame.ny + 1) for i in range(frame.nx + 1)])
    s.table("CONSTRAINT DEFINITIONS - DIAPHRAGM",
            [dict(Name=f"DIAPH{k}", CoordSys="GLOBAL", Axis="Z", MultiLevel=False)
             for k in range(1, frame.nf + 1)])
    s.table("JOINT CONSTRAINT ASSIGNMENTS",
            [dict(Joint=n, Constraint=f"DIAPH{k}") for k in range(1, frame.nf + 1) for n in frame.floor_nodes(k)])
    s.table("FRAME SECTION ASSIGNMENTS",
            [dict(Frame=tag, AutoSelect="N.A.", AnalSect="COL", MatProp="Default") for tag, *_ in frame.columns]
            + [dict(Frame=tag, AutoSelect="N.A.", AnalSect="BEAM", MatProp="Default")
               for tag, *_ in frame.beams_x + frame.beams_y])
    # Beams stay a rectangular SAP section (the web is what SAP designs); their
    # vertical-bending stiffness is the design frame's T/L section on the
    # 6.3.2 flange, carried as an I33 modifier per line so SAP analyses the
    # same stiffness the record was designed with.
    t_over_rect = _beam_stiffness_ratios(record, frame)

    def beam_i33(axis, i, j):
        edge = (j in (0, frame.ny)) if axis == "x" else (i in (0, frame.nx))
        return beam_mod * t_over_rect[(axis, "edge" if edge else "interior")]

    s.table("FRAME PROPERTY MODIFIERS",
            [dict(Frame=tag, AMod=1, AS2Mod=1, AS3Mod=1, JMod=col_mod, I22Mod=col_mod, I33Mod=col_mod, MMod=1, WMod=weight_mod["column"])
             for tag, *_ in frame.columns]
            + [dict(Frame=tag, AMod=1, AS2Mod=1, AS3Mod=1, JMod=beam_mod, I22Mod=beam_mod, I33Mod=beam_i33(axis, i, j),
                    MMod=1, WMod=weight_mod[axis])
               for axis, beams in (("x", frame.beams_x), ("y", frame.beams_y)) for tag, _ni, _nj, _k, i, j in beams])
    s.table("FRAME DESIGN PROCEDURES",
            [dict(Frame=tag, DesignProc="From Material") for tag, *_ in frame.columns + frame.beams_x + frame.beams_y])

    # --- slab variant: shell areas -------------------------------------------
    if variant == "slab":
        t_slab = float(slab.get("thickness_in") or floor.get("slab_thickness_in") or 5.0)
        s.table("AREA SECTION PROPERTIES", [dict(
            Section="SLAB", Material="CONC_SLAB", MatAngle=0, AreaType="Shell", Type="Shell-Thin",
            DrillDOF=True, Thickness=t_slab, BendThick=t_slab, Color="Cyan")])
        areas, area_assign, mesh = [], [], []
        aid = 1
        for k in range(1, frame.nf + 1):
            for j in range(frame.ny):
                for i in range(frame.nx):
                    areas.append(dict(Area=aid, NumJoints=4, Joint1=frame.node(k, i, j), Joint2=frame.node(k, i + 1, j),
                                      Joint3=frame.node(k, i + 1, j + 1), Joint4=frame.node(k, i, j + 1)))
                    area_assign.append(dict(Area=aid, Section="SLAB", MatProp="Default"))
                    n = int(transfer["mesh_per_bay"])
                    if n < 1 or n != transfer["mesh_per_bay"]:
                        raise ValueError("mesh_per_bay must be a positive integer subdivision count")
                    mesh.append(dict(Area=aid, MeshOption="Mesh N x N", N1=n, N2=n, RestraintsOnEdge=False,
                                     RestraintsOnFace=False, LocalAxesOnEdge=False, LocalAxesOnFace=False, SubMesh=False))
                    aid += 1
        s.table("CONNECTIVITY - AREA", areas)
        s.table("AREA SECTION ASSIGNMENTS", area_assign)
        s.table("AREA AUTO MESH ASSIGNMENTS", mesh)

    # --- load patterns -------------------------------------------------------
    s.table("LOAD PATTERN DEFINITIONS", [
        dict(LoadPat="DEAD_FLOOR", DesignType="Dead", SelfWtMult=0),
        dict(LoadPat="SELF_WT", DesignType="Dead", SelfWtMult=1),
        dict(LoadPat="LIVE", DesignType="Live", SelfWtMult=0),
        dict(LoadPat="EQX", DesignType="Quake", SelfWtMult=0),
        dict(LoadPat="EQY", DesignType="Quake", SelfWtMult=0),
    ] + [dict(LoadPat=name, DesignType="Live", SelfWtMult=0)
         for pid, name in live_patterns.items() if pid != "all"])

    joint_loads, frame_point_loads, area_loads = [], [], []

    if variant == "frame":
        # Exact per-member loads from the record's shell floor transfer, per floor.
        beams_by_key = {}
        for tag, ni, nj, k, i, j in frame.beams_x:
            beams_by_key[("x", k, j, i)] = tag
        for tag, ni, nj, k, i, j in frame.beams_y:
            beams_by_key[("y", k, i, j)] = tag
        unit_patterns = [("DEAD_FLOOR", "dead"), ("LIVE", "live")]
        unit_patterns += [(live_patterns[pid], "live_pattern_" + pid) for pid in patterns]
        for pattern, case_key in unit_patterns:
            unit = transfer["unit_cases"].get(case_key)
            if not unit:
                raise ValueError(f"Missing floor-transfer case {case_key}")
            for k in range(1, frame.nf + 1):
                for beam in unit.get("beams", []):
                    axis, line, span = beam["axis"], int(beam["line_index"]), int(beam["span_index"])
                    tag = beams_by_key.get((axis, k, line, span))
                    if tag is None:
                        continue
                    for rel, force in beam.get("node_loads", []):
                        frame_point_loads.append(dict(Frame=tag, LoadPat=pattern, CoordSys="GLOBAL", Type="Force",
                                                      Dir="Gravity", DistType="RelDist", RelDist=float(rel),
                                                      AbsDist=0, Force=float(force)))
                    for rel, local_x, local_y in beam.get("node_couples", []):
                        mx, my = global_couple(axis, local_x, local_y)
                        if abs(mx) > 1e-12:
                            frame_point_loads.append(dict(Frame=tag, LoadPat=pattern, CoordSys="GLOBAL", Type="Moment",
                                                          Dir="X", DistType="RelDist", RelDist=float(rel), AbsDist=0,
                                                          Force=float(mx)))
                        if abs(my) > 1e-12:
                            frame_point_loads.append(dict(Frame=tag, LoadPat=pattern, CoordSys="GLOBAL", Type="Moment",
                                                          Dir="Y", DistType="RelDist", RelDist=float(rel), AbsDist=0,
                                                          Force=float(my)))
                for col in unit.get("columns", []):
                    n = frame.node(k, int(col["grid_i"]), int(col["grid_j"]))
                    joint_loads.append(dict(Joint=n, LoadPat=pattern, CoordSys="GLOBAL", F1=0, F2=0,
                                            F3=-float(col.get("direct_load_kip", 0.0)),
                                            M1=float(col.get("couple_global_mx_kip_in", 0.0)),
                                            M2=float(col.get("couple_global_my_kip_in", 0.0)), M3=0))
    else:
        sdl = float(floor.get("floor_superimposed_dead_load_ksf", 0.05)) / 144.0
        live = float(floor.get("floor_live_load_ksf", 0.05)) / 144.0
        for aid in range(1, frame.nf * frame.nx * frame.ny + 1):
            area_loads.append(dict(Area=aid, LoadPat="DEAD_FLOOR", CoordSys="GLOBAL", Dir="Gravity", UnifLoad=sdl))
            area_loads.append(dict(Area=aid, LoadPat="LIVE", CoordSys="GLOBAL", Dir="Gravity", UnifLoad=live))
            panel = (aid - 1) % (frame.nx * frame.ny)
            ij = [panel % frame.nx, panel // frame.nx]
            for pid, pattern in patterns.items():
                if ij in pattern["panels"]:
                    area_loads.append(dict(Area=aid, LoadPat=live_patterns[pid], CoordSys="GLOBAL",
                                           Dir="Gravity", UnifLoad=live))

    # ELF: mass-proportional distribution to every floor joint (resultant at the
    # centre of mass under a rigid diaphragm), plus one torsional couple for the
    # accidental eccentricity ASCE 7-22 12.8.4.2.
    for k, force in enumerate(elf["story_forces_kip"], start=1):
        nodes = frame.floor_nodes(k)
        weights = {n: frame.tributary_area_in2(i, j) for n, (i, j) in
                   zip(nodes, [(i, j) for j in range(frame.ny + 1) for i in range(frame.nx + 1)])}
        total = sum(weights.values())
        for n, w in weights.items():
            share = force * w / total
            joint_loads.append(dict(Joint=n, LoadPat="EQX", CoordSys="GLOBAL", F1=share, F2=0, F3=0, M1=0, M2=0, M3=0))
            joint_loads.append(dict(Joint=n, LoadPat="EQY", CoordSys="GLOBAL", F1=0, F2=share, F3=0, M1=0, M2=0, M3=0))
        anchor = nodes[0]
        joint_loads.append(dict(Joint=anchor, LoadPat="EQX", CoordSys="GLOBAL", F1=0, F2=0, F3=0, M1=0, M2=0,
                                M3=force * ecc_ratio * frame.plan_y))
        joint_loads.append(dict(Joint=anchor, LoadPat="EQY", CoordSys="GLOBAL", F1=0, F2=0, F3=0, M1=0, M2=0,
                                M3=force * ecc_ratio * frame.plan_x))

    s.table("JOINT LOADS - FORCE", joint_loads)
    if frame_point_loads:
        s.table("FRAME LOADS - POINT", frame_point_loads)
    if area_loads:
        s.table("AREA LOADS - UNIFORM", area_loads)

    # --- cases, combinations, mass -----------------------------------------
    static = ["DEAD_FLOOR", "SELF_WT", "LIVE", "EQX", "EQY"] + [live_patterns[pid] for pid in patterns]
    s.table("LOAD CASE DEFINITIONS",
            [dict(Case=c, Type="LinStatic", InitialCond="Zero", DesTypeOpt="Prog Det",
                  DesignType=("Live" if c.startswith("LIVE") else
                              {"DEAD_FLOOR": "Dead", "SELF_WT": "Dead"}.get(c, "Quake")),
                  DesActOpt="Prog Det", DesignAct="Non-Composite", AutoType="None", RunCase=True) for c in static]
            + [dict(Case="MODAL", Type="LinModal", InitialCond="Zero", DesTypeOpt="Prog Det", DesignType="Other",
                    DesActOpt="Prog Det", DesignAct="Other", AutoType="None", RunCase=True)])
    s.table("CASE - STATIC 1 - LOAD ASSIGNMENTS",
            [dict(Case=c, LoadType="Load pattern", LoadName=c, LoadSF=1) for c in static])
    s.table("CASE - MODAL 1 - GENERAL", [dict(Case="MODAL", ModeType="Eigen", MaxNumModes=min(12, 3 * frame.nf),
                                              MinNumModes=1, EigenShift=0, EigenCutoff=0, EigenTol=1e-9,
                                              AutoShift=True)])
    live_seismic = float(floor.get("seismic_live_load_fraction", 0.0))
    mass_rows = [dict(MassSource="MSSSRC1", Elements=False, Masses=False, Loads=True, IsDefault=True)]
    s.table("MASS SOURCE", mass_rows)
    s.table("MASS SOURCE LOADS", [dict(MassSource="MSSSRC1", LoadPat="DEAD_FLOOR", Multiplier=1.0),
                                  dict(MassSource="MSSSRC1", LoadPat="SELF_WT", Multiplier=1.0)]
            + ([dict(MassSource="MSSSRC1", LoadPat="LIVE", Multiplier=live_seismic)] if live_seismic > 0 else []))

    rho_strength = 1.3 if str(drift_assumptions.get("seismic_design_category", "D")) in ("D", "E", "F") else 1.0
    def combo(name, terms):
        return [dict(ComboName=name, CaseType="Linear Static", CaseName=case, ScaleFactor=sf) for case, sf in terms]

    combo_cases = []
    combo_cases += combo("DRIFT_X", [("DEAD_FLOOR", 1), ("SELF_WT", 1), ("LIVE", 1), ("EQX", 1)])
    combo_cases += combo("DRIFT_Y", [("DEAD_FLOOR", 1), ("SELF_WT", 1), ("LIVE", 1), ("EQY", 1)])
    for saved_combo in combinations:
        terms = [("DEAD_FLOOR", saved_combo["dead"]), ("SELF_WT", saved_combo["dead"]),
                 (live_patterns[saved_combo.get("live_pattern", "all")], saved_combo["live"]),
                 ("EQX", saved_combo["ex"]), ("EQY", saved_combo["ey"])]
        combo_cases += combo(saved_combo["id"], [(name, sf) for name, sf in terms if sf != 0])
    for row in combo_cases:
        row.update(ComboType="Linear Additive", AutoDesign=False)
    s.table("COMBINATION DEFINITIONS", combo_cases)

    # --- concrete design set-up --------------------------------------------
    sdc = str(drift_assumptions.get("seismic_design_category", "D"))
    s.table("PREFERENCES - CONCRETE DESIGN - ACI 318-19", [dict(
        THDesign="Envelopes", NumCurves=24, NumPoints=11, MinEccen=True, BCCDesign=True, IgnoreBPu=True,
        CTorsion=True, PatLLF=0.75, UFLimit=0.95, SeisCat=sdc, Rho=rho_strength,
        Sds=float(seismic.get("sds", sp.ASCE_SDS)), PhiT=0.9, PhiCTied=0.65, PhiCSpiral=0.75, PhiV=0.75,
        PhiVSeismic=0.6, PhiVJoint=0.85, TanTheta=1)])
    s.table("OVERWRITES - CONCRETE DESIGN - ACI 318-19",
            [dict(Frame=tag, FrameType="Sway Special") for tag, *_ in frame.columns + frame.beams_x + frame.beams_y])

    # --- comparison targets --------------------------------------------------
    stories = (record.get("drift_screen") or {}).get("stories") or []
    targets = {
        "case": record.get("request_identity", {}).get("case_id") or Path(".").name,
        "schema_version": record.get("schema_version"),
        "variant": variant,
        "validation_status": "export-only; live SAP import and selected design combinations require verification",
        "required_strength_combinations": [c["id"] for c in combinations],
        "member_weight_modifiers": weight_mod,
        "geometry": {"num_bay_x": frame.nx, "num_bay_y": frame.ny, "num_floor": frame.nf,
                     "bay_x_in": frame.bx, "bay_y_in": frame.by, "story_h_in": frame.sh},
        "period_T1_sec": demand.get("model_period_sec"),
        "elf": {"base_shear_kip": elf["base_shear_kip"], "story_forces_kip": elf["story_forces_kip"],
                "design_period_sec": elf["design_period_sec"], "cs": elf["cs"],
                "seismic_weight_kip": elf["seismic_weight_kip"],
                "record_base_shear_kip": demand.get("base_shear_kip")},
        "torsion": {"eccentricity_ratio": ecc_ratio, "moment_x_case_kip_in": [f * ecc_ratio * frame.plan_y for f in elf["story_forces_kip"]],
                    "moment_y_case_kip_in": [f * ecc_ratio * frame.plan_x for f in elf["story_forces_kip"]]},
        "drift_screen": {"assumptions": drift_assumptions, "cd": drift_assumptions.get("cd", 5.5),
                         "ie": float(sp.ASCE_IE), "stories": stories},
        "dcr": record.get("dcr"),
        "scwb": record.get("scwb"),
        "floor_transfer_totals": {key: {k2: v for k2, v in (unit or {}).items() if k2 in ("applied_kip", "beam_kip", "column_direct_kip", "column_direct_fraction")}
                                  for key, unit in ((transfer.get("unit_cases") or {}).items()) if key in ("dead", "live")},
        "floor_loads": floor,
        "coupled_comparison": {k: v for k, v in (record.get("coupled_comparison") or {}).items()
                               if k in ("base_moment_ratio_frame_over_coupled", "max_column_vertical_relative_difference",
                                        "frame_total_vertical_kip", "coupled_total_vertical_kip")},
        "stiffness_modifiers": {"column": col_mod, "beam": beam_mod},
        "labels": {"joint": "Model.Build_Model.node_tag(k, i, j)", "frame": "OpenSees element tag, columns then beam_x then beam_y",
                   "diaphragm": "DIAPH<k>, one per elevated floor"},
        "known_differences": [
            "SAP linear static has no P-Delta unless you enable it; the OpenSees drift screen includes it (theta ~0.02 -> ~2% drift).",
            "Member WMod reproduces clear-span beam drops and story-minus-slab column weights; confirm SAP imports these modifiers.",
            "SAP load-derived mass distribution must be checked against OpenSees joint masses; matching total weight alone does not validate modal mass distribution.",
            "Select the listed required_strength_combinations for SAP concrete design; do not substitute automatic combinations.",
            "SAP beam required steel area is not a beam DCR; no direct utilization comparison is made without a true ratio field.",
            "ELF is recomputed here from the record's period and hazard; SAP's own auto-seismic case is an additional check, not the same forces.",
        ],
    }
    return s, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Case directory or design.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", choices=("frame", "slab", "both"), default="both")
    args = parser.parse_args()

    record, path = load_record(args.record)
    case = path.parent.name if path.name == "design.json" else path.stem
    out = Path(args.output_dir) if args.output_dir else path.parent / "sap2000"
    out.mkdir(parents=True, exist_ok=True)

    for variant in (("frame", "slab") if args.variant == "both" else (args.variant,)):
        model, targets = build(record, variant)
        targets["case"] = case
        s2k_path = out / f"{case}_{variant}.$2k"
        s2k_path.write_text(model.render(), encoding="utf-8")
        (out / f"{case}_{variant}_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
        print(f"wrote {s2k_path}  ({len(model.tables)} tables)")
        print(f"      ELF V = {targets['elf']['base_shear_kip']:.2f} kip vs record {targets['elf']['record_base_shear_kip']}")


if __name__ == "__main__":
    main()
