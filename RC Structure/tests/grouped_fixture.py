"""Shared fixture for the grouped-design tests: a small frame with explicit uniform and grouped designs.

Not a screening case and not a designed structure: sections and cages are written down, so every
number a test compares can be worked by hand. 2 x 3 bays, four stories (bands 1-2 and 3-4), unequal X
and Y spans, a 7 in slab without an established reinforcement layout unless a test supplies one.
"""
import contextlib
import math
import sys
from dataclasses import replace
from pathlib import Path

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import openseespy.opensees as ops                                              # noqa: E402
import Structure_Parameters as sp                                              # noqa: E402
from Model import Member_Groups as mg                                           # noqa: E402

NX, NY, NZ = 2, 3, 4
BAY_X, BAY_Y, STORY_H, SLAB = 240.0, 216.0, 168.0, 7.0
COLUMN = mg.MemberDesign("column", 26.0, 26.0, 5.0, 9, 4, 4, 2, 4, 4, 4.0, (4, 4))
BEAM = mg.MemberDesign("beam_x", 18.0, 28.0, 4.0, 8, 4, 3, 0, 4, 2, 5.0)
# The two deliberately changed groups of the perturbation tests.
SMALL_COLUMN = replace(COLUMN, b_in=22.0, h_in=22.0, bar_size=8, top_bars=3, bot_bars=3, side_bars=1, stirrup_spacing_in=5.0)
DEEP_BEAM = replace(BEAM, member_type="beam_y", b_in=24.0, h_in=32.0, top_bars=5, bot_bars=4)
CHANGED = {"s03_04__column__interior": SMALL_COLUMN, "s01_02__beam_y__edge": DEEP_BEAM}


def set_uniform(column=COLUMN, beam=BEAM):
    """Write one column and one beam design into Structure_Parameters (the uniform state)."""
    sp.B_COL, sp.H_COL, sp.FC_COL_KSI = column.b_in, column.h_in, column.fc_ksi
    sp.COL_BAR_SIZE, sp.COL_TOP_BARS, sp.COL_BOT_BARS, sp.COL_SIDE_BARS = column.bar_size, column.top_bars, column.bot_bars, column.side_bars
    sp.COL_BAR_AREA = sp.rebar_area(column.bar_size)
    sp.COL_STIRRUP_BAR_SIZE, sp.COL_STIRRUP_LEGS, sp.COL_STIRRUP_SPACING = column.stirrup_bar_size, column.stirrup_legs, column.stirrup_spacing_in
    sp.COL_STIRRUP_LEGS_BY_DIRECTION = column.legs_by_direction
    sp.B_BEAM, sp.H_BEAM, sp.FC_BEAM_KSI = beam.b_in, beam.h_in, beam.fc_ksi
    sp.BEAM_BAR_SIZE, sp.BEAM_TOP_BARS, sp.BEAM_BOT_BARS, sp.BEAM_SIDE_BARS = beam.bar_size, beam.top_bars, beam.bot_bars, beam.side_bars
    sp.BEAM_BAR_AREA = sp.rebar_area(beam.bar_size)
    sp.BEAM_STIRRUP_BAR_SIZE, sp.BEAM_STIRRUP_LEGS, sp.BEAM_STIRRUP_SPACING = beam.stirrup_bar_size, beam.stirrup_legs, beam.stirrup_spacing_in


@contextlib.contextmanager
def frame(slab=SLAB, column=COLUMN, beam=BEAM, **overrides):
    """The fixture frame in the uniform state; everything is restored (and any grouped design cleared) on exit."""
    mg.clear()
    snapshot = dict(vars(sp))
    try:
        sp.NUM_BAY_X, sp.NUM_BAY_Y, sp.NUM_FLOOR = NX, NY, NZ
        sp.BAY_X, sp.BAY_Y, sp.STORY_H = BAY_X, BAY_Y, STORY_H
        sp.NUM_MODES = 6
        sp.SLAB_THICKNESS_IN = slab
        sp.SLAB_REINFORCEMENT, sp.FLOOR_TRANSFER, sp.SLAB_ACTIONS = None, None, None
        sp.GRAVITY_LOAD_MODEL = "nodal"
        sp.ELEMENT_FORMULATION, sp.IMK_APPLY_TO_COLUMNS, sp.IMK_APPLY_TO_BEAMS = "imk", True, True
        sp.IMK_MATERIAL_TYPE, sp.JOINT_MODEL = "IMKPeakOriented", "rigid_centerline"
        sp.IMK_ENERGY_MAPPING_MODE = "legacy_unmapped"
        set_uniform(column, beam)
        for key, value in overrides.items():
            setattr(sp, key, value)
        yield
    finally:
        ops.wipe()
        mg.clear()
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


def uniform_state(column=COLUMN, beam=BEAM):
    """The uniform design expanded into its groups (the equivalence fixture)."""
    return mg.GroupedDesign.uniform(NX, NY, NZ, column, beam)


def changed_state():
    """The expanded uniform design with the two changed groups."""
    return uniform_state().with_designs(CHANGED)


# ---- the same designs worked by hand, without the resolver ---------------------------------------------
def hand_design(kind, k, i, j):
    """(b, h, fc) and the design of a member of the CHANGED state from its position, written out by hand."""
    if kind == "column":
        interior = 0 < i < NX and 0 < j < NY
        return SMALL_COLUMN if (interior and k in (3, 4)) else COLUMN
    if kind == "beam_y" and i in (0, NX) and k in (1, 2):
        return DEEP_BEAM
    return replace(BEAM, member_type=kind)


def hand_core(k, i, j):
    """(along X, along Y) of the joint core at floor k: the column below."""
    design = hand_design("column", k, i, j)
    return design.h_in, design.b_in


def hand_members():
    """(tag, kind, k, i, j) in the builders' order."""
    tag, rows = 0, []
    for k in range(1, NZ + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                tag += 1
                rows.append((tag, "column", k, i, j))
    for k in range(1, NZ + 1):
        for j in range(NY + 1):
            for i in range(NX):
                tag += 1
                rows.append((tag, "beam_x", k, i, j))
    for k in range(1, NZ + 1):
        for j in range(NY):
            for i in range(NX + 1):
                tag += 1
                rows.append((tag, "beam_y", k, i, j))
    return rows


def hand_clear_span(kind, k, i, j):
    if kind == "beam_x":
        return BAY_X - 0.5 * hand_core(k, i, j)[0] - 0.5 * hand_core(k, i + 1, j)[0]
    return BAY_Y - 0.5 * hand_core(k, i, j)[1] - 0.5 * hand_core(k, i, j + 1)[1]


def hand_flange_width(kind, k, i, j):
    """ACI 318-19 Table 6.3.2.1 flange of a beam of the CHANGED state, per slab side."""
    design = hand_design(kind, k, i, j)
    clear_span = hand_clear_span(kind, k, i, j)
    if kind == "beam_x":
        neighbours = [(i, line) for line in (j - 1, j + 1) if 0 <= line <= NY]
        transverse = BAY_Y
    else:
        neighbours = [(line, j) for line in (i - 1, i + 1) if 0 <= line <= NX]
        transverse = BAY_X
    one_sided = len(neighbours) == 1
    width = design.b_in
    for ni, nj in neighbours:
        other = hand_design(kind, k, ni, nj)
        clear_web = transverse - 0.5 * design.b_in - 0.5 * other.b_in
        width += min((6.0 if one_sided else 8.0) * SLAB, clear_web / 2.0, clear_span / (12.0 if one_sided else 8.0))
    return width


def hand_weight_kip(kind, k, i, j):
    """Accounted self weight of one member of the CHANGED state (slab-aware convention)."""
    design = hand_design(kind, k, i, j)
    gamma = sp.CONCRETE_UNIT_WEIGHT_KCF / 1728.0
    if kind == "column":
        return gamma * design.b_in * design.h_in * (STORY_H - SLAB)
    return gamma * design.b_in * (design.h_in - SLAB) * hand_clear_span(kind, k, i, j)


def hand_node_mass(k, i, j):
    tx = BAY_X / 2.0 if i in (0, NX) else BAY_X
    ty = BAY_Y / 2.0 if j in (0, NY) else BAY_Y
    weight = (sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF + sp.CONCRETE_UNIT_WEIGHT_KCF * SLAB / 12.0
              + sp.SEISMIC_LIVE_LOAD_FRACTION * sp.FLOOR_LIVE_LOAD_KSF) * tx * ty / 144.0
    weight += hand_weight_kip("column", k, i, j)
    for di in (-1, 0):
        if 0 <= i + di < NX:
            weight += 0.5 * hand_weight_kip("beam_x", k, i + di, j)
    for dj in (-1, 0):
        if 0 <= j + dj < NY:
            weight += 0.5 * hand_weight_kip("beam_y", k, i, j + dj)
    return weight / sp.G


def hand_elastic_properties(kind, k, i, j):
    """(A, E, G, J, Iy, Iz) of the elastic design frame element, modifiers applied, by hand."""
    design = hand_design(kind, k, i, j)
    b, h = design.b_in, design.h_in
    e = 57.0 * math.sqrt(design.fc_ksi * 1000.0)
    modifier = sp.COLUMN_STIFFNESS_MODIFIER if kind == "column" else sp.BEAM_STIFFNESS_MODIFIER
    iy_rect, iz = b * h ** 3 / 12.0, h * b ** 3 / 12.0
    if kind == "column":
        iy = iy_rect
    else:
        bf = hand_flange_width(kind, k, i, j)
        # T-section about its centroid: web b x h, flange overhangs (bf - b) x SLAB at the top.
        a_web, a_fl = b * h, (bf - b) * SLAB
        y_bar = (a_web * h / 2.0 + a_fl * SLAB / 2.0) / (a_web + a_fl)
        iy = (b * h ** 3 / 12.0 + a_web * (h / 2.0 - y_bar) ** 2
              + (bf - b) * SLAB ** 3 / 12.0 + a_fl * (SLAB / 2.0 - y_bar) ** 2)
    return b * h, e, 0.4 * e, modifier * (iy_rect + iz), modifier * iy, modifier * iz
