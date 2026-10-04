"""Roof column extension: the one place its length, weight and mass are computed (2026-10-04).

Every column line is carried one column depth above the roof joint with the bars and hoops of the column
below (ACI 318-19 15.2.6; ``Structure_Parameters.ROOF_COLUMN_EXTENSION``). The joint classification credits
that extension, so the same extension is carried as weight and mass wherever the frame's own weight is:

  * gravity: a vertical load at each roof node, with the member self-weight factor (Loads/Gravity_Loads);
  * seismic mass: added to the roof node of its column line (Model/Build_Model);
  * seismic weight: added to the roof level (Loads/Seismic_ELF);
  * column gravity axial estimate and the story stability weight.

It is a short free cantilever with nothing attached to it, so it is represented by its weight and mass at
the roof joint and not by an element: it has no stiffness role and carries no demand of its own. The per-floor
weight functions of Structure_Parameters and Model/Member_Properties describe the framed members only and are
unchanged and the uniform-mode consumers add this module's value at the roof. Under a grouped design the
member ledger (Model/Member_Properties) is floor by floor already and carries the extension in its roof
values, so nothing is added to it a second time. In the legacy bundled load model the seismic mass omits
member weight altogether, and so it omits the extension as well.
"""
import Structure_Parameters as sp
from Model import Member_Groups as mg


def _roof_column_section(i, j):
    if mg.is_grouped():
        design = mg.column_at(sp.NUM_FLOOR, i, j).design
        return design.b_in, design.h_in
    return sp.B_COL, sp.H_COL


def length_in(i=0, j=0):
    """Extension above the roof joint of column line (i, j), in; 0 when the extension is not declared."""
    return sp.roof_column_extension_in(*_roof_column_section(i, j))


def weight_kip(i, j):
    """Weight of the extension of column line (i, j), kip."""
    return sp.roof_column_extension_weight_kip(*_roof_column_section(i, j))


def total_weight_kip():
    return sum(weight_kip(i, j) for j in range(sp.NUM_BAY_Y + 1) for i in range(sp.NUM_BAY_X + 1))


def in_seismic_mass():
    """Member weight is part of the seismic mass only in the slab-aware load model."""
    return sp.SLAB_THICKNESS_IN is not None


def node_seismic_mass(floor, i, j):
    """Mass the extension adds at grid node (i, j) of ``floor`` (kip s2/in): the roof node only."""
    if floor != sp.NUM_FLOOR or not in_seismic_mass():
        return 0.0
    return weight_kip(i, j) / sp.G


def seismic_weight_kip():
    """Seismic weight the extensions add to the roof level, kip; equal to the mass they add times g."""
    return total_weight_kip() if in_seismic_mass() else 0.0


def declaration():
    """What is represented and how, for the design record and the exported model description."""
    return {"declared": bool(getattr(sp, "ROOF_COLUMN_EXTENSION", False)),
            "length_rule": "one column depth (the larger section dimension) above the roof joint",
            "length_in_by_column_line": sorted({round(length_in(i, j), 6) for j in range(sp.NUM_BAY_Y + 1)
                                                for i in range(sp.NUM_BAY_X + 1)}),
            "total_weight_kip": total_weight_kip(),
            "seismic_weight_kip": seismic_weight_kip(),
            "reinforcement": "longitudinal bars and hoops of the roof-story column continued through the extension",
            "representation": ("weight as a vertical load and mass at the roof node of each column line; no element, since a "
                               "free stub has no stiffness role"),
            "in_seismic_mass": in_seismic_mass()}
