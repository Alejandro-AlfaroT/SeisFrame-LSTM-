import math

import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import (apply_analysis_constraints, floor_master_node, installed_node_seismic_mass, node_tag,
                               roof_master_node)


def _massed_nodes():
    """Grid nodes carrying seismic mass, with their translational mass."""
    entries = []
    for k in range(1, sp.NUM_FLOOR + 1):
        for j in range(sp.NUM_BAY_Y + 1):
            for i in range(sp.NUM_BAY_X + 1):
                entries.append((node_tag(k, i, j), installed_node_seismic_mass(k, i, j)))
    return entries


def _modal_participation(mode, massed):
    """Participation factors and effective modal masses for one mode.

    Gamma = sum(m phi) / sum(m phi^2) and M* = (sum(m phi))^2 / sum(m phi^2),
    with the generalized mass taken over both horizontal components so the
    two directions share one normalization. Mass is lumped at the grid nodes,
    so the sums run over those rather than the diaphragm masters.
    """
    numerator_x = numerator_y = generalized = 0.0
    for tag, mass in massed:
        phi_x = ops.nodeEigenvector(tag, mode, 1)
        phi_y = ops.nodeEigenvector(tag, mode, 2)
        numerator_x += mass * phi_x
        numerator_y += mass * phi_y
        generalized += mass * (phi_x * phi_x + phi_y * phi_y)

    if generalized <= 0.0:
        return None

    total_mass = sum(mass for _tag, mass in massed)
    effective_x = numerator_x**2 / generalized
    effective_y = numerator_y**2 / generalized
    return {
        "generalized_mass": generalized,
        "participation_factor_x": numerator_x / generalized,
        "participation_factor_y": numerator_y / generalized,
        "effective_modal_mass_x": effective_x,
        "effective_modal_mass_y": effective_y,
        "modal_mass_ratio_x": effective_x / total_mass if total_mass > 0 else None,
        "modal_mass_ratio_y": effective_y / total_mass if total_mass > 0 else None,
    }


def _floor_mode_shape(mode):
    """Horizontal mode shape at each diaphragm master, floor order."""
    shape = []
    for k in range(1, sp.NUM_FLOOR + 1):
        master = floor_master_node(k)
        shape.append(
            (
                ops.nodeEigenvector(master, mode, 1),
                ops.nodeEigenvector(master, mode, 2),
            )
        )
    return shape


def run_modal_analysis():
    ops.wipeAnalysis()
    apply_analysis_constraints()
    ops.numberer("RCM")
    ops.system("BandGeneral")

    eigenvalues = ops.eigen(sp.NUM_MODES)
    roof_node = roof_master_node()
    massed = _massed_nodes()

    modes = []
    for mode, eigenvalue in enumerate(eigenvalues, start=1):
        if eigenvalue <= sp.EIGENVALUE_TOL:
            modes.append(
                {
                    "mode": mode,
                    "lambda": eigenvalue,
                    "valid": False,
                    "period": None,
                    "omega": None,
                    "frequency": None,
                    "roof_eigenvector": None,
                    "floor_mode_shape": None,
                    "participation": None,
                }
            )
            continue

        omega = math.sqrt(eigenvalue)
        period = 2.0 * math.pi / omega
        frequency = omega / (2.0 * math.pi)

        modes.append(
            {
                "mode": mode,
                "lambda": eigenvalue,
                "valid": True,
                "period": period,
                "omega": omega,
                "frequency": frequency,
                "roof_eigenvector": (
                    ops.nodeEigenvector(roof_node, mode, 1),
                    ops.nodeEigenvector(roof_node, mode, 2),
                    ops.nodeEigenvector(roof_node, mode, 3),
                ),
                # The full shape and participation factors are what let a
                # surrogate reconstruct an elastic modal-superposition
                # response; a single roof value cannot.
                "floor_mode_shape": _floor_mode_shape(mode),
                "participation": _modal_participation(mode, massed),
            }
        )

    return modes
