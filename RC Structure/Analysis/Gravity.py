import openseespy.opensees as ops

import Structure_Parameters as sp
from Model.Build_Model import apply_analysis_constraints, node_tag, roof_master_node


def _roof_floor_vertical_stats():
    uz_values = []

    for j in range(sp.NUM_BAY_Y + 1):
        for i in range(sp.NUM_BAY_X + 1):
            node = node_tag(sp.NUM_FLOOR, i, j)
            uz_values.append(ops.nodeDisp(node, 3))

    return {
        "roof_floor_uz_min": min(uz_values),
        "roof_floor_uz_max": max(uz_values),
        "roof_floor_uz_avg": sum(uz_values) / len(uz_values),
    }


def run_gravity_analysis():
    ops.system("BandGeneral")
    apply_analysis_constraints()
    ops.numberer("RCM")
    ops.test("NormDispIncr", sp.GRAVITY_TOL, sp.GRAVITY_MAX_ITER)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / sp.GRAVITY_STEPS)
    ops.analysis("Static")

    ok = 0
    for step in range(sp.GRAVITY_STEPS):
        ok = ops.analyze(1)

        if ok != 0:
            ops.algorithm("NewtonLineSearch")
            ok = ops.analyze(1)
            ops.algorithm("Newton")

        if ok != 0:
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1)
            ops.algorithm("Newton")

        if ok != 0:
            raise RuntimeError(f"Static gravity analysis failed at step {step}.")

    ops.loadConst("-time", 0.0)

    roof_node = roof_master_node()
    results = {
        "ok": ok,
        "roof_node": roof_node,
        "ux": ops.nodeDisp(roof_node, 1),
        "uy": ops.nodeDisp(roof_node, 2),
        "uz": ops.nodeDisp(roof_node, 3),
    }
    results.update(_roof_floor_vertical_stats())
    return results
