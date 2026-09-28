"""Asymmetric beam hinges: hogging/sagging strengths land on the right spring sign.

The IMK beam springs report hogging as positive deformation at end i and
negative at end j (measured here, not assumed). With different hogging and
sagging yield moments, each end's IMKBilin must therefore receive them in
the opposite order. The tests push one end into yield in each direction and
read the moment the spring actually delivers.
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops  # noqa: E402

import Structure_Parameters as sp  # noqa: E402
from Model import IMK_Hinges  # noqa: E402
from Model.IMK_Hinges import create_imk_member, hinge_element_tag, reset_hinge_registry, hinge_registry  # noqa: E402

HOGGING, SAGGING = 2000.0, 1000.0


def _cantilever(member_type, fixed_end, tip_load):
    """One IMK beam, fixed at one end, vertical tip load at the other.

    Returns the dir-5 spring moment at the fixed end. A downward load hogs
    the fixed end; an upward load sags it.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    reset_hinge_registry()
    ops.node(1, 0.0, 0.0, 0.0)
    if member_type == "beam_x":
        ops.node(2, 240.0, 0.0, 0.0)
    else:
        ops.node(2, 0.0, 240.0, 0.0)
    ops.geomTransf("Linear", 9, 0, 0, 1)
    fixed, free = (1, 2) if fixed_end == "i" else (2, 1)
    ops.fix(fixed, 1, 1, 1, 1, 1, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        create_imk_member(1, 1, 2, member_type, 9)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(free, 0.0, 0.0, tip_load, 0.0, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Penalty", sp.PENALTY_ALPHA_SP, sp.PENALTY_ALPHA_MP)
    ops.test("NormDispIncr", 1e-8, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.05)
    ops.analysis("Static")
    for _ in range(20):
        if ops.analyze(1) != 0:
            raise RuntimeError("cantilever step failed")
    end_id = 1 if fixed == 1 else 2
    tag = hinge_element_tag(1, end_id)
    moment = ops.eleResponse(tag, "material", 1, "stress")[0]
    rotation = ops.eleResponse(tag, "material", 1, "strain")[0]
    ops.wipe()
    return moment, rotation


class BeamHingeAsymmetryTests(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch.object(
            IMK_Hinges, "beam_yield_moments",
            return_value=(HOGGING, SAGGING, {"basis": "test", "family": "x_interior"}))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        ops.wipe()

    def test_each_end_yields_at_its_own_hogging_and_sagging_strengths(self):
        # Tip load of 4.375 kip on 240 in demands 1050 kip-in: past the
        # sagging yield (1000, peak 1.1x) but well inside hogging (2000). The
        # moment is fixed by statics either way; yielding shows in the
        # spring rotation, which stays M/Ke when elastic.
        for material_type in ("IMKBilin", "IMKPeakOriented"):
            with mock.patch.object(sp, "IMK_MATERIAL_TYPE", material_type):
                for member_type in ("beam_x", "beam_y"):
                    for fixed_end in ("i", "j"):
                        hog_m, hog_r = _cantilever(member_type, fixed_end, -4.375)
                        sag_m, sag_r = _cantilever(member_type, fixed_end, +4.375)
                        with self.subTest(material=material_type, member=member_type, end=fixed_end):
                            self.assertAlmostEqual(abs(hog_m), 1050.0, delta=1.0)
                            self.assertAlmostEqual(abs(sag_m), 1050.0, delta=1.0)
                            elastic = 1050.0 / IMK_Hinges.imk_hinge_stiffness(member_type, "rot_y", 240.0)
                            self.assertAlmostEqual(abs(hog_r), elastic, delta=0.02 * elastic)
                            self.assertGreater(abs(sag_r), 5.0 * elastic)
                            self.assertEqual(hog_r > 0, fixed_end == "i")

    def test_registry_records_both_strengths_and_the_weaker_yield_rotation(self):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        reset_hinge_registry()
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 240.0, 0.0, 0.0)
        ops.geomTransf("Linear", 9, 0, 0, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            create_imk_member(1, 1, 2, "beam_x", 9)
        entry = hinge_registry()[1]
        self.assertEqual(entry["yield_moment_y_hogging_kip_in"], HOGGING)
        self.assertEqual(entry["yield_moment_y_sagging_kip_in"], SAGGING)
        self.assertEqual(entry["yield_moment_y_kip_in"], HOGGING)
        self.assertAlmostEqual(entry["theta_y_spring_y_hogging"], 2.0 * entry["theta_y_spring_y"])


class BeamYieldMomentSourceTests(unittest.TestCase):
    def test_slab_aware_beam_uses_composite_family_strengths(self):
        from Design.SMRF_Beam_Slab_Strength import composite_beam_strengths
        layout = {"layers": {
            f"{axis}_{face}": {"bar_area_in2": 0.20, "spacing_in": 10.0,
                               "effective_depth_in": 4.0 if axis == "x" else 3.5}
            for axis in ("x", "y") for face in ("top", "bottom")}}
        values = {"NUM_BAY_X": 3, "NUM_BAY_Y": 3, "BAY_X": 120.0, "BAY_Y": 120.0,
                  # corner bars only: the two #7 beam bars pass in one layer (the two-layer rule of
                  # 2026-09-27 would otherwise stack them and shift the x hinge's depth)
                  "COL_TOP_BARS": 2, "COL_SIDE_BARS": 0,
                  "B_COL": 18.0, "H_COL": 18.0, "B_BEAM": 10.0, "H_BEAM": 18.0, "FC_BEAM_KSI": 4.0,
                  "FY_KSI": 60.0, "BEAM_BAR_SIZE": 7, "BEAM_TOP_BARS": 2, "BEAM_BOT_BARS": 2,
                  "BEAM_BAR_AREA": 0.60, "SLAB_THICKNESS_IN": 5.0,
                  "SLAB_REINFORCEMENT": {"layout": layout}}
        with contextlib.ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(mock.patch.object(sp, name, value, create=True))
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0.0, 120.0, 0.0)      # interior x line (j = 1)
            ops.node(2, 120.0, 120.0, 0.0)
            ops.node(3, 0.0, 0.0, 0.0)        # perimeter x line (j = 0)
            ops.node(4, 120.0, 0.0, 0.0)
            hog_int, sag_int, basis_int = IMK_Hinges.beam_yield_moments("beam_x", 1, 2)
            hog_edge, sag_edge, basis_edge = IMK_Hinges.beam_yield_moments("beam_x", 3, 4)
            # The x direction's own top/bottom offsets: the orthogonal layers stack at the joints
            # (2026-09-27), and the hinge prices the family with this direction's depths.
            beam = {"b_in": 10.0, "h_in": 18.0, "fc_ksi": 4.0, "fy_ksi": 60.0, "bar_size": 7,
                    "top_bars": 2, "bot_bars": 2, "centroid_offset_in": sp.beam_bar_stacking_offsets_in()["x"]}
            geometry = {"bay_x_in": 120.0, "bay_y_in": 120.0, "h_col_in": 18.0, "b_col_in": 18.0}
            expected = composite_beam_strengths(beam, {"thickness_in": 5.0}, layout, geometry, "x", "interior")
            ops.wipe()
        self.assertEqual(basis_int["family"], "x_interior")
        self.assertEqual(basis_edge["family"], "x_edge")
        self.assertAlmostEqual(hog_int, expected["mn_negative_kip_in"])
        self.assertAlmostEqual(sag_int, expected["mn_positive_kip_in"])
        self.assertGreater(hog_int, hog_edge)          # wider flange, more slab steel
        # The slab makes hogging much stronger (1.6x at one elevation); x is the lower direction under
        # the stacking convention here, so its top bars sit a layer lower and the ratio is about 1.4x.
        self.assertGreater(hog_int, 1.3 * sag_int)


if __name__ == "__main__":
    unittest.main()
