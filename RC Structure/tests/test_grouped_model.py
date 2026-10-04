"""Both analysis frames, the gravity and mass ledgers and the hinges under a grouped design (2026-10-02).

Acceptance items of the grouped design that concern the model layer:
  A2  the uniform design expanded into its groups builds the same elastic and nonlinear frames, with the
      same masses, weights, periods, gravity state and hinge materials as the uniform code path;
  A3  one deliberately changed column group and one changed beam group reach the elements, the hinges,
      the member weights and the nodal masses: OpenSees queries against totals and properties worked by
      hand, and an elastic frame assembled independently in this file;
  A5  what is not wired for groups is refused: fiber members, joint springs, a uniform value, a stale tag.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_fixture as gf                                                     # noqa: E402

import openseespy.opensees as ops                                               # noqa: E402
import Structure_Parameters as sp                                               # noqa: E402
from Analysis.Gravity import run_gravity_analysis                                # noqa: E402
from Analysis.Modal import run_modal_analysis                                    # noqa: E402
from Design.SMRF_Elastic import build_design_model, gravity_weight_above_story   # noqa: E402
from Loads.Gravity_Loads import apply_gravity_loads                              # noqa: E402
from Loads.Seismic_ELF import apply_elf_loads, seismic_weight_per_floor          # noqa: E402
from Model import Member_Bar_Layers as layers                                    # noqa: E402
from Model import Member_Groups as mg                                            # noqa: E402
from Model import Member_Properties as mp                                        # noqa: E402
from Model.Build_Model import build_model, create_nodes, create_rigid_diaphragms, fix_base_nodes, node_tag  # noqa: E402
from Model.IMK_Calibration import column_moment_at_axial, column_pm_nominal_for  # noqa: E402
from Model.IMK_Hinges import hinge_registry                                      # noqa: E402

N_MEMBERS = len(gf.hand_members())


def grid_nodes():
    return [(k, i, j, node_tag(k, i, j)) for k in range(1, gf.NZ + 1) for j in range(gf.NY + 1) for i in range(gf.NX + 1)]


def base_reaction():
    ops.reactions()
    return sum(ops.nodeReaction(node_tag(0, i, j), 3) for j in range(gf.NY + 1) for i in range(gf.NX + 1))


def snapshot(builder):
    """What a built frame is, as OpenSees reports it: masses, periods, gravity state, lateral response."""
    ops.wipe()
    builder()
    data = {"mass": {tag: ops.nodeMass(tag, 1) for _k, _i, _j, tag in grid_nodes()},
            "periods": [m["period"] for m in run_modal_analysis()]}
    apply_gravity_loads()
    run_gravity_analysis()
    data["gravity_reaction"] = base_reaction()
    data["gravity_forces"] = {tag: list(ops.eleResponse(tag, "localForce")) for tag in range(1, N_MEMBERS + 1)}
    elf = apply_elf_loads("x", model_period_sec=data["periods"][0])
    ops.loadConst("-time", 0.0)
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("lateral step failed")
    data["roof_ux"] = ops.nodeDisp(node_tag(gf.NZ, 0, 0), 1)
    data["base_shear"] = elf["base_shear_kip"]
    data["story_weights"] = list(elf["story_weights_kip"])
    data["registry"] = hinge_registry()
    ops.wipe()
    return data


def close(test, a, b, rel=1e-9, msg=None):
    test.assertTrue(math.isclose(a, b, rel_tol=rel, abs_tol=1e-9), msg or (a, b))


# The eigen solver is not repeatable below about 3e-9 on the SAME model built twice (measured on this
# fixture, 2026-10-02), so periods and what follows from them (the ELF base shear) are compared at 1e-7.
SOLVER = 1e-7


class UniformEquivalence(unittest.TestCase):
    """A2: expanding the uniform design into its groups changes no physics."""

    def compare(self, builder, slab):
        with gf.frame(slab):
            legacy = snapshot(builder)
            legacy_layers = sp.beam_bar_layers()
            mg.install(gf.uniform_state())
            grouped = snapshot(builder)
            grouped_layers = layers.arrangement()
            for tag, mass in legacy["mass"].items():
                close(self, grouped["mass"][tag], mass, 1e-12)
            for a, b in zip(grouped["periods"], legacy["periods"]):
                close(self, a, b, SOLVER)
            close(self, grouped["gravity_reaction"], legacy["gravity_reaction"])
            close(self, grouped["roof_ux"], legacy["roof_ux"], SOLVER)
            close(self, grouped["base_shear"], legacy["base_shear"], SOLVER)
            for a, b in zip(grouped["story_weights"], legacy["story_weights"]):
                close(self, a, b, 1e-12)
            for tag, forces in legacy["gravity_forces"].items():
                for a, b in zip(grouped["gravity_forces"][tag], forces):
                    self.assertTrue(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6), (tag, a, b))
            for gid, rows in grouped_layers.items():
                if gid == "_basis":
                    continue
                axis = "x" if "__beam_x__" in gid else "y"
                for face in ("top", "bottom"):
                    self.assertEqual(rows[face]["per_layer"], legacy_layers[axis][face]["per_layer"])
                    for a, b in zip(rows[face]["offsets_in"], legacy_layers[axis][face]["offsets_in"]):
                        close(self, a, b, 1e-12)
            return legacy, grouped

    def test_elastic_design_frame_is_the_same(self):
        for slab in (None, gf.SLAB):
            self.compare(build_design_model, slab)

    def test_nonlinear_frame_and_every_hinge_are_the_same(self):
        for slab in (None, gf.SLAB):
            legacy, grouped = self.compare(build_model, slab)
            self.assertEqual(set(legacy["registry"]), set(grouped["registry"]))
            self.assertEqual(len(grouped["registry"]), N_MEMBERS)
            keys = ("ke_y_kip_in_per_rad", "ke_z_kip_in_per_rad", "yield_moment_y_kip_in", "yield_moment_y_hogging_i_kip_in",
                    "yield_moment_y_hogging_j_kip_in", "yield_moment_y_sagging_i_kip_in", "yield_moment_y_sagging_j_kip_in",
                    "yield_moment_z_kip_in", "iy_effective_in4", "theta_p", "theta_pc", "theta_u", "axial_kip", "axial_ratio",
                    "theta_y_spring_y", "theta_y_spring_z", "lambda_opensees_rad")
            for tag, entry in legacy["registry"].items():
                other = grouped["registry"][tag]
                for key in keys:
                    if key in entry:
                        close(self, other[key], entry[key], 1e-11, (tag, key, other[key], entry[key]))
                self.assertEqual(other["beam_family"], entry["beam_family"])
                self.assertEqual(other["strength_basis"], entry["strength_basis"])
                self.assertEqual(other["member_type"], entry["member_type"])
                # the installed OpenSees material commands themselves
                for end in ("i", "j"):
                    for axis in ("y", "z"):
                        a, b = other["installed_materials"][end][axis], entry["installed_materials"][end][axis]
                        self.assertEqual(a["command"][0:2], b["command"][0:2])
                        for x, y in zip(a["command"][2:], b["command"][2:]):
                            close(self, x, y, 1e-11, (tag, end, axis))

    def test_ledgers_are_the_same(self):
        with gf.frame():
            uniform = (seismic_weight_per_floor(), [gravity_weight_above_story(k) for k in range(1, gf.NZ + 1)])
            mg.install(gf.uniform_state())
            grouped = (seismic_weight_per_floor(), [gravity_weight_above_story(k) for k in range(1, gf.NZ + 1)])
            for a, b in zip(uniform[0] + uniform[1], grouped[0] + grouped[1]):
                close(self, a, b, 1e-12)


class ChangedGroupPropagation(unittest.TestCase):
    """A3: two changed groups reach every consumer of the model layer."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.addCleanup(self.stack.__exit__, None, None, None)
        mg.install(gf.changed_state())

    def test_elastic_frame_equals_a_frame_assembled_by_hand(self):
        installed = snapshot(build_design_model)

        def by_hand():
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            create_nodes()
            fix_base_nodes()
            ops.geomTransf("PDelta", 1, 1, 0, 0)
            ops.geomTransf("Linear", 2, 0, 0, 1)
            ops.geomTransf("Linear", 3, 0, 0, 1)
            for tag, kind, k, i, j in gf.hand_members():
                if kind == "column":
                    ni, nj, transform = node_tag(k - 1, i, j), node_tag(k, i, j), 1
                elif kind == "beam_x":
                    ni, nj, transform = node_tag(k, i, j), node_tag(k, i + 1, j), 2
                else:
                    ni, nj, transform = node_tag(k, i, j), node_tag(k, i, j + 1), 3
                ops.element("elasticBeamColumn", tag, ni, nj, *gf.hand_elastic_properties(kind, k, i, j), transform)
            create_rigid_diaphragms()
            for k, i, j, tag in grid_nodes():
                m = gf.hand_node_mass(k, i, j)
                ops.mass(tag, m, m, 1.0e-8, 0.0, 0.0, 0.0)

        self.assertEqual((sp.COL_TRANSF_TAG, sp.BEAM_X_TRANSF_TAG, sp.BEAM_Y_TRANSF_TAG), (1, 2, 3))
        reference = snapshot(by_hand)
        for tag, mass in reference["mass"].items():
            close(self, installed["mass"][tag], mass, 1e-11)
        for a, b in zip(installed["periods"], reference["periods"]):
            close(self, a, b, SOLVER)
        close(self, installed["roof_ux"], reference["roof_ux"], SOLVER)
        for tag, forces in reference["gravity_forces"].items():
            for a, b in zip(installed["gravity_forces"][tag], forces):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6), (tag, a, b))
        # and it is a different frame from the uniform one
        mg.install(gf.uniform_state())
        uniform = snapshot(build_design_model)
        self.assertGreater(abs(uniform["periods"][0] - installed["periods"][0]) / uniform["periods"][0], 1e-3)

    def test_weights_masses_and_base_reaction_follow_the_installed_members(self):
        installed = snapshot(build_design_model)
        members = sum(gf.hand_weight_kip(kind, k, i, j) for _tag, kind, k, i, j in gf.hand_members())
        floor_load = ((sp.FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF + sp.CONCRETE_UNIT_WEIGHT_KCF * gf.SLAB / 12.0 + sp.FLOOR_LIVE_LOAD_KSF)
                      * gf.BAY_X * gf.NX * gf.BAY_Y * gf.NY / 144.0) * gf.NZ
        close(self, installed["gravity_reaction"], floor_load + members, 1e-8)
        for k, i, j, tag in grid_nodes():
            close(self, installed["mass"][tag], gf.hand_node_mass(k, i, j), 1e-11)
        for k, weight in enumerate(installed["story_weights"], start=1):
            close(self, weight, sum(gf.hand_node_mass(k, i, j) for j in range(gf.NY + 1) for i in range(gf.NX + 1)) * sp.G, 1e-11)
        # the floors of the two bands no longer weigh the same, and the old uniform floor weight is in neither
        self.assertNotAlmostEqual(installed["story_weights"][0], installed["story_weights"][2], places=2)
        mg.install(gf.uniform_state())
        uniform = snapshot(build_design_model)
        self.assertNotAlmostEqual(uniform["gravity_reaction"], installed["gravity_reaction"], places=2)

    def test_hinges_are_built_from_each_members_own_design(self):
        installed = snapshot(build_model)
        registry = installed["registry"]
        self.assertEqual(len(registry), N_MEMBERS)
        n = sp.IMK_HINGE_STIFFNESS_FACTOR
        for tag, kind, k, i, j in gf.hand_members():
            entry, design = registry[tag], gf.hand_design(kind, k, i, j)
            self.assertEqual(entry["installed_design"], mg.design_record(design), tag)
            self.assertEqual((entry["story_or_floor"], entry["grid_i"], entry["grid_j"]), (k, i, j))
            _a, e, _g, _j, iy, iz = gf.hand_elastic_properties(kind, k, i, j)
            close(self, entry["iy_effective_in4"], iy, 1e-11, (tag, kind))
            length = gf.STORY_H if kind == "column" else gf.BAY_X if kind == "beam_x" else gf.BAY_Y
            close(self, entry["ke_y_kip_in_per_rad"], n * 6.0 * e * iy / length, 1e-11)
            close(self, entry["ke_z_kip_in_per_rad"], n * 6.0 * e * iz / length, 1e-11)
        # one changed interior column of story 3, checked against the published expressions by hand
        tag = next(t for t, kind, k, i, j in gf.hand_members() if (kind, k, i, j) == ("column", 3, 1, 1))
        entry, d = registry[tag], gf.SMALL_COLUMN
        self.assertEqual(entry["group_id"], "s03_04__column__interior")
        axial = sum(gf.hand_node_mass(k, 1, 1) * sp.G for k in (3, 4)) + 2 * sp.FLOOR_LIVE_LOAD_KSF * gf.BAY_X * gf.BAY_Y / 144.0
        close(self, entry["axial_kip"], axial, 1e-11)                 # SEISMIC_LIVE_LOAD_FRACTION is 0: mass weight + live
        ab = sp.rebar_area(d.bar_size)
        cover = sp.COL_CLEAR_COVER_IN + sp.rebar_diameter(d.stirrup_bar_size) + 0.5 * sp.rebar_diameter(d.bar_size)
        rows = [(d.top_bars * ab, cover), (2 * ab, cover + (d.h_in - 2 * cover) / 2.0), (d.bot_bars * ab, d.h_in - cover)]
        my = column_moment_at_axial(axial, column_pm_nominal_for(d.b_in, d.h_in, d.fc_ksi, rows))
        close(self, entry["yield_moment_y_kip_in"], my, 1e-11)
        nu = axial / (d.b_in * d.h_in * d.fc_ksi)
        close(self, entry["axial_ratio"], nu, 1e-11)
        rho_sh = d.stirrup_legs * sp.rebar_area(d.stirrup_bar_size) / (d.b_in * d.stirrup_spacing_in)
        rho = d.top_bars * ab / (d.b_in * (d.h_in - cover))
        s_n = d.stirrup_spacing_in / sp.rebar_diameter(d.bar_size) * math.sqrt(sp.FY_KSI * 6.894757 / 100.0)
        theta_p = (0.12 * 1.55 * 0.16 ** nu * (0.02 + 40.0 * rho_sh) ** 0.43 * 0.54 ** (0.01 * d.fc_ksi * 6.894757)
                   * 0.66 ** (0.1 * s_n) * 2.27 ** (10.0 * rho))
        close(self, entry["theta_p"], theta_p, 1e-11)
        close(self, entry["theta_pc"], min(0.10, 0.76 * 0.031 ** nu * (0.02 + 40.0 * rho_sh) ** 1.02), 1e-11)
        close(self, entry["lambda_haselton_dimensionless"], 170.7 * 0.27 ** nu * 0.10 ** (d.stirrup_spacing_in / d.h_in), 1e-11)
        # an unchanged column of the same story keeps the uniform cage and section
        corner = registry[next(t for t, kind, k, i, j in gf.hand_members() if (kind, k, i, j) == ("column", 3, 0, 0))]
        self.assertEqual(corner["installed_design"], mg.design_record(gf.COLUMN))

    def test_beam_strengths_follow_the_changed_neighbours(self):
        changed = snapshot(build_model)["registry"]
        mg.install(gf.uniform_state())
        uniform = snapshot(build_model)["registry"]
        tags = {(kind, k, i, j): tag for tag, kind, k, i, j in gf.hand_members()}

        def differs(key, field="yield_moment_y_sagging_i_kip_in"):
            return not math.isclose(changed[tags[key]][field], uniform[tags[key]][field], rel_tol=1e-9)

        # the deeper edge Y beams themselves, floors 1-2 only
        self.assertTrue(differs(("beam_y", 1, 0, 0)) and differs(("beam_y", 2, 2, 1)))
        self.assertFalse(differs(("beam_y", 3, 0, 0)))
        self.assertFalse(differs(("beam_y", 1, 1, 0)) and differs(("beam_y", 1, 1, 0), "yield_moment_y_hogging_i_kip_in"))
        # an interior X beam of floor 3 frames into the smaller interior column: longer clear span, wider flange
        beam = changed[tags[("beam_x", 3, 0, 1)]]
        close(self, beam["flange_width_in"], gf.hand_flange_width("beam_x", 3, 0, 1), 1e-12)
        self.assertGreater(beam["flange_width_in"], uniform[tags[("beam_x", 3, 0, 1)]]["flange_width_in"])
        # an edge X beam of floor 3 touches nothing that changed
        for field in ("yield_moment_y_sagging_i_kip_in", "yield_moment_y_hogging_i_kip_in", "iy_effective_in4", "ke_y_kip_in_per_rad"):
            self.assertFalse(differs(("beam_x", 3, 0, 0), field), field)
        # the edge X beam of floor 1 crosses the deeper Y edge beam at the corner: its bars are unchanged
        # because the deeper beam's bottom bars pass below them
        rows = layers.arrangement()
        self.assertEqual(rows["s01_02__beam_x__edge"]["bottom"]["offsets_in"], rows["s03_04__beam_x__edge"]["bottom"]["offsets_in"])


class GroupedRefusals(unittest.TestCase):
    """A5 at the model layer: nothing uniform is installed behind a grouped design."""

    def setUp(self):
        self.stack = gf.frame()
        self.stack.__enter__()
        self.addCleanup(self.stack.__exit__, None, None, None)
        mg.install(gf.changed_state())

    def test_fiber_members_and_joint_springs_are_refused(self):
        sp.IMK_APPLY_TO_BEAMS = False
        with self.assertRaises(mg.GroupedStateError):
            build_model()
        sp.IMK_APPLY_TO_BEAMS, sp.JOINT_MODEL = True, "imk_pinching_scissors"
        with self.assertRaises(mg.GroupedStateError):
            build_model()
        sp.JOINT_MODEL, sp.ELEMENT_FORMULATION = "rigid_centerline", "fiber"
        with self.assertRaises(mg.GroupedStateError):
            build_model()

    def test_a_uniform_floor_transfer_is_not_applied_to_a_grouped_frame(self):
        sp.GRAVITY_LOAD_MODEL = "slab_transfer"
        sp.FLOOR_TRANSFER = {"schema": "smrf_floor_transfer_v3_force_and_couple_export", "load_model": "slab_transfer"}
        build_design_model()
        with self.assertRaises(mg.GroupedStateError):
            apply_gravity_loads()

    def test_a_tag_that_is_not_the_groups_member_is_refused(self):
        from Model.IMK_Hinges import create_imk_member
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        create_nodes()
        ops.geomTransf("Linear", 2, 0, 0, 1)
        with self.assertRaises(mg.GroupedStateError):                # tag 1 is a column in the grouped design
            create_imk_member(1, node_tag(1, 0, 0), node_tag(1, 1, 0), "beam_x", 2)

    def test_the_one_section_helpers_of_the_hinge_code_refuse(self):
        from Model import IMK_Calibration as calibration
        from Model import IMK_Hinges as hinges
        for call in (lambda: calibration.column_pm_nominal(), lambda: calibration.axial_load_ratio(100.0, "column"),
                     lambda: calibration.haselton_theta_p("beam_x", 0.0), lambda: calibration.deterioration_for_member("column", 0.1),
                     lambda: hinges.imk_member_properties("column"), lambda: hinges.imk_member_properties("beam_x")):
            with self.assertRaises(mg.GroupedStateError):
                call()


# These hand totals describe the framed members; the column extension above the roof is verified on its own
# (tests/test_roof_extension.py), so this module runs with the terminating roof column.
def setUpModule():
    global _ROOF_PATCH
    from unittest import mock as _mock
    import Structure_Parameters as _sp
    _ROOF_PATCH = _mock.patch.object(_sp, "ROOF_COLUMN_EXTENSION", False)
    _ROOF_PATCH.start()


def tearDownModule():
    _ROOF_PATCH.stop()


if __name__ == "__main__":
    unittest.main()
