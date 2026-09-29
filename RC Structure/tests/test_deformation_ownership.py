"""Per-end ownership of bar slip (Model/Deformation_Ownership, 2026-09-27 evening).

Pins the policy Codex's slip resolution asked for: the member hinge keeps Haselton's slip share
(a_sl = 1) at every end, column bases included, unless a face interface is registered for that
end; a declared joint scope is not an owner; duplicate and unowned claims are refused; the
ownership travels into the hinge registry and the NTHA end inventory.
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RC_DIR = Path(__file__).resolve().parents[1]
if str(RC_DIR) not in sys.path:
    sys.path.insert(0, str(RC_DIR))

import openseespy.opensees as ops  # noqa: E402
import Structure_Parameters as sp  # noqa: E402
from Model import Build_Model, Deformation_Ownership as own, IMK_Calibration  # noqa: E402
from Model.IMK_Hinges import hinge_registry  # noqa: E402
from Model.Joint_Springs import joint_registry  # noqa: E402

SMALL = dict(NUM_BAY_X=1, NUM_BAY_Y=1, NUM_FLOOR=2, ELEMENT_FORMULATION="imk", JOINT_MODEL="imk_pinching_scissors",
             JOINT_DEFORMATION_SCOPE="joint_shear_only")


def end(tag, member_type, which, location, floor_index=None):
    return own.member_end(tag, member_type, which, location=location, floor_index=floor_index)


class Policy(unittest.TestCase):
    def setUp(self):
        own.reset_slip_interfaces()

    def tearDown(self):
        own.reset_slip_interfaces()

    def test_locations_from_floor_index(self):
        with mock.patch.object(sp, "NUM_FLOOR", 3):
            self.assertEqual(own.end_location("column", 0), "column_base")
            self.assertEqual(own.end_location("column", 1), "floor_joint")
            self.assertEqual(own.end_location("beam_x", 3), "roof_joint")
            # only a column end at the base level is a base; a fixture beam at z = 0 is a floor joint
            self.assertEqual(own.end_location("beam_x", 0), "floor_joint")
            self.assertEqual(own.end_location("column", 4), "roof_joint")

    def test_member_hinge_owns_slip_everywhere_without_interfaces(self):
        for location in ("column_base", "floor_joint", "roof_joint"):
            e = end(7, "column", "i", location)
            self.assertEqual(own.slip_owner(e), "member_hinge")
            self.assertEqual(own.bond_slip_indicator_for_end(e), 1.0)
            record = own.ownership(e)
            self.assertEqual((record["bar_slip"], record["flexure"]), ("member_hinge", "member_hinge"))
            self.assertEqual(record["member_deformation_scope"], "flexure_and_slip")
            self.assertEqual(record["decomposition_status"], "legacy_included_slip_scope")
            self.assertFalse(record["interface_present"])
        # the base has no joint spring; elevated ends do under the scissors model
        with mock.patch.object(sp, "JOINT_MODEL", "imk_pinching_scissors"):
            self.assertIsNone(own.ownership(end(7, "column", "i", "column_base"))["panel_shear"])
            self.assertEqual(own.ownership(end(7, "column", "j", "floor_joint"))["panel_shear"], "joint_spring")
        self.assertEqual(IMK_Calibration.bond_slip_indicator(), 1.0)

    def test_declared_joint_scope_is_not_an_owner(self):
        for scope in ("joint_shear_only", "joint_shear_and_slip"):
            with mock.patch.object(sp, "JOINT_DEFORMATION_SCOPE", scope):
                self.assertEqual(own.bond_slip_indicator_for_end(end(1, "beam_x", "i", "floor_joint")), 1.0)
        with mock.patch.object(sp, "JOINT_DEFORMATION_SCOPE", "joint_shear_and_slip"):
            with self.assertRaisesRegex(ValueError, "retired"):
                own.validate_joint_scope()
        with mock.patch.object(sp, "JOINT_DEFORMATION_SCOPE", "joint_shear_only"):
            self.assertEqual(own.validate_joint_scope(), "joint_shear_only")

    def test_registered_interface_takes_slip_from_that_end_only(self):
        own.register_slip_interface(3, "j", calibration_id="provisional_bond_law", status="provisional_diagnostic_only")
        taken, kept = end(3, "beam_x", "j", "floor_joint"), end(3, "beam_x", "i", "floor_joint")
        self.assertEqual(own.bond_slip_indicator_for_end(taken), 0.0)
        self.assertEqual(own.bond_slip_indicator_for_end(kept), 1.0)
        record = own.ownership(taken)
        self.assertEqual(record["bar_slip"], "face_interface")
        self.assertEqual(record["member_deformation_scope"], "flexure_only")
        self.assertEqual(record["decomposition_status"], "diagnostic_unresolved")
        self.assertEqual(record["interface_calibration_id"], "provisional_bond_law")
        own.register_slip_interface(4, "i", calibration_id="reviewed", status="calibrated")
        self.assertEqual(own.ownership(end(4, "column", "i", "column_base"))["decomposition_status"],
                         "calibrated_separate_interface")
        # the plastic rotation follows the indicator: 1.55x between the two ends of the same member
        with mock.patch.object(sp, "JOINT_MODEL", "imk_pinching_scissors"):
            with_slip = IMK_Calibration.haselton_theta_p("beam_x", 0.0, kept)
            without = IMK_Calibration.haselton_theta_p("beam_x", 0.0, taken)
        if with_slip > IMK_Calibration.THETA_P_FLOOR and without > IMK_Calibration.THETA_P_FLOOR:
            self.assertAlmostEqual(without / with_slip, 1.0 / 1.55, places=9)

    def test_duplicate_unowned_and_per_axis_claims_are_refused(self):
        own.register_slip_interface(5, "i", calibration_id="x", status="provisional_diagnostic_only")
        with self.assertRaisesRegex(ValueError, "duplicate slip interface"):
            own.register_slip_interface(5, "i", calibration_id="y", status="provisional_diagnostic_only")
        with self.assertRaisesRegex(ValueError, "duplicate slip scope"):
            own.bond_slip_indicator_for_end(end(5, "beam_x", "i", "floor_joint"), member_scope="flexure_and_slip")
        with self.assertRaisesRegex(ValueError, "unowned slip"):
            own.bond_slip_indicator_for_end(end(5, "beam_x", "j", "floor_joint"), member_scope="flexure_only")
        with self.assertRaises(NotImplementedError):
            own.register_slip_interface(6, "i", axis="rot_y", calibration_id="z", status="provisional_diagnostic_only")
        with self.assertRaises(ValueError):
            own.register_slip_interface(6, "i", calibration_id="z", status="approved")
        # the legacy 2026-09-27 state (slip removed, nobody owns it) is reproducible only as a flagged override
        legacy = end(8, "column", "i", "column_base")
        self.assertEqual(own.bond_slip_indicator_for_end(legacy, member_scope=own.LEGACY_UNOWNED_OVERRIDE), 0.0)
        self.assertEqual(own.ownership(legacy, member_scope=own.LEGACY_UNOWNED_OVERRIDE)["decomposition_status"],
                         "slip_removed_without_owner_legacy_error")
        with self.assertRaisesRegex(ValueError, "cannot coexist"):
            own.bond_slip_indicator_for_end(end(5, "beam_x", "i", "floor_joint"), member_scope=own.LEGACY_UNOWNED_OVERRIDE)


class FrameInventory(unittest.TestCase):
    def setUp(self):
        own.reset_slip_interfaces()
        ops.wipe()

    def tearDown(self):
        ops.wipe()
        own.reset_slip_interfaces()

    def test_every_hinge_end_is_classified_and_keeps_its_slip_share(self):
        with mock.patch.multiple(sp, **SMALL):
            Build_Model.build_model()
            registry = hinge_registry()
            joints = joint_registry()
            rows = own.end_inventory(registry)
        self.assertEqual(len(rows), 2 * len(registry))
        locations = {(r["member_type"], r["location"]) for r in rows}
        self.assertIn(("column", "column_base"), locations)
        self.assertIn(("column", "floor_joint"), locations)
        self.assertIn(("column", "roof_joint"), locations)
        self.assertIn(("beam_x", "floor_joint"), locations)
        self.assertNotIn(("beam_x", "column_base"), locations)
        for r in rows:
            self.assertEqual(r["bar_slip"], "member_hinge")
            self.assertEqual(r["bond_slip_indicator"], 1.0)
            self.assertEqual(r["decomposition_status"], "legacy_included_slip_scope")
            self.assertEqual(r["panel_shear"], None if r["location"] == "column_base" else "joint_spring")
            self.assertEqual(r["policy"], own.POLICY_ID)
        for member in registry.values():
            self.assertFalse(member["ends_differ"])
            for e in ("i", "j"):
                provenance = member["installed_materials"][e]["y"]["provenance"]
                self.assertEqual(provenance["bond_slip_indicator"], 1.0)
                self.assertEqual(provenance["slip_owner"], "member_hinge")
                self.assertIn("legacy included scope", provenance["deformation_scope"])
        self.assertTrue(joints)
        # base hinges: first-story columns, end i at floor 0
        bases = [r for r in rows if r["location"] == "column_base"]
        self.assertEqual(len(bases), (SMALL["NUM_BAY_X"] + 1) * (SMALL["NUM_BAY_Y"] + 1))
        self.assertTrue(all(r["end"] == "i" for r in bases))

    def test_retired_scope_is_refused_at_build(self):
        with mock.patch.multiple(sp, **{**SMALL, "JOINT_DEFORMATION_SCOPE": "joint_shear_and_slip"}):
            with self.assertRaisesRegex(ValueError, "retired"):
                Build_Model.build_model()

    def test_inventory_table_is_written_with_the_ntha_outputs(self):
        from Ground_Motion_Main import _deformation_ownership_rows, _write_csv
        with mock.patch.multiple(sp, **SMALL):
            Build_Model.build_model()
            rows = _deformation_ownership_rows()
        self.assertTrue(rows)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "deformation_ownership.csv"
            _write_csv(path, list(rows[0].keys()), rows)
            with path.open(encoding="utf-8") as handle:
                read = list(csv.DictReader(handle))
        self.assertEqual(len(read), len(rows))
        self.assertEqual(set(read[0]), set(own.INVENTORY_COLUMNS))


class DomainLifecycle(unittest.TestCase):
    """Codex Unit 1 review (2026-09-28), commit blocker 1: a registration from a previous domain must not
    survive build_model/ops.wipe, and a registration is honoured only while its element is installed."""

    def setUp(self):
        own.reset_slip_interfaces()
        ops.wipe()

    def tearDown(self):
        ops.wipe()
        own.reset_slip_interfaces()

    def test_consecutive_builds_do_not_inherit_registrations(self):
        with mock.patch.multiple(sp, **SMALL):
            Build_Model.build_model()
            rows = own.end_inventory(hinge_registry())
            before = {(r["member_tag"], r["end"]): r["theta_p"] for r in rows}
            base = next(r for r in rows if r["location"] == "column_base")
            elevated = next(r for r in rows if r["location"] == "floor_joint" and r["member_type"] == "column")
            # stale declarations against a base end and an elevated end; the next build installs no replacement
            for r in (base, elevated):
                own.register_slip_interface(r["member_tag"], r["end"], calibration_id="stale_prior_model",
                                            status="provisional_diagnostic_only")
            Build_Model.build_model()
            after_rows = own.end_inventory(hinge_registry())
        after = {(r["member_tag"], r["end"]): r["theta_p"] for r in after_rows}
        self.assertEqual(before, after)
        for r in after_rows:
            self.assertEqual(r["bar_slip"], "member_hinge")
            self.assertEqual(r["bond_slip_indicator"], 1.0)
            self.assertFalse(r["interface_present"])
            self.assertEqual(r["decomposition_status"], "legacy_included_slip_scope")
        domain = own.slip_domain()
        self.assertEqual(domain["owner"], "Build_Model.build_model")
        self.assertEqual(domain["stale_discarded"], 2)
        self.assertEqual({(r["member_tag"], r["end"]) for r in domain["stale_records"]},
                         {(base["member_tag"], base["end"]), (elevated["member_tag"], elevated["end"])})
        self.assertEqual(own.slip_interfaces(), {})

    def test_registration_is_honoured_only_while_its_element_is_installed(self):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        own.begin_domain("test_fixture")
        own.register_slip_interface(9, "i", calibration_id="x", status="provisional_diagnostic_only", element_tag=901)
        e = end(9, "column", "i", "column_base")
        with self.assertRaisesRegex(ValueError, "not in the current OpenSees domain"):
            own.slip_owner(e)
        with self.assertRaisesRegex(ValueError, "absent from the domain"):
            own.validate_installed()
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 0.0, 0.0, 0.0)
        ops.uniaxialMaterial("Elastic", 1, 1.0)
        ops.element("zeroLength", 901, 1, 2, "-mat", 1, "-dir", 5)
        self.assertEqual(own.slip_owner(e), "face_interface")
        self.assertEqual(own.validate_installed(), 1)
        # a builder that installs no interfaces refuses any registration
        with self.assertRaisesRegex(ValueError, "installs no face interfaces"):
            own.validate_installed(expect_none=True)
        # a registration that names no element is refused by the installed check
        own.register_slip_interface(9, "j", calibration_id="x", status="provisional_diagnostic_only")
        with self.assertRaisesRegex(ValueError, "names no interface element"):
            own.validate_installed()
        # opening the next domain discards both and reports them
        domain = own.begin_domain("next")
        self.assertEqual(domain["stale_discarded"], 2)
        self.assertIsNone(own.registered_slip_interface(9, "i"))


if __name__ == "__main__":
    unittest.main()
