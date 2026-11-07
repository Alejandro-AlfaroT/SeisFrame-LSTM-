"""The frozen V2 member spring calibration (seisframe_imk_v2_frozen_20261029; principal researcher and RA approval,
2026-10-29).

Golden values are the archived approved evidence of the twenty-run MCER/DBE study (case_0064, beam 173, end j, local
axis y, material 3017325), written down here and never recomputed by the code under test. The mechanics are checked
against the hand algebra of the nominal half-member reference and against the native OpenSees IMKPeakOriented
backbone. The full ten-design parity, the native probe replay and the frame regression against the archived study
are in tools/imk_frozen_v2 (they need the study evidence); these tests need none.
"""
import contextlib
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openseespy.opensees as ops                                              # noqa: E402
import Structure_Parameters as sp                                              # noqa: E402
from Model import Analysis_Profile as ap                                       # noqa: E402
from Model import IMK_Calibration as cal                                       # noqa: E402
from Model import IMK_Hinges as hinges                                         # noqa: E402
from Model import IMK_Materials as mm                                          # noqa: E402
from Model import IMK_Spring_Mapping as mapping                                # noqa: E402
from Model.IMK_Materials import RotationalBackbone                             # noqa: E402

# ---- golden evidence: approved study, case_0064 beam 173 (beam_y, L = 216 in) end j axis y, material 3017325 ------
TAG = 3017325
KH = 47188289.00237646
K0 = 2359414.450118823
KB = 2477385.172624764
MY_HOGGING = 8249.551921462938                 # canonical physical direction of a beam's vertical bending
MY_SAGGING = 5805.419123130262
THETA_P = 0.06616077795789096                  # member interval (Eq. 3.10), both branches
THETA_PC = 0.1                                 # member interval at the published 0.10 ceiling (Eq. 3.16)
DU = 0.17116077795789097                       # retained native cutoff: theta_p + theta_pc + 0.005
LAMBDA_H = 105.65746528473817                  # Eq. 3.20, spring y
GOLDEN_COMMAND = ["IMKPeakOriented", 3017325, 47188289.00237646,
                  0.06592058298671885, 0.10257770212965206, 0.17116077795789097, 5805.419123130262, 1.1, 0.2,
                  0.06581945877939033, 0.10366293752537273, 0.17116077795789097, 8249.551921462938, 1.1, 0.2,
                  0.524956230127582, 0.524956230127582, 1421008844752.4194, 1421008844752.4194,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
GOLDEN_THETA_Y_REF = 0.0034964403651285094
GOLDEN_EREF_SC = 3047.590937189035
CONTEXT = {"physical_member_tag": 173, "end": "j", "member_type": "beam_y", "spring_local_direction": 5}
PROVENANCE = {"deformation_scope": "member_end_spring; test fixture", "backbone_source": "haselton_2008"}


def member(direction, my, theta_p=THETA_P, theta_pc=THETA_PC, du=DU):
    return mapping.MemberBranch(direction, my, theta_p, theta_pc, du, 1.10, 0.20)


def install(tag=TAG, kh=KH, hogging=MY_HOGGING, sagging=MY_SAGGING, *, reverse=True, lambda_h=LAMBDA_H,
            directions=("hogging", "sagging"), provenance=None, **branch):
    """One spring on the frozen calibration, physical positive (canonical) first; reverse = end j."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    return mapping.define_frozen_member_imk(
        "IMKPeakOriented", tag, kh, member(directions[0], hogging, **branch), member(directions[1], sagging, **branch),
        lambda_haselton=lambda_h, c_s=1.0, c_c=1.0, c_a=1.0, c_k=1.0, d_pos=1.0, d_neg=1.0, reverse=reverse,
        physical_directions=directions, provenance=dict(provenance or PROVENANCE), context=CONTEXT)


def moment_at(command, rotation, steps=4000):
    """Monotonic native response of a FRESH copy of the installed command at ``rotation`` (no reversal, so no cyclic
    deterioration): the material is re-created and pushed in small steps from zero."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.uniaxialMaterial(*command)
    ops.testUniaxialMaterial(command[1])
    for k in range(1, steps + 1):
        ops.setStrain(float(rotation) * k / steps)
    return ops.getStress()


@contextlib.contextmanager
def restored_parameters(**values):
    snapshot = dict(vars(sp))
    try:
        for key, value in values.items():
            setattr(sp, key, value)
        yield
    finally:
        for key in set(vars(sp)) - set(snapshot):
            delattr(sp, key)
        vars(sp).update(snapshot)


class GoldenCheckpoint(unittest.TestCase):
    """The installed command of beam 173 end j axis y is the approved one, argument for argument."""

    def test_installed_command_is_the_approved_command(self):
        installed = install()
        self.assertEqual(installed["command"], GOLDEN_COMMAND)
        ops.wipe()

    def test_reference_and_energy_records_are_the_approved_values(self):
        installed = install()
        record = installed["provenance"]["member_spring_calibration"]
        self.assertEqual(record["calibration_id"], "seisframe_imk_v2_frozen_20261029")
        self.assertEqual(record["numerically_equivalent_to"], "v2_imk_half_member_reference_research_20261028")
        self.assertIs(record["approval"]["experimental_validation"], False)
        self.assertEqual(record["reference"]["K0_kip_in_per_rad"], K0)
        self.assertEqual(record["reference"]["Kb_kip_in_per_rad"], KB)
        self.assertEqual(record["energy"]["energy_reference_theta_y_rad"], GOLDEN_THETA_Y_REF)
        self.assertEqual(installed["reference_energies_kip_in_rad"]["S"], GOLDEN_EREF_SC)
        self.assertEqual(installed["reference_energies_kip_in_rad"]["C"], GOLDEN_EREF_SC)
        mapped = installed["provenance"]["energy_mapping"]
        self.assertEqual((mapped["positive_input_direction"], mapped["negative_input_direction"]), ("sagging", "hogging"))
        self.assertEqual(mapped["canonical_physical_direction"], "hogging")
        ops.wipe()


class HalfMemberMechanics(unittest.TestCase):
    """Hand algebra of the reference against the native OpenSees backbone of the installed spring."""

    def setUp(self):
        self.installed = install()
        self.addCleanup(ops.wipe)

    def branch(self, sign):
        return self.installed[sign]

    def test_reference_stiffnesses_and_split(self):
        self.assertEqual(KH / 20.0, K0)
        self.assertEqual(1.05 * K0, KB)
        # The retained split assembles to 420/421 K0 in double curvature; it is stated, not corrected.
        assembled = 1.0 / (1.0 / KH + 1.0 / KB)
        self.assertAlmostEqual(assembled / K0, 420.0 / 421.0, places=14)
        self.assertAlmostEqual(mapping.ASSEMBLED_INITIAL_STIFFNESS_OVER_K0, 420.0 / 421.0, places=14)
        self.assertNotAlmostEqual(KB / K0, 20.0 / 19.0, places=6)          # the interior factor is not n/(n-1)
        self.assertEqual(self.installed["ke"], KH)                        # the spring is n K0, not (n+1) K0

    def test_cap_target_and_post_capping_slope_on_the_reference(self):
        for sign, physical in (("positive", MY_SAGGING), ("negative", MY_HOGGING)):
            with self.subTest(sign=sign):
                b = self.branch(sign)
                s = 1.0 if sign == "positive" else -1.0
                fy, mc, mr = b["fy"], 1.1 * b["fy"], 0.2 * b["fy"]
                self.assertEqual(fy, physical)
                native_cap = fy / KH + b["dp"]
                self.assertAlmostEqual(abs(moment_at(self.installed["command"], s * native_cap)), mc, delta=1e-9 * mc)
                # The capping point of the native spring is the member-reference cap theta_y,ref + theta_p.
                self.assertAlmostEqual(native_cap + mc / KB, fy / K0 + THETA_P, delta=1e-15)
                # Post-capping: zero intercept of the reference at cap + theta_pc, slope -Mc/theta_pc.
                residual_onset = native_cap + b["dpc"] * (1.0 - mr / mc)
                self.assertAlmostEqual(abs(moment_at(self.installed["command"], s * residual_onset)), mr, delta=1e-7 * mc)
                self.assertAlmostEqual(residual_onset + mr / KB, fy / K0 + THETA_P + THETA_PC * (1.0 - mr / mc), delta=1e-15)
                a, c = native_cap + 0.25 * b["dpc"] * (1 - mr / mc), native_cap + 0.75 * b["dpc"] * (1 - mr / mc)
                ma, mb = abs(moment_at(self.installed["command"], s * a)), abs(moment_at(self.installed["command"], s * c))
                reference_slope = (mb - ma) / ((c + mb / KB) - (a + ma / KB))
                self.assertAlmostEqual(reference_slope, -mc / THETA_PC, delta=1e-9 * mc / THETA_PC)

    def test_yield_hardening_and_unchanged_cutoff(self):
        for sign in ("positive", "negative"):
            with self.subTest(sign=sign):
                b = self.branch(sign)
                s = 1.0 if sign == "positive" else -1.0
                self.assertAlmostEqual(abs(moment_at(self.installed["command"], s * b["fy"] / KH)), b["fy"], delta=1e-9 * b["fy"])
                self.assertEqual(b["du"], DU)
                self.assertAlmostEqual(DU, THETA_P + THETA_PC + 0.005, delta=1e-15)
                self.assertAlmostEqual(abs(moment_at(self.installed["command"], s * (DU - 1e-6))), 0.2 * b["fy"], delta=1e-9 * b["fy"])
                self.assertEqual(moment_at(self.installed["command"], s * (DU + 1e-6)), 0.0)
                # Native dpc exceeds the member 0.10 ceiling and is not clamped back to it.
                self.assertGreater(b["dpc"], 0.10)


class EnergyRule(unittest.TestCase):

    def tearDown(self):
        ops.wipe()

    def test_mirrored_ends_keep_every_physical_energy(self):
        start, end = install(reverse=False), install(reverse=True)
        for mode in ("S", "C", "A", "K"):
            self.assertEqual(start["reference_energies_kip_in_rad"][mode], end["reference_energies_kip_in_rad"][mode])
        # Complete branches move between the native signs; the native Lamda follows the native positive strength.
        self.assertEqual(start["positive"], end["negative"])
        self.assertEqual(start["negative"], end["positive"])
        self.assertEqual(start["positive"]["fy"], MY_HOGGING)
        self.assertAlmostEqual(end["cyclic"]["lamda_s"] / start["cyclic"]["lamda_s"], MY_HOGGING / MY_SAGGING, places=14)

    def test_canonical_direction_is_hogging_even_when_sagging_is_stronger(self):
        weak_hogging = install(hogging=MY_SAGGING, sagging=MY_HOGGING, reverse=False)
        expected = LAMBDA_H * MY_SAGGING * (MY_SAGGING / K0)
        self.assertAlmostEqual(weak_hogging["reference_energies_kip_in_rad"]["S"], expected, delta=1e-12 * expected)
        self.assertEqual(weak_hogging["provenance"]["energy_mapping"]["canonical_physical_direction"], "hogging")

    def test_active_energy_is_member_reference_not_nominal_or_spring_yield(self):
        installed = install(reverse=False)
        e = installed["reference_energies_kip_in_rad"]["S"]
        self.assertAlmostEqual(e, LAMBDA_H * MY_HOGGING ** 2 / K0, delta=1e-12 * e)
        self.assertNotAlmostEqual(e, LAMBDA_H * MY_HOGGING * 0.005, delta=1.0)          # the retired nominal rule
        self.assertNotAlmostEqual(e, LAMBDA_H * MY_HOGGING ** 2 / KH, delta=1.0)        # not My/Kh
        self.assertEqual(installed["reference_energies_kip_in_rad"]["S"], installed["reference_energies_kip_in_rad"]["C"])

    def test_suppressed_modes_keep_the_canonical_capacity_without_the_energy_rotation(self):
        for kh in (KH, 3.0 * KH):
            installed = install(kh=kh, reverse=True)
            self.assertEqual(installed["reference_energies_kip_in_rad"]["A"], 1.0e12 * MY_HOGGING)
            self.assertEqual(installed["reference_energies_kip_in_rad"]["K"], 1.0e12 * MY_HOGGING)
        self.assertNotEqual(install(kh=KH)["reference_energies_kip_in_rad"]["S"],
                            install(kh=3.0 * KH)["reference_energies_kip_in_rad"]["S"])

    def test_cutoff_padding_does_not_anchor_the_energy(self):
        base = install(reverse=False)
        padded = install(reverse=False, du=DU + 0.01)
        self.assertEqual(padded["reference_energies_kip_in_rad"], base["reference_energies_kip_in_rad"])
        self.assertEqual(padded["positive"]["du"], DU + 0.01)
        self.assertEqual(padded["positive"]["dp"], base["positive"]["dp"])


class Refusals(unittest.TestCase):

    def tearDown(self):
        ops.wipe()

    def test_a_native_branch_is_never_converted_again(self):
        reference = mapping.half_member_reference(KH)
        with self.assertRaises(TypeError):
            mapping.native_branch(RotationalBackbone(THETA_P, THETA_PC, DU, MY_HOGGING, 1.1, 0.2), reference,
                                  input_sign="positive", context=CONTEXT)
        installed = install()
        with self.assertRaisesRegex(ValueError, "already records a member-to-native mapping"):
            install(provenance=installed["provenance"])
        with mock.patch.object(mm, "_research_half_member_installed", True, create=True), \
                self.assertRaisesRegex(RuntimeError, "second time"):
            install()

    def test_cutoff_at_or_before_the_mapped_cap_is_refused_with_context(self):
        native_cap = MY_SAGGING / KH + (MY_SAGGING / K0 + THETA_P - 1.1 * MY_SAGGING / KB - MY_SAGGING / KH)
        with self.assertRaisesRegex(ValueError, r"beam_y 173 end j axis y.*cutoff"):
            install(du=native_cap)

    def test_invalid_inputs_are_refused_not_repaired(self):
        for bad in ({"theta_p": 0.0}, {"theta_pc": -0.01}, {"du": float("nan")}, {"theta_p": float("inf")}):
            with self.subTest(**{k: str(v) for k, v in bad.items()}), self.assertRaises(ValueError):
                install(**bad)
        for kwargs in ({"lambda_h": 0.0}, {"hogging": -1.0}, {"kh": float("nan")}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                install(**kwargs)
        with self.assertRaisesRegex(ValueError, "maps to native dp"):
            install(theta_p=1e-6)                     # cap inside the elastic reference: negative native dp
        with self.assertRaisesRegex(ValueError, "strength ratios"):
            mapping.native_branch(replace(member("hogging", MY_HOGGING), fmax_fy=1.13), mapping.half_member_reference(KH),
                                  input_sign="positive", context=CONTEXT)

    def test_reference_must_match_the_element(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            mapping.half_member_reference(KH, k0_from_section=K0 * 1.001)
        with self.assertRaisesRegex(ValueError, "inertia factor"):
            mapping.half_member_reference(KH, element_factor=21.0 / 19.0)
        self.assertEqual(mapping.half_member_reference(KH, k0_from_section=K0, element_factor=1.05).kb, KB)


class Configuration(unittest.TestCase):

    def test_frozen_is_the_repository_default_and_refuses_mixed_policies(self):
        self.assertEqual(sp.IMK_MEMBER_SPRING_CALIBRATION, mapping.FROZEN_V2)
        self.assertEqual(mapping.configuration_problems(), [])
        for key, value in (("IMK_ENERGY_MAPPING_MODE", "legacy_unmapped"),
                           ("IMK_ENERGY_MAPPING_MODE", "provisional_min_yield_anchor_v1"),
                           ("IMK_MATERIAL_TYPE", "IMKBilin"), ("IMK_HINGE_STIFFNESS_MODE", "yield_rotation"),
                           ("IMK_HINGE_STIFFNESS_FACTOR", 21.0), ("IMK_FMAXFY_POS", 1.13), ("IMK_FRESFY_NEG", 0.0),
                           ("IMK_DETERIORATION_MODE", "direct"), ("IMK_USE_CALIBRATED_BACKBONE", False),
                           ("IMK_D_POS", 0.9)):
            with self.subTest(key=key, value=value), restored_parameters(**{key: value}):
                self.assertTrue(mapping.configuration_problems())
                with self.assertRaises(ValueError):
                    mapping.require_configuration()
        with restored_parameters(IMK_ENERGY_MAPPING_MODE="legacy_unmapped",
                                 IMK_MEMBER_SPRING_CALIBRATION=mapping.LEGACY_REPLAY):
            self.assertEqual(mapping.configuration_problems(), [])
        with restored_parameters(IMK_MEMBER_SPRING_CALIBRATION="something_else"):
            with self.assertRaises(ValueError):
                mapping.is_frozen()

    def test_global_thresholds_are_refused_under_the_frozen_calibration(self):
        with self.assertRaisesRegex(ValueError, "installed_spring_points"):
            hinges.imk_hinge_thresholds("beam_x", "rot_y", 240.0)

    def test_profiles_carry_and_require_the_calibration(self):
        with restored_parameters():
            identity = ap.apply_profile(ap.V2_FLEXURE)
            self.assertEqual(identity["state"]["IMK_MEMBER_SPRING_CALIBRATION"], mapping.FROZEN_V2)
            replay = ap.apply_profile(ap.V2_FLEXURE_LEGACY_IMK_REPLAY)
            self.assertEqual(sp.IMK_MEMBER_SPRING_CALIBRATION, mapping.LEGACY_REPLAY)
            self.assertNotEqual(replay["sha256"], identity["sha256"])
            self.assertEqual(ap.PROFILES[ap.V2_FLEXURE_LEGACY_IMK_REPLAY]["design_profile"], ap.V2_FLEXURE)
            # A process configured for the replay is refused by the V2 profile rather than silently switched.
            with self.assertRaisesRegex(ValueError, "IMK_MEMBER_SPRING_CALIBRATION"):
                ap.apply_profile(ap.V2_FLEXURE)
        with restored_parameters():
            ap.apply_profile(ap.V2_FLEXURE_ANCHORED)
            self.assertEqual(sp.IMK_MEMBER_SPRING_CALIBRATION, mapping.LEGACY_REPLAY)
        self.assertEqual(ap.FROZEN_IMK, mapping.FROZEN_V2)
        self.assertEqual(ap.LEGACY_IMK, mapping.LEGACY_REPLAY)

    def test_recorded_model_differences_are_reported(self):
        from Analysis.Fixed_Design_Diagnostics import recorded_model_differences
        recorded = {"settings": {"JOINT_MODEL": "rigid_centerline"}, "state": {"IMK_ENERGY_MAPPING_MODE": "legacy_unmapped"}}
        installed = {"settings": {"JOINT_MODEL": "rigid_centerline"},
                     "state": {"IMK_ENERGY_MAPPING_MODE": "canonical_physical_orientation_baseline_v1",
                               "IMK_MEMBER_SPRING_CALIBRATION": mapping.FROZEN_V2}}
        differences = {d["key"]: d for d in recorded_model_differences(recorded, installed)}
        self.assertEqual(set(differences), {"IMK_ENERGY_MAPPING_MODE", "IMK_MEMBER_SPRING_CALIBRATION"})
        self.assertIsNone(differences["IMK_MEMBER_SPRING_CALIBRATION"]["record"])


class MemberRegressions(unittest.TestCase):
    """The member side: Haselton intervals, guards and the retained cutoff are unchanged; the nominal theta_y is not
    an active energy or cutoff quantity under the frozen calibration."""

    BEAM = None

    @classmethod
    def setUpClass(cls):
        import grouped_fixture as gf
        cls.BEAM = gf.BEAM

    def backbone(self, **values):
        with restored_parameters(**values):
            return cal.backbone_for_member("beam_x", 0.0, design=self.BEAM)

    def test_retained_cutoff_and_reported_guards(self):
        frozen = self.backbone()
        self.assertEqual(frozen["member_spring_calibration"], mapping.FROZEN_V2)
        self.assertEqual(frozen["legacy_cutoff_padding_rad"], 0.005)
        self.assertEqual(frozen["theta_u"], frozen["theta_p"] + frozen["theta_pc"] + 0.005)
        guards = frozen["regression_guards"]
        self.assertEqual(guards["theta_pc_published_ceiling"], 0.10)
        self.assertEqual(guards["theta_pc_ceiling_active"], guards["theta_pc_regression"] > 0.10)
        self.assertEqual(frozen["theta_pc"], min(0.10, max(0.010, guards["theta_pc_regression"])))
        legacy = self.backbone(IMK_MEMBER_SPRING_CALIBRATION=mapping.LEGACY_REPLAY)
        for key in ("theta_p", "theta_pc", "theta_u", "lambda_haselton_dimensionless"):
            self.assertEqual(frozen[key], legacy[key])

    def test_nominal_theta_y_is_not_active_under_the_frozen_calibration(self):
        frozen = self.backbone()
        for key in ("lambda_opensees_rad", "lambda_opensees_by_mode_rad", "energy_reference_member_theta_y_rad"):
            self.assertNotIn(key, frozen)
        self.assertEqual(frozen["deterioration_historical_nominal"]["nominal_member_theta_y_rad"], 0.005)
        changed = self.backbone(IMK_BEAM_THETA_Y=0.007)
        self.assertEqual(changed["theta_u"], frozen["theta_u"])
        self.assertEqual(changed["deterioration_by_axis"]["y"]["lambda_haselton_dimensionless"],
                         frozen["deterioration_by_axis"]["y"]["lambda_haselton_dimensionless"])
        # The pre-freeze rule is still reproduced by the explicit replay.
        legacy = self.backbone(IMK_MEMBER_SPRING_CALIBRATION=mapping.LEGACY_REPLAY)
        self.assertAlmostEqual(legacy["lambda_opensees_rad"], legacy["lambda_haselton_dimensionless"] * 0.005, places=15)
        self.assertEqual(self.backbone(IMK_MEMBER_SPRING_CALIBRATION=mapping.LEGACY_REPLAY, IMK_BEAM_THETA_Y=0.007)["theta_u"],
                         legacy["theta_p"] + legacy["theta_pc"] + 0.007)


class AssembledFrame(unittest.TestCase):
    """Every member-end spring of a built frame is installed once on the frozen calibration (hand formulas)."""

    def test_every_spring_of_the_fixture_frame(self):
        import grouped_fixture as gf
        from Model.Build_Model import build_model
        with gf.frame(IMK_ENERGY_MAPPING_MODE=mm.CANONICAL_BASELINE_VERSION, IMK_MEMBER_SPRING_CALIBRATION=mapping.FROZEN_V2):
            build_model()
            registry = hinges.hinge_registry()
            self.assertEqual(len(registry), len(gf.hand_members()))
            springs = 0
            for tag, entry in registry.items():
                self.assertEqual(entry["member_spring_calibration"], mapping.FROZEN_V2)
                self.assertIsNone(entry["theta_y_target"])
                by_end = {}
                for end in ("i", "j"):
                    energies = {}
                    for axis in ("y", "z"):
                        material = entry["installed_materials"][end][axis]
                        springs += 1
                        command = material["command"]
                        kh = command[2]
                        k0 = kh / 20.0
                        kb = 1.05 * k0
                        mapped = material["provenance"]["energy_mapping"]
                        for offset, sign in ((3, "positive"), (9, "negative")):
                            dp, dpc, du, fy, fmax, fres = command[offset:offset + 6]
                            self.assertEqual((fmax, fres), (1.1, 0.2))
                            theta_p, theta_pc = entry["backbone_by_end"][end]["theta_p"], entry["backbone_by_end"][end]["theta_pc"]
                            self.assertAlmostEqual(dp, fy / k0 + theta_p - 1.1 * fy / kb - fy / kh, delta=1e-15)
                            self.assertAlmostEqual(dpc, theta_pc + 1.1 * fy / kb, delta=1e-15)
                            self.assertEqual(du, entry["backbone_by_end"][end]["theta_u"])
                        canonical = mapped["reference_moment_kip_in"]
                        lam = entry["backbone_by_end"][end]["deterioration_by_axis"][axis]["lambda_haselton_dimensionless"]
                        expected = lam * canonical * canonical / k0
                        self.assertAlmostEqual(material["reference_energies_kip_in_rad"]["S"], expected, delta=1e-12 * expected)
                        self.assertAlmostEqual(material["reference_energies_kip_in_rad"]["A"], 1e12 * canonical,
                                               delta=4e-16 * 1e12 * canonical)   # recomputed as Lamda x My+ (2 ulp)
                        expected_direction = ("hogging" if entry["member_type"].startswith("beam") and axis == "y"
                                              else f"section_positive_{axis}")
                        self.assertEqual(mapped["canonical_physical_direction"], expected_direction)
                        # The physical energies as mapped (the identity's Lamda x My+ recomputation rounds A/K by an ulp).
                        energies[axis] = mapped["reference_energies_kip_in_rad"]
                        points = entry["spring_points"][end][axis]
                        self.assertEqual(points["calibration_id"], mapping.FROZEN_V2)
                        self.assertEqual(points["positive"]["native_theta_cap_rad"],
                                         command[6] / kh + command[3])
                    by_end[end] = energies
                # Both ends of a member with equal physical strengths at its ends carry the same physical energies.
                if entry.get("yield_moment_y_hogging_i_kip_in") == entry.get("yield_moment_y_hogging_j_kip_in"):
                    for axis in ("y", "z"):
                        self.assertEqual(by_end["i"][axis], by_end["j"][axis])
            self.assertEqual(springs, 4 * len(registry))
            topology = ap.installed_topology()
            self.assertEqual(topology["installed_member_spring_calibrations"], {mapping.FROZEN_V2: springs})
            ops.wipe()

    def test_the_replay_installs_the_pre_freeze_rule(self):
        import grouped_fixture as gf
        from Model.Build_Model import build_model
        with gf.frame(IMK_ENERGY_MAPPING_MODE=mm.CANONICAL_BASELINE_VERSION,
                      IMK_MEMBER_SPRING_CALIBRATION=mapping.LEGACY_REPLAY):
            build_model()
            for entry in hinges.hinge_registry().values():
                for end in ("i", "j"):
                    command = entry["installed_materials"][end]["y"]["command"]
                    self.assertEqual(command[3], entry["backbone_by_end"][end]["theta_p"])
                    self.assertEqual(command[4], entry["backbone_by_end"][end]["theta_pc"])
                    self.assertNotIn("member_spring_calibration", entry["installed_materials"][end]["y"]["provenance"])
            ops.wipe()


class ExportVersions(unittest.TestCase):
    """The changed export meanings carry new versions, and the hinge table's producer and reader name the same one."""

    def test_producer_and_reader_agree_and_versions_are_bumped(self):
        import Ground_Motion_Main as gm
        from Analysis import Hinge_Moment_Rotation as hmr
        from Data_Generation import Hybrid_Exporter as hx
        self.assertEqual(gm.HINGE_BACKBONE_VERSION, hx.HINGE_BACKBONE_VERSION)
        self.assertNotEqual(hmr.SCHEMA_VERSION, "hinge_moment_rotation_v1")
        self.assertNotEqual(hx.SCHEMA_VERSION, "hybrid_gnn_lstm_v3")
        self.assertIn("imk_member_spring_calibration", gm.OUTPUT_IDENTITY_KEYS)


class SpringPointsAndDemand(unittest.TestCase):

    def test_points_and_demand_of_the_golden_spring(self):
        installed = install()
        ops.wipe()
        points = mapping.spring_points(installed)
        positive = points["positive"]
        self.assertEqual(positive["physical_direction"], "sagging")
        self.assertEqual(positive["native_theta_y_rad"], MY_SAGGING / KH)
        self.assertEqual(positive["native_theta_cap_rad"], MY_SAGGING / KH + GOLDEN_COMMAND[3])
        self.assertEqual(positive["member_theta_y_reference_rad"], MY_SAGGING / K0)
        self.assertEqual(positive["member_theta_p_rad"], THETA_P)
        self.assertEqual(points["member_reference"]["Kb_kip_in_per_rad"], KB)
        below = mapping.rotation_demand(points, 0.99 * positive["native_theta_cap_rad"], 0.0)
        self.assertFalse(below["past_capping"])
        self.assertLess(below["capping_ratio"], 1.0)
        at = mapping.rotation_demand(points, positive["native_theta_cap_rad"], 0.0)
        self.assertTrue(at["past_capping"])
        self.assertAlmostEqual(at["capping_ratio"], 1.0, places=12)
        cut = mapping.rotation_demand(points, 0.0, -DU)
        self.assertEqual(cut["governing_sign"], "negative")
        self.assertTrue(cut["past_native_cutoff"])
        self.assertIsNone(mapping.spring_points({"material_type": "Elastic", "released": True}))


if __name__ == "__main__":
    unittest.main()
