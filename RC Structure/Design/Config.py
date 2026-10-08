"""
Design/Config.py
Configuration dataclasses for the RC frame design pipeline.

All design-decision parameters live here.  Structure_Parameters.py retains
geometry scalars and OpenSees solver settings that feed the model builder
directly; this module is the design-decision layer on top of it.

Usage
-----
    from Design.Config import DesignConfig

    cfg = DesignConfig()                          # all defaults from sp
    cfg = DesignConfig(dcr=DCRTargets(dcr_band_lo=0.70, dcr_target=0.80))
    cfg = DesignConfig.from_structure_parameters()   # explicit mirror of sp
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import Structure_Parameters as sp


# ---------------------------------------------------------------------------
# Scalar-or-range type
# ---------------------------------------------------------------------------
# A parameter that can be fixed (scalar) or drawn uniformly from [lo, hi]
# during a batch generation run.
ScalarOrRange = Union[float, Tuple[float, float]]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RebarConfig:
    """Longitudinal and transverse reinforcement constraints.

    bar_sizes_col / bar_sizes_beam
        ASTM standard bar numbers available for selection (e.g. 8 → #8).

    col_n_top_range / beam_n_range
        Inclusive (min, max) bar count search bounds.
        Column: n_top == n_bot enforced (symmetric section).
        Beam: equal top/bottom counts by default; set beam_symmetric=False
        explicitly to reproduce the independent-layer design search.

    col_n_side_options
        Candidate side-bar counts per face (total side = 2 × value).
        Face bar counts stay within what the capacity-design hoop ladder can
        tie with a leg or crosstie on every face bar (six), so the steel the
        strong-column rule can demand at the top of the section ladder comes
        from bar size (#10, #11), not from denser faces; the bar-spacing and
        18.7.5.2(f) filters in Redesign._col_candidates remove what does not
        fit.

    rho_col_min / rho_col_max
        ACI 318-19 §18.7.4.1 SMRF longitudinal steel ratio limits.

    rho_col_practical_max
        Proportioning preference, not a code limit: the largest column steel
        ratio the strong-column/weak-beam escalation may select before the
        search steps to a larger column section instead. ACI 318-19 18.7.4.1
        permits 0.06; R18.7.4.1 names congestion, splicing and shear as the
        reasons practice stays well below it. Ratios above this are never
        selected automatically.

    cover_in
        Distance from concrete face to longitudinal bar centroid (in),
        NOT clear cover to the outside of transverse reinforcement.

    stirrup_*
        Transverse reinforcement properties used for shear checks.
        These were previously hardcoded in RC_Design_Check.py.

    section_round_increment_in
        Snap section dimensions to this increment (0 = no snapping).
    """
    # No. 14 joined the column ladder on 2026-09-27 with the column cage rule
    # (fewer, larger bars per face keep the beam-bar lanes open); it cannot be
    # lap spliced (ACI 318-19 25.5.1.1), so its splices are Type 2 mechanical
    # (SMRF_Capacity_Design.design_splices). No. 18 is not offered.
    bar_sizes_col: List[int] = field(
        default_factory=lambda: [5, 6, 7, 8, 9, 10, 11, 14]
    )
    # No. 11 joined the beam ladder on 2026-09-27 with the fewest-layers
    # preference (larger bars in one layer); 18.8.2.3 (20 db through the
    # joint) still screens it per column in Redesign._beam_candidates.
    bar_sizes_beam: List[int] = field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9, 10, 11]
    )

    col_n_top_range: Tuple[int, int] = (2, 6)     # inclusive; n_bot = n_top
    col_n_side_options: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]    # bars per face
    )
    # Total bars per face, including both permitted layers. The former seven-bar
    # ceiling excluded reinforced candidates even when the two-layer cage fit.
    # Actual lanes, layer count, steel ratio and joint depth still screen each cage.
    beam_n_range: Tuple[int, int] = (2, 14)
    beam_symmetric: bool = True                  # same bar size and count on both faces
    # 2026-09-27 (user decision, "the larger bars per layer option"): among
    # beam cages whose estimated DCR is within the ceiling, the pick takes the
    # fewest layers first (larger bars in one layer keep the depth the second
    # layer would cost), then the DCR objective and the least steel as before.
    beam_prefer_fewest_layers: bool = True
    # 2026-09-27 (user decision, "wire the column-side one-layer preference"):
    # among column cages that keep the estimated DCR within the ceiling, the
    # strength pick and the strong-column escalation take the cages the
    # current beam bars pass in the fewest layers first (fewer, larger bars
    # per face open the lanes), then the objective and the least steel. A
    # tie-break among cages that pass; strength and the joint rule still
    # decide. Ordinary joint coordination in practice.
    col_prefer_fewest_beam_layers: bool = True

    rho_col_min: float = 0.01
    rho_col_max: float = 0.06
    rho_col_practical_max: float = 0.04

    cover_in: float = field(default_factory=lambda: sp.COVER)
    beam_clear_cover_in: float = field(default_factory=lambda: sp.BEAM_CLEAR_COVER_IN)
    col_clear_cover_in: float = field(default_factory=lambda: sp.COL_CLEAR_COVER_IN)
    aggregate_max_size_in: float = field(default_factory=lambda: sp.AGGREGATE_MAX_SIZE_IN)

    def centroid_cover_in(self, member_type, bar_size=None):
        """Current/candidate bar offset; explicit clear cover only in new mode."""
        if sp.SLAB_THICKNESS_IN is None:
            return self.cover_in
        if member_type == "beam":
            clear, hoop, selected = self.beam_clear_cover_in, self.beam_stirrup_bar_size, sp.BEAM_BAR_SIZE
        elif member_type == "column":
            clear, hoop, selected = self.col_clear_cover_in, self.col_stirrup_bar_size, sp.COL_BAR_SIZE
        else:
            raise ValueError("member_type must be beam or column")
        return clear + sp.rebar_diameter(hoop) + 0.5 * sp.rebar_diameter(selected if bar_size is None else bar_size)

    # Transverse reinforcement
    stirrup_bar_size: int = field(default_factory=lambda: sp.COL_STIRRUP_BAR_SIZE)
    stirrup_legs: int = field(default_factory=lambda: sp.COL_STIRRUP_LEGS)
    col_stirrup_bar_size: int = field(default_factory=lambda: sp.COL_STIRRUP_BAR_SIZE)
    col_stirrup_legs: int = field(default_factory=lambda: sp.COL_STIRRUP_LEGS)
    beam_stirrup_bar_size: int = field(default_factory=lambda: sp.BEAM_STIRRUP_BAR_SIZE)
    beam_stirrup_legs: int = field(default_factory=lambda: sp.BEAM_STIRRUP_LEGS)
    stirrup_spacing_col_in: float = field(default_factory=lambda: sp.COL_STIRRUP_SPACING)
    stirrup_spacing_beam_in: float = field(default_factory=lambda: sp.BEAM_STIRRUP_SPACING)
    stirrup_spacing_min_in: float = field(default_factory=lambda: sp.STIRRUP_MIN_SPACING)
    stirrup_spacing_step_in: float = field(default_factory=lambda: sp.STIRRUP_SPACING_STEP)

    section_round_increment_in: float = 2.0

    # ---------------------------------------------------------------------------
    def col_n_top_iter(self):
        lo, hi = self.col_n_top_range
        return range(lo, hi + 1)

    def beam_n_iter(self):
        lo, hi = self.beam_n_range
        return range(lo, hi + 1)

    @property
    def stirrup_area_in2(self) -> float:
        """Total stirrup shear area (both legs) in²."""
        return self.stirrup_legs * sp.rebar_area(self.stirrup_bar_size)

    @property
    def col_stirrup_area_in2(self) -> float:
        """Total column stirrup/tie shear area in in^2."""
        return self.col_stirrup_legs * sp.rebar_area(self.col_stirrup_bar_size)

    @property
    def beam_stirrup_area_in2(self) -> float:
        """Total beam stirrup shear area in in^2."""
        return self.beam_stirrup_legs * sp.rebar_area(self.beam_stirrup_bar_size)


@dataclass
class MaterialConfig:
    """Concrete and steel material properties.

    Each value may be a fixed float or a (lo, hi) tuple for randomised
    sampling.  Call .resolve() to get a MaterialConfig with all fields
    as scalars before using them in calculations.
    """
    fc_col_ksi: ScalarOrRange = field(default_factory=lambda: sp.FC_COL_KSI)
    fc_beam_ksi: ScalarOrRange = field(default_factory=lambda: sp.FC_BEAM_KSI)
    fy_ksi: ScalarOrRange = field(default_factory=lambda: sp.FY_KSI)
    es_ksi: float = field(default_factory=lambda: sp.ES_KSI)
    reinforcement_specification: str = field(default_factory=lambda: sp.REINFORCEMENT_SPECIFICATION)
    exposure: str = field(default_factory=lambda: sp.MATERIAL_EXPOSURE)

    def resolve(self, rng=None) -> "MaterialConfig":
        """Return a MaterialConfig with all ScalarOrRange fields as floats.

        Parameters
        ----------
        rng : numpy.random.Generator or None
            If provided, uses rng.uniform for reproducible draws.
            If None, falls back to the standard library random module.
        """
        def _draw(v):
            if isinstance(v, tuple):
                lo, hi = v
                return rng.uniform(lo, hi) if rng is not None else random.uniform(lo, hi)
            return float(v)

        return MaterialConfig(
            fc_col_ksi=_draw(self.fc_col_ksi),
            fc_beam_ksi=_draw(self.fc_beam_ksi),
            fy_ksi=_draw(self.fy_ksi),
            es_ksi=self.es_ksi,
            reinforcement_specification=self.reinforcement_specification,
            exposure=self.exposure,
        )

    def is_resolved(self) -> bool:
        return all(
            isinstance(v, (int, float))
            for v in (self.fc_col_ksi, self.fc_beam_ksi, self.fy_ksi)
        )


@dataclass
class SectionConfig:
    """Member cross-section dimensions."""
    b_col_in: float = field(default_factory=lambda: sp.B_COL)
    h_col_in: float = field(default_factory=lambda: sp.H_COL)
    b_beam_in: float = field(default_factory=lambda: sp.B_BEAM)
    h_beam_in: float = field(default_factory=lambda: sp.H_BEAM)


@dataclass
class LoadConfig:
    """Gravity and lateral load combination settings.

    mode
        "gravity"          → gravity-only analysis (D + L).
        "gravity_lateral"  → gravity + seismic (ASCE 7 load combination).

    factor_dead / factor_live
        ASCE 7-22 §2.3.1 load combination factors for strength design.
        Default: 1.2D + 1.6L (combination 2).

    strong_column_weak_beam
        When True and mode == "gravity_lateral", enforce ACI 318-19 §18.7.3:
        ΣφMnc / ΣφMnb ≥ scwb_ratio_min at every beam-column joint.
        TODO: the joint-level capacity ratio check for this legacy mode is
        not implemented; the SMRF path applies its own column-strength and
        joint checks (Design/SMRF_Capacity_Design.py, Design/SMRF_Joints.py).
    """
    mode: str = "gravity"

    factor_dead: float = 1.2
    factor_live: float = 1.6

    strong_column_weak_beam: bool = False
    scwb_ratio_min: float = 1.2

    def is_lateral(self) -> bool:
        return self.mode == "gravity_lateral"


@dataclass
class SlabConfig:
    """Uniform building slab policy, not a preselected thickness.

    Normalweight monolithic beam-supported two-way panels, Grade 60, with
    the same thickness at every level including the roof. Bounds/increment
    are research search choices, not ACI-prescribed thicknesses. The 50 psf
    superimposed allowance is provisional; it EXCLUDES concrete self-weight.
    Zero live-load mass is the provisional ordinary-office assumption, not
    a completed ASCE effective-seismic-weight inventory.
    """
    minimum_thickness_in: float = 5.0
    maximum_thickness_in: float = 14.0
    thickness_increment_in: float = 0.5
    superimposed_dead_load_ksf: float = 0.05
    live_load_mass_fraction: float = 0.0


@dataclass
class SlabActionAssertions:
    """Engineering assertions about the slab action evidence. Default: none.

    ``SMRF_Slab_Actions`` computes strip demands from the flexible-beam floor
    model and records the numerical basis of each item below. Whether that
    analysis is applicable, enveloped and verified is an engineering
    judgement made after reviewing it (methodology item 7); it is asserted
    here, never by the code that produced the numbers. Until every flag is
    True the strip routine selects no reinforcement, the beam-plus-slab
    strengths are not established, and the joint SCWB checks stay
    not_evaluated. These fields are part of the design request identity, so
    a design made under an assertion records who made it and on what basis.
    """
    analysis_applicability_verified: bool = False
    all_floors_enveloped: bool = False
    load_pattern_envelope_verified: bool = False
    spatial_envelope_per_unit_width: bool = False
    twisting_moment_resolution_verified: bool = False
    zero_membrane_force_verified: bool = False
    verified: bool = False
    two_way_shear_path_assessed: bool = False
    asserted_by: str = ""
    assertion_date: str = ""
    assertion_basis: str = ""

    def all_asserted(self):
        from Design.SMRF_Common import assertion_provenance_valid
        return assertion_provenance_valid(vars(self)) and all(getattr(self, name) is True for name in (
            "analysis_applicability_verified", "all_floors_enveloped", "load_pattern_envelope_verified",
            "spatial_envelope_per_unit_width", "twisting_moment_resolution_verified",
            "zero_membrane_force_verified", "verified"))


@dataclass
class DemandPolicy:
    """Declared demand-scope assumptions (ASCE 7-22 Chapters 2, 4, 11, 12).

    These are the engineer's declarations about the research archetype; the
    demand checks evaluate the design against them and stay not_evaluated
    while ``declaration_basis`` is empty. The site class is the declared basis
    (Structure_Parameters.ASCE_SITE_CLASS, Site Class C since the user decision
    of 2026-10-02); under ASCE 7-22 it bears on 11.4.7 (site response analysis
    for Site Class F only) even though SDS/SD1/S1 are given directly.
    Occupancy fixes the 12.7.2 effective-weight inventory: an office carries
    a partition allowance (>= 10 psf where partitions exist) and no storage
    live-load fraction. Wind, snow and rain are declared not to govern this
    seismic archetype; the 50 psf floor live load is applied to the roof
    and envelopes the Table 4.3-1 roof live load.
    """
    # One source: Structure_Parameters.ASCE_RISK_CATEGORY (Risk Category III since the user decision of
    # 2026-10-01), which also fixes Ie and the drift criterion (sp.seismic_design_basis). A policy that
    # names another category is refused by the design driver, not silently designed with a stale Ie.
    # The category does not select the occupancy, the gravity loading or the hazard declared below.
    risk_category: str = field(default_factory=lambda: sp.ASCE_RISK_CATEGORY)
    site_class: str = field(default_factory=lambda: sp.ASCE_SITE_CLASS)
    occupancy: str = "office"
    partition_allowance_ksf: float = 0.010
    storage_live_fraction_in_weight: float = 0.0
    roof_live_load_ksf: float = 0.020
    snow_load_ksf: float = 0.0
    wind_governs: bool = False
    rain_ponding_excluded: bool = True
    accidental_torsion_ratio: float = 0.05
    live_load_patterning: bool = True
    declared_by: str = ""
    declaration_date: str = ""
    declaration_basis: str = ""

    def declared(self):
        """True when this is a complete, valid declaration (SMRF_Demands.demand_policy_problems)."""
        from Design.SMRF_Demands import demand_policy_problems
        return not demand_policy_problems(vars(self))

    def problems(self):
        from Design.SMRF_Demands import demand_policy_problems
        return demand_policy_problems(vars(self))


@dataclass
class IndependentVerification:
    """Assertions that a person has done the checks code cannot do about itself.

    Each flag replaces one not_evaluated qualification item with an
    evaluated one carrying ``asserted_by``/``basis``; all default False.
    """
    floor_hand_check_verified: bool = False          # floor.independent_hand_verification
    strength_model_verified: bool = False            # qualification.strength_model_verification
    detailing_model_consistency_verified: bool = False   # qualification.detailing_model_consistency
    slab_column_local_steel_assessed: bool = False   # slab_column_local_minimum_steel (8.6.1.2)
    fire_resistance_scope_accepted: bool = False     # slab_fire_resistance
    congestion_and_placement_accepted: bool = False  # detailing.congestion_and_placement
    floor_frame_compatibility_reviewed: bool = False   # floor.compatibility_idealization_reviewed
    # demands.torsional_irregularity, strength-distribution branch (review item
    # M1, 2026-09-20): the frame-line story-strength model behind the Table
    # 12.3-1 "75% at or on one side" criterion is a declared approximation
    # until a person asserts it. Unasserted, the criterion can establish a
    # Type 1 irregularity (conservative) but never its absence.
    story_strength_model_verified: bool = False
    asserted_by: str = ""
    assertion_date: str = ""
    assertion_basis: str = ""


@dataclass
class DCRTargets:
    """Demand/capacity ratio acceptance criteria and optimisation target.

    Hard constraint
        DCR > dcr_hard_max → member fails; design is rejected.
        Default = 1.0 (ACI strength design).

    Soft band  [dcr_band_lo, dcr_band_hi]
        Optimisation feasibility region.  Designs within the band are
        preferred; those outside incur a penalty in the objective function
        but are not hard-rejected unless DCR > dcr_hard_max.

    dcr_target
        Centroid used for the "min_deviation" objective:
            objective = Σ (DCR_i − dcr_target)²
        A value inside the soft band (e.g. 0.85) centres the distribution.

    objective
        "min_deviation"  minimize Σ (DCR_i − dcr_target)²  ← default
        "min_volume"     minimize total longitudinal steel volume
    """
    dcr_hard_max: float = 1.0

    dcr_band_lo: float = field(default_factory=lambda: sp.DESIGN_DCR_MIN)
    dcr_band_hi: float = field(default_factory=lambda: sp.DESIGN_DCR_MAX)
    dcr_target: float = 0.85

    objective: str = "min_deviation"

    # Biaxial column interaction (ACI 318-19 R22.4; Bresler load contour):
    # (Muy/phiMny)^alpha + (Muz/phiMnz)^alpha <= 1. A stated engineering
    # parameter, covered by IndependentVerification.strength_model_verified:
    # 1.0 is the linear contour (conservative for every section), 1.5 the
    # customary design value; Bresler's measured range is about 1.15-1.55.
    biaxial_contour_exponent: float = 1.5

    def in_band(self, dcr: float) -> bool:
        return self.dcr_band_lo <= dcr <= self.dcr_band_hi

    def hard_fail(self, dcr: float) -> bool:
        return dcr > self.dcr_hard_max


@dataclass
class FloorAnalysisConfig:
    """Rigid-line shell diagnostic, plus the flexible-beam slab-to-frame transfer.

    transfer_to_frame
        Run the flexible-beam floor model for unit dead and unit live
        pressure and load the frame beams/columns with its transfer instead
        of tributary nodal loads (Design/SMRF_Floor_Transfer). The transfer
        mesh is chosen by SMRF_Floor_Analysis.transfer_mesh_per_bay unless
        transfer_mesh_per_bay is set.
    """
    enabled: bool = True
    mesh_per_bay: int = 4
    refinement_mesh_per_bay: int = 8
    transfer_to_frame: bool = True
    transfer_mesh_per_bay: Optional[int] = None
    # Bounded slab-demand refinement, recorded in the request identity: an
    # explicit plan ({meshes, moment_tolerance, shear_tolerance,
    # tolerance_basis}) or a named recipe ({recipe, levels, max_shells, the
    # same three}) that Design/SMRF_Floor_Mesh resolves for the current bay
    # dimensions and beam face at every section iteration, so the policy
    # stays geometry-independent and the resolved coordinates travel with
    # the evidence. None keeps single-mesh demands unverified, so asserted
    # slab actions cannot select reinforcement without a plan. This does not
    # change frame-transfer stiffness/kinematics.
    slab_refinement: Optional[dict] = None


# The one PROBE refinement plan both probe factories (Design/Verify_Designs
# and Design/Evidence_Summary) carry. A recipe, not coordinates, so the
# methodology hash agrees across geometries; the 5% is the inherited
# investigation screen of the 2026-09-24 fixed-candidate benchmark, not an
# ACI acceptance limit, and the recipe is not validated for every geometry:
# a floor whose affordable levels are fewer than two is an explicit
# unresolved result, and a genuine convergence rejection stays a rejection.
# 2026-10-03: allow all four levels on the pilot's largest 6x6 floor
# (129,600 shells at 60 cells/bay). Explicit smaller user budgets still apply.
# 2026-10-07: after a completed but failing final comparison, permit at most
# two nested face-band extensions selected from the failed strip witnesses.
# The same tolerances and shell budget apply; unresolved or failed solves
# remain unverified. This is a numerical investigation policy, not acceptance.
PROBE_SLAB_REFINEMENT = {
    "recipe": "graded_face_v1", "levels": 4, "max_shells": 130000,
    "local_extension": {"method": "failed_witness_face_band_v1", "max_extra_levels": 2},
    "moment_tolerance": 0.05, "shear_tolerance": 0.05,
    "tolerance_basis": "PROBE -- inherited 5% all-strip investigation screen of the 2026-09-24 "
                       "fixed-candidate benchmark (graded face meshes 12/24/48/60 per bay with at most "
                       "two failed-witness face-band extensions inside the same shell budget); not an ACI "
                       "acceptance limit, not a validated recipe for every geometry",
}


@dataclass
class IterationConfig:
    """Controls for the iterative redesign loop.

    convergence_tol
        The loop considers itself converged when the bar specification does
        not change between successive iterations (discrete bar search).
        This tolerance is also used as a guard for near-zero objective
        improvement in future continuous optimisation extensions.

    seed
        Random seed for reproducible batch generation runs (passed to
        MaterialConfig.resolve()).  None → non-reproducible.

    penalty_weight
        Multiplier applied to the objective for infeasible designs (DCR > 1.0
        or ACI limit violations) so that the optimizer still has a gradient to
        follow rather than encountering a silent rejection.
    """
    max_iter: int = 10
    convergence_tol: float = 1e-3
    seed: Optional[int] = None
    penalty_weight: float = 1e3
    # Full evaluations after first feasibility; untested alternatives are reported.
    uniform_reduction_trials: int = 24


@dataclass
class CapacityPolicy:
    """Capacity-design method selection (ACI 318-19 Chapter 18), part of the request identity.

    ``column_shear_method`` names the 18.7.6.1.1 Ve rule
    (Design.SMRF_Capacity_Design.COLUMN_SHEAR_METHODS):
    ``beam_joint_delivery_limited_v2`` is the rule every design saved before
    2026-09-27 was produced with (records without the key are read as it,
    never relabelled); ``column_own_probable_envelope_v3`` is the column-own
    probable-strength rule prepared for review on 2026-09-20 and selected as
    the design basis on 2026-09-27 (repair review: the joint-limited
    rule rested on an unsupported 50/50 sharing reduction, and the existing
    cage was inadequate under column-own demands). ``column_clear_height_convention``
    applies to the column-own rule: ``uniform_face_to_face`` (story_h - h_beam
    at every story, the engineering note's conservative convention) or
    ``physical_base`` (the base story's base-to-soffit height), selected with it.
    """
    column_shear_method: str = "column_own_probable_envelope_v3"
    column_clear_height_convention: str = "physical_base"
    # The uniform design's Ve rule (user decision 2026-10-02): 18.7.6.1.1 lets the column shear be limited
    # by what the beams' probable strengths deliver through the joints, and the column-own rule left 42-in
    # columns designing for about three times that. The grouped design is not defined for the joint-limited
    # rule and keeps ``column_shear_method``. None makes the uniform design read ``column_shear_method`` too.
    # User decision 2026-10-04 (pre-generation review): the beam limit stays, but the beam moments are divided
    # between the column above and the column below by the analysis, not equally
    # (``beam_delivery_analysis_split_v4``; ``beam_joint_delivery_limited_v2`` is the equal split it replaces).
    uniform_column_shear_method: Optional[str] = "beam_delivery_analysis_split_v4"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class DesignConfig:
    """Root design configuration object.

    Aggregates all sub-configs.  Construct with defaults (reads from
    Structure_Parameters) or override any sub-config explicitly.

    Examples
    --------
    cfg = DesignConfig()
    cfg = DesignConfig(dcr=DCRTargets(dcr_band_lo=0.70, dcr_target=0.80))
    cfg = DesignConfig(loads=LoadConfig(mode="gravity_lateral",
                                        strong_column_weak_beam=True))
    """
    rebar: RebarConfig = field(default_factory=RebarConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
    loads: LoadConfig = field(default_factory=LoadConfig)
    slab_actions: SlabActionAssertions = field(default_factory=SlabActionAssertions)
    demands: DemandPolicy = field(default_factory=DemandPolicy)
    verification: IndependentVerification = field(default_factory=IndependentVerification)
    slab: SlabConfig = field(default_factory=SlabConfig)
    floor_analysis: FloorAnalysisConfig = field(default_factory=FloorAnalysisConfig)
    dcr: DCRTargets = field(default_factory=DCRTargets)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    capacity: CapacityPolicy = field(default_factory=CapacityPolicy)

    @classmethod
    def from_structure_parameters(cls) -> "DesignConfig":
        """Construct a DesignConfig that mirrors the current Structure_Parameters."""
        return cls(
            rebar=RebarConfig(
                cover_in=sp.COVER,
                col_stirrup_bar_size=sp.COL_STIRRUP_BAR_SIZE,
                col_stirrup_legs=sp.COL_STIRRUP_LEGS,
                beam_stirrup_bar_size=sp.BEAM_STIRRUP_BAR_SIZE,
                beam_stirrup_legs=sp.BEAM_STIRRUP_LEGS,
                stirrup_spacing_col_in=sp.COL_STIRRUP_SPACING,
                stirrup_spacing_beam_in=sp.BEAM_STIRRUP_SPACING,
                stirrup_spacing_min_in=sp.STIRRUP_MIN_SPACING,
                stirrup_spacing_step_in=sp.STIRRUP_SPACING_STEP,
            ),
            materials=MaterialConfig(
                fc_col_ksi=sp.FC_COL_KSI,
                fc_beam_ksi=sp.FC_BEAM_KSI,
                fy_ksi=sp.FY_KSI,
                es_ksi=sp.ES_KSI,
            ),
            sections=SectionConfig(
                b_col_in=sp.B_COL,
                h_col_in=sp.H_COL,
                b_beam_in=sp.B_BEAM,
                h_beam_in=sp.H_BEAM,
            ),
            dcr=DCRTargets(
                dcr_band_lo=sp.DESIGN_DCR_MIN,
                dcr_band_hi=sp.DESIGN_DCR_MAX,
            ),
        )
