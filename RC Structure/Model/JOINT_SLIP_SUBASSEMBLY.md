# Joint slip subassembly: force/rotation map, slip ownership and provisional calibration

2026-09-27, revised the same evening after Codex's slip-calibration resolution
(`Codex/SeisFrame_Slip_Calibration_Resolution_2026-09-27.md`). Diagnostic only: the provisional bond
law, its invariant interface stiffness and its suppressed reference energies live in
`Analysis/Joint_Slip_Subassembly.py` and its outputs, never in `Model/Build_Model`. What the evening
DID change in the production model is the ownership of bar slip per member end
(`Model/Deformation_Ownership`, section 2), because the daytime frame had removed it from every
hinge without a replacement. Results for the review candidate
(`outputs/dv_local_col_one_layer_20260927/case_0002`, design record request-identity sha256
fb2f1418..., design file sha256 40f0b635...) are in
`outputs/diag_joint_slip_ownership_tol12_v2_20260927/case_0002/REVIEW_RESULTS.md` (v2: the energy
exports relabelled per Codex's next-calibration package, same numbers); the first edition
(`outputs/diag_joint_slip_subassembly_20260927`), the 1e-8-tolerance edition
(`outputs/diag_joint_slip_ownership_20260927`) and the first 1e-12 edition
(`outputs/diag_joint_slip_ownership_tol12_20260927`) are kept as records, not superseded in place.

## 0. Source revision reconciliation

Codex's package was captured with the working tree on `main` at 91afdd40 (2026-09-17), checked out
at 22:19 that evening after the branch commits; its `source_snapshot` therefore lacked
`imk_hinge_stiffness(props=)` (added in 9a587d6c) and everything the subassembly reads. The working
tree was returned to `continuation/slab-refinement-20260924` at bed537cb before any of the work
below. The material helper Codex verified (`reference_helpers/IMK_Materials.py`, sha256 7199…) is
byte-identical to the branch's `Model/IMK_Materials.py`, so nothing from the package needed porting:
`define_mapped_rotational_imk` already existed; the repair was in how the diagnostic used it.

## 1. The installed joint: what the spring's moment and rotation are

The frame joint (`Model/JOINT_PANEL.md`) is a scissors element: one column core node J and one beam
core node B at the joint centre, translations tied, one IMKPinching rotational spring per vertical
plane between them. In the x-z plane:

    q       = theta_B - theta_J                     spring rotation (strain of the zeroLength)
    q       = -gamma_xz                             panel shear strain (JOINT_PANEL.md; the sign follows
                                                    the right-hand convention, the magnitude is what matters)
    M       = tau_xz * Aj * h_b                     spring moment, conjugate to q by virtual work
    Mn      = Vn * h_b,  Vn = gamma sqrt(f'c) Aj    yield moment of the installed law
    K       = G * Aj * h_b                          elastic stiffness (JOINT_STIFFNESS_MODIFIER = 1.0)
    theta_y = Mn / K = tau_n / G                    = 0.000632 rad for the review candidate

Equilibrium at the two cores fixes what M is in terms of member actions. Every beam hinge in the
plane is attached to B and every column hinge to J (the hinge springs are zero-length, so the hinge
moments act at the cores). With no other rotational element at the cores:

    M = sum over beam faces of M_beam,face = -(sum over column faces of M_column,face)

so the spring carries the aggregate joint moment, not any individual face moment. The subassembly
checks this at every step: with the pairs recorded core-side first, the panel spring has B as its
second node and each beam hinge has B as its first, so M_panel = M_beam,L + M_beam,R; the largest
residual over the protocol is 1e-9 of the peak panel moment at the 1e-12 solver tolerance (1e-5 at
1e-8, which is how the tolerance was chosen).

Relation to horizontal joint shear. From the classical joint free body between the beam faces,
with the beam tension and compression resultants at the lever arm jd and the column shear V_col
carried across the panel:

    V_jh  = (M_b1 + M_b2) / jd - V_col
    V_col = (M_b1 + M_b2) / H_c                     inflection points at mid-height, no gravity term

    sum M_beam = V_jh * jd / (1 - jd / H_c)

At V_jh = Vn this gives the beam-moment sum at which the joint reaches its nominal shear strength.
For the review candidate (jd = 21.2 in hogging, 24.0 in sagging, average 22.6 in; H_c = 168 in;
h_b = 28 in):

    sum M_beam at Vn (free body)    43,774 kip-in
    installed Mn = Vn h_b           46,857 kip-in     ratio 1.07

The installed law therefore yields 7 % later than the free-body mapping for this section, because
h_b exceeds jd / (1 - jd / H_c). This is a mapping difference, not a strength change, and it does
not alter the conclusion below: the capacity-designed demand never reaches either value.

Demand cap. With beam hinges yielding first (strong-column design), the largest joint moment the
frame can deliver is the sum of the two beam face moments, at most 1.25 (My_hog + My_sag) =
26,585 kip-in, which is 0.567 Mn. The installed member hinges reach only 1.10 My before cyclic
deterioration, so the subassembly peaks at 0.455 Mn, and the ground-motion diagnostic at scale 2.5
reached 0.458 Mn at its worst joint. Under the installed law the panel spring is elastic at every
intensity, its rotation is 0.46 theta_y = 0.029 %, and its IMKPinching pinching branch is never
entered. Slip cannot appear in this spring because its yield threshold is a panel-shear threshold.

## 2. Where beam, column and base slip are represented: ownership per member end

Daytime state (the frame the first edition of this note described). `bond_slip_indicator()` returned
a_sl = 0 whenever `JOINT_MODEL = "imk_pinching_scissors"` and
`JOINT_DEFORMATION_SCOPE = "joint_shear_and_slip"`, with no location argument, so every hinge the
frame created lost the (1 + 0.55 a_sl) share of Haselton's theta_p (PEER 2007/03 eq. 3.10; a factor
1/1.55, -35.5 %), column bases included, while the panel spring, calibrated to shear, never yielded.
Slip was in neither place. The factor enters theta_p only; it never entered the hinge stiffness
(Ke = 20 x 6EI/L), the 0.35 EI member modifier, theta_pc or the deterioration energies, so the
switch never changed the elastic frame, as the review said.

Evening state (installed). `Model/Deformation_Ownership` gives every mechanism exactly one owner at
each member end (policy `member_hinge_unless_registered_face_interface_v1`):

    mechanism      owner                                             where recorded
    ------------------------------------------------------------------------------------------------
    flexure        the member hinge                                  hinge registry, per end
    bar slip       the member hinge (a_sl = 1, the legacy included    hinge registry, per end;
                   scope) unless a face slip interface is REGISTERED  provenance.slip_owner,
                   for that end, then the interface (a_sl = 0)        provenance.bond_slip_indicator
    panel shear    the joint spring, at floor and roof joints only    joint registry

`create_imk_member` now builds one backbone per END (`backbone_for_member(..., end=...)`), classifies
the end from the joint node it frames into (`column_base`, `floor_joint`, `roof_joint`; a fixture
member at elevation zero is a floor joint, since nothing at that end is a footing), and records the
ownership in the hinge registry (`deformation_ownership`, `backbone_by_end`, `ends_differ`). The
NTHA outputs write it as `deformation_ownership.csv`, one row per end (member, end, location, axis,
owners of the three mechanisms, member scope, interface presence and calibration id, decomposition
status, a_sl, theta_p); the output identity carries `slip_ownership_policy`.

Rules the module enforces rather than resolves silently:

- a declared joint scope is not an owner: `JOINT_DEFORMATION_SCOPE = "joint_shear_and_slip"` is
  retired and refused at build time (`validate_joint_scope`), and the default is `joint_shear_only`;
- a member hinge that excludes slip while no interface is registered is an "unowned slip" error;
- a member hinge that includes slip while an interface is registered is a "duplicate slip scope"
  error;
- a provisional interface makes the decomposition `diagnostic_unresolved`; only a reviewed
  `calibrated` interface makes it `calibrated_separate_interface`;
- the daytime state is reproducible only through a flagged override
  (`legacy_unowned_slip_removed`, status `slip_removed_without_owner_legacy_error`), which the
  subassembly's `legacy_no_slip` variant uses;
- a per-axis interface is refused (`NotImplementedError`) until a member hinge can carry separate
  backbones for rot_y and rot_z.

Member-by-member accounting of the review candidate's frame as now built (no interface is
registered in production):

    member end                       hinge theta_p         joint spring present?   bar slip owner
    -----------------------------------------------------------------------------------------------
    beam end, floor joint            a_sl = 1 (0.0904)     yes, panel-shear law    member hinge
    beam end, roof joint             a_sl = 1              yes (level "roof")      member hinge
    column end, floor or roof joint  a_sl = 1 (0.0637 at   yes, panel-shear law    member hinge
                                     nu = 0.064)
    column base                      a_sl = 1              no (fix_base_nodes)     member hinge
                                                                                   (restored)

Consequence for identity: every NTHA output written under `joint_shear_and_slip` on 2026-09-27
carries that value and a_sl = 0 in its identity and is not comparable with outputs from the evening
state; the matched frame rerun (Codex package item 4) has not been run.

## 3. Preferred topology: slip at the member faces, in series with the flexural hinge

The subassembly keeps the panel spring for shear and places a slip interface at each participating
member face, in series with that member's flexural hinge, using the same moment and rotation
reference (rotation about global Y, zeroLength strain = rotation(j) - rotation(i), i on the core
side). Two zero-length elements in series between the core and the member end give, exactly:

    M_slip = M_hinge = M_member,end                 equilibrium at the interface node (no other element)
    theta_member,end = theta_core + theta_slip + theta_hinge
    1 / K_total = 1 / K_slip + 1 / K_hinge          in the elastic range

Separate elements were used so that each component's moment-rotation pair is read directly from
its own material (`eleResponse ... material 1 stress/strain`) with no Series-material iteration
setting to verify. Opposite faces have their own interfaces, own strengths (hogging on one face is
sagging on the other at the same instant) and own histories; nothing is shared through the core
except the panel spring.

Subassembly (interior joint, one vertical plane, one story between inflection points):

    top node N_TOP (0, 0, +H/2): lateral displacement imposed; column axial load N applied
      elastic upper column (H/2)
    column hinge (frame column law)                 [optional column-face slip interface]
    J = column core --- panel spring (installed law) --- B = beam core
    beam hinge (frame beam law) --- beam slip interface --- B, on the left and right faces
      elastic half beams (L/2); far ends (+-L/2, 0, 0) on vertical rollers, free to rotate
    column hinge (frame column law)
      elastic lower column (H/2)
    base N_BASE (0, 0, -H/2): pin

Boundary conditions: out-of-plane DOFs (uy, rx, rz) fixed everywhere; base pinned; beam far ends
uz fixed, ux and ry free; top free except the imposed ux. Constraint handler Penalty (1e9, the
frame's) because the interface ties are chained. Members are elasticBeamColumn with the frame's
properties (`_member_properties`: 0.35 EI beam, the column modifier, Linear transformation, so no
P-delta). Hinges are the frame's IMKPeakOriented materials from `_define_imk_peak_material` with the
frame's backbones (`backbone_for_member(..., end=...)`, so the ownership above applies inside the
fixture too), the beam yield moments from the actual bar rows (`_beam_strength_families`), and the
frame's sign rule (hogging positive where the beam leaves the joint toward +x, end i; sagging
positive where it leaves toward -x, end j). The column carries the floor-2 interior gravity axial
load (453 kip, nu = 0.064) before the lateral protocol.

Both in-plane bending axes are run: `--axis x` uses the x beams (bay 144 in, x_interior family,
hogging 10,538 / sagging 10,730 kip-in, the x bars' rows) and `--axis y` the y beams (bay 180 in,
y_interior family, hogging 12,088 / sagging 8,568 kip-in, the y bars' rows: the physically unequal
slab/cage branches the resolution asked to keep). The square column uses its own inertia and
strength about the plane's axis.

Variants:

    installed         the frame as now built: a_sl = 1 in every hinge, panel spring, no interfaces
    slip_interfaces   beam-face interfaces registered (section 4), so those hinges drop the share
                      (a_sl = 0); decomposition diagnostic/unresolved
    legacy_no_slip    the daytime frame: a_sl = 0 everywhere and nothing replacing it (flagged override)
    --column-slip     adds column-face interfaces (unverified for through-bars; section 5)
    --reverse-springs every spring's node order reversed with the materials mapped accordingly
                      (the orientation regression, section 6)
    --suppress-member-deterioration   member hinges given suppressed energies in-process

Protocol: displacement control at N_TOP, drift levels 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0,
3.0 %, two cycles each, 0.005 % drift per step (0.0025 % for the convergence run), Newton with
NormDispIncr 1e-12 and 300 iterations, KrylovNewton retry.

## 4. Provisional slip law and the invariant interface (diagnostic only)

Source: Sezen, Lodhi, Setzler and Chowdhury (2008), 14WCEE paper S15-019, section 2.2, and Sezen
and Setzler (2008), ACI Structural Journal 105(3). Uniform elastic bond stress ub over the
development length, triangular strain profile:

    ld          = fy db / (4 ub)                    elastic development length at yield
    s_y         = eps_y fy db / (8 ub)              slip at first bar yield
    ld'         = (fs - fy) db / (4 ub')            yielded length at a bar stress fs > fy
    s_u         = s_y + (eps_s + eps_y) / 2 * ld'   slip at fs, adding the yielded length's strain
    theta_slip  = s / (d - c_y)                     tension-side slip, rotation about the neutral axis

Choices made here, all declared and all provisional:

    ub          = 12 sqrt(f'c) psi (elastic), ub' = 6 sqrt(f'c) psi (yielded), Sezen et al.
    f'c         = the concrete surrounding the anchored bar = the column concrete (4 ksi for the
                  review candidate; the beam is 8 ksi and is not used)
    fs          = 1.25 fy, the probable stress; eps_s = 0.02 at that stress (A706 assumption; not
                  measured; the resolution notes that integrating the same strain distribution
                  monotonically would give LESS slip at 1.10 fy than this endpoint interpolation)
    c_y         = elastic cracked transformed-section neutral axis with compression steel
                  (`cracked_neutral_axis`), computed per face from the actual bar rows
    d           = depth to the tension steel centroid of the actual rows
    My          = the installed nominal face moment, so the interface yields with the hinge. The
                  resolution's independent cracked-section extrapolation puts first steel yield at
                  8,190 (hogging, slab mat first) and 8,813 kip-in (sagging), below the nominal
                  10,538 / 10,730; that mismatch is recorded as evidence and NOT substituted here.

Interface stiffness, the repaired mapping (resolution section 2). The bond law gives a secant to
first bar yield per face, My / theta_slip,y, and the two faces differ because their lever arms
differ. The first prototype took Ke from whichever face the element called positive and shared it,
so the same physical interface got 5.90e6 or 6.71e6 kip-in/rad depending on node order: a 13.9 %
coordinate artefact that put the negative branch's yield rotation +13.9 % or -12.2 % off its target.
Now:

    Ke          = 2 / (1/K_hog + 1/K_sag)          the flexibility mean of the two face secants, one number,
                                                    a property of the calibration (PROVISIONAL_INTERFACE_KE_RULE)
    theta_y,b   = My,b / Ke                         each branch's own elastic yield rotation
    dp,b        = theta_slip,u,b - My,b / Ke        each branch's plastic input to its capping target;
                                                    a target inside My,b / Ke is refused, not hidden
    branches    hogging and sagging both installed, directional D_pos / D_neg mapped with them
    energies    every mode E = suppression x max(My), declared verification-only (no bond
                deterioration data), mapped as Lambda = E / Fy_positive whichever branch is positive

installed through `Model/IMK_Materials.define_mapped_rotational_imk(..., reverse=...)`, with
`reverse` true wherever the element's positive strain is physically sagging (the left beam, or any
reversed node order). Fmax / Fy = 1.25 (above the hinge's 1.10, so the interface cannot cap the
member strength), dpc = 10 dp and residual 0.2 (no data, kept far away), kappa = 0.25 (borrowed from
the joint law; an Ibarra-Medina-Krawinkler representative level, not an RC fit).

Parameters for the review candidate (floor 2 interior joint):

    quantity                             x beams: hogging / sagging      y beams: hogging / sagging
    ------------------------------------------------------------------------------------------------
    d (in)                               22.48 / 25.30                   25.30 / 22.48
    c_y (in)                             6.35 / 7.25                     (per run JSON)
    theta_slip,y target (rad)            0.00179 / 0.00160               (per run JSON)
    theta_slip,u at 1.25 fy (rad)        0.01132 / 0.01012               (per run JSON)
    My (kip-in)                          10,538 / 10,730                 12,088 / 8,568
    face secants (kip-in/rad)            5.90e6 / 6.71e6                 (per run JSON)
    interface Ke (kip-in/rad)            6.28e6                          5.87e6
    branch yield-rotation mismatch       -6.1 % / +6.9 %                 +28.9 % / -18.3 %

The y-beam mismatch is the finding that matters: an IMK material has one Ke, and where the two
faces' secants differ by the y beams' 41 % strength asymmetry no single number represents both
elastic branches; the rule is declared and the error recorded, but a slip law with a direction-
dependent elastic stiffness would need a different material form. Compare the frame hinge at the
same face: Ke = 8.10e7 kip-in/rad (rotation at My 0.013 %), theta_p = 0.0904 rad with a_sl = 1,
0.0583 without, Mc/My = 1.10, post-yield slope 0.10 My / theta_p; and the elastic half beam's end
rotation at My, My L'/(3EI) = 0.264 % (x).

Haselton comparison, not a target (resolution decision 1): the 0.55 share is 0.0321 rad; the bond
law's travel to the probable stress is 0.0101 to 0.0113 rad (x), one third of it; at first yield
0.0016 to 0.0018 rad.

## 5. Column faces and bases

The `--column-slip` run places the same law at the column faces with the column's bars (No. 11,
d = 39.3 in, c_y = 8.6 in, theta_slip,y = 0.00094 rad at My = 34,329 kip-in) and registers those
interfaces too, so the column hinges drop a_sl. At this joint the column moment is capacity-limited
to 0.31 My by the beams, so the interfaces stay elastic, rotate 0.029 % and dissipate nothing; they
do add 5 % to the drift at 0.1 % (secant 137 against 145 kip/in). The resolution's point stands:
elastic here does not mean omissible in the 3D building, and through-bars transfer force through
bond and so do not imply zero slip; the law's independent-anchorage assumption is what is unverified
for them (a #11 bar from +fy to -fy needs 55.7 in at the elastic bond stress against a 42-in joint).

Column bases are not in the subassembly. The daytime frame removed their slip share without
replacement; the ownership module restores it (a_sl = 1, `legacy_included_slip_scope`) and will
only hand it to a base interface that is registered with footing evidence (anchorage length,
termination, confinement, axial/moment history; the Zhao-Sritharan strain-penetration reference
and `Bond_SP01` are the resolution's candidates). None exists.

## 6. Separate deformation and energy outputs, and the checks

Recorded per step, every pair in the physical sign (core-side node first, whatever the element's
node order): u_top, u_J, u_z(J), u_z(top), column shear P (base reaction), theta_J, theta_B, and
for every component its own moment and rotation from its material, plus the member end moments and
the column axial force. Exports: `<label>_result.json`, `<label>_histories.npz`,
`<label>_histories.csv`, two PNGs per run, `summary.json`.

Kinematic identities (small displacements). A member-end tangent is the core rotation plus the slip
rotation plus the hinge rotation. An elastic segment loaded by M at its hinge end and moment-free at
the far end has tangent - chord = -M L'/(3EI), with M the spring's own moment (equal to the
member-end moment):

    column half:  phi = theta_J + theta_slip + theta_hinge + M L'/(3EI),   drift = (phi_T + phi_B) / 2
    beam half:    theta_B + theta_slip + theta_hinge + M L'/(3EI) = chord = -/+ u_z(J) / L'  (left / right)

The beam chord term is the core's vertical displacement (column axial shortening under N, 0.006
in) over the half bay; without it the beam identity closes only to 8e-5 rad. No sign search.
Closure: drift reconstruction 2.3e-9, beam residual 4e-9 to 6e-9 rad, core equilibrium 1e-9 to
3e-9 relative (all at the 1e-12 solver tolerance).

Energy. External work W_ext = integral P du_top + N du_z(top). Internal = sum over components of
the signed work integral M dtheta (each from its own pair) + U_elastic, with U_elastic = M^2 L'/(6EI)
per elastic segment. Balance error at most 7e-5 of W_ext (y), 6e-6 (x). The signed component work
is the reliable quantity and the only one the balance uses. Per cycle and per component the change
of the linear-elastic reference M^2 / (2 Ke) on the installed stiffness is also tabulated, and the
difference work minus that change is labelled EXACT dissipation only for a component that never
left its elastic branch (the panel and the column hinges: zero to solver precision, so a small
negative work increment there is stored energy, not negative dissipation) and an elastic-reference
ESTIMATE for a yielded IMK component, whose recoverable energy under path-dependent unloading and
internal variables is not established (Codex next-calibration package, 2026-09-27). The estimates
are not to be used to fit energy capacities.

Elastic flexibility. At the first peaks of the 0.10 % and 0.25 % levels: the secant stiffness
P / u_top and the drift decomposition through the column identity (core rotation, column hinge,
column slip, column elastic) and the beam-core rotation decomposition through the beam identity
(beam slip, beam hinge, beam elastic, chord). The flexibility relation the resolution states,
C_total = C_flexure + C_slip + C_shear at one moment and rotation reference, is what the beam
identity reads off face by face.

Orientation regression. Every spring's node order reversed (`--reverse-springs`), the frame hinges
installed with their branches swapped, the interfaces with `reverse` toggled, the same loading;
the two runs are compared component by component in the physical signs. Acceptance 1e-9 relative.
With the member hinges' deterioration suppressed in-process the fixture is invariant to 1.5e-13
(installed, both axes), 1.8e-13 (slip interfaces, x) and 1.5e-10 (slip interfaces, y); at the 1e-8
solver tolerance the slip fixture had sat at 3.6e-7 (x) and 2.6e-5 (y), which is why the tolerance
is 1e-12. The material
alone (`tests/test_joint_slip_subassembly.py`, 17,680 samples) gives zero discrepancy with the
repaired mapping and 12.2 % of the peak moment (1,632 kip-in) with the prototype's positive-selected
Ke, the negative control the resolution's package reported. With the PRODUCTION member law the
fixture is orientation-dependent: 0.4 % (x) and 6.5 to 6.9 % (y) of the peak. That is the legacy
member energy anchor E_ref = Lambda x Fy_positive of `IMK_ENERGY_MAPPING_MODE = "legacy_unmapped"`
(a beam framing into a joint from the left is end j and anchors its deterioration on sagging, from
the right end i on hogging; the y beams' 41 % strength asymmetry makes it 7 %). The corrected member
mapping exists (`explicit_reference_energy_v1`) but stays gated behind experimentally supported
member energies (user decision 2026-09-24, preserved by the resolution); the magnitude is reported
for that decision, not fixed here.

Convergence. The slip_interfaces fixture at half the drift step changes the peak shear by 7e-15 (x)
and 1e-8 (y) and the component work by less than the printed precision.

## 7. What the subassembly shows (summary; numbers in REVIEW_RESULTS.md)

1. Under the installed law the joint is elastic to 3 % drift (0.46 theta_y, 0.455 Mn) on both axes;
   the column hinges are elastic (0.31 My); all dissipation is in the beam hinges. This is a
   capacity-design bound, not an intensity effect.
2. Restoring a_sl = 1 (`installed` against `legacy_no_slip`) changes nothing within 3 % drift: the
   theta_p difference is beyond the protocol. It matters at capping, and in the frame's identity.
3. The slip interfaces yield with the hinges and do not cap strength (127.1 against 126.8 kip).
   After yield their rotation saturates near their yield rotation (0.17 to 0.21 %): the interface's
   post-yield slope (2.8e5 to 3.2e5 kip-in/rad, keyed to reaching 1.25 fy) is fifteen times the
   hinge's (1.8e4), so the incremental slip share while both ascend is Kh / (Kh + Ks) = 5.5 to
   6.1 %, and the face moment barely rises above My under Mc/My = 1.10 with cyclic deterioration.
   The interface is not capped at yield; its additional rotation is small because of its slope and
   the moment the member delivers (resolution section 1). Per interface the protocol dissipates 6 to
   7 kip-in against about 2,400 per hinge.
4. The Haselton difference is a comparison only: the law reaches one third of it at 1.25 fy.
5. The interfaces add elastic flexibility: secant stiffness 214 to 145 kip/in (x), 190 to 130 (y)
   at 0.1 % drift; first beam yield 0.355 to 0.525 % drift (x). The first edition read this as
   double counting against the 0.35 EI modifier; the resolution rejects that inference: the 0.35 is
   attributed to ACI in the source, what it contains has not been established, and restoring the
   old compliance algebraically would need 0.88 to 1.12 gross EI, which is not a recommendation. The
   partition is to be built from a flexure-only reference plus separately identified mechanisms
   (package item 3, not started here), not by raising EI.
6. The y beams expose the single-Ke limitation of the interface form (section 4).
7. Column-face interfaces are elastic here for a capacity reason and are not thereby omissible.

## 8. Not covered, and prerequisites before any frame integration

- Package item 3, the reference comparison: reproduce Park-Ruitong Unit 1 (the OpenSees PR1 example
  is a starting point, not a verification), then the project section and anchorage with a
  section-equilibrium slip model (signed bar slip s_i = u + z_i theta, N = sum F_i, M = sum F_i z_i);
  fit member and interface together; keep nominal strength, first bar yield, peak and residual
  distinct. The `BarSlip` and `Bond_SP01` materials are candidates for that reference, not for a
  rotational zeroLength.
- Package item 4, the matched frame rerun of the same design and record: not started; the
  production frame's identity changed (a_sl = 1, `joint_shear_only`, `slip_ownership_policy`).
- Exterior, edge and corner joints, cyclic bond deterioration, the kappa values, the 1.25 fy /
  eps_s = 0.02 endpoint and the post-yield form of the law, the direction-dependent elastic
  stiffness of the interface, the footing anchorage basis, and the production member energy anchor
  (section 6).

None of the above is a request to change the production model further, and no readiness flag moved.

## References

- Sezen, H., Lodhi, M.S., Setzler, E.J. and Chowdhury, S.R. (2008). Bond-slip behavior of
  reinforced concrete members. 14th World Conference on Earthquake Engineering, paper S15-019,
  section 2.2.
- Sezen, H. and Setzler, E.J. (2008). Reinforcement slip in reinforced concrete columns. ACI
  Structural Journal 105(3).
- Haselton, C.B., Liel, A.B., Taylor Lange, S. and Deierlein, G.G. (2008). PEER 2007/03, eq. 3.10,
  sections 2.1.2.2 and 3.5.4.
- Lowes, L.N., Mitra, N. and Altoontash, A. (2003). PEER 2003/10, sections 4.2 and 5 (bar slip and
  panel shear separated; Viwathanatepa and Park-Ruitong comparisons).
- Zhao, J. and Sritharan, S. (2007). Modeling of strain penetration effects in fiber-based analysis
  of reinforced concrete structures. ACI Structural Journal 104(2).
- Alath, S. and Kunnath, S.K. (1995); Celik, O.C. and Ellingwood, B.R. (2008): scissors joint
  element and its conjugacy.
- Ibarra, L.F., Medina, R.A. and Krawinkler, H. (2005): pinching parameters kappa.
- `Codex/SeisFrame_Joint_Slip_Review_2026-09-27.md`,
  `Codex/SeisFrame_Slip_Calibration_Resolution_2026-09-27.md`; `Model/JOINT_PANEL.md`;
  `Model/Deformation_Ownership.py`; `Model/IMK_Materials.py` (`define_mapped_rotational_imk`).

## 9. Domain scope of the slip-interface registry (2026-09-29)

Codex's Unit 1 review (2026-09-28) reproduced a lifecycle defect: a registration left from a previous
model survived `build_model()`'s `ops.wipe()` and took bar slip away from a member hinge in a frame that
installed no replacement interface. The registry (`Model/Deformation_Ownership`) is now scoped to one
OpenSees domain: every builder calls `begin_domain()` right after its wipe and before any member exists
(discarded stale registrations are counted and listed in the domain record), a registration names the
element that realises the interface and is honoured only while that element is in the domain, and
`validate_installed()` closes the build (the production builder passes `expect_none=True`; this
subassembly registers and installs inside the same `build()` and validates before returning).
Registrations are never cleared in the middle of installing an interface assembly. Tests:
`tests/test_deformation_ownership.py::DomainLifecycle`.
