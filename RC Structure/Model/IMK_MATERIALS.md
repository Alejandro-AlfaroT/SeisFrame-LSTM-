# Modern IMK material inputs

The rotational-material adapter in `IMK_Materials.py` uses modern OpenSees
`E_ref,m = Lamda_m * Fy_positive`. The reference strength is the positive
input strength for BOTH loading directions, for every active deterioration
mode. For these springs Fy is a moment in kip-in and the
deformation coordinate is rotation in rad. No yield-rotation multiplier is
applied. Paper coefficients normalized by yield energy must be reconciled
outside this adapter before being supplied as command-level Lamda values.

`RotationalBackbone` holds positive magnitudes separately for each direction:
`dp, dpc, du, fy, fmax_fy, fres_fy`. Ke is shared by both directions. The
strength ratios are not hardening slopes. `CyclicParameters` provides the
material-specific deterioration and pinching arguments explicitly.

`define_rotational_imk` validates and installs one material, then returns its
exact command, units, OpenSees version, parameter SHA-256 and provenance.
Provenance requires `calibration_id`, `status` and `deformation_scope`; an
experimental set should also identify its paper, specimen, table/fit,
loading domain and deformation mapping. Recorded status does not establish
scientific validity or production acceptance.

## Current transition

- The member builder supports IMKBilin and IMKPeakOriented with separate
  signatures. The historical `_define_imk_peak_material` function name is
  retained for compatibility. Installed inputs are stored per end and axis
  under the hinge registry's `installed_materials` field.
- The production frame material is IMKPeakOriented with Haselton Eq. 3.20
  deterioration (S and C per member; A and K suppressed) since 2026-09-24;
  IMKBilin remains selectable for legacy reproduction. The fixed-design
  diagnostic runner defaults to the production material and accepts
  `--member-material IMKBilin`. Under the direct deterioration convention
  only, its PeakOriented profile selects Lamda_A = 10 and c_A = 1, approved
  on 2026-09-22 as provisional diagnostic inputs. The S/C/K/A constants are
  defaults, not experimental calibration. Since 2026-09-26 the deterioration
  mode, the backbone source and the installed policy are output identity keys.
- Measured OpenSees 3.8 reloading rules, pinned by
  `tests/test_imk_peak_oriented_verification.py`: after a full reversal
  IMKPeakOriented reloads on a straight line from the zero-force crossing to
  the previous peak (deteriorated), whereas IMKBilin reloads at Ke until it
  meets the backbone. IMKPinching first heads to a break point at rotation
  (1 - kappaD) x the permanent rotation left after unloading from the previous
  peak (its zero-moment crossing), at kappaF times the force of that straight
  line there, then continues to the peak (measured 2026-09-27 with springs whose
  yield rotation is 0.5 % to 50 % of the peak; the two definitions coincide only
  for a stiff spring).
- IMKPinching carries joint shear and slip in the frame since 2026-09-27
  (user decision): `Model/Joint_Springs.py` installs, at every elevated joint,
  a beam core coincident with the joint node and one zeroLength with
  IMKPinching in the two vertical shear planes (the centreline variant of the
  `Joint_Panel.py` scissors topology; `JOINT_MODEL`, `JOINT_DEFORMATION_SCOPE`
  in Structure_Parameters). Strength from ACI 318-19 Table 18.8.4.3 (the
  design record's joint-shear category), deformation from ASCE/SEI 41-17
  Table 10-11 (conforming rows, verified 2026-09-27 against the Elwood et al. 2007 PEER update that
  produced them), Mn = Vn h_b and K = G Aj h_b, no cyclic
  deterioration, kappaF = kappaD = 0.25 (Ibarra, Medina & Krawinkler 2005, a
  representative level, not an RC joint fit). Under the shear-and-slip scope
  the member hinges drop Haselton's bond-slip term (a_sl = 0) so slip is
  counted once. Status: provisional, not experimentally calibrated. The
  finite-size `Joint_Panel.py` subassembly stays a diagnostic prototype; the
  2026-09-23 review's shared-core torsion benchmark is still owed. Fixture
  kappaF = kappaD = 0.5 is not a joint value.
- For PeakOriented/Pinching, the hysteresis diagnostic uses virgin-envelope
  exceedance to identify yielded loops. This may miss degraded reversal
  yielding below the original envelope. Accumulated plastic rotation is
  unavailable: a reduced reloading tangent is not a valid plastic-strain
  decomposition. The raw work integral is reported alongside an initial-Ke
  storage-corrected energy estimate; the latter is not exact dissipation
  under stiffness degradation.
- Global output identity includes the modern energy convention and cyclic
  parameters, so output from different deterioration settings cannot be
  silently reused in the same output directory.

## Verification

### Explicit-energy mapping (2026-09-24)

The user requires **experimentally supported reference-energy capacities
before enabling the corrected structural member model**. The default remains
`IMK_ENERGY_MAPPING_MODE = "legacy_unmapped"`, with an empty
`IMK_MEMBER_ENERGY_CALIBRATIONS`. This preserves the historical diagnostic
baseline, including its known coordinate-dependent energy limitation. It is
not an accepted corrected model. Production release remains false.

The implemented optional mode is `explicit_reference_energy_v1`.
`define_mapped_rotational_imk` receives physical positive/negative backbones,
directional D values, explicit mode energies, physical labels and a reversal
flag. It maps the complete branches and D, then computes each command input
as `Lamda_m = E_ref,m / mapped_positive_Fy`. It does not infer energy from
hogging, sagging, their mean/maximum, or old Lambda values. S/C/K are required
for Bilin; S/C/A/K for PeakOriented and Pinching.

Member profiles are keyed by string physical element tag, then `i`/`j`, then
`y`/`z`. Every profile requires:

- `status: "experimentally_supported"`, `calibration_id`, matching
  `material_type`, and `units: "kip-in*rad"`;
- positive finite `energies_kip_in_rad` for exactly the active modes;
- `source_refs`, `specimen_ids`, `derivation`, `deformation_scope` and
  `applicability_basis`;
- `reviewed_by`, `review_date` (YYYY-MM-DD), and `review_basis`.

These fields require a documented review; code cannot establish the truth of
an experimental claim. Derivation must reconcile the published parameter's
normalization with the native reference energy and the modeled deformation
coordinate. A measured loop area is not automatically the IMK reference-energy
capacity. No real member profile is supplied by this change. Existing rotation,
D/exponent, bond-slip partition and backbone choices still need independent
evidence even after reference energies are established.

All four profiles are validated before the member builder creates hinge nodes.
The private `_verification_calibrations` argument accepts explicitly synthetic
profiles only for assembled test fixtures; normal builders never supply it.
Such installations are marked `verification_only` in the registry and material
provenance. Invalid/missing profiles never fall back to legacy Lambda.

The current corrected member mapping supports axis-aligned X/Y beams and
vertical columns. Each spring is ordered retained joint → duplicate member
node. With canonical increasing-coordinate member direction, the low end
uses the physical positive branch and the high end uses its reverse. This
depends on physical endpoint position, not just the connectivity's i/j labels.
Beam physical positive/negative in the principal flexural plane mean
hogging/sagging; real slab/anchorage strength differences are retained.

Provenance records local zeroLength directions 5/6, their signed global axes,
node order, tied DOFs, physical branch labels, energies, mapped Lambda and
calibration identity. Native directions 4/5/6 are rotation about local X/Y/Z.
The measured spring rotation is the projected relative nodal rotation; it is
not automatically member chord rotation, accumulated plastic rotation or joint
shear distortion. The deformation-partition research task remains open.

Output identity includes mapping mode and the complete member energy profiles,
and refuses reuse after either changes. The raw material adapter and separate
joint prototype remain available for explicit diagnostic inputs; no joint
Pinching calibration or frame integration is enabled here.

`python -B -m unittest discover -s tests -p test_imk_energy_mapping.py -v`

This regression exercises degrading native 3D springs for all three materials,
complete asymmetric-branch/D reversal, explicit-energy invariance, numerical
moment rescaling, assembled X/Y beams and both column bending axes with either
end fixed/reversed connectivity, calibration rejection and output identity.
Column assembly fixtures use NormDispIncr 1e-8 and independently check moment
equilibrium within 1e-5 of nominal strength; spring/beam fixtures use 1e-10.
No production solver tolerances change. Tests demonstrate implementation
behavior, not experimentally correct component parameters.

`python -B -m unittest discover -s tests -p test_imk_materials.py -v`

Real OpenSees fixtures check both backbone signs, capping and post-capping,
cyclic response, force-unit scaling, conjugate response queries, and local
directions 5/6 in all three member orientations. The existing beam-end
asymmetry test exercises both Bilin and PeakOriented in actual members.
These tests establish implementation behavior, not experimental accuracy.

Primary command references (checked 2026-09-22):

- https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKPeakOriented.html
- https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKPinching.html
- https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKBilin.html

Reference-energy source checks (2026-09-24):

- https://github.com/OpenSees/OpenSees/blob/master/SRC/material/uniaxial/IMKPeakOriented.cpp
- https://github.com/OpenSees/OpenSees/blob/master/SRC/material/uniaxial/IMKPinching.cpp
- https://github.com/OpenSees/OpenSees/blob/master/SRC/material/uniaxial/IMKBilin.cpp
