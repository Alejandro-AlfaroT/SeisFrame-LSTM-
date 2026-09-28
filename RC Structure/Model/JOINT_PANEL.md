# Diagnostic 3D joint panel

`Joint_Panel.py` supplies a finite-size, orthogonal scissors subassembly with
IMKPinching in both vertical shear planes. It is separate from `Build_Model`
and does not change the production frame. Input lengths are inches, moments
kip-in, and rotations radians. Global Z is vertical.

Follow-up disposition: see `JOINT_PANEL_REVIEW_20260923.md`. The transverse
beam test quantified the shared-core torsion coupling. This prototype remains
diagnostic; physical applicability and 3D compatibility are not established
by the passing implementation checks.

## Topology and scope

Two distinct six-DOF center nodes occupy identical coordinates. The column
core C and beam core B share UX, UY, UZ, and RZ through `equalDOF(1,2,3,6)`.
An explicitly oriented zeroLength element connects C to B, with IMKPinching
in direction 4 (global Rx) and direction 5 (global Ry). Direction 6 would be
Rz for this orientation; it is not the vertical-plane bending coordinate.

Six face nodes lie at x = +/-dx/2, y = +/-dy/2, z = +/-hz/2 from the center.
Column faces connect to C; the four horizontal beam faces connect to B with
`rigidLink('beam', core, face)`. These are kinematic constraints, not elastic
elements with artificially large stiffness. All eight nodes are initially
massless. The builder supplies no loads, fixities, or analysis settings.

This is a proposed 3D extension of a scissors idealization, not a claim that
independent planar calibrations validate a biaxially loaded RC joint. It
assumes global-axis alignment, a common panel height, small rotations, rigid
horizontal-plane distortion, and independent vertical-plane shear laws.
Axial-load interaction, biaxial strength interaction, anchorage failure, and
explicit concrete crushing are not separate mechanisms in this module.
An IMKPinching law can represent their aggregate effect only if a suitable
calibration and deformation definition establish that interpretation.

All four beam faces share the beam core's rotation. Consequently rotation
that bends one beam set is also transmitted as torsion to the perpendicular
set. This is an explicit kinematic assumption of this two-core topology;
the independent material laws do not eliminate that structural coupling.
The orthogonal beams and their torsional compatibility need a 3D benchmark
before adopting this topology in the building. These mechanical tests do not
establish equivalence to a continuum joint or the OpenSees Joint3D element.

## Work-conjugate coordinates: derivation for this topology

Define qx = theta_Bx - theta_Cx and qy = theta_By - theta_Cy. Small-rotation
rigid-arm kinematics u_face = u_core + theta_core cross r give:

    gamma_yz = (uy(z+) - uy(z-))/hz + (uz(y+) - uz(y-))/dy =  qx
    gamma_xz = (ux(z+) - ux(z-))/hz + (uz(x+) - uz(x-))/dx = -qy

The signs differ because of right-hand rotation conventions. Spring basic
forces Mx, My are conjugate to qx, qy. For a uniform rectangular shear-panel
idealization with volume V = dx*dy*hz:

    Mx =  V*tau_yz      My = -V*tau_xz
    Mx*dqx + My*dqy = V*(tau_yz*dgamma_yz + tau_xz*dgamma_xz)
    K_rotation = G_effective * V

Here G_effective is the initial slope of the chosen panel shear stress-strain
law, not automatically the uncracked concrete shear modulus. The conversion
uses the same effective geometry as that law. For horizontal panel shear
Vj = tau*(dx*dy), the corresponding moment magnitude is |M| = |Vj|*hz.
Vj is joint shear demand, not a column shear or specimen applied load copied
without joint equilibrium. Parameters from a different spring topology must
be converted through its own deformation and virtual-work definitions.

The positive branch of the Ry material describes negative gamma_xz. This
mapping matters when fitting asymmetric experimental responses.

## Explicit parameter contract

Both planes require separate `JointShearCalibration` inputs: Ke, positive and
negative rotational backbones, all active modern IMK cyclic parameters,
kappaF/kappaD, calibration ID, status, and source references. No beam or column
strength is reused as joint strength. There is no numerical joint default.
The installed material commands and parameter hashes are returned in panel
metadata. The modern equation remains E_ref = Lamda * Fy; no extra yield
rotation multiplier is introduced.

The current scope is **joint shear only**. The existing member calibration
still has `BOND_SLIP_INDICATOR = 1`. A combined shear/slip joint law must not be
added to that model without reconciling the deformation partition. Setting
that indicator to zero by itself does not establish a calibrated flexure-only
member law. The current builder rejects a combined deformation scope.

Synthetic values in `tests/test_joint_panel.py` are explicitly for mechanics
verification. In particular kappaF = kappaD = 0.5 and Lamda = 10, c = 1 do not
constitute a fit to RC joints. The user's A=10, cA=1 diagnostic approval is
not interpreted as approval of experimental or production joint parameters.

## Verification and constraints

Run from `RC Structure` with the OpenSees environment:

    python -B -m unittest discover -s tests -p test_joint_panel.py -v

The tests cover coincident distinct core nodes, finite faces, zero added mass,
input rejection, force couples, face shear reconstruction, virtual work,
rigid-body motion, axial/torsional transfer, simultaneous nonlinear pinching,
clear spans, a floor diaphragm, and a PeakOriented face hinge plus clear beam.

The combined elastic subassembly has a fixed column core, rigid arm a, joint
stiffness Kj, face hinge stiffness Kh, and clear cantilever length L, with
downward tip force P. Its checked vertical tip displacement is:

    uz_tip = -P * [(L+a)^2/Kj + L^2/Kh + L^3/(3*E*Iy)]
    M_joint = P*(L+a)       M_member_hinge = P*L

Exact fixture constraints use Lagrange multipliers and BandGeneral because
the resulting matrix is indefinite. Chained rigidLink/equalDOF constraints
must not be silently put under Plain or assumed safe under Transformation.
A separate Penalty check measures convergence to the exact fixture. This
does not select a penalty factor for a full building or change its handler.

## Before frame integration

1. Choose an applicable joint calibration and effective panel geometry by
   joint type, confinement, anchorage, axial loading, and loading direction.
2. Resolve shear/slip partition consistently with the member calibration.
3. Check the shared beam-core/torsion assumption in a full orthogonal joint
   subassembly. Attach member hinges at faces; reconcile clear lengths, member identities,
   gravity load transfer, self-weight, masses and diaphragm constraints. The
   current frame still uses its prior centerline geometry.
4. Verify one assembled design before a matched ground-motion comparison.

Implementation tests are not experimental validation. A shear-only panel
may be a useful initial model, but this module does not decide whether joint
degradation governs a code-conforming SMRF or which calibration is adequate.

## Primary references checked 2026-09-22

- OpenSees [IMKPinching command](https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/IMKPinching.html): material signature, pinching coordinates and modern reference energy.
- OpenSees [rigidLink](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html): small-rotation constraint matrix.
- OpenSees [Lagrange constraints](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/lagrangeMultipliers.html): indefinite matrix requirement.
- Golias and De Risi (2026), [full-scale exterior joint experiments](https://doi.org/10.3390/buildings16081638), Section 5: a scissors panel and separate fixed-end rotation contribution. Those low-standard exterior specimens are a topology/deformation-accounting reference, not a calibration source for this SMRF model. The two-plane extension and work mapping above are our stated idealization.

## Frame integration (2026-09-27)

At the user's decision the building now carries joint springs (`Model/Joint_Springs.py`,
`JOINT_MODEL = "imk_pinching_scissors"`), overriding the 2026-09-23 "not yet": a centreline
variant of the topology above (two coincident cores per elevated joint, no finite faces, members
keep their centreline lengths, loads and masses), IMKPinching in directions 4 and 5, calibrated
per joint from ACI 318-19 Table 18.8.4.3 strength (the design record's saved category) and
ASCE/SEI 41-17 Table 10-11 deformation, with `Mn = Vn h_b`, `K = G Aj h_b`, no cyclic
deterioration and kappaF = kappaD = 0.25. The deformation scope is `joint_shear_and_slip`: the
member hinges drop Haselton's bond-slip term (`IMK_Calibration.bond_slip_indicator`). Items 1
and 2 of the list above are therefore answered by declared, provisional choices (transcribed
table rows, a representative pinching level, slip attributed to the joint), not by experimental
calibration (the ASCE 41 rows were verified on 2026-09-27 against Table 6-9 of the Elwood et al.
2007 PEER update that produced them; an ASCE 41 "interior joint" has beams on both faces in the
plane's direction); item 3's torsion benchmark remains open and item 4's matched comparison was
run on 2026-09-27 (`outputs/diag_gm_peakoriented_joints_20260927`). The joint
keys are part of the ground-motion output identity, and `joint_springs.csv` in every NTHA output
records what each joint plane was given and the rotation it reached. This module's finite-size
subassembly and its `JointShearCalibration` contract (shear only) are unchanged.
