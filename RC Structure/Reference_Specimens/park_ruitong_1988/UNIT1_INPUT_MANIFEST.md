# Park and Ruitong (1988) Unit 1: source-traceable input manifest

2026-09-27, revised 2026-09-29 after Codex's Unit 1 review of 2026-09-28 (corrections marked
**[rev]**). First package of the reference reproduction Codex asked for
(`Codex/SeisFrame_Next_Calibration_Package_2026-09-27.md`). Isolated diagnostic: the model in
`pr1_unit1.py` is the official OpenSees example PR1.tcl ported to OpenSeesPy with two switchable
input corrections (measured beam steel, drawn beam confinement); nothing here touches the project
frame, no parameter has been fitted, and no readiness flag moves.

Status codes: **M** measured in the paper; **P** paper-model (a value the paper computed from its
measurements, not measured); **E** example-model (the OpenSees example's input, with or without a
paper basis); **I** inferred here; **U** unresolved.

Sources: [paper] R. Park and Dai Ruitong (1988), Bulletin NZNSEE 21(4), 255-278, DOI
10.5459/bnzsee.21.4.255-278 (scanned; read page by page from the bulletin PDF). [PR1] OpenSees
example PR1.tcl, N. Mitra, 16 Feb 2003, opensees.berkeley.edu/OpenSees/manuals/usermanual/1178.htm,
with procMKPC.tcl (1179), procRC.tcl (1180), procUniaxialPinching.tcl (1181). [PEER] Lowes, Mitra
and Altoontash (2003), PEER 2003/10, sections 4.2, 4.3, 4.4, 5. [src] OpenSees source
`SRC/material/uniaxial/BarSlipMaterial.cpp`, `SRC/element/joint/BeamColumnJoint2d.cpp`.

## 1. Geometry and supports

| input | value | status | source |
|---|---|---|---|
| column section (depth in the plane of bending x width) | 406 x 305 mm | M | paper Fig. 3 section A-A; PEER Fig. 5.1 |
| beam section (depth x width) | 457 x 229 mm | M | paper Fig. 3 section B-B |
| column pin to pin | 2473 mm (1008 + 457 + 1008) | M | paper Fig. 3; H = 2.473 m in section 2.6 |
| beam pin to column centreline | 2119 mm (1916 clear + 203) | M | paper Fig. 3; L = 2.119 m; PEER Fig. 5.1 (4238 between pins) |
| gravity load point from column centreline | 848 mm (y = 0.4, yL) | M | paper Fig. 11, section 2.6 |
| example node coordinates | nodes 1..10 as in PR1: base (0,0), column bottom (0,1008), beam nodes at y = 1236.5 (= 1008 + 457/2, **[rev]** the first edition printed 1233; the code always used 1008 + 457/2), column top (0,1465), tip (0,2473); beam nodes at x = +-203, +-848, +-2119 | E, consistent with M | PR1 node lines: BeamLengthIn 645, BeamLengthOut 1271, ColumnLengthClear 1008, JointWidth 406, JointHeight 457 |
| joint panel volume | 406 x 457 x 305 mm^3 | E | PR1 JointVolume |
| supports | base pinned (ux, uy); beam far ends vertical rollers (uy) with free rotation and free horizontal movement; tip free | M for the test rig (paper section 3.1, Fig. 12: 40 mm pins through 152 x 76 channels; column top hinge and jack) | PR1 fix lines: 1 1 1 0; 3 0 1 0; 8 0 1 0 |
| plane of the test | in-plane, cross-braced against out-of-plane instability | M | paper section 3.1 |

## 2. Reinforcement (Unit 1)

| input | value | status | source |
|---|---|---|---|
| column longitudinal | 8 HD16 (Grade 380 deformed): 3 per face + 2 intermediate mid-depth, rho_t = 1.30 % | M | paper Fig. 7 section 2-2; section 2.3 |
| column bar cover to bar centre | 43 mm | M | paper Fig. 7 (43, 160, 160, 43) |
| beam top | 5 D16 (Grade 275 deformed) in two layers, 3 + 2; rho = 1.09 % | M | paper Fig. 7 section 3-3 (5-D16, 42, 33); section 2.3 |
| beam bottom | 2 D16; rho' = 0.44 % | M | paper Fig. 7 section 3-3 (2-D16, 42); section 2.3 |
| beam bar cover to bar centre | 42 mm first layer, 33 mm between layers | M | paper Fig. 7 section 3-3 |
| bar area used in the fibre sections | 201.06 mm^2 (16 mm) | E (= pi 16^2 / 4) | PR1 CAs, BAs |
| joint core hoops (station J) | 5 sets of R12 rectangular + R8 diamond ties at 75 crs, inside the beam depth: 75 + 4 x 75 + 20 + 20 + 42 = 457 mm **[rev]** | M | paper Fig. 7 (section 1-1, "5 sets of R12 and R8 ties"; the "5 R12/R8 at 75 crs = 300" callout is this joint core); Table 3 provided Vsh = 470 kN |
| column ties next to the joint (station C1) | 8 R6(A) at 60 crs = 420 mm, both sides of the joint; rectangle 1078 + diamond 775.5 = 1853.5 mm of R6 per set **[rev]** (the first edition claimed a 300 mm column region with R12/R8 ties: withdrawn) | M | paper Fig. 7 |
| column ties toward the pins (station C2) | 2 R6(B) at 120, 3 R6(B) at 50 crs; low-moment region, columns stayed elastic | M | paper Fig. 7 |
| beam stirrups (stations B1-B4) | B1: 9 R6(B) at 80 crs (0-640 from the face); B2: 4 R6(B) at 110 (640-1080); B3: 4 R6(A) at 180 (1080-1780); B4: 5 R6(A) at 90 (1780-2230); 640 + 110 + 330 + 160 + 540 + 90 + 360 + 16 = 2246 | M | paper Fig. 7 |
| confinement input to procMKPC, column | tie spacing 60, total hoop length per set 1853.53 mm, fy 282, bar area 28.3 mm^2 (R6) = station C1 **[rev]**: the example's input is the drawing's | E = M | PR1 CTSspace, CTSlength, CTSFy, CTSarea |
| confinement input to procMKPC, beam (example) | spacing 80, hoop length 1036 mm, fy 282, area 28.3 (R6), one section for every beam element | E | PR1 BTSspace, BTSlength, BTSFy, BTSarea |
| confinement input to procMKPC, beam (case `verified_confinement_only`) **[rev]** | inner elements (face to gravity point): station B1, spacing 80, fy 366 (R6(B), Table 2(a)); outer elements: station B2, spacing 110, fy 366; hoop length 1036, area 28.3 | M (stations and fy), E (single rectangular hoop length) | paper Fig. 7, Table 2(a); `pr1_unit1.BEAM_STATIONS` |

**[rev]** Station map (Fig. 7, read with the dimension strings): (a) the example's column
confinement input is station C1, the column next to the joint; the earlier discrepancy (a) is
withdrawn. (b) The beam stirrups next to the column are R6(B) with fy = 366 MPa (Table 2(a)) where
the example uses 282 MPa; the `verified_confinement_only` and `both_corrections` cases carry the
station values (B1 inner, B2 outer). (c) The joint hoops (station J) belong to the panel and are
never smeared into a member section. None of these affects the bar-slip or panel components
directly; they affect the fibre sections' confined core.

## 3. Materials

| input | value | status | source |
|---|---|---|---|
| concrete f'c at test, Unit 1 | 45.9 MPa (age 112 days, slump 50 mm, 13 mm aggregate) | M | paper Table 1 |
| Concrete01 unconfined | fc = -45.9, ec0 = -0.002, fcu and ecu from procMKPC (-29.99 MPa, -0.004) | E | PR1 CUnconfFc, CUnconfEc; procMKPC |
| Concrete01 confined (modified Kent-Park) | beam: -47.81 MPa at -0.00208, fcu -9.56 at -0.01444; column: -49.42 at -0.00215, fcu -9.88 at -0.03217 | E (computed by the ported procMKPC from the E inputs above) | procMKPC.tcl; values printed by `pr1_unit1.build` |
| D16 beam steel, measured | fy 294, eps_y 0.00140, Es 210,400, eps_sh 0.0255, Esh 3,580, fsu 434, eps_sf 0.288 (MPa) | M | paper Table 2(b), Fig. 5 |
| Steel02 beam steel in the example | Fy 294, E 210,400, b = 0.002322 (b E = 488.5 MPa), R0 18.5, cR1 0.925, cR2 0.15, a1 0, a2 0.4, a3 0, a4 0.5 | E | PR1 Steel02 30 |
| ReinforcingSteel beam steel, cases `measured_beam_steel_only` and `both_corrections` **[rev]** | fy 294, fu 434, Es 210,400, Esh 3,580, eps_sh 0.0255, eps_ult 0.20 with declared bounds 0.15-0.25 (Fig. 5 ends at 0.055 strain; 0.288 is the fracture strain, not the strain at fsu); coupon check `steel_coupon_check.py` (plateau at 0.012 exact, stress at 0.05 within the Fig. 5 reading, bound spread 0.8 % at 0.05) | M (fy, fu, Es, Esh, eps_sh), U bounded (eps_ult) | paper Table 2(b), Fig. 5 |
| HD16 column steel, measured | fy 498, eps_y 0.00253, Es 196,600, fsu 660, eps_sf 0.198; no Esh or eps_sh printed | M (partial) | paper Table 2(c), Fig. 6 |
| Steel02 column steel in the example | Fy 498, E 196,600, b = 0.004216 (b E = 828.9 MPa), same R0/cR/a | E | PR1 Steel02 60 |
| transverse steel | R6(A) 282, R6(B) 366, R8 360, R12 283, R10 320 MPa at 0.005 strain (plain round, no yield plateau) | M | paper Table 2(a), Fig. 4 |

Discrepancy recorded (Codex package, section "Specimen reference sources"): the example's bilinear
hardening b E = 488.5 MPa (beam) is not the measured hardening tangent Esh = 3,580 MPa, and the
measured steel has a yield plateau to eps_sh = 0.0255 that Steel02 with b > 0 does not have. The
example's value is treated as a deliberate effective idealization of plateau plus hardening over the
strain range of interest; it is an E value, not M.

## 4. Bar-slip springs (BarSlip material, one per bar group and joint face)

| input | value | status | source |
|---|---|---|---|
| beam bottom bars, both faces (tags 21, 22) | fc 45.9, fy 294, Es 210,400, fu 434, Eh 488.5, db 16, ld = 406 (the joint width), nb 2, width 305, depth 457, bond "strong", type "beamBot" | E (fy, Es, fu, db, nb M; ld, width, depth M geometry; Eh E; bond flag E) | PR1 BarSlip 21/22; bs_* lines |
| beam top bars, both faces (tags 31, 32) | as above with nb 5, type "beamTop" | E | PR1 BarSlip 31/32 |
| column bars, both faces (tags 41-44) | fc 45.9, fy 498, Es 196,600, fu 660, Eh 828.9, db 16, ld = 457 (the beam depth), nb 3, width 305, depth 406, "strong", "column" | E | PR1 BarSlip 41-44; cs_* lines |
| bond strengths implied (MPa units, strong) | tau_ET = 1.8 sqrt(f'c), tau_YT = 0.4 sqrt(f'c), tau_EC = 2.2 sqrt(f'c), tau_YC = 3.7 sqrt(f'c), residual 0.15 sqrt(f'c) | E (PEER Table 4.2 gives 3.6 sqrt(f'c) for tau_YC; the source code uses 3.7) | src BarSlipMaterial.cpp; PEER p. 31 Table 4.2 |
| slip at bar yield implied, D16 in 45.9 MPa | s_y = fy^2 db / (8 tau_ET E) = 0.067 mm | I (from the E inputs and PEER Eq. 4.5a) | PEER p. 29 Eq. 4.5a |
| interface shear springs (tag 1, four of them) | Elastic, E = 1.0e10 N/mm (effectively rigid) | E; matches PEER's assumption that the interface stays elastic at closed-crack stiffness because the rig's axial restraint is unknown | PR1 "uniaxialMaterial Elastic 1 10000000000.0"; PEER p. 49 |
| bar-slip pinching and damage | rDisp 0.25, rForce 0.25, uForce from the residual bond; gammaK (0.3, 0, 0.1, 0, 0.4), gammaD (0.6, 0, 0.2, 0, 0.25), gammaF per the damage flag; gammaE 10 | E (fixed inside the material, not user inputs) | src BarSlipMaterial.cpp |

## 5. Shear panel (Pinching4 material 5)

| input | value | status | source |
|---|---|---|---|
| envelope stresses | 2.1932, 4.0872, 4.4862, 4.4862e-3 MPa, each times the joint volume (moment = tau x volume, the scissors conjugacy) | E; PEER's MCFT-based envelope for this joint (section 4.3.1) | PR1 p1..p4, JointVolume |
| envelope strains | 0.0002, 0.004465, 0.0131, 0.0269 rad | E | PR1 pEnvStnsp |
| pinching | rDisp 0.25, rForce 0.15, uForce 0.0 | E; PEER 4.3.2 (reloading at 25 % of the maximum historic strain, zero strength) | PR1 |
| degradation | gammaK (1.1336, 0, 0.1011, 0, 0.9165), gammaD (0.12, 0, 0.23, 0, 0.95), gammaF (1.11, 0, 0.319, 0, 0.125), gammaE 10, "energy" | E | PR1 |
| observed comparison basis | design joint shear stress below 0.7 sqrt(f'c) MPa; joint hoop strains at most about twice yield; joint crack 0.6 mm max | M | PEER p. 47; paper section 4.2, Fig. 17(a) |

## 6. Elements and analysis

| input | value | status | source |
|---|---|---|---|
| beam-column elements | nonlinearBeamColumn (force-based, Gauss-Lobatto): columns 5 integration points; outer beams 3; inner beams (between gravity load and joint face) 2 | E | PR1 element lines |
| geometric transformation | Linear (no P-delta); no column axial load in the test | E, consistent with M (axial load zero, paper sections 2.1, 2.6) | PR1; paper |
| joint element | beamColumnJoint 7, nodes 2, 6, 9, 5 (bottom, right, top, left, anticlockwise), materials 41 42 1 21 31 1 43 44 1 22 32 1 5 | E | PR1; src BeamColumnJoint2d.cpp |
| PEER's own model of this specimen | lumped-plasticity beam-columns (cracked-section EI, bilinear moment-rotation hinge from moment-curvature with a plastic-hinge length of half the depth) with the same joint element | P (different from the example's fibre elements) | PEER p. 48 |
| solver | ProfileSPD, Plain constraints, NormDispIncr 1e-8 / 150, Newton, RCM | E | PR1 |

## 7. Loading

| input | value | status | source |
|---|---|---|---|
| gravity | 55 kN per beam at 848 mm, applied first and held constant | M | paper Table 4 (P = 55.0 kN), section 2.6, 3.1 |
| column axial load | zero | M | paper sections 2.1, 2.6, Table 4 |
| example gravity implementation | "load 4 0 -55000 0 -const", same at node 7, one zero-increment LoadControl step, then loadConst | E; ported as a Constant time series pattern (OpenSeesPy's load command has no -const flag) | PR1; pr1_unit1.build |
| theoretical loads | V1 = 54.2 kN (first hinge, positive moment, near the gravity load point), V2 = 72.6 kN (both hinges); M1u = 54.8, M2u = 114.8 kN m; phi = 1, 0.85 f'c block, no strain hardening | P | paper Table 4, Eqs. 1-2, section 2.6 |
| test protocol | run 1-2 load-controlled to +-0.75 V2 = +-54.45 kN; Delta_y = (4/3)(Delta_y1 + Delta_y2)/2; then displacement control, two full cycles at each of mu = 2, 3, 4, 5, 6, 7 (runs 3-26) | M | paper section 3.2, Eqs. 3-4, Fig. 15; Fig. 16(a) run numbers |
| experiment protocol in the model (`--protocol experiment`) **[rev]** | the load-controlled +-54.45 kN cycle, then FIXED amplitudes 30, 30, 45, 45, 60, 60, 75, 75, 90, 90, 105, 105 mm (mu x 15); the model's own Delta_y is reported, never used; `--amplitude-shift +-3` applies the reading bound coherently | I (amplitudes audited: pixel tips within about 1.5 mm of mu x 15, `unit1_observed_digitized_v2.json`) | Fig. 16(a) pixel audit |
| paper protocol in the model (`--protocol paper`) **[rev]** | SELF-SCALED: mu x the model-derived Delta_y (11.34 mm for the example inputs, so mu = 7 reaches 79 mm, not 105); archived diagnostic, not an experiment-matched comparison | E (diagnostic) | `pr1_unit1.run_paper_protocol` |
| measured Delta_y, Unit 1 | about 15 mm | I (mu = 2 at about 30 mm and mu = 7 at about 105 mm on Fig. 16(a); the bulletin does not print the value, it is in Ref. 8) | paper Fig. 16(a); `unit1_observed_digitized.json` |
| example protocol (procRC) | displacement control only; for each entry y of peakpts (0.1, 10, 10, 30, 30, 45, 45, 60, 60, 75, 75, 90, 90, 105, 105 mm): 0 -> +y -> -y -> 0; step dU = 0.1 / 10 = 0.01 mm; NormDispIncr 1e-8 / 150; ArcLength fallback on failure | E; the 10 mm cycle stands in for the load-controlled +-0.75 V2 cycle and the mu cycles assume Delta_y = 15 mm | procRC.tcl; PR1 peakpts, increment |
| loading rate | quasi-static (300 kN MTS jack, load or displacement controlled) | M | paper section 3.1 |

Three protocols are kept in `pr1_unit1.py`: `example` (frozen control), `paper` (self-scaled
diagnostic) and `experiment` (the comparison basis) **[rev]**.

## 8. Observations the comparison is made against

**[rev]** `unit1_observed_digitized_v2.json` (pixel audit of a 400 dpi render, `digitize_fig16a.py`,
2026-09-29) supersedes the manual v1 reading: every reversal, axis crossing and unloading chord
carries its pixel coordinates and the affine axis transformation (skew retained); obscured reversals
carry an interval; cycle pairs that share one tip are marked not separable; the v1 residual targets
(35 and 42 mm) and the single unsigned pinching target (28 kN) are marked not accepted (the outer
positive unloading branches cross zero force at 41 to 59 mm); negative-side residuals for runs 8 to
22 and the per-cycle force at zero displacement are unavailable (branches merge). Retained from the
paper: Vmax = 80.3 kN (text), Vmax / V2 = 1.11, strength at the end of the test close to the
theoretical (PEER Table 5.1), first beam cover spalling at mu = 3, joint core share of the tip
displacement 9 to 14 % at high ductility (a virtual diagonal-gauge quantity: the model's
component-times-height number is NOT that measurement and is not compared to it), joint hoop strain
at most about twice yield, beam bar strains: yielding penetrated into the joint core, stress below
yield in the middle quarter of the joint (Fig. 20a), top bars in tension only, bottom bars in
tension and compression in the beam, tension only in the core.

Not available from the paper: the loops as digital data (only the scanned figure), the measured
Delta_y (Ref. 8), bar slip at the column face (strain profiles only), the joint distortion history
(only its share of the tip displacement), and the column tip force at the load-controlled cycle
beyond the target.

## 9. Unresolved and inferred items (carried into the discrepancy report)

- U: the joint hoops' contribution to the panel envelope in the example (the MCFT envelope values
  are given as numbers; the transverse steel ratio used to produce them is in PEER p. 48-49:
  horizontal ratio from the total horizontal transverse steel in the joint, vertical from the
  column's longitudinal steel; not recomputed here).
- **[rev]** resolved: the example's column confinement input is the drawing's station C1; the beam
  stations B1/B2 with fy 366 are the `verified_confinement_only` case.
- I: Delta_y = 15 mm for the example protocol and for the reconstructed experiment amplitudes (audited).
- **[rev]** E vs M: beam steel hardening is the `measured_beam_steel_only` case (ReinforcingSteel);
  the BarSlip springs keep the example's bilinear Eh = 488.5 MPa in every case (no plateau-end input).
- **[rev]** U bounded: eps_ult of the D16 steel (0.15-0.25; immaterial below 0.10 strain).
- U: PEER Table 4.2 lists tau_YC = 3.6 sqrt(f'c) MPa; the source code uses 3.7 (PSI: 43); the code is
  what runs.
