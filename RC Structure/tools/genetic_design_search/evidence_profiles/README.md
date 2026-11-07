**Design-evidence profiles (design_evidence_profile_v1)**

A profile is the explicit slab evidence configuration of one search run. It is passed with `--design-evidence PATH` to
`tools/genetic_design_search/run.py`. It is loaded and validated by `Design/Design_Evidence_Profile.py` and checked
against the run before anything is evaluated. Without a profile, a run keeps the unreviewed defaults:

- slab actions are computed on a single mesh;
- no engineering assertion is made;
- no slab layout is selected;
- the strong-column checks that need the developed slab stay unknown.

**What a profile can change**

| Key | Effect |
|---|---|
| `slab_refinement` | The numerical refinement plan of the slab actions: a recipe or explicit meshes, the tolerances and their basis (Design/SMRF_Slab_Refinement). It is loaded apart from the assertions, so numerical evidence can be produced while every assertion stays false. The tolerance is the declared investigation criterion, not an ACI requirement or a validated error bound. |
| `slab_action_assertions` | The engineering assertions (Design/Config.py SlabActionAssertions), every field written out. A true flag is accepted only in a reviewed profile (see the rules below). |
| `scope` | The case it covers (`case_id`, and `case_sha256`: the sha256 of the run's `--case-file`, or of its `--manifest` file for a manifest run), the analysis `profile`, the `geometry`, the slab `materials` and `loads`, and the `slab_policy` (frame live load patterning, transfer mesh, the slab thickness ladder). For assertions, `floor_sections_sha256` lists the floor section sets (Design/SMRF_Floor_Sections signatures) and `slab_thickness_in` the slab thickness the review covers, which must be a thickness of the declared ladder (`slab_policy` minimum, maximum and increment). A candidate with a floor outside them, or another slab thickness, is evaluated unasserted. The slab stage does not step past the reviewed thickness: no admissible layout there makes the candidate infeasible at the slab stage. |
| `provenance` | Who prepared the file, when (ISO date) and why. |
| `kind` | `unreviewed_example`, `test_fixture` or `reviewed`. The runner refuses a test fixture. |
| `review_status` | `unreviewed`, `under_review` or `reviewed_for_declared_scope`. |
| `review` | `reviewed_by`, `review_date`, `basis`, `decision`, and the `evidence` files reviewed (`path`, `sha256`, `description`). Each file is hashed on load; a missing file or a changed hash refuses the profile. |

**Rules checked on load**

- Every key is required at every level, no other key is accepted and no key may appear twice in one object. Use
  `null` where the schema allows it.
- Booleans must be JSON booleans. A string `"true"` is refused, and so are NaN and Infinity.
- Provenance must be non-blank and dated.
- A refinement plan needs a `slab_refinement_basis`. Its recipe name, levels (2 to 4) and shell budget (4 to
  130,000), or an explicit plan's meshes (each axis's offsets a list of numbers, or a list of such lists), are checked
  on load. The runner then resolves the plan on the seed's floors before anything is evaluated. A plan is refused
  there unless its recipe resolves to at least two levels inside the shell budget, or to all four recipe levels when a
  `local_extension` is declared; the resolved levels of each seed floor are recorded in the request
  (`design_evidence.seed_plan_resolution`).
- Any true assertion flag needs all of the following:
  - `kind` `reviewed` and `review_status` `reviewed_for_declared_scope`;
  - the review block;
  - a refinement plan;
  - `scope.floor_sections_sha256` and `scope.slab_thickness_in`;
  - `asserted_by` and `assertion_date` equal to the review's own `reviewed_by` and `review_date`.
- No approver is filled in by the code, and blank assertions may not name one.
- A partial review (some flags true) is accepted. Its flags and provenance are recorded in the evidence of the
  candidates inside its scope, but no slab layout is selected until every numerical flag is asserted.
- A PROBE stamp (the upper-case word PROBE, as in Design/Verify_Designs.PROBE) in the assertions, or anywhere in a
  review block (reviewer, basis, decision, evidence descriptions), is refused outside a test fixture, whether or not
  anything is asserted. A word such as "probed", or a name such as "Probert", is not a stamp.
- A PROBE-derived refinement plan (stamped, or numerically the plan of Design/Config.py PROBE_SLAB_REFINEMENT) must say
  in its basis that its tolerance is an investigation criterion.
- Nothing is inherited from Design/Verify_Designs.probe_config.
- The scope must equal the run's case id, case hash (as above), analysis profile, geometry, slab steel grade, concrete unit
  weight, floor live load, superimposed dead load and slab policy.

**Identity**

The profile enters the run before any evaluation:

- Its refinement plan, assertions and covered floor sections become part of the run's DesignConfig, which is inside
  the evaluation identity.
- Its full summary goes into the request (`design_evidence`); the generation_v3 basis carries its profile id, kind,
  review status, whether it asserts, and `identity_sha256`. `identity_sha256` covers every field, and the evidence files
  through their recorded hashes.

A changed plan, assertion, review or scope is therefore another identity. The cache, the accepted-design registry and
any checkpoint made under the old one do not carry over. A resume under another profile is refused. `--plan-only`
prints the request with the profile's summary.

**Files here**

- `case_0007_slab_refinement_unreviewed.json` is an unreviewed example. For case_0007 it declares the graded-face
  refinement plan used by the 2026-11-01 validation bundle, with every assertion false. It produces numerical slab
  evidence only, and it is not a production-approved profile.
- A clearly marked test fixture lives with the tests: `RC Structure/tests/fixtures/design_evidence_test_fixture.json`.
  It is loadable only by tests and refused by the runner.

No reviewed profile exists. A reviewed profile records an engineering decision taken elsewhere, for the scope it names.
It is never a blanket declaration that the slab evidence is verified. The two-way shear path, slab-column local minimum
steel, fire resistance, the floor and frame idealization, and independent verification remain separate review items.
