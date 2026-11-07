"""Population selection and accepted-design retention of the genetic search (2026-10-30).

Selection policy SELECTION_POLICY (research policy population_archive_semantics_v1, adapted to the production
code-check contract), an explicit starting option, not a tuned or validated rule:

  eligibility   a search parent is a valid, numerically complete record that is code-accepted or has at least one
                known failure. A record without an observed failure but with missing evidence goes to the evidence
                queue; a stopped, erroring or numerically incomplete one to the analysis-recovery queue; an invalid
                one is never a parent. Missing evidence is never a zero violation.
  tiers         code-accepted (0), complete with known failures (1), unresolved with known failures (2).
  coverage      within the unresolved tier, more known required checks is preferred only between designs with the
                same expected inventory (inventory_id); different inventories are never compared on coverage.
  progress      (maximum known exceedance, mean of the per-family maximum known exceedances), the family being the
                contract family and check id (PROGRESS_FAMILY_RULE). Compared only between designs of the same tier
                whose comparison signature is equal: same inventory, same known-check mask and the same basis (clause,
                comparison, units, branch) of every known check. Code-accepted designs never dominate one another.
  fronts        nondominated sorting under that (acyclic) dominance; a front larger than the open slots is thinned
                by fixed-domain diversity: a seeded anchor, then greedy maximum-minimum distance over the declared
                section and grade ranges (each size gene the mean of its width and depth distances, each grade gene
                its grade distance), never rescaled to the observed population.
  witnesses     every selected design keeps its local regressions against the other candidates: worsened violations,
                worsened utilizations (passing checks included), evidence lost, checks whose basis or applicability
                changed. Survival is not acceptance and not an engineering improvement.
  quarantine    a realized design observed with contradictory evidence anywhere in the history (``history``) is no
                parent and is listed in the queues (v2: v1 looked only at the candidates passed in).

The archive views are recomputed from the full evaluation history and never delete it: every distinct code-accepted
realized design (keyed by case, evaluation basis and realized identity, material dominance notwithstanding), a
bounded diversity view of that registry, the evidence queue grouped by missing source (each realized design counted
once, quarantined ones left to the quarantine view), the analysis-recovery queue, the unroutable failures (failing
rows the router's first gates keep from every operator: evidence that did not authenticate, no declared mutation
reason, incomplete trigger dependencies; later abstentions such as no legal move are counted in the run's
abstentions instead), quarantined identities with contradictory observations and descriptor collisions. Qualification and
nonlinear statuses are carried unchanged.
"""
from __future__ import annotations

from Design import Code_Check_Contract as contract
from Design.Genetic_Diagnostics import RULES, STRENGTH_SOURCE, digest, finite

SELECTION_POLICY = "seisframe_ga_selection_v2"
PROGRESS_FAMILY_RULE = "contract_family_and_check_id_v1"
DESCRIPTOR = "section_grade_fixed_catalog_range_v1"
TIERS = {"accepted": 0, "check_failed": 1, "unresolved": 2}
KNOWN = ("pass", "fail")
EVIDENCE_SOURCES = {
    "slab_action_review_missing": ("slab_action_engineering_assertion",
                                   "An engineer reviews the slab-action evidence (stages.slab.review_bundle) and records "
                                   "a named, dated assertion in Design.Config.SlabActionAssertions. The search never "
                                   "sets it."),
    "no_slab_action_evidence": ("slab_layout_selection", "No slab layout was selected for this design; the slab "
                                                         "stage must produce one before the strong-column checks."),
    "no_slab_layout": ("slab_layout_selection", "No slab layout was selected for this design."),
    "slab_bar_development_failed": ("slab_bar_development", "The slab mats are not developed; the slab design must "
                                                            "change before slab credit is defined."),
    "slab_bar_development_unknown": ("slab_bar_development", "Slab-bar development is not established."),
    "slab_evidence_stale": ("slab_evidence_currency", "The slab record was not priced on the final floor context."),
    "slab_credit_not_established": ("slab_bar_development", "Slab credit is not established by the slab record."),
    "action_inventory_invalid": ("factored_action_inventory", "The solved combination inventory is incomplete or "
                                                              "invalid; analysis must be repeated, not the sections."),
    "column_axial_outside_section_domain": ("column_axial_domain", "A column's axial load is outside its section "
                                                                   "domain; its strength check carries the failure."),
    "no_slab_in_model": ("slab_model", "The model has no slab; the developed beam-plus-slab strength is undefined."),
    "hoops_not_selected": ("upstream_hoop_selection", "The column has no selected hoops (its hoop selection or cage "
                                                      "layout failed): the capacity-shear row follows that design "
                                                      "outcome; it is not evidence to obtain separately."),
    "two_way_shear_path_assessment_required": (
        "slab_two_way_shear_assessment",
        "The floor meets the beam-supported premise (alpha_f1 l2/l1 >= 1.0 on every edge, columns at beam "
        "intersections); an engineer assesses whether that shear path replaces a slab-column two-way shear check and "
        "records it in Design.Config.SlabActionAssertions.two_way_shear_path_assessed. The search never sets it."),
    "slab_two_way_shear_not_implemented": (
        "slab_two_way_shear_design",
        "Some panel edge has alpha_f1 l2/l1 < 1.0 or a column bears on the slab: the beams do not take the whole panel "
        "shear and slab-column two-way shear (ACI 318-19 22.6) is not implemented. The ratio follows from the beam "
        "sections and the slab thickness."),
    "slab_column_local_steel_assessment_required": (
        "slab_column_local_steel_assessment",
        "The floor meets the beam-supported premise; an engineer assesses whether ACI 318-19 8.6.1.2 applies to it and "
        "records it in Design.Config.SlabActionAssertions.slab_column_local_steel_assessed. The search never sets it."),
    "slab_column_local_steel_not_implemented": (
        "slab_two_way_shear_design",
        "Without the beam-supported premise a slab-column critical section exists and its local minimum steel "
        "(ACI 318-19 8.6.1.2) is not implemented."),
}


# ---- normalized checks ---------------------------------------------------------------------------------------------
def utilization(row):
    """demand / capacity for '<=', capacity / demand for '>=' (at most 1 passes); None for an equality or bad data."""
    demand, capacity, comparison = row.get("demand"), row.get("capacity"), row.get("comparison")
    if not finite(demand) or not finite(capacity):
        return None
    if comparison == "<=" and capacity > 0:
        return demand / capacity
    if comparison == ">=" and demand > 0:
        return capacity / demand
    return None


def row_basis(row):
    details = row.get("details") or {}
    return digest([row.get("clause"), row.get("comparison"), row.get("units", ""), details.get("branch"),
                   details.get("category")])[:16]


def canonical_key(row):
    """The comparison key of one row; a beam capacity-shear row drops its governing member tag (kept as a witness)."""
    family, check_id, location = row.get("family"), row.get("id"), str(row.get("location", ""))
    if family == "joints" and check_id in ("beam_capacity_shear_left", "beam_capacity_shear_right") and "/member_" in location:
        gid, rest = location.split("/member_", 1)
        tail = rest.split("/", 1)[1] if "/" in rest else ""
        return f"{family}|{check_id}|{gid}/{tail}"
    return f"{family}|{check_id}|{location}"


def normalize_checks(rows, expected):
    """({key: {"s", "v", "u", "b"}}, inventory) of an evaluated record's compact rows and its expected inventory.

    ``s`` is the contract state (pass, fail, unknown, review, invalid, or missing for a required row that is absent),
    ``v`` the contract's dimensionless exceedance (0 unless failed), ``u`` the utilization, ``b`` the row basis.
    The inventory lists the required items: the exact and triggered conditional keys, each canonical beam
    capacity-shear key, and one detailing item per group; its id fixes what coverage may be compared against.
    """
    checks, states = {}, {}
    declared = ({(f, n, l) for f, n, l, _basis in expected.get("not_applicable", [])}
                | {(f, n, l) for f, n, l, _item in expected.get("excluded", [])})
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (row.get("family"), row.get("id"), row.get("location", "")) in declared and row.get("status") == "not_evaluated":
            continue                                  # declared not applicable, or outside the declared scope
        key = canonical_key(row)
        state = contract.classify_row(row)[0]
        entry = {"s": state, "v": contract.violation(row), "u": utilization(row), "b": row_basis(row)}
        if key in checks:
            # A duplicated identity makes the record invalid upstream; keep the worse of the two visibly.
            entry = max(checks[key], entry, key=lambda e: (e["s"] == "fail", e["v"]))
        checks[key] = entry
        states[(row.get("family"), row.get("id"), row.get("location", ""))] = state
    required = {f"{f}|{n}|{l}" for f, n, l in expected["exact"]}
    for rule in expected.get("conditional", []):
        if states.get(tuple(rule["trigger"])) == "pass":
            required |= {f"{f}|{n}|{l}" for f, n, l in rule["requires"]}
    patterns = []
    for pattern in expected.get("patterns", []):
        if "group" in pattern:
            patterns.append(f"detailing|pattern|{pattern['group']}")
        else:
            gid = pattern["prefix"].split("/member_")[0]
            for check_id in pattern["id_in"]:
                required.add(f"{pattern['family']}|{check_id}|{gid}{pattern['suffix']}")
    for key in required:
        checks.setdefault(key, {"s": "missing", "v": 0.0, "u": None, "b": None})
    governed = {(f, n) for f, n, _ in expected["exact"]}
    known_items = sum(1 for key in required if checks[key]["s"] in KNOWN)
    for item in patterns:
        gid = item.rsplit("|", 1)[1]
        members = [e for key, e in checks.items() if key.startswith("detailing|")
                   and (key.split("|", 2)[2] == gid or key.split("|", 2)[2].startswith(gid + "/"))
                   and tuple(key.split("|", 2)[:2]) not in governed]
        known_items += bool(members) and all(e["s"] in KNOWN for e in members)
    items = sorted(required) + sorted(patterns)
    not_applicable = sorted(f"{f}|{n}|{l}" for f, n, l, _basis in expected.get("not_applicable", []))
    inventory = {"inventory_id": digest({"items": items, "not_applicable": not_applicable}), "items": len(items),
                 "known_items": known_items}
    return checks, inventory


def features(genes, genome):
    """Fixed-domain descriptor: per size gene its normalized width and depth (weight 1/2 each), per grade gene its
    normalized grade, each over the declared catalog range; a dimension without range contributes nothing."""
    values, weights = [], []
    for gene, allele in zip(genes, genome):
        options = gene["options"]
        if gene["kind"] == "size":
            for dim in (0, 1):
                low, high = min(o[dim] for o in options), max(o[dim] for o in options)
                if high > low:
                    values.append((options[allele][dim] - low) / (high - low))
                    weights.append(0.5)
        else:
            low, high = min(options), max(options)
            if high > low:
                values.append((options[allele] - low) / (high - low))
                weights.append(1.0)
    return values, weights


def distance(a, b, weights):
    total = sum(weights)
    return sum(w * abs(x - y) for w, x, y in zip(weights, a["features"], b["features"])) / total if total else 0.0


# ---- audit and selection -------------------------------------------------------------------------------------------
def audit(record, run):
    """Classification, eligibility and comparison keys of one selection record within ``run`` (case and basis)."""
    bad = {"classification": "invalid", "eligible": False, "reason": "invalid_record", "tier": None}
    if record.get("case_id") != run["case_id"] or record.get("basis_id") != run["basis_id"]:
        return {**bad, "reason": "unbound_record"}
    if not record.get("id"):
        return bad
    if record.get("outcome") in ("infeasible", "error") or (record.get("valid") is True and not record.get("realized_identity")):
        # A stopped evaluation (no code check reached) or an exception: no realized design to compare or mutate.
        if record.get("classification") == "accepted":
            return {**bad, "reason": "contradictory_acceptance"}
        return {"classification": record.get("classification"), "eligible": False, "tier": None,
                "reason": "analysis_recovery_required"}
    if record.get("valid") is not True or not record.get("realized_identity"):
        return bad
    checks = record.get("checks") or {}
    for entry in checks.values():
        if entry["s"] in KNOWN and (not finite(entry["v"]) or entry["v"] < 0):
            return {**bad, "reason": "invalid_exceedance"}
    known = {key: entry for key, entry in checks.items() if entry["s"] in KNOWN}
    failures = sum(1 for e in known.values() if e["s"] == "fail")
    classification = record["classification"]
    if classification == "accepted" and (failures or record.get("numerical_complete") is not True):
        return {**bad, "reason": "contradictory_acceptance"}
    families = {}
    for key, entry in known.items():
        family = "|".join(key.split("|", 2)[:2])
        families[family] = max(families.get(family, 0.0), entry["v"])
    progress = (max((e["v"] for e in known.values()), default=0.0),
                sum(families.values()) / len(families) if families else 0.0)
    signature = digest({"inventory": record.get("inventory_id"),
                        "known": sorted((key, e["b"]) for key, e in known.items())})
    eligible = record.get("numerical_complete") is True and (classification == "accepted" or failures > 0)
    if record.get("numerical_complete") is not True:
        reason = "analysis_recovery_required"
    elif not eligible:
        reason = "evidence_acquisition_required"
    else:
        reason = "search_parent"
    return {"classification": classification, "eligible": eligible, "reason": reason, "tier": TIERS.get(classification),
            "inventory_id": record.get("inventory_id"), "known_items": record.get("known_items"),
            "known_failures": failures, "signature": signature, "progress": progress, "family_maxima": families}


def dominates(a, b):
    """The selection order between two eligible audits (a strict partial order)."""
    if a["tier"] != b["tier"]:
        return a["tier"] < b["tier"]
    if a["tier"] == TIERS["unresolved"] and a["inventory_id"] == b["inventory_id"] and a["known_items"] != b["known_items"]:
        return a["known_items"] > b["known_items"]
    if a["tier"] == TIERS["accepted"] or a["signature"] != b["signature"]:
        return False
    x, y = a["progress"], b["progress"]
    return all(i <= j for i, j in zip(x, y)) and any(i < j for i, j in zip(x, y))


def spread(candidates, count, weights, seed, anchors=()):
    """Seeded anchor, then greedy maximum-minimum distance (ties by a seeded hash), including ``anchors``."""
    remaining, selected = list(candidates), []

    def tie(c):
        return digest([seed, c["realized_identity"], c["id"]])
    while remaining and len(selected) < count:
        previous = list(anchors) + selected
        if not previous:
            winner = min(remaining, key=tie)
        else:
            winner = min(remaining, key=lambda c: (-min(distance(c, p, weights) for p in previous), tie(c)))
        selected.append(winner)
        remaining.remove(winner)
    return selected


def evidence_payload(record):
    """What two observations of one realized design must agree on (origins, ids, timings and inputs excluded)."""
    return digest({"classification": record.get("classification"), "numerical_complete": record.get("numerical_complete"),
                   "code_check_status": record.get("code_check_status"), "checks": record.get("checks"),
                   "inventory_id": record.get("inventory_id"), "quantities": record.get("quantities")})


def identity_key(record):
    return (record.get("case_id"), record.get("basis_id"), record.get("realized_identity"))


def observation_order(observation_id):
    """Observation order of an id ('o0042' before 'o10000'); other ids after, as text."""
    text = str(observation_id)
    return (0, int(text[1:]), text) if text[1:].isdigit() else (1, 0, text)


def deduplicate(records):
    """(clean, duplicates, conflicts) by exact final identity. Identical observations merge (the first observation
    kept, every origin and input identity listed); contradictory ones are quarantined. An observation id names one
    design."""
    buckets, identifiers = {}, {}
    for r in records:
        if not r.get("realized_identity"):
            continue
        key = identity_key(r)
        if r["id"] in identifiers and identifiers[r["id"]] != key:
            raise ValueError(f"observation id {r['id']} names two different designs")
        identifiers[r["id"]] = key
        buckets.setdefault(key, []).append(r)
    clean, duplicates, conflicts = [], [], []
    for key, rows in sorted(buckets.items(), key=lambda item: [str(x) for x in item[0]]):
        rows = sorted(rows, key=lambda r: observation_order(r["id"]))
        ids = [r["id"] for r in rows]
        if len({evidence_payload(r) for r in rows}) != 1:
            conflicts.append({"identity": list(key), "ids": ids,
                              "reason": "same realized design and basis, different evidence: quarantined"})
            continue
        chosen = rows[0]
        if len(rows) > 1:
            chosen = {**chosen, "origins": [o for r in rows for o in r.get("origins", [])], "duplicate_ids": ids,
                      "input_identities": sorted({r.get("input_identity") for r in rows if r.get("input_identity")})}
            duplicates.append({"kept": chosen["id"], "ids": ids})
        clean.append(chosen)
    return clean, duplicates, conflicts


def regressions(a, b, limit=20):
    """Local witnesses of ``b`` relative to ``a`` (worse violations and utilizations, evidence lost, basis or
    applicability changes), each list truncated to ``limit`` with its full count."""
    result = {"violation_worse": [], "utilization_worse": [], "evidence_lost": [], "basis_changed": [],
              "applicability_changed": []}
    ca, cb = a.get("checks") or {}, b.get("checks") or {}
    for key in sorted(set(ca) | set(cb)):
        x, y = ca.get(key), cb.get(key)
        if x is None or y is None:
            result["applicability_changed"].append({"id": key, "in": "selected" if x is None else "reference"})
            continue
        if x["s"] in KNOWN and y["s"] not in KNOWN:
            result["evidence_lost"].append({"id": key, "to": y["s"]})
        if x["s"] not in KNOWN or y["s"] not in KNOWN:
            continue
        if x["b"] != y["b"]:
            result["basis_changed"].append({"id": key})
            continue
        if y["v"] > x["v"]:
            result["violation_worse"].append({"id": key, "from": x["v"], "to": y["v"]})
        if finite(x["u"]) and finite(y["u"]) and y["u"] > x["u"]:
            result["utilization_worse"].append({"id": key, "from": x["u"], "to": y["u"]})
    counts = {name: len(items) for name, items in result.items()}
    return {"counts": counts, **{name: items[:limit] for name, items in result.items()}}


def select(records, count, run, seed, generation=0, history=None):
    """The next working population from ``records`` (selection records). Returns the decision with its fronts,
    audits, queues, duplicates, conflicts, witnesses and whether the population is underfilled.

    ``history`` (default ``records``) is every observation of the run: a candidate whose realized design any
    observation contradicts is quarantined, whether or not the contradicting observation is a candidate."""
    if type(count) is not int or count < 1:
        raise ValueError("population size must be a positive integer")
    candidate_ids = {r["id"] for r in records}
    conflicts = [c for c in deduplicate(records if history is None else history)[2] if candidate_ids & set(c["ids"])]
    quarantined = {i for c in conflicts for i in c["ids"]}
    clean, duplicates, _ = deduplicate([r for r in records if r["id"] not in quarantined])
    audits = {r["id"]: audit(r, run) for r in clean}
    eligible = [r for r in clean if audits[r["id"]]["eligible"]]
    weights = run["feature_weights"]
    selected, fronts, pool = [], [], list(eligible)
    tie_seed = [seed, generation]
    while pool and len(selected) < count:
        front = [c for c in pool if not any(dominates(audits[d["id"]], audits[c["id"]]) for d in pool if d is not c)]
        chosen = spread(front, count - len(selected), weights, tie_seed, selected)
        fronts.append({"ids": sorted(c["id"] for c in front), "selected": sorted(c["id"] for c in chosen),
                       "tiers": sorted({audits[c["id"]]["tier"] for c in front})})
        selected.extend(chosen)
        ids = {c["id"] for c in front}
        pool = [c for c in pool if c["id"] not in ids]
    witnesses = []
    for b in selected:
        for a in clean:
            if a is b or audits[a["id"]]["classification"] == "invalid":
                continue
            witness = regressions(a, b)
            if any(witness["counts"].values()):
                witnesses.append({"reference": a["id"], "selected": b["id"], **witness})
    queues = [{"id": r["id"], "reason": audits[r["id"]]["reason"]} for r in clean if not audits[r["id"]]["eligible"]]
    # Records without a realized design (stopped, raised) are never parents but stay visible in the decision.
    queues += [{"id": r["id"], "reason": audit(r, run)["reason"]} for r in records if not r.get("realized_identity")]
    queues += [{"id": i, "reason": "quarantined_conflicting_evidence"}
               for i in sorted(candidate_ids & quarantined, key=observation_order)]
    return {"policy": SELECTION_POLICY, "generation": generation, "requested": count,
            "selected_ids": [c["id"] for c in selected], "fronts": fronts,
            "audits": {k: {key: v for key, v in a.items() if key != "family_maxima"} for k, a in audits.items()},
            "queues": queues, "duplicates": duplicates, "conflicts": conflicts, "regression_witnesses": witnesses,
            "underfilled": len(selected) < count, "eligible": len(eligible),
            "survivor_is_not_acceptance": True, "structural_benefit_established": False}


# ---- archive views -------------------------------------------------------------------------------------------------
def accepted_registry(records, run):
    """Every distinct code-accepted realized design, with quarantined conflicts and descriptor collisions.

    Every observation is deduplicated first, whatever its classification: an identity observed once accepted and
    once with other evidence is contradictory and quarantined, never kept as accepted. Every accepted-classified
    observation that is not an entry is either quarantined or refused (with its reason)."""
    clean, duplicates, conflicts = deduplicate(records)
    entries, refused = [], []
    for r in records:
        if r.get("classification") == "accepted" and not r.get("realized_identity"):
            reason = audit(r, run)["reason"]
            refused.append({"id": r["id"], "duplicate_ids": [r["id"]],
                            "reason": reason if reason == "contradictory_acceptance" else "accepted_without_realized_identity"})
    for r in clean:
        if r.get("classification") != "accepted":
            continue
        a = audit(r, run)
        if a["classification"] != "accepted" or a["reason"] == "contradictory_acceptance":
            refused.append({"id": r["id"], "duplicate_ids": r.get("duplicate_ids", [r["id"]]), "reason": a["reason"]})
            continue
        entries.append({"id": r["id"], "realized_identity": r["realized_identity"], "duplicate_ids": r.get("duplicate_ids", [r["id"]]),
                        "input_identities": r.get("input_identities", [r.get("input_identity")]),
                        "genome": r["genome"], "features": r["features"], "quantities": r.get("quantities"),
                        "qualification": r.get("qualification"), "nonlinear_performance": r.get("nonlinear_performance"),
                        "review_dependencies": r.get("review_dependencies"), "origins": r.get("origins", []),
                        "artifacts": r.get("artifacts")})
    by_profile = {}
    for e in entries:
        by_profile.setdefault(tuple(e["genome"]), []).append(e["id"])
    collisions = [{"genome": list(k), "ids": v} for k, v in sorted(by_profile.items()) if len(v) > 1]
    kept = {e["id"] for e in entries}
    duplicates = [d for d in duplicates if d["kept"] in kept]
    refused.sort(key=lambda item: observation_order(item["id"]))
    return {"entries": entries, "duplicates": duplicates, "conflicts": conflicts, "refused": refused,
            "descriptor_collisions": collisions,
            "distinct_realized_designs": len(entries), "distinct_genomes": len(by_profile),
            "distinct_section_profiles": len({tuple(a for a, g in zip(e["genome"], run["gene_kinds"]) if g == "size")
                                              for e in entries}),
            "material_dominance_removes_entries": False}


def diversity_view(registry, count, run, seed):
    """A bounded, recomputed view of the registry; eviction from it erases nothing."""
    view = spread(registry["entries"], count, run["feature_weights"], [seed, "view"])
    pairs = [distance(a, b, run["feature_weights"]) for i, a in enumerate(view) for b in view[i + 1:]]
    return {"ids": [e["id"] for e in view], "size": len(view), "requested": count,
            "min_pair_distance": min(pairs) if pairs else None, "descriptor": DESCRIPTOR,
            "not_training_ready_by_code_acceptance_alone": True}


def evidence_queue(records, run):
    """Missing evidence grouped by its source (reason code, missing row or review item), with the affected checks and
    observations. An observation without a known failure is ``blocked`` (no search route applies to it). Each
    realized design counts once (its duplicate observations are listed); a quarantined one is left to the
    quarantine view, not reported as waiting for evidence."""
    groups = {}

    def add(source, action, check, rows, record, blocked, code=None):
        entry = groups.setdefault(source, {"source": source, "action": action, "reason_codes": set(), "checks": {},
                                           "observations": set(), "blocked_observations": set()})
        if code:
            entry["reason_codes"].add(code)
        entry["checks"][check] = entry["checks"].get(check, 0) + rows
        observed = record.get("duplicate_ids", [record["id"]])
        entry["observations"].update(observed)
        if blocked:
            entry["blocked_observations"].update(observed)

    for r in deduplicate(records)[0]:
        if r.get("valid") is not True or r.get("numerical_complete") is not True or r.get("classification") == "accepted":
            continue
        blocked = audit(r, run)["reason"] == "evidence_acquisition_required"
        before = sum(1 for e in groups.values() if r["id"] in e["observations"])
        unavailable = sorted(k for k, v in (r.get("quantities") or {}).items() if v is None)
        if unavailable:
            # Listed before it is the last blocker: a missing quantity is never zero and blocks every acceptance.
            add("material_quantities_unavailable",
                "A material objective is unavailable; the evaluator must report it before any design can be "
                "accepted (a missing quantity is never zero).",
                "quantities|" + ",".join(unavailable), 1, r, blocked)
        for item in r.get("unknown") or []:
            for code in item["reason_codes"]:
                source, action = EVIDENCE_SOURCES.get(code, (f"reason_code:{code}", "Inspect the producer's unknown "
                                                                                    "reason; no section change is implied."))
                if code == "unclassified":
                    source, action = f"unclassified:{item['family']}|{item['check_id']}", "The row gives no reason code."
                add(source, action, f"{item['family']}|{item['check_id']}", item["count"], r, blocked, code)
        counts = r.get("missing_counts")
        if counts is None:
            # A record without the full counts (this version's records always carry them): its item list, one source
            # per pattern description (missing_counts keys a pattern by its family and checks across groups).
            counts = {}
            for item in r.get("missing") or []:
                source = (f"pattern|{item['pattern']}" if item.get("pattern") is not None
                          else f"{item.get('family')}|{item.get('id')}")
                counts[source] = counts.get(source, 0) + 1
        for source, n in sorted(counts.items()):
            add(f"missing_row:{source}", "The evaluator did not produce a required row: an evidence or software gap.",
                source, n, r, blocked)
        if r.get("classification") == "unresolved" and sum(1 for e in groups.values() if r["id"] in e["observations"]) == before:
            # Unresolved, yet no unknown, missing or quantity item explains it: shown, never silently absent.
            add("unexplained_unresolved_record", "The record is unresolved without an unknown, missing or quantity item; "
                "inspect its code-check result.", f"code_check|{r.get('code_check_status')}", 1, r, blocked)
    rows = []
    for entry in groups.values():
        rows.append({**entry, "reason_codes": sorted(entry["reason_codes"]),
                     "observations": sorted(entry["observations"], key=observation_order),
                     "blocked_observations": sorted(entry["blocked_observations"], key=observation_order),
                     "affected_rows": sum(entry["checks"].values())})
    return sorted(rows, key=lambda e: (-len(e["blocked_observations"]), -e["affected_rows"], e["source"]))


def recovery_queue(records):
    """Observations that stopped, raised, were invalid or did not complete numerically: no structural diagnosis."""
    rows = []
    for r in records:
        if r.get("outcome") in ("infeasible", "error") or r.get("valid") is not True or r.get("numerical_complete") is not True:
            kind = ("evaluation_exception" if r.get("outcome") == "error" else
                    "evaluator_stage_stopped" if r.get("outcome") == "infeasible" else
                    "invalid_record" if r.get("valid") is not True else "numerical_incompletion")
            rows.append({"id": r["id"], "kind": kind, "stage": r.get("failed_stage") or (r.get("error") or {}).get("stage"),
                         "reason": r.get("stop_reason"), "problems": (r.get("problems") or [])[:6],
                         "incomplete_flags": r.get("incomplete_flags") or [], "artifacts": r.get("artifacts") or {},
                         "action": "analysis or software diagnosis; not a strengthening trigger"})
    return rows


UNROUTABLE_ACTIONS = {
    "invalid_evidence": "The failing row's evidence did not authenticate against its producer record (see problems): "
                        "a producer or adapter question. No section or grade step is aimed at it, and it still "
                        "blocks acceptance.",
    "no_operator": "No mutation reason is declared for this failure: reported, never strengthened blindly; it blocks "
                   "acceptance until a rule or a design change elsewhere resolves it.",
    "dependencies_incomplete": "The failure is mapped, but the record does not show the dependencies its operator "
                               "needs (listed under problems): evidence to complete before it can be routed.",
}


def unroutable_failures(records, run):
    """Failing rows the router's first gates keep from every operator, by kind and check: ``invalid_evidence`` (the
    adapter could not authenticate the row: a mapped diagnostic whose evidence is invalid, or a strength row whose
    DCR does not follow from its components or whose group has no strength evidence; a whole-row strength item is never
    authenticated, however it is marked), ``no_operator`` (no mutation
    reason declared; a strength component without one, the analysis shear, names its component) and
    ``dependencies_incomplete`` (a mapped failure whose operator dependencies the record does not show). Each blocks
    acceptance like any failure and none is a strengthening trigger. A failure that passes these gates but on which
    every operator still abstains (no legal move at the catalog's end, a target group the parent lacks, a numeric
    check that contradicts its status) is not listed: it is counted in the run's abstentions, once per routed parent.
    Each realized design counts once (its duplicate observations are listed), and ``rows`` counts failing rows, not
    diagnostics (a strength row and its components are one row)."""
    groups = {}
    for r in deduplicate(records)[0]:
        if r.get("valid") is not True or r.get("classification") == "accepted" or audit(r, run)["reason"] == "unbound_record":
            continue
        for d in (r.get("diagnostics") or {}).get("diagnostics") or []:
            if d.get("status") != "fail":
                continue
            reason, problems = d.get("reason"), list(d.get("evidence_problems") or [])
            source = d.get("source") or {}
            check = f"{source.get('family')}|{source.get('check_id')}"
            # The adapter's whole-row strength item (no component) exists only when the row did not authenticate.
            strength_row = (source.get("family"), source.get("check_id")) == STRENGTH_SOURCE and not source.get("component")
            if reason not in RULES and not strength_row:
                kind = "no_operator"
                if source.get("component"):
                    check += f"#{source['component']}"
            elif strength_row or d.get("evidence_valid") is not True:
                kind = "invalid_evidence"            # a whole-row strength item is unauthenticated, however marked
            else:
                missing = [k for k in RULES[reason][1] if (d.get("dependencies") or {}).get(k) is not True]
                if not missing:
                    continue
                kind, problems = "dependencies_incomplete", [f"dependency not shown: {k}" for k in missing]
            entry = groups.setdefault((kind, check), {"kind": kind, "check": check, "action": UNROUTABLE_ACTIONS[kind],
                                                      "problems": set(), "rows": set(), "observations": set()})
            entry["rows"].add((r["id"], source.get("location")))
            entry["problems"].update(problems)
            entry["observations"].update(r.get("duplicate_ids", [r["id"]]))
    rows = [{**e, "rows": len(e["rows"]), "problems": sorted(e["problems"])[:8], "problem_count": len(e["problems"]),
             "observations": sorted(e["observations"], key=observation_order)} for e in groups.values()]
    return sorted(rows, key=lambda e: (-len(e["observations"]), -e["rows"], e["kind"], e["check"]))


def views(records, run, seed, view_size):
    registry = accepted_registry(records, run)
    return {"policy": SELECTION_POLICY, "accepted_registry": registry,
            "diversity_view": diversity_view(registry, view_size, run, seed),
            "evidence_queue": evidence_queue(records, run), "analysis_recovery_queue": recovery_queue(records),
            "unroutable_failures": unroutable_failures(records, run), "quarantine": registry["conflicts"]}
