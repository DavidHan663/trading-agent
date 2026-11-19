"""Fresh semantic holdout v5 with explicit multi-label adjudication rules."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from tradingagents.evaluation.semantic_v2 import (
    DECISIONS,
    REASON_FLAGS,
    semantic_v2_prompt,
)

SEMANTIC_V5_DATASET = Path(__file__).parent / "fixtures" / "semantic_claims.v5.json"
SEMANTIC_V5_SHA256 = "1ad1b4fd50eeb9953cff07ae733901deeded810e8b79fc6fb3ff7f078bd622fb"


def load_semantic_v5_dataset(path=SEMANTIC_V5_DATASET):
    raw = Path(path).read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if fingerprint != SEMANTIC_V5_SHA256:
        raise ValueError("Version-5 holdout hash mismatch; dataset is frozen")
    dataset = json.loads(raw)
    if (dataset.get("schema_version") != 5
            or dataset.get("kind") != "blinded_hierarchical_semantic_claim_support"):
        raise ValueError("Expected version-5 blinded semantic holdout")
    if set(dataset.get("decision_definitions", {})) != set(DECISIONS):
        raise ValueError("Decision definitions must cover allow and review")
    if set(dataset.get("reason_definitions", {})) != set(REASON_FLAGS):
        raise ValueError("Reason definitions must cover all review reasons")

    catalog_items = dataset.get("evidence_catalog", [])
    catalog = {item["id"]: item for item in catalog_items}
    if len(catalog) != len(catalog_items) or not catalog:
        raise ValueError("Evidence IDs must be unique and non-empty")
    for item in catalog.values():
        date.fromisoformat(item["available_at"])
        if not all(item.get(field) for field in ("content", "source", "ticker")):
            raise ValueError("Evidence metadata and content are required")

    case_ids = set()
    claims = set()
    decisions = Counter()
    reasons = Counter()
    multi_reason_cases = 0
    for case in dataset.get("cases", []):
        if case["id"] in case_ids:
            raise ValueError("Duplicate version-5 case ID")
        case_ids.add(case["id"])
        normalized_claim = " ".join(case.get("claim", "").casefold().split())
        if not normalized_claim or normalized_claim in claims:
            raise ValueError("Version-5 claims must be non-empty and unique")
        claims.add(normalized_claim)
        date.fromisoformat(case["as_of"])

        decision = case.get("gate_decision")
        flags = case.get("reason_flags", [])
        if (decision not in DECISIONS or len(flags) != len(set(flags))
                or not set(flags) <= set(REASON_FLAGS)):
            raise ValueError("Invalid version-5 decision or reason flags")
        if (decision == "allow" and flags) or (decision == "review" and not flags):
            raise ValueError("Allow must have no reasons and review must have at least one")
        evidence_ids = case.get("evidence_ids", [])
        if not case.get("rationale") or not evidence_ids:
            raise ValueError("Rationale and evidence are required")
        if len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids) <= set(catalog):
            raise ValueError("Case evidence IDs must be unique and known")
        for evidence_id in evidence_ids:
            evidence = catalog[evidence_id]
            if evidence["available_at"] > case["as_of"]:
                raise ValueError("Version-5 case includes future evidence")
            if evidence["ticker"] != case["ticker"]:
                raise ValueError("Evidence ticker must match claim subject")
        decisions[decision] += 1
        reasons.update(flags)
        multi_reason_cases += len(flags) > 1

    if len(case_ids) != 40 or decisions != Counter({"allow": 16, "review": 24}):
        raise ValueError("Version-5 holdout must contain 16 allow and 24 review cases")
    if any(reasons[flag] < 4 for flag in REASON_FLAGS) or multi_reason_cases < 8:
        raise ValueError("Version-5 reasons need broad and multi-label coverage")
    if not dataset.get("reason_adjudication_policy"):
        raise ValueError("Version-5 reason adjudication policy is required")
    return dataset, fingerprint


def semantic_v5_prompt(case, dataset):
    return semantic_v2_prompt(case, dataset, include_ticker=True)
