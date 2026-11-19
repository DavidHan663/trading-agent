"""Fresh, hash-frozen semantic holdout, authored before any v4 model run."""

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

SEMANTIC_V4_DATASET = Path(__file__).parent / "fixtures" / "semantic_claims.v4.json"
SEMANTIC_V4_SHA256 = "cacba43eac47ae4babe6e7ceec2653dd6b35e23eed82d6ccfa60aa004933520f"


def load_semantic_v4_dataset(path=SEMANTIC_V4_DATASET):
    raw = Path(path).read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if fingerprint != SEMANTIC_V4_SHA256:
        raise ValueError("Version-4 holdout hash mismatch; dataset is frozen")
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 4 or dataset.get("kind") != "blinded_hierarchical_semantic_claim_support":
        raise ValueError("Expected version-4 blinded semantic holdout")
    if set(dataset.get("decision_definitions", {})) != set(DECISIONS):
        raise ValueError("Decision definitions must cover allow and review")
    if set(dataset.get("reason_definitions", {})) != set(REASON_FLAGS):
        raise ValueError("Reason definitions must cover all review reasons")

    catalog = {item["id"]: item for item in dataset.get("evidence_catalog", [])}
    if len(catalog) != len(dataset.get("evidence_catalog", [])) or not catalog:
        raise ValueError("Evidence IDs must be unique and non-empty")
    for item in catalog.values():
        date.fromisoformat(item["available_at"])
        if not all(item.get(field) for field in ("content", "source", "ticker")):
            raise ValueError("Evidence metadata and content are required")

    ids = set()
    decisions = Counter()
    reasons = Counter()
    multilabel = 0
    for case in dataset.get("cases", []):
        if case["id"] in ids:
            raise ValueError("Duplicate holdout case ID")
        ids.add(case["id"])
        date.fromisoformat(case["as_of"])
        flags = case.get("reason_flags", [])
        decision = case.get("gate_decision")
        if (decision not in DECISIONS or len(flags) != len(set(flags))
                or not set(flags) <= set(REASON_FLAGS)):
            raise ValueError("Invalid holdout decision or reason flags")
        if (decision == "allow" and flags) or (decision == "review" and not flags):
            raise ValueError("Allow must have no reasons and review must have at least one")
        evidence_ids = case.get("evidence_ids", [])
        if not case.get("claim") or not case.get("rationale") or not evidence_ids:
            raise ValueError("Claim, rationale, and evidence are required")
        if not set(evidence_ids) <= set(catalog):
            raise ValueError("Holdout case references unknown evidence")
        if any(catalog[evidence_id]["available_at"] > case["as_of"]
               for evidence_id in evidence_ids):
            raise ValueError("Holdout case includes future evidence")
        if any(catalog[evidence_id]["ticker"] != case["ticker"]
               for evidence_id in evidence_ids):
            raise ValueError("Holdout evidence ticker must match claim subject")
        decisions[decision] += 1
        reasons.update(flags)
        multilabel += len(flags) > 1

    if len(ids) != 40 or decisions != Counter({"allow": 16, "review": 24}):
        raise ValueError("Version-4 holdout must contain 16 allow and 24 review cases")
    if any(reasons[flag] < 4 for flag in REASON_FLAGS) or multilabel < 8:
        raise ValueError("Holdout reasons need broad single-label and multi-label coverage")
    return dataset, fingerprint


def semantic_v4_prompt(case, dataset):
    return semantic_v2_prompt(case, dataset, include_ticker=True)
