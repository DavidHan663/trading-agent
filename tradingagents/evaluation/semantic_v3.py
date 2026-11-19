"""Version-3 semantic dataset with explicit multi-label adjudication rules."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tradingagents.evaluation.semantic_v2 import (
    DECISIONS,
    REASON_FLAGS,
    load_semantic_v2_dataset,
    semantic_v2_prompt,
)

SEMANTIC_V3_DATASET = Path(__file__).parent / "fixtures" / "semantic_claims.v3.json"


def load_semantic_v3_dataset(path=SEMANTIC_V3_DATASET):
    patch_path = Path(path)
    patch_bytes = patch_path.read_bytes()
    patch = json.loads(patch_bytes)
    if patch.get("schema_version") != 3 or patch.get("kind") != "hierarchical_semantic_claim_support_overrides":
        raise ValueError("Expected version-3 semantic dataset overrides")
    parent_path = patch_path.with_name(patch["parent_dataset"])
    parent_bytes = parent_path.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    if parent_hash != patch.get("parent_dataset_sha256"):
        raise ValueError("Version-3 parent dataset hash mismatch")

    parent, _ = load_semantic_v2_dataset(parent_path)
    dataset = copy.deepcopy(parent)
    dataset.update({
        "schema_version": 3,
        "dataset_version": patch["dataset_version"],
        "annotation_status": patch["annotation_status"],
        "parent_dataset": patch["parent_dataset"],
        "parent_dataset_sha256": parent_hash,
        "reason_adjudication_policy": list(patch["reason_adjudication_policy"]),
    })
    dataset["reason_definitions"]["missing_evidence"] = (
        "Information required to establish or assess a material part of the claim is absent or explicitly unavailable. "
        "This may coexist with unsupported_inference when the claim also adds an unsupported causal, predictive, or certainty step."
    )
    dataset["reason_definitions"]["unsupported_inference"] = (
        "The claim adds a causal, predictive, certainty, valuation, or investment implication not established by evidence. "
        "This may coexist with missing_evidence when a specific required fact is absent."
    )

    cases_by_id = {case["id"]: case for case in dataset["cases"]}
    overrides = patch.get("case_overrides", [])
    override_ids = [override.get("id") for override in overrides]
    if len(override_ids) != len(set(override_ids)):
        raise ValueError("Duplicate version-3 case override")
    for override in overrides:
        case = cases_by_id.get(override.get("id"))
        if case is None:
            raise ValueError(f"Unknown version-3 case override: {override.get('id')}")
        reasons = override.get("reason_flags", [])
        if (len(reasons) != len(set(reasons)) or not set(reasons) <= set(REASON_FLAGS)
                or case["gate_decision"] not in DECISIONS or not override.get("rationale")):
            raise ValueError(f"Invalid version-3 labels for {case['id']}")
        if (case["gate_decision"] == "allow" and reasons) or (
            case["gate_decision"] == "review" and not reasons
        ):
            raise ValueError(f"Invalid version-3 reason cardinality for {case['id']}")
        case["reason_flags"] = reasons
        case["rationale"] = override["rationale"]

    canonical = json.dumps(dataset, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return dataset, hashlib.sha256(canonical).hexdigest()


def semantic_v3_prompt(case, dataset):
    return semantic_v2_prompt(case, dataset, include_ticker=True)
