"""Hierarchical allow/review evaluation for frozen claim/evidence pairs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.evaluation.semantic import EvidenceQuote

SEMANTIC_V2_DATASET = Path(__file__).parent / "fixtures" / "semantic_claims.v2.json"
DECISIONS = ("allow", "review")
REASON_FLAGS = (
    "partial_support",
    "contradiction",
    "missing_evidence",
    "unsupported_inference",
)
ReasonFlag = Literal[
    "partial_support",
    "contradiction",
    "missing_evidence",
    "unsupported_inference",
]


class HierarchicalSemanticReview(BaseModel):
    decision: Literal["allow", "review"]
    reason_flags: list[ReasonFlag] = Field(
        default_factory=list,
        description="Zero reasons for allow; one or more distinct reasons for review.",
    )
    evidence_quotes: list[EvidenceQuote] = Field(
        default_factory=list,
        description="Short exact substrings copied from supplied evidence records.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Material information absent from the supplied evidence.",
    )
    reasoning: str = Field(description="Brief explanation limited to the claim and evidence.")
    confidence: float = Field(ge=0, le=1)


def load_semantic_v2_dataset(path=SEMANTIC_V2_DATASET):
    raw = Path(path).read_bytes()
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 2 or dataset.get("kind") != "hierarchical_semantic_claim_support":
        raise ValueError("Expected version-2 hierarchical semantic dataset")
    if set(dataset.get("decision_definitions", {})) != set(DECISIONS):
        raise ValueError("Decision definitions must cover allow and review")
    if set(dataset.get("reason_definitions", {})) != set(REASON_FLAGS):
        raise ValueError("Reason definitions must cover the four review reasons")

    catalog = {item["id"]: item for item in dataset.get("evidence_catalog", [])}
    if len(catalog) != len(dataset.get("evidence_catalog", [])) or not catalog:
        raise ValueError("Evidence IDs must be unique and non-empty")
    for item in catalog.values():
        date.fromisoformat(item["available_at"])
        if not item.get("content") or not item.get("source") or not item.get("ticker"):
            raise ValueError("Evidence metadata and content are required")

    seen = set()
    decisions = Counter()
    reasons = Counter()
    multi_reason_cases = 0
    for case in dataset.get("cases", []):
        if case["id"] in seen:
            raise ValueError("Duplicate semantic-v2 case ID")
        seen.add(case["id"])
        date.fromisoformat(case["as_of"])
        decision = case.get("gate_decision")
        flags = case.get("reason_flags", [])
        if decision not in DECISIONS or len(flags) != len(set(flags)) or not set(flags) <= set(REASON_FLAGS):
            raise ValueError("Invalid semantic-v2 decision or reason flags")
        if (decision == "allow" and flags) or (decision == "review" and not flags):
            raise ValueError("Allow must have no reasons and review must have at least one")
        if not case.get("claim") or not case.get("rationale"):
            raise ValueError("Claim and proxy-label rationale are required")
        if not set(case.get("evidence_ids", [])) <= set(catalog):
            raise ValueError("Semantic-v2 case references unknown evidence")
        if any(catalog[evidence_id]["available_at"] > case["as_of"]
               for evidence_id in case.get("evidence_ids", [])):
            raise ValueError("Semantic-v2 case includes future evidence")
        decisions[decision] += 1
        reasons.update(flags)
        multi_reason_cases += len(flags) > 1

    if len(seen) != 40 or decisions != Counter({"allow": 16, "review": 24}):
        raise ValueError("Semantic-v2 dataset must contain 16 allow and 24 review cases")
    if any(reasons[flag] < 4 for flag in REASON_FLAGS) or multi_reason_cases < 8:
        raise ValueError("Semantic-v2 reasons need broad single-label and multi-label coverage")
    return dataset, hashlib.sha256(raw).hexdigest()


def case_evidence_v2(case, dataset):
    catalog = {item["id"]: item for item in dataset["evidence_catalog"]}
    return [catalog[evidence_id] for evidence_id in case.get("evidence_ids", [])]


def semantic_v2_prompt(case, dataset, *, include_ticker=False):
    evidence = case_evidence_v2(case, dataset)
    evidence_text = "\n\n".join(
        f"[{item['id']}] "
        f"{('ticker=' + item['ticker'] + ' ') if include_ticker else ''}"
        f"available_at={item['available_at']}\n{item['content']}"
        for item in evidence
    ) or "No evidence records were supplied."
    decisions = "\n".join(
        f"- {name}: {dataset['decision_definitions'][name]}" for name in DECISIONS
    )
    reasons = "\n".join(
        f"- {name}: {dataset['reason_definitions'][name]}" for name in REASON_FLAGS
    )
    return f"""You are an evidence-support gate, not an investment analyst.

Judge only the exact claim against the supplied evidence. Do not use outside knowledge,
search, infer missing facts, or offer a trading recommendation.

Gate decisions:
{decisions}

Review reasons (multi-label; select every reason that materially applies):
{reasons}

Analysis date: {case['as_of']}
Claim: {case['claim']}

Evidence:
{evidence_text}

Return one structured result. Use allow only when every material part of the claim is
directly supported or follows from transparent arithmetic over supplied values. Allow
must have no reason flags. Review must have one or more distinct reason flags. Evidence
quotes must be short exact substrings using only supplied IDs. A valid evidence ID does
not itself establish support. Strong qualifiers, causes, forecasts, guarantees, valuation
claims, and investment implications require direct support. Reasons may coexist."""


def validate_semantic_v2_review(review: HierarchicalSemanticReview, case, dataset):
    evidence = {item["id"]: item for item in case_evidence_v2(case, dataset)}
    issues = []
    if len(review.reason_flags) != len(set(review.reason_flags)):
        issues.append("duplicate_reason_flag")
    if review.decision == "allow" and review.reason_flags:
        issues.append("allow_has_reason_flags")
    if review.decision == "review" and not review.reason_flags:
        issues.append("review_missing_reason_flags")
    for quote in review.evidence_quotes:
        record = evidence.get(quote.evidence_id)
        if record is None:
            issues.append(f"unknown_evidence_id:{quote.evidence_id}")
        elif not quote.quote.strip() or quote.quote.strip() not in record["content"]:
            issues.append(f"quote_not_literal:{quote.evidence_id}")
    if review.decision == "allow" and not review.evidence_quotes:
        issues.append("allow_missing_evidence_quote")
    if "missing_evidence" in review.reason_flags and not review.missing_information:
        issues.append("missing_evidence_without_missing_information")
    return issues


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _f1(precision, recall):
    return (2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall else None)


def _percentile(values, proportion):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def semantic_v2_metrics(records, *, input_price=None, output_price=None):
    completed = [record for record in records if record["status"] == "completed"]
    primary = [record for record in completed if record.get("repetition", 1) == 1]
    primary_attempts = sum(record.get("repetition", 1) == 1 for record in records)
    confusion = {actual: dict.fromkeys(DECISIONS, 0) for actual in DECISIONS}
    for record in primary:
        confusion[record["expected_decision"]][record["predicted_decision"]] += 1

    decision_metrics = {}
    for decision in DECISIONS:
        tp = confusion[decision][decision]
        fp = sum(confusion[actual][decision] for actual in DECISIONS if actual != decision)
        fn = sum(confusion[decision][predicted] for predicted in DECISIONS if predicted != decision)
        precision = _fraction(tp, tp + fp)
        recall = _fraction(tp, tp + fn)
        decision_metrics[decision] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": sum(confusion[decision].values()),
        }

    reason_metrics = {}
    for flag in REASON_FLAGS:
        tp = sum(flag in record["expected_reasons"] and flag in record["predicted_reasons"]
                 for record in primary)
        fp = sum(flag not in record["expected_reasons"] and flag in record["predicted_reasons"]
                 for record in primary)
        fn = sum(flag in record["expected_reasons"] and flag not in record["predicted_reasons"]
                 for record in primary)
        precision = _fraction(tp, tp + fp)
        recall = _fraction(tp, tp + fn)
        reason_metrics[flag] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": tp + fn,
        }

    expected_review = [record for record in primary if record["expected_decision"] == "review"]
    exact_reason_matches = sum(
        set(record["expected_reasons"]) == set(record["predicted_reasons"])
        for record in expected_review
    )
    any_reason_matches = sum(
        bool(set(record["expected_reasons"]) & set(record["predicted_reasons"]))
        for record in expected_review
    )
    unsafe_allows = sum(
        record["expected_decision"] == "review" and record["predicted_decision"] == "allow"
        for record in primary
    )

    by_case = defaultdict(list)
    for record in completed:
        by_case[record["case_id"]].append((
            record["predicted_decision"], tuple(sorted(record["predicted_reasons"])),
        ))
    repeated = [predictions for predictions in by_case.values() if len(predictions) > 1]
    consistent = sum(len(set(predictions)) == 1 for predictions in repeated)

    usages = [record["usage"] for record in records]
    input_tokens = sum(usage.get("input_tokens", 0) for usage in usages)
    output_tokens = sum(usage.get("output_tokens", 0) for usage in usages)
    complete_usage = bool(usages) and all(usage.get("token_usage_complete") for usage in usages)
    cost = None
    if input_price is not None and output_price is not None and complete_usage:
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    return {
        "runs": len(records),
        "completed": len(completed),
        "primary_runs": primary_attempts,
        "primary_completed": len(primary),
        "structured_success_rate": _fraction(len(completed), len(records)),
        "primary_structured_success_rate": _fraction(len(primary), primary_attempts),
        "gate_accuracy": _fraction(sum(
            record["expected_decision"] == record["predicted_decision"] for record in primary
        ), len(primary)),
        "gate_confusion_matrix": confusion,
        "decision_metrics": decision_metrics,
        "allow_precision": decision_metrics["allow"]["precision"],
        "allow_recall": decision_metrics["allow"]["recall"],
        "review_precision": decision_metrics["review"]["precision"],
        "review_recall": decision_metrics["review"]["recall"],
        "unsafe_allow_count": unsafe_allows,
        "reason_metrics": reason_metrics,
        "review_reason_exact_match_rate": _fraction(exact_reason_matches, len(expected_review)),
        "review_any_correct_reason_rate": _fraction(any_reason_matches, len(expected_review)),
        "repeat_consistency": _fraction(consistent, len(repeated)),
        "repeated_case_count": len(repeated),
        "latency_p50_seconds": _percentile([record["seconds"] for record in completed], 0.5),
        "latency_p95_seconds": _percentile([record["seconds"] for record in completed], 0.95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_usage_complete": complete_usage,
        "estimated_cost_usd": cost,
        "label_source": "Codex expert-proxy labels; not independent financial-expert consensus.",
    }
