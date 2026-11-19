"""Frozen claim/evidence pairs and metrics for semantic support evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SEMANTIC_DATASET = Path(__file__).parent / "fixtures" / "semantic_claims.v1.json"
LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "insufficient_evidence",
)


class EvidenceQuote(BaseModel):
    evidence_id: str = Field(description="ID of one supplied evidence record.")
    quote: str = Field(description="Short exact substring copied from that evidence record.")


class SemanticReview(BaseModel):
    verdict: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "insufficient_evidence",
    ]
    evidence_quotes: list[EvidenceQuote] = Field(
        default_factory=list,
        description="Exact evidence excerpts supporting the verdict; empty only when no relevant evidence exists.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information needed to support or refute the claim but absent from the evidence.",
    )
    reasoning: str = Field(description="Brief explanation limited to the supplied claim and evidence.")
    confidence: float = Field(ge=0, le=1, description="Confidence in this support-relation verdict.")


def load_semantic_dataset(path=SEMANTIC_DATASET):
    raw = Path(path).read_bytes()
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 1 or dataset.get("kind") != "semantic_claim_support":
        raise ValueError("Expected version-1 semantic claim-support dataset")
    catalog = {item["id"]: item for item in dataset.get("evidence_catalog", [])}
    if len(catalog) != len(dataset.get("evidence_catalog", [])) or not catalog:
        raise ValueError("Evidence IDs must be unique and non-empty")
    for item in catalog.values():
        date.fromisoformat(item["available_at"])
        if not item.get("content") or not item.get("source") or not item.get("ticker"):
            raise ValueError("Evidence metadata and content are required")
    seen = set()
    labels = Counter()
    for case in dataset.get("cases", []):
        if case["id"] in seen:
            raise ValueError("Duplicate semantic case ID")
        seen.add(case["id"])
        date.fromisoformat(case["as_of"])
        if case["label"] not in LABELS:
            raise ValueError("Invalid semantic label")
        if not case.get("claim") or not case.get("rationale"):
            raise ValueError("Claim and proxy-label rationale are required")
        if not set(case.get("evidence_ids", [])) <= set(catalog):
            raise ValueError("Semantic case references unknown evidence")
        if any(catalog[evidence_id]["available_at"] > case["as_of"]
               for evidence_id in case.get("evidence_ids", [])):
            raise ValueError("Semantic case includes future evidence")
        labels[case["label"]] += 1
    if len(seen) != 40 or labels != Counter(dict.fromkeys(LABELS, 8)):
        raise ValueError("Semantic dataset must contain eight cases per label")
    return dataset, hashlib.sha256(raw).hexdigest()


def case_evidence(case, dataset):
    catalog = {item["id"]: item for item in dataset["evidence_catalog"]}
    return [catalog[evidence_id] for evidence_id in case.get("evidence_ids", [])]


def semantic_prompt(case, dataset):
    evidence = case_evidence(case, dataset)
    evidence_text = "\n\n".join(
        f"[{item['id']}] available_at={item['available_at']}\n{item['content']}"
        for item in evidence
    ) or "No evidence records were supplied."
    definitions = "\n".join(
        f"- {label}: {dataset['label_definitions'][label]}" for label in LABELS
    )
    return f"""You are an evidence-support reviewer, not an investment analyst.

Judge only whether the supplied evidence supports the exact claim as written.
Do not use outside knowledge, infer missing facts, search, or offer a trading recommendation.

Verdict definitions:
{definitions}

Analysis date: {case['as_of']}
Claim: {case['claim']}

Evidence:
{evidence_text}

Return one structured verdict. Evidence quotes must be short exact substrings of the supplied records and must use only their IDs. A valid source or evidence ID does not by itself prove support. If wording adds a stronger qualifier, causal claim, forecast, guarantee, or investment implication, judge that added meaning explicitly."""


def validate_semantic_review(review: SemanticReview, case, dataset):
    evidence = {item["id"]: item for item in case_evidence(case, dataset)}
    issues = []
    for quote in review.evidence_quotes:
        record = evidence.get(quote.evidence_id)
        if record is None:
            issues.append(f"unknown_evidence_id:{quote.evidence_id}")
        elif not quote.quote.strip() or quote.quote.strip() not in record["content"]:
            issues.append(f"quote_not_literal:{quote.evidence_id}")
    if review.verdict != "insufficient_evidence" and evidence and not review.evidence_quotes:
        issues.append("missing_evidence_quote")
    return issues


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _percentile(values, proportion):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def semantic_metrics(records, *, input_price=None, output_price=None):
    completed = [record for record in records if record["status"] == "completed"]
    primary = [record for record in completed if record.get("repetition", 1) == 1]
    confusion = {actual: dict.fromkeys(LABELS, 0) for actual in LABELS}
    for record in primary:
        confusion[record["expected"]][record["predicted"]] += 1
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][predicted] for predicted in LABELS if predicted != label)
        precision = _fraction(tp, tp + fp)
        recall = _fraction(tp, tp + fn)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision is not None and recall is not None and precision + recall else None),
            "support": sum(confusion[label].values()),
        }
    risky = {"unsupported", "contradicted"}
    risky_cases = [record for record in primary if record["expected"] in risky]
    risky_detected = sum(record["predicted"] in risky for record in risky_cases)
    by_case = defaultdict(list)
    for record in completed:
        by_case[record["case_id"]].append(record["predicted"])
    repeated = [predictions for predictions in by_case.values() if len(predictions) > 1]
    consistent = sum(len(set(predictions)) == 1 for predictions in repeated)
    usages = [record["usage"] for record in records]
    input_tokens = sum(usage.get("input_tokens", 0) for usage in usages)
    output_tokens = sum(usage.get("output_tokens", 0) for usage in usages)
    cost = None
    if input_price is not None and output_price is not None and all(
        usage.get("token_usage_complete") for usage in usages
    ):
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return {
        "runs": len(records),
        "completed": len(completed),
        "primary_runs": sum(record.get("repetition", 1) == 1 for record in records),
        "primary_completed": len(primary),
        "structured_success_rate": _fraction(len(completed), len(records)),
        "primary_structured_success_rate": _fraction(
            len(primary), sum(record.get("repetition", 1) == 1 for record in records)
        ),
        "accuracy": _fraction(sum(record["expected"] == record["predicted"] for record in primary),
                              len(primary)),
        "confusion_matrix": confusion,
        "per_label": per_label,
        "supported_precision": per_label["supported"]["precision"],
        "unsupported_or_contradicted_recall": _fraction(risky_detected, len(risky_cases)),
        "repeat_consistency": _fraction(consistent, len(repeated)),
        "repeated_case_count": len(repeated),
        "latency_p50_seconds": _percentile([record["seconds"] for record in completed], 0.5),
        "latency_p95_seconds": _percentile([record["seconds"] for record in completed], 0.95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_usage_complete": bool(usages) and all(
            usage.get("token_usage_complete") for usage in usages
        ),
        "estimated_cost_usd": cost,
        "label_source": "Codex expert-proxy labels; not independent financial-expert consensus.",
    }
