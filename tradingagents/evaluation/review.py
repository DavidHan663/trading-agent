"""Aggregate human-reviewed judgments; unreviewed runs have no quality score."""

import math
from collections import Counter


def review_template(case, repetition):
    return {"case_id": case["id"], "repetition": repetition, "reviewed": False,
            "reviewer": "", "claim_count": None, "supported_claim_count": None,
            "citation_count": None, "correct_citation_count": None,
            "reference_fact_count": len(case["reference"]["facts"]), "covered_fact_count": None,
            "predicted_insufficient": None,
            "expected_insufficient": case["reference"]["evidence_status"] == "insufficient",
            "unjustified_direction": None, "notes": ""}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, proportion):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def aggregate(records, reviews):
    eligible = {(r["case_id"], r["repetition"]): r for r in records
                if r["mode"] == "model" and r["status"] == "completed"}
    seen = set()
    totals = Counter()
    reviewed = []
    for item in reviews:
        key = item["case_id"], item["repetition"]
        if key in seen:
            raise ValueError("Duplicate review")
        seen.add(key)
        if not item["reviewed"]:
            continue
        if key not in eligible:
            raise ValueError("Only completed real-model runs may receive quality reviews")
        if not item["reviewer"].strip():
            raise ValueError("Reviewed items require a reviewer")
        record = eligible[key]
        if (item["expected_insufficient"] != record["expected_insufficient"]
                or item["reference_fact_count"] != record["reference_fact_count"]):
            raise ValueError("Reference labels must not be changed during review")
        for count, correct in (("claim_count", "supported_claim_count"),
                               ("citation_count", "correct_citation_count"),
                               ("reference_fact_count", "covered_fact_count")):
            if any(type(item[name]) is not int for name in (count, correct)):
                raise ValueError("Review counts must be integers")
            if not 0 <= item[correct] <= item[count]:
                raise ValueError("Invalid review counts")
            totals[count] += item[count]
            totals[correct] += item[correct]
        if any(type(item[name]) is not bool for name in (
            "expected_insufficient", "predicted_insufficient", "unjustified_direction"
        )):
            raise ValueError("Review decisions must be booleans")
        reviewed.append(item)
    tp = sum(r["predicted_insufficient"] and r["expected_insufficient"] for r in reviewed)
    fp = sum(r["predicted_insufficient"] and not r["expected_insufficient"] for r in reviewed)
    fn = sum(not r["predicted_insufficient"] and r["expected_insufficient"] for r in reviewed)
    real = [r for r in records if r["mode"] == "model"]
    latencies = [r["seconds"] for r in real if r["status"] == "completed"]
    costs = [r["estimated_cost_usd"] for r in real if r["estimated_cost_usd"] is not None]
    return {
        "runs": len(records), "real_model_runs": len(real), "reviewed_runs": len(reviewed),
        "quality_evaluated": bool(reviewed),
        "completion_rate": fraction(sum(r["status"] == "completed" for r in real), len(real)),
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "supported_claim_rate": fraction(totals["supported_claim_count"], totals["claim_count"]),
        "citation_correctness": fraction(totals["correct_citation_count"], totals["citation_count"]),
        "reference_fact_coverage": fraction(totals["covered_fact_count"], totals["reference_fact_count"]),
        "insufficiency_precision": fraction(tp, tp + fp),
        "insufficiency_recall": fraction(tp, tp + fn),
        "unjustified_direction_rate": fraction(sum(r["unjustified_direction"] for r in reviewed), len(reviewed)),
        "estimated_cost_usd": sum(costs) if real and len(costs) == len(real) else None,
        "note": "Synthetic diagnostic cases; human-reviewed metrics only. Not investment performance.",
    }
