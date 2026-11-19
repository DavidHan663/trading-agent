"""Blinded semantic-support dataset, reviewer validation, and metrics."""

import json
from collections import Counter

import pytest

from tradingagents.evaluation.semantic import (
    EvidenceQuote,
    SemanticReview,
    load_semantic_dataset,
    semantic_metrics,
    semantic_prompt,
    validate_semantic_review,
)
from tradingagents.evaluation.semantic_run import run_semantic_suite


def test_semantic_dataset_is_balanced_and_versioned():
    dataset, fingerprint = load_semantic_dataset()
    assert len(fingerprint) == 64
    assert len(dataset["cases"]) == 40
    assert Counter(case["label"] for case in dataset["cases"]) == {
        "supported": 8,
        "partially_supported": 8,
        "unsupported": 8,
        "contradicted": 8,
        "insufficient_evidence": 8,
    }


def test_prompt_is_blinded_and_contains_only_selected_evidence():
    dataset, _ = load_semantic_dataset()
    case = next(item for item in dataset["cases"] if item["id"] == "supported-01")
    prompt = semantic_prompt(case, dataset)
    assert case["claim"] in prompt
    assert case["rationale"] not in prompt
    assert "SYN-F1" in prompt
    assert "SYN-I1" not in prompt
    assert "proxy_rationale" not in prompt and "expected" not in prompt


def test_review_rejects_unknown_ids_and_nonliteral_quotes():
    dataset, _ = load_semantic_dataset()
    case = next(item for item in dataset["cases"] if item["id"] == "supported-01")
    valid = SemanticReview(
        verdict="supported",
        evidence_quotes=[EvidenceQuote(evidence_id="SYN-F1", quote="120 million USD")],
        reasoning="The comparison supports the calculation.", confidence=0.9,
    )
    assert validate_semantic_review(valid, case, dataset) == []
    unknown = valid.model_copy(update={"evidence_quotes": [
        EvidenceQuote(evidence_id="OTHER", quote="120 million USD"),
    ]})
    assert validate_semantic_review(unknown, case, dataset) == ["unknown_evidence_id:OTHER"]
    paraphrase = valid.model_copy(update={"evidence_quotes": [
        EvidenceQuote(evidence_id="SYN-F1", quote="revenue rose strongly"),
    ]})
    assert validate_semantic_review(paraphrase, case, dataset) == ["quote_not_literal:SYN-F1"]


def _record(case_id, expected, predicted, seconds=1.0, status="completed"):
    return {
        "case_id": case_id, "repetition": 1, "expected": expected,
        "predicted": predicted, "status": status, "seconds": seconds,
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "token_usage_complete": status == "completed"},
    }


def test_metrics_have_explicit_confusion_cost_and_consistency():
    records = [
        _record("s", "supported", "supported", 1),
        _record("u", "unsupported", "contradicted", 2),
        _record("c", "contradicted", "supported", 3),
        _record("s", "supported", "supported", 4) | {"repetition": 2},
    ]
    summary = semantic_metrics(records, input_price=1, output_price=2)
    assert summary["primary_completed"] == 3
    assert summary["accuracy"] == 1 / 3
    assert summary["supported_precision"] == 0.5
    assert summary["unsupported_or_contradicted_recall"] == 0.5
    assert summary["repeat_consistency"] == 1
    assert summary["latency_p50_seconds"] == 2
    assert summary["latency_p95_seconds"] == 4
    assert summary["estimated_cost_usd"] == pytest.approx(80 / 1_000_000)


class FixedStructuredModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is SemanticReview
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.result


def test_runner_writes_blinded_artifacts_without_free_text_fallback(tmp_path):
    model = FixedStructuredModel(SemanticReview(
        verdict="supported",
        evidence_quotes=[EvidenceQuote(evidence_id="SYN-F1", quote="120 million USD")],
        reasoning="Directly supported.", confidence=0.9,
    ))
    output = run_semantic_suite(tmp_path / "run", model, case_ids=["supported-01"])
    record = json.loads((output / "runs.json").read_text())[0]
    assert record["status"] == "completed"
    assert record["predicted"] == "supported"
    model_prompt = model.prompts[0]
    reference = json.loads((output / "supported-01-r1" / "reference.json").read_text())
    assert reference["proxy_rationale"] not in model_prompt
    assert (output / "supported-01-r1" / "prediction.json").exists()


def test_runner_stops_after_three_failures_and_stores_error_type_only(tmp_path):
    model = FixedStructuredModel(error=RuntimeError("SECRET_VALUE"))
    output = run_semantic_suite(
        tmp_path / "run", model,
        case_ids=["supported-01", "supported-02", "supported-03", "supported-04"],
    )
    records = json.loads((output / "runs.json").read_text())
    assert len(records) == 3
    assert all(record["error_type"] == "RuntimeError" for record in records)
    assert "SECRET_VALUE" not in json.dumps(records)
    assert json.loads((output / "stopped.json").read_text())["reason"] == "three_consecutive_failures"
