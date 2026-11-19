"""Hierarchical semantic dataset, validation, runner, and metrics."""

import json
import re
from collections import Counter

import pytest

from tradingagents.evaluation.semantic import EvidenceQuote, load_semantic_dataset
from tradingagents.evaluation.semantic_v2 import (
    HierarchicalSemanticReview,
    REASON_FLAGS,
    load_semantic_v2_dataset,
    semantic_v2_metrics,
    semantic_v2_prompt,
    validate_semantic_v2_review,
)
from tradingagents.evaluation.semantic_v2_run import run_semantic_v2_suite
from tradingagents.evaluation.semantic_v3 import (
    load_semantic_v3_dataset,
    semantic_v3_prompt,
)
from tradingagents.evaluation.semantic_v3_run import run_semantic_v3_suite
from tradingagents.evaluation.semantic_v2_run import _semantic_error_summary
from tradingagents.evaluation.trace import EvaluationTrace
from pydantic import ValidationError


def test_semantic_v2_dataset_is_frozen_balanced_and_multilabel():
    dataset, fingerprint = load_semantic_v2_dataset()
    assert len(fingerprint) == 64
    assert len(dataset["cases"]) == 40
    assert Counter(case["gate_decision"] for case in dataset["cases"]) == {
        "allow": 16, "review": 24,
    }
    reason_counts = Counter(
        flag for case in dataset["cases"] for flag in case["reason_flags"]
    )
    assert set(reason_counts) == set(REASON_FLAGS)
    assert sum(len(case["reason_flags"]) > 1 for case in dataset["cases"]) >= 8


def _mechanical_form(claim):
    normalized = claim.lower()
    normalized = re.sub(r"\b(?:aapl|vtx|company|shares?|stock)\b", "<entity>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def test_semantic_v2_claims_are_not_copied_or_numeric_ticker_substitutions_of_v1():
    v1, _ = load_semantic_dataset()
    v2, _ = load_semantic_v2_dataset()
    v1_forms = {_mechanical_form(case["claim"]) for case in v1["cases"]}
    v2_forms = [_mechanical_form(case["claim"]) for case in v2["cases"]]
    assert len(v2_forms) == len(set(v2_forms))
    assert not (v1_forms & set(v2_forms))


def test_semantic_v2_prompt_is_blinded_and_contains_only_selected_evidence():
    dataset, _ = load_semantic_v2_dataset()
    case = next(item for item in dataset["cases"] if item["id"] == "v2-review-04")
    prompt = semantic_v2_prompt(case, dataset)
    assert case["claim"] in prompt
    assert case["rationale"] not in prompt
    assert "V2-N1" in prompt
    assert "V2-F1" not in prompt
    assert "expected_decision" not in prompt and "proxy_rationale" not in prompt


def test_semantic_v2_review_enforces_gate_reason_and_literal_quote_rules():
    dataset, _ = load_semantic_v2_dataset()
    allow_case = next(item for item in dataset["cases"] if item["id"] == "v2-allow-01")
    valid = HierarchicalSemanticReview(
        decision="allow",
        evidence_quotes=[EvidenceQuote(evidence_id="V2-F1", quote="75 million USD")],
        reasoning="The supplied comparison supports the calculation.",
        confidence=0.9,
    )
    assert validate_semantic_v2_review(valid, allow_case, dataset) == []
    unknown = valid.model_copy(update={"evidence_quotes": [
        EvidenceQuote(evidence_id="OTHER", quote="75 million USD"),
    ]})
    assert validate_semantic_v2_review(unknown, allow_case, dataset) == [
        "unknown_evidence_id:OTHER",
    ]
    paraphrase = valid.model_copy(update={"evidence_quotes": [
        EvidenceQuote(evidence_id="V2-F1", quote="revenue rose substantially"),
    ]})
    assert validate_semantic_v2_review(paraphrase, allow_case, dataset) == [
        "quote_not_literal:V2-F1",
    ]
    invalid_allow = valid.model_copy(update={"reason_flags": ["unsupported_inference"]})
    assert "allow_has_reason_flags" in validate_semantic_v2_review(
        invalid_allow, allow_case, dataset,
    )


def test_semantic_v2_missing_evidence_requires_missing_information():
    dataset, _ = load_semantic_v2_dataset()
    case = next(item for item in dataset["cases"] if item["id"] == "v2-review-21")
    review = HierarchicalSemanticReview(
        decision="review", reason_flags=["missing_evidence"],
        evidence_quotes=[], missing_information=[],
        reasoning="The required value is absent.", confidence=0.9,
    )
    assert validate_semantic_v2_review(review, case, dataset) == [
        "missing_evidence_without_missing_information",
    ]


def _record(case_id, expected_decision, predicted_decision, expected_reasons=(),
            predicted_reasons=(), seconds=1.0, repetition=1, status="completed"):
    return {
        "case_id": case_id, "repetition": repetition,
        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "expected_reasons": list(expected_reasons),
        "predicted_reasons": list(predicted_reasons),
        "status": status, "seconds": seconds,
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "token_usage_complete": status == "completed"},
    }


def test_semantic_v2_metrics_separate_primary_quality_from_repeat_consistency():
    records = [
        _record("a", "allow", "allow", seconds=1),
        _record("r1", "review", "review", ["partial_support", "contradiction"],
                ["partial_support"], seconds=2),
        _record("r2", "review", "allow", ["missing_evidence"], [], seconds=3),
        _record("a", "allow", "review", [], ["unsupported_inference"],
                seconds=4, repetition=2),
    ]
    summary = semantic_v2_metrics(records, input_price=1, output_price=2)
    assert summary["primary_completed"] == 3
    assert summary["gate_accuracy"] == 2 / 3
    assert summary["allow_precision"] == 0.5
    assert summary["review_recall"] == 0.5
    assert summary["unsafe_allow_count"] == 1
    assert summary["reason_metrics"]["partial_support"]["recall"] == 1
    assert summary["reason_metrics"]["contradiction"]["recall"] == 0
    assert summary["review_reason_exact_match_rate"] == 0
    assert summary["review_any_correct_reason_rate"] == 0.5
    assert summary["repeat_consistency"] == 0
    assert summary["latency_p50_seconds"] == 2
    assert summary["latency_p95_seconds"] == 4
    assert summary["estimated_cost_usd"] == pytest.approx(80 / 1_000_000)


class FixedHierarchicalModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is HierarchicalSemanticReview
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.result


def test_semantic_v2_runner_writes_blinded_artifacts(tmp_path):
    model = FixedHierarchicalModel(HierarchicalSemanticReview(
        decision="allow", reason_flags=[],
        evidence_quotes=[EvidenceQuote(evidence_id="V2-F1", quote="75 million USD")],
        reasoning="Direct support.", confidence=0.9,
    ))
    output = run_semantic_v2_suite(
        tmp_path / "run", model, case_ids=["v2-allow-01"],
    )
    record = json.loads((output / "runs.json").read_text())[0]
    assert record["status"] == "completed"
    assert record["predicted_decision"] == "allow"
    reference = json.loads((output / "v2-allow-01-r1" / "reference.json").read_text())
    assert reference["proxy_rationale"] not in model.prompts[0]
    assert "expected_decision" not in model.prompts[0]
    assert (output / "v2-allow-01-r1" / "prediction.json").exists()


def test_semantic_v2_runner_stops_after_three_failures_without_exception_text(tmp_path):
    model = FixedHierarchicalModel(error=RuntimeError("SECRET_VALUE"))
    output = run_semantic_v2_suite(
        tmp_path / "run", model,
        case_ids=["v2-allow-01", "v2-allow-02", "v2-allow-03", "v2-allow-04"],
    )
    records = json.loads((output / "runs.json").read_text())
    assert len(records) == 3
    assert all(record["error_type"] == "RuntimeError" for record in records)
    assert "SECRET_VALUE" not in json.dumps(records)
    assert json.loads((output / "stopped.json").read_text())["reason"] == (
        "three_consecutive_failures"
    )


def test_semantic_v3_is_versioned_and_preserves_v2_parent():
    v2, v2_hash = load_semantic_v2_dataset()
    v3, v3_hash = load_semantic_v3_dataset()
    assert v2_hash == "0758fda38906bec1f20c974646bec9ee783b928997414311462a1eb8fc7e4d46"
    assert v3_hash != v2_hash
    assert v3["schema_version"] == 3
    assert v3["parent_dataset_sha256"] == v2_hash
    assert len(v3["cases"]) == 40
    assert [case["id"] for case in v3["cases"]] == [case["id"] for case in v2["cases"]]
    updated = {case["id"]: case for case in v3["cases"]}
    assert "missing_evidence" in updated["v2-review-02"]["reason_flags"]
    assert set(updated["v2-review-03"]["reason_flags"]) == {
        "partial_support", "contradiction", "missing_evidence",
    }
    assert v2["cases"] != v3["cases"]


def test_semantic_v3_prompt_supplies_ticker_metadata_and_stays_blinded():
    dataset, _ = load_semantic_v3_dataset()
    case = next(item for item in dataset["cases"] if item["id"] == "v2-review-02")
    prompt = semantic_v3_prompt(case, dataset)
    assert "ticker=VTX" in prompt
    assert case["claim"] in prompt
    assert case["rationale"] not in prompt
    assert "expected_reasons" not in prompt and "proxy_rationale" not in prompt


def test_semantic_error_summary_is_field_level_and_does_not_store_inputs():
    with pytest.raises(ValidationError) as caught:
        HierarchicalSemanticReview.model_validate({
            "decision": "allow", "reasoning": "SECRET_RESPONSE",
        })
    summary = _semantic_error_summary(caught.value)
    assert summary["error_type"] == "ValidationError"
    assert summary["validation_error_count"] >= 1
    assert "confidence" in summary["validation_error_locations"]
    assert "missing" in summary["validation_error_types"]
    assert "SECRET_RESPONSE" not in json.dumps(summary)


def test_evaluation_trace_can_omit_prompt_and_response_content():
    from langchain_core.outputs import Generation, LLMResult

    trace = EvaluationTrace(capture_content=False)
    trace.on_llm_start({}, ["SECRET_PROMPT"], run_id="run-1")
    trace.on_llm_end(LLMResult(generations=[[Generation(text="SECRET_RESPONSE")]]), run_id="run-1")
    assert trace.summary()["llm_calls"] == 1
    assert all("SECRET" not in json.dumps(event) for event in trace.events)
    assert trace.events[0] == {"kind": "llm_start", "id": "run-1"}
    assert "output" not in trace.events[1]


def test_semantic_v3_runner_persists_sanitized_failure_and_safe_trace(tmp_path):
    with pytest.raises(ValidationError) as caught:
        HierarchicalSemanticReview.model_validate({"decision": "allow", "reasoning": "SECRET_RESPONSE"})
    model = FixedHierarchicalModel(error=caught.value)
    output = run_semantic_v3_suite(
        tmp_path / "run", model, case_ids=["v2-review-09"],
    )
    record = json.loads((output / "runs.json").read_text())[0]
    assert record["error_summary"]["error_type"] == "ValidationError"
    assert "confidence" in record["error_summary"]["validation_error_locations"]
    assert "SECRET_RESPONSE" not in json.dumps(record)
    trace = json.loads((output / "v2-review-09-r1" / "trace.json").read_text())
    assert all(not ({"messages", "prompts", "output", "arguments"} & set(event)) for event in trace)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["trace_content_captured"] is False
