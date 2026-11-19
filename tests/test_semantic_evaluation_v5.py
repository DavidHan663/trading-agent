"""Offline validation for the fresh semantic-v5 holdout."""

import json
from collections import Counter

from tradingagents.evaluation.semantic import EvidenceQuote
from tradingagents.evaluation.semantic_v2 import HierarchicalSemanticReview
from tradingagents.evaluation.semantic_v4 import load_semantic_v4_dataset
from tradingagents.evaluation.semantic_v5 import (
    SEMANTIC_V5_SHA256,
    load_semantic_v5_dataset,
    semantic_v5_prompt,
)
from tradingagents.evaluation.semantic_v5_run import run_semantic_v5_suite


def test_v5_holdout_is_hash_frozen_balanced_and_distinct_from_v4():
    dataset, fingerprint = load_semantic_v5_dataset()
    v4, _ = load_semantic_v4_dataset()
    assert fingerprint == SEMANTIC_V5_SHA256
    assert len(dataset["cases"]) == 40
    assert Counter(case["gate_decision"] for case in dataset["cases"]) == {
        "allow": 16, "review": 24,
    }
    assert not ({case["claim"] for case in dataset["cases"]}
                & {case["claim"] for case in v4["cases"]})
    reasons = Counter(flag for case in dataset["cases"] for flag in case["reason_flags"])
    assert set(reasons) == {
        "partial_support", "contradiction", "missing_evidence", "unsupported_inference",
    }
    assert min(reasons.values()) >= 4
    assert sum(len(case["reason_flags"]) > 1 for case in dataset["cases"]) >= 8
    arithmetic = next(case for case in dataset["cases"] if case["id"] == "v5-02")
    assert arithmetic["gate_decision"] == "allow"
    assert not arithmetic["reason_flags"]


def test_v5_prompt_blinds_labels_and_rationale_and_includes_ticker():
    dataset, _ = load_semantic_v5_dataset()
    case = dataset["cases"][20]
    prompt = semantic_v5_prompt(case, dataset)
    assert case["claim"] in prompt
    assert f"ticker={case['ticker']}" in prompt
    assert case["rationale"] not in prompt
    assert "gate_decision" not in prompt
    assert "expected_decision" not in prompt
    assert "reason_adjudication_policy" not in prompt


class _FixedReviewModel:
    def __init__(self, error=None):
        self.prompts = []
        self.error = error

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return HierarchicalSemanticReview(
            decision="allow",
            reason_flags=[],
            evidence_quotes=[EvidenceQuote(evidence_id="V5-E01", quote="2.4 million")],
            reasoning="The recorded count matches the claim.",
            confidence=0.9,
        )


def test_v5_runner_records_one_call_with_sanitized_trace(tmp_path):
    model = _FixedReviewModel()
    output = run_semantic_v5_suite(tmp_path / "run", model, case_ids=["v5-01"])
    run = json.loads((output / "runs.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    trace = json.loads((output / "v5-01-r1" / "trace.json").read_text())
    assert len(model.prompts) == 1
    assert run[0]["status"] == "completed"
    assert manifest["dataset_sha256"] == SEMANTIC_V5_SHA256
    assert manifest["trace_content_captured"] is False
    assert all(not ({"messages", "prompts", "output", "arguments"} & event.keys())
               for event in trace)


def test_v5_runner_stops_after_three_failures(tmp_path):
    model = _FixedReviewModel(error=RuntimeError("private-provider-message"))
    output = run_semantic_v5_suite(
        tmp_path / "run", model,
        case_ids=["v5-01", "v5-02", "v5-03", "v5-04"],
    )
    runs = json.loads((output / "runs.json").read_text())
    stopped = json.loads((output / "stopped.json").read_text())
    assert len(runs) == 3
    assert len(model.prompts) == 3
    assert stopped["reason"] == "three_consecutive_failures"
    assert "private-provider-message" not in json.dumps(runs)
