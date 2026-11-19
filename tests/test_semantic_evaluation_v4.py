"""Offline checks for the independently frozen v4 semantic holdout."""

import json
from collections import Counter

from tradingagents.evaluation.semantic_v2 import HierarchicalSemanticReview
from tradingagents.evaluation.semantic_v4 import (
    SEMANTIC_V4_SHA256,
    load_semantic_v4_dataset,
    semantic_v4_prompt,
)
from tradingagents.evaluation.semantic_v4_run import run_semantic_v4_suite
from tradingagents.evaluation.semantic import EvidenceQuote


def test_v4_holdout_is_frozen_balanced_and_distinct_from_prior_cases():
    dataset, fingerprint = load_semantic_v4_dataset()
    assert fingerprint == SEMANTIC_V4_SHA256
    assert len(dataset["cases"]) == 40
    assert Counter(case["gate_decision"] for case in dataset["cases"]) == {
        "allow": 16, "review": 24,
    }
    assert all(case["id"].startswith("holdout-") for case in dataset["cases"])
    reason_counts = Counter(flag for case in dataset["cases"] for flag in case["reason_flags"])
    assert set(reason_counts) == {"partial_support", "contradiction", "missing_evidence", "unsupported_inference"}
    assert min(reason_counts.values()) >= 4
    assert sum(len(case["reason_flags"]) > 1 for case in dataset["cases"]) >= 8


def test_v4_prompt_excludes_labels_and_rationale_but_includes_ticker():
    dataset, _ = load_semantic_v4_dataset()
    case = dataset["cases"][20]
    prompt = semantic_v4_prompt(case, dataset)
    assert case["claim"] in prompt
    assert f"ticker={case['ticker']}" in prompt
    assert case["rationale"] not in prompt
    assert "gate_decision" not in prompt and "expected_decision" not in prompt
    assert "expected_reasons" not in prompt


class _FixedReviewModel:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return HierarchicalSemanticReview(
            decision="allow", reason_flags=[],
            evidence_quotes=[EvidenceQuote(evidence_id="HOLD-E01", quote="$120 million")],
            reasoning="Directly supported.", confidence=0.9,
        )


def test_v4_runner_records_one_call_and_omits_model_content(tmp_path):
    model = _FixedReviewModel()
    output = run_semantic_v4_suite(tmp_path / "run", model, case_ids=["holdout-01"])
    record = json.loads((output / "runs.json").read_text())[0]
    manifest = json.loads((output / "manifest.json").read_text())
    trace = json.loads((output / "holdout-01-r1" / "trace.json").read_text())
    assert len(model.prompts) == 1
    assert record["status"] == "completed"
    assert manifest["dataset_sha256"] == SEMANTIC_V4_SHA256
    assert manifest["trace_content_captured"] is False
    assert all("messages" not in event and "output" not in event for event in trace)
