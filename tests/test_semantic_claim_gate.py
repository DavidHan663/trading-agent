"""Shadow semantic gate behavior, budget, failure handling, and persistence."""

import json
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.semantic_claim_gate import (
    MAX_SEMANTIC_CLAIMS,
    RuntimeEvidenceQuote,
    RuntimeSemanticReview,
    create_semantic_claim_gate,
)
from tradingagents.evaluation.semantic_gate_integration_run import REPEAT_CASE_IDS, controlled_cases
from tradingagents.evidence import validate_claims
from tradingagents.reporting import write_report_tree


def _record(evidence_id="F1", content="Revenue was 120 million USD."):
    return {
        "id": evidence_id, "tool": "get_fundamentals", "target_ticker": "EVAL01",
        "requested_ticker": "EVAL01", "source_ticker": "EVAL01",
        "source_url": "https://example.test/source", "available_at": "2025-01-30",
        "period_end": "2024-12-31", "fetched_at": "2025-01-31T00:00:00Z",
        "content": content,
    }


def _fact(claim_id="C1", status="fact_verified", evidence_id="F1"):
    return {
        "claim_id": claim_id, "claim_type": "fact", "claim": "Revenue was 120 million USD.",
        "kind": "financial", "evidence_ids": [evidence_id], "basis_claim_ids": [],
        "issues": [], "status": status,
    }


def _allow(evidence_id="F1", quote="Revenue was 120 million USD"):
    return RuntimeSemanticReview(
        decision="allow", reason_flags=[],
        evidence_quotes=[RuntimeEvidenceQuote(evidence_id=evidence_id, quote=quote)],
        reasoning="The exact claim is directly supported.", confidence=0.95,
    )


class FixedSemanticModel:
    def __init__(self, results=None, error=None, binding_error=False):
        self.results = list(results or [])
        self.error = error
        self.binding_error = binding_error
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is RuntimeSemanticReview
        if self.binding_error:
            raise NotImplementedError
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.results.pop(0)


def _state(claims, evidence=None):
    return {
        "citation_validation": {"status": "checked", "claims": claims},
        "evidence": evidence if evidence is not None else [_record()],
        "final_trade_decision": "**Rating**: Buy",
    }


def test_supported_fact_is_allowed_without_changing_final_decision():
    model = FixedSemanticModel([_allow()])
    state = _state([_fact()])
    update = create_semantic_claim_gate(model)(state)
    gate = update["semantic_claim_validation"]
    assert gate["status"] == "allow"
    assert gate["model_invocations"] == 1
    assert gate["claims"][0]["decision"] == "allow"
    assert gate["final_decision_changed"] is False
    assert state["final_trade_decision"] == "**Rating**: Buy"
    assert "final_trade_decision" not in update


def test_prompt_distinguishes_requested_ticker_from_unconfirmed_source_ticker():
    record = _record()
    record.update(target_ticker="AAPL", requested_ticker="AAPL", source_ticker=None)
    model = FixedSemanticModel([_allow()])
    gate = create_semantic_claim_gate(model)(_state([_fact()], [record]))[
        "semantic_claim_validation"
    ]
    assert gate["status"] == "allow"
    assert "target_ticker=AAPL" in model.prompts[0]
    assert "requested_ticker=AAPL" in model.prompts[0]
    assert "source_ticker=unknown" in model.prompts[0]
    assert "Quotes must come from record content, not metadata" in model.prompts[0]

    metadata_quote = _allow(quote="target_ticker=AAPL")
    rejected = create_semantic_claim_gate(FixedSemanticModel([metadata_quote]))(
        _state([_fact()], [record])
    )["semantic_claim_validation"]
    assert rejected["status"] == "review"
    assert rejected["claims"][0]["failure_reason"] == "invalid_semantic_review"


def test_invalid_quote_and_model_exception_fail_closed_without_secret_text():
    invalid = _allow(quote="invented paraphrase")
    invalid_gate = create_semantic_claim_gate(FixedSemanticModel([invalid]))(_state([_fact()]))[
        "semantic_claim_validation"
    ]
    assert invalid_gate["status"] == "review"
    assert invalid_gate["claims"][0]["failure_reason"] == "invalid_semantic_review"
    assert invalid_gate["claims"][0]["decision"] == "review"

    failed_gate = create_semantic_claim_gate(
        FixedSemanticModel(error=RuntimeError("SECRET_VALUE"))
    )(_state([_fact()]))["semantic_claim_validation"]
    encoded = json.dumps(failed_gate)
    assert failed_gate["status"] == "review"
    assert failed_gate["claims"][0]["error_type"] == "RuntimeError"
    assert "SECRET_VALUE" not in encoded


def test_deterministic_failure_and_missing_structured_claims_do_not_call_model():
    model = FixedSemanticModel([_allow()])
    failed = create_semantic_claim_gate(model)(_state([_fact(status="fact_failed")]))[
        "semantic_claim_validation"
    ]
    assert failed["status"] == "review"
    assert failed["model_invocations"] == 0
    assert failed["claims"][0]["failure_reason"] == "deterministic_validation_failed"
    assert not model.prompts

    unvalidated = create_semantic_claim_gate(model)({
        "citation_validation": {"status": "unvalidated", "claims": []},
        "final_trade_decision": "**Rating**: Hold",
    })["semantic_claim_validation"]
    assert unvalidated["status"] == "review"
    assert unvalidated["reason"] == "no_structured_claims"
    assert unvalidated["model_invocations"] == 0


def test_calculation_and_judgment_receive_typed_contexts():
    calculation = {
        "claim_id": "C2", "claim_type": "calculation", "claim": "Free cash flow was 10 million USD.",
        "kind": "financial", "evidence_ids": [], "basis_claim_ids": [], "issues": [],
        "status": "calculation_verified", "formula": "cash - debt", "computed_value": 10.0,
        "result_value": 10.0, "result_unit": "million USD",
        "inputs": [{"name": "cash", "value": 30, "evidence_ids": ["B1"], "issues": []},
                   {"name": "debt", "value": 20, "evidence_ids": ["B1"], "issues": []}],
    }
    basis = _fact()
    judgment = {
        "claim_id": "C3", "claim_type": "judgment", "claim": "The balance sheet supports holding.",
        "kind": "rating_reason", "evidence_ids": [], "basis_claim_ids": ["C1"],
        "issues": [], "status": "judgment_review_required",
    }
    model = FixedSemanticModel([
        _allow("CALC-C2", "computed_value=10.0"),
        _allow(),
        _allow("CLAIM-C1", "claim=Revenue was 120 million USD."),
    ])
    gate = create_semantic_claim_gate(model)(_state(
        [calculation, basis, judgment],
        [_record("B1", "Cash was 30 million USD and debt was 20 million USD."), _record()],
    ))["semantic_claim_validation"]
    assert gate["status"] == "allow"
    assert gate["model_invocations"] == 3
    assert "[CALC-C2]" in model.prompts[0]
    assert "[CLAIM-C1]" in model.prompts[2]
    assert "Revenue was 120 million USD" in model.prompts[2]


def test_budget_is_bounded_and_excess_claims_are_reviewed():
    claims = [_fact(f"C{index}") for index in range(1, MAX_SEMANTIC_CLAIMS + 2)]
    model = FixedSemanticModel([_allow() for _ in range(MAX_SEMANTIC_CLAIMS)])
    gate = create_semantic_claim_gate(model)(_state(claims))["semantic_claim_validation"]
    assert gate["model_invocations"] == MAX_SEMANTIC_CLAIMS
    assert len(model.prompts) == MAX_SEMANTIC_CLAIMS
    assert gate["status"] == "review"
    assert gate["claims"][-1]["failure_reason"] == "semantic_budget_exceeded"


def test_binding_unavailable_fails_closed_without_free_text_retry():
    model = FixedSemanticModel(binding_error=True)
    gate = create_semantic_claim_gate(model)(_state([_fact()]))["semantic_claim_validation"]
    assert gate["status"] == "review"
    assert gate["model_invocations"] == 0
    assert gate["claims"][0]["failure_reason"] == "structured_binding_unavailable"
    assert not model.prompts


def test_report_persists_shadow_gate_in_json_and_markdown(tmp_path):
    gate = create_semantic_claim_gate(FixedSemanticModel([_allow()]))(_state([_fact()]))[
        "semantic_claim_validation"
    ]
    state = _state([_fact()])
    state["semantic_claim_validation"] = gate
    report = write_report_tree(state, "EVAL01", tmp_path)
    markdown = report.read_text()
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert saved["semantic_validation"]["status"] == "allow"
    assert "Semantic claim gate (shadow mode): **allow**" in markdown
    assert "did not change the final rating" in markdown
    assert "Explanation: The exact claim is directly supported." in markdown
    assert "Evidence F1: Revenue was 120 million USD" in markdown
    assert "[source](https://example.test/source)" in markdown


def test_report_shows_missing_information_and_rejects_unsafe_source_link(tmp_path):
    review = RuntimeSemanticReview(
        decision="review", reason_flags=["missing_evidence"],
        evidence_quotes=[RuntimeEvidenceQuote(evidence_id="F1", quote="Revenue was")],
        missing_information=["Prior-period revenue"], reasoning="Comparison is missing.",
        confidence=0.8,
    )
    record = _record()
    record["source_url"] = "javascript:alert(1)"
    state = _state([_fact()], [record])
    state["semantic_claim_validation"] = create_semantic_claim_gate(
        FixedSemanticModel([review])
    )(state)["semantic_claim_validation"]
    markdown = write_report_tree(state, "EVAL01", tmp_path).read_text()
    assert "Missing information: Prior-period revenue" in markdown
    assert "Explanation: Comparison is missing." in markdown
    assert "Evidence F1: Revenue was" in markdown
    assert "[source](javascript:" not in markdown


def test_initial_state_and_graph_place_shadow_gate_after_portfolio_manager(monkeypatch):
    initial = Propagator().create_initial_state("EVAL01", "2025-01-31")
    assert initial["semantic_claim_validation"] == {}

    from tradingagents.graph import setup

    node = lambda state: {}
    factory_names = (
        "create_aggressive_debator", "create_bear_researcher", "create_bull_researcher",
        "create_conservative_debator", "create_fundamentals_analyst", "create_market_analyst",
        "create_neutral_debator", "create_news_analyst", "create_portfolio_manager",
        "create_research_manager", "create_sentiment_analyst", "create_trader",
        "create_semantic_claim_gate",
    )
    for name in factory_names:
        monkeypatch.setattr(setup, name, lambda llm, result=node: result)
    monkeypatch.setattr(setup, "build_analyst_subgraph", lambda *args, **kwargs: node)
    monkeypatch.setattr(setup, "make_analyst_wrapper", lambda *args, **kwargs: node)
    workflow = setup.GraphSetup(
        MagicMock(), MagicMock(), {"market": node}, ConditionalLogic(1, 1),
    ).setup_graph(["market"])
    assert ("Portfolio Manager", "Semantic Claim Gate") in workflow.edges
    assert ("Semantic Claim Gate", "__end__") in workflow.edges
    assert ("Portfolio Manager", "__end__") not in workflow.edges


def test_controlled_live_plan_is_eligible_and_capped_at_twelve_calls():
    cases = controlled_cases()
    expected_calls = 0
    for case in cases:
        validation = validate_claims(case["claims"], case["records"], "EVAL01", "2025-01-31")
        assert validation["status"] == "checked"
        assert all(claim["status"] in {
            "fact_verified", "calculation_verified", "judgment_review_required",
        } for claim in validation["claims"])
        expected_calls += len(validation["claims"]) * (2 if case["id"] in REPEAT_CASE_IDS else 1)
    assert expected_calls == 12


def test_controlled_summary_handles_first_record_before_any_repeat_exists():
    from tradingagents.evaluation.semantic_gate_integration_run import _summary

    record = {
        "case_id": "first", "repetition": 1, "expected": "allow", "predicted": "allow",
        "gate": {
            "model_invocations": 1, "seconds": 1.0,
            "usage": {"provider_calls_observed": 1, "input_tokens": 10, "output_tokens": 5,
                      "token_usage_complete": True},
        },
    }
    assert _summary([record])["repeat_consistency"] is None
