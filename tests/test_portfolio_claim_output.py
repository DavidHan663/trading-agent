"""Strict provider-facing Portfolio Manager claim schema and compatibility adapter."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.portfolio_manager import (
    PM_EVIDENCE_LIMIT,
    _invoke_portfolio_structured,
    _select_portfolio_evidence,
    _structured_error_summary,
)
from tradingagents.agents.schemas import (
    CalculationInput,
    PortfolioCalculationClaimOutput,
    PortfolioDecision,
    PortfolioDecisionOutput,
    PortfolioFactClaimOutput,
    PortfolioJudgmentClaimOutput,
    PortfolioRating,
    render_pm_decision,
    to_portfolio_decision,
)


def _fact(claim_id="C1"):
    return PortfolioFactClaimOutput(
        claim_id=claim_id, text="Revenue was 120 million USD.", kind="financial",
        evidence_ids=["F1"], numeric_value=120, unit="million USD",
    )


def _calculation(claim_id="C2"):
    return PortfolioCalculationClaimOutput(
        claim_id=claim_id, text="Revenue growth was 20 percent.", kind="financial",
        calculation_inputs=[
            CalculationInput(name="current", value=120, unit="million USD", evidence_ids=["F1"]),
            CalculationInput(name="prior", value=100, unit="million USD", evidence_ids=["F1"]),
        ],
        formula="(current - prior) / prior * 100", result_value=20, result_unit="percent",
    )


def _judgment(claim_id="C3", basis=None):
    return PortfolioJudgmentClaimOutput(
        claim_id=claim_id, text="The reported comparison shows positive historical growth.",
        kind="rating_reason", basis_claim_ids=basis or ["C1", "C2"],
    )


def _output(**updates):
    values = {
        "rating": PortfolioRating.HOLD,
        "executive_summary": "Hold pending broader evidence.",
        "investment_thesis": "Revenue increased, but the evidence set is narrow.",
        "fact_claims": [_fact()],
        "calculation_claims": [_calculation()],
        "judgment_claims": [_judgment()],
    }
    values.update(updates)
    return PortfolioDecisionOutput(**values)


def test_provider_schema_separates_claim_fields_by_type():
    full_schema = PortfolioDecisionOutput.model_json_schema()
    schema = full_schema["$defs"]
    fact_fields = set(schema["PortfolioFactClaimOutput"]["properties"])
    calculation_fields = set(schema["PortfolioCalculationClaimOutput"]["properties"])
    judgment_fields = set(schema["PortfolioJudgmentClaimOutput"]["properties"])
    assert judgment_fields == {"claim_id", "text", "kind", "basis_claim_ids"}
    assert "basis_claim_ids" not in fact_fields | calculation_fields
    assert not ({"evidence_ids", "numeric_value", "calculation_inputs", "formula",
                 "result_value"} & judgment_fields)
    properties = full_schema["properties"]
    assert properties["fact_claims"]["maxItems"] == 3
    assert properties["calculation_claims"]["maxItems"] == 2
    assert properties["judgment_claims"]["maxItems"] == 3


def test_strict_output_converts_to_stable_public_decision_and_markdown():
    decision = to_portfolio_decision(_output())
    assert isinstance(decision, PortfolioDecision)
    assert [claim.claim_type for claim in decision.key_claims] == [
        "fact", "calculation", "judgment",
    ]
    assert decision.key_claims[-1].basis_claim_ids == ["C1", "C2"]
    markdown = render_pm_decision(decision)
    assert "**Rating**: Hold" in markdown
    assert "[C3:judgment; rating_reason]" in markdown
    assert "basis: C1, C2" in markdown
    assert to_portfolio_decision(decision) is decision


def test_judgment_forbids_direct_evidence_and_unknown_or_judgment_basis():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PortfolioJudgmentClaimOutput(
            claim_id="C3", text="Hold.", kind="rating_reason",
            basis_claim_ids=["C1"], evidence_ids=["F1"],
        )
    with pytest.raises(ValidationError, match="non-fact/calculation basis IDs"):
        _output(judgment_claims=[_judgment(basis=["C9"])])
    with pytest.raises(ValidationError, match="non-fact/calculation basis IDs"):
        _output(judgment_claims=[_judgment("C3", ["C1"]), _judgment("C4", ["C3"])])


def test_numeric_value_and_unit_must_be_provided_together_without_unit_conversion():
    with pytest.raises(ValidationError, match="provided together"):
        PortfolioFactClaimOutput(
            claim_id="C1", text="营收为1.20亿美元。", kind="financial",
            evidence_ids=["F1"], numeric_value=1.2,
        )
    fact = PortfolioFactClaimOutput(
        claim_id="C1", text="营收为1.20亿美元。", kind="financial",
        evidence_ids=["F1"], numeric_value=120, unit="million USD",
    )
    assert fact.numeric_value == 120
    assert fact.unit == "million USD"


def test_claim_ids_are_unique_and_each_category_is_capped():
    with pytest.raises(ValidationError, match="claim IDs must be unique"):
        _output(calculation_claims=[_calculation("C1")])
    facts = [_fact(f"C{index}") for index in range(1, 5)]
    with pytest.raises(ValidationError, match="at most 3 items"):
        _output(fact_claims=facts, calculation_claims=[], judgment_claims=[])
    calculations = [_calculation("C2"), _calculation("C3"), _calculation("C4")]
    with pytest.raises(ValidationError, match="at most 2 items"):
        _output(fact_claims=[_fact()], calculation_claims=calculations, judgment_claims=[])


def test_portfolio_manager_binds_strict_schema_but_accepts_legacy_mock_result():
    captured = {}
    llm = MagicMock()

    def bind(schema):
        captured["schema"] = schema
        return llm.with_structured_output.return_value

    llm.with_structured_output.side_effect = bind
    llm.with_structured_output.return_value.invoke.return_value = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="Wait.", investment_thesis="Limited data.",
    )
    captured_prompts = []
    llm.with_structured_output.return_value.invoke.side_effect = lambda prompt: (
        captured_prompts.append(prompt) or PortfolioDecision(
            rating=PortfolioRating.HOLD, executive_summary="Wait.", investment_thesis="Limited data.",
        )
    )
    risk = dict.fromkeys((
        "history", "aggressive_history", "conservative_history", "neutral_history",
        "current_aggressive_response", "current_conservative_response", "current_neutral_response",
    ), "")
    risk.update(latest_speaker="Neutral", count=1)
    result = create_portfolio_manager(llm)({
        "company_of_interest": "EVAL01", "trade_date": "2025-01-31",
        "investment_plan": "Wait", "trader_investment_plan": "Hold",
        "risk_debate_state": risk, "evidence": [],
    })
    assert captured["schema"] is PortfolioDecisionOutput
    assert result["final_trade_decision"].startswith("**Rating**: Hold")
    assert len(captured_prompts) == 1
    assert "numeric_value and unit are an all-or-nothing pair" in captured_prompts[0]
    assert "never return numeric_value without unit" in captured_prompts[0]


def _record(evidence_id, available_at="2025-01-01"):
    return {
        "id": evidence_id, "content": f"Evidence {evidence_id}",
        "available_at": available_at, "fetched_at": available_at,
    }


def test_portfolio_evidence_compaction_keeps_referenced_ids_and_newest_supplement():
    records = [_record(f"E{index:03d}", f"2025-01-{(index % 9) + 1:02d}") for index in range(100)]
    selected, diagnostics = _select_portfolio_evidence(
        {"market_report": "Use E050 for the price claim.", "risk_debate_state": {"history": "E050"}},
        records,
    )
    assert len(selected) == 9
    assert "E050" in {record["id"] for record in selected}
    assert diagnostics["mode"] == "referenced_plus_latest"
    assert diagnostics["omitted_count"] == 91


class _RecoveryStructured:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_portfolio_structured_recovery_retries_strict_schema_before_fallback():
    structured = _RecoveryStructured([None, _output()])
    plain = MagicMock()
    plain.invoke.return_value.content = "fallback"
    diagnostics = {}
    text, result = _invoke_portfolio_structured(
        structured, plain, "original prompt",
        lambda decision: render_pm_decision(to_portfolio_decision(decision)), diagnostics,
    )
    assert result is not None
    assert text.startswith("**Rating**: Hold")
    assert len(structured.prompts) == 2
    assert "RECOVERY" in structured.prompts[1]
    assert diagnostics["structured_attempts"] == 2
    assert diagnostics["status"] == "structured"
    assert diagnostics["recovery_used"] is True
    assert diagnostics["first_error_type"] == "ValueError"
    plain.invoke.assert_not_called()


def test_portfolio_structured_recovery_falls_back_only_after_two_failures():
    structured = _RecoveryStructured(error=RuntimeError("SECRET_VALUE"))
    plain = MagicMock()
    plain.invoke.return_value.content = "fallback"
    diagnostics = {}
    text, result = _invoke_portfolio_structured(
        structured, plain, "original prompt",
        lambda decision: render_pm_decision(to_portfolio_decision(decision)), diagnostics,
    )
    assert (text, result) == ("fallback", None)
    assert len(structured.prompts) == 2
    assert diagnostics["status"] == "fallback"
    assert diagnostics["first_error_type"] == "RuntimeError"
    assert diagnostics["second_error_type"] == "RuntimeError"
    assert "SECRET_VALUE" not in str(diagnostics)


def test_structured_error_summary_records_validation_shape_without_raw_content():
    with pytest.raises(ValidationError) as caught:
        PortfolioDecisionOutput.model_validate({"rating": "Hold", "executive_summary": "raw-response-secret"})
    summary = _structured_error_summary(caught.value)
    assert summary["error_type"] == "ValidationError"
    assert summary["validation_error_count"] >= 1
    assert "investment_thesis" in summary["validation_error_locations"]
    assert "missing" in summary["validation_error_types"]
    assert "raw-response-secret" not in str(summary)


def test_portfolio_diagnostics_store_sanitized_validation_summary():
    with pytest.raises(ValidationError) as caught:
        PortfolioDecisionOutput.model_validate({"rating": "Hold", "executive_summary": "raw-response-secret"})
    structured = _RecoveryStructured(error=caught.value)
    plain = MagicMock()
    plain.invoke.return_value.content = "fallback"
    diagnostics = {}
    _invoke_portfolio_structured(
        structured, plain, "original prompt",
        lambda decision: render_pm_decision(to_portfolio_decision(decision)), diagnostics,
    )
    assert diagnostics["first_error"]["error_type"] == "ValidationError"
    assert "investment_thesis" in diagnostics["first_error"]["validation_error_locations"]
    assert "raw-response-secret" not in str(diagnostics)
