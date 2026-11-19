"""Evidence provenance and citation checks without a model or network."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    CalculationInput,
    EvidenceClaim,
    PortfolioDecision,
    PortfolioRating,
)
from tradingagents.agents.utils.structured import invoke_structured_with_result
from tradingagents.evidence import (
    check_analyst_report,
    records_from_tool,
    unique_records,
    validate_claims,
)
from tradingagents.dataflows.interface import _attach_provider_reference
from tradingagents.graph.analyst_execution import AnalystNodeSpec
from tradingagents.graph.analyst_subgraph import build_analyst_subgraph, make_analyst_wrapper
from tradingagents.reporting import write_report_tree


def _record(**updates):
    record = records_from_tool(
        "get_fundamentals", {"ticker": "EVAL01"},
        json.dumps({"records": [{"id": "F1", "source": "https://example.invalid/f1",
                                "available_at": "2025-01-30", "period_end": "2024-12-31",
                                "content": "Revenue=120 million USD."}]}),
        "EVAL01",
    )[0]
    return record | updates


def test_record_preserves_explicit_metadata_and_unknowns():
    record = _record()
    assert record["id"] == "F1"
    assert record["source_url"] == "https://example.invalid/f1"
    assert record["available_at"] == "2025-01-30"
    plain = records_from_tool("get_news", {"ticker": "EVAL01"}, "Product announced.", "EVAL01")[0]
    assert plain["id"].startswith("E-")
    assert plain["available_at"] is None and plain["source_url"] is None
    mixed = records_from_tool("get_news", {"ticker": "EVAL01"},
                              "https://one.invalid https://two.invalid", "EVAL01")[0]
    assert mixed["source_url"] is None
    assert records_from_tool("get_news", {}, "<unavailable> No evidence", "EVAL01") == []
    assert records_from_tool("get_stock_data", {}, "NO_DATA_AVAILABLE: no price", "EVAL01") == []
    assert records_from_tool("get_news", {}, "Error fetching news for EVAL01: down", "EVAL01") == []
    assert records_from_tool("get_news", {}, "No news found for EVAL01", "EVAL01") == []
    assert records_from_tool("get_fundamentals", {},
                             "# Point-in-time as of: 2026-11-13\nProfile fundamentals are withheld for this date.",
                             "EVAL01") == []
    assert records_from_tool("stocktwits", {}, "<no StockTwits messages for $AAPL>", "AAPL") == []
    assert records_from_tool("reddit", {}, "<Reddit unavailable: every source failed>", "AAPL") == []
    assert records_from_tool("get_prediction_markets", {},
                             '## Polymarket prediction markets: "x"\nNo open prediction markets matched x',
                             "AAPL") == []


def test_real_tool_text_preserves_safe_dates_and_news_provenance():
    snapshot = records_from_tool(
        "get_verified_market_snapshot", {"symbol": "AAPL"},
        "Latest trading row used: 2026-11-13\n| Close | 100.00 |", "AAPL",
    )[0]
    assert snapshot["available_at"] == "2026-11-13"
    assert snapshot["period_end"] == "2026-11-13"
    financial = records_from_tool(
        "get_income_statement", {"ticker": "AAPL"},
        "# Income Statement data\n# Data retrieved on: 2026-11-14\n\n,2026-09-30,2026-06-30\nRevenue,100,90", "AAPL",
    )[0]
    assert financial["period_end"] == "2026-09-30"
    assert financial["available_at"] is None
    series = records_from_tool(
        "get_indicators", {"symbol": "AAPL"},
        "## rsi values\n\n2026-11-13: 55\n2026-11-12: 54", "AAPL",
    )[0]
    assert series["available_at"] == "2026-11-13"
    assert series["period_end"] == "2026-11-13"
    news = records_from_tool(
        "get_news", {"ticker": "AAPL"},
        "## AAPL News\n\n### Product ships (source: Example Wire)\nSummary.\nPublished: 2026-11-12\nLink: https://example.invalid/a\n\n"
        "### Earnings scheduled (source: Other Wire)\nPublished: 2026-11-13\nLink: https://example.invalid/b\n", "AAPL",
    )
    assert [(r["available_at"], r["source_url"]) for r in news] == [
        ("2026-11-12", "https://example.invalid/a"),
        ("2026-11-13", "https://example.invalid/b"),
    ]
    repeated = records_from_tool(
        "get_news", {"ticker": "AAPL", "start_date": "2026-11-01"},
        "### Product ships (source: Example Wire)\nSummary.\nPublished: 2026-11-12\n"
        "Link: https://example.invalid/a\n", "AAPL",
    ) + records_from_tool(
        "get_news", {"ticker": "AAPL", "start_date": "2026-11-10"},
        "### Product ships (source: Example Wire)\nSummary.\nPublished: 2026-11-12\n"
        "Link: https://example.invalid/a\n", "AAPL",
    )
    assert repeated[0]["id"] == repeated[1]["id"]
    assert len(unique_records(repeated)) == 1


def test_provider_reference_is_traceable_but_never_invents_financial_availability():
    wrapped = _attach_provider_reference(
        "# Cash Flow data for AAPL\n\n,2026-09-30\nFree Cash Flow,100",
        "get_cashflow", "yfinance", ("AAPL",), {"freq": "annual"},
    )
    record = records_from_tool("get_cashflow", {"ticker": "AAPL"}, wrapped, "AAPL")[0]
    assert record["source_url"] == "https://finance.yahoo.com/quote/AAPL/cash-flow"
    assert record["source_provider"] == "yfinance"
    assert record["source_kind"] == "provider_reference"
    assert record["period_end"] == "2026-09-30"
    assert record["available_at"] is None

    market = _attach_provider_reference(
        "Latest trading row used: 2026-11-13\n| Close | 100.00 |",
        "get_stock_data", "yfinance", ("AAPL",), {},
    )
    market_record = records_from_tool("get_stock_data", {"symbol": "AAPL"}, market, "AAPL")[0]
    assert market_record["source_url"] == "https://finance.yahoo.com/quote/AAPL/history"
    assert market_record["available_at"] == "2026-11-13"

    alpha = _attach_provider_reference(
        "timestamp,close\n2026-11-13,100", "get_stock_data", "alpha_vantage", ("AAPL",), {},
    )
    alpha_record = records_from_tool("get_stock_data", {"symbol": "AAPL"}, alpha, "AAPL")[0]
    assert alpha_record["source_url"] == (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL"
    )
    assert alpha_record["source_kind"] == "provider_reference"


def test_validation_flags_bad_identity_time_number_and_collision():
    claim = EvidenceClaim(claim_id="C1", claim_type="fact",
                          text="Revenue was 120 million USD", kind="financial",
                          evidence_ids=["F1"], numeric_value=120, unit="million USD")
    assert validate_claims([claim], [_record()], "EVAL01", "2025-01-31")["claims"][0]["status"] == "fact_verified"
    wrong = validate_claims([claim], [_record(source_ticker="OTHER")], "EVAL01", "2025-01-31")
    assert "wrong_source_ticker:F1" in wrong["claims"][0]["issues"]
    future = validate_claims([claim], [_record(available_at="2025-02-01")], "EVAL01", "2025-01-31")
    assert "future_evidence:F1" in future["claims"][0]["issues"]
    mismatch = validate_claims([claim], [_record(content="Revenue=180 million USD.")], "EVAL01", "2025-01-31")
    assert "numeric_not_literal" in mismatch["claims"][0]["issues"]
    assert mismatch["claims"][0]["status"] == "fact_needs_review"
    derived = validate_claims([EvidenceClaim(claim_id="C1", claim_type="fact",
                                             text="Net cash is 10 million USD",
                                             kind="financial", evidence_ids=["F1"],
                                             numeric_value=10, unit="million USD")],
                              [_record(content="Cash=30 million USD; debt=20 million USD.")],
                              "EVAL01", "2025-01-31")
    assert derived["claims"][0]["status"] == "fact_needs_review"
    uncited = validate_claims([EvidenceClaim(claim_id="C1", claim_type="fact",
                                            text="Valuation data are missing",
                                            kind="rating_reason", evidence_ids=[])],
                              [_record()], "EVAL01", "2025-01-31")
    assert uncited["claims"][0]["issues"] == ["no_citation"]
    assert uncited["claims"][0]["status"] == "fact_needs_review"
    collision = validate_claims([claim], [_record(), _record(content="Revenue=180 million USD.")], "EVAL01", "2025-01-31")
    assert "id_collision:F1" in collision["claims"][0]["issues"]
    unknown = validate_claims([claim], [], "EVAL01", "2025-01-31")
    assert "unknown_id:F1" in unknown["claims"][0]["issues"]
    unclear = validate_claims([claim], [_record(content="Revenue rose by 120M USD.")],
                              "EVAL01", "2025-01-31")
    assert unclear["claims"][0]["status"] == "fact_needs_review"


def test_text_numeric_fallback_requires_every_non_date_literal_to_be_in_cited_evidence():
    market = _record(content="2025-01-31 close=100 USD.")
    literal = EvidenceClaim(claim_id="C1", claim_type="fact", kind="price",
                            text="On 2025-01-31, the close was 100 USD.", evidence_ids=["F1"])
    assert validate_claims([literal], [market], "EVAL01", "2025-01-31")["claims"][0]["status"] == "fact_verified"

    mismatch = literal.model_copy(update={"text": "On 2025-01-31, the close was 101 USD."})
    checked = validate_claims([mismatch], [market], "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "fact_needs_review"
    assert "numeric_needs_review" in checked["issues"]

    without_number = literal.model_copy(update={"text": "The close was elevated."})
    assert validate_claims([without_number], [market], "EVAL01", "2025-01-31")["claims"][0]["status"] == "fact_needs_review"


def test_text_numeric_fallback_handles_chinese_iso_dates_and_display_rounding():
    evidence = _record(content=(
        "Latest trading row used: 2025-01-31; close=100 USD. "
        "10 EMA=99.7320639742041 USD."
    ))
    claim = EvidenceClaim(
        claim_id="C1", claim_type="fact", kind="price",
        text="截至2025-01-31，收盘价为100美元，10 EMA为99.7321美元。",
        evidence_ids=["F1"],
    )
    checked = validate_claims([claim], [evidence], "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "fact_verified"

    mismatch = claim.model_copy(update={"text": "截至2025-01-31，收盘价为100美元，10 EMA为99.7329美元。"})
    checked = validate_claims([mismatch], [evidence], "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "fact_needs_review"
    assert "numeric_needs_review" in checked["issues"]


def test_provenance_fixture_distinguishes_verified_missing_and_future_metadata():
    fixture_path = Path("tradingagents/evaluation/fixtures/evidence_provenance.v1.json")
    records = json.loads(fixture_path.read_text())["records"]
    claim = EvidenceClaim(claim_id="C1", claim_type="fact", kind="financial",
                          text="Revenue was 120 million USD.", evidence_ids=["FINANCIAL-VERIFIED"],
                          numeric_value=120, unit="million USD")
    assert validate_claims([claim], records, "EVAL01", "2025-01-31")["claims"][0]["status"] == "fact_verified"

    period_only = claim.model_copy(update={"evidence_ids": ["FINANCIAL-PERIOD-ONLY"]})
    checked = validate_claims([period_only], records, "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "fact_needs_review"
    assert "unknown_publication_date:FINANCIAL-PERIOD-ONLY" in checked["issues"]

    future = EvidenceClaim(claim_id="C2", claim_type="fact", kind="price",
                           text="Close was 120 USD.", evidence_ids=["MARKET-FUTURE"],
                           numeric_value=120, unit="USD")
    checked = validate_claims([future], records, "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "fact_failed"
    assert "future_evidence:MARKET-FUTURE" in checked["issues"]


def test_atomic_calculation_reproduces_inputs_and_rejects_bad_math():
    revenue = _record()
    costs = _record(id="F2", content="Costs=80 million USD.",
                    source_url="https://example.invalid/f2")
    inputs = [
        CalculationInput(name="revenue", value=120, unit="million USD", evidence_ids=["F1"]),
        CalculationInput(name="costs", value=80, unit="million USD", evidence_ids=["F2"]),
    ]
    claim = EvidenceClaim(
        claim_id="C1", claim_type="calculation", kind="financial",
        text="Gross profit was 40 million USD.", evidence_ids=[],
        calculation_inputs=inputs, formula="revenue - costs", result_value=40,
        result_unit="million USD",
    )
    checked = validate_claims([claim], [revenue, costs], "EVAL01", "2025-01-31")["claims"][0]
    assert checked["status"] == "calculation_verified"
    assert checked["arithmetic_status"] == "verified"
    assert checked["computed_value"] == 40

    wrong = claim.model_copy(update={"result_value": 41})
    bad_result = validate_claims([wrong], [revenue, costs], "EVAL01", "2025-01-31")["claims"][0]
    assert bad_result["status"] == "calculation_failed"
    assert bad_result["arithmetic_status"] == "failed"
    assert "result_mismatch" in bad_result["issues"]

    illegal = claim.model_copy(update={"formula": "__import__('os').system('id')"})
    bad_formula = validate_claims([illegal], [revenue, costs], "EVAL01", "2025-01-31")["claims"][0]
    assert bad_formula["status"] == "calculation_failed"
    assert "formula_invalid" in bad_formula["issues"]

    zero = claim.model_copy(update={"formula": "revenue / (costs - costs)"})
    assert validate_claims([zero], [revenue, costs], "EVAL01", "2025-01-31")["claims"][0]["status"] == "calculation_failed"

    missing = claim.model_copy(update={"calculation_inputs": [
        CalculationInput(name="revenue", value=120, evidence_ids=["UNKNOWN"]),
    ], "formula": "revenue", "result_value": 120})
    assert validate_claims([missing], [revenue], "EVAL01", "2025-01-31")["claims"][0]["status"] == "calculation_failed"


def test_judgment_never_becomes_evidence_verified():
    fact = EvidenceClaim(claim_id="C1", claim_type="fact", kind="financial",
                         text="Revenue was 120 million USD.", evidence_ids=["F1"],
                         numeric_value=120, unit="million USD")
    judgment = EvidenceClaim(claim_id="C2", claim_type="judgment", kind="rating_reason",
                             text="The position should be held.", evidence_ids=[],
                             basis_claim_ids=["C1"])
    results = validate_claims([fact, judgment], [_record()], "EVAL01", "2025-01-31")["claims"]
    assert results[0]["status"] == "fact_verified"
    assert results[1]["status"] == "judgment_review_required"
    assert results[1]["issues"] == []

    broken = judgment.model_copy(update={"basis_claim_ids": ["C9"]})
    invalid = validate_claims([fact, broken], [_record()], "EVAL01", "2025-01-31")["claims"][1]
    assert invalid["status"] == "judgment_invalid"
    assert "unknown_basis_claim:C9" in invalid["issues"]


def test_legacy_composite_claim_is_never_marked_verified():
    legacy = EvidenceClaim(text="Revenue supports holding.", kind="rating_reason",
                           evidence_ids=["F1"])
    result = validate_claims([legacy], [_record()], "EVAL01", "2025-01-31")["claims"][0]
    assert result["status"] == "legacy_needs_review"
    assert "legacy_composite_claim" in result["issues"]


def test_atomic_schema_rejects_mixed_claim_shapes():
    with pytest.raises(ValueError, match="fact claims cannot"):
        EvidenceClaim(claim_id="C1", claim_type="fact", kind="financial",
                      text="Mixed claim", evidence_ids=["F1"], formula="x + 1")
    with pytest.raises(ValueError, match="judgment claims must"):
        EvidenceClaim(claim_id="C2", claim_type="judgment", kind="rating_reason",
                      text="Hold", evidence_ids=["F1"], basis_claim_ids=["C1"])


def test_analyst_tool_result_reaches_report_and_parent_evidence():
    @tool
    def get_fundamentals(ticker: str) -> str:
        """Return a frozen fundamental record."""
        return _attach_provider_reference(
            json.dumps({"records": [{"id": "F1", "source": "https://example.invalid/f1",
                                       "available_at": "2025-01-30", "period_end": "2024-12-31",
                                       "content": "Revenue=120 million USD."}]}),
            "get_fundamentals", "yfinance", (ticker,), {},
        )

    spec = AnalystNodeSpec("market", "Market Analyst", "Msg Clear Market", "tools_market", "market_report")

    def analyst(state):
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content="Revenue 120 million USD [F1].")],
                    "market_report": "Revenue 120 million USD [F1]."}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "get_fundamentals", "args": {"ticker": "EVAL01"}, "id": "call-1"}
        ])]}

    def route(state):
        last = state["messages"][-1]
        return spec.tool_node if getattr(last, "tool_calls", None) else spec.clear_node

    subgraph = build_analyst_subgraph(spec, analyst, ToolNode([get_fundamentals]), route)
    result = make_analyst_wrapper(subgraph, spec)({
        "company_of_interest": "EVAL01", "trade_date": "2025-01-31", "asset_type": "stock",
    })
    assert result["market_report"].endswith("[F1].")
    assert result["evidence"][0]["id"] == "F1"
    assert result["evidence"][0]["source_provider"] == "yfinance"
    assert result["analyst_citation_checks"][0]["status"] == "ids_present"
    assert check_analyst_report("market", "Revenue rose.", result["evidence"])["status"] == "missing_citation_ids"


def test_portfolio_validation_and_report_artifact(tmp_path):
    llm = MagicMock()
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="Hold pending evidence.",
        investment_thesis="Revenue 120 million USD [F1].",
        key_claims=[EvidenceClaim(claim_id="C1", claim_type="fact",
                                  text="Revenue 120 million USD", kind="financial",
                                  evidence_ids=["F1"], numeric_value=120, unit="million USD")],
    )
    llm.with_structured_output.return_value.invoke.return_value = decision
    risk = dict.fromkeys(("history", "aggressive_history", "conservative_history",
                          "neutral_history", "current_aggressive_response",
                          "current_conservative_response", "current_neutral_response"), "")
    risk.update(latest_speaker="Neutral", count=1)
    state = {"company_of_interest": "EVAL01", "trade_date": "2025-01-31",
             "investment_plan": "Wait", "trader_investment_plan": "Hold",
             "risk_debate_state": risk, "evidence": [_record()]}
    result = create_portfolio_manager(llm)(state)
    assert result["citation_validation"]["claims"][0]["status"] == "fact_verified"
    assert result["citation_validation"]["structured_output"]["status"] == "structured"
    assert "[F1]" in result["final_trade_decision"]
    state.update(result)
    report = write_report_tree(state, "EVAL01", tmp_path)
    assert "Evidence and Citation Checks" in report.read_text()
    assert "fact_verified" in report.read_text()
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert saved["records"][0]["id"] == "F1"


def test_free_text_decision_is_explicitly_unvalidated():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value.content = "**Rating**: Hold"
    risk = dict.fromkeys(("history", "aggressive_history", "conservative_history",
                          "neutral_history", "current_aggressive_response",
                          "current_conservative_response", "current_neutral_response"), "")
    risk.update(latest_speaker="Neutral", count=1)
    result = create_portfolio_manager(llm)({
        "company_of_interest": "EVAL01", "trade_date": "2025-01-31",
        "investment_plan": "Wait", "trader_investment_plan": "Hold",
        "risk_debate_state": risk,
    })
    assert result["citation_validation"]["status"] == "unvalidated"
    assert result["citation_validation"]["structured_output"]["reason"] == "binding_unavailable"


def test_structured_fallback_records_safe_failure_reason():
    llm = MagicMock()
    llm.invoke.return_value.content = "fallback"
    structured = MagicMock()
    structured.invoke.return_value = None
    details = {}
    text, parsed = invoke_structured_with_result(structured, llm, "prompt", str,
                                                  "Portfolio Manager", diagnostics=details)
    assert (text, parsed) == ("fallback", None)
    assert details == {"status": "fallback", "reason": "no_parsed_result"}
    structured.invoke.side_effect = RuntimeError("secret-bearing message must not be stored")
    details = {}
    invoke_structured_with_result(structured, llm, "prompt", str,
                                  "Portfolio Manager", diagnostics=details)
    assert details == {"status": "fallback", "reason": "structured_exception",
                       "error_type": "RuntimeError"}


def test_missing_structured_claims_and_legacy_report_are_not_validated(tmp_path):
    decision = PortfolioDecision(rating=PortfolioRating.HOLD,
                                 executive_summary="Wait for evidence.",
                                 investment_thesis="Insufficient data.")
    assert validate_claims(decision.key_claims, [_record()], "EVAL01", "2025-01-31")["status"] == "unvalidated"
    path = write_report_tree({"market_report": "Prior report."}, "EVAL01", tmp_path)
    assert "Prior report." in path.read_text()
    assert not (tmp_path / "evidence.json").exists()
