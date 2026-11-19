"""Offline evidence routing on the existing frozen diagnostic cases."""

import json

import pytest

from tradingagents.evaluation.dataset import available_evidence, frozen_response, load_dataset
from tradingagents.evidence import records_from_tool
from tradingagents.graph.evidence_gate import (
    MAX_REQUERIES,
    assess_evidence,
    limited_decision,
    requery_evidence,
    route_evidence,
)

_TOOLS = {"market": "get_verified_market_snapshot", "fundamentals": "get_fundamentals",
          "balance": "get_balance_sheet", "cashflow": "get_cashflow", "income": "get_income_statement",
          "news": "get_news", "macro": "get_global_news", "social": "stocktwits"}


def _state(case_id):
    case = next(c for c in load_dataset()[0]["cases"] if c["id"] == case_id)
    records = []
    for item in available_evidence(case):
        tool = _TOOLS.get(item["domain"])
        if tool:
            records += records_from_tool(tool, {"ticker": case["ticker"]},
                                         json.dumps({"records": [item]}), case["ticker"])
    return {"company_of_interest": case["ticker"], "trade_date": case["as_of"],
            "asset_type": "stock", "evidence": records,
            "requery_count": 0, "requery_attempted": []}


@pytest.mark.parametrize(("case_id", "expected"), [
    ("normal-01", "sufficient"), ("missing-01", "insufficient"),
    ("conflict-01", "insufficient"), ("stale-01", "insufficient"),
    ("cutoff-01", "sufficient"),
])
def test_frozen_case_routing(case_id, expected):
    state = _state(case_id)
    state.update(assess_evidence(state))
    assert state["evidence_assessment"]["status"] == expected
    assert route_evidence(state) == ("Bull Researcher" if expected == "sufficient" else "Evidence Requery")


def test_missing_price_requery_is_bounded_and_ends_without_rating(monkeypatch):
    state = _state("normal-01")
    state["evidence"] = [r for r in state["evidence"] if r["id"] != "M1"]
    case = next(c for c in load_dataset()[0]["cases"] if c["id"] == "normal-01")
    case = {**case, "evidence": [x for x in case["evidence"] if x["id"] != "M1"]}
    from tradingagents.graph import evidence_gate
    monkeypatch.setattr(evidence_gate.get_verified_market_snapshot, "func",
                        lambda **kwargs: frozen_response(case, "get_verified_market_snapshot", kwargs))
    state.update(assess_evidence(state))
    assert state["evidence_assessment"]["missing"] == ["market_price_unverified"]
    update = requery_evidence(state)
    state["evidence"] += update["evidence"]
    state["requery_count"] = update["requery_count"]
    state["requery_attempted"] += update["requery_attempted"]
    state.update(assess_evidence(state))
    assert route_evidence(state) == "Evidence Limited"
    decision = limited_decision(state)["final_trade_decision"]
    assert "无法判断" in decision and "止损" in decision
    assert "缺少可核验价格" in decision
    assert "market_price_unverified" not in decision
    assert state["requery_count"] <= MAX_REQUERIES


def test_recent_indicator_without_a_close_is_not_a_verified_price():
    state = _state("normal-01")
    market = next(r for r in state["evidence"] if r["id"] == "M1")
    market["content"] = "RSI=55, MACD=1.2; no price observation."
    result = assess_evidence(state)["evidence_assessment"]
    assert "market_price_unverified" in result["missing"]


def test_conflicting_revenue_is_not_reconciled_by_same_frozen_response(monkeypatch):
    state = _state("conflict-01")
    case = next(c for c in load_dataset()[0]["cases"] if c["id"] == "conflict-01")
    from tradingagents.graph import evidence_gate
    monkeypatch.setattr(evidence_gate.get_fundamentals, "func",
                        lambda **kwargs: frozen_response(case, "get_fundamentals", kwargs))
    state.update(assess_evidence(state))
    assert state["evidence_assessment"]["conflicts"][0]["evidence_ids"] == ["F1", "F2"]
    update = requery_evidence(state)
    state["evidence"] += update["evidence"]
    state["requery_count"] = update["requery_count"]
    state["requery_attempted"] += update["requery_attempted"]
    state.update(assess_evidence(state))
    assert route_evidence(state) == "Evidence Limited"
