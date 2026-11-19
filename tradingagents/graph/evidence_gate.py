"""Deterministic evidence gate between analyst research and the trading debate."""

import re
from datetime import date

from tradingagents.agents.utils.agent_utils import get_fundamentals, get_verified_market_snapshot
from tradingagents.evidence import records_from_tool, unique_records

MAX_REQUERIES = 2
MAX_PRICE_AGE_DAYS = 7
_MARKET_TOOLS = {"get_stock_data", "get_indicators", "get_verified_market_snapshot"}
_FUNDAMENTAL_TOOLS = {"get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"}
_REVENUE = re.compile(r"\brevenue\s*=\s*([\d,.]+)\s+(million|billion)\s+([A-Z]{3})\b", re.I)
_SNAPSHOT_DATE = re.compile(r"Latest trading row used:\s*(\d{4}-\d{2}-\d{2})")
_CLOSE = re.compile(r"(?:\bclose\s*[=:]\s*\$?\d|\|\s*Close\s*\|\s*\$?\d)", re.I)
_GAP_LABELS = {"market_price_unverified": "缺少可核验价格",
               "market_price_stale": "最近价格已过期",
               "financial_data_missing": "缺少可用财务数据"}


def _market_date(record):
    match = _SNAPSHOT_DATE.search(record.get("content", ""))
    if match:
        return match.group(1)
    return str(record["available_at"])[:10] if record.get("available_at") else None


def assess_evidence(state):
    """Require a recent market observation and financial data for a stock."""
    records = unique_records(state.get("evidence", []))
    as_of = date.fromisoformat(state["trade_date"])
    missing, conflicts = [], []
    market = [r for r in records if r.get("tool") in _MARKET_TOOLS
              and _CLOSE.search(r.get("content", ""))]
    dated = []
    for record in market:
        raw = _market_date(record)
        if raw:
            try:
                observed = date.fromisoformat(raw)
                if observed <= as_of:
                    dated.append(observed)
            except ValueError:
                pass
    if not dated:
        missing.append("market_price_unverified")
    elif (as_of - max(dated)).days > MAX_PRICE_AGE_DAYS:
        missing.append("market_price_stale")

    if state.get("asset_type", "stock") != "crypto":
        financial = [r for r in records if r.get("tool") in _FUNDAMENTAL_TOOLS]
        if not financial:
            missing.append("financial_data_missing")
        values = {}
        for record in financial:
            # Only compare records with explicit period metadata and a literal
            # same-unit revenue field. Unstructured prose is not comparable.
            match = _REVENUE.search(record.get("content", ""))
            if match and record.get("period_end"):
                key = (str(record["period_end"]), match.group(2).lower(), match.group(3).upper())
                values.setdefault(key, {}).setdefault(float(match.group(1).replace(",", "")), set()).add(record["id"])
        for (period, unit, currency), amounts in values.items():
            if len(amounts) > 1:
                conflicts.append({"metric": "revenue", "period_end": period,
                                  "unit": f"{unit} {currency}",
                                  "evidence_ids": sorted(set().union(*amounts.values()))})

    attempted = set(state.get("requery_attempted", []))
    next_tool = None
    if any(x.startswith("market_price") for x in missing) and "get_verified_market_snapshot" not in attempted:
        next_tool = "get_verified_market_snapshot"
    elif ("financial_data_missing" in missing or conflicts) and "get_fundamentals" not in attempted:
        next_tool = "get_fundamentals"
    status = "sufficient" if not missing and not conflicts else "insufficient"
    return {"evidence_assessment": {"status": status, "missing": missing,
                                    "conflicts": conflicts, "next_tool": next_tool,
                                    "requery_count": state.get("requery_count", 0)}}


def route_evidence(state):
    result = state["evidence_assessment"]
    if result["status"] == "sufficient":
        return "Bull Researcher"
    if result["next_tool"] and state.get("requery_count", 0) < MAX_REQUERIES:
        return "Evidence Requery"
    return "Evidence Limited"


def requery_evidence(state):
    name = state["evidence_assessment"]["next_tool"]
    ticker, as_of = state["company_of_interest"], state["trade_date"]
    arguments = ({"symbol": ticker, "curr_date": as_of} if name == "get_verified_market_snapshot"
                 else {"ticker": ticker, "curr_date": as_of})
    tool = get_verified_market_snapshot if name == "get_verified_market_snapshot" else get_fundamentals
    try:
        output = str(tool.invoke(arguments))
    except Exception:
        output = "<unavailable> Tool call failed."
    return {"evidence": records_from_tool(name, arguments, output, ticker),
            "requery_count": state.get("requery_count", 0) + 1,
            "requery_attempted": [name]}


def limited_decision(state):
    result = state["evidence_assessment"]
    items = [_GAP_LABELS.get(gap, gap) for gap in result["missing"]]
    items += [f"同期营收来源冲突（{', '.join(c['evidence_ids'])}）" for c in result["conflicts"]]
    return {"final_trade_decision": "证据不足，无法判断是否调整敞口。\n\n待解决：" + "；".join(items)
            + "。本轮不提供确定价格、止损或方向性交易建议。",
            "evidence_assessment": {**result, "status": "insufficient", "requery_count": state.get("requery_count", 0)}}
