"""Load and validate versioned synthetic research cases and reference labels."""

import hashlib
import json
from datetime import date
from pathlib import Path

DATASET = Path(__file__).parent / "fixtures" / "cases.v1.json"


def load_dataset(path=DATASET):
    raw = Path(path).read_bytes()
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 1 or dataset.get("kind") != "synthetic":
        raise ValueError("Expected version-1 synthetic evaluation dataset")
    seen = set()
    for case in dataset["cases"]:
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        date.fromisoformat(case["as_of"])
        evidence_ids = {item["id"] for item in case["evidence"]}
        if len(evidence_ids) != len(case["evidence"]):
            raise ValueError("Duplicate evidence ID")
        for item in case["evidence"]:
            date.fromisoformat(item["available_at"])
            date.fromisoformat(item["period_end"])
            if not item["source"] or not item["content"]:
                raise ValueError("Evidence needs a source and content")
        if not case["reference"]["facts"] or not case["reference"]["forbidden_claims"]:
            raise ValueError("Reference facts and forbidden claims are required")
        for fact in case["reference"]["facts"]:
            if not set(fact["evidence_ids"]) <= evidence_ids:
                raise ValueError("Reference points to unknown evidence")
        if case["reference"]["evidence_status"] not in {"sufficient", "insufficient", "conflicting"}:
            raise ValueError("Invalid reference evidence status")
    if not seen:
        raise ValueError("Dataset cannot be empty")
    return dataset, hashlib.sha256(raw).hexdigest()


def available_evidence(case):
    """Availability date, not fiscal-period date, controls historical access."""
    return [item for item in case["evidence"] if item["available_at"] <= case["as_of"]]


TOOL_DOMAINS = {
    "get_stock_data": "market", "get_indicators": "market",
    "get_verified_market_snapshot": "market", "get_fundamentals": "fundamentals",
    "get_balance_sheet": "balance", "get_cashflow": "cashflow",
    "get_income_statement": "income", "get_news": "news",
    "get_global_news": "macro", "get_macro_indicators": "macro",
    "get_insider_transactions": "insiders", "get_prediction_markets": "prediction",
    "stocktwits": "social", "reddit": "social",
}


def frozen_response(case, tool_name, arguments=None):
    """Replay a tool's frozen domain; never fall through to a network fetch.

    This evaluates reasoning on supplied facts, not vendor adapters or indicator
    calculation. Arguments are recorded by the runner and checked for identity
    and future requests. It is not an exact API-response recording.
    """
    arguments = arguments or {}
    for key in ("symbol", "ticker"):
        if key in arguments and str(arguments[key]).upper() != case["ticker"]:
            return "<unavailable> Requested instrument is outside this frozen case."
    for key in ("curr_date", "current_date", "end_date"):
        if key in arguments and str(arguments[key])[:10] > case["as_of"]:
            return "<unavailable> Requested date is after the evaluation cutoff."
    if tool_name not in TOOL_DOMAINS:
        raise ValueError(f"Unrecorded tool: {tool_name}")
    records = [item for item in available_evidence(case)
               if item["domain"] == TOOL_DOMAINS[tool_name]]
    start = arguments.get("start_date")
    if start and TOOL_DOMAINS[tool_name] in {"news", "social", "macro"}:
        records = [item for item in records if item["available_at"] >= str(start)[:10]]
    if not records:
        return "<unavailable> No evidence available for this source at the analysis cutoff."
    return json.dumps({"ticker": case["ticker"], "as_of": case["as_of"], "records": records}, ensure_ascii=False)
