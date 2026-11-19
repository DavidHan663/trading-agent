import json

from tradingagents.evaluation.aapl_provenance_state import (
    FINANCIAL_RECORD_IDS,
    MARKET_RECORD_IDS,
    SEC_DOCUMENT_URL,
    build_provenance_state,
)


def _state():
    records = []
    for evidence_id in FINANCIAL_RECORD_IDS | MARKET_RECORD_IDS:
        records.append({"id": evidence_id, "source_url": None, "available_at": None,
                        "period_end": "2026-09-30", "content": "fixture"})
    return {"company_of_interest": "AAPL", "trade_date": "2026-11-13", "evidence": records}


def _submissions():
    return {"filings": {"recent": {
        "form": ["10-K"], "filingDate": ["2026-11-03"], "reportDate": ["2026-09-26"],
        "accessionNumber": ["0000320193-26-000026"], "primaryDocument": ["aapl-20260926.htm"],
    }}}


def _fact(value):
    return {"units": {"USD": [{"form": "10-K", "fy": 2026, "filed": "2026-11-03",
                                  "end": "2026-09-26", "val": value}]}}


def _companyfacts():
    return {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": _fact(145_097_000_000),
        "PaymentsToAcquirePropertyPlantAndEquipment": _fact(9_041_000_000),
    }}}


def _yahoo():
    return {"chart": {"result": [{"indicators": {"quote": [{"close": [356.269989]}]}}]}}


def test_builds_new_state_and_preserves_input_records():
    state = _state()
    original = json.loads(json.dumps(state))
    result, manifest = build_provenance_state(state, _submissions(), _companyfacts(), _yahoo())
    assert state == original
    assert len(manifest["updated_evidence_ids"]) == len(FINANCIAL_RECORD_IDS | MARKET_RECORD_IDS)
    financial = next(record for record in result["evidence"] if record["id"] in FINANCIAL_RECORD_IDS)
    assert financial["source_url"] == SEC_DOCUMENT_URL
    assert financial["available_at"] == "2026-11-03"
    annual = next(record for record in result["evidence"] if record["id"] == "E-71DDA3222977")
    assert annual["source_kind"] == "primary_filing"
    market = next(record for record in result["evidence"] if record["id"] in MARKET_RECORD_IDS)
    assert market["source_provider"] == "yfinance"


def test_rejects_wrong_snapshot_values():
    bad = _yahoo()
    bad["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [1]
    try:
        build_provenance_state(_state(), _submissions(), _companyfacts(), bad)
    except ValueError as error:
        assert "unexpected Yahoo close" in str(error)
    else:
        raise AssertionError("invalid public snapshot was accepted")
