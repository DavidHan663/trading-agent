"""Create a provenance-complete AAPL replay state from frozen public snapshots.

The generator never edits its input state.  Financial records are tied to the
SEC filing that made FY2026 information public; market and indicator records
retain their Yahoo Finance provider reference and historical observation date.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SEC_CIK = "0000320193"
SEC_ACCESSION = "000032019326000026"
SEC_DOCUMENT = "aapl-20260926.htm"
SEC_FILING_DATE = "2026-11-03"
SEC_PERIOD_END = "2026-09-26"
SEC_DOCUMENT_URL = (
    f"https://www.sec.gov/Archives/edgar/data/{int(SEC_CIK)}/{SEC_ACCESSION}/{SEC_DOCUMENT}"
)
YAHOO_HISTORY_URL = "https://finance.yahoo.com/quote/AAPL/history"

FINANCIAL_RECORD_IDS = {
    "E-5E89FBA52455", "E-A73D91EFE377", "E-60CF9D18428B",
    "E-602589DDAD14", "E-E1AB2695E30F", "E-71DDA3222977",
}
ANNUAL_FINANCIAL_RECORD_IDS = {
    "E-602589DDAD14", "E-E1AB2695E30F", "E-71DDA3222977",
}
MARKET_RECORD_IDS = {
    "E-718F38137E79", "E-16398D30A6A6", "E-68172898B7A2",
    "E-FB314DA560CF", "E-326FB5B80DFE", "E-24D59C5A489A",
    "E-7A3C18C17EAE", "E-18BB0549BA2F", "E-F6E554429D8C",
    "E-74A5BB0DEDAE",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sec_annual_fact(payload: dict, tag: str) -> float:
    units = payload["facts"]["us-gaap"][tag]["units"]
    for observations in units.values():
        for observation in observations:
            if (observation.get("form") == "10-K" and observation.get("fy") == 2026
                    and observation.get("filed") == SEC_FILING_DATE
                    and observation.get("end") == SEC_PERIOD_END):
                return float(observation["val"])
    raise ValueError(f"SEC snapshot is missing FY2026 10-K fact: {tag}")


def _validate_snapshots(submissions: dict, companyfacts: dict, yahoo_chart: dict) -> dict:
    recent = submissions["filings"]["recent"]
    filing = next(
        (
            {
                "filing_date": recent["filingDate"][index],
                "report_date": recent["reportDate"][index],
                "accession": recent["accessionNumber"][index].replace("-", ""),
                "document": recent["primaryDocument"][index],
            }
            for index, form in enumerate(recent["form"])
            if form == "10-K" and recent["filingDate"][index] == SEC_FILING_DATE
        ),
        None,
    )
    expected = {
        "filing_date": SEC_FILING_DATE,
        "report_date": SEC_PERIOD_END,
        "accession": SEC_ACCESSION,
        "document": SEC_DOCUMENT,
    }
    if filing != expected:
        raise ValueError(f"SEC filing snapshot does not match expected FY2026 10-K: {filing}")

    operating_cashflow = _sec_annual_fact(
        companyfacts, "NetCashProvidedByUsedInOperatingActivities",
    )
    capital_expenditure = _sec_annual_fact(
        companyfacts, "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    free_cashflow = operating_cashflow - capital_expenditure
    if free_cashflow != 136_056_000_000:
        raise ValueError(f"unexpected SEC FY2026 free cash flow: {free_cashflow}")

    chart = yahoo_chart["chart"]["result"][0]
    quote = chart["indicators"]["quote"][0]
    close = float(quote["close"][0])
    if abs(close - 356.27) > 0.01:
        raise ValueError(f"unexpected Yahoo close for 2026-11-13: {close}")
    return {
        "filing": filing,
        "operating_cashflow": operating_cashflow,
        "capital_expenditure": capital_expenditure,
        "free_cashflow": free_cashflow,
        "market_close": close,
    }


def build_provenance_state(state: dict, submissions: dict, companyfacts: dict, yahoo_chart: dict) -> tuple[dict, dict]:
    """Copy one AAPL state and enrich only records with absent provenance."""
    if state.get("company_of_interest") != "AAPL" or state.get("trade_date") != "2026-11-13":
        raise ValueError("expected the fixed AAPL state for 2026-11-13")
    verification = _validate_snapshots(submissions, companyfacts, yahoo_chart)
    result = copy.deepcopy(state)
    updated_ids = []
    for record in result.get("evidence", []):
        record_id = record.get("id")
        if record_id in FINANCIAL_RECORD_IDS:
            record.update({
                "source_url": SEC_DOCUMENT_URL,
                "source_provider": "sec_edgar",
                "source_kind": ("primary_filing" if record_id in ANNUAL_FINANCIAL_RECORD_IDS
                                else "derived_from_primary_filing"),
                "available_at": SEC_FILING_DATE,
                "period_end": SEC_PERIOD_END,
                "provenance_note": (
                    "The FY2026 annual filing was public on this date; free cash flow "
                    "was checked as operating cash flow minus capital expenditure."
                    if record_id == "E-71DDA3222977" else
                    "This financial series is tied to the FY2026 SEC 10-K; quarterly values "
                    "are derived from that filing where applicable."
                ),
            })
            updated_ids.append(record_id)
        elif record_id in MARKET_RECORD_IDS:
            record.update({
                "source_url": YAHOO_HISTORY_URL,
                "source_provider": "yfinance",
                "source_kind": "provider_reference",
                "provenance_note": "Historical market observation and derived indicators are tied to the Yahoo Finance AAPL history endpoint.",
            })
            updated_ids.append(record_id)
    missing = (FINANCIAL_RECORD_IDS | MARKET_RECORD_IDS) - set(updated_ids)
    if missing:
        raise ValueError(f"input state lacks expected provenance records: {sorted(missing)}")
    return result, {
        "updated_evidence_ids": sorted(updated_ids),
        "sec_document_url": SEC_DOCUMENT_URL,
        "sec_filing_date": SEC_FILING_DATE,
        "sec_period_end": SEC_PERIOD_END,
        "sec_free_cashflow": verification["free_cashflow"],
        "yahoo_close": verification["market_close"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-state", type=Path, required=True)
    parser.add_argument("--sec-submissions", type=Path, required=True)
    parser.add_argument("--sec-companyfacts", type=Path, required=True)
    parser.add_argument("--yahoo-chart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_hash_before = file_sha256(args.input_state)
    state = json.loads(args.input_state.read_text())
    submissions = json.loads(args.sec_submissions.read_text())
    companyfacts = json.loads(args.sec_companyfacts.read_text())
    yahoo_chart = json.loads(args.yahoo_chart.read_text())
    result, details = build_provenance_state(state, submissions, companyfacts, yahoo_chart)

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "state.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    snapshots = args.output / "source_snapshots"
    snapshots.mkdir()
    for source in (args.sec_submissions, args.sec_companyfacts, args.yahoo_chart):
        destination = snapshots / source.name
        destination.write_bytes(source.read_bytes())
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tradingagents.evaluation.aapl_provenance_state",
        "input_state": str(args.input_state),
        "input_state_sha256": input_hash_before,
        "output_state_sha256": file_sha256(args.output / "state.json"),
        "source_snapshots": {
            path.name: file_sha256(snapshots / path.name)
            for path in (args.sec_submissions, args.sec_companyfacts, args.yahoo_chart)
        },
        "source_files_unchanged": file_sha256(args.input_state) == input_hash_before,
        **details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
