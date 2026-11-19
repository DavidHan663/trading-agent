"""Controlled live validation for the runtime shadow semantic claim gate."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from tradingagents.agents.schemas import CalculationInput, EvidenceClaim
from tradingagents.evaluation.run import current_commit, source_fingerprint, write_json
from tradingagents.evaluation.semantic_run import _public_config, create_reviewer_llm
from tradingagents.evidence import validate_claims
from tradingagents.graph.semantic_claim_gate import create_semantic_claim_gate

REPEAT_CASE_IDS = {"runtime-fact-supported", "runtime-fact-contradicted", "runtime-judgment-supported"}


def _record(evidence_id, content):
    return {
        "id": evidence_id, "tool": "controlled_fixture", "target_ticker": "EVAL01",
        "requested_ticker": "EVAL01", "source_ticker": "EVAL01",
        "source_url": f"https://example.test/{evidence_id.lower()}",
        "available_at": "2025-01-30", "period_end": "2024-12-31",
        "fetched_at": "2025-01-31T00:00:00+00:00", "content": content,
    }


def controlled_cases():
    revenue = _record("F1", "Quarterly revenue was 120 million USD versus 100 million USD previously.")
    market = _record("M1", "The closing price was 50 USD and the 50-day moving average was 49 USD.")
    cash = _record("CASH1", "Cash was 30 million USD and total debt was 20 million USD.")
    supported_fact = EvidenceClaim(
        claim_id="C1", claim_type="fact", text="Quarterly revenue was 120 million USD.",
        kind="financial", evidence_ids=["F1"], numeric_value=120, unit="million USD",
    )
    contradicted_fact = EvidenceClaim(
        claim_id="C1", claim_type="fact", text="Quarterly revenue declined from the prior period.",
        kind="event", evidence_ids=["F1"],
    )
    missing_fact = EvidenceClaim(
        claim_id="C1", claim_type="fact", text="The chief executive resigned during the quarter.",
        kind="event", evidence_ids=["M1"],
    )
    calculation = EvidenceClaim(
        claim_id="C1", claim_type="calculation", text="Cash exceeded total debt by 10 million USD.",
        kind="financial", evidence_ids=[],
        calculation_inputs=[
            CalculationInput(name="cash", value=30, unit="million USD", evidence_ids=["CASH1"]),
            CalculationInput(name="debt", value=20, unit="million USD", evidence_ids=["CASH1"]),
        ],
        formula="cash - debt", result_value=10, result_unit="million USD",
    )
    momentum_judgment = EvidenceClaim(
        claim_id="C2", claim_type="judgment",
        text="The reported revenue comparison indicates positive historical revenue momentum.",
        kind="rating_reason", evidence_ids=[], basis_claim_ids=["C1"],
    )
    guarantee_judgment = EvidenceClaim(
        claim_id="C2", claim_type="judgment",
        text="The revenue increase guarantees that the share price will rise next year.",
        kind="rating_reason", evidence_ids=[], basis_claim_ids=["C1"],
    )
    return [
        {"id": "runtime-fact-supported", "expected": "allow", "records": [revenue],
         "claims": [supported_fact]},
        {"id": "runtime-fact-contradicted", "expected": "review", "records": [revenue],
         "claims": [contradicted_fact]},
        {"id": "runtime-fact-missing", "expected": "review", "records": [market],
         "claims": [missing_fact]},
        {"id": "runtime-calculation-supported", "expected": "allow", "records": [cash],
         "claims": [calculation]},
        {"id": "runtime-judgment-supported", "expected": "allow", "records": [revenue],
         "claims": [supported_fact, momentum_judgment]},
        {"id": "runtime-judgment-unsupported", "expected": "review", "records": [revenue],
         "claims": [supported_fact, guarantee_judgment]},
    ]


def _percentile(values, proportion):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def _summary(records):
    primary = [record for record in records if record["repetition"] == 1]
    repeated = {}
    for record in records:
        repeated.setdefault(record["case_id"], []).append(record["predicted"])
    repeated = [values for values in repeated.values() if len(values) > 1]
    usages = [record["gate"]["usage"] for record in records]
    return {
        "runs": len(records),
        "primary_runs": len(primary),
        "logical_model_invocations": sum(record["gate"]["model_invocations"] for record in records),
        "provider_calls_observed": sum(usage["provider_calls_observed"] for usage in usages),
        "accuracy": sum(record["expected"] == record["predicted"] for record in primary) / len(primary),
        "unsafe_allow_count": sum(
            record["expected"] == "review" and record["predicted"] == "allow" for record in primary
        ),
        "repeat_consistency": (
            sum(len(set(values)) == 1 for values in repeated) / len(repeated)
            if repeated else None
        ),
        "latency_p50_seconds": _percentile([record["gate"]["seconds"] for record in records], 0.5),
        "latency_p95_seconds": _percentile([record["gate"]["seconds"] for record in records], 0.95),
        "input_tokens": sum(usage["input_tokens"] for usage in usages),
        "output_tokens": sum(usage["output_tokens"] for usage in usages),
        "token_usage_complete": all(usage["token_usage_complete"] for usage in usages),
        "label_source": "Codex expert-proxy expectations; controlled synthetic integration cases.",
    }


def run(output, llm):
    output = Path(output).resolve()
    node = create_semantic_claim_gate(llm)
    cases = controlled_cases()
    if output.exists():
        if not (output / "manifest.json").is_file() or not (output / "runs.json").is_file():
            raise ValueError("Cannot resume an incomplete directory without manifest and runs")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("initial_source_sha256", manifest.get("source_sha256"))
        manifest["resume_source_sha256"] = source_fingerprint()
        manifest["resumed_after"] = "empty-repeat-summary fix; runtime gate logic unchanged"
        write_json(manifest_path, manifest)
        records = json.loads((output / "runs.json").read_text())
    else:
        output.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_fingerprint(),
            "git_commit": current_commit(),
            "case_ids": [case["id"] for case in cases],
            "repeat_case_ids": sorted(REPEAT_CASE_IDS),
            "maximum_provider_calls": 12,
            "config": {},
            "external_data_calls": 0,
        }
        write_json(output / "manifest.json", manifest)
        records = []
    completed = {(record["case_id"], record["repetition"]) for record in records}
    for case in cases:
        repetitions = 2 if case["id"] in REPEAT_CASE_IDS else 1
        for repetition in range(1, repetitions + 1):
            if (case["id"], repetition) in completed:
                continue
            validation = validate_claims(case["claims"], case["records"], "EVAL01", "2025-01-31")
            state = {
                "company_of_interest": "EVAL01", "trade_date": "2025-01-31",
                "citation_validation": validation, "evidence": case["records"],
                "final_trade_decision": "**Rating**: Hold",
            }
            gate = node(state)["semantic_claim_validation"]
            record = {
                "case_id": case["id"], "repetition": repetition,
                "expected": case["expected"], "predicted": gate["status"],
                "gate": gate,
            }
            records.append(record)
            write_json(output / "runs.json", records)
            write_json(output / "summary.json", _summary(records))
            if _summary(records)["provider_calls_observed"] > 12:
                raise RuntimeError("Provider call budget exceeded")
    return output


def main():
    parser = argparse.ArgumentParser(description="Controlled runtime semantic-gate validation")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("Runtime semantic-gate validation requires --allow-model-calls")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["llm_max_retries"] = 0
    llm = create_reviewer_llm(config)
    output = args.output or Path("eval_results") / (
        f"semantic-gate-runtime-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    directory = run(output, llm)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = _public_config(config)
    write_json(manifest_path, manifest)
    print(f"Artifacts saved: {directory}")


if __name__ == "__main__":
    main()
