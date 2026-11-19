"""Run the blinded semantic-support reviewer on frozen claim/evidence pairs."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from dotenv import load_dotenv

from tradingagents.evaluation.run import current_commit, source_fingerprint, write_json
from tradingagents.evaluation.semantic import (
    SemanticReview,
    load_semantic_dataset,
    semantic_metrics,
    semantic_prompt,
    validate_semantic_review,
)
from tradingagents.evaluation.trace import EvaluationTrace

CONSISTENCY_CASE_IDS = {
    "supported-01", "supported-02",
    "partial-01", "partial-02",
    "unsupported-01", "unsupported-02",
    "contradicted-01", "contradicted-02",
    "insufficient-01", "insufficient-02",
}


def _public_config(config):
    return {key: config.get(key) for key in (
        "llm_provider", "deep_think_llm", "backend_url", "temperature",
        "max_tokens", "llm_max_retries",
    )}


def create_reviewer_llm(config):
    from tradingagents.llm_clients import create_llm_client

    kwargs = {}
    for key in ("temperature", "max_tokens", "llm_max_retries"):
        if config.get(key) is not None:
            kwargs[{"llm_max_retries": "max_retries"}.get(key, key)] = config[key]
    client = create_llm_client(
        provider=config["llm_provider"],
        model=config["deep_think_llm"],
        base_url=config.get("backend_url"),
        **kwargs,
    )
    return client.get_llm()


def run_semantic_suite(output, llm, *, case_ids=None, consistency=False,
                       input_price=None, output_price=None):
    dataset, dataset_hash = load_semantic_dataset()
    cases = dataset["cases"]
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown semantic case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in case_ids]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    structured = llm.with_structured_output(SemanticReview)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": dataset_hash,
        "source_sha256": source_fingerprint(),
        "git_commit": current_commit(),
        "case_ids": [case["id"] for case in cases],
        "consistency_case_ids": sorted(CONSISTENCY_CASE_IDS & {case["id"] for case in cases})
        if consistency else [],
        "config": {},
        "cost_assumption": {
            "input_usd_per_million": input_price,
            "output_usd_per_million": output_price,
        },
        "annotation_status": dataset["annotation_status"],
        "blinding": "label and proxy rationale excluded from model prompt",
    }
    write_json(output / "manifest.json", manifest)
    records = []
    consecutive_failures = 0
    for case in cases:
        repetitions = 2 if consistency and case["id"] in CONSISTENCY_CASE_IDS else 1
        for repetition in range(1, repetitions + 1):
            directory = output / f"{case['id']}-r{repetition}"
            directory.mkdir()
            prompt = semantic_prompt(case, dataset)
            write_json(directory / "input.json", {
                "case_id": case["id"], "as_of": case["as_of"], "claim": case["claim"],
                "evidence_ids": case["evidence_ids"],
            })
            write_json(directory / "reference.json", {
                "expected": case["label"], "proxy_rationale": case["rationale"],
            })
            trace = EvaluationTrace()
            started = monotonic()
            record = {
                "case_id": case["id"], "repetition": repetition,
                "expected": case["label"], "status": "failed", "error_type": None,
                "predicted": None, "validation_issues": [],
            }
            try:
                prediction = structured.invoke(prompt, config={"callbacks": [trace]})
                if prediction is None:
                    raise ValueError("No parsed structured result")
                issues = validate_semantic_review(prediction, case, dataset)
                record["validation_issues"] = issues
                if issues:
                    record["error_type"] = "InvalidSemanticReview"
                else:
                    record.update(status="completed", predicted=prediction.verdict)
                    write_json(directory / "prediction.json", prediction)
            except Exception as exc:
                record["error_type"] = type(exc).__name__
            record["seconds"] = monotonic() - started
            record["usage"] = trace.summary()
            write_json(directory / "trace.json", trace.events)
            write_json(directory / "result.json", record)
            records.append(record)
            consecutive_failures = consecutive_failures + 1 if record["status"] == "failed" else 0
            write_json(output / "runs.json", records)
            write_json(output / "summary.json", semantic_metrics(
                records, input_price=input_price, output_price=output_price,
            ))
            if consecutive_failures >= 3:
                write_json(output / "stopped.json", {
                    "reason": "three_consecutive_failures",
                    "last_error_types": [item["error_type"] for item in records[-3:]],
                })
                return output
    return output


def main():
    parser = argparse.ArgumentParser(description="Blinded frozen semantic-support evaluation")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--review", type=Path,
                        help="Recompute summary for an existing semantic run without model calls")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--consistency", action="store_true",
                        help="Repeat a fixed stratified set of ten cases once")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    args = parser.parse_args()
    if args.review:
        records = json.loads((args.review / "runs.json").read_text())
        manifest = json.loads((args.review / "manifest.json").read_text())
        prices = manifest.get("cost_assumption", {})
        write_json(args.review / "summary.json", semantic_metrics(
            records,
            input_price=prices.get("input_usd_per_million"),
            output_price=prices.get("output_usd_per_million"),
        ))
        print(f"Summary updated: {args.review / 'summary.json'}")
        return
    if not args.allow_model_calls:
        parser.error("Semantic evaluation requires --allow-model-calls")
    prices = [args.input_usd_per_million, args.output_usd_per_million]
    if any(value is not None and value < 0 for value in prices):
        parser.error("Token prices cannot be negative")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    llm = create_reviewer_llm(config)
    output = args.output or Path("eval_results") / f"semantic-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    logging.disable(logging.CRITICAL)
    directory = run_semantic_suite(
        output, llm, case_ids=args.cases, consistency=args.consistency,
        input_price=prices[0], output_price=prices[1],
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = _public_config(config)
    write_json(manifest_path, manifest)
    print(f"Artifacts saved: {directory}")


if __name__ == "__main__":
    main()
