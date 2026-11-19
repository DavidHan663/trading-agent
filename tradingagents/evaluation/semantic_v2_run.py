"""Run the blinded hierarchical semantic reviewer on frozen v2 cases."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import ValidationError

from tradingagents.evaluation.run import current_commit, source_fingerprint, write_json
from tradingagents.evaluation.semantic_run import _public_config, create_reviewer_llm
from tradingagents.evaluation.semantic_v2 import (
    HierarchicalSemanticReview,
    load_semantic_v2_dataset,
    semantic_v2_metrics,
    semantic_v2_prompt,
    validate_semantic_v2_review,
)
from tradingagents.evaluation.trace import EvaluationTrace

CONSISTENCY_CASE_IDS_V2 = {
    "v2-allow-01", "v2-allow-05", "v2-allow-09", "v2-allow-13",
    "v2-review-01", "v2-review-04", "v2-review-09", "v2-review-13",
    "v2-review-21", "v2-review-23",
}


def _semantic_error_summary(exc):
    """Describe parser/schema failures without persisting prompts or model text."""
    summary = {"error_type": type(exc).__name__}
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            errors = current.errors(include_url=False)
            locations, error_types = [], []
            for error in errors:
                parts = []
                for part in error.get("loc", ()):
                    if isinstance(part, int):
                        parts.append(str(part))
                    elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", part):
                        parts.append(part)
                    else:
                        parts.append("<field>")
                location = ".".join(parts)
                if location and location not in locations:
                    locations.append(location)
                error_type = error.get("type")
                if isinstance(error_type, str) and error_type not in error_types:
                    error_types.append(error_type)
            summary.update({
                "validation_error_count": len(errors),
                "validation_error_locations": locations[:8],
                "validation_error_types": error_types[:8],
            })
            break
        current = current.__cause__ or current.__context__
    return summary


def run_semantic_v2_suite(output, llm, *, case_ids=None, consistency=False,
                          input_price=None, output_price=None,
                          dataset_loader=load_semantic_v2_dataset,
                          prompt_builder=semantic_v2_prompt,
                          trace_content=True):
    dataset, dataset_hash = dataset_loader()
    cases = dataset["cases"]
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown semantic-v2 case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in case_ids]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    structured = llm.with_structured_output(HierarchicalSemanticReview)
    manifest = {
        "schema_version": dataset.get("schema_version", 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": dataset_hash,
        "source_sha256": source_fingerprint(),
        "git_commit": current_commit(),
        "case_ids": [case["id"] for case in cases],
        "consistency_case_ids": sorted(
            CONSISTENCY_CASE_IDS_V2 & {case["id"] for case in cases}
        ) if consistency else [],
        "config": {},
        "cost_assumption": {
            "input_usd_per_million": input_price,
            "output_usd_per_million": output_price,
        },
        "annotation_status": dataset["annotation_status"],
        "trace_content_captured": trace_content,
        "blinding": "gate decision, reason flags, and proxy rationale excluded from model prompt",
        "maximum_model_calls": len(cases) + (
            len(CONSISTENCY_CASE_IDS_V2 & {case["id"] for case in cases}) if consistency else 0
        ),
    }
    write_json(output / "manifest.json", manifest)
    records = []
    consecutive_failures = 0
    for case in cases:
        repetitions = 2 if consistency and case["id"] in CONSISTENCY_CASE_IDS_V2 else 1
        for repetition in range(1, repetitions + 1):
            directory = output / f"{case['id']}-r{repetition}"
            directory.mkdir()
            prompt = prompt_builder(case, dataset)
            write_json(directory / "input.json", {
                "case_id": case["id"], "as_of": case["as_of"], "claim": case["claim"],
                "evidence_ids": case["evidence_ids"],
            })
            write_json(directory / "reference.json", {
                "expected_decision": case["gate_decision"],
                "expected_reasons": case["reason_flags"],
                "proxy_rationale": case["rationale"],
            })
            trace = EvaluationTrace(capture_content=trace_content)
            started = monotonic()
            record = {
                "case_id": case["id"], "repetition": repetition,
                "expected_decision": case["gate_decision"],
                "expected_reasons": case["reason_flags"],
                "status": "failed", "error_type": None,
                "predicted_decision": None, "predicted_reasons": [],
                "validation_issues": [],
            }
            try:
                prediction = structured.invoke(prompt, config={"callbacks": [trace]})
                if prediction is None:
                    raise ValueError("No parsed structured result")
                issues = validate_semantic_v2_review(prediction, case, dataset)
                record["validation_issues"] = issues
                if issues:
                    record["error_type"] = "InvalidHierarchicalSemanticReview"
                    record["error_summary"] = {"error_type": "InvalidHierarchicalSemanticReview"}
                else:
                    record.update(
                        status="completed",
                        predicted_decision=prediction.decision,
                        predicted_reasons=prediction.reason_flags,
                    )
                    write_json(directory / "prediction.json", prediction)
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                record["error_summary"] = _semantic_error_summary(exc)
            record["seconds"] = monotonic() - started
            record["usage"] = trace.summary()
            write_json(directory / "trace.json", trace.events)
            write_json(directory / "result.json", record)
            records.append(record)
            consecutive_failures = consecutive_failures + 1 if record["status"] == "failed" else 0
            write_json(output / "runs.json", records)
            write_json(output / "summary.json", semantic_v2_metrics(
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
    parser = argparse.ArgumentParser(description="Blinded hierarchical semantic evaluation")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--review", type=Path,
                        help="Recompute summary for an existing v2 run without model calls")
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
        write_json(args.review / "summary.json", semantic_v2_metrics(
            records,
            input_price=prices.get("input_usd_per_million"),
            output_price=prices.get("output_usd_per_million"),
        ))
        print(f"Summary updated: {args.review / 'summary.json'}")
        return
    if not args.allow_model_calls:
        parser.error("Semantic-v2 evaluation requires --allow-model-calls")
    prices = [args.input_usd_per_million, args.output_usd_per_million]
    if any(value is not None and value < 0 for value in prices):
        parser.error("Token prices cannot be negative")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    llm = create_reviewer_llm(config)
    output = args.output or Path("eval_results") / (
        f"semantic-v2-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    logging.disable(logging.CRITICAL)
    directory = run_semantic_v2_suite(
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
