"""Run the version-3 blinded hierarchical semantic evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from tradingagents.evaluation.run import write_json
from tradingagents.evaluation.semantic_v2 import semantic_v2_metrics
from tradingagents.evaluation.semantic_v2_run import run_semantic_v2_suite
from tradingagents.evaluation.semantic_run import _public_config, create_reviewer_llm
from tradingagents.evaluation.semantic_v3 import load_semantic_v3_dataset, semantic_v3_prompt


def run_semantic_v3_suite(output, llm, *, case_ids=None, consistency=False,
                          input_price=None, output_price=None):
    return run_semantic_v2_suite(
        output,
        llm,
        case_ids=case_ids,
        consistency=consistency,
        input_price=input_price,
        output_price=output_price,
        dataset_loader=load_semantic_v3_dataset,
        prompt_builder=semantic_v3_prompt,
        trace_content=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Run blinded semantic-v3 evaluation")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--review", type=Path,
                        help="Recompute summary for an existing v3 run without model calls")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--consistency", action="store_true")
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
        parser.error("Semantic-v3 evaluation requires --allow-model-calls")
    prices = [args.input_usd_per_million, args.output_usd_per_million]
    if any(value is not None and value < 0 for value in prices):
        parser.error("Token prices cannot be negative")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    llm = create_reviewer_llm(config)
    output = args.output or Path("eval_results") / (
        f"semantic-v3-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    logging.disable(logging.CRITICAL)
    directory = run_semantic_v3_suite(
        output,
        llm,
        case_ids=args.cases,
        consistency=args.consistency,
        input_price=prices[0],
        output_price=prices[1],
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = _public_config(config)
    write_json(manifest_path, manifest)
    print(f"Artifacts saved: {directory}")


if __name__ == "__main__":
    main()
