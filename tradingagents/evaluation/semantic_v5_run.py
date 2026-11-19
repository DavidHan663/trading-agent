"""Run the frozen semantic-v5 holdout when explicitly enabled."""

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
from tradingagents.evaluation.semantic_v5 import load_semantic_v5_dataset, semantic_v5_prompt


def run_semantic_v5_suite(output, llm, *, case_ids=None):
    """Run cases once each with sanitized traces and a frozen dataset loader."""
    return run_semantic_v2_suite(
        output,
        llm,
        case_ids=case_ids,
        dataset_loader=load_semantic_v5_dataset,
        prompt_builder=semantic_v5_prompt,
        trace_content=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Run blinded semantic-v5 holdout")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("Semantic-v5 evaluation requires --allow-model-calls")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["llm_max_retries"] = 0
    llm = create_reviewer_llm(config)
    output = args.output or Path("eval_results") / (
        f"semantic-v5-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    logging.disable(logging.CRITICAL)
    directory = run_semantic_v5_suite(output, llm)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = _public_config(config)
    manifest["maximum_model_calls"] = len(manifest["case_ids"])
    manifest["holdout_policy"] = (
        "one provider attempt per case; SDK retries disabled; no case repeats; "
        "stop after three consecutive failures"
    )
    write_json(manifest_path, manifest)
    print(f"Artifacts saved: {directory}")


if __name__ == "__main__":
    main()
