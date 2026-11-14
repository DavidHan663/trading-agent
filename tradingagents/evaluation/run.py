"""Run the current graph on frozen evidence, or smoke-test artifacts offline."""

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from uuid import uuid4

from .dataset import available_evidence, frozen_response, load_dataset
from .review import aggregate, review_template
from .trace import EvaluationTrace


def write_json(path, value):
    def serialize(item):
        from pydantic import BaseModel

        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"Unsupported artifact value: {type(item).__name__}")

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serialize) + "\n", encoding="utf-8")


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("agents", "graph", "evaluation", "llm_clients"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    digest.update((root / "default_config.py").read_bytes())
    return digest.hexdigest()


def current_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_current_graph(case, config, trace):
    """Use existing prompts, role graph and schemas, replacing all market I/O.

    This patch scope is evaluation-only and process-global. Run cases serially
    in a dedicated CLI process, never inside the web server.
    """
    import tradingagents.dataflows.config as data_config
    from tradingagents.agents.utils import agent_utils
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sentiment = importlib.import_module("tradingagents.agents.analysts.sentiment_analyst")
    saved_config = copy.deepcopy(data_config._config)

    def replay(name, original):
        @wraps(original)
        def replacement(*args, **kwargs):
            arguments = dict(inspect.signature(original).bind_partial(*args, **kwargs).arguments)
            result = frozen_response(case, name, arguments)
            trace.event("frozen_tool", name=name, arguments=arguments, output=result)
            return result
        return replacement

    class FrozenGraph(TradingAgentsGraph):
        def resolve_instrument_context(self, ticker, asset_type="stock"):
            return case["instrument"]

    try:
        with ExitStack() as stack:
            for name in agent_utils.__all__:
                candidate = getattr(agent_utils, name)
                if name.startswith("get_") and hasattr(candidate, "func"):
                    stack.enter_context(patch.object(candidate, "func", replay(name, candidate.func)))
            for name, alias in (("fetch_stocktwits_messages", "stocktwits"),
                                ("fetch_reddit_posts", "reddit")):
                stack.enter_context(patch.object(sentiment, name, replay(alias, getattr(sentiment, name))))
            graph = FrozenGraph(config=config, callbacks=[trace])
            state, signal = graph.propagate(case["ticker"], case["as_of"])
            return state, signal
    finally:
        data_config._config = saved_config


def run_suite(output, *, mode="smoke", case_ids=None, repeat=1, limit=None,
              input_price=None, output_price=None):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.reporting import write_report_tree

    if mode not in {"smoke", "model"} or repeat < 1 or (limit is not None and limit < 1):
        raise ValueError("Invalid run mode or count")
    dataset, dataset_hash = load_dataset()
    cases = dataset["cases"]
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in case_ids]
    if limit:
        cases = cases[:limit]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update({"memory_log_path": None, "checkpoint_enabled": False,
                   "results_dir": str(output / "engine"), "data_cache_dir": str(output / "cache")})
    public_config = {key: config.get(key) for key in (
        "llm_provider", "deep_think_llm", "quick_think_llm", "temperature", "max_tokens",
        "llm_max_retries", "max_debate_rounds", "max_risk_discuss_rounds", "output_language",
    )}
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode, "dataset_version": dataset["dataset_version"],
                "dataset_sha256": dataset_hash, "source_sha256": source_fingerprint(),
                "git_commit": current_commit(), "case_ids": [c["id"] for c in cases],
                "repetitions": repeat, "config": public_config,
                "cost_assumption": {"input_usd_per_million": input_price,
                                    "output_usd_per_million": output_price},
                "annotation_status": dataset["annotation_status"],
                "market_data_mode": "frozen domain evidence; not exact API response replay"}
    manifest["python_version"] = sys.version.split()[0]
    manifest["library_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("langchain-core", "langgraph", "pydantic")
    }
    write_json(output / "manifest.json", manifest)
    records, reviews = [], []
    for case in cases:
        for repetition in range(1, repeat + 1):
            directory = output / f"{case['id']}-r{repetition}"
            directory.mkdir()
            # Only the admissible evidence enters model/tool context. Reference
            # judgments stay in a separate reviewer artifact, never in prompts.
            write_json(directory / "input.json", {key: case[key] for key in (
                "id", "ticker", "as_of", "instrument"
            )} | {"evidence": available_evidence(case)})
            write_json(directory / "reference.json", case["reference"])
            trace = EvaluationTrace()
            started = monotonic()
            record = {"case_id": case["id"], "repetition": repetition,
                      "mode": mode, "status": "completed", "error_type": None,
                      "expected_insufficient": case["reference"]["evidence_status"] == "insufficient",
                      "reference_fact_count": len(case["reference"]["facts"]),
                      "estimated_cost_usd": None}
            try:
                if mode == "smoke":
                    state = {"market_report": "OFFLINE SMOKE ONLY — no model or quality evaluation.\n\n"
                             + json.dumps(available_evidence(case), ensure_ascii=False, indent=2)}
                    signal = None
                else:
                    state, signal = run_current_graph(case, config, trace)
                write_report_tree(state, case["ticker"], directory)
                write_json(directory / "state.json", state)
                record["signal"] = signal
            except Exception as exc:
                # SDK exception messages can include tokens/URLs. Persist type only.
                record.update(status="failed", error_type=type(exc).__name__)
            record.update(seconds=monotonic() - started, usage=trace.summary())
            if mode == "model" and record["usage"]["token_usage_complete"] and input_price is not None and output_price is not None:
                record["estimated_cost_usd"] = (
                    record["usage"]["input_tokens"] * input_price
                    + record["usage"]["output_tokens"] * output_price
                ) / 1_000_000
            write_json(directory / "trace.json", trace.events)
            write_json(directory / "result.json", record)
            records.append(record)
            reviews.append(review_template(case, repetition))
            # Checkpoint the suite artifacts after every case for interruption safety.
            write_json(output / "runs.json", records)
            write_json(output / "reviews.json", reviews)
            write_json(output / "summary.json", aggregate(records, reviews))
    return output


def main():
    parser = argparse.ArgumentParser(description="Frozen TradingAgents evaluation")
    parser.add_argument("--mode", choices=["smoke", "model"], default="smoke")
    parser.add_argument("--allow-model-calls", action="store_true",
                        help="Explicitly enable potentially billable model calls")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path, help="Recompute summary for an existing run directory")
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    args = parser.parse_args()
    if args.review:
        records = json.loads((args.review / "runs.json").read_text())
        reviews = json.loads((args.review / "reviews.json").read_text())
        write_json(args.review / "summary.json", aggregate(records, reviews))
        print(f"Review summary updated: {args.review / 'summary.json'}")
        return
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeat and limit must be positive")
    prices = [args.input_usd_per_million, args.output_usd_per_million]
    if any(value is not None and value < 0 for value in prices):
        parser.error("token prices cannot be negative")
    if args.mode == "model" and not args.allow_model_calls:
        parser.error("Model mode requires --allow-model-calls; may incur API charges")
    if args.mode == "model" and not args.limit and not args.cases:
        parser.error("Start with --case or --limit to bound billable runs")
    logging.disable(logging.CRITICAL)
    output = args.output or Path("eval_results") / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    try:
        directory = run_suite(output, mode=args.mode, case_ids=args.cases, repeat=args.repeat,
                              limit=args.limit, input_price=prices[0], output_price=prices[1])
    except Exception as exc:
        print(f"Evaluation stopped: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Artifacts saved: {directory}")
    print("Smoke mode does not evaluate model quality. Complete human reviews for quality metrics.")


if __name__ == "__main__":
    main()
