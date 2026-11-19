"""Collect current-schema judgment claims by replaying saved Portfolio Manager inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.evaluation.run import current_commit, source_fingerprint, write_json
from tradingagents.evaluation.semantic_run import _public_config, create_reviewer_llm
from tradingagents.evidence import unique_records

SOURCE_STATES = (
    Path("eval_results/live-aapl-full-fixed-20261114/state.json"),
    Path("eval_results/phase3-normal-20261114/normal-01-r1/state.json"),
    Path("eval_results/phase3-edge-20261114/cutoff-01-r1/state.json"),
    Path("eval_results/phase2-small-edge-20261114/conflict-01-r1/state.json"),
    Path("eval_results/phase2-small-edge-20261114/stale-01-r1/state.json"),
)
REQUIRED_STATE_FIELDS = (
    "company_of_interest", "trade_date", "investment_plan",
    "trader_investment_plan", "risk_debate_state", "evidence",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replay_state(path):
    path = Path(path)
    state = json.loads(path.read_text())
    missing = [field for field in REQUIRED_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"Replay state missing required fields: {missing}")
    risk = state.get("risk_debate_state")
    if not isinstance(risk, dict) or not risk.get("history"):
        raise ValueError("Replay state requires non-empty risk debate history")
    if not isinstance(state.get("evidence"), list) or not state["evidence"]:
        raise ValueError("Replay state requires registered evidence")
    return state


def _claim_evidence_ids(claim):
    if claim.get("claim_type") == "calculation":
        return [
            evidence_id
            for item in claim.get("inputs", [])
            for evidence_id in item.get("evidence_ids", [])
        ]
    return list(claim.get("evidence_ids", []))


def judgment_candidates(record):
    validation = record.get("citation_validation") or {}
    claims = validation.get("claims") or []
    claims_by_id = {claim["claim_id"]: claim for claim in claims if claim.get("claim_id")}
    evidence_by_id = {item["id"]: item for item in record.get("evidence", [])}
    candidates = []
    for judgment in claims:
        if judgment.get("claim_type") != "judgment":
            continue
        basis_claims = [claims_by_id.get(claim_id) for claim_id in judgment.get("basis_claim_ids", [])]
        basis_claims = [claim for claim in basis_claims if claim is not None]
        evidence_ids = []
        for basis in basis_claims:
            evidence_ids.extend(_claim_evidence_ids(basis))
        evidence_ids = list(dict.fromkeys(evidence_ids))
        candidates.append({
            "id": f"{record['source_id']}--{judgment.get('claim_id') or 'legacy'}",
            "source_id": record["source_id"],
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "ticker": record["ticker"],
            "as_of": record["as_of"],
            "judgment": judgment,
            "basis_claims": basis_claims,
            "evidence": [evidence_by_id[evidence_id] for evidence_id in evidence_ids
                         if evidence_id in evidence_by_id],
        })
    return candidates


def valid_judgment_candidate(candidate):
    judgment = candidate.get("judgment") or {}
    bases = candidate.get("basis_claims") or []
    return (
        judgment.get("status") == "judgment_review_required"
        and not judgment.get("issues")
        and bool(bases)
        and all(basis.get("status") in {"fact_verified", "calculation_verified"}
                for basis in bases)
        and bool(candidate.get("evidence"))
    )


def _public_record(source_id, relative_path, source_hash, state, result):
    validation = result.get("citation_validation") or {}
    cited_ids = set()
    for claim in validation.get("claims") or []:
        cited_ids.update(_claim_evidence_ids(claim))
    evidence = [record for record in unique_records(state.get("evidence", []))
                if record["id"] in cited_ids]
    return {
        "source_id": source_id,
        "source_path": str(relative_path),
        "source_sha256": source_hash,
        "ticker": state["company_of_interest"],
        "as_of": state["trade_date"],
        "final_trade_decision": result.get("final_trade_decision", ""),
        "citation_validation": validation,
        "evidence": evidence,
    }


def collect_replays(output, portfolio_node, *, source_paths=SOURCE_STATES, workspace=None):
    workspace = Path(workspace or Path.cwd()).resolve()
    output = Path(output).resolve()
    resolved = [(workspace / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
                for path in source_paths]
    initial_hashes = {str(path): file_sha256(path) for path in resolved}

    if output.exists():
        if not (output / "collection.json").is_file() or not (output / "manifest.json").is_file():
            raise ValueError("Cannot resume collection without manifest and collection files")
        records = json.loads((output / "collection.json").read_text())["records"]
        manifest = json.loads((output / "manifest.json").read_text())
    else:
        output.mkdir(parents=True)
        records = []
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_fingerprint(),
            "git_commit": current_commit(),
            "source_states": [
                {"path": str(path.relative_to(workspace)), "sha256": initial_hashes[str(path)]}
                for path in resolved
            ],
            "portfolio_manager_provider_call_upper_bound": len(resolved) * 2,
            "semantic_gate_calls": 0,
            "config": {},
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "collection.json", {"records": [], "judgment_candidates": []})

    completed = {record["source_path"] for record in records}
    for index, (relative_path, path) in enumerate(zip(source_paths, resolved), start=1):
        source_path = str(Path(relative_path))
        if source_path in completed:
            continue
        state = load_replay_state(path)
        result = portfolio_node(state)
        records.append(_public_record(
            f"replay-{index:02d}", source_path, initial_hashes[str(path)], state, result,
        ))
        candidates = [candidate for record in records for candidate in judgment_candidates(record)]
        write_json(output / "collection.json", {
            "records": records, "judgment_candidates": candidates,
        })

    final_hashes = {str(path): file_sha256(path) for path in resolved}
    if final_hashes != initial_hashes:
        raise RuntimeError("A replay source changed during collection")
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["source_files_unchanged"] = True
    manifest["record_count"] = len(records)
    candidates = [candidate for record in records for candidate in judgment_candidates(record)]
    manifest["judgment_candidate_count"] = len(candidates)
    manifest["valid_judgment_candidate_count"] = sum(
        valid_judgment_candidate(candidate) for candidate in candidates
    )
    manifest["portfolio_manager_logical_invocations"] = (
        len(records) + sum(
            (record.get("citation_validation", {}).get("structured_output", {}).get("status")
             == "fallback")
            for record in records
        )
    )
    write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description="Replay saved states through current Portfolio Manager")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("Judgment replay collection requires --allow-model-calls")
    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["llm_max_retries"] = 0
    llm = create_reviewer_llm(config)
    directory = collect_replays(args.output, create_portfolio_manager(llm))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = _public_config(config)
    write_json(manifest_path, manifest)
    print(f"Artifacts saved: {directory}")


if __name__ == "__main__":
    main()
