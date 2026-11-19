"""Historical-state replay collection is non-destructive and prediction-blind."""

import json

import pytest

from tradingagents.evaluation.judgment_replay import (
    collect_replays,
    file_sha256,
    judgment_candidates,
    load_replay_state,
    valid_judgment_candidate,
)


def _evidence():
    return {
        "id": "F1", "tool": "fixture", "target_ticker": "EVAL01",
        "requested_ticker": "EVAL01", "source_ticker": "EVAL01",
        "source_url": "https://example.test/f1", "available_at": "2025-01-30",
        "period_end": "2024-12-31", "fetched_at": "2025-01-31T00:00:00Z",
        "content": "Revenue was 120 million USD versus 100 million USD previously.",
    }


def _state():
    return {
        "company_of_interest": "EVAL01", "trade_date": "2025-01-31",
        "investment_plan": "Hold", "trader_investment_plan": "No change",
        "risk_debate_state": {"history": "Risk discussion."},
        "evidence": [_evidence()],
    }


def _portfolio_result():
    return {
        "final_trade_decision": "**Rating**: Hold",
        "citation_validation": {
            "status": "checked",
            "semantic_support_checked": False,
            "claims": [
                {"claim_id": "C1", "claim_type": "fact", "claim": "Revenue was 120 million USD.",
                 "kind": "financial", "evidence_ids": ["F1"], "basis_claim_ids": [],
                 "issues": [], "status": "fact_verified"},
                {"claim_id": "C2", "claim_type": "judgment", "claim": "Revenue improved.",
                 "kind": "rating_reason", "evidence_ids": [], "basis_claim_ids": ["C1"],
                 "issues": [], "status": "judgment_review_required"},
            ],
        },
    }


def test_load_replay_state_requires_complete_saved_context(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"company_of_interest": "EVAL01"}))
    with pytest.raises(ValueError, match="missing required fields"):
        load_replay_state(path)


def test_candidate_links_judgment_to_basis_and_evidence():
    record = {
        "source_id": "replay-01", "source_path": "state.json", "source_sha256": "a" * 64,
        "ticker": "EVAL01", "as_of": "2025-01-31", "evidence": [_evidence()],
        "citation_validation": _portfolio_result()["citation_validation"],
    }
    candidates = judgment_candidates(record)
    assert len(candidates) == 1
    assert candidates[0]["judgment"]["claim_id"] == "C2"
    assert candidates[0]["basis_claims"][0]["claim_id"] == "C1"
    assert candidates[0]["evidence"][0]["id"] == "F1"
    assert valid_judgment_candidate(candidates[0]) is True


def test_collection_preserves_sources_is_resumable_and_never_runs_semantic_gate(tmp_path):
    source = tmp_path / "state.json"
    source.write_text(json.dumps(_state()))
    original_hash = file_sha256(source)
    calls = []

    def portfolio_node(state):
        calls.append(state["company_of_interest"])
        return _portfolio_result()

    output = collect_replays(
        tmp_path / "out", portfolio_node, source_paths=[source], workspace=tmp_path,
    )
    assert calls == ["EVAL01"]
    assert file_sha256(source) == original_hash
    collection = json.loads((output / "collection.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(collection["judgment_candidates"]) == 1
    assert "semantic_prediction" not in json.dumps(collection)
    assert manifest["semantic_gate_calls"] == 0
    assert manifest["source_files_unchanged"] is True
    assert manifest["valid_judgment_candidate_count"] == 1
    assert manifest["portfolio_manager_logical_invocations"] == 1

    collect_replays(output, portfolio_node, source_paths=[source], workspace=tmp_path)
    assert calls == ["EVAL01"]
