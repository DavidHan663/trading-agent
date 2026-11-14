"""Evaluation integrity tests: frozen evidence, honest metrics and isolated I/O."""

import copy
import json
import socket
from collections import Counter
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tradingagents.evaluation.dataset import available_evidence, frozen_response, load_dataset
from tradingagents.evaluation.review import aggregate, review_template
from tradingagents.evaluation.run import run_current_graph, run_suite
from tradingagents.evaluation.trace import EvaluationTrace


def test_dataset_has_30_distinct_labeled_cases():
    dataset, fingerprint = load_dataset()
    assert len(fingerprint) == 64
    assert Counter(case["category"] for case in dataset["cases"]) == {
        "normal": 6, "missing": 6, "conflict": 6, "stale": 6, "cutoff": 6,
    }
    assert len({json.dumps(c["evidence"]) for c in dataset["cases"]}) == 30


def test_future_evidence_is_excluded_by_publication_date():
    dataset, _ = load_dataset()
    case = next(c for c in dataset["cases"] if c["id"] == "cutoff-02")
    assert not any(e["domain"] == "income" for e in available_evidence(case))
    assert "999" not in frozen_response(case, "get_income_statement")
    assert "unavailable" in frozen_response(case, "get_income_statement")
    case = next(c for c in dataset["cases"] if c["id"] == "cutoff-01")
    response = frozen_response(case, "get_news")
    assert "FUTURE_NEWS_SENTINEL" not in response
    assert "product launch" in response


def test_unrecorded_requests_never_fall_back_to_network():
    case = load_dataset()[0]["cases"][0]
    assert "unavailable" in frozen_response(case, "get_news", {"ticker": "OTHER"})
    assert "unavailable" in frozen_response(case, "get_news", {"end_date": "2099-01-01"})
    with pytest.raises(ValueError, match="Unrecorded"):
        frozen_response(case, "unknown_tool")


def test_smoke_artifacts_cannot_be_reported_as_quality(tmp_path):
    output = run_suite(tmp_path / "run", repeat=2)
    results = json.loads((output / "runs.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert len(results) == 60
    assert all(r["status"] == "completed" for r in results)
    assert summary["quality_evaluated"] is False
    assert summary["completion_rate"] is None
    assert summary["citation_correctness"] is None
    assert summary["estimated_cost_usd"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert "API_KEY" not in json.dumps(manifest)
    inputs = (output / "cutoff-02-r1" / "input.json").read_text()
    assert "999" not in inputs and "reference" not in inputs
    with pytest.raises(FileExistsError):
        run_suite(output)


def test_unknown_case_fails_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="Unknown case"):
        run_suite(tmp_path / "run", case_ids=["typo"])
    assert not (tmp_path / "run").exists()


def test_model_artifact_serializes_messages_and_preserves_unknown_cost(tmp_path, monkeypatch):
    def fake_graph(case, config, trace):
        return {"messages": [AIMessage(content="Synthetic test")],
                "market_report": "Report from fake model, not a real evaluation."}, "Hold"

    monkeypatch.setattr("tradingagents.evaluation.run.run_current_graph", fake_graph)
    output = run_suite(tmp_path / "run", mode="model", limit=1)
    record = json.loads((output / "runs.json").read_text())[0]
    assert record["status"] == "completed"
    state = json.loads((output / "normal-01-r1" / "state.json").read_text())
    assert state["messages"][0]["content"] == "Synthetic test"
    assert record["estimated_cost_usd"] is None
    assert json.loads((output / "summary.json").read_text())["quality_evaluated"] is False


def reviewed_data():
    case = load_dataset()[0]["cases"][0]
    review = review_template(case, 1)
    review.update(reviewed=True, reviewer="test-reviewer", claim_count=4,
                  supported_claim_count=3, citation_count=2, correct_citation_count=1,
                  covered_fact_count=1, predicted_insufficient=False, unjustified_direction=False)
    record = {"case_id": case["id"], "repetition": 1, "mode": "model", "status": "completed",
              "seconds": 2, "estimated_cost_usd": None,
              "expected_insufficient": False, "reference_fact_count": 2}
    return record, review


def test_human_metrics_have_explicit_denominators():
    record, review = reviewed_data()
    summary = aggregate([record], [review])
    assert summary["supported_claim_rate"] == 0.75
    assert summary["citation_correctness"] == 0.5
    assert summary["reference_fact_coverage"] == 0.5
    assert summary["insufficiency_recall"] is None
    assert summary["estimated_cost_usd"] is None


def test_review_integrity_and_zero_citations():
    record, review = reviewed_data()
    review.update(citation_count=0, correct_citation_count=0)
    assert aggregate([record], [review])["citation_correctness"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([record], [review, review])
    record["mode"] = "smoke"
    with pytest.raises(ValueError, match="real-model"):
        aggregate([record], [review])
    record["mode"] = "model"
    review["reference_fact_count"] = 3
    with pytest.raises(ValueError, match="Reference labels"):
        aggregate([record], [review])


def test_trace_counts_usage_without_serialized_secrets():
    trace = EvaluationTrace()
    trace.on_chat_model_start({"api_key": "DO_NOT_STORE"}, [[AIMessage(content="Synthetic prompt")]], run_id="one")
    message = AIMessage(content="Hold", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    trace.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="one")
    assert trace.summary()["token_usage_complete"]
    assert trace.summary()["input_tokens"] == 10
    assert "DO_NOT_STORE" not in json.dumps(trace.events)
    trace.on_llm_error(RuntimeError("DO_NOT_STORE"), run_id="two")
    assert "DO_NOT_STORE" not in json.dumps(trace.events)


class OfflineModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError


def test_existing_graph_runs_with_frozen_social_fetches_and_no_network(tmp_path, monkeypatch):
    import tradingagents.dataflows.config as data_config
    import tradingagents.graph.trading_graph as graph_module
    from tradingagents.agents.utils import agent_utils
    from tradingagents.default_config import DEFAULT_CONFIG

    def fail_network(*args, **kwargs):
        raise AssertionError("Evaluation must not fetch live market data")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    llm = OfflineModel(responses=["**Rating**: Hold\nInsufficient evidence for a directional call."])
    monkeypatch.setattr(graph_module, "create_llm_client", lambda **kwargs: SimpleNamespace(get_llm=lambda: llm))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(memory_log_path=None, checkpoint_enabled=False,
                  results_dir=str(tmp_path / "logs"), data_cache_dir=str(tmp_path / "cache"))
    original_news = agent_utils.get_news.func
    prior_config = copy.deepcopy(data_config._config)
    case = load_dataset()[0]["cases"][0]
    trace = EvaluationTrace()
    state, signal = run_current_graph(case, config, trace)
    assert signal == "Hold"
    assert state["final_trade_decision"]
    assert agent_utils.get_news.func is original_news
    assert data_config._config == prior_config
    names = {event["name"] for event in trace.events if event["kind"] == "frozen_tool"}
    assert {"get_news", "stocktwits", "reddit"} <= names
