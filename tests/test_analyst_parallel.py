"""Tests for parallel analyst execution with isolated message channels."""

import unittest

from langgraph.graph import END, START, StateGraph

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.analyst_subgraph import (
    build_analyst_subgraph,
    make_analyst_wrapper,
)


def _stub_analyst(report_key, marker):
    def node(state):
        seen = " | ".join(str(getattr(message, "content", message)) for message in state["messages"])
        return {
            "messages": [("ai", f"{marker} done")],
            report_key: f"{marker}::saw[{seen}]",
        }

    return node


def _stub_should_continue(spec):
    def should_continue(state):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return spec.tool_node
        return spec.clear_node

    return should_continue


class AnalystWrapperIsolationTests(unittest.TestCase):
    def test_wrapper_returns_only_report_and_seeds_fresh_messages(self):
        spec = build_analyst_execution_plan(["market"]).specs[0]
        captured = {}

        class FakeCompiled:
            def invoke(self, input_state):
                captured["input"] = input_state
                return {spec.report_key: "REPORT", "messages": [("ai", "noise")] * 50}

        wrapper = make_analyst_wrapper(FakeCompiled(), spec)
        output = wrapper(
            {
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "instrument_context": "ctx",
                "trade_date": "2026-01-15",
                "messages": [("ai", "OTHER ANALYST LEAK")] * 10,
            }
        )

        self.assertEqual(output, {spec.report_key: "REPORT"})
        self.assertEqual(len(captured["input"]["messages"]), 1)
        self.assertEqual(captured["input"]["company_of_interest"], "NVDA")


class AnalystFanOutFanInTests(unittest.TestCase):
    def _build_parent(self, analyst_keys):
        plan = build_analyst_execution_plan(analyst_keys)
        workflow = StateGraph(AgentState)
        workflow.add_node("Bull Researcher", lambda state: {})
        workflow.add_edge("Bull Researcher", END)

        for spec in plan.specs:
            subgraph = build_analyst_subgraph(
                spec,
                _stub_analyst(spec.report_key, spec.report_key.upper()),
                lambda state: {},
                _stub_should_continue(spec),
            )
            workflow.add_node(spec.agent_node, make_analyst_wrapper(subgraph, spec))
            workflow.add_edge(START, spec.agent_node)
            workflow.add_edge(spec.agent_node, "Bull Researcher")
        return workflow.compile()

    def _invoke(self, analyst_keys):
        return self._build_parent(analyst_keys).invoke(
            {
                "messages": [("human", "NVDA")],
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "instrument_context": "ctx",
                "trade_date": "2026-01-15",
            }
        )

    def test_all_reports_populated_after_fan_in(self):
        final = self._invoke(["market", "social", "news", "fundamentals"])
        for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
            self.assertTrue(final.get(key), f"missing report: {key}")

    def test_analysts_do_not_see_each_others_messages(self):
        final = self._invoke(["market", "news"])
        self.assertNotIn("NEWS_REPORT", final["market_report"])
        self.assertNotIn("MARKET_REPORT", final["news_report"])
        self.assertIn("NVDA", final["market_report"])
        self.assertIn("NVDA", final["news_report"])


if __name__ == "__main__":
    unittest.main()
