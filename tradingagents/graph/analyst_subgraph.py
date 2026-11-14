"""Per-analyst ReAct subgraphs with isolated message channels."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import AnalystNodeSpec


def build_analyst_subgraph(
    spec: AnalystNodeSpec,
    analyst_node: Callable,
    tool_node: Any,
    should_continue: Callable,
):
    """Compile an isolated ReAct subgraph for one analyst.

    The analyst and its tools retain their existing loop, but the graph owns a
    private ``messages`` channel.  A completed analyst terminates the subgraph
    instead of flowing through a shared message-clear node.
    """
    subgraph = StateGraph(AgentState)
    subgraph.add_node(spec.agent_node, analyst_node)
    subgraph.add_node(spec.tool_node, tool_node)
    subgraph.add_edge(START, spec.agent_node)
    subgraph.add_conditional_edges(
        spec.agent_node,
        should_continue,
        {spec.tool_node: spec.tool_node, spec.clear_node: END},
    )
    subgraph.add_edge(spec.tool_node, spec.agent_node)
    return subgraph.compile()


def make_analyst_wrapper(compiled_subgraph, spec: AnalystNodeSpec) -> Callable:
    """Expose a subgraph as a parent-graph node without leaking messages."""

    def analyst_wrapper_node(state):
        result = compiled_subgraph.invoke(
            {
                "messages": [("human", state["company_of_interest"])],
                "company_of_interest": state["company_of_interest"],
                "asset_type": state.get("asset_type", "stock"),
                "instrument_context": state.get("instrument_context", ""),
                "trade_date": state["trade_date"],
            }
        )
        return {spec.report_key: result.get(spec.report_key, "")}

    return analyst_wrapper_node
