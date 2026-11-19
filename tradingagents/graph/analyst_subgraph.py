"""Per-analyst ReAct subgraphs with isolated message channels."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.evidence import check_analyst_report, evidence_label, records_from_tool, strip_provider_provenance

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
    def evidence_tool_node(state):
        result = tool_node.invoke(state)
        calls = {call["id"]: call for call in state["messages"][-1].tool_calls}
        messages, evidence = [], []
        for message in result["messages"]:
            if not isinstance(message, ToolMessage):
                messages.append(message)
                continue
            call = calls.get(message.tool_call_id, {})
            records = records_from_tool(
                call.get("name") or message.name or "unknown",
                call.get("args") or {}, str(message.content), state["company_of_interest"],
            )
            evidence.extend(records)
            messages.append(message.model_copy(update={
                "content": evidence_label(records) + strip_provider_provenance(message.content),
            }))
        return {"messages": messages, "evidence": evidence}

    subgraph = StateGraph(AgentState)
    subgraph.add_node(spec.agent_node, analyst_node)
    subgraph.add_node(spec.tool_node, evidence_tool_node)
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
        report = result.get(spec.report_key, "")
        records = result.get("evidence", [])
        return {spec.report_key: report, "evidence": records,
                "analyst_citation_checks": [check_analyst_report(spec.key, report, records)]}

    return analyst_wrapper_node
