"""Isolated analysis process; API keys and endpoints stay on the server."""

import copy
import json
import sys
from pathlib import Path


def run(request, directory):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = copy.deepcopy(DEFAULT_CONFIG)
    # A web job always starts fresh; never inherit another job's checkpoint.
    config["checkpoint_enabled"] = False
    config["output_language"] = "简体中文"
    config["max_debate_rounds"] = request["rounds"]
    config["max_risk_discuss_rounds"] = request["rounds"]
    graph = TradingAgentsGraph(selected_analysts=request["analysts"], config=config)
    state, signal = graph.propagate(request["ticker"], request["trade_date"], request["asset_type"])
    graph.save_reports(state, request["ticker"], directory)
    (directory / "result.json").write_text(json.dumps({"signal": str(signal)}), encoding="utf-8")


if __name__ == "__main__":
    run(json.load(sys.stdin), Path(sys.argv[1]))
