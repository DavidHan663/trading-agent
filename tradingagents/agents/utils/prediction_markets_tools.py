from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_prediction_markets(
    topic: Annotated[
        str,
        "Event topic/keyword, e.g. 'Fed rate cut', 'recession 2026', "
        "'US election', or a sector/company event.",
    ],
    limit: Annotated[int | None, "Max markets to return; omit for a default of 6"] = None,
    curr_date: Annotated[
        str | None,
        "Analysis date in YYYY-MM-DD. Required for historical analyses so live odds are withheld.",
    ] = None,
) -> str:
    """
    Retrieve live, market-implied probabilities for forward-looking events from
    prediction markets (Polymarket): Fed decisions, recession, elections,
    geopolitics, crypto. Returns the most-traded open markets matching the
    topic, each with its implied probability, traded volume, resolution date,
    and recent move. Uses the configured prediction_markets vendor.

    Args:
        topic (str): Event keyword(s) to search
        limit (int): Max markets to return; omit for a default of 6
        curr_date (str): Analysis date; live probabilities are withheld for past dates

    Returns:
        str: A formatted markdown report of matching prediction markets
    """
    if not curr_date:
        return (
            "<unavailable> Analysis date is required for prediction-market data; "
            "live odds cannot be used without a point-in-time boundary."
        )
    return route_to_vendor("get_prediction_markets", topic, limit, curr_date)
