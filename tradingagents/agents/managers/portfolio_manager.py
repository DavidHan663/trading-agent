"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import re
import logging

from pydantic import ValidationError

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioDecisionOutput,
    render_pm_decision,
    to_portfolio_decision,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
)
from tradingagents.evidence import evidence_index, unique_records, validate_claims

logger = logging.getLogger(__name__)
PM_EVIDENCE_LIMIT = 64
PM_EVIDENCE_SUPPLEMENT = 8
FACT_NUMERIC_FIELD_INSTRUCTION = (
    "For each fact_claim, numeric_value and unit are an all-or-nothing pair: "
    "if you provide one, provide both; if the evidence does not give a directly "
    "copyable number and its unit, omit both. For example, a source literal "
    "'Close=356.27 USD' may use numeric_value=356.27 and unit='USD'; never return "
    "numeric_value without unit or unit without numeric_value."
)


def _structured_error_summary(exc):
    """Return diagnostic-safe structured-output failure metadata.

    Never persist exception text: parser exceptions can include a raw model
    response, prompt fragment, or provider detail.  Pydantic locations and
    error codes are sufficient to distinguish schema failures from transport or
    parser failures without retaining model content.
    """
    summary = {"error_type": type(exc).__name__}
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            errors = current.errors(include_url=False)
            locations = []
            error_types = []
            for error in errors:
                location = ".".join(str(part) for part in error.get("loc", ()))
                if location and location not in locations:
                    locations.append(location)
                error_type = error.get("type")
                if error_type and error_type not in error_types:
                    error_types.append(error_type)
            summary.update({
                "validation_error_count": len(errors),
                "validation_error_locations": locations[:8],
                "validation_error_types": error_types[:8],
            })
            break
        current = current.__cause__ or current.__context__
    return summary


def _referenced_evidence_ids(state, evidence):
    searchable = "\n".join(
        str(state.get(key, "")) for key in (
            "market_report", "sentiment_report", "news_report", "fundamentals_report",
            "investment_plan", "trader_investment_plan",
        )
    )
    risk = state.get("risk_debate_state") or {}
    searchable += "\n" + "\n".join(str(risk.get(key, "")) for key in (
        "history", "bull_history", "bear_history", "aggressive_history",
        "conservative_history", "neutral_history",
    ))
    return {
        record["id"] for record in evidence
        if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(record["id"])
                    + r"(?![A-Za-z0-9_-])", searchable)
    }


def _select_portfolio_evidence(state, evidence):
    records = unique_records(evidence)
    if len(records) <= PM_EVIDENCE_LIMIT:
        return records, {
            "mode": "full", "original_count": len(records),
            "included_count": len(records), "omitted_count": 0,
        }
    referenced_ids = _referenced_evidence_ids(state, records)
    selected = [record for record in records if record["id"] in referenced_ids]
    selected = selected[:PM_EVIDENCE_LIMIT - PM_EVIDENCE_SUPPLEMENT]
    selected_ids = {record["id"] for record in selected}
    newest = sorted(
        (record for record in records if record["id"] not in selected_ids),
        key=lambda record: (
            record.get("available_at") or "", record.get("fetched_at") or "", record["id"],
        ),
        reverse=True,
    )
    selected.extend(newest[:PM_EVIDENCE_SUPPLEMENT])
    return selected, {
        "mode": "referenced_plus_latest",
        "original_count": len(records), "included_count": len(selected),
        "omitted_count": len(records) - len(selected),
        "referenced_count": len(referenced_ids),
    }


def _invoke_portfolio_structured(structured_llm, llm, prompt, render, diagnostics):
    """Try strict output twice, then use the existing free-text fallback."""
    diagnostics["structured_attempts"] = 0
    if structured_llm is None:
        diagnostics.update(status="fallback", reason="binding_unavailable")
        return llm.invoke(prompt).content, None

    recovery_prompt = prompt + (
        "\n\nRECOVERY: The previous response was not a valid PortfolioDecisionOutput. "
        "Return only one object matching the provided schema. Use no extra fields, "
        "respect the per-list item limits, and ensure every judgment basis_claim_id "
        "points to a fact_claim or calculation_claim."
    )
    for attempt, attempt_prompt in enumerate((prompt, recovery_prompt), start=1):
        diagnostics["structured_attempts"] = attempt
        try:
            result = structured_llm.invoke(attempt_prompt)
            if result is None:
                raise ValueError("structured output returned no parsed result")
            diagnostics.update(
                status="structured", recovery_used=attempt == 2,
            )
            return render(result), result
        except Exception as exc:
            error_key = "first_error" if attempt == 1 else "second_error"
            error_summary = _structured_error_summary(exc)
            diagnostics[error_key] = error_summary
            diagnostics["first_error_type" if attempt == 1 else "second_error_type"] = error_summary["error_type"]
            logger.warning(
                "Portfolio Manager structured attempt %s failed (%s)", attempt, type(exc).__name__,
            )

    diagnostics.update(status="fallback", reason="structured_exception", recovery_used=True)
    return llm.invoke(prompt).content, None


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecisionOutput, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        evidence = state.get("evidence", [])
        prompt_evidence, evidence_context = _select_portfolio_evidence(state, evidence)

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

**Registered Evidence (IDs identify records, not verified truth):**
{evidence_index(prompt_evidence)}

Evidence context: {evidence_context['mode']}; included {evidence_context['included_count']} of {evidence_context['original_count']} records. Omitted records cannot be cited.

---

Return no more than eight key claims total: at most 3 fact_claims, 2 calculation_claims, and 3 judgment_claims. Put directly observed propositions in fact_claims, reproducible arithmetic in calculation_claims, and interpretations or actions in judgment_claims. Assign unique claim IDs (C1, C2, ...). A fact cites registered evidence IDs. {FACT_NUMERIC_FIELD_INSTRUCTION} Copy numeric literals and units exactly from evidence; do not translate, rescale, or convert units even when the claim text is Chinese. Do not put a computed number in a fact; use a calculation_claim instead. A calculation declares exact cited inputs, a formula using their variable names, and its result. A judgment contains only its own text, kind, and basis_claim_ids; every basis ID must point directly to a fact or calculation in this same output. Never make a judgment depend on another judgment, and never put evidence IDs, numeric values, formulas, or calculation inputs into a judgment. Prefer the smallest claim set that actually determines the final rating.

Ground every conclusion in specific evidence from the analysts. Never combine an observed fact, derived arithmetic, and investment opinion in one claim. Never invent evidence or claim IDs. Do not treat a source link or an ID as proof that the source supports an inference. Commit to a directional call only when the evidence clearly supports one; choose Hold when the case is balanced, materially conflicting, ambiguous, or insufficient to justify changing exposure, rather than forcing a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        structured_diagnostics = {}
        final_trade_decision, structured_output = _invoke_portfolio_structured(
            structured_llm,
            llm,
            prompt,
            lambda decision: render_pm_decision(to_portfolio_decision(decision)),
            structured_diagnostics,
        )
        structured_diagnostics["evidence_context"] = evidence_context
        structured_decision = (
            to_portfolio_decision(structured_output)
            if structured_output is not None
            else None
        )
        citation_validation = (
            validate_claims(structured_decision.key_claims, evidence,
                            state["company_of_interest"], state.get("trade_date", ""))
            if structured_decision is not None
            else {"status": "unvalidated", "claims": [],
                  "semantic_support_checked": False, "reason": "free_text_fallback"}
        )
        citation_validation["structured_output"] = structured_diagnostics

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "citation_validation": citation_validation,
        }

    return portfolio_manager_node
