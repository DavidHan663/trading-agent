"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from tradingagents.evidence import unique_records


def _source_link(value):
    """Only render ordinary HTTP(S) URLs as Markdown links."""
    if not isinstance(value, str) or any(c.isspace() or c in '<>()' for c in value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        return value
    except ValueError:
        return None


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    assessment = final_state.get("evidence_assessment")
    if assessment and assessment.get("status") == "insufficient":
        sections.append(f"## V. Limited Decision\n\n{final_state.get('final_trade_decision', '')}")

    evidence = unique_records(final_state.get("evidence", []))
    validation = final_state.get("citation_validation")
    semantic_validation = final_state.get("semantic_claim_validation")
    analyst_checks = final_state.get("analyst_citation_checks", [])
    if evidence or validation or semantic_validation or analyst_checks or assessment:
        (save_path / "evidence.json").write_text(
            json.dumps({"records": evidence, "analyst_checks": analyst_checks,
                        "validation": validation, "semantic_validation": semantic_validation,
                        "assessment": assessment}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        evidence_lines = ["## VI. Evidence and Citation Checks", ""]
        if assessment:
            evidence_lines.append(f"Evidence sufficiency: **{assessment['status']}**; "
                                  f"requeries: {assessment.get('requery_count', 0)}.")
            for gap in assessment.get("missing", []):
                evidence_lines.append(f"- Gap: {gap}")
            for conflict in assessment.get("conflicts", []):
                evidence_lines.append(f"- Conflict: {conflict['metric']} · IDs: {', '.join(conflict['evidence_ids'])}")
        for record in evidence:
            excerpt = record["content"].replace("\n", " ")[:240]
            evidence_lines.append(
                f"- **{record['id']}** · {record['tool']} · source: {record['source_url'] or 'unknown'} "
                f"· published/available: {record['available_at'] or 'unknown'} "
                f"· period: {record['period_end'] or 'unknown'} · {excerpt}"
            )
        if validation:
            evidence_lines.extend([
                "",
                f"Atomic claim check: **{validation['status']}**. Facts check citation metadata and literal values; "
                "calculations check arithmetic reproducibility; judgments always require semantic review. "
                "Source authenticity, formula choice, and investment correctness were not verified.",
            ])
            structured = validation.get("structured_output", {})
            if structured.get("status") == "fallback":
                evidence_lines.append(f"Structured output fallback: {structured.get('reason', 'unknown')}"
                                      f" ({structured.get('error_type', 'type unavailable')}).")
            for claim in validation.get("claims", []):
                issues = ", ".join(claim["issues"]) or "none"
                identity = claim.get("claim_id") or "legacy"
                calculation = (
                    f" · arithmetic: {claim.get('arithmetic_status', 'unknown')}"
                    f" · computed: {claim.get('computed_value')} {claim.get('result_unit') or ''}".rstrip()
                    if claim.get("claim_type") == "calculation" else ""
                )
                basis = (
                    f" · basis: {', '.join(claim.get('basis_claim_ids', [])) or 'none'}"
                    if claim.get("claim_type") == "judgment" else ""
                )
                evidence_lines.append(
                    f"- **{claim['status']}** · {identity}/{claim.get('claim_type', 'legacy')}: "
                    f"{claim['claim']}{calculation}{basis} · issues: {issues}"
                )
        if semantic_validation:
            evidence_by_id = {record['id']: record for record in evidence}
            usage = semantic_validation.get("usage", {})
            evidence_lines.extend([
                "",
                f"Semantic claim gate (shadow mode): **{semantic_validation.get('status', 'review')}**. "
                f"Model invocations: {semantic_validation.get('model_invocations', 0)}/"
                f"{semantic_validation.get('max_model_invocations', 0)}. "
                "This check did not change the final rating.",
                f"Reviewer independence: {semantic_validation.get('reviewer_independence', 'unknown')}. "
                f"Input/output tokens: {usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}; "
                f"usage complete: {usage.get('token_usage_complete', False)}.",
            ])
            if semantic_validation.get("reason"):
                evidence_lines.append(f"- Gate-level reason: {semantic_validation['reason']}")
            for claim in semantic_validation.get("claims", []):
                flags = ", ".join(claim.get("reason_flags", [])) or "none"
                failure = claim.get("failure_reason") or "none"
                evidence_lines.append(
                    f"- **{claim.get('decision', 'review')}** · "
                    f"{claim.get('claim_id') or 'legacy'}/{claim.get('claim_type', 'legacy')}: "
                    f"{claim.get('claim', '')} · reasons: {flags} · failure: {failure}"
                )
                if claim.get("reasoning"):
                    evidence_lines.append(f"  - Explanation: {claim['reasoning']}")
                for missing in claim.get("missing_information", []):
                    evidence_lines.append(f"  - Missing information: {missing}")
                for quote in claim.get("evidence_quotes", []):
                    evidence_id = quote.get("evidence_id", "unknown")
                    source = _source_link(evidence_by_id.get(evidence_id, {}).get("source_url"))
                    source_text = f" · [source]({source})" if source else ""
                    evidence_lines.append(
                        f"  - Evidence {evidence_id}: {quote.get('quote', '')}{source_text}"
                    )
        if analyst_checks:
            evidence_lines.extend(["", "Analyst report ID checks (presence only):"])
            for check in analyst_checks:
                evidence_lines.append(f"- {check['analyst']}: {check['status']}")
        sections.append("\n".join(evidence_lines))

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
