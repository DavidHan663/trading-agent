"""Run-local evidence records and deterministic citation checks.

An ID identifies a tool result. It does not certify the original source or
prove that the source supports an analyst's interpretation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

_URL = re.compile(r"https?://[^\s\]>)\"']+")
_NUMBER = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])")
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_SNAPSHOT_DATE = re.compile(r"Latest trading row used:\s*(\d{4}-\d{2}-\d{2})")
_FINANCIAL_PERIOD = re.compile(r"(?m)^,(\d{4}-\d{2}-\d{2})(?:,|$)")
_SERIES_DATE = re.compile(r"(?m)^(\d{4}-\d{2}-\d{2})(?=[:,])")
_NEWS_BLOCK = re.compile(
    r"(?ms)^### (?P<title>.+?) \(source: (?P<publisher>.+?)\)\n"
    r"(?P<body>.*?)(?=^### |\Z)"
)
_PROVENANCE_HEADER = re.compile(r"\A# Evidence provenance:\s*(?P<payload>\{[^\n]+\})\s*\n")


def _stable_id(tool_name: str, arguments: dict, content: str) -> str:
    payload = json.dumps([tool_name, arguments, content], sort_keys=True, default=str)
    return "E-" + hashlib.sha256(payload.encode()).hexdigest()[:12].upper()


def _is_unavailable(content: str) -> bool:
    lowered = content.lower().strip()
    return (not lowered or lowered.startswith("<unavailable>")
            or lowered.startswith("error:") or lowered.startswith("error fetching")
            or lowered.startswith("error retrieving")
            or lowered.startswith("no_data_available:")
            or lowered.startswith("data_unavailable:")
            or lowered.startswith("no news found")
            or lowered.startswith("no global news found")
            or lowered.startswith("polymarket data is currently unavailable")
            or "no open prediction markets matched" in lowered
            or lowered.startswith("<no ")
            or lowered.startswith("<reddit unavailable:")
            or "profile fundamentals are withheld for this date" in lowered)


def _plain_output_records(tool_name: str, output: str) -> list[dict]:
    if tool_name in {"get_news", "get_global_news"}:
        articles = []
        for match in _NEWS_BLOCK.finditer(output):
            body = match.group("body").strip()
            link_match = re.search(r"(?m)^Link:\s*(https?://\S+)", body)
            date_match = re.search(r"(?m)^Published:\s*(\d{4}-\d{2}-\d{2})", body)
            articles.append({
                "content": f"{match.group('title')} (source: {match.group('publisher')})\n{body}",
                "source": link_match.group(1) if link_match else None,
                "available_at": date_match.group(1) if date_match else None,
            })
        if articles:
            return articles
    item = {"content": output}
    snapshot = _SNAPSHOT_DATE.search(output)
    period = _FINANCIAL_PERIOD.search(output)
    series_dates = _SERIES_DATE.findall(output)
    if snapshot:
        item["available_at"] = snapshot.group(1)
        item["period_end"] = snapshot.group(1)
    elif period:
        item["period_end"] = period.group(1)
    elif tool_name in {"get_stock_data", "get_indicators"} and series_dates:
        latest = max(series_dates)
        item["available_at"] = latest
        item["period_end"] = latest
    return [item]


def _provider_provenance(output: str) -> dict:
    """Read a routing header while retaining the original tool payload.

    A provider reference establishes where a response came from.  It never
    supplies an availability date, because a retrieval timestamp is not proof
    that the data was publicly available on a historical analysis date.
    """
    match = _PROVENANCE_HEADER.match(output)
    if not match:
        return {}
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    source_url = payload.get("source_url")
    if not isinstance(source_url, str) or not _URL.fullmatch(source_url):
        return {}
    return _normalise_provider_provenance(payload)


def strip_provider_provenance(output: str) -> str:
    """Remove the transport-only provenance header before model prompting."""
    return _PROVENANCE_HEADER.sub("", str(output), count=1)


def _normalise_provider_provenance(payload: dict) -> dict:
    source_url = payload.get("source_url")
    if not isinstance(source_url, str) or not _URL.fullmatch(source_url):
        return {}
    return {
        "source_url": source_url,
        "source_provider": payload.get("provider") if isinstance(payload.get("provider"), str) else None,
        "source_kind": payload.get("source_kind") if isinstance(payload.get("source_kind"), str) else None,
    }


def records_from_tool(tool_name: str, arguments: dict, output: str, target_ticker: str) -> list[dict]:
    """Preserve explicit metadata; never infer publication dates from prose."""
    output = str(output)
    if _is_unavailable(output):
        return []
    provider_provenance = _provider_provenance(output)
    payload_output = strip_provider_provenance(output)
    try:
        parsed = json.loads(payload_output)
    except (ValueError, TypeError):
        parsed = None
    source_records = parsed.get("records") if isinstance(parsed, dict) else None
    if not isinstance(source_records, list) or not source_records:
        source_records = _plain_output_records(tool_name, payload_output)

    requested_ticker = arguments.get("ticker", arguments.get("symbol"))
    fetched_at = datetime.now(timezone.utc).isoformat()
    result = []
    for item in source_records:
        if not isinstance(item, dict):
            item = {"content": str(item)}
        content = str(item.get("content") or payload_output)
        explicit_id = item.get("id")
        # Published items can be returned by several overlapping query windows.
        # Their identity comes from source metadata and content, not the window
        # that happened to discover them.
        identity_arguments = (
            {"source": item.get("source"), "available_at": item.get("available_at")}
            if item.get("source") else arguments
        )
        evidence_id = (str(explicit_id) if explicit_id and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", str(explicit_id))
                       else _stable_id(tool_name, identity_arguments, content))
        explicit_urls = _URL.findall(str(item.get("source") or ""))
        content_urls = set(_URL.findall(content))
        source_url = (explicit_urls[0] if explicit_urls else next(iter(content_urls))
                      if len(content_urls) == 1 else provider_provenance.get("source_url"))
        source_kind = (item.get("source_kind") if isinstance(item.get("source_kind"), str)
                       else "primary" if (explicit_urls or len(content_urls) == 1)
                       else provider_provenance.get("source_kind"))
        result.append({
            "id": evidence_id,
            "tool": tool_name,
            "target_ticker": target_ticker,
            "requested_ticker": str(requested_ticker) if requested_ticker is not None else None,
            "source_ticker": item.get("ticker"),
            "source_url": source_url,
            "source_provider": (item.get("source_provider") if isinstance(item.get("source_provider"), str)
                                else provider_provenance.get("source_provider")),
            "source_kind": source_kind,
            "available_at": item.get("available_at"),
            "period_end": item.get("period_end"),
            "fetched_at": fetched_at,
            "content": content,
        })
    return result


def unique_records(records: list[dict]) -> list[dict]:
    """Repeated identical results collapse; conflicting ID reuse remains visible."""
    seen = set()
    result = []
    for record in records:
        signature = (record["id"], record["content"], record.get("source_url"),
                     record.get("source_provider"), record.get("source_kind"),
                     record.get("available_at"), record.get("period_end"),
                     record.get("target_ticker"), record.get("requested_ticker"),
                     record.get("source_ticker"))
        if signature not in seen:
            result.append(record)
            seen.add(signature)
    return result


def evidence_label(records: list[dict]) -> str:
    ids = ", ".join(record["id"] for record in records)
    return f"[Evidence ID: {ids}. Cite these IDs beside key factual claims.]\n" if ids else ""


def evidence_index(records: list[dict]) -> str:
    lines = []
    for record in unique_records(records):
        lines.append(
            f"{record['id']} | tool={record['tool']} | target={record['target_ticker']} "
            f"| source={record['source_url'] or 'unknown'} "
            f"| source_kind={record.get('source_kind') or 'unknown'} "
            f"| available_at={record['available_at'] or 'unknown'} "
            f"| period_end={record['period_end'] or 'unknown'} "
            f"| excerpt={json.dumps(record['content'][:240], ensure_ascii=False)}"
        )
    return "\n".join(lines) or "No registered evidence."


def check_analyst_report(analyst: str, report: str, records: list[dict]) -> dict:
    """Check ID presence in a report; claim-level support remains unverified."""
    ids = {record["id"] for record in records}
    cited = sorted(evidence_id for evidence_id in ids if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(evidence_id) + r"(?![A-Za-z0-9_-])", report))
    status = ("no_report" if not report else "no_evidence" if not ids
              else "ids_present" if cited else "missing_citation_ids")
    return {"analyst": analyst, "status": status, "cited_ids": cited,
            "claim_support_checked": False}


def _numeric_check(value: float, unit: str | None, content: str) -> str:
    if not unit:
        return "matched" if _literal_number_present(value, content) else "needs_review"
    comparable_found = False
    for match in _NUMBER.finditer(content):
        nearby = content[match.end():match.end() + len(unit) + 4]
        if re.match(r"\s*" + re.escape(unit) + r"(?!\w)", nearby, re.I):
            comparable_found = True
            if float(match.group()) == value:
                return "matched"
    # An absent literal can be an arithmetic derivative or a decision rule.
    # Neither can be disproved by string comparison alone.
    return "not_literal" if comparable_found else "needs_review"


def _citation_issues(evidence_ids, lookup, counts, ticker, as_of):
    issues = []
    cited = []
    for evidence_id in evidence_ids:
        record = lookup.get(evidence_id)
        if record is None:
            issues.append(f"unknown_id:{evidence_id}")
            continue
        cited.append(record)
        if counts[evidence_id] > 1:
            issues.append(f"id_collision:{evidence_id}")
        if record["target_ticker"].upper() != ticker.upper():
            issues.append(f"wrong_target:{evidence_id}")
        if record.get("requested_ticker") and record["requested_ticker"].upper() != ticker.upper():
            issues.append(f"wrong_requested_ticker:{evidence_id}")
        if record.get("source_ticker") and str(record["source_ticker"]).upper() != ticker.upper():
            issues.append(f"wrong_source_ticker:{evidence_id}")
        available_at = record.get("available_at")
        if available_at:
            try:
                if not as_of:
                    issues.append(f"unknown_analysis_date:{evidence_id}")
                elif date.fromisoformat(str(available_at)[:10]) > date.fromisoformat(as_of):
                    issues.append(f"future_evidence:{evidence_id}")
            except ValueError:
                issues.append(f"invalid_date:{evidence_id}")
        else:
            issues.append(f"unknown_publication_date:{evidence_id}")
        if not record.get("source_url"):
            issues.append(f"unknown_source_url:{evidence_id}")
    return issues, cited


def _literal_number_present(value: float, content: str) -> bool:
    return any(math.isclose(float(match.group()), value, rel_tol=1e-12, abs_tol=1e-12)
               for match in _NUMBER.finditer(content))


def _non_date_numeric_literals(text: str) -> list[tuple[Decimal, int]]:
    """Return claim numbers after removing full ISO dates from consideration."""
    without_dates = _ISO_DATE.sub("", text)
    return [
        (Decimal(match.group()), len(match.group().partition(".")[2]))
        for match in _NUMBER.finditer(without_dates)
    ]


def _rounded_literal_number_present(value: Decimal, precision: int, content: str) -> bool:
    """Match a source literal when it rounds to the precision shown in a claim."""
    quantum = Decimal(1).scaleb(-precision)
    expected = value.quantize(quantum, rounding=ROUND_HALF_UP)
    for match in _NUMBER.finditer(content):
        literal = Decimal(match.group())
        if literal.quantize(quantum, rounding=ROUND_HALF_UP) == expected:
            return True
    return False


_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate_formula(formula: str, values: dict[str, float]) -> float:
    if len(formula) > 240:
        raise ValueError("formula_too_long")
    tree = ast.parse(formula, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in values:
            return float(values[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
            return _ARITHMETIC[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("unsupported_formula")

    result = float(evaluate(tree))
    if not math.isfinite(result):
        raise ValueError("non_finite_result")
    return result


def _fatal_metadata_issue(issue: str) -> bool:
    return issue.startswith(("unknown_id", "wrong_", "future_evidence",
                             "invalid_date", "id_collision"))


def validate_claims(claims: list, records: list[dict], ticker: str, as_of: str) -> dict:
    """Validate atomic facts and arithmetic; judgments remain semantic opinions."""
    deduplicated = unique_records(records)
    lookup = {record["id"]: record for record in deduplicated}
    counts = {}
    for record in deduplicated:
        counts[record["id"]] = counts.get(record["id"], 0) + 1
    items = [claim.model_dump() if hasattr(claim, "model_dump") else dict(claim)
             for claim in claims]
    claim_id_counts = {}
    for item in items:
        if item.get("claim_id"):
            claim_id_counts[item["claim_id"]] = claim_id_counts.get(item["claim_id"], 0) + 1

    results = []
    for item in items:
        claim_type = item.get("claim_type")
        base = {"claim_id": item.get("claim_id"), "claim_type": claim_type or "legacy",
                "claim": item.get("text", ""), "kind": item.get("kind"),
                "evidence_ids": item.get("evidence_ids", [])}
        duplicate_id = (item.get("claim_id") and claim_id_counts[item["claim_id"]] > 1)

        if claim_type == "calculation":
            issues = ["duplicate_claim_id"] if duplicate_id else []
            values = {}
            input_results = []
            for calculation_input in item.get("calculation_inputs", []):
                name = calculation_input["name"]
                input_issues, cited = _citation_issues(
                    calculation_input.get("evidence_ids", []), lookup, counts, ticker, as_of,
                )
                if name in values:
                    input_issues.append("duplicate_input_name")
                values[name] = float(calculation_input["value"])
                if cited and not any(_literal_number_present(values[name], record["content"])
                                     for record in cited):
                    input_issues.append("input_value_not_literal")
                issues.extend(f"{issue}:{name}" for issue in input_issues)
                input_results.append({"name": name, "value": values[name],
                                      "evidence_ids": calculation_input.get("evidence_ids", []),
                                      "issues": input_issues})
            computed = None
            arithmetic_status = "verified"
            try:
                computed = _evaluate_formula(item.get("formula") or "", values)
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
                issues.append("formula_invalid")
                arithmetic_status = "failed"
            expected = item.get("result_value")
            if computed is not None and expected is not None:
                # Permit ordinary report rounding (for example 0.2555 -> 0.26)
                # while still rejecting materially different arithmetic.
                tolerance = max(1e-9, abs(float(expected)) * 0.005, 0.005)
                if not math.isclose(computed, float(expected), rel_tol=0.0, abs_tol=tolerance):
                    issues.append("result_mismatch")
                    arithmetic_status = "failed"
            fatal = any(_fatal_metadata_issue(issue.split(":", 1)[0]) for issue in issues)
            failed = fatal or any(issue.startswith(("duplicate_claim_id", "duplicate_input_name",
                                                    "input_value_not_literal", "formula_invalid",
                                                    "result_mismatch")) for issue in issues)
            status = "calculation_failed" if failed else "calculation_needs_review" if issues else "calculation_verified"
            results.append(base | {"formula": item.get("formula"), "result_value": expected,
                                   "result_unit": item.get("result_unit"),
                                   "computed_value": computed, "arithmetic_status": arithmetic_status,
                                   "inputs": input_results,
                                   "issues": issues, "status": status})
            continue

        if claim_type == "judgment":
            issues = ["duplicate_claim_id"] if duplicate_id else []
            results.append(base | {"basis_claim_ids": item.get("basis_claim_ids", []),
                                   "issues": issues, "status": "judgment_review_required"})
            continue

        issues, cited = _citation_issues(item.get("evidence_ids", []), lookup, counts, ticker, as_of)
        if duplicate_id:
            issues.append("duplicate_claim_id")
        if not item.get("evidence_ids"):
            issues.append("no_citation")
        value = item.get("numeric_value")
        if value is not None and cited:
            checks = [_numeric_check(float(value), item.get("unit"), record["content"]) for record in cited]
            if "matched" not in checks:
                issues.append("numeric_not_literal" if all(x == "not_literal" for x in checks)
                              else "numeric_needs_review")
        elif item.get("kind") in {"financial", "price"}:
            text_values = _non_date_numeric_literals(item.get("text", ""))
            if not text_values or not all(
                any(_rounded_literal_number_present(number, precision, record["content"])
                    for record in cited)
                for number, precision in text_values
            ):
                issues.append("numeric_needs_review")

        if claim_type == "fact":
            fatal = duplicate_id or any(_fatal_metadata_issue(issue) for issue in issues)
            status = "fact_failed" if fatal else "fact_needs_review" if issues else "fact_verified"
        else:
            issues.append("legacy_composite_claim")
            status = "legacy_failed" if any(_fatal_metadata_issue(issue) for issue in issues) else "legacy_needs_review"
        results.append(base | {"basis_claim_ids": item.get("basis_claim_ids", []),
                               "issues": issues, "status": status})

    result_by_id = {result["claim_id"]: result for result in results
                    if result.get("claim_id") and claim_id_counts[result["claim_id"]] == 1}
    for result in results:
        if result["claim_type"] != "judgment":
            continue
        basis_ids = result.get("basis_claim_ids", [])
        for basis_id in basis_ids:
            basis = result_by_id.get(basis_id)
            if basis is None:
                result["issues"].append(f"unknown_basis_claim:{basis_id}")
            elif basis_id == result.get("claim_id"):
                result["issues"].append(f"self_basis_claim:{basis_id}")
            elif basis["claim_type"] not in {"fact", "calculation"}:
                result["issues"].append(f"invalid_basis_type:{basis_id}")
            elif basis["status"] not in {"fact_verified", "calculation_verified"}:
                result["issues"].append(f"basis_not_verified:{basis_id}")
        if any(issue.startswith(("unknown_basis_claim", "self_basis_claim",
                                "invalid_basis_type", "duplicate_claim_id"))
               for issue in result["issues"]):
            result["status"] = "judgment_invalid"

    return {"status": "unvalidated" if not results else "checked", "claims": results,
            "semantic_support_checked": False,
            "calculation_scope": "Arithmetic reproducibility only; formula choice and investment meaning are not validated."}
