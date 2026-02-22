"""Validator node — cross-subtask claim verification with structured evidence.

P2: Dedicated Validator Node in the LangGraph workflow (executor → validator → critic).
P3: Structured Evidence Representation (EvidenceClaim objects).

Runs ONCE after all subtasks complete. For each subtask result:
1. Uses MiniMax-M2.1 to extract specific claims from the LLM analysis
2. Checks each claim against the subtask's fetched tool content
3. Produces structured EvidenceClaim objects: {claim, evidence, source_url, confidence, grounded}
4. Updates each subtask's validation with richer structured data
5. Writes `evidence_claims` list to state for Critic and Reporter to consume
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_engine.agents.state import GraphState, SubtaskStatus
from agent_engine.config import get_settings
from agent_engine.llm import get_provider
from agent_engine.tools.builtin.web_search import sanitize_text_for_llm

logger = logging.getLogger(__name__)

# Max chars of fetched content to pass to the validator per subtask
_MAX_EVIDENCE_CHARS = 12000
# Max chars to keep from a single source block
_MAX_EVIDENCE_PER_SOURCE = 3000
# Max chars of subtask response to validate
_MAX_RESPONSE_CHARS = 5000

# Trust executor outputs by default: bypass validator-side numeric/unit/currency
# consistency checks and regex-based matching.
TRUST_EXECUTOR_OUTPUT = True


VALIDATOR_SYSTEM_PROMPT = """You are an evidence-consistency verification specialist.
Your ONLY job is to verify numerical claims against provided source evidence.

## INPUT FORMAT
You will receive:
1. A "Subtask Analysis" — the LLM's written analysis for a research subtask
2. "Source Evidence" — actual text fetched from URLs (ground truth)

## YOUR TASK
1. Identify up to 10 specific factual claims in the analysis (focus on numbers, dates,
   statistics, rankings, identifiers, and named facts — NOT qualitative statements)
2. For each claim, search the Source Evidence for supporting text
3. Classify each claim as:
   - GROUNDED: number can be located in evidence and unit/currency/period are consistent
   - NEEDS_VERIFICATION: insufficient evidence or ambiguous mismatch
   - UNGROUNDED: clear contradiction with cited evidence
4. For GROUNDED or NEEDS_VERIFICATION, copy the best evidence snippet
5. For each claim, include a short reason

## RULES
- Do NOT use external knowledge — verify ONLY against the provided Source Evidence
- Do NOT adjudicate truth across conflicting sources; mark as NEEDS_VERIFICATION
- Check these dimensions explicitly: number locateability, unit/currency/period consistency,
  and arithmetic consistency for derived calculations
- Qualitative claims ("strong growth", "market leader") need NOT be verified
- If no Source Evidence is provided, classify as NEEDS_VERIFICATION (not UNGROUNDED)

## OUTPUT FORMAT
Respond with a JSON object:
{
  "evidence_claims": [
    {
      "claim": "exact quote of the specific claim from the analysis",
      "verification_status": "grounded|needs_verification|ungrounded",
      "grounded": true/false,
      "reason": "short reason for this status",
      "evidence": "supporting snippet from source (empty if unavailable)",
      "source_url": "URL of supporting source (empty if unavailable)",
      "confidence": 0.0-1.0
    }
  ],
  "grounded_count": <integer>,
  "needs_verification_count": <integer>,
  "ungrounded_count": <integer>,
  "grounding_ratio": <float 0.0-1.0>
}"""


def _extract_numeric_tokens(text: str) -> list[str]:
    return re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")


def _to_float(token: str) -> float | None:
    try:
        return float(str(token).replace(",", ""))
    except Exception:
        return None


def _numbers_match(a: str, b: str) -> bool:
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return False
    if fa == fb:
        return True
    scale = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / scale <= 0.02


def _extract_meta_tokens(text: str) -> dict[str, set[str]]:
    t = (text or "").lower()
    currency: set[str] = set()
    unit: set[str] = set()
    period: set[str] = set()

    currency_map = {
        "usd": [r"\busd\b", r"\bus\$\b", r"\$"],
        "cny": [r"\bcny\b", r"\brmb\b", r"¥", r"￥"],
        "hkd": [r"\bhkd\b", r"\bhk\$\b"],
        "eur": [r"\beur\b", r"€"],
        "gbp": [r"\bgbp\b", r"£"],
    }
    unit_map = {
        "billion": [r"\bbillion\b", r"\bbn\b", r"\bb\b", r"十亿", r"亿"],
        "million": [r"\bmillion\b", r"\bmn\b", r"\bm\b", r"百万"],
        "thousand": [r"\bthousand\b", r"\bk\b", r"千"],
        "percent": [r"%", r"百分比", r"pct"],
    }

    for key, pats in currency_map.items():
        if any(re.search(p, t) for p in pats):
            currency.add(key)
    for key, pats in unit_map.items():
        if any(re.search(p, t) for p in pats):
            unit.add(key)

    for y in re.findall(r"\b(19\d{2}|20\d{2})\b", t):
        period.add(y)
    for q in re.findall(r"\bq([1-4])\b", t):
        period.add(f"q{q}")
    for q in re.findall(r"\b([1-4])q\b", t):
        period.add(f"q{q}")
    for fy in re.findall(r"\bfy\s*(\d{2,4})\b", t):
        period.add(f"fy{fy}")
    return {"currency": currency, "unit": unit, "period": period}


def _infer_metric_key(text: str) -> str:
    t = (text or "").lower()
    metric_keywords = {
        "revenue": ["revenue", "sales", "营收", "收入"],
        "r_and_d": ["r&d", "research and development", "研发"],
        "net_income": ["net income", "net profit", "净利润", "净利"],
        "net_loss": ["net loss", "loss", "净亏损", "亏损"],
        "cash": ["cash", "cash equivalents", "现金", "现金储备"],
    }
    for metric, kws in metric_keywords.items():
        if any(k in t for k in kws):
            return metric
    return ""


def _check_arithmetic_consistency(text: str) -> str | None:
    t = str(text or "")
    eq_patterns = [
        (r"(-?\d[\d,]*(?:\.\d+)?)\s*\+\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)", lambda a, b: a + b),
        (r"(-?\d[\d,]*(?:\.\d+)?)\s*-\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)", lambda a, b: a - b),
        (r"(-?\d[\d,]*(?:\.\d+)?)\s*[xX\*]\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)", lambda a, b: a * b),
    ]
    for pat, fn in eq_patterns:
        for m in re.finditer(pat, t):
            a = _to_float(m.group(1))
            b = _to_float(m.group(2))
            c = _to_float(m.group(3))
            if a is None or b is None or c is None:
                continue
            expected = fn(a, b)
            if abs(expected - c) > max(abs(expected), 1.0) * 0.02:
                return f"Arithmetic mismatch in '{m.group(0)[:80]}'"

    tl = t.lower()
    if "%" in tl and any(k in tl for k in ("yoy", "growth", "increase", "decrease", "同比", "增长", "下降")):
        nums = [_to_float(x) for x in _extract_numeric_tokens(t)]
        nums = [x for x in nums if x is not None]
        pct_vals = [_to_float(x) for x in re.findall(r"(-?\d[\d,]*(?:\.\d+)?)\s*%", t)]
        pct_vals = [x for x in pct_vals if x is not None]
        if len(nums) >= 2 and pct_vals and abs(nums[0]) > 1e-9:
            derived = (nums[1] - nums[0]) / abs(nums[0]) * 100.0
            if not any(abs(derived - p) <= 3.0 for p in pct_vals):
                return (
                    f"Derived growth {derived:.2f}% inconsistent with stated {pct_vals[0]:.2f}%."
                )
    return None


def _extract_contextual_numeric_claims(response_text: str, max_claims: int = 10) -> list[str]:
    """Extract numeric claims with context; avoid flooding with bare numbers."""
    text = (response_text or "")[:_MAX_RESPONSE_CHARS]
    if not text.strip():
        return []

    # Split on sentence boundaries, but avoid cutting decimal numbers like "1.69".
    parts = re.split(
        r"(?<=[。！？；;\n])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])",
        text,
    )
    claims: list[str] = []
    for p in parts:
        s = p.strip()
        if len(s) < 8:
            continue
        if not re.search(r"-?\d[\d,]*(?:\.\d+)?", s):
            continue
        # Keep only lines with at least one metric/unit signal to reduce noise numbers like "2", "12.9".
        if not re.search(
            r"(revenue|sales|r&d|net|profit|loss|cash|margin|营收|收入|研发|利润|亏损|现金|市占|同比|增长|下降|%"
            r"|USD|CNY|RMB|HKD|\$|¥|￥|亿|万|million|billion)",
            s,
            re.IGNORECASE,
        ):
            continue
        if s not in claims:
            claims.append(s[:240])
        if len(claims) >= max_claims:
            break
    return claims


async def _validate_subtask_claims(
    subtask_id: str,
    subtask_desc: str,
    response_text: str,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use MiniMax to validate claims in one subtask against fetched evidence.

    Returns structured EvidenceClaim list + grounding stats.
    """
    if not response_text or not response_text.strip():
        return {
            "evidence_claims": [],
            "grounded_count": 0,
            "needs_verification_count": 0,
            "ungrounded_count": 0,
            "grounding_ratio": 1.0,
            "strict_grounding_ratio": 1.0,
            "no_evidence": True,
        }

    response_excerpt = sanitize_text_for_llm(response_text[:_MAX_RESPONSE_CHARS])

    if TRUST_EXECUTOR_OUTPUT:
        source_url = ""
        for tr in tool_results:
            if not tr.get("success"):
                continue
            if tr.get("discarded"):
                continue
            r = tr.get("result")
            if isinstance(r, dict):
                candidate = str(r.get("url", "") or "").strip()
                if candidate:
                    source_url = candidate
                    break

        return {
            "evidence_claims": [
                {
                    "claim": "Executor output accepted as trusted evidence.",
                    "verification_status": "grounded",
                    "grounded": True,
                    "reason": (
                        "Trusted-executor mode enabled: validator does not "
                        "re-check numeric/unit/currency consistency."
                    ),
                    "evidence": response_excerpt[:500],
                    "source_url": source_url,
                    "confidence": 1.0,
                    "subtask_id": subtask_id,
                }
            ],
            "grounded_count": 1,
            "needs_verification_count": 0,
            "ungrounded_count": 0,
            "grounding_ratio": 1.0,
            "strict_grounding_ratio": 1.0,
            "no_evidence": False,
            "validator_error": "",
        }

    # Build evidence block from tool results, prioritizing focused snippets.
    evidence_parts: list[str] = []
    total_evidence_chars = 0
    for tr in tool_results:
        if not tr.get("success"):
            continue
        if tr.get("discarded"):
            continue
        r = tr.get("result")
        if not isinstance(r, dict):
            continue
        tool_name = tr.get("tool_name", "")
        block = ""

        if tool_name == "fetch_url":
            url = str(r.get("url", "") or "")
            evidence_snippets = r.get("evidence_snippets") or []
            lines: list[str] = []
            if isinstance(evidence_snippets, list):
                for item in evidence_snippets[:8]:
                    if not isinstance(item, dict):
                        continue
                    snip = str(item.get("snippet", "") or "").strip()
                    term = str(item.get("term", "") or "").strip()
                    if not snip:
                        continue
                    lines.append(f"- [{term}] {snip}" if term else f"- {snip}")
            excerpt = str(r.get("excerpt") or r.get("content") or "")
            payload = ""
            if lines:
                payload = "Evidence snippets:\n" + "\n".join(lines)
            if excerpt:
                excerpt = sanitize_text_for_llm(excerpt[:_MAX_EVIDENCE_PER_SOURCE])
                payload = (payload + "\n\nExcerpt:\n" + excerpt).strip()
            if payload:
                block = f"### Source: {url}\n{payload}"

        elif tool_name == "search_document":
            url = str(r.get("url", "") or "")
            parts: list[str] = []
            for key in ("combined_text", "content", "excerpt", "relevant_text", "text"):
                val = r.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    break
            for key in ("relevant_chunks", "hits", "results", "snippets"):
                val = r.get(key)
                if isinstance(val, list):
                    for hit in val[:6]:
                        if isinstance(hit, dict):
                            htxt = str(
                                hit.get("snippet")
                                or hit.get("text")
                                or hit.get("combined_text")
                                or ""
                            ).strip()
                            if htxt:
                                parts.append(htxt)
                        elif isinstance(hit, str) and hit.strip():
                            parts.append(hit.strip())
            if parts:
                payload = "\n".join(parts)
                payload = sanitize_text_for_llm(payload[:_MAX_EVIDENCE_PER_SOURCE])
                block = f"### Source: {url}\n{payload}"

        elif tool_name in ("sec_edgar_financials", "sec_edgar_filings"):
            entity = str(r.get("entity_name", "") or "")
            cik = str(r.get("cik", "") or "")
            fins = r.get("financials") or r.get("filings") or []
            if fins:
                raw = json.dumps(
                    {"entity": entity, "cik": cik, "data": fins},
                    ensure_ascii=False,
                )[:_MAX_EVIDENCE_PER_SOURCE]
                block = f"### {tool_name} (entity={entity}, cik={cik}):\n{raw}"
            elif r.get("error"):
                logger.info("[Validator] %s reported error: %s", tool_name, str(r.get("error", ""))[:120])

        elif tool_name not in ("fetch_url", "web_search"):
            raw = json.dumps(r, ensure_ascii=False)[:_MAX_EVIDENCE_PER_SOURCE]
            if raw:
                block = f"### {tool_name}:\n{raw}"

        if block and total_evidence_chars < _MAX_EVIDENCE_CHARS:
            remaining = _MAX_EVIDENCE_CHARS - total_evidence_chars
            clipped = block[:remaining]
            evidence_parts.append(clipped)
            total_evidence_chars += len(clipped)

    if not evidence_parts:
        # No source evidence: extract contextual numeric claims and mark as NEEDS_VERIFICATION.
        claims = _extract_contextual_numeric_claims(response_excerpt, max_claims=10)
        evidence_claims = [
            {
                "claim": c,
                "verification_status": "needs_verification",
                "grounded": False,
                "reason": "No source evidence available for this subtask.",
                "evidence": "",
                "source_url": "",
                "confidence": 0.0,
                "subtask_id": subtask_id,
            }
            for c in claims
        ]
        total = len(evidence_claims)
        return {
            "evidence_claims": evidence_claims,
            "grounded_count": 0,
            "needs_verification_count": total,
            "ungrounded_count": 0,
            "grounding_ratio": 0.5 if total > 0 else 1.0,
            "strict_grounding_ratio": 0.0 if total > 0 else 1.0,
            "no_evidence": True,
            "validator_error": "",
        }

    evidence_text = "\n\n".join(evidence_parts)
    user_content = (
        f"## Subtask: {subtask_desc[:200]}\n\n"
        f"## Subtask Analysis (to verify):\n{response_excerpt}\n\n"
        f"## Source Evidence (ground truth fetched from URLs):\n{evidence_text}\n\n"
        f"Please verify the claims in the analysis against the source evidence."
    )

    raw_claims: list[dict[str, Any]] = []
    llm_error = ""
    try:
        settings = get_settings()
        provider = get_provider(provider="qwen", model=settings.planner_model)  # MiniMax-M2.1 via DashScope
        resp = await provider.invoke(
            [
                SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ],
        )
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content)
        raw_claims = [
            ec for ec in (parsed.get("evidence_claims") or [])
            if isinstance(ec, dict)
        ]
    except Exception as e:
        llm_error = str(e)
        logger.warning(f"[Validator] LLM call failed for {subtask_id}: {e}")

    # LLM fallback: extract claims from response text.
    if not raw_claims:
        raw_claims = [
            {
                "claim": c,
                "verification_status": "needs_verification",
                "grounded": False,
                "reason": (
                    "Validator LLM unavailable; extracted claim requires manual verification."
                    if llm_error else
                    "Claim extracted programmatically; requires verification."
                ),
                "evidence": "",
                "source_url": "",
                "confidence": 0.0,
            }
            for c in _extract_contextual_numeric_claims(response_excerpt, max_claims=10)
        ]

    evidence_preview = evidence_text[:3000]
    evidence_metric = _infer_metric_key(evidence_preview)
    processed_claims: list[dict[str, Any]] = []
    grounded_count = 0
    needs_verification_count = 0
    ungrounded_count = 0

    for ec in raw_claims[:12]:
        claim = str(ec.get("claim", "") or "").strip()
        if not claim:
            continue
        claim_nums = _extract_numeric_tokens(claim)
        if not claim_nums:
            continue

        evidence = str(ec.get("evidence", "") or "").strip()
        source_url = str(ec.get("source_url", "") or "").strip()
        confidence = ec.get("confidence", 0.5)
        status_from_model = str(ec.get("verification_status", "") or "").strip().lower()
        reason_from_model = str(ec.get("reason", "") or "").strip()

        source_blob = evidence if evidence else evidence_preview
        source_nums = _extract_numeric_tokens(source_blob)
        locatable = any(_numbers_match(a, b) for a in claim_nums for b in source_nums)

        claim_meta = _extract_meta_tokens(claim)
        source_meta = _extract_meta_tokens(source_blob)
        mismatch_tags: list[str] = []
        if (
            claim_meta["currency"] and source_meta["currency"]
            and claim_meta["currency"].isdisjoint(source_meta["currency"])
        ):
            mismatch_tags.append("currency")
        if (
            claim_meta["unit"] and source_meta["unit"]
            and claim_meta["unit"].isdisjoint(source_meta["unit"])
        ):
            mismatch_tags.append("unit")
        if (
            claim_meta["period"] and source_meta["period"]
            and claim_meta["period"].isdisjoint(source_meta["period"])
        ):
            mismatch_tags.append("period")

        arithmetic_issue = _check_arithmetic_consistency(claim)
        claim_metric = _infer_metric_key(claim)
        contradiction = (
            bool(claim_metric)
            and claim_metric == evidence_metric
            and bool(source_nums)
            and not locatable
        )

        if locatable and not mismatch_tags and not arithmetic_issue:
            status = "grounded"
            reason = "Number located in evidence with consistent unit/currency/period."
        elif contradiction:
            status = "ungrounded"
            reason = "Evidence appears to contain the same metric but with conflicting numeric value."
        else:
            status = "needs_verification"
            reason_parts = []
            if not locatable:
                reason_parts.append("number not clearly located in provided snippet")
            if mismatch_tags:
                reason_parts.append(f"{'/'.join(mismatch_tags)} mismatch")
            if arithmetic_issue:
                reason_parts.append(arithmetic_issue)
            if not reason_parts and status_from_model:
                reason_parts.append(f"model_status={status_from_model}")
            reason = "; ".join(reason_parts) if reason_parts else "insufficient evidence"

        if status_from_model == "ungrounded" and status == "needs_verification":
            # Keep weak-rebuttal policy: do not auto-upgrade to hard contradiction unless programmatic signal exists.
            reason = (reason + "; model suggested ungrounded").strip("; ")

        if reason_from_model and status != "grounded":
            reason = f"{reason}; model_note={reason_from_model[:120]}"

        grounded = status == "grounded"
        if grounded:
            grounded_count += 1
        elif status == "ungrounded":
            ungrounded_count += 1
        else:
            needs_verification_count += 1

        processed_claims.append(
            {
                "claim": claim,
                "verification_status": status,
                "grounded": grounded,
                "reason": reason,
                "evidence": evidence,
                "source_url": source_url,
                "confidence": confidence,
                "subtask_id": subtask_id,
            }
        )

    total_checked = grounded_count + needs_verification_count + ungrounded_count
    weighted_grounding = (
        (grounded_count + 0.5 * needs_verification_count) / total_checked
        if total_checked > 0 else 1.0
    )
    strict_grounding = (
        grounded_count / total_checked
        if total_checked > 0 else 1.0
    )

    result = {
        "evidence_claims": processed_claims,
        "grounded_count": grounded_count,
        "needs_verification_count": needs_verification_count,
        "ungrounded_count": ungrounded_count,
        "grounding_ratio": round(weighted_grounding, 3),
        "strict_grounding_ratio": round(strict_grounding, 3),
        "no_evidence": False,
        "validator_error": llm_error[:300] if llm_error else "",
    }
    logger.info(
        "[Validator] %s: grounded=%d, verify=%d, ungrounded=%d, ratio=%.2f (strict=%.2f)%s",
        subtask_id,
        grounded_count,
        needs_verification_count,
        ungrounded_count,
        result["grounding_ratio"],
        result["strict_grounding_ratio"],
        " [llm_fallback]" if llm_error else "",
    )
    return result


async def validator_node(state: GraphState) -> dict[str, Any]:
    """Validator node: structured cross-subtask claim verification.

    Runs once after all executor subtasks complete, before the Critic.
    For each subtask, uses MiniMax to:
    1. Extract specific factual claims from the LLM analysis
    2. Verify each claim against the fetched tool evidence
    3. Produce structured EvidenceClaim objects

    Outputs:
    - `evidence_claims`: list of all EvidenceClaim dicts across subtasks
    - Updated subtask `validation` with richer structured data
    """
    subtasks = state.get("subtasks", [])
    execution_results = state.get("execution_results", [])
    metrics = state.get("metrics", {})

    # Build lookup map: merge ALL execution_result entries for the same subtask.
    # When a subtask is replanned and runs multiple times, different runs may have
    # different (and complementary) tool_results. The "last entry wins" dict
    # comprehension would lose fetch_url data from earlier runs if a later run
    # used a different strategy (e.g., first run fetched a PDF, second run used
    # sec_edgar). Merging ensures the validator has ALL available evidence.
    exec_map: dict[str, dict[str, Any]] = {}
    for er in execution_results:
        sid = er.get("subtask_id")
        if not sid:
            continue
        if sid not in exec_map:
            exec_map[sid] = dict(er)
        else:
            existing = exec_map[sid]
            merged_tools = list(existing.get("tool_results", []))
            # Add new tools, avoiding duplicates (same tool_name + same URL/query)
            existing_sigs = {
                (t.get("tool_name"), str(t.get("tool_args", {})))
                for t in merged_tools
            }
            for t in er.get("tool_results", []):
                sig = (t.get("tool_name"), str(t.get("tool_args", {})))
                if sig not in existing_sigs:
                    merged_tools.append(t)
                    existing_sigs.add(sig)
            # Use the response from the run with the longer / more complete answer
            if len(er.get("response", "")) > len(existing.get("response", "")):
                exec_map[sid] = {**er, "tool_results": merged_tools}
            else:
                exec_map[sid] = {**existing, "tool_results": merged_tools}
            logger.info(
                "[Validator] Merged tool_results for %s: now %d total entries",
                sid, len(merged_tools),
            )

    all_evidence_claims: list[dict[str, Any]] = []
    updated_subtasks = list(subtasks)  # shallow copy to avoid mutation issues

    for i, subtask in enumerate(updated_subtasks):
        sid = subtask.get("id", "")
        status = subtask.get("status", "")

        # Only validate completed subtasks
        if status != SubtaskStatus.COMPLETED.value:
            continue

        exec_result = exec_map.get(sid, {})
        response_text = exec_result.get("response", "") or str(subtask.get("result", ""))
        tool_results = exec_result.get("tool_results", [])
        subtask_type = subtask.get("subtask_type", "research")
        subtask_desc = subtask.get("description", "")

        # For synthesis/computation subtasks that have no own tool_results,
        # borrow the tool_results from their dependency subtasks so the
        # validator has evidence to verify against (prevents structural false negatives).
        if subtask_type in ("synthesis", "computation") and not tool_results:
            dep_ids = subtask.get("dependencies", [])
            for dep_id in dep_ids:
                dep_er = exec_map.get(dep_id, {})
                dep_tools = dep_er.get("tool_results", [])
                if dep_tools:
                    tool_results = tool_results + dep_tools
                    logger.info(
                        "[Validator] %s is %s — borrowing %d tool_results from dep %s",
                        sid, subtask_type, len(dep_tools), dep_id,
                    )
                if len(tool_results) >= 20:
                    break

        if not response_text:
            continue

        logger.info(f"[Validator] Validating subtask {sid} ({len(response_text)} chars response)")

        validation_result = await _validate_subtask_claims(
            subtask_id=sid,
            subtask_desc=subtask_desc,
            response_text=response_text,
            tool_results=tool_results,
        )

        # Merge structured evidence into existing validation data
        existing_validation = subtask.get("validation") or {}
        merged_validation = {
            **existing_validation,
            "structured_claims": validation_result.get("evidence_claims", []),
            "grounded_claims": validation_result.get("grounded_count", existing_validation.get("grounded_claims", 0)),
            "needs_verification_count": validation_result.get("needs_verification_count", 0),
            "ungrounded_claims": validation_result.get("ungrounded_count", existing_validation.get("ungrounded_claims", 0)),
            "grounding_ratio": validation_result.get("grounding_ratio", 0.0),
            "strict_grounding_ratio": validation_result.get("strict_grounding_ratio", validation_result.get("grounding_ratio", 0.0)),
            "no_evidence": validation_result.get("no_evidence", False),
            "validator_error": validation_result.get("validator_error", ""),
            "verified_by_validator": True,
        }
        updated_subtasks[i] = {**subtask, "validation": merged_validation}

        # Also sync to execution_results
        if sid in exec_map:
            exec_map[sid]["validation"] = merged_validation

        # Collect all evidence claims
        for ec in validation_result.get("evidence_claims", []):
            all_evidence_claims.append(ec)

        # Update metrics for LLM call
        metrics["step_count"] = metrics.get("step_count", 0) + 1

    # Sync updated execution_results back
    updated_exec_results = list(execution_results)
    for j, er in enumerate(updated_exec_results):
        sid = er.get("subtask_id", "")
        if sid in exec_map and exec_map[sid].get("validation"):
            updated_exec_results[j] = {**er, "validation": exec_map[sid]["validation"]}

    total_grounded = sum(1 for ec in all_evidence_claims if ec.get("grounded"))
    total_ungrounded = sum(1 for ec in all_evidence_claims if not ec.get("grounded"))
    logger.info(
        f"[Validator] Complete: {len(all_evidence_claims)} claims checked, "
        f"grounded={total_grounded}, ungrounded={total_ungrounded}"
    )

    # ── P8: Compute data_quality_level ──────────────────────────────────────
    # Aggregate grounding ratios from all validated subtasks.
    grounding_ratios = []
    for st in updated_subtasks:
        val = st.get("validation", {})
        if val.get("verified_by_validator") and isinstance(val.get("grounding_ratio"), float):
            grounding_ratios.append(val["grounding_ratio"])

    if grounding_ratios:
        avg_grounding = sum(grounding_ratios) / len(grounding_ratios)
        # Weight: any subtask below 25% pulls the level down hard
        any_very_low = any(r < 0.25 for r in grounding_ratios)
        if avg_grounding >= 0.65 and not any_very_low:
            data_quality_level = "good"
        elif avg_grounding >= 0.35:
            data_quality_level = "partial"
        else:
            data_quality_level = "poor"
    else:
        # No validated claims — cannot assess quality; assume partial
        data_quality_level = "partial"

    logger.info(
        "[Validator] data_quality_level=%s (avg_grounding=%.2f, %d subtasks)",
        data_quality_level,
        sum(grounding_ratios) / len(grounding_ratios) if grounding_ratios else 0.0,
        len(grounding_ratios),
    )

    # ── P7: Build tried_strategies from low-grounding subtasks ───────────────
    tried_strategies: list[str] = []
    for er in updated_exec_results:
        val = er.get("validation", {})
        gr = val.get("grounding_ratio", 1.0)
        if val.get("no_evidence"):
            desc = er.get("subtask_description", "")[:80]
            tried_strategies.append(
                f"Subtask '{desc}' had no source evidence for verification — enforce fetch_url/search_document before synthesis"
            )
        if gr < 0.25:
            desc = er.get("subtask_description", "")[:80]
            tried_strategies.append(
                f"Subtask '{desc}' had grounding_ratio={gr:.0%} — "
                "change approach: use different data source or tool"
            )
        # Flag low-quality fetch_url URLs
        for tr in er.get("tool_results", []):
            if not tr.get("success"):
                continue
            if tr.get("discarded"):
                continue
            r = tr.get("result", {})
            if tr.get("tool_name") == "fetch_url" and isinstance(r, dict):
                url = r.get("url", "")
                qs = r.get("quality_score", 1.0)
                is_js = r.get("is_js_rendered", False)
                if url and (is_js or qs < 0.3):
                    strategy_str = f"fetch_url({url[:80]}) returned JS-rendered/low-quality content — skip"
                    if strategy_str not in tried_strategies:
                        tried_strategies.append(strategy_str)

    # Policy: do not re-verify/re-fetch citation URLs after subtask completion.
    # Keep executor-produced citations unchanged; report-stage checks will
    # validate claim-source consistency against citation summaries/snippets.
    citations = list(state.get("citations", []))
    enriched_citations = citations

    return {
        "subtasks": updated_subtasks,
        "execution_results": updated_exec_results,
        "evidence_claims": all_evidence_claims,
        "citations": enriched_citations,
        "metrics": metrics,
        "data_quality_level": data_quality_level,
        "tried_strategies": tried_strategies,
    }
