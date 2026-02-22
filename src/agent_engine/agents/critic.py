"""Critic agent for evaluating execution results."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_engine.agents.state import GraphState, SubtaskStatus
from agent_engine.config import get_settings
from agent_engine.llm import get_provider
from agent_engine.tools.builtin.web_search import sanitize_text_for_llm


CRITIC_SYSTEM_PROMPT = """You are a weak-rebuttal critic agent. Your role is to:
1. Evaluate if the executed tasks address the user's request
2. Perform evidence-consistency checks ONLY (no external-knowledge fact rebuttal)
3. Flag verification gaps and internal inconsistencies conservatively
4. Provide constructive next-step suggestions

Original request: {user_request}

Execution summary:
{execution_summary}

CRITICAL EVALUATION CRITERIA (EVIDENCE-CONSISTENCY ONLY):
- Source Traceability: Can each key number be located in cited evidence snippets?
- Unit/Currency/Period Consistency: Are unit, currency, and period aligned between claim and evidence?
- Arithmetic Consistency: Are derived calculations internally consistent?
- Multi-source Conflicts: If sources disagree, mark conflict only; do NOT adjudicate who is right.
- Completeness: Does the output cover all major aspects of the request?

STRICT BOUNDARY:
- Do NOT use external/world knowledge to declare claims true/false.
- Do NOT "correct" numbers using memory.
- If uncertain, output needs_verification items instead of rebuttal.

Domain legitimacy guidance (avoid false positives):
- Do NOT label a domain as "spoof/phishing" merely because it is unfamiliar.
- Only flag spoof/phishing when there is **concrete evidence** (e.g., typosquatting, mismatch between claimed brand and domain, security warnings, or red flags seen in fetched content).
- If legitimacy is unclear, state "domain legitimacy uncertain — needs verification" and suggest verifying via official filings and cross-links from already verified primary sources.

Tool usage guidance (avoid false negatives):
- The executor has access to specialized data tools beyond web_search/fetch_url.
  ALL tools listed in the "Registered tools" section of the execution summary are
  legitimate system-registered tools.
- Using domain-specific structured API tools (listed in "Registered tools" above) is
  PREFERRED over generic web scraping when available — they provide more reliable data.
- Do NOT penalize the executor for choosing specialized tools over web_search.
  Only penalize if the tool output is incorrect, irrelevant, or unverified.

## OUTPUT FORMAT — MANDATORY
You MUST respond with ONLY a raw JSON object. No markdown headers, no prose before or
after, no ```json fences. The ENTIRE response must be valid JSON that can be parsed
directly by json.loads(). Start your response with {{ and end with }}.

Required JSON structure:
{{
    "is_complete": true/false,
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "feedback": "Detailed feedback on the execution, including specific issues found",
    "suggestions": ["specific suggestion 1", "specific suggestion 2"],
    "needs_revision": true/false,
    "needs_verification": ["item that requires verification"],
    "conflicts": ["multi-source conflict description"],
    "data_quality_issues": ["issue 1", "issue 2"],
    "missing_sources": ["claim without source"],
    "incorrect_facts": []
}}

IMPORTANT: Do NOT write any explanation or headers before or after the JSON.
The first character of your response must be {{ and the last must be }}."""


def _extract_critic_json(raw: str) -> dict:
    """Robustly extract the critic JSON from LLM output.

    Handles three response styles:
    1. Pure JSON (ideal)
    2. JSON wrapped in ```json ... ``` fences
    3. Markdown prose that contains a JSON block somewhere inside it
       (e.g. MiniMax returns a full analysis then embeds JSON at the end)

    Falls back to extracting key signals from prose text rather than
    blindly returning needs_revision=True.
    """
    text = raw.strip()

    # Strategy 1: pure JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: ```json ... ``` fences
    if "```json" in text:
        try:
            block = text.split("```json", 1)[1].split("```", 1)[0]
            return json.loads(block.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # Strategy 3: any ``` ... ``` fence
    if "```" in text:
        try:
            block = text.split("```", 1)[1].split("```", 1)[0]
            candidate = block.strip()
            if candidate.startswith("{"):
                return json.loads(candidate)
        except (json.JSONDecodeError, IndexError):
            pass

    # Strategy 4: find the LAST {...} block in the text (works when LLM writes prose then JSON)
    # Walk backwards from the end to find the outermost JSON object.
    last_brace = text.rfind("}")
    if last_brace != -1:
        # Find the matching opening brace by scanning backwards
        depth = 0
        for i in range(last_brace, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i : last_brace + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 5: find the FIRST {...} block (some models put JSON before prose)
    first_brace = text.find("{")
    if first_brace != -1:
        depth = 0
        for i in range(first_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 6: prose fallback — extract key signals from Markdown text
    # Better than blindly returning needs_revision=True
    logger.warning(
        f"[Critic] JSON extraction failed on all strategies. "
        f"Attempting prose signal extraction. Preview: {raw[:200]!r}"
    )
    text_lower = text.lower()

    # Detect key boolean signals from prose
    is_complete = any(p in text_lower for p in ("is complete", "完整", "fully addressed", "完全覆盖"))
    needs_rev = any(p in text_lower for p in (
        "needs revision", "needs_revision", "建议修订", "需要修订", "needs improvement",
        "should revise", "质量不足", "数据缺失", "unverified", "hallucin",
    ))
    is_correct = not any(p in text_lower for p in (
        "incorrect", "inaccurate", "错误", "不准确", "hallucin", "fabricat",
    ))

    # Rough confidence from prose: look for explicit percentage or phrase
    confidence = 0.5
    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', text_lower)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            if confidence > 1.0:
                confidence /= 100.0
        except ValueError:
            pass

    return {
        "is_complete": is_complete,
        "is_correct": is_correct,
        "confidence": confidence,
        "feedback": f"[Parsed from prose — JSON extraction failed] {raw[:800]}",
        "suggestions": ["Review Critic LLM output format; ensure JSON-only response"],
        "needs_revision": needs_rev,
        "needs_verification": [],
        "conflicts": [],
        "data_quality_issues": ["Critic response was not valid JSON; signals extracted from prose"],
        "missing_sources": [],
        "incorrect_facts": [],
        "review_mode": "evidence_consistency_only",
        "_parse_strategy": "prose_fallback",
    }


async def critic_node(state: GraphState) -> dict[str, Any]:
    """Critic node that evaluates execution results.

    Args:
        state: Current graph state.

    Returns:
        Updated state with critic feedback.
    """
    # ── Build LEAN execution summary (context slimming) ──
    # Old approach: dump raw tool logs + full result strings → ~20k chars.
    # New approach: per-subtask compact digest → ~5k chars.
    # Each subtask gets a ~300-char summary with key signals for the critic.
    subtasks = state.get("subtasks", [])
    execution_results = state.get("execution_results", [])
    tool_call_log = state.get("tool_call_log", [])
    citations = state.get("citations", [])

    # Group tool calls by subtask for stats
    subtask_tool_stats: dict[str, dict] = {}
    for log_entry in tool_call_log:
        sid = log_entry.get("subtask_id", "unknown")
        if sid not in subtask_tool_stats:
            subtask_tool_stats[sid] = {"total": 0, "success": 0, "tools": set()}
        subtask_tool_stats[sid]["total"] += 1
        if log_entry.get("success"):
            subtask_tool_stats[sid]["success"] += 1
        subtask_tool_stats[sid]["tools"].add(log_entry.get("tool_name", "?"))

    # Inject registered tool list so the Critic knows all tools are legitimate
    try:
        from agent_engine.tools import get_tool_registry
        registry = get_tool_registry()
        all_tools = registry.list_tools()
        tool_names = [t.name for t in all_tools]
        summary_parts = [
            "## Registered Tools (all are legitimate system tools)",
            f"Available tools: {', '.join(tool_names)}",
            "",
            "## Subtask Digests",
        ]
    except Exception:
        summary_parts = ["## Subtask Digests"]

    for subtask in subtasks:
        sid = subtask.get("id", "?")
        status = subtask.get("status", "unknown")
        status_emoji = {
            SubtaskStatus.COMPLETED.value: "✓",
            SubtaskStatus.FAILED.value: "✗",
            SubtaskStatus.SKIPPED.value: "⊘",
            SubtaskStatus.PENDING.value: "○",
            SubtaskStatus.IN_PROGRESS.value: "◐",
        }.get(status, "?")

        desc = subtask.get("description", "")[:120]
        summary_parts.append(f"\n### {status_emoji} {sid}: {desc}")
        summary_parts.append(f"Status: {status}")

        # Tool stats (compact)
        stats = subtask_tool_stats.get(sid, {})
        if stats:
            tools_used = ", ".join(sorted(stats.get("tools", set())))
            summary_parts.append(
                f"Tools: {stats.get('success',0)}/{stats.get('total',0)} succeeded ({tools_used})"
            )

        # Result digest — compact, not raw dump
        result = subtask.get("result")
        if result:
            if isinstance(result, dict):
                analysis = result.get("analysis", "")
                tool_count = len(result.get("tool_results", []))
                if analysis:
                    summary_parts.append(f"Analysis ({len(analysis)} chars): {analysis[:400]}")
                else:
                    summary_parts.append(f"Analysis: (empty — {tool_count} tool results only)")
            else:
                result_str = str(result)
                summary_parts.append(f"Result: {result_str[:400]}")

        # Quality warnings (important for critic)
        warnings = subtask.get("quality_warnings", [])
        if warnings:
            summary_parts.append(f"Quality warnings: {'; '.join(str(w)[:80] for w in warnings[:3])}")

        # Validation (grounded/ungrounded claims)
        validation = subtask.get("validation", {})
        if validation:
            summary_parts.append(
                f"Validation: grounded={validation.get('grounded_claims', 0)}, "
                f"ungrounded={validation.get('ungrounded_claims', 0)}, "
                f"verified={validation.get('verified', '?')}"
            )

        if subtask.get("error"):
            summary_parts.append(f"Error: {subtask['error'][:200]}")

    # Citation stats (compact — just counts and top verified sources)
    summary_parts.append("\n## Citation Stats")
    verified_citations = [c for c in citations if c.get("verified")]
    with_content = [c for c in citations if c.get("fetched_content", "").strip()]
    summary_parts.append(
        f"Total: {len(citations)}, Verified: {len(verified_citations)}, "
        f"With content: {len(with_content)}"
    )
    # Show top 5 verified sources (title + URL only, no content)
    for c in verified_citations[:5]:
        title = c.get("title", "") or "(no title)"
        url = c.get("url", "")[:80]
        content_len = len(c.get("fetched_content", ""))
        summary_parts.append(f"  ✓ {title[:60]} ({url}) [{content_len} chars]")
    if len(verified_citations) > 5:
        summary_parts.append(f"  ... and {len(verified_citations) - 5} more verified sources")

    # Metrics (compact)
    metrics = state.get("metrics", {})
    summary_parts.append(
        f"\n## Metrics: {metrics.get('total_tokens', 0)} tokens, "
        f"{metrics.get('step_count', 0)} steps, "
        f"{metrics.get('tool_call_count', 0)} tool calls"
    )

    # Sanitize to prevent content moderation triggers
    execution_summary = sanitize_text_for_llm("\n".join(summary_parts))
    logger.info(
        f"[Critic] Execution summary: {len(execution_summary)} chars "
        f"(subtasks={len(subtasks)}, citations={len(citations)})"
    )

    # Get LLM with critic-specific model
    settings = get_settings()
    provider = get_provider(model=settings.critic_model)

    # Build messages
    system_msg = SystemMessage(
        content=CRITIC_SYSTEM_PROMPT.format(
            user_request=state.get("user_request", ""),
            execution_summary=execution_summary,
        )
    )

    user_msg = HumanMessage(
        content="""Evaluate the execution with weak-rebuttal policy.

Allowed checks only:
1. Whether numerical claims can be located in cited evidence snippets
2. Unit/currency/period consistency between claim and evidence
3. Arithmetic consistency for derived calculations
4. Multi-source conflicts (mark conflict only, do not adjudicate)
5. Completeness of requested coverage

Do NOT use external knowledge to rebut claims.
When uncertain, add items to needs_verification instead of declaring incorrect facts."""
    )

    messages = [system_msg, user_msg]

    # Invoke LLM — with 400 error resilience for content moderation blocks
    try:
        response = await provider.invoke(messages)
    except Exception as e:
        error_msg = str(e)
        is_moderation_block = (
            "400" in error_msg
            or "data_inspection_failed" in error_msg
            or "inappropriate content" in error_msg.lower()
            or "content moderation" in error_msg.lower()
        )
        if is_moderation_block:
            logger.error(
                f"[Critic] Content moderation blocked the request (400). "
                f"Returning conservative fallback evaluation. Error: {error_msg[:200]}"
            )
            # Moderation block: mark needs_revision=True on FIRST occurrence
            # so the pipeline attempts a re-plan with sanitized content.
            # On subsequent moderation blocks (iteration >= 2), allow the pipeline
            # to proceed to avoid infinite loops.
            iteration = state.get("iteration_count", 0) + 1
            is_first_moderation = iteration <= 1
            feedback = {
                "is_complete": not is_first_moderation,
                "is_correct": False,
                "confidence": 0.15,
                "feedback": (
                    "Critic evaluation was blocked by content moderation. "
                    + (
                        "Requesting revision to sanitize content before proceeding."
                        if is_first_moderation
                        else "Second moderation block — accepting results with very low confidence."
                    )
                ),
                "suggestions": [
                    "Review search result quality — some results may contain spam content",
                    "Consider narrowing search queries to avoid contaminated results",
                    "Sanitize fetched content before passing to Critic",
                ],
                "needs_revision": is_first_moderation,
                "needs_verification": [
                    "Critic evaluation blocked by moderation; verify key numbers manually from cited sources."
                ],
                "conflicts": [],
                "data_quality_issues": [
                    "Content moderation triggered during evaluation",
                    "Search results may contain inappropriate material that must be filtered",
                ],
                "missing_sources": [],
                "incorrect_facts": [],
                "review_mode": "evidence_consistency_only",
            }
            ai_msg = AIMessage(
                content=f"Critic evaluation (content moderation fallback):\n{json.dumps(feedback, indent=2)}"
            )
            metrics["step_count"] = metrics.get("step_count", 0) + 1
            return {
                "critic_feedback": feedback,
                "status": "completed",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [ai_msg],
                "metrics": metrics,
                "reuse_mode": state.get("reuse_mode", "fuzzy"),
            }
        else:
            raise

    # Update metrics
    metrics["input_tokens"] = metrics.get("input_tokens", 0) + response.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + response.output_tokens
    metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
    metrics["step_count"] = metrics.get("step_count", 0) + 1

    # Parse response — multi-strategy extractor for LLMs that return Markdown prose
    feedback = _extract_critic_json(response.content)

    # Normalize optional fields returned by critic LLM.
    feedback.setdefault("suggestions", [])
    feedback.setdefault("data_quality_issues", [])
    feedback.setdefault("missing_sources", [])
    feedback.setdefault("incorrect_facts", [])
    feedback.setdefault("needs_verification", [])
    feedback.setdefault("conflicts", [])
    feedback["review_mode"] = "evidence_consistency_only"

    # Convert LLM "incorrect_facts" into conservative verification tasks.
    llm_incorrect = [
        str(x).strip() for x in (feedback.get("incorrect_facts") or [])
        if str(x).strip()
    ]
    if llm_incorrect:
        feedback["needs_verification"].extend(
            [f"Potential issue (not adjudicated): {x}" for x in llm_incorrect[:20]]
        )
        feedback["incorrect_facts"] = []

    # ── Programmatic evidence-consistency checks (weak-rebuttal mode) ──
    consistency = _programmatic_evidence_consistency_check(
        subtasks=subtasks,
        citations=citations,
        evidence_claims=state.get("evidence_claims", []),
    )
    if consistency["needs_verification"] or consistency["conflicts"]:
        logger.info(
            "[Critic] Evidence consistency: verify=%d, conflicts=%d, hard_failures=%d, high_trust_ratio=%.2f",
            len(consistency["needs_verification"]),
            len(consistency["conflicts"]),
            consistency["hard_failures"],
            consistency["high_trust_ratio"],
        )
    feedback["needs_verification"] = _dedupe_items(
        list(feedback.get("needs_verification", []))
        + consistency["needs_verification"]
    )
    feedback["conflicts"] = _dedupe_items(
        list(feedback.get("conflicts", []))
        + consistency["conflicts"]
    )

    if consistency["needs_verification"]:
        feedback["data_quality_issues"].append(
            f"{len(consistency['needs_verification'])} evidence-consistency items require verification."
        )
    if consistency["conflicts"]:
        feedback["data_quality_issues"].append(
            f"{len(consistency['conflicts'])} multi-source numerical conflicts detected (not adjudicated)."
        )
    feedback["data_quality_issues"] = _dedupe_items(list(feedback.get("data_quality_issues", [])))

    # Weak-rebuttal decision policy:
    # - uncertainty -> needs_verification (default, no forced rebuttal)
    # - only escalate to revision on repeated low-trust hard failures.
    hard_failures = int(consistency.get("hard_failures", 0))
    high_trust_ratio = float(consistency.get("high_trust_ratio", 0.0))
    if hard_failures >= 2 and high_trust_ratio < 0.6:
        feedback["needs_revision"] = True
        feedback["is_correct"] = False
    elif feedback.get("needs_revision") and hard_failures == 0:
        # Avoid unnecessary replan when issues are only "unverified", not contradicted.
        feedback["needs_revision"] = False
        if feedback.get("is_complete", False):
            feedback["is_correct"] = True

    # ── Programmatic confidence adjustment ──
    # The LLM's confidence is purely subjective. We compute an objective
    # score based on measurable signals and blend them together.
    # This prevents scenarios where the LLM gives 0.28 but the data
    # actually exists in the citations (just not in the analysis text).
    llm_confidence = feedback.get("confidence", 0.5)
    prog_confidence = _compute_programmatic_confidence(subtasks, citations, execution_results)

    # Blend: 40% programmatic, 60% LLM (LLM has richer context but can be harsh)
    blended_confidence = prog_confidence * 0.4 + llm_confidence * 0.6
    logger.info(
        f"[Critic] Confidence: LLM={llm_confidence:.2f}, "
        f"programmatic={prog_confidence:.2f}, "
        f"blended={blended_confidence:.2f}"
    )
    feedback["confidence"] = round(blended_confidence, 2)
    feedback["confidence_detail"] = {
        "llm_raw": llm_confidence,
        "programmatic": round(prog_confidence, 2),
        "blended": round(blended_confidence, 2),
    }

    # Determine final status
    iteration = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 10)

    # Limit revision loops: allow at most 3 revisions (increased from 2 to leverage
    # critic feedback for smarter replanning — each replan now uses specific feedback
    # from the critic to improve data source strategy and avoid repeated mistakes)
    max_revisions = min(max_iterations, 3)

    # If token budget already exceeded, don't allow more revisions
    # With smart replan (reusing completed subtask results), replanning is much
    # cheaper, so we use a higher threshold (95% instead of 90%).
    total_tokens = metrics.get("total_tokens", 0)
    max_tokens = state.get("max_tokens", 500000)
    budget_exhausted = total_tokens > max_tokens * 0.95  # 95% threshold (was 90%)

    # Also check step and tool budgets
    max_steps = state.get("max_steps", 100)
    max_tool_calls = state.get("max_tool_calls", 200)
    step_count = metrics.get("step_count", 0)
    tool_count = metrics.get("tool_call_count", 0)
    step_exhausted = step_count > max_steps * 0.95
    tool_exhausted = tool_count > max_tool_calls * 0.95
    any_budget_exhausted = budget_exhausted or step_exhausted or tool_exhausted

    logger.info(
        f"[Critic] Decision inputs: needs_revision={feedback.get('needs_revision')}, "
        f"iteration={iteration}, max_revisions={max_revisions}, "
        f"budget_exhausted={any_budget_exhausted} "
        f"(tokens={total_tokens}/{max_tokens} [{total_tokens/max(max_tokens,1):.0%}], "
        f"steps={step_count}/{max_steps} [{step_count/max(max_steps,1):.0%}], "
        f"tools={tool_count}/{max_tool_calls} [{tool_count/max(max_tool_calls,1):.0%}])"
    )

    if feedback.get("needs_revision") and iteration < max_revisions and not any_budget_exhausted:
        new_status = "planning"
        logger.info("[Critic] → status=planning (revision allowed)")
    elif feedback.get("is_complete") and feedback.get("is_correct"):
        new_status = "completed"
        logger.info("[Critic] → status=completed (all good)")
    else:
        new_status = "completed"  # Accept results after review, don't loop endlessly
        reason_parts = []
        if feedback.get("needs_revision"):
            if iteration >= max_revisions:
                reason_parts.append(f"max_revisions reached ({iteration}>={max_revisions})")
            if any_budget_exhausted:
                reason_parts.append("budget exhausted")
        else:
            reason_parts.append(f"is_complete={feedback.get('is_complete')}, is_correct={feedback.get('is_correct')}")
        logger.info(
            f"[Critic] → status=completed (reason: {'; '.join(reason_parts) or 'default fallthrough'})"
        )

    # Add AI message
    ai_msg = AIMessage(
        content=f"Critic evaluation:\n{json.dumps(feedback, indent=2)}"
    )

    # Control executor reuse strategy for the next iteration.
    # If we need revision, enforce strict reuse (fingerprint match only) to avoid
    # repeating evaluation of the exact same old results.
    reuse_mode = state.get("reuse_mode", "fuzzy")
    if feedback.get("needs_revision"):
        reuse_mode = "strict"
    elif feedback.get("is_complete") and feedback.get("is_correct"):
        reuse_mode = "fuzzy"

    return {
        "critic_feedback": feedback,
        "status": new_status,
        "iteration_count": iteration,
        "messages": [ai_msg],
        "metrics": metrics,
        "reuse_mode": reuse_mode,
    }


def route_after_critic(state: GraphState) -> str:
    """Route after critic evaluation.

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    feedback = state.get("critic_feedback", {})
    status = state.get("status", "")
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    logger.info(
        f"[ROUTE] route_after_critic: status={status!r}, "
        f"needs_revision={feedback.get('needs_revision')}, "
        f"iteration={iteration}, max_iterations={max_iterations}"
    )

    if status == "completed":
        logger.info("[ROUTE] → reporter (status=completed)")
        return "reporter"

    if status == "failed":
        logger.info("[ROUTE] → end (status=failed)")
        return "end"

    if feedback.get("needs_revision", False):
        if iteration < max_iterations:
            logger.info(f"[ROUTE] → planner (needs revision, iter {iteration}<{max_iterations})")
            return "planner"

    logger.info("[ROUTE] → reporter (default fallthrough)")
    return "reporter"


def _dedupe_items(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = str(raw or "").strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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

    # Explicit equations.
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
                return f"Arithmetic mismatch in claim: '{m.group(0)[:80]}'"

    # Growth-style consistency: compare (to-from)/from with stated percentage.
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
                    f"Derived growth {derived:.2f}% inconsistent with stated percentage "
                    f"{pct_vals[0]:.2f}%."
                )
    return None


def _programmatic_evidence_consistency_check(
    subtasks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    evidence_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consistency checks only; never adjudicates factual truth via external knowledge."""
    needs_verification: list[str] = []
    conflicts: list[str] = []
    hard_failures = 0

    url_tier: dict[str, str] = {}
    for c in citations:
        url = str(c.get("url", "") or "")
        if url:
            url_tier[url] = str(c.get("source_tier", "unknown") or "unknown")

    high_trust_tiers = {"regulator", "company_ir"}
    checked_claims = 0
    high_trust_claims = 0

    # Merge structured claims from subtasks + state-level evidence_claims.
    merged_claims: list[dict[str, Any]] = []
    for st in subtasks:
        val = st.get("validation") or {}
        for ec in val.get("structured_claims", []):
            if isinstance(ec, dict):
                merged_claims.append(ec)
    for ec in evidence_claims or []:
        if isinstance(ec, dict):
            merged_claims.append(ec)

    metric_records: dict[tuple[str, str], list[tuple[float, str, str]]] = {}

    for ec in merged_claims:
        claim = str(ec.get("claim", "") or "").strip()
        evidence = str(ec.get("evidence", "") or "").strip()
        source_url = str(ec.get("source_url", "") or "").strip()
        grounded = bool(ec.get("grounded", False))

        claim_nums = _extract_numeric_tokens(claim)
        if not claim_nums:
            continue

        checked_claims += 1
        tier = url_tier.get(source_url, "unknown")
        high_trust = tier in high_trust_tiers
        if high_trust:
            high_trust_claims += 1

        if not grounded or not evidence:
            needs_verification.append(
                f"Missing grounded evidence for numeric claim: {claim[:120]}"
            )
            if not high_trust:
                hard_failures += 1
            continue

        evidence_nums = _extract_numeric_tokens(evidence)
        if not any(_numbers_match(a, b) for a in claim_nums for b in evidence_nums):
            needs_verification.append(
                f"Number not located in cited snippet: claim='{claim[:90]}', source='{source_url[:80]}'"
            )
            if not high_trust:
                hard_failures += 1

        claim_meta = _extract_meta_tokens(claim)
        evidence_meta = _extract_meta_tokens(evidence)
        mismatch_tags: list[str] = []
        if (
            claim_meta["currency"] and evidence_meta["currency"]
            and claim_meta["currency"].isdisjoint(evidence_meta["currency"])
        ):
            mismatch_tags.append("currency")
        if (
            claim_meta["unit"] and evidence_meta["unit"]
            and claim_meta["unit"].isdisjoint(evidence_meta["unit"])
        ):
            mismatch_tags.append("unit")
        if (
            claim_meta["period"] and evidence_meta["period"]
            and claim_meta["period"].isdisjoint(evidence_meta["period"])
        ):
            mismatch_tags.append("period")
        if mismatch_tags:
            needs_verification.append(
                f"Claim/evidence {', '.join(mismatch_tags)} mismatch: '{claim[:100]}'"
            )
            if not high_trust:
                hard_failures += 1

        arithmetic_issue = _check_arithmetic_consistency(claim)
        if arithmetic_issue:
            needs_verification.append(arithmetic_issue)
            if not high_trust:
                hard_failures += 1

        # Prepare conflict detection records (metric + period).
        metric = _infer_metric_key(f"{claim} {evidence}")
        period_tokens = claim_meta["period"] or evidence_meta["period"]
        period = sorted(period_tokens)[0] if period_tokens else ""
        v = _to_float(claim_nums[0]) if claim_nums else None
        if metric and period and v is not None and source_url:
            metric_records.setdefault((metric, period), []).append((v, source_url, tier))

    # Multi-source conflict detection: mark only, never adjudicate.
    for (metric, period), recs in metric_records.items():
        if len(recs) < 2:
            continue
        unique_urls = {u for _, u, _ in recs}
        if len(unique_urls) < 2:
            continue

        def _far(v1: float, v2: float) -> bool:
            return abs(v1 - v2) / max(abs(v1), abs(v2), 1.0) > 0.02

        base = recs[0][0]
        if any(_far(base, v) for v, _, _ in recs[1:]):
            sample = "; ".join(
                f"{v:g} @ {u[:50]}" for v, u, _ in recs[:3]
            )
            conflicts.append(
                f"Multi-source conflict for {metric} ({period}): {sample}"
            )
            # Escalate only when all conflicting sources are low trust.
            if all(t not in high_trust_tiers for _, _, t in recs):
                hard_failures += 1

    high_trust_ratio = high_trust_claims / max(checked_claims, 1)
    return {
        "needs_verification": _dedupe_items(needs_verification)[:40],
        "conflicts": _dedupe_items(conflicts)[:20],
        "hard_failures": hard_failures,
        "high_trust_ratio": round(high_trust_ratio, 3),
    }


def _compute_programmatic_confidence(
    subtasks: list[dict],
    citations: list[dict],
    execution_results: list[dict],
) -> float:
    """Compute an objective confidence score based on measurable signals.

    Factors considered:
    1. Subtask completion rate (how many completed vs total)
    2. Non-empty analysis rate (how many subtasks have actual analysis text)
    3. Citation coverage (how many citations are verified with content)
    4. Evidence density (total chars of fetched_content)
    5. Tool success rate (successful vs failed tool calls)
    6. **Grounding ratio** — fraction of numerical claims verified against evidence

    Returns:
        Float between 0.0 and 1.0.
    """
    if not subtasks:
        return 0.5  # No subtasks to evaluate

    scores = []

    # Factor 1: Subtask completion rate (weight: 0.20)
    total = len(subtasks)
    completed = sum(
        1 for s in subtasks
        if s.get("status") in ("completed", SubtaskStatus.COMPLETED.value)
    )
    completion_rate = completed / total if total > 0 else 0
    scores.append(("completion", completion_rate, 0.20))

    # Factor 2: Non-empty analysis rate (weight: 0.15)
    with_analysis = 0
    for s in subtasks:
        result = s.get("result", "")
        if isinstance(result, dict):
            analysis = result.get("analysis", "")
        elif isinstance(result, str):
            analysis = result
        else:
            analysis = ""
        if analysis and len(str(analysis).strip()) > 50:
            with_analysis += 1
    analysis_rate = with_analysis / total if total > 0 else 0
    scores.append(("analysis", analysis_rate, 0.15))

    # Factor 3: Citation quality (weight: 0.15)
    if citations:
        verified = sum(1 for c in citations if c.get("verified"))
        with_content = sum(
            1 for c in citations
            if c.get("fetched_content") and len(c.get("fetched_content", "")) > 100
        )
        citation_score = min(with_content / 5.0, 1.0)
    else:
        citation_score = 0
    scores.append(("citations", citation_score, 0.15))

    # Factor 4: Evidence density — total quality of fetched content (weight: 0.10)
    total_content_chars = sum(
        len(c.get("fetched_content", ""))
        for c in citations if c.get("verified")
    )
    density_score = min(total_content_chars / 15000.0, 1.0)
    scores.append(("density", density_score, 0.10))

    # Factor 5: Tool success rate (weight: 0.10)
    all_tool_results = []
    for er in execution_results:
        for tr in er.get("tool_results", []):
            all_tool_results.append(tr)
    if all_tool_results:
        successful = sum(1 for tr in all_tool_results if tr.get("success"))
        tool_rate = successful / len(all_tool_results)
    else:
        tool_rate = 0.5
    scores.append(("tools", tool_rate, 0.10))

    # Factor 6: GROUNDING RATIO — fraction of numerical claims verified (weight: 0.30)
    # This is the most important anti-hallucination signal.
    # It directly measures how many numbers in the analysis can be traced to evidence.
    total_grounded = 0
    total_ungrounded = 0
    no_evidence_subtasks = 0
    for s in subtasks:
        validation = s.get("validation") or {}
        g = validation.get("grounded_claims", 0)
        u = validation.get("ungrounded_claims", 0)
        total_grounded += g
        total_ungrounded += u
        # Extra penalty: subtasks that answered from LLM memory with no tool evidence
        if validation.get("no_tool_evidence") and (g + u) > 0:
            no_evidence_subtasks += 1

    total_claims = total_grounded + total_ungrounded
    if total_claims > 0:
        grounding_score = total_grounded / total_claims
        # Heavy penalty for subtasks with zero evidence
        if no_evidence_subtasks > 0:
            penalty = no_evidence_subtasks * 0.15
            grounding_score = max(0.0, grounding_score - penalty)
    else:
        # No numerical claims — neutral (could be a qualitative task)
        grounding_score = 0.7
    scores.append(("grounding", grounding_score, 0.30))

    # Compute weighted average
    weighted_sum = sum(score * weight for _, score, weight in scores)
    total_weight = sum(weight for _, _, weight in scores)
    final = weighted_sum / total_weight if total_weight > 0 else 0.5

    logger.info(
        f"[Critic] Programmatic confidence breakdown: "
        + ", ".join(f"{name}={score:.2f}*{weight}" for name, score, weight in scores)
        + f" → {final:.2f}"
    )

    return final
