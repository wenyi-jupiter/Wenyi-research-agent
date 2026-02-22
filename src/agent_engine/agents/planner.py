"""Planner agent for task decomposition."""

import json
import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_engine.agents.state import GraphState, SubtaskStatus
from agent_engine.config import get_settings
from agent_engine.llm import get_provider
from agent_engine.agents.domain_profile import detect_domain_profile
from agent_engine.tools.builtin.web_search import sanitize_text_for_llm

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a task planning agent. Your role is to:
1. Analyze the user's request and break it down into actionable subtasks
2. Identify dependencies between subtasks
3. Create a clear execution plan

For each subtask, provide:
- A unique ID (use format: subtask_001, subtask_002, etc.)
- A clear description of what needs to be done
- Any dependencies on other subtasks (list their IDs)
- Which tools might be needed
- A "subtask_type" field (REQUIRED, see below)
- An "expected_outputs" list (REQUIRED for synthesis/computation, optional for research)

## REQUIRED: subtask_type field

Every subtask MUST include a "subtask_type" field. Choose from:
- "research"     — needs to fetch external data from the web (use web_search + fetch_url)
- "computation"  — only needs to compute/calculate from data already in previous subtasks
- "synthesis"    — only needs to aggregate/summarize results from previous subtasks (no new data needed)

Rules:
- A subtask with no dependencies is almost always "research"
- A subtask that says "综合前述" / "基于以上" / "整合所有" / "compile/synthesize" is "synthesis"
- A subtask that only does math/calculation from prior results is "computation"
- If unsure, prefer "research" over "synthesis"

## Sequential Workflow Orchestration

For synthesis and computation subtasks, declare what outputs you expect from each
dependency. This allows the executor to precisely extract only the needed fields.

Use the "expected_inputs" field:
{{
  "id": "subtask_003",
  "subtask_type": "synthesis",
  "dependencies": ["subtask_001", "subtask_002"],
  "expected_inputs": [
    {{"from": "subtask_001", "fields": ["metric_a_value", "metric_b_value"]}},
    {{"from": "subtask_002", "fields": ["item_count", "key_items_list"]}}
  ],
  "expected_outputs": ["summary_result", "risk_assessment"]
}}

Available tools:
{tools}

Respond with a JSON object containing:
{{
    "analysis": "Brief analysis of the request",
    "subtasks": [
        {{
            "id": "subtask_001",
            "description": "What needs to be done",
            "subtask_type": "research",
            "dependencies": [],
            "expected_inputs": [],
            "expected_outputs": ["entity_id", "latest_filing_url"],
            "suggested_tools": ["tool_name"]
        }}
    ],
    "execution_order": ["subtask_001", "subtask_002"]
}}

Keep the plan focused and actionable. Break complex tasks into manageable pieces.
If the task is simple, a single subtask is acceptable.

{domain_strategy}

## IMPORTANT: Content safety in subtask descriptions
- Subtask descriptions are sent to an LLM API that has content moderation.
- Keep all subtask descriptions **objective, neutral, and analytical**.
- Use descriptive verbs: "分析", "评估", "总结", "梳理", "比较", "调研", "整理"
- Do NOT use subjective, prescriptive, or directive phrasing that could be
  interpreted as personal advice, endorsement, or incitement in any domain.
- This is critical because subjective/directive phrasing can trigger API content
  moderation filters and crash the entire task.

## IMPORTANT: Domain legitimacy (avoid false "spoof" bans)
- Do NOT ban a domain as "spoof" merely because it is unfamiliar.
- Only mark a domain as spoof/phishing when there is **evidence**, such as:
  - clear typosquatting patterns (look-alike characters, extra words, suspicious subdomains),
  - content mismatch (page claims Brand A but domain is unrelated),
  - security warnings, or other concrete red flags observed in fetched content.
- If domain legitimacy is uncertain, phrase it as "needs verification" and plan verification via:
  - official filings or regulatory announcements that reference the domain,
  - cross-links from already verified official sources,
  - multiple independent reputable sources pointing to the same domain."""


async def planner_node(state: GraphState) -> dict[str, Any]:
    """Planner node that decomposes tasks into subtasks.

    Args:
        state: Current graph state.

    Returns:
        Updated state with subtasks.
    """
    # Get LLM provider with planner-specific model.
    # Planner uses MiniMax-M2.1 via DashScope for higher quality task decomposition.
    settings = get_settings()
    provider = get_provider(provider="qwen", model=settings.planner_model)

    # Get available tools
    from agent_engine.tools import get_tool_registry

    registry = get_tool_registry()
    tools_list = registry.list_tools()
    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools_list])

    # Detect domain and inject pluggable data source strategy
    user_request = state.get("user_request", "")
    domain = detect_domain_profile(user_request)
    domain_strategy = domain.data_source_strategy

    # Build messages
    system_msg = SystemMessage(
        content=PLANNER_SYSTEM_PROMPT.format(
            tools=tools_desc,
            domain_strategy=domain_strategy,
        )
    )

    # Include memory context if available
    context = ""
    if state.get("memory_context"):
        context = f"\n\nRelevant context from memory:\n{state['memory_context']}"

    # ── Include Critic feedback for replanning ──
    critic_context = ""
    critic_feedback = state.get("critic_feedback")
    prev_subtasks = state.get("subtasks", [])
    reuse_mode = state.get("reuse_mode", "fuzzy")
    if critic_feedback and critic_feedback.get("needs_revision"):
        # When replanning after critic feedback, make executor reuse conservative:
        # only reuse results for subtasks whose descriptions are identical (fingerprint match).
        reuse_mode = "strict"
        feedback_text = critic_feedback.get("feedback", "")
        suggestions = critic_feedback.get("suggestions", [])
        data_issues = critic_feedback.get("data_quality_issues", [])
        missing = critic_feedback.get("missing_sources", [])
        incorrect = critic_feedback.get("incorrect_facts", [])

        critic_parts = ["\n\n## ⚠ PREVIOUS ATTEMPT FAILED — Critic Feedback:"]
        if feedback_text:
            critic_parts.append(f"Feedback: {feedback_text[:2000]}")
        if suggestions:
            critic_parts.append(f"Suggestions: {'; '.join(suggestions[:5])}")
        if data_issues:
            critic_parts.append(f"Data quality issues: {'; '.join(data_issues[:5])}")
        if missing:
            critic_parts.append(f"Missing sources: {'; '.join(missing[:5])}")
        if incorrect:
            critic_parts.append(f"Incorrect facts: {'; '.join(incorrect[:5])}")

        # ── Show which subtasks were already completed (so the planner can reuse them) ──
        completed_prev = [
            s for s in prev_subtasks
            if s.get("status") == SubtaskStatus.COMPLETED.value
        ]
        if completed_prev:
            critic_parts.append(
                "\n## COMPLETED SUBTASKS FROM PREVIOUS ATTEMPT "
                "(you SHOULD keep these unless the critic flagged them as incorrect):"
            )
            for cs in completed_prev:
                result_preview = ""
                if cs.get("result"):
                    result_preview = str(cs["result"])[:300]
                critic_parts.append(
                    f"  - [{cs['id']}] {cs['description']}"
                    f"\n    Result preview: {result_preview}"
                )

        # ── P7: Strategy blacklist from tried_strategies ──
        tried = list(state.get("tried_strategies", []))
        if not tried:
            # Auto-derive from execution_results: collect failed/low-grounding approaches
            exec_results = state.get("execution_results", [])
            # Track JS-rendered domains to create domain-level blacklist
            _js_domains: dict[str, int] = {}  # domain → count of JS failures

            for er in exec_results:
                validation = er.get("validation", {})
                grounding = validation.get("grounding_ratio", 1.0)
                tool_results = er.get("tool_results", [])
                # Collect failed fetch_url URLs and domains
                for tr in tool_results:
                    if not tr.get("success"):
                        continue
                    tn = tr.get("tool_name", "")
                    r = tr.get("result", {})
                    if tn == "fetch_url" and isinstance(r, dict):
                        url = r.get("url", "")
                        qs = r.get("quality_score", 1.0)
                        is_js = r.get("is_js_rendered", False)
                        if url and (is_js or qs < 0.3):
                            tried.append(
                                f"fetch_url({url[:80]}) — JS-rendered / low-quality, skip this URL"
                            )
                            # Extract domain for domain-level blacklist
                            try:
                                from urllib.parse import urlparse as _urlparse
                                _domain = _urlparse(url).netloc
                                if _domain:
                                    _js_domains[_domain] = _js_domains.get(_domain, 0) + 1
                            except Exception:
                                pass
                # Flag subtasks with very low grounding
                if grounding < 0.25 and er.get("subtask_description"):
                    desc_preview = er["subtask_description"][:80]
                    tried.append(
                        f"Strategy for '{desc_preview}' produced <25% grounding — use different sources"
                    )

            # Add domain-level blacklist for repeatedly failing JS domains
            for _domain, _cnt in _js_domains.items():
                if _cnt >= 2:
                    tried.append(
                        f"Domain '{_domain}' consistently returns JS-rendered pages (failed {_cnt}x) "
                        f"— AVOID ALL URLs from this domain entirely"
                    )

        if tried:
            critic_parts.append(
                "\n## ❌ FAILED STRATEGIES — DO NOT RETRY THESE:\n"
                + "\n".join(f"  - {s}" for s in tried[:15])
                + "\n  → For each failed strategy above, plan an ALTERNATIVE approach: "
                "use a different tool, different URL pattern, or different data source.\n"
                "  → JS-rendered domains require BROWSER rendering — use SEC EDGAR API, "
                "news articles, or direct PDF links instead."
            )

        critic_parts.append(
            "\nYou MUST address these issues in your new plan. "
            "IMPORTANT: Re-use completed subtasks that the critic did NOT flag as incorrect "
            "by keeping the same ID and description — these will be skipped by the executor. "
            "Only create new subtasks for the UNFINISHED or INCORRECT work. "
            "Use different data sources, more specific search queries, "
            "and avoid the same mistakes. "
            "If portal pages failed due to JS rendering, "
            "plan to search news articles and official source pages instead.\n"
            "If a URL pattern consistently fails, switch to a completely different source "
            "(e.g., if SEC EDGAR HTML fails, try the XBRL API or news summaries; "
            "if company IR pages fail, try financial news wires or regulatory filings)."
        )
        # Sanitize critic context — it may contain unsafe snippets from
        # search results that propagated through tool results → critic feedback
        critic_context = sanitize_text_for_llm("\n".join(critic_parts))

    # Place the user request FIRST and prominently marked so the LLM keeps focus
    # on the original task even when critic feedback is long (8-11K chars).
    # user_request already declared above (used for domain detection)
    if not user_request:
        logger.warning(
            "[Planner] user_request is empty at replan time! "
            "This should never happen — state may be corrupted."
        )
    if critic_context:
        user_msg = HumanMessage(
            content=(
                f"## ORIGINAL USER REQUEST (your plan MUST stay focused on this):\n"
                f"{user_request}\n"
                f"{context}"
                f"{critic_context}\n\n"
                f"## REMINDER: All subtasks MUST directly serve the original request above. "
                f"Do NOT create subtasks unrelated to the user's topic."
            )
        )
    else:
        user_msg = HumanMessage(
            content=f"Please create a plan for the following request:{context}\n\n{user_request}"
        )

    messages = [system_msg, user_msg]

    # Invoke LLM — with content moderation resilience.
    # Planner input may contain critic feedback with contaminated snippets.
    metrics = state.get("metrics", {})
    response = None
    for planner_attempt in range(3):
        try:
            response = await provider.invoke(messages)
            break
        except Exception as e:
            error_msg = str(e)
            is_moderation = (
                "data_inspection_failed" in error_msg
                or "inappropriate content" in error_msg.lower()
                or ("400" in error_msg and "bad request" in error_msg.lower())
            )
            if is_moderation and planner_attempt < 2:
                logger.warning(
                    f"[Planner] Content moderation blocked (attempt "
                    f"{planner_attempt + 1}): {error_msg[:200]}"
                )
                # Sanitize all message content and retry
                sanitized = []
                for msg in messages:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        clean = sanitize_text_for_llm(msg.content)
                        if clean != msg.content:
                            msg = msg.copy(update={"content": clean})
                    sanitized.append(msg)
                messages = sanitized

                # If this is a replan with heavy critic context, strip it
                # to reduce the chance of triggering moderation again.
                if planner_attempt == 1 and critic_context:
                    logger.warning(
                        "[Planner] Stripping critic context for retry"
                    )
                    user_msg = HumanMessage(
                        content=(
                            f"Please create a plan for the following request:"
                            f"{context}\n\n{user_request}"
                        )
                    )
                    messages = [system_msg, user_msg]
                continue
            else:
                raise

    if response is None:
        # Should not happen (loop either breaks or raises), but guard anyway
        raise RuntimeError("[Planner] Failed to get LLM response after retries")

    # Update metrics
    metrics["input_tokens"] = metrics.get("input_tokens", 0) + response.input_tokens
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + response.output_tokens
    metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
    metrics["step_count"] = metrics.get("step_count", 0) + 1

    # Parse response
    try:
        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan = json.loads(content.strip())

        # Convert to subtask format
        # When replanning, preserve validation/quality metadata from previously
        # completed subtasks so the Critic can still see grounding data.
        old_subtasks = {s["id"]: s for s in state.get("subtasks", [])}

        subtasks = []
        for st in plan.get("subtasks", []):
            new_subtask = {
                "id": st.get("id", f"subtask_{uuid.uuid4().hex[:8]}"),
                "description": st.get("description", ""),
                "subtask_type": st.get("subtask_type", "research"),
                "status": SubtaskStatus.PENDING.value,
                "dependencies": st.get("dependencies", []),
                # Sequential Orchestration: carry declared I/O contracts from planner
                "expected_inputs": st.get("expected_inputs", []),
                "expected_outputs": st.get("expected_outputs", []),
                "tool_calls": st.get("suggested_tools", []),
                "result": None,
                "error": None,
            }
            # Carry over validation metadata from previous iteration
            old = old_subtasks.get(new_subtask["id"])
            if old and old.get("status") == SubtaskStatus.COMPLETED.value:
                for key in ("validation", "quality_warnings"):
                    if old.get(key):
                        new_subtask[key] = old[key]
            subtasks.append(new_subtask)

        # Add AI message to conversation
        ai_msg = AIMessage(content=f"Plan created with {len(subtasks)} subtasks:\n{json.dumps(plan, indent=2)}")

        return {
            "subtasks": subtasks,
            "status": "executing",
            "messages": [ai_msg],
            "metrics": metrics,
            "current_subtask_index": 0,
            # Pass through iteration_count so frontend knows if this is a replan
            "iteration_count": state.get("iteration_count", 0),
            "reuse_mode": reuse_mode,
            # Reset tried_strategies so validator can rebuild fresh ones next iteration
            "tried_strategies": [],
        }

    except json.JSONDecodeError as e:
        # Fallback: create single subtask from the request
        subtasks = [{
            "id": f"subtask_{uuid.uuid4().hex[:8]}",
            "description": state["user_request"],
            "subtask_type": "research",
            "status": SubtaskStatus.PENDING.value,
            "dependencies": [],
            "tool_calls": [],
            "result": None,
            "error": None,
        }]

        ai_msg = AIMessage(content=f"Created fallback plan with 1 subtask (parsing error: {e})")

        return {
            "subtasks": subtasks,
            "status": "executing",
            "messages": [ai_msg],
            "metrics": metrics,
            "current_subtask_index": 0,
            "iteration_count": state.get("iteration_count", 0),
            "reuse_mode": reuse_mode,
            "tried_strategies": [],
        }


async def should_replan(state: GraphState) -> str:
    """Determine if replanning is needed based on critic feedback.

    Args:
        state: Current graph state.

    Returns:
        Next node name: "planner" for replan, "end" for complete.
    """
    feedback = state.get("critic_feedback")

    if not feedback:
        return "end"

    if feedback.get("needs_revision", False):
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)

        if iteration < max_iterations:
            return "planner"

    if feedback.get("is_complete", False) and feedback.get("is_correct", True):
        return "end"

    return "end"
