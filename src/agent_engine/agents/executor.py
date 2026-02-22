"""Executor agent for tool execution with multi-turn reasoning.

Key improvements over the original single-turn executor:
1. Multi-turn tool calling loop: LLM can iteratively search → fetch → analyze → refine
2. Pre-search with fetch_url: automatically fetches full page content, not just snippets
3. Rich cross-subtask context: previous subtask results include actual content excerpts
4. Improved system prompt: guides LLM through the iterative research workflow
"""

import json
import logging
import re
import hashlib
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_engine.agents.state import GraphState, SubtaskStatus
from agent_engine.agents.extractor import extract_fields, validate_subtask_result
from agent_engine.agents.domain_profile import detect_domain_profile
from agent_engine.agents.entity_resolver import (
    expand_query_with_alias_anchor,
    get_domain_noise_patterns,
    get_domain_search_vocabulary,
    is_entity_anchored,
    resolve_entity_profile,
)
from agent_engine.config import get_settings
from agent_engine.llm import get_provider
from agent_engine.llm_logger import set_caller_context
from agent_engine.tools import MCPToolRequest, get_tool_executor, get_tool_registry
from agent_engine.tools.builtin.web_search import sanitize_text_for_llm

logger = logging.getLogger(__name__)

# Per-task query deduplication cache (reset at the start of each subtask).
# Maps normalized query string -> summary of results for that query.
_search_query_cache: dict[str, str] = {}

# Topic-level dedup: tracks core keyword sets that have been searched.
# Prevents LLM from searching the same topic with rephrased queries.
# Each entry is a frozenset of normalized keywords.
# Reset at the start of each subtask (same as _search_query_cache).
_searched_topics: list[frozenset[str]] = []


# ── Configuration constants ──
# Maximum reasoning rounds per subtask (each round = one LLM call + tool execution)
# Reduced from 10 to 5: LLM can gather sufficient data in 5 rounds (pre-search +
# 4 rounds of search→fetch→reason). More rounds waste tokens on diminishing returns.
MAX_TOOL_ROUNDS = 5
# Hard cap for LLM-initiated tool calls per subtask.
# Prevents runaway loops even when global budget is still available.
MAX_LLM_TOOL_CALLS_PER_SUBTASK = 6
# Maximum URLs to pre-fetch full content for (LLM judges relevance after fetch)
PRE_FETCH_TOP_N = 3
# Maximum fetch attempts when many get discarded by LLM (try more until we have PRE_FETCH_TOP_N)
PRE_FETCH_MAX_ATTEMPTS = 8
# Maximum content length per pre-fetched page
PRE_FETCH_MAX_CONTENT = 5000
# Minimum meaningful chars for fetched content before treating as usable evidence
MIN_FETCH_CONTENT_CHARS = 80
# Max chars of fetched evidence sent to LLM for discard decision
FETCH_DISCARD_EVIDENCE_CHARS = 2200
# Maximum context chars from each previous subtask
MAX_PREV_CONTEXT_PER_SUBTASK = 1500
# ── Token budget control ──
# Maximum chars per ToolMessage in conversation (prevents context explosion).
# Reduced from 8000 to 6000: LLMs rarely extract useful data from chars 6000-8000
# of a single tool result; the most relevant content is always near the top.
MAX_TOOL_MSG_CHARS = 6000
# Maximum total conversation chars before compaction (rough: 4 chars ≈ 1 token).
# Reduced from 100000 to 60000: triggers compaction earlier, cutting "Lost in the
# Middle" effects and reducing token costs significantly.
MAX_CONVERSATION_CHARS = 60000  # ~15k tokens — compacts when conversation grows large
# When compacting, keep last N rounds fully detailed
KEEP_RECENT_ROUNDS = 3  # Increased from 2: more recent context = fewer extra rounds
# Convergence guards: stop looping when rounds add no new evidence.
MAX_NO_GAIN_ROUNDS = 2
MAX_ALL_DISCARDED_ROUNDS = 2
# If the agent executes tools, force an explicit "observe/decide" step afterwards.
# Also forces a final no-tool "observe" pass when round/budget limits hit right after tools.
FORCE_OBSERVE_AFTER_TOOLS = True
# Trust subtask output text from LLM and skip numeric/unit/currency verification in executor.
# This disables:
# 1) P4 numeric redaction ([DATA NOT VERIFIED] replacement)
# 2) per-subtask numeric grounding validation (validate_subtask_result)
TRUST_EXECUTOR_SUBTASK_OUTPUT = True

# Cache repeated discard decisions to reduce LLM triage overhead.
# Keyed by URL + relevance query + short content fingerprint.
_discard_decision_cache: dict[str, tuple[bool, str]] = {}


def _append_unique_human_message(
    conversation: list,
    content: str,
    *,
    lookback: int = 8,
) -> bool:
    """Append a HumanMessage unless an identical recent message already exists."""
    target = (content or "").strip()
    if not target:
        return False
    recent = conversation[-lookback:] if lookback > 0 else conversation
    for msg in reversed(recent):
        if not isinstance(msg, HumanMessage):
            continue
        if str(getattr(msg, "content", "") or "").strip() == target:
            return False
    conversation.append(HumanMessage(content=target))
    return True


def _safe_debug_print(message: str) -> None:
    """Best-effort console print that never breaks execution."""
    try:
        print(message, flush=True)
    except OSError:
        # In threaded routing on Windows, stdout flush can sporadically raise
        # OSError(22). Logging already captured the message; skip console print.
        pass


def _resolve_entity_profile_for_subtask(
    subtask_description: str,
) -> dict[str, Any]:
    """Resolve entity aliases from the current subtask only.

    This avoids leaking unrelated entities from the global user request into
    subtask-local web queries.
    """
    raw = resolve_entity_profile(subtask_description or "")
    aliases = list(raw.get("aliases") or [])
    if not aliases:
        return {"canonical_name": "", "aliases": [], "timeline_note": ""}

    desc_fold = (subtask_description or "").casefold()
    anchored_aliases: list[str] = []
    for alias in aliases:
        a = str(alias or "").strip()
        if not a:
            continue
        if a.casefold() in desc_fold and a not in anchored_aliases:
            anchored_aliases.append(a)

    if not anchored_aliases:
        return {"canonical_name": "", "aliases": [], "timeline_note": ""}

    canonical = str(raw.get("canonical_name") or "").strip()
    if not canonical or canonical.casefold() not in desc_fold:
        canonical = anchored_aliases[0]
    elif canonical not in anchored_aliases:
        anchored_aliases.insert(0, canonical)

    return {
        "canonical_name": canonical,
        "aliases": anchored_aliases[:6],
        "timeline_note": str(raw.get("timeline_note") or ""),
    }


EXECUTOR_SYSTEM_PROMPT = """You are an execution agent with iterative reasoning capabilities.

Current subtask: {subtask_description}

Available tools:
{tools}

## EXECUTION STRATEGY — You MUST follow this multi-step workflow:

### Step 1: Search broadly
Use `web_search` to find relevant sources. Review the titles, URLs, and snippets.

**CRITICAL — How to write good search queries:**
- GOOD: `"ENTITY_NAME YEAR report filing"` (entity + topic + year + type)
- GOOD: `"ENTITY_NAME EVENT YEAR press release"` (entity + event + year + source type)
- BAD:  `"example.com"` (bare domain — use `fetch_url` for known URLs instead)
- BAD:  `"pdf"` or `"report"` (too generic, will return completely irrelevant results)
- BAD:  `"newswire.com"` (domain name — add entity + topic keywords)

If you already know a specific URL, use `fetch_url` directly instead of searching.

### Step 2: Fetch full content from the most promising URLs
Use `fetch_url` on the most relevant URLs from search results to get **actual page content**.
Search snippets alone are NOT sufficient — you MUST fetch full pages for key claims.

### Step 3: Use specialized data tools when available
- If the task involves structured data (filings, financial records, regulatory databases),
  check whether the available tool list includes a specialized API tool for that domain.
  Structured API tools provide more reliable data than parsing raw HTML/PDF pages.
- For general web content: use `fetch_url` on the most relevant URLs.

### Step 3b: Use `search_document` for long filings and annual reports
When you encounter a LONG document (SEC 10-K/20-F, HKEX annual report, PDF > 30 pages)
where `fetch_url` only returns the beginning and the financial data you need is not there:
→ Use `search_document(url=URL, query="specific financial metric you need")` instead.
`search_document` fetches the FULL file, splits it into chunks, and returns ONLY the
sections most relevant to your query — no truncation, no missed tables.
- Good query examples: "R&D expenses full year 2024", "operating income fiscal 2024",
  "revenue breakdown by segment 2024", "net profit attributable to shareholders"

### Step 4: Analyze, extract, and COMPUTE
After fetching content, analyze it. Extract relevant data, facts, quotes, and statistics.
When you have partial data that can be combined to derive a missing value,
use `code_execute` or `python_eval` to compute it rather than leaving it as "unknown".

### Step 5: Search deeper if needed
If initial results are insufficient, perform more targeted searches with refined queries.
Then fetch those results too. You can repeat steps 1-4 multiple times.

### Step 6: Synthesize and respond
Once you have enough verified information, provide a comprehensive summary.
Always cite source URLs for every factual claim.

## MANDATORY WORKFLOW RULES (SYSTEM-ENFORCED — you cannot skip these):

**RULE 1 — fetch_url is MANDATORY after web_search:**
After calling `web_search`, you MUST call `fetch_url` (or `search_document`) on at least
one of the returned URLs before you can write your final answer.
⚠ If you try to finish after web_search without any fetch_url call, the system will
  AUTOMATICALLY REJECT your response and demand a fetch. You cannot bypass this.

**RULE 2 — Never answer from snippets alone:**
Search snippets are 1-2 sentence previews. They NEVER contain the actual data, tables,
or figures you need. Always fetch the source page to get real numbers.

**RULE 3 — No facts from memory:**
NEVER provide facts, numbers, or statistics from your training data.
All data MUST come from tool results in this conversation.

- You may call tools MULTIPLE rounds. Keep going until you have comprehensive, verified data.
- When you have gathered enough information AND have fetched at least one URL, respond WITHOUT any tool calls to finish.
- For each fact or data point, note which URL it came from so citations are traceable.
- If a `fetch_url` fails or returns irrelevant content, try another URL or a different query.

## P4 — TOOL OUTPUT CONSTRAINED GENERATION (CRITICAL):
Every number, percentage, date, ID, or fact you write in your final answer
MUST appear VERBATIM in the tool results above. This is a hard constraint — not a guideline.

**VERBATIM means character-for-character identical to what the tool returned.**
- If the tool summary shows `$3.81 billion`, write exactly `$3.81 billion` — NOT `$3.8 billion`, NOT `3,810 million`, NOT `38.1亿`.
- If the tool shows `3,810,241,000`, write exactly `3,810,241,000` — do NOT convert to any other unit.
- Copy the number AND its surrounding unit/currency token as one inseparable string.

If a specific value is NOT present in any tool result:
  → Write "[DATA NOT VERIFIED — not found in fetched sources]" instead of the value.
  → Do NOT fill in values from your training data.
  → Do NOT estimate, approximate, or infer missing values.
  → Do NOT round, truncate, or reformat numbers pulled from tool results.

Examples of COMPLIANT output:
  ✓ "Revenue was $3.81 billion [source: SEC EDGAR summary above]"
  ✗ "Revenue was $3.8 billion"  ← WRONG — rounding not allowed
  ✗ "Revenue was $3,810 million" ← WRONG — unit conversion not allowed
  ✗ "Revenue was 38.1亿美元"    ← WRONG — currency/unit conversion not allowed

## CRITICAL: HANDLING JAVASCRIPT-RENDERED PAGES
Many data portal sites use JavaScript to render content dynamically.
When `fetch_url` returns a result with `is_js_rendered: true` or `quality_score < 0.3`,
the page content is EMPTY TEMPLATES, not real data.

**When you encounter a JS-rendered page, you MUST change strategy:**
1. Do NOT extract data from that page — it contains placeholders, NOT real data.
2. Instead, search for the same information via:
   - **News articles / press releases** that contain actual data in static text
   - **Official filings**: government regulatory filing databases, exchange announcement portals
   - **Investor relations pages**: company IR sites often have pre-rendered HTML or PDF links
   - **Reputable news agencies**: major wire services publish data in article text
3. Search: "ENTITY_NAME KEY_METRIC YEAR results" — prefer pages that embed actual numbers in text.

## CRITICAL — COPY ORIGINAL VALUES WITHOUT ANY CONVERSION
- Copy all numbers EXACTLY as they appear in tool outputs — same digits, same unit, same scale.
- NEVER perform unit conversions (USD→CNY), currency conversions, or scale changes (billion→万).
- NEVER round a number differently from how the tool rounded it.
- Always include the unit token that appeared next to the number in the source.

{domain_executor_hints}"""


async def executor_node(state: GraphState) -> dict[str, Any]:
    """Executor node that runs tools for current subtask with multi-turn reasoning.

    The executor supports iterative tool usage within a single subtask:
    LLM → tool calls → results fed back → LLM → more tool calls → ... → final response

    This allows deep research patterns like:
    web_search → fetch_url → analyze → refined web_search → fetch_url → synthesize

    Args:
        state: Current graph state.

    Returns:
        Updated state with execution results.
    """
    subtasks = state.get("subtasks", [])
    current_idx = state.get("current_subtask_index", 0)

    if current_idx >= len(subtasks):
        return {
            "status": "reviewing",
            "citations": list(state.get("citations", [])),
            "tool_call_log": list(state.get("tool_call_log", [])),
        }

    current_subtask = subtasks[current_idx]

    # Check dependencies
    for dep_id in current_subtask.get("dependencies", []):
        dep_task = next((s for s in subtasks if s["id"] == dep_id), None)
        if dep_task and dep_task.get("status") != SubtaskStatus.COMPLETED.value:
            current_subtask["status"] = SubtaskStatus.SKIPPED.value
            current_subtask["error"] = f"Dependency {dep_id} not completed"
            return {
                "subtasks": subtasks,
                "current_subtask_index": current_idx + 1,
                "citations": list(state.get("citations", [])),
                "tool_call_log": list(state.get("tool_call_log", [])),
            }

    # ── Check if this subtask was already completed in a previous iteration ──
    # When replanning, the planner may keep the same subtask IDs/descriptions for
    # work that was already done.  Instead of re-executing, reuse the old result
    # to save budget for the subtasks that actually need re-doing.
    execution_results = state.get("execution_results", [])
    prev_result = _find_reusable_result(
        current_subtask,
        execution_results,
        reuse_mode=state.get("reuse_mode", "fuzzy"),
    )
    if prev_result is not None:
        logger.info(
            f"[Executor] Reusing previous result for subtask {current_subtask['id']} "
            f"(matched from earlier iteration)"
        )
        current_subtask["status"] = SubtaskStatus.COMPLETED.value
        current_subtask["result"] = prev_result.get("response") or prev_result.get("tool_results")
        # Restore validation and quality_warnings from previous execution
        if prev_result.get("validation"):
            current_subtask["validation"] = prev_result["validation"]
        if prev_result.get("quality_issues"):
            current_subtask["quality_warnings"] = prev_result["quality_issues"]
        # Don't re-append to execution_results — the old entry is already there
        next_idx = current_idx + 1
        all_done = next_idx >= len(subtasks)
        return {
            "subtasks": subtasks,
            "current_subtask_index": next_idx,
            "execution_results": execution_results,
            "messages": [],
            "metrics": state.get("metrics", {}),
            "status": "reviewing" if all_done else "executing",
            "citations": list(state.get("citations", [])),
            "tool_call_log": list(state.get("tool_call_log", [])),
        }

    current_subtask["status"] = SubtaskStatus.IN_PROGRESS.value

    # Reset per-subtask deduplication caches
    global _search_query_cache, _searched_topics
    _search_query_cache = {}
    _searched_topics = []

    # Get LLM with executor-specific model
    settings = get_settings()
    provider = get_provider(model=settings.executor_model)
    registry = get_tool_registry()
    tool_executor = get_tool_executor()

    tools = registry.get_langchain_tools()
    all_tool_schemas = registry.list_tools()
    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in all_tool_schemas])

    # ── Initialize tracking (MUST be before pre-search) ──
    metrics = state.get("metrics", {})
    max_tokens = state.get("max_tokens", 100000)
    max_steps = state.get("max_steps", 50)
    max_tool_calls = state.get("max_tool_calls", 100)
    citations = list(state.get("citations", []))
    tool_call_log = list(state.get("tool_call_log", []))
    execution_results = state.get("execution_results", [])

    # ── Build system message (domain hints injected from profile) ──
    _domain_for_sys = detect_domain_profile(
        f"{state.get('user_request', '')}\n{current_subtask.get('description', '')}"
    )
    system_msg = SystemMessage(
        content=EXECUTOR_SYSTEM_PROMPT.format(
            subtask_description=current_subtask["description"],
            tools=tools_desc,
            domain_executor_hints=_domain_for_sys.executor_hints or "",
        )
    )

    # ── Build tiered cross-subtask context (Progressive Disclosure) ──
    context_parts = _build_cross_subtask_context(state, current_subtask=current_subtask)
    context = "\n".join(context_parts) if context_parts else ""

    # ── P6: Inject sibling evidence from global_evidence_pool ──
    # When a sibling subtask already fetched content relevant to this subtask,
    # reuse it without re-fetching (solving the "information silo" problem).
    sibling_evidence_context = _build_sibling_evidence_context(
        state, current_subtask=current_subtask
    )
    if sibling_evidence_context:
        context = sibling_evidence_context + "\n\n" + context if context else sibling_evidence_context

    # ── Pre-search with content fetch for research tasks ──
    search_context = ""
    prefetch_tool_results: list[dict[str, Any]] = []
    # Determine if the subtask needs web research (pre-search + forced tool use).
    # Default: YES — almost all subtasks benefit from real-time data retrieval.
    # Only skip for purely computational/formatting subtasks that explicitly
    # say they don't need external information.
    subtask_desc_lower = current_subtask["description"].lower()
    entity_profile = _resolve_entity_profile_for_subtask(
        current_subtask.get("description", "")
    )
    # P1: Use subtask_type from planner to decide if web search is needed.
    # "synthesis" and "computation" tasks get data from prior subtask results,
    # not from new web searches. Only "research" tasks trigger pre-search.
    subtask_type = current_subtask.get("subtask_type", "research")
    _no_search_indicators = [
        "format", "reformat", "convert", "summarize the above",
        "汇总以上", "格式化", "转换格式", "整理已有",
    ]
    _desc_no_search = any(kw in subtask_desc_lower for kw in _no_search_indicators)
    is_research_task = (subtask_type == "research") and not _desc_no_search

    if is_research_task:
        (
            search_context,
            prefetch_tool_results,
            citations,
            tool_call_log,
            metrics,
        ) = await _pre_search_with_fetch(
            current_subtask,
            provider,
            tool_executor,
            citations,
            tool_call_log,
            metrics,
            user_request=state.get("user_request", ""),
            entity_profile=entity_profile,
        )

    # ── Sequential Orchestration: inject dependency outputs for synthesis/computation ──
    # Uses expected_inputs (from planner) to extract ONLY the declared fields,
    # avoiding full LLM-analysis dumps that propagate hallucinations.
    dep_context = ""
    if subtask_type in ("synthesis", "computation"):
        dep_ids = current_subtask.get("dependencies", [])
        expected_inputs: list[dict] = current_subtask.get("expected_inputs", [])
        # Build a lookup: dep_id → requested fields (empty list = all available)
        fields_wanted: dict[str, list[str]] = {}
        for ei in expected_inputs:
            src = ei.get("from", "")
            if src:
                fields_wanted[src] = ei.get("fields", [])

        if dep_ids:
            subtasks_map = {s["id"]: s for s in state.get("subtasks", [])}
            exec_map = {r["subtask_id"]: r for r in state.get("execution_results", [])}
            dep_parts = ["## Dependency Results (use as PRIMARY data source)\n"]
            for dep_id in dep_ids:
                dep_subtask = subtasks_map.get(dep_id, {})
                dep_result = exec_map.get(dep_id, {})
                dep_desc = dep_subtask.get("description", dep_id)
                wanted = fields_wanted.get(dep_id, [])

                # Prefer raw tool evidence over LLM analysis (avoids hallucination relay)
                tool_results = dep_result.get("tool_results", [])
                evidence_parts: list[str] = []
                for tr in tool_results:
                    if not tr.get("success") or not tr.get("result"):
                        continue
                    tn = tr.get("tool_name", "")
                    r = tr["result"]
                    if tn == "fetch_url" and isinstance(r, dict):
                        url = r.get("url", "")
                        content = r.get("content", "") or r.get("excerpt", "")
                        if content:
                            safe = sanitize_text_for_llm(
                                _extract_text_excerpt(content, max_len=800)
                            )
                            evidence_parts.append(f"  [{url}]\n  {safe}")
                    elif tn in ("sec_edgar_financials", "sec_edgar_filings") and r:
                        evidence_parts.append(
                            f"  [SEC/{tn}]\n  {sanitize_text_for_llm(str(r)[:800])}"
                        )

                # LLM analysis as supplement ONLY when no tool evidence OR declared fields needed
                dep_response = dep_result.get("response", "") or dep_subtask.get("result", "")
                if wanted and dep_response:
                    # Filter to declared fields only (extract sentences mentioning those fields)
                    resp_str = str(dep_response)
                    filtered_lines = []
                    for line in resp_str.splitlines():
                        if any(f.lower() in line.lower() for f in wanted):
                            filtered_lines.append(line)
                    if filtered_lines:
                        filtered_text = sanitize_text_for_llm(
                            "\n".join(filtered_lines)[:1500]
                        )
                        evidence_parts.append(
                            f"  [LLM summary — declared fields: {', '.join(wanted)}]\n"
                            f"  ⚠ Cross-check against tool evidence above.\n"
                            f"  {filtered_text}"
                        )
                elif not evidence_parts and dep_response:
                    # No tool evidence and no field filter — fall back to full response (with warning)
                    safe_resp = sanitize_text_for_llm(str(dep_response)[:2000])
                    evidence_parts.append(
                        f"  [LLM summary ⚠ unverified — cross-check numbers against sources]\n"
                        f"  {safe_resp}"
                    )

                if evidence_parts:
                    dep_parts.append(f"### [{dep_id}] {dep_desc}")
                    dep_parts.extend(evidence_parts)
                    dep_parts.append("")

            if len(dep_parts) > 1:
                dep_context = "\n".join(dep_parts)
                logger.info(
                    f"[Executor] Sequential orchestration: injected {len(dep_ids)} dependency "
                    f"results for {subtask_type} subtask {current_subtask['id']}"
                )

    # ── Build initial user message ──
    entity_hint = ""
    if entity_profile.get("canonical_name"):
        aliases = ", ".join(entity_profile.get("aliases", [])[:5])
        entity_hint = (
            f"\n\nEntity resolution note:\n"
            f"- Canonical name: {entity_profile.get('canonical_name')}\n"
            f"- Aliases for retrieval: {aliases}\n"
            f"- {entity_profile.get('timeline_note', '')}\n"
            f"Use aliases during retrieval, but keep conclusions mapped to the canonical name."
        )

    if subtask_type in ("synthesis", "computation") and dep_context:
        # Synthesis/computation: lead with dependency data, no search context needed
        user_content = (
            f"{dep_context}\n\n"
            f"{context}\n\n"
            f"Execute this subtask using ONLY the dependency data above:\n"
            f"{current_subtask['description']}\n\n"
            f"Do NOT call web_search — all required data is provided above.\n"
            f"Respond with your synthesis/computation result directly."
            f"{entity_hint}"
        )
    elif search_context:
        user_content = (
            f"{context}\n\n{search_context}\n\n"
            f"Based on the search results and fetched content above, execute this subtask:\n"
            f"{current_subtask['description']}\n\n"
            f"If you need more information, use `web_search` and `fetch_url` to find it.\n"
            f"When you have enough verified data, respond WITHOUT tool calls to provide your final analysis."
            f"{entity_hint}"
        )
    else:
        user_content = (
            f"{context}\n\n"
            f"Execute this subtask: {current_subtask['description']}\n\n"
            f"Use `web_search` to find relevant information, then `fetch_url` to get full page content.\n"
            f"When you have enough verified data, respond WITHOUT tool calls to provide your final analysis."
            f"{entity_hint}"
        )

    user_msg = HumanMessage(content=user_content)

    # Start conversation fresh for each subtask (no orphaned tool messages)
    conversation = [system_msg, user_msg]

    # ── Multi-turn tool calling loop ──
    # CRITICAL: Disable DashScope built-in search when tools are provided.
    # If enable_search=True (default), the LLM can use DashScope's internal search
    # to answer directly WITHOUT calling our web_search/fetch_url tools, which
    # bypasses the entire citation tracking pipeline and produces unverifiable results.
    try:
        model = provider.get_langchain_model(
            tools=tools if tools else None,
            enable_search=False,
        )
    except TypeError:
        # Some providers don't support enable_search (only relevant for DashScope/Qwen).
        model = provider.get_langchain_model(tools=tools if tools else None)
    # A secondary model WITHOUT tools, used for a forced "observe/summarize" pass
    # when we hit round/budget limits right after tool execution.
    try:
        analysis_model = provider.get_langchain_model(
            tools=None,
            enable_search=False,
        )
    except TypeError:
        analysis_model = provider.get_langchain_model(tools=None)

    all_tool_results = []  # Accumulate all tool results across rounds
    if prefetch_tool_results:
        all_tool_results.extend(prefetch_tool_results)
    non_prefetch_tool_calls = 0  # tool calls initiated by the LLM inside the loop
    quality_issues = []
    final_response = None
    moderation_retries = 0  # Track content moderation retry attempts
    budget_exceeded = False
    pending_observation = False  # True if we executed tools and must "read" them next

    # ── "Search-then-Fetch" enforcement tracking ──
    # URLs discovered by LLM-initiated web_search calls (not pre-search).
    # We inject a mandatory fetch nudge if the LLM tries to finish after
    # searching without fetching any full pages.
    _llm_search_found_urls: list[str] = []   # URLs returned by web_search results
    _llm_fetch_called_urls: set[str] = set() # URLs the LLM actually fetched
    no_gain_rounds = 0
    all_discard_rounds = 0

    # Tiered budget management:
    # - Soft limit (70%): reduce max rounds for current subtask to finish faster
    # - Hard limit (90%): break current subtask's loop, but still allow next subtasks
    # - Emergency limit (100%): stop all execution, go to reviewing
    token_ratio = metrics.get("total_tokens", 0) / max(max_tokens, 1)
    step_ratio = metrics.get("step_count", 0) / max(max_steps, 1)
    tool_ratio = metrics.get("tool_call_count", 0) / max(max_tool_calls, 1)

    if token_ratio > 1.0 or step_ratio > 1.0 or tool_ratio >= 1.0:
        # Emergency: already over budget — skip this subtask entirely
        logger.warning(
            f"[Executor] Budget already exceeded before subtask {current_subtask['id']} "
            f"(tokens: {token_ratio:.0%}, steps: {step_ratio:.0%}, tools: {tool_ratio:.0%})"
        )
        budget_exceeded = True
        current_subtask["status"] = SubtaskStatus.COMPLETED.value
        current_subtask["result"] = "(Skipped — execution budget exceeded)"
        execution_results.append({
            "subtask_id": current_subtask["id"],
            "response": "(Budget exceeded, subtask skipped)",
            "summary": "Skipped due to budget constraints",
        })
    else:
        # Determine effective max rounds based on budget usage
        max_budget_ratio = max(token_ratio, step_ratio, tool_ratio)
        if max_budget_ratio > 0.7:
            # Soft limit: reduce rounds to conserve budget for remaining subtasks
            effective_max_rounds = min(MAX_TOOL_ROUNDS, 3)
            logger.info(
                f"[Executor] Budget at {max_budget_ratio:.0%} — reducing max rounds to "
                f"{effective_max_rounds} for subtask {current_subtask['id']}"
            )
        else:
            effective_max_rounds = MAX_TOOL_ROUNDS

    if not budget_exceeded:
        for round_num in range(effective_max_rounds):
            # Hard limit check: stop current subtask loop but don't skip remaining subtasks
            token_ratio = metrics.get("total_tokens", 0) / max(max_tokens, 1)
            step_ratio = metrics.get("step_count", 0) / max(max_steps, 1)
            tool_ratio = metrics.get("tool_call_count", 0) / max(max_tool_calls, 1)

            if token_ratio > 0.9 or step_ratio > 0.9 or tool_ratio >= 0.9:
                # If we just executed tools, force ONE final observe/summarize pass
                # so the agent doesn't "fetch but never read".
                if pending_observation and FORCE_OBSERVE_AFTER_TOOLS:
                    logger.warning(
                        f"[Executor] Hard budget limit (90%) reached but tools were just executed; "
                        f"forcing one final observe pass (round={round_num})"
                    )
                    _append_unique_human_message(
                        conversation,
                        "FINAL PASS (no more tools): Read the tool results above, "
                        "extract key facts, and write the final answer with citations.",
                    )
                    try:
                        with set_caller_context("executor"):
                            final_response = await analysis_model.ainvoke(conversation)
                        pending_observation = False
                    except Exception as llm_err:
                        logger.error(
                            f"[Executor] Final observe pass failed: {str(llm_err)[:200]}"
                        )
                    break

                logger.warning(
                    f"[Executor] Hard budget limit (90%) reached at round {round_num} "
                    f"(tokens: {token_ratio:.0%}, steps: {step_ratio:.0%}, "
                    f"tools: {tool_ratio:.0%})"
                )
                break

            # ── Compact conversation if too large ──
            # Prevents context window explosion from accumulated tool results.
            # Keep system + user messages intact, compress old ToolMessages to summaries.
            total_chars = sum(
                len(m.content) if hasattr(m, 'content') else 0
                for m in conversation
            )
            if total_chars > MAX_CONVERSATION_CHARS and round_num >= KEEP_RECENT_ROUNDS:
                logger.info(
                    f"[Executor] Compacting conversation: {total_chars} chars → "
                    f"keeping last {KEEP_RECENT_ROUNDS} rounds detailed"
                )
                conversation = _compact_conversation(conversation, KEEP_RECENT_ROUNDS)

            # Invoke LLM
            logger.info(
                f"[Executor] Subtask {current_subtask['id']} "
                f"round {round_num + 1}/{effective_max_rounds}"
            )
            try:
                with set_caller_context("executor"):
                    response = await model.ainvoke(conversation)
            except Exception as llm_err:
                err_str = str(llm_err)
                # Handle DashScope content moderation rejection
                is_moderation = (
                    "data_inspection_failed" in err_str
                    or "inappropriate content" in err_str.lower()
                    or ("400" in err_str and "bad request" in err_str.lower())
                )
                if is_moderation:
                    moderation_retries = moderation_retries + 1
                    logger.warning(
                        f"[Executor] Content moderation blocked subtask "
                        f"{current_subtask['id']} round {round_num} "
                        f"(attempt {moderation_retries}): {err_str[:200]}"
                    )
                    if moderation_retries <= 2:
                        # Dynamic fix: sanitize all messages in conversation,
                        # then retry.  This catches toxic content that leaked
                        # into ToolMessages from fetched pages or search snippets.
                        sanitized_conversation = []
                        for msg in conversation:
                            if hasattr(msg, "content") and isinstance(msg.content, str):
                                clean = sanitize_text_for_llm(msg.content)
                                if clean != msg.content:
                                    logger.info(
                                        f"[Executor] Sanitized {type(msg).__name__} "
                                        f"({len(msg.content)} → {len(clean)} chars)"
                                    )
                                    msg = msg.copy(update={"content": clean})
                            sanitized_conversation.append(msg)
                        conversation = sanitized_conversation

                        # If conversation is very large, aggressively compact
                        # to drop older messages that are most likely to contain
                        # the offending content.
                        total_chars = sum(
                            len(m.content) for m in conversation
                            if hasattr(m, "content") and isinstance(m.content, str)
                        )
                        if total_chars > 20000:
                            conversation = _compact_conversation(
                                conversation, keep_recent_rounds=1
                            )
                            logger.info(
                                f"[Executor] Aggressively compacted conversation "
                                f"after moderation block ({total_chars} → "
                                f"{sum(len(m.content) for m in conversation if hasattr(m, 'content') and isinstance(m.content, str))} chars)"
                            )
                        continue  # Retry with sanitized conversation
                    else:
                        # Exhausted retries — skip this subtask gracefully
                        current_subtask["status"] = SubtaskStatus.FAILED.value
                        current_subtask["error"] = (
                            "Content moderation: input rejected by API after "
                            f"{moderation_retries} sanitization attempts"
                        )
                        break
                else:
                    # Other LLM errors — fail the subtask
                    logger.error(f"[Executor] LLM error: {err_str[:300]}")
                    current_subtask["status"] = SubtaskStatus.FAILED.value
                    current_subtask["error"] = f"LLM error: {err_str[:200]}"
                    break

            # Update metrics — handle multiple token metadata formats.
            # ChatOpenAI with DashScope may report tokens under different keys:
            #   - usage_metadata: {"input_tokens": N, "output_tokens": N}  (LangChain standard)
            #   - response_metadata.token_usage: {"prompt_tokens": N, "completion_tokens": N}
            # We check both paths to ensure token counting works with DashScope.
            _inp_tok = 0
            _out_tok = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                _inp_tok = (
                    response.usage_metadata.get("input_tokens", 0)
                    or response.usage_metadata.get("prompt_tokens", 0)
                )
                _out_tok = (
                    response.usage_metadata.get("output_tokens", 0)
                    or response.usage_metadata.get("completion_tokens", 0)
                )
            if _inp_tok == 0 and _out_tok == 0:
                # Fallback: check response_metadata.token_usage (OpenAI-compat path)
                _rm = getattr(response, "response_metadata", None) or {}
                _tu = _rm.get("token_usage") or _rm.get("usage") or {}
                _inp_tok = _tu.get("prompt_tokens", 0) or _tu.get("input_tokens", 0)
                _out_tok = _tu.get("completion_tokens", 0) or _tu.get("output_tokens", 0)
            if _inp_tok or _out_tok:
                metrics["input_tokens"] = metrics.get("input_tokens", 0) + _inp_tok
                metrics["output_tokens"] = metrics.get("output_tokens", 0) + _out_tok
                metrics["total_tokens"] = (
                    metrics["input_tokens"] + metrics["output_tokens"]
                )
            metrics["step_count"] = metrics.get("step_count", 0) + 1

            # Check if LLM wants to call tools
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            if not has_tool_calls:
                # ── Force tool usage for research tasks on early rounds ──
                # Research tasks MUST use tools to find real-time data.  If the LLM
                # answers without calling tools in the first 2 rounds, inject a nudge.
                # The nudge triggers even if pre-search ran, because the LLM should
                # still call fetch_url on additional sources (not just pre-search top 3).
                if is_research_task and subtask_type == "research" and round_num <= 1 and non_prefetch_tool_calls == 0:
                    logger.warning(
                        f"[Executor] Research subtask {current_subtask['id']} "
                        f"answered without tools on round {round_num} — forcing tool usage"
                    )
                    # Add the LLM's response to conversation, then nudge
                    conversation.append(response)
                    nudge_text = (
                        "STOP — you answered from memory without calling any tools. "
                        "This is a research task that requires REAL, CURRENT data. "
                        "You MUST call `web_search` NOW to find up-to-date information, "
                        "then use `fetch_url` to get full page content from the results. "
                        "Do NOT answer without tool calls."
                    )
                    _append_unique_human_message(conversation, nudge_text)
                    continue  # Go to next round

                # ── Search-then-Fetch enforcement ──
                # If the LLM used web_search but never called fetch_url / search_document,
                # it is answering from snippets alone (search-only = unverified data).
                # Inject a mandatory fetch nudge with the specific URLs found.
                _has_unfetched = _llm_search_found_urls and not _llm_fetch_called_urls
                _still_has_rounds = round_num < effective_max_rounds - 1
                _within_tool_cap = non_prefetch_tool_calls < MAX_LLM_TOOL_CALLS_PER_SUBTASK
                if (
                    is_research_task
                    and _has_unfetched
                    and _still_has_rounds
                    and _within_tool_cap
                    and not budget_exceeded
                ):
                    # Find top un-fetched URLs (deduplicate, cap at 3)
                    seen_for_nudge: set[str] = set()
                    urls_to_fetch: list[str] = []
                    for u in _llm_search_found_urls:
                        if u not in _llm_fetch_called_urls and u not in seen_for_nudge:
                            seen_for_nudge.add(u)
                            urls_to_fetch.append(u)
                        if len(urls_to_fetch) >= 3:
                            break

                    url_list = "\n".join(f"  - {u}" for u in urls_to_fetch)
                    logger.warning(
                        "[Executor] Subtask %s: LLM finished after web_search "
                        "but never called fetch_url — enforcing fetch on %d URL(s)",
                        current_subtask["id"], len(urls_to_fetch),
                    )
                    conversation.append(response)
                    fetch_nudge = (
                        "MANDATORY FETCH REQUIRED — you searched the web but did NOT fetch "
                        "any full pages. Answering from search snippets alone is NOT allowed: "
                        "snippets are too short to contain the actual data you need.\n\n"
                        "You MUST now call `fetch_url` (or `search_document` for long PDFs) "
                        "on the most relevant URL from your search results. "
                        "Here are un-fetched URLs from your search — pick the most relevant:\n"
                        f"{url_list}\n\n"
                        "Call fetch_url on ONE of these URLs NOW. "
                        "After reading the full page content, you can write your final answer."
                    )
                    _append_unique_human_message(conversation, fetch_nudge)
                    continue  # Force another round

                # No tool calls → LLM has finished reasoning for this subtask
                final_response = response
                pending_observation = False
                logger.info(
                    f"[Executor] Subtask {current_subtask['id']} completed "
                    f"after {round_num + 1} round(s), "
                    f"{len(all_tool_results)} total tool call(s)"
                )
                break

            # ── Execute tool calls and feed results back to LLM ──
            conversation.append(response)  # Add AI message with tool_calls

            round_tool_results = []
            round_tool_message_count = 0
            round_had_executed_tool = False
            urls_before_round = len(set(_llm_search_found_urls))
            fetched_before_round = len(_llm_fetch_called_urls)

            for tc_idx, tool_call in enumerate(response.tool_calls):
                # Extract tool call info (handle both object and dict formats)
                if hasattr(tool_call, "name"):
                    tool_name = tool_call.name
                    tool_args = getattr(tool_call, "args", {}) or {}
                    tool_id = getattr(tool_call, "id", "") or ""
                else:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                if not tool_id:
                    tool_id = f"missing_call_id_{round_num}_{tc_idx}"

                if non_prefetch_tool_calls >= MAX_LLM_TOOL_CALLS_PER_SUBTASK:
                    skip_reason = (
                        f"SKIPPED: Per-subtask tool cap reached "
                        f"({MAX_LLM_TOOL_CALLS_PER_SUBTASK}). "
                        "Choose the single most relevant remaining call next round."
                    )
                    logger.warning(
                        f"[Executor] Per-subtask tool cap reached "
                        f"({MAX_LLM_TOOL_CALLS_PER_SUBTASK}) for {current_subtask['id']}"
                    )
                    conversation.append(ToolMessage(content=skip_reason, tool_call_id=tool_id))
                    round_tool_message_count += 1
                    round_tool_results.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": None,
                        "error": skip_reason,
                        "success": False,
                        "skipped": True,
                    })
                    continue
                if metrics.get("tool_call_count", 0) >= max_tool_calls:
                    skip_reason = (
                        "SKIPPED: Global tool_call budget reached. "
                        "No further tool executions are allowed in this task."
                    )
                    logger.warning("[Executor] Tool call limit hit during round")
                    conversation.append(ToolMessage(content=skip_reason, tool_call_id=tool_id))
                    round_tool_message_count += 1
                    round_tool_results.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": None,
                        "error": skip_reason,
                        "success": False,
                        "skipped": True,
                    })
                    continue

                # ── Tool Intent Guard: validate & normalize web_search queries ──
                if tool_name == "web_search" and isinstance(tool_args, dict):
                    q = str(tool_args.get("query", "") or "")
                    q_norm = _normalize_search_query(q)
                    q_norm = expand_query_with_alias_anchor(
                        q_norm or q,
                        entity_profile.get("aliases", []),
                        max_len=50,
                    )

                    rejection = _validate_search_query(
                        q_norm or q,
                        required_aliases=entity_profile.get("aliases", []),
                    )
                    if rejection:
                        # Return an error to the LLM so it reformulates
                        logger.info(
                            f"[Executor] Rejected web_search query: "
                            f"{q[:60]!r} -> {rejection}"
                        )
                        error_msg = (
                            f"QUERY REJECTED: {rejection}\n"
                            f"Your query was: \"{q[:80]}\"\n"
                            f"Please provide a proper search query with "
                            f"specific keywords, e.g. "
                            f"\"ENTITY_NAME YEAR report type\" "
                            f"instead of bare domains or single words. "
                            f"To navigate to a known URL directly, use "
                            f"fetch_url instead of web_search."
                        )
                        conversation.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_id or f"reject_{round_num}",
                            )
                        )
                        round_tool_message_count += 1
                        metrics["tool_call_count"] = metrics.get("tool_call_count", 0) + 1
                        tool_call_log.append({
                            "tool_name": "web_search",
                            "args": {"query": q},
                            "result_summary": f"Rejected: {rejection}",
                            "success": False,
                            "timestamp": datetime.now().isoformat(),
                            "subtask_id": current_subtask["id"],
                        })
                        continue  # skip execution, let LLM retry

                    # Check per-subtask dedup cache (exact match)
                    if q_norm in _search_query_cache:
                        cached = _search_query_cache[q_norm]
                        logger.info(
                            f"[Executor] Dedup: query {q_norm[:40]!r} "
                            f"already executed, returning cached result"
                        )
                        conversation.append(
                            ToolMessage(
                                content=(
                                    f"This exact query was already searched. "
                                    f"Previous results:\n{cached}\n\n"
                                    f"Try a DIFFERENT query with different "
                                    f"keywords if you need more information."
                                ),
                                tool_call_id=tool_id or f"dedup_{round_num}",
                            )
                        )
                        round_tool_message_count += 1
                        continue

                    # Check topic-level dedup (fuzzy match)
                    topic_dup = _is_topic_searched(q_norm or q)
                    if topic_dup:
                        logger.info(
                            f"[Executor] Topic dedup: {(q_norm or q)[:40]!r} — {topic_dup}"
                        )
                        conversation.append(
                            ToolMessage(
                                content=f"QUERY REJECTED: {topic_dup}",
                                tool_call_id=tool_id or f"topic_dedup_{round_num}",
                            )
                        )
                        round_tool_message_count += 1
                        continue

                    if q_norm and q_norm != q:
                        tool_args = dict(tool_args)
                        tool_args["query"] = q_norm

                # ── Evidence Guard: add focus_terms for fetch_url so we can cite exact passages ──
                if tool_name == "fetch_url" and isinstance(tool_args, dict):
                    if not tool_args.get("focus_terms"):
                        focus_terms = _derive_focus_terms(
                            current_subtask.get("description", ""),
                            user_request=state.get("user_request", ""),
                        )
                        if focus_terms:
                            tool_args = dict(tool_args)
                            tool_args["focus_terms"] = focus_terms

                    # ── PDF auto-routing: redirect to search_document ──
                    # fetch_url truncates PDFs to the first N chars (usually the cover
                    # page), missing financial tables deep in the document.
                    # search_document fetches the FULL PDF, chunks it, and retrieves
                    # only the sections relevant to the query — much better for filings.
                    _pdf_url = str(tool_args.get("url", ""))
                    _is_pdf_url = _pdf_url.lower().split("?")[0].endswith((".pdf",))
                    _search_doc_available = any(
                        getattr(t, "name", None) == "search_document" for t in tools
                    )
                    if _is_pdf_url and _search_doc_available:
                        _query = " ".join(
                            str(t) for t in (
                                tool_args.get("focus_terms") or []
                            )
                        ) or current_subtask.get("description", "")[:120]
                        tool_name = "search_document"
                        tool_args = {"url": _pdf_url, "query": _query, "top_k": 5}
                        logger.info(
                            "[Executor] PDF URL detected — auto-routing %s → search_document "
                            "(query: %s)", _pdf_url[:60], _query[:60],
                        )

                # Execute tool
                request = MCPToolRequest(
                    tool_name=tool_name,
                    arguments=tool_args,
                    request_id=tool_id,
                )

                result = await tool_executor.execute(request)
                metrics["tool_call_count"] = (
                    metrics.get("tool_call_count", 0) + 1
                )
                non_prefetch_tool_calls += 1
                round_had_executed_tool = True

                tool_result_entry = {
                    "tool_name": tool_name,
                    "args": tool_args,
                    "result": result.result if result.is_success else None,
                    "error": result.error,
                    "success": result.is_success,
                }
                discard_fetch_result = False
                discard_fetch_reason = ""
                if (
                    tool_name == "fetch_url"
                    and result.is_success
                    and isinstance(result.result, dict)
                ):
                    discard_fetch_result, discard_fetch_reason = await _llm_should_discard_fetch_result(
                        provider=provider,
                        subtask_description=current_subtask.get("description", ""),
                        relevance_query=current_subtask.get("description", "") or state.get("user_request", ""),
                        fetch_payload=result.result,
                        metrics=metrics,
                    )
                    if discard_fetch_result:
                        tool_result_entry["discarded"] = True
                        tool_result_entry["discard_reason"] = discard_fetch_reason
                        logger.info(
                            "[Executor] Discarded fetch_url result for subtask %s: %s (%s)",
                            current_subtask["id"],
                            str(result.result.get("url", ""))[:120],
                            discard_fetch_reason,
                        )
                round_tool_results.append(tool_result_entry)
                all_tool_results.append(tool_result_entry)

                # Record to detailed tool call log
                result_summary = ""
                if result.is_success and result.result:
                    result_summary = str(result.result)[:300]
                elif result.error:
                    result_summary = f"Error: {result.error}"
                if tool_name == "fetch_url" and discard_fetch_result:
                    result_summary = f"Discarded fetch_url: {discard_fetch_reason}"
                tool_call_log.append({
                    "tool_name": tool_name,
                    "args": tool_args,
                    "result_summary": result_summary,
                    "success": result.is_success,
                    "timestamp": datetime.now().isoformat(),
                    "subtask_id": current_subtask["id"],
                    "execution_time_ms": result.execution_time_ms,
                })

                # Track citations
                citations = _update_citations(
                    citations,
                    tool_name,
                    result,
                    subtask_id=current_subtask["id"],
                    skip_fetch_url=(tool_name == "fetch_url" and discard_fetch_result),
                )

                # Populate dedup cache for web_search results
                if (
                    tool_name == "web_search"
                    and result.is_success
                    and isinstance(result.result, dict)
                ):
                    cache_key = _normalize_search_query(
                        str(tool_args.get("query", ""))
                    )
                    if cache_key:
                        sr_list = result.result.get("results", [])
                        summary_parts = []
                        for sr in sr_list[:5]:
                            t = sanitize_text_for_llm(sr.get("title", ""))
                            u = sr.get("url", "")
                            s = sanitize_text_for_llm(sr.get("snippet", "")[:100])
                            summary_parts.append(f"- {t} ({u}): {s}")
                        _search_query_cache[cache_key] = "\n".join(
                            summary_parts
                        ) or "(no results)"
                        # Record topic for fuzzy dedup
                        _searched_topics.append(_extract_topic_keys(cache_key))

                    # Track URLs found by LLM-initiated web_search for fetch enforcement
                    for sr in result.result.get("results", [])[:5]:
                        u = sr.get("url", "")
                        if u and u not in _llm_fetch_called_urls:
                            _llm_search_found_urls.append(u)

                # Track URLs fetched by LLM for fetch enforcement
                if tool_name in ("fetch_url", "search_document") and result.is_success:
                    fetched_url = ""
                    if isinstance(result.result, dict):
                        fetched_url = result.result.get("url", "")
                    if fetched_url and not (tool_name == "fetch_url" and discard_fetch_result):
                        _llm_fetch_called_urls.add(fetched_url)

                # Build tool result message for conversation
                # CRITICAL: Truncate content to prevent conversation context explosion.
                # A single fetch_url can return 50k+ chars of HTML, and with 5+ tool
                # calls per round across 5+ rounds, the conversation can exceed 500k chars.
                raw_content = result.to_message_content()

                # For fetch_url, extract just the text content from the result dict
                if (
                    tool_name == "fetch_url"
                    and result.is_success
                    and isinstance(result.result, dict)
                ):
                    # Extract the most useful fields, not the full HTML
                    url = result.result.get("url", "")
                    quality_score = result.result.get("quality_score", 1.0)
                    is_js = result.result.get("is_js_rendered", False)
                    is_citable = result.result.get("is_citable", True)
                    not_citable_reason = result.result.get("not_citable_reason")
                    # fetch_url guarantees "content" is a readable excerpt now.
                    clean_text = (result.result.get("content") or "").strip()
                    evidence_snippets = result.result.get("evidence_snippets") or []
                    # Truncate to limit
                    if len(clean_text) > MAX_TOOL_MSG_CHARS:
                        clean_text = clean_text[:MAX_TOOL_MSG_CHARS] + f"\n... [truncated, {len(clean_text)} total chars]"
                    if discard_fetch_result:
                        clean_text = "(discarded - irrelevant or empty page)"
                    evidence_block = ""
                    if (
                        not discard_fetch_result
                        and isinstance(evidence_snippets, list)
                        and evidence_snippets
                    ):
                        lines = []
                        for item in evidence_snippets[:8]:
                            if not isinstance(item, dict):
                                continue
                            term = str(item.get("term") or "").strip()
                            snip = str(item.get("snippet") or "").strip()
                            if not snip:
                                continue
                            if term:
                                lines.append(f"- [{term}] {snip}")
                            else:
                                lines.append(f"- {snip}")
                        if lines:
                            evidence_block = "\n\nEvidence snippets:\n" + "\n".join(lines)

                    # ── Extractor: precise field extraction for large citable documents ──
                    extractor_block = ""
                    extracted_text_full = (result.result.get("extracted_text") or "")
                    if (not discard_fetch_result) and is_citable and len(extracted_text_full) > 5000:
                        focus = tool_args.get("focus_terms") or []
                        if focus:
                            try:
                                extractions = await extract_fields(
                                    text=extracted_text_full,
                                    fields=focus,
                                    source_url=url,
                                    subtask_description=current_subtask.get("description", ""),
                                )
                                if extractions:
                                    ext_lines = []
                                    for ext in extractions:
                                        v = ext.get("value")
                                        ctx = ext.get("context", "")
                                        f_name = ext.get("field", "")
                                        conf = ext.get("confidence", 0)
                                        if v is not None:
                                            ext_lines.append(
                                                f"- {f_name}: {v} (confidence={conf:.1f})\n  Context: \"{ctx[:300]}\""
                                            )
                                        else:
                                            ext_lines.append(f"- {f_name}: NOT FOUND in document")
                                    if ext_lines:
                                        extractor_block = (
                                            "\n\nExtracted fields (verbatim from document):\n"
                                            + "\n".join(ext_lines)
                                        )
                                    # Store extractions for validation later
                                    tool_result_entry["extractions"] = extractions
                            except Exception as ext_err:
                                logger.warning(f"[Executor] Extractor failed: {ext_err}")

                    # ── Truncation detection: if extracted_text >> content,
                    # the document was severely truncated.  Recommend search_document.
                    extracted_text_len = result.result.get("extracted_text_length", 0)
                    content_len = len(clean_text)
                    _truncation_note = ""
                    _search_doc_avail = any(
                        getattr(t, "name", None) == "search_document" for t in tools
                    )
                    if (
                        not discard_fetch_result
                        and _search_doc_avail
                        and extracted_text_len > content_len * 2
                        and extracted_text_len > 8000
                        and not is_js
                    ):
                        _truncation_note = (
                            f"\n\n⚠ DOCUMENT TRUNCATED: This page has {extracted_text_len:,} chars "
                            f"of content but only {content_len:,} chars were shown above. "
                            f"Critical financial data is likely in the hidden sections.\n"
                            f"RECOMMENDED ACTION: Use `search_document(url=\"{url}\", "
                            f"query=\"<specific metric>\")` to semantically retrieve the "
                            f"relevant sections from the full document."
                        )

                    msg_content = (
                        f"URL: {url}\n"
                        f"Quality: {quality_score}/1.0 {'(JS-rendered)' if is_js else ''}\n"
                        f"Citable: {is_citable}\n"
                        f"Not citable reason: {not_citable_reason or ''}\n"
                        f"Content:\n{clean_text}"
                        f"{evidence_block}"
                        f"{extractor_block}"
                        f"{_truncation_note}"
                    )
                    if discard_fetch_result:
                        msg_content += (
                            "\n\nDISCARDED PAGE:\n"
                            f"Reason: {discard_fetch_reason}\n"
                            "Do not use this page for analysis/citations. "
                            "Fetch a different URL."
                        )
                        quality_issues.append(
                            f"fetch_url discarded: {url} ({discard_fetch_reason})"
                        )
                else:
                    # For other tools, just truncate
                    if len(raw_content) > MAX_TOOL_MSG_CHARS:
                        msg_content = raw_content[:MAX_TOOL_MSG_CHARS] + f"\n... [truncated, {len(raw_content)} total chars]"
                    else:
                        msg_content = raw_content

                # ── Inject quality warnings for JS-rendered pages ──
                if (
                    result.is_success
                    and tool_name == "fetch_url"
                    and isinstance(result.result, dict)
                ):
                    is_js = result.result.get("is_js_rendered", False)
                    q_score = result.result.get("quality_score", 1.0)
                    q_issues = result.result.get("quality_issues", [])
                    warning = result.result.get("warning", "")

                    if is_js or q_score < 0.3:
                        quality_warning = (
                            f"\n\n⚠ CONTENT QUALITY WARNING ⚠\n"
                            f"Quality score: {q_score}/1.0\n"
                            f"Issues: {'; '.join(q_issues)}\n"
                            f"{warning}\n"
                            f"DO NOT extract data from this page. "
                            f"Search for news articles, official filings, "
                            f"or press releases instead."
                        )
                        msg_content += quality_warning
                        quality_issues.append(
                            f"fetch_url JS-rendered: "
                            f"{result.result.get('url', '')}"
                        )

                # Sanitize tool message content before adding to conversation
                # to prevent content moderation triggers in subsequent LLM calls
                msg_content = sanitize_text_for_llm(msg_content)

                tool_msg = ToolMessage(
                    content=msg_content,
                    tool_call_id=tool_id,
                )
                conversation.append(tool_msg)
                round_tool_message_count += 1

            # Encourage recursive "observe → decide" behavior, instead of batching tools linearly.
            if FORCE_OBSERVE_AFTER_TOOLS and round_tool_message_count > 0:
                _append_unique_human_message(
                    conversation,
                    "Read the tool results above. "
                    "If more information is needed, call the next tool. "
                    "Otherwise, answer now with citations.",
                )
            pending_observation = round_tool_message_count > 0

            # Validate quality of this round's results
            quality_issues.extend(_validate_tool_quality(round_tool_results))

            # ── JS-rendered page fallback nudge ──
            # If ALL fetch_url calls in this round returned JS-rendered / low-quality
            # content, the LLM is stuck in a dead-end strategy.  Inject an immediate
            # nudge to pivot to news articles or static page alternatives BEFORE the
            # LLM tries to read the useless templates.
            _round_fetches = [
                t for t in round_tool_results
                if t.get("tool_name") == "fetch_url" and t.get("success")
            ]
            _js_rendered_fetches = [
                t for t in _round_fetches
                if isinstance(t.get("result"), dict)
                and (
                    t["result"].get("is_js_rendered")
                    or t["result"].get("quality_score", 1.0) < 0.2
                    and len(t["result"].get("content", "")) < 500
                )
            ]
            _discarded_fetches = [t for t in _round_fetches if t.get("discarded")]
            _new_url_candidates = len(set(_llm_search_found_urls)) - urls_before_round
            _new_fetched_sources = len(_llm_fetch_called_urls) - fetched_before_round
            _new_primary_evidence = any(
                t.get("success")
                and not t.get("discarded")
                and t.get("tool_name") in (
                    "fetch_url",
                    "search_document",
                    "sec_edgar_financials",
                    "sec_edgar_filings",
                )
                for t in round_tool_results
            )
            _round_gain = (
                _new_fetched_sources > 0
                or _new_primary_evidence
                or (_new_url_candidates > 0 and _new_fetched_sources > 0)
            )
            if round_had_executed_tool:
                if _round_gain:
                    no_gain_rounds = 0
                else:
                    no_gain_rounds += 1
            elif round_tool_message_count > 0:
                # Tool calls were requested but none executed (typically cap reached).
                no_gain_rounds += 1

            if _round_fetches and len(_js_rendered_fetches) == len(_round_fetches):
                _js_urls = [
                    t.get("result", {}).get("url", "?")[:70]
                    for t in _js_rendered_fetches
                ]
                logger.warning(
                    "[Executor] All %d fetch_url(s) in round %d returned JS-rendered content "
                    "— injecting alternative strategy nudge for subtask %s",
                    len(_js_rendered_fetches), round_num, current_subtask["id"],
                )
                js_nudge = (
                    "⚠ JS-RENDERED PAGE(S) DETECTED — the pages you fetched contain "
                    "JavaScript templates with NO real data:\n"
                    + "\n".join(f"  - {u}" for u in _js_urls)
                    + "\n\nThese sites require a browser to render content and cannot be "
                    "scraped directly. DO NOT try fetching the same domain again.\n\n"
                    "MANDATORY PIVOT — choose one of these alternative strategies:\n"
                    "1. Search for PRESS RELEASES: query \"{company} Q4 2024 results press release\"\n"
                    "2. Search for NEWS ARTICLES: query \"{company} annual revenue 2024 site:reuters.com OR site:bloomberg.com\"\n"
                    "3. Use SEC EDGAR API: call sec_edgar_financials with the company name\n"
                    "4. Search for PDF filings: query \"{company} 10-K 2024 annual report filetype:pdf\"\n\n"
                    "Pick ONE strategy and execute it NOW."
                )
                _append_unique_human_message(conversation, js_nudge)
            elif _round_fetches and len(_discarded_fetches) == len(_round_fetches):
                all_discard_rounds += 1
                _discarded_lines = []
                for t in _discarded_fetches[:4]:
                    _u = str(t.get("result", {}).get("url", "?"))[:100]
                    _r = str(t.get("discard_reason", "irrelevant or empty"))
                    _discarded_lines.append(f"  - {_u} ({_r})")
                discard_nudge = (
                    "ALL FETCHED PAGES WERE DISCARDED (irrelevant or empty):\n"
                    + "\n".join(_discarded_lines)
                    + "\n\nMANDATORY ACTION:\n"
                    "1. Drop these URLs.\n"
                    "2. Pick different URLs from web_search results and fetch again.\n"
                    "3. Do not write final analysis until at least one non-discarded source is fetched."
                )
                _append_unique_human_message(conversation, discard_nudge)
            elif _round_fetches:
                all_discard_rounds = 0
            else:
                all_discard_rounds = 0

            if round_had_executed_tool and no_gain_rounds >= MAX_NO_GAIN_ROUNDS:
                logger.warning(
                    "[Executor] Converging subtask %s after %d no-gain rounds",
                    current_subtask["id"],
                    no_gain_rounds,
                )
                _append_unique_human_message(
                    conversation,
                    "CONVERGENCE TRIGGERED: Recent rounds added no new usable evidence. "
                    "Do not call more tools. Summarize the best-supported conclusions, "
                    "state remaining evidence gaps, and cite fetched sources only.",
                )
                try:
                    with set_caller_context("executor"):
                        final_response = await analysis_model.ainvoke(conversation)
                    pending_observation = False
                except Exception as llm_err:
                    logger.error(
                        f"[Executor] Convergence final pass failed: {str(llm_err)[:200]}"
                    )
                break

            if all_discard_rounds >= MAX_ALL_DISCARDED_ROUNDS:
                logger.warning(
                    "[Executor] Converging subtask %s after %d all-discard fetch rounds",
                    current_subtask["id"],
                    all_discard_rounds,
                )
                _append_unique_human_message(
                    conversation,
                    "CONVERGENCE TRIGGERED: Multiple fetch rounds were discarded as irrelevant/empty. "
                    "Stop fetching. Provide a concise report of verified findings so far and "
                    "explicitly list unresolved data gaps.",
                )
                try:
                    with set_caller_context("executor"):
                        final_response = await analysis_model.ainvoke(conversation)
                    pending_observation = False
                except Exception as llm_err:
                    logger.error(
                        f"[Executor] Discard-convergence final pass failed: {str(llm_err)[:200]}"
                    )
                break

        else:
            # for-else: Exhausted all rounds without LLM finishing (no break)
            logger.warning(
                f"[Executor] Subtask {current_subtask['id']} "
                f"hit max rounds ({effective_max_rounds}) without finishing"
            )
            # If we ended right after tool execution, force one final no-tool observe pass
            # so the agent doesn't "fetch but never read".
            if pending_observation and FORCE_OBSERVE_AFTER_TOOLS:
                _append_unique_human_message(
                    conversation,
                    "FINAL PASS (no more tools): Using the tool results above, "
                    "extract insights and provide the final response with citations.",
                )
                try:
                    with set_caller_context("executor"):
                        final_response = await analysis_model.ainvoke(conversation)
                    pending_observation = False
                except Exception as llm_err:
                    logger.error(
                        f"[Executor] Final observe pass failed: {str(llm_err)[:200]}"
                    )
                    final_response = response
            else:
                # Use the last response as final
                final_response = response

    # ── Finalize subtask ──
    if final_response is not None:
        response_content = (
            final_response.content
            if hasattr(final_response, "content")
            else str(final_response)
        )
    else:
        response_content = ""

    # ── Fallback: if analysis is empty but tools produced data, synthesize a summary ──
    # This prevents "false completions" where subtasks are marked done with empty analysis.
    if not response_content.strip() and all_tool_results:
        successful_results = [
            r for r in all_tool_results
            if r.get("success") and not r.get("discarded")
        ]
        if successful_results:
            summary_parts = []
            for tr in successful_results:
                tn = tr.get("tool_name", "")
                r = tr.get("result", "")
                if tn == "fetch_url" and isinstance(r, dict):
                    url = r.get("url", "")
                    content = r.get("content", "") or r.get("excerpt", "")
                    if content and not _is_low_quality_content(url, content):
                        summary_parts.append(
                            f"[{url}]: {content[:1500]}"
                        )
                elif tn == "web_search" and isinstance(r, dict):
                    for sr in r.get("results", [])[:3]:
                        title = sr.get("title", "")
                        snippet = sr.get("snippet", "")
                        url = sr.get("url", "")
                        summary_parts.append(f"- {title} ({url}): {snippet}")
            if summary_parts:
                response_content = (
                    f"Tool execution produced the following data "
                    f"(auto-summarized from {len(successful_results)} tool results):\n\n"
                    + "\n\n".join(summary_parts)
                )
                logger.warning(
                    f"[Executor] Subtask {current_subtask['id']}: LLM analysis was empty. "
                    f"Auto-synthesized summary from {len(successful_results)} tool results "
                    f"({len(response_content)} chars)"
                )

    # ── P4: Tool Output Constrained Generation ──
    # When TRUST_EXECUTOR_SUBTASK_OUTPUT is enabled, skip numeric replacement and
    # trust the LLM's final subtask text as-is.
    if not TRUST_EXECUTOR_SUBTASK_OUTPUT:
        # After the LLM response is finalized, replace ungrounded numerical claims.
        # Only apply when tool results exist (evidence available to verify against).
        # For synthesis/computation tasks with no tools, skip (handled by P5 context injection).
        # SKIP P4 when the ONLY evidence sources are sec_edgar_* API results — these are
        # structured trusted APIs whose XBRL integers don't match human-readable text
        # (e.g., 3783241000 vs "37.83亿元"), causing massive false-positive replacements.
        _has_sec_only = all_tool_results and all(
            tr.get("tool_name", "").startswith("sec_edgar")
            for tr in all_tool_results
            if tr.get("success")
        )
        _has_fetch_url = any(
            tr.get("tool_name") in ("fetch_url", "search_document")
            for tr in all_tool_results
            if tr.get("success")
        )
        _skip_p4 = _has_sec_only and not _has_fetch_url
        if all_tool_results and response_content.strip() and not _skip_p4:
            constrained_content, p4_replacements = _constrain_output_to_tool_evidence(
                response_text=response_content,
                tool_results=all_tool_results,
            )
            if p4_replacements > 0:
                response_content = constrained_content
                quality_issues.append(
                    f"P4: {p4_replacements} ungrounded numerical claims replaced with [DATA NOT VERIFIED]"
                )
    else:
        logger.info(
            "[Executor] TRUST_EXECUTOR_SUBTASK_OUTPUT enabled — skipping P4 numeric redaction"
        )

    if all_tool_results:
        # Had tool calls — store both analysis and tool results
        execution_results.append({
            "subtask_id": current_subtask["id"],
            "subtask_description": current_subtask.get("description", ""),
            "subtask_fingerprint": _subtask_fingerprint(current_subtask.get("description", "")),
            "tool_results": all_tool_results,
            "summary": (
                f"Executed {len(all_tool_results)} tool call(s) across multiple rounds. "
                f"Final analysis: {response_content[:500]}"
            ),
            "response": response_content,
            "quality_issues": quality_issues if quality_issues else None,
        })

        successful = [r for r in all_tool_results if r["success"]]
        failed = [r for r in all_tool_results if not r["success"]]

        if successful:
            # At least some tools succeeded → mark as completed
            current_subtask["status"] = SubtaskStatus.COMPLETED.value
            current_subtask["result"] = {
                "analysis": response_content,
                "tool_results": all_tool_results,
            }
            if quality_issues:
                current_subtask["quality_warnings"] = quality_issues
            if failed:
                current_subtask["quality_warnings"] = current_subtask.get(
                    "quality_warnings", []
                ) + [f"Some tools failed: {[r['error'] for r in failed]}"]
        else:
            current_subtask["status"] = SubtaskStatus.FAILED.value
            current_subtask["error"] = (
                f"All tool calls failed: {[r['error'] for r in failed]}"
            )

    else:
        # No tool calls at all — LLM responded directly
        current_subtask["status"] = SubtaskStatus.COMPLETED.value
        current_subtask["result"] = response_content

        # Extract URLs from LLM response
        urls_in_content = _extract_urls_from_text(response_content)
        for url in urls_in_content:
            if not any(c.get("url") == url for c in citations):
                snippet = _extract_context_around_url(response_content, url, window=200)
                citations.append({
                    "id": len(citations) + 1,
                    "title": "",
                    "url": url,
                    "snippet": snippet,
                    "source_tool": "llm_builtin_search",
                    "accessed_at": datetime.now().isoformat(),
                    "verified": False,
                    "fetched_content": "",
                    "subtask_id": current_subtask["id"],
                })

        tool_call_log.append({
            "tool_name": "llm_response",
            "args": {"subtask": current_subtask["description"]},
            "result_summary": response_content[:300] if response_content else "",
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "subtask_id": current_subtask["id"],
            "execution_time_ms": 0,
        })

        execution_results.append({
            "subtask_id": current_subtask["id"],
            "subtask_description": current_subtask.get("description", ""),
            "subtask_fingerprint": _subtask_fingerprint(current_subtask.get("description", "")),
            "response": response_content,
            "summary": f"Completed via LLM (found {len(urls_in_content)} URLs)",
        })

    # ── Per-subtask validation: verify claims are grounded in evidence ──
    # In trusted mode, skip numeric validation in executor and mark the subtask
    # as trusted for downstream nodes.
    if (
        current_subtask.get("status") == SubtaskStatus.COMPLETED.value
        and response_content
        and not budget_exceeded
    ):
        if TRUST_EXECUTOR_SUBTASK_OUTPUT:
            trusted_validation = {
                "verified": True,
                "grounded_claims": 0,
                "ungrounded_claims": 0,
                "details": [],
                "grounding_ratio": 1.0,
                "strict_grounding_ratio": 1.0,
                "no_tool_evidence": False,
                "validation_mode": "trusted_executor_output",
            }
            current_subtask["validation"] = trusted_validation
            for er in execution_results:
                if er.get("subtask_id") == current_subtask["id"]:
                    er["validation"] = trusted_validation
                    break
            logger.info(
                "[Executor] TRUST_EXECUTOR_SUBTASK_OUTPUT enabled — skipping "
                "per-subtask numeric validation"
            )
        else:
            # Collect evidence from this subtask's tool results
            _evidence_snippets = []
            _fetched_contents: dict[str, str] = {}
            for tr in all_tool_results:
                if tr.get("discarded"):
                    continue
                if tr.get("tool_name") == "fetch_url" and tr.get("success") and isinstance(tr.get("result"), dict):
                    r = tr["result"]
                    url = r.get("url", "")
                    excerpt = (r.get("excerpt") or r.get("content") or "")[:3000]
                    if excerpt:
                        _fetched_contents[url] = excerpt
                    for snip in (r.get("evidence_snippets") or []):
                        if isinstance(snip, dict):
                            _evidence_snippets.append(snip)
                    # Include extractor results too
                    for ext in tr.get("extractions", []):
                        if isinstance(ext, dict) and ext.get("value") is not None:
                            _evidence_snippets.append({
                                "term": ext.get("field", ""),
                                "snippet": f"{ext.get('value')} — {ext.get('context', '')}",
                            })

            try:
                validation = await validate_subtask_result(
                    subtask_result=response_content,
                    evidence_snippets=_evidence_snippets,
                    fetched_contents=_fetched_contents,
                )
                _g = validation.get("grounded_claims", 0)
                _u = validation.get("ungrounded_claims", 0)
                _verified = validation.get("verified", True)

                # If the subtask had NO tool calls (pure LLM response) and contains
                # numerical claims, those claims are entirely from LLM memory —
                # treat them as HIGH RISK.
                if not all_tool_results and (_g + _u) > 0:
                    logger.warning(
                        f"[Executor] Subtask {current_subtask['id']} answered from LLM memory "
                        f"with {_g + _u} numerical claims — ALL marked as ungrounded "
                        f"(no tool evidence to verify against)"
                    )
                    # Override: mark ALL claims as ungrounded since there's no source
                    for det in validation.get("details", []):
                        det["grounded"] = False
                        det["found_in"] = None
                    validation["grounded_claims"] = 0
                    validation["ungrounded_claims"] = _g + _u
                    validation["verified"] = False
                    validation["no_tool_evidence"] = True
                    _g, _u, _verified = 0, _g + _u, False

                logger.info(
                    f"[Executor] Subtask {current_subtask['id']} validation: "
                    f"grounded={_g}, ungrounded={_u}, verified={_verified}"
                )

                # Store validation result
                current_subtask["validation"] = validation

                # Sync validation to execution_results so it survives replan.
                # When the Planner creates new subtask dicts on replan, validation
                # stored only on the subtask dict is lost.  By also storing it in
                # execution_results, the Executor can restore it when reusing results.
                for er in execution_results:
                    if er.get("subtask_id") == current_subtask["id"]:
                        er["validation"] = validation
                        break

                if not _verified and _u > 0:
                    current_subtask.setdefault("quality_warnings", []).append(
                        f"Validation: {_u} numerical claims not found in fetched evidence"
                    )
                    # Add ungrounded details to quality_issues for critic visibility
                    for det in validation.get("details", []):
                        if not det.get("grounded"):
                            quality_issues.append(
                                f"Ungrounded claim: '{det.get('claim', '')}' not found in any fetched content"
                            )
            except Exception as val_err:
                logger.warning(f"[Executor] Subtask validation failed: {val_err}")

    # Move to next subtask
    next_idx = current_idx + 1
    all_done = next_idx >= len(subtasks)

    # Determine next status:
    # - Only go to "reviewing" when ALL subtasks are done OR emergency budget exceeded
    # - budget_exceeded from the emergency check (100%+) means we should stop
    # - budget_exceeded from the tiered system just means this subtask was shortened
    token_ratio_final = metrics.get("total_tokens", 0) / max(max_tokens, 1)
    step_ratio_final = metrics.get("step_count", 0) / max(max_steps, 1)
    tool_ratio_final = metrics.get("tool_call_count", 0) / max(max_tool_calls, 1)
    emergency_exceeded = (
        token_ratio_final > 1.0 or step_ratio_final > 1.0 or tool_ratio_final >= 1.0
    )

    if emergency_exceeded:
        logger.warning(
            f"[Executor] Emergency budget exceeded — going to review "
            f"(tokens: {token_ratio_final:.0%}, steps: {step_ratio_final:.0%}, "
            f"tools: {tool_ratio_final:.0%})"
        )
        new_status = "reviewing"
    elif all_done:
        new_status = "reviewing"
    else:
        new_status = "executing"

    _return_msg = (
        f"[Executor] RETURN: new_status={new_status!r}, next_idx={next_idx}, "
        f"all_done={all_done}, emergency_exceeded={emergency_exceeded}, "
        f"budget_exceeded={budget_exceeded}, "
        f"subtask={current_subtask['id']}, "
        f"subtask_status={current_subtask.get('status', '?')}, "
        f"tool_results_count={len(all_tool_results)}, "
        f"tokens={metrics.get('total_tokens', 0)}/{max_tokens}, "
        f"steps={metrics.get('step_count', 0)}/{max_steps}, "
        f"tools={metrics.get('tool_call_count', 0)}/{max_tool_calls}"
    )
    logger.info(_return_msg)
    _safe_debug_print(_return_msg)

    # Output clean AI summary message for state (no orphaned tool_calls)
    output_messages = []
    if response_content:
        ai_summary = AIMessage(
            content=f"[Subtask {current_subtask['id']}] {response_content[:2000]}"
        )
        output_messages.append(ai_summary)

    # ── P6: Update global evidence pool with this subtask's high-quality fetch results ──
    updated_pool = _update_global_evidence_pool(
        existing_pool=list(state.get("global_evidence_pool", [])),
        subtask=current_subtask,
        tool_results=all_tool_results,
    )

    return {
        "subtasks": subtasks,
        "current_subtask_index": next_idx,
        "execution_results": execution_results,
        "messages": output_messages,
        "metrics": metrics,
        "status": new_status,
        "citations": citations,
        "tool_call_log": tool_call_log,
        "global_evidence_pool": updated_pool,
    }


# ═══════════════════════════════════════════════════════════════
# P6: Global Evidence Pool — cross-subtask evidence sharing
# ═══════════════════════════════════════════════════════════════

def _extract_keywords_from_text(text: str, max_words: int = 20) -> list[str]:
    """Extract meaningful keywords from subtask description or content."""
    import re as _re
    # Remove punctuation and split
    words = _re.sub(r"[^\w\s]", " ", text.lower()).split()
    # Skip stop words
    stops = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "and", "or",
        "is", "are", "was", "were", "be", "been", "has", "have", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "from",
        "with", "that", "this", "these", "those", "what", "how", "when", "where",
        "which", "who", "its", "it", "we", "they", "their", "our", "your", "his",
        "her", "可以", "对于", "以下", "如何", "什么", "包括", "分析", "评估", "整合",
        "综合", "基于", "包含", "进行", "通过", "获取", "提供",
    }
    meaningful = [w for w in words if len(w) > 2 and w not in stops]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for w in meaningful:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_words]


def _keyword_overlap(kw_a: list[str], kw_b: list[str]) -> float:
    """Jaccard-like keyword overlap score between two keyword lists."""
    if not kw_a or not kw_b:
        return 0.0
    set_a, set_b = set(kw_a), set(kw_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _update_global_evidence_pool(
    existing_pool: list[dict[str, Any]],
    subtask: dict[str, Any],
    tool_results: list[dict[str, Any]],
    max_pool_entries: int = 50,
    min_quality_score: float = 0.05,   # Lowered: qs=0.10 is common even for good pages
    min_content_chars: int = 300,
) -> list[dict[str, Any]]:
    """Add evidence from this subtask's tool results to the global pool.

    Accepts fetch_url, search_document, and sec_edgar_* results.
    Prefers extracted_text over content (extracted_text is the full
    cleaned version; content is the truncated conversation version).
    Pool is capped at max_pool_entries (oldest entries removed first).
    """
    subtask_id = subtask.get("id", "")
    subtask_keywords = _extract_keywords_from_text(subtask.get("description", ""))
    new_entries: list[dict[str, Any]] = []
    existing_urls = {e.get("url") for e in existing_pool}

    for tr in tool_results:
        if not tr.get("success"):
            continue
        if tr.get("discarded"):
            continue
        tool_name = tr.get("tool_name", "")
        result = tr.get("result", {})
        if not isinstance(result, dict):
            continue

        if tool_name == "fetch_url":
            url = result.get("url", "")
            qs = result.get("quality_score", 1.0)
            # Prefer extracted_text (larger, cleaner) over content
            extracted = (result.get("extracted_text") or "").strip()
            content = (result.get("content") or result.get("excerpt") or "").strip()
            best_text = extracted if len(extracted) > len(content) else content

            if (
                url
                and url not in existing_urls
                and best_text
                and len(best_text) >= min_content_chars
                and qs >= min_quality_score
            ):
                new_entries.append({
                    "url": url,
                    "content": best_text[:8000],
                    "keywords": subtask_keywords,
                    "subtask_id": subtask_id,
                    "quality_score": qs,
                    "tool_name": tool_name,
                })
                existing_urls.add(url)

        elif tool_name == "search_document":
            # RAG results: semantically relevant chunks
            combined = result.get("combined_text", "")
            url = result.get("url", "")
            if url and url not in existing_urls and combined and len(combined) >= min_content_chars:
                new_entries.append({
                    "url": url,
                    "content": combined[:8000],
                    "keywords": subtask_keywords,
                    "subtask_id": subtask_id,
                    "quality_score": 0.9,
                    "tool_name": "search_document",
                })
                existing_urls.add(url)

        elif tool_name in ("sec_edgar_financials", "sec_edgar_filings"):
            # Structured financial API — always high-trust evidence
            fins = result.get("financials") or result.get("filings") or []
            cik = result.get("cik", "")
            entity = result.get("entity_name", "")
            if fins:
                raw = sanitize_text_for_llm(
                    json.dumps({"entity": entity, "cik": cik, "data": fins},
                               ensure_ascii=False)[:8000]
                )
                pool_url = f"sec_edgar://{cik}/{tool_name}"
                if pool_url not in existing_urls and len(raw) >= min_content_chars:
                    new_entries.append({
                        "url": pool_url,
                        "content": raw,
                        "keywords": subtask_keywords,
                        "subtask_id": subtask_id,
                        "quality_score": 0.95,
                        "tool_name": tool_name,
                    })
                    existing_urls.add(pool_url)

    updated = existing_pool + new_entries
    # Trim to cap
    if len(updated) > max_pool_entries:
        updated = updated[-max_pool_entries:]
    if new_entries:
        logger.info(
            "[EvidencePool] Added %d entries from %s (pool size: %d)",
            len(new_entries), subtask_id, len(updated),
        )
    return updated


def _build_sibling_evidence_context(
    state: GraphState,
    current_subtask: dict[str, Any],
    max_total_chars: int = 3000,
    min_overlap: float = 0.12,
) -> str:
    """Find evidence pool entries relevant to *current_subtask* and format them.

    Uses keyword overlap to score relevance. Returns a formatted context block
    (empty string if nothing is relevant enough).
    """
    pool: list[dict[str, Any]] = state.get("global_evidence_pool", [])
    if not pool:
        return ""

    current_sid = current_subtask.get("id", "")
    current_kw  = _extract_keywords_from_text(current_subtask.get("description", ""))
    if not current_kw:
        return ""

    # Score each pool entry by keyword overlap with current subtask
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in pool:
        # Never inject a subtask's own evidence back as "sibling"
        if entry.get("subtask_id") == current_sid:
            continue
        entry_kw = entry.get("keywords", [])
        overlap   = _keyword_overlap(current_kw, entry_kw)
        if overlap >= min_overlap:
            scored.append((overlap, entry))

    if not scored:
        return ""

    # Sort by overlap descending, then take greedily up to max_total_chars
    scored.sort(key=lambda x: -x[0])
    parts: list[str] = []
    used_chars = 0
    for overlap, entry in scored:
        content  = entry.get("content", "")
        url      = entry.get("url", "")
        sid      = entry.get("subtask_id", "")
        if not content:
            continue
        budget   = max_total_chars - used_chars
        if budget <= 0:
            break
        snippet  = sanitize_text_for_llm(content[:budget])
        parts.append(
            f"### Evidence from sibling subtask [{sid}] (overlap={overlap:.2f})\n"
            f"Source: {url}\n{snippet}"
        )
        used_chars += len(snippet)

    if not parts:
        return ""

    header = (
        "## 🔄 Evidence already fetched by other subtasks — use this to avoid re-fetching:\n"
        "(The following content was fetched by peer subtasks and may contain data "
        "relevant to your current task. Treat it as additional source evidence.)\n\n"
    )
    return header + "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Helper: Compact conversation to control token usage
# ═══════════════════════════════════════════════════════════════


def _compact_conversation(
    messages: list,
    keep_recent_rounds: int = 2,
) -> list:
    """Compact conversation by summarizing old ToolMessages.

    Keeps SystemMessage, initial HumanMessage, and the last N rounds
    of AI+Tool messages fully detailed. Older ToolMessages are replaced
    with short summaries to drastically reduce token usage.

    A 'round' is an AIMessage (with tool_calls) followed by its ToolMessages.

    Args:
        messages: The full conversation message list.
        keep_recent_rounds: Number of recent rounds to keep fully detailed.

    Returns:
        Compacted message list.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # Determine each ToolMessage's tool-call round index while preserving message order.
    tool_round_by_index: dict[int, int] = {}
    round_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                round_idx += 1
        elif isinstance(msg, ToolMessage):
            if round_idx >= 0:
                tool_round_by_index[idx] = round_idx

    total_rounds = round_idx + 1
    if total_rounds <= keep_recent_rounds:
        return messages  # Nothing to compact

    # Compact older rounds in place: summarize only old ToolMessages, keep all
    # non-tool messages exactly in original order to avoid role-order corruption.
    cutoff = total_rounds - keep_recent_rounds
    compacted = []
    summarized_count = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            compacted.append(msg)
            continue
        msg_round = tool_round_by_index.get(idx, -1)
        if msg_round >= cutoff:
            compacted.append(msg)
            continue
        raw = str(msg.content or "")
        summary = raw[:450] if raw else "(empty)"
        if len(raw) > 450:
            summary += f"... [{len(raw)} chars total]"
        compacted.append(
            ToolMessage(
                content=f"[SUMMARIZED] {summary}",
                tool_call_id=msg.tool_call_id,
            )
        )
        summarized_count += 1

    before_chars = sum(len(m.content) if hasattr(m, 'content') else 0 for m in messages)
    after_chars = sum(len(m.content) if hasattr(m, 'content') else 0 for m in compacted)
    logger.info(
        f"[Executor] Conversation compacted: {before_chars:,} → {after_chars:,} chars "
        f"({total_rounds} rounds, {cutoff} compacted, {summarized_count} ToolMessages summarized)"
    )

    return compacted


# ═══════════════════════════════════════════════════════════════
# Helper: Rich cross-subtask context
# ═══════════════════════════════════════════════════════════════


def _build_cross_subtask_context(
    state: GraphState,
    *,
    current_subtask: dict | None = None,
) -> list[str]:
    """Build tiered context from previous subtask results.

    Implements Progressive Disclosure (from Anthropic Skills guide):
    - Layer 1 (always): subtask ID + one-line summary + verified source URLs only.
      This is safe to show every subtask — no LLM analysis text, no risk of
      hallucination amplification.
    - Layer 2 (dependency or synthesis subtask): raw tool results (fetched content
      from fetch_url / sec_edgar tools). These are primary evidence, not LLM text.
    - Layer 3 (synthesis/computation only): abbreviated LLM response with an
      explicit ⚠ hallucination warning. Never injected for research subtasks.

    The key invariant: LLM-generated analysis text is NEVER passed to a downstream
    research subtask as if it were verified ground truth.

    Args:
        state: Current graph state.
        current_subtask: The subtask about to be executed (used to decide layer depth).

    Returns:
        List of context strings.
    """
    context_parts = []
    execution_results = state.get("execution_results", [])

    if not execution_results:
        return context_parts

    subtask_type = (current_subtask or {}).get("subtask_type", "research")
    dep_ids: set[str] = set((current_subtask or {}).get("dependencies", []))

    context_parts.append("## Previous Subtask Results\n")

    for result in execution_results[-5:]:
        subtask_id = result.get("subtask_id", "unknown")
        summary = sanitize_text_for_llm(result.get("summary", "No summary"))[:200]
        is_dependency = subtask_id in dep_ids

        # ── Layer 1 (always): ID + summary + source URLs ──────────────────
        context_parts.append(f"### {subtask_id}")
        context_parts.append(f"Summary: {summary}")

        # Collect verified source URLs (no content — just the references)
        tool_results = result.get("tool_results", [])
        verified_urls: list[str] = []
        for tr in tool_results:
            if not tr.get("success") or not tr.get("result"):
                continue
            if tr.get("tool_name") == "fetch_url" and isinstance(tr["result"], dict):
                url = tr["result"].get("url", "")
                if url:
                    verified_urls.append(url)
        if verified_urls:
            context_parts.append(
                "Verified sources: " + ", ".join(verified_urls[:4])
            )

        # ── Layer 2 (dependency or synthesis): raw tool evidence ──────────
        # Only inject actual fetched content when this prior subtask is a
        # declared dependency OR the current subtask is a synthesis/computation.
        # This provides real evidence without amplifying LLM errors.
        if is_dependency or subtask_type in ("synthesis", "computation"):
            fetch_items: list[str] = []
            for tr in tool_results:
                if not tr.get("success") or not tr.get("result"):
                    continue
                tn = tr.get("tool_name", "")
                r = tr["result"]
                if tn == "fetch_url" and isinstance(r, dict):
                    url = r.get("url", "")
                    content = r.get("content", "") or r.get("excerpt", "")
                    if content:
                        safe = sanitize_text_for_llm(
                            _extract_text_excerpt(content, max_len=600)
                        )
                        fetch_items.append(f"  [{url}] {safe}")
                elif tn in ("sec_edgar_financials", "sec_edgar_filings") and r:
                    safe = sanitize_text_for_llm(str(r)[:600])
                    fetch_items.append(f"  [SEC EDGAR/{tn}] {safe}")

            if fetch_items:
                context_parts.append("Evidence from tool results:")
                context_parts.extend(fetch_items[:3])

        # ── Layer 3 (synthesis/computation only): LLM analysis (with warning) ──
        # For research subtasks this layer is deliberately OMITTED to prevent
        # hallucination propagation. A research subtask must go to primary sources.
        if subtask_type in ("synthesis", "computation") and is_dependency:
            response = result.get("response", "")
            if response:
                response_str = sanitize_text_for_llm(
                    str(response)[:MAX_PREV_CONTEXT_PER_SUBTASK]
                )
                context_parts.append(
                    "⚠ LLM analysis from prior subtask (treat as unverified summary; "
                    "numbers NOT guaranteed accurate — cross-check against tool evidence above):\n"
                    f"{response_str}"
                )

        # Quality warnings
        quality_issues = result.get("quality_issues")
        if quality_issues:
            context_parts.append(f"Quality issues: {quality_issues}")

        context_parts.append("")  # Blank line between subtasks

    return context_parts


# ═══════════════════════════════════════════════════════════════
# Helper: Pre-search with automatic fetch
# ═══════════════════════════════════════════════════════════════


async def _pre_search_with_fetch(
    subtask: dict,
    provider,
    tool_executor,
    citations: list,
    tool_call_log: list,
    metrics: dict,
    *,
    user_request: str = "",
    entity_profile: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], list, list, dict]:
    """Pre-search with automatic fetch_url for top results.

    1. Run web_search to find relevant URLs
    2. Fetch full content from top N results via fetch_url
    3. Return enriched context with actual page content for the LLM

    Args:
        subtask: Current subtask dict.
        provider: LLM provider used for fetch-page discard decisions.
        tool_executor: Tool executor instance.
        citations: Current citations list (mutated in place).
        tool_call_log: Current tool call log (mutated in place).
        metrics: Current metrics dict (mutated in place).

    Returns:
        Tuple of (search_context_str, prefetch_tool_results, citations, tool_call_log, metrics).
    """
    search_context = ""
    prefetch_tool_results: list[dict[str, Any]] = []

    try:
        raw_desc = subtask["description"]
        profile = entity_profile or _resolve_entity_profile_for_subtask(raw_desc)
        entity_aliases = profile.get("aliases", [])
        # If this subtask explicitly depends on other subtasks' outputs (e.g. "从 subtask_001..."),
        # pre-search tends to become instruction-driven and can pull irrelevant documents.
        # In these cases, rely on cross-subtask context instead of doing a blind web_search.
        if re.search(r"\bsubtask_\d+\b", raw_desc.lower()):
            return search_context, prefetch_tool_results, citations, tool_call_log, metrics

        # ── Step 0: Extract and fetch URLs explicitly mentioned in subtask description ──
        # When the planner specifies exact URLs (e.g., SEC filings, press releases),
        # those should be fetched FIRST, before any web search.
        explicit_urls = re.findall(
            r'https?://[^\s\)\]\}\"\'<>,，。；]+', raw_desc
        )
        explicit_fetched: dict[str, str] = {}
        explicit_discarded: dict[str, str] = {}
        for url in explicit_urls[:3]:
            url = url.rstrip(".,;:)")
            if not url or any(c.get("url") == url and c.get("fetched_content") for c in citations):
                continue
            try:
                fetch_request = MCPToolRequest(
                    tool_name="fetch_url",
                    arguments={"url": url},
                )
                fetch_result = await tool_executor.execute(fetch_request)
                metrics["tool_call_count"] = metrics.get("tool_call_count", 0) + 1

                if fetch_result.is_success and isinstance(fetch_result.result, dict):
                    content_excerpt = (
                        fetch_result.result.get("excerpt")
                        or fetch_result.result.get("content")
                        or ""
                    )
                    if len(content_excerpt) > PRE_FETCH_MAX_CONTENT:
                        content_excerpt = content_excerpt[:PRE_FETCH_MAX_CONTENT]
                    discard, discard_reason = await _llm_should_discard_fetch_result(
                        provider=provider,
                        subtask_description=raw_desc,
                        relevance_query=raw_desc,
                        fetch_payload=fetch_result.result,
                        metrics=metrics,
                    )
                    if discard:
                        explicit_discarded[url] = discard_reason
                        tool_call_log.append({
                            "tool_name": "fetch_url",
                            "args": {"url": url},
                            "result_summary": (
                                f"Pre-fetched explicit URL discarded: {discard_reason}"
                            ),
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "subtask_id": subtask["id"],
                            "execution_time_ms": fetch_result.execution_time_ms,
                        })
                        prefetch_tool_results.append({
                            "tool_name": "fetch_url",
                            "args": {"url": url},
                            "result": fetch_result.result,
                            "error": None,
                            "success": True,
                            "prefetch": True,
                            "discarded": True,
                            "discard_reason": discard_reason,
                        })
                        logger.info(
                            "[Executor] Discarded explicit URL %s: %s",
                            url[:120],
                            discard_reason,
                        )
                        continue

                    explicit_fetched[url] = content_excerpt

                    is_citable = bool(fetch_result.result.get("is_citable", True))
                    not_citable_reason = fetch_result.result.get("not_citable_reason")

                    existing = next((c for c in citations if c.get("url") == url), None)
                    if existing:
                        existing["fetched_content"] = content_excerpt
                        existing["verified"] = is_citable
                        existing["is_citable"] = is_citable
                        if not_citable_reason:
                            existing["not_citable_reason"] = not_citable_reason
                    else:
                        safe_title = sanitize_text_for_llm(
                            fetch_result.result.get("title", "")
                        )
                        source_tier = _classify_source_tier(url)
                        citations.append({
                            "id": len(citations) + 1,
                            "title": safe_title,
                            "url": url,
                            "snippet": "",
                            "source_tool": "fetch_url",
                            "accessed_at": datetime.now().isoformat(),
                            "verified": is_citable,
                            "is_citable": is_citable,
                            "fetched_content": content_excerpt,
                            "source_tier": source_tier,
                            "subtask_id": subtask["id"],
                        })

                    tool_call_log.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result_summary": f"Pre-fetched explicit URL: {len(content_excerpt)} chars",
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "subtask_id": subtask["id"],
                        "execution_time_ms": fetch_result.execution_time_ms,
                    })
                    prefetch_tool_results.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result": fetch_result.result,
                        "error": None,
                        "success": True,
                        "prefetch": True,
                    })
                    logger.info(
                        f"[Executor] Pre-fetched explicit URL from subtask description: "
                        f"{url} ({len(content_excerpt)} chars)"
                    )
            except Exception as e:
                logger.warning(
                    f"[Executor] Failed to pre-fetch explicit URL {url}: {e}"
                )

        # Build atomic search queries via LLM (P0 fix: entity-anchored, semantics-aware)
        queries = await _llm_generate_search_queries(subtask, user_request, profile)
        if not queries:
            # Fallback: legacy regex approach
            queries = _split_to_atomic_queries(raw_desc)
            if not queries:
                fallback = _normalize_search_query(raw_desc)
                queries = [fallback] if fallback else []
        # Verify entity anchoring; add alias prefix where missing.
        anchored_queries: list[str] = []
        for q in queries:
            q2 = expand_query_with_alias_anchor(q, entity_aliases, max_len=60)
            if q2 and _validate_search_query(q2, required_aliases=entity_aliases) is None:
                anchored_queries.append(q2)
        if anchored_queries:
            queries = anchored_queries

        if not queries:
            # Even without search queries, we may have fetched explicit URLs
            if explicit_fetched:
                parts = ["## Pre-fetched content from explicit URLs:\n"]
                for url, content in explicit_fetched.items():
                    safe = sanitize_text_for_llm(content)
                    parts.append(f"### {url}\n{safe}\n")
                search_context = "\n".join(parts)
            return search_context, prefetch_tool_results, citations, tool_call_log, metrics

        # ── Step 1: Web search with top 2 atomic queries ──
        # Using 2 queries is safer than 1: query decomposition doesn't
        # guarantee perfect ordering, and the marginal cost is negligible.
        # If all queries return 0 relevant results, ask LLM to regenerate
        # queries with failure context before falling back to raw results.
        web_results: list[dict] = []
        _failed_queries: list[str] = []
        _irrelevant_snippets: list[str] = []

        async def _run_search_batch(
            query_list: list[str],
            *,
            is_retry: bool = False,
        ) -> None:
            """Run a batch of search queries and collect relevant results.

            Populates web_results in the enclosing scope. Tracks failed
            queries and irrelevant snippets for potential LLM retry.
            """
            for search_query in query_list:
                if not search_query:
                    continue

                search_request = MCPToolRequest(
                    tool_name="web_search",
                    arguments={"query": search_query, "num_results": 5},
                )
                search_result = await tool_executor.execute(search_request)
                metrics["tool_call_count"] = metrics.get("tool_call_count", 0) + 1

                if not (search_result.is_success and isinstance(search_result.result, dict)):
                    continue

                batch_results = search_result.result.get("results", [])
                # Use all results; relevance is judged by LLM after fetch_url (not regex here).
                relevant_batch_results = batch_results
                filter_mode = "llm_post_fetch"

                if not batch_results:
                    _failed_queries.append(search_query)

                tagged_batch_results: list[dict[str, Any]] = []
                for sr in relevant_batch_results:
                    url = sr.get("url", "")
                    if url and not any(c.get("url") == url for c in citations):
                        safe_snippet = sanitize_text_for_llm(sr.get("snippet", "")[:500])
                        safe_title = sanitize_text_for_llm(sr.get("title", ""))
                        source_tier = _classify_source_tier(url)
                        citations.append({
                            "id": len(citations) + 1,
                            "title": safe_title,
                            "url": url,
                            "snippet": safe_snippet,
                            "source_tool": "web_search",
                            "accessed_at": datetime.now().isoformat(),
                            "verified": False,
                            "fetched_content": "",
                            "source_tier": source_tier,
                            "subtask_id": subtask["id"],
                        })
                    sr_tagged = dict(sr)
                    sr_tagged["_matched_query"] = search_query
                    tagged_batch_results.append(sr_tagged)

                tool_call_log.append({
                    "tool_name": "web_search",
                    "args": {"query": search_query, "num_results": 5},
                    "result_summary": (
                        f"Found {len(batch_results)} results, "
                        f"kept {len(relevant_batch_results)} relevant "
                        f"(mode={filter_mode})"
                    ),
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "subtask_id": subtask["id"],
                    "execution_time_ms": search_result.execution_time_ms,
                })
                prefetch_tool_results.append({
                    "tool_name": "web_search",
                    "args": {"query": search_query, "num_results": 5},
                    "result": search_result.result,
                    "error": None,
                    "success": True,
                    "prefetch": True,
                })

                web_results.extend(tagged_batch_results)
                _searched_topics.append(_extract_topic_keys(search_query))

                logger.info(
                    f"[Executor] Pre-search query: {search_query!r} → "
                    f"{len(batch_results)} raw / {len(relevant_batch_results)} relevant "
                    f"(mode={filter_mode})"
                )

        await _run_search_batch(queries[:2])

        # ── LLM query retry: if initial queries returned 0 relevant results,
        # ask LLM to generate new queries informed by the failure context.
        if not web_results and _failed_queries:
            logger.info(
                f"[Executor] All pre-search queries returned 0 relevant results for "
                f"{subtask['id']}. Asking LLM to regenerate queries."
            )
            retry_queries = await _llm_retry_search_queries(
                subtask=subtask,
                user_request=user_request,
                entity_profile=entity_profile or {},
                failed_queries=_failed_queries,
                irrelevant_snippets=_irrelevant_snippets,
            )
            if retry_queries:
                anchored_retry: list[str] = []
                for q in retry_queries:
                    q2 = expand_query_with_alias_anchor(q, entity_aliases, max_len=60)
                    if q2 and _validate_search_query(q2, required_aliases=entity_aliases) is None:
                        anchored_retry.append(q2)
                if anchored_retry:
                    retry_queries = anchored_retry
                await _run_search_batch(retry_queries[:2], is_retry=True)

        # Deduplicate URLs across the two searches
        seen_urls: set[str] = set()
        unique_web_results: list[dict] = []
        for wr in web_results:
            url = wr.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(wr)

        # ── Step 2: Source-policy gate + fetch full content from top N URLs ──
        # Prioritize official/regulatory/company domains and recent items first.
        unique_web_results.sort(
            key=lambda sr: _prefetch_priority_score(sr, entity_aliases),
            reverse=True,
        )
        fetched_contents = {}
        discarded_prefetch: dict[str, str] = {}
        accepted_prefetch = 0
        fetch_attempts = 0
        for sr in unique_web_results:
            if accepted_prefetch >= PRE_FETCH_TOP_N:
                break
            if fetch_attempts >= PRE_FETCH_MAX_ATTEMPTS:
                break
            url = sr.get("url", "")
            if not url:
                continue
            fetch_attempts += 1
            try:
                fetch_request = MCPToolRequest(
                    tool_name="fetch_url",
                    arguments={"url": url},
                )
                fetch_result = await tool_executor.execute(fetch_request)
                metrics["tool_call_count"] = metrics.get("tool_call_count", 0) + 1

                if fetch_result.is_success and isinstance(fetch_result.result, dict):
                    # fetch_url returns a readable excerpt in "content"
                    content_excerpt = (fetch_result.result.get("excerpt") or fetch_result.result.get("content") or "")
                    if len(content_excerpt) > PRE_FETCH_MAX_CONTENT:
                        content_excerpt = content_excerpt[:PRE_FETCH_MAX_CONTENT]
                    matched_query = str(sr.get("_matched_query") or raw_desc)
                    discard, discard_reason = await _llm_should_discard_fetch_result(
                        provider=provider,
                        subtask_description=raw_desc,
                        relevance_query=matched_query,
                        fetch_payload=fetch_result.result,
                        metrics=metrics,
                    )
                    if discard:
                        discarded_prefetch[url] = discard_reason
                        tool_call_log.append({
                            "tool_name": "fetch_url",
                            "args": {"url": url},
                            "result_summary": (
                                f"Discarded fetched page: {discard_reason}"
                            ),
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "subtask_id": subtask["id"],
                            "execution_time_ms": fetch_result.execution_time_ms,
                        })
                        prefetch_tool_results.append({
                            "tool_name": "fetch_url",
                            "args": {"url": url},
                            "result": fetch_result.result,
                            "error": None,
                            "success": True,
                            "prefetch": True,
                            "discarded": True,
                            "discard_reason": discard_reason,
                        })
                        logger.info(
                            "[Executor] Discarded pre-fetched URL %s: %s",
                            url[:120],
                            discard_reason,
                        )
                        continue

                    fetched_contents[url] = content_excerpt

                    is_citable = bool(fetch_result.result.get("is_citable", True))
                    not_citable_reason = fetch_result.result.get("not_citable_reason")

                    # Update citation with verified fetched content
                    for c in citations:
                        if c.get("url") == url:
                            c["fetched_content"] = content_excerpt
                            c["verified"] = is_citable
                            c["is_citable"] = is_citable
                            if not_citable_reason:
                                c["not_citable_reason"] = not_citable_reason
                            c["content_type"] = fetch_result.result.get("content_type", "")
                            c["source_tier"] = _classify_source_tier(url)
                            break

                    # Log fetch
                    tool_call_log.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result_summary": (
                            f"Fetched {len(content_excerpt)} chars from {url}"
                        ),
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "subtask_id": subtask["id"],
                        "execution_time_ms": fetch_result.execution_time_ms,
                    })
                    # Persist into this subtask's tool_results so later subtasks can reuse it
                    prefetch_tool_results.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result": fetch_result.result,
                        "error": None,
                        "success": True,
                        "prefetch": True,
                    })
                    accepted_prefetch += 1
                else:
                    tool_call_log.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result_summary": f"Failed: {fetch_result.error}",
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                        "subtask_id": subtask["id"],
                        "execution_time_ms": fetch_result.execution_time_ms,
                    })
                    prefetch_tool_results.append({
                        "tool_name": "fetch_url",
                        "args": {"url": url},
                        "result": None,
                        "error": fetch_result.error,
                        "success": False,
                        "prefetch": True,
                    })

            except Exception as e:
                logger.warning(f"[Executor] Pre-fetch failed for {url}: {e}")
                tool_call_log.append({
                    "tool_name": "fetch_url",
                    "args": {"url": url},
                    "result_summary": f"Error: {str(e)}",
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "subtask_id": subtask["id"],
                    "execution_time_ms": 0,
                })
                prefetch_tool_results.append({
                    "tool_name": "fetch_url",
                    "args": {"url": url},
                    "result": None,
                    "error": str(e),
                    "success": False,
                    "prefetch": True,
                })

        # ── Build enriched context with actual content ──
        search_parts = []

        # Include explicit URL content first (highest priority)
        if explicit_fetched:
            search_parts.append("## Content from URLs specified in subtask description\n")
            for url, content in explicit_fetched.items():
                safe = sanitize_text_for_llm(content)
                search_parts.append(
                    f"### {url}\n"
                    f"**Full page content (excerpt, {len(content)} chars):**\n"
                    f"{safe}\n"
                )
        if explicit_discarded:
            search_parts.append("## Discarded explicit URLs\n")
            for url, reason in explicit_discarded.items():
                search_parts.append(f"- {url} (discarded: {reason})")

        search_parts.append("## Pre-search Results (with fetched page content)\n")
        for i, sr in enumerate(unique_web_results, 1):
            url = sr.get("url", "")
            search_parts.append(
                f"### [{i}] {sr.get('title', '')}\n"
                f"URL: {url}\n"
                f"Snippet: {sr.get('snippet', '')}"
            )
            if url in fetched_contents:
                content = fetched_contents[url]
                search_parts.append(
                    f"\n**Full page content (excerpt, {len(content)} chars):**\n"
                    f"{content}\n"
                )
            elif url in discarded_prefetch:
                search_parts.append(
                    f"(Fetched then discarded: {discarded_prefetch[url]})\n"
                )
            else:
                search_parts.append("(Page content not fetched or fetch failed)\n")

        # Sanitize the entire search context to prevent content moderation issues
        search_context = sanitize_text_for_llm("\n".join(search_parts))

    except Exception as e:
        logger.warning(f"[Executor] Pre-search failed: {e}")

    return search_context, prefetch_tool_results, citations, tool_call_log, metrics


# ═══════════════════════════════════════════════════════════════
# Helper: Search relevance and source policy
# ═══════════════════════════════════════════════════════════════

# Official domains — merged from all loaded domain profiles at import time.
# This avoids hardcoding domain-specific lists here; add new official domains
# in the corresponding domains/*.md file instead.
def _build_merged_official_domains() -> tuple[str, ...]:
    try:
        from agent_engine.agents.domain_profile import _PROFILES as _dp
        return tuple(d for p in _dp for d in p.official_domains)
    except Exception:
        return ()

_OFFICIAL_DOMAINS: tuple[str, ...] = _build_merged_official_domains()

# Generic quality-signal media domains — not domain-profile-specific.
_TRUSTED_MEDIA_DOMAINS = (
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
)


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _classify_source_tier(url: str) -> str:
    """Classify URL into source tiers for quality control."""
    domain = _extract_domain(url)
    if not domain:
        return "unknown"
    if any(domain == d or domain.endswith(f".{d}") for d in _OFFICIAL_DOMAINS):
        return "regulator"
    if "investor" in domain or domain.endswith("ir"):
        return "company_ir"
    if any(domain == d or domain.endswith(f".{d}") for d in _TRUSTED_MEDIA_DOMAINS):
        return "trusted_media"
    return "unknown"


def _extract_year(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _prefetch_priority_score(sr: dict[str, Any], entity_aliases: list[str]) -> int:
    """Higher score means should fetch earlier."""
    url = str(sr.get("url", "") or "")
    title = str(sr.get("title", "") or "")
    snippet = str(sr.get("snippet", "") or "")
    tier = _classify_source_tier(url)

    tier_score = {
        "regulator": 50,
        "company_ir": 40,
        "trusted_media": 25,
        "unknown": 5,
    }.get(tier, 0)

    rel_bonus = 12 if _is_search_result_relevant(
        sr,
        search_query="",
        entity_aliases=entity_aliases,
        require_entity_alias=bool(entity_aliases),
    ) else 0

    year = _extract_year(f"{title} {snippet}")
    freshness_bonus = 0
    if year is not None:
        freshness_bonus = max(0, 8 - max(0, datetime.now().year - year))

    return tier_score + rel_bonus + freshness_bonus


_QUERY_FILTER_STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that",
    "report", "reports", "latest", "official", "company",
    "year", "data", "news", "update",
}


def _extract_query_terms_for_relevance(
    search_query: str,
    entity_aliases: list[str],
) -> list[str]:
    """Extract stable topic terms from a query for second-pass filtering."""
    q = (search_query or "").lower().strip()
    if not q:
        return []

    raw_terms = re.findall(r"[a-z0-9]{2,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}", q)

    alias_terms: set[str] = set()
    for alias in entity_aliases:
        a = (alias or "").strip().lower()
        if not a:
            continue
        alias_terms.add(a)
        alias_terms.update(
            re.findall(r"[a-z0-9]{2,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}", a)
        )

    terms: list[str] = []
    for t in raw_terms:
        if t in _QUERY_FILTER_STOP_WORDS:
            continue
        if t in alias_terms:
            continue
        if re.search(r"(?:%[0-9a-f]{2}){2,}", t):
            continue
        if t not in terms:
            terms.append(t)
    return terms[:12]


def _is_search_result_relevant(
    sr: dict[str, Any],
    *,
    search_query: str,
    entity_aliases: list[str],
    require_entity_alias: bool = True,
) -> bool:
    """Filter noisy web results before fetch_url."""
    title = str(sr.get("title", "") or "").lower()
    snippet = str(sr.get("snippet", "") or "").lower()
    url = str(sr.get("url", "") or "").lower()
    blob = f"{title} {snippet} {url}"

    # Strict mode for entity-specific lookups.
    if require_entity_alias and entity_aliases:
        if not any(a.lower() in blob for a in entity_aliases if a):
            return False

    query_terms = _extract_query_terms_for_relevance(search_query, entity_aliases)
    if query_terms:
        overlap = sum(1 for t in query_terms if t in blob)
        if overlap == 0:
            return False

    # Drop obvious unrelated social/news noise (sourced from domain profile)
    noise_terms = get_domain_noise_patterns(f"{search_query} {blob}")
    if any(n.lower() in blob for n in noise_terms):
        return False

    return True


# ═══════════════════════════════════════════════════════════════
# Helper: Citation tracking
# ═══════════════════════════════════════════════════════════════


def _extract_first_json_object(raw: str) -> dict[str, Any]:
    """Extract first JSON object from LLM text output."""
    text = str(raw or "").strip()
    if not text:
        return {}
    if "```json" in text:
        try:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        except Exception:
            pass
    elif "```" in text:
        try:
            text = text.split("```", 1)[1].split("```", 1)[0]
        except Exception:
            pass
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _build_discard_cache_key(
    *,
    relevance_query: str,
    fetch_payload: dict[str, Any],
) -> str:
    """Build a stable cache key for discard judgments."""
    url = str(fetch_payload.get("url", "") or "").strip()
    excerpt = str(fetch_payload.get("excerpt") or fetch_payload.get("content") or "").strip()
    snippets = fetch_payload.get("evidence_snippets") or []
    snippet_join = ""
    if isinstance(snippets, list):
        parts = []
        for item in snippets[:4]:
            if not isinstance(item, dict):
                continue
            s = str(item.get("snippet") or "").strip()
            if s:
                parts.append(s[:180])
        snippet_join = "|".join(parts)
    raw = f"{url}|{relevance_query[:220]}|{excerpt[:700]}|{snippet_join}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _quick_discard_decision(fetch_payload: dict[str, Any]) -> tuple[bool, str] | None:
    """Cheap heuristic triage before calling LLM (returns None when uncertain)."""
    excerpt = str(fetch_payload.get("excerpt") or fetch_payload.get("content") or "").strip()
    snippets = fetch_payload.get("evidence_snippets") or []
    is_js = bool(fetch_payload.get("is_js_rendered", False))
    quality_score = float(fetch_payload.get("quality_score", 1.0) or 1.0)

    snippet_texts: list[str] = []
    if isinstance(snippets, list):
        for item in snippets[:8]:
            if not isinstance(item, dict):
                continue
            snip = str(item.get("snippet") or "").strip()
            if snip:
                snippet_texts.append(snip)

    combined = " ".join([excerpt, *snippet_texts]).strip()
    if not combined:
        return True, "empty content"
    if is_js and quality_score < 0.2 and len(excerpt) < 350 and not snippet_texts:
        return True, "js-rendered placeholder page"
    if quality_score < 0.08 and len(excerpt) < 200:
        return True, f"very low quality ({quality_score:.2f})"
    if snippet_texts and len(" ".join(snippet_texts)) >= 140:
        return False, "heuristic keep: has evidence snippets"
    if len(excerpt) >= 900 and quality_score >= 0.35:
        return False, "heuristic keep: sufficient content"
    return None


async def _llm_should_discard_fetch_result(
    *,
    provider,
    subtask_description: str,
    relevance_query: str,
    fetch_payload: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Use LLM to decide whether fetched page should be discarded.

    Fallback: if LLM call fails or returns invalid JSON, fall back to heuristic decision.
    """
    url = str(fetch_payload.get("url", "") or "")
    cache_key = _build_discard_cache_key(
        relevance_query=relevance_query,
        fetch_payload=fetch_payload,
    )
    cached = _discard_decision_cache.get(cache_key)
    if cached is not None:
        return cached

    quick = _quick_discard_decision(fetch_payload)
    if quick is not None:
        _discard_decision_cache[cache_key] = quick
        return quick

    excerpt = str(fetch_payload.get("excerpt") or fetch_payload.get("content") or "").strip()
    evidence_snippets = fetch_payload.get("evidence_snippets") or []

    lines: list[str] = []
    if isinstance(evidence_snippets, list):
        for item in evidence_snippets[:8]:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or "").strip()
            snip = str(item.get("snippet") or "").strip()
            if not snip:
                continue
            if term:
                lines.append(f"- [{term}] {snip}")
            else:
                lines.append(f"- {snip}")

    evidence_parts = []
    if lines:
        evidence_parts.append("Evidence snippets:\n" + "\n".join(lines))
    if excerpt:
        evidence_parts.append("Excerpt:\n" + excerpt[:1400])
    evidence_text = sanitize_text_for_llm("\n\n".join(evidence_parts))
    if len(evidence_text) > FETCH_DISCARD_EVIDENCE_CHARS:
        evidence_text = evidence_text[:FETCH_DISCARD_EVIDENCE_CHARS]

    system_prompt = (
        "You are a strict web-evidence triage judge for a research subtask.\n"
        "Decide whether a fetched webpage should be discarded.\n"
        "Return JSON only: {\"discard\": true|false, \"reason\": \"...\"}\n"
        "Rules:\n"
        "1) discard=true ONLY when page is empty, template/placeholder, inaccessible, "
        "or clearly irrelevant to the subtask/query.\n"
        "2) If page has partial but relevant evidence, keep it (discard=false).\n"
        "3) Be conservative: avoid false discard."
    )
    user_prompt = (
        f"Subtask: {subtask_description[:500]}\n"
        f"Current query: {relevance_query[:300]}\n"
        f"Fetched evidence:\n{evidence_text or '(empty)'}"
    )

    try:
        resp = await provider.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        if metrics is not None:
            in_tok = int(getattr(resp, "input_tokens", 0) or 0)
            out_tok = int(getattr(resp, "output_tokens", 0) or 0)
            if in_tok or out_tok:
                metrics["input_tokens"] = metrics.get("input_tokens", 0) + in_tok
                metrics["output_tokens"] = metrics.get("output_tokens", 0) + out_tok
                metrics["total_tokens"] = (
                    metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
                )
        parsed = _extract_first_json_object(getattr(resp, "content", "") or "")
        if parsed:
            discard = bool(parsed.get("discard", False))
            reason = str(parsed.get("reason", "") or "").strip()[:200]
            if discard:
                decision = (True, reason or "llm judged irrelevant/empty")
            else:
                decision = (False, reason or "llm judged usable")
            _discard_decision_cache[cache_key] = decision
            if len(_discard_decision_cache) > 2048:
                _discard_decision_cache.pop(next(iter(_discard_decision_cache)))
            return decision
        raise ValueError("invalid json from llm discard judge")
    except Exception as e:
        logger.warning(
            "[Executor] LLM discard judge failed for %s: %s; using heuristic fallback",
            url[:120],
            str(e)[:200],
        )
        decision = _should_discard_fetch_result(
            fetch_payload,
            relevance_query=relevance_query,
        )
        _discard_decision_cache[cache_key] = decision
        return decision


def _should_discard_fetch_result(
    fetch_payload: dict[str, Any],
    *,
    relevance_query: str,
) -> tuple[bool, str]:
    """Decide whether a fetched page should be discarded for this subtask."""
    if not isinstance(fetch_payload, dict):
        return True, "invalid fetch payload"

    url = str(fetch_payload.get("url", "") or "")
    excerpt = str(fetch_payload.get("excerpt") or fetch_payload.get("content") or "").strip()
    evidence_snippets = fetch_payload.get("evidence_snippets") or []
    is_js = bool(fetch_payload.get("is_js_rendered", False))
    quality_score = float(fetch_payload.get("quality_score", 1.0) or 1.0)

    snippet_texts: list[str] = []
    if isinstance(evidence_snippets, list):
        for item in evidence_snippets[:8]:
            if not isinstance(item, dict):
                continue
            snip = str(item.get("snippet") or "").strip()
            if snip:
                snippet_texts.append(snip)

    combined_text = " ".join([excerpt, *snippet_texts]).strip()
    if not combined_text:
        return True, "empty content"
    if len(combined_text) < MIN_FETCH_CONTENT_CHARS and not snippet_texts:
        return True, f"content too short ({len(combined_text)} chars)"
    if is_js and len(excerpt) < 500:
        return True, "js-rendered placeholder page"
    if quality_score < 0.2 and len(excerpt) < 500:
        return True, f"very low quality ({quality_score:.2f})"
    if _is_low_quality_content(url, combined_text, domain_text=""):
        return True, "low-signal quote/dashboard content"

    blob = combined_text.lower()
    query_terms = _extract_query_terms_for_relevance(relevance_query, [])
    if query_terms:
        overlap = sum(1 for t in query_terms if t in blob)
        if overlap == 0:
            return True, "no overlap with subtask terms"

    noise_terms = get_domain_noise_patterns(relevance_query)
    if any((n or "").lower() in blob for n in noise_terms):
        return True, "matched domain noise patterns"

    return False, ""


def _update_citations(
    citations: list,
    tool_name: str,
    result,
    *,
    subtask_id: str = "",
    skip_fetch_url: bool = False,
) -> list:
    """Update citations list from tool execution result.

    web_search results are added as unverified; fetch_url and search_document
    results are verified (fetched content available).

    Args:
        citations: Current citations list.
        tool_name: Name of the tool that produced the result.
        result: Tool execution result.
        subtask_id: ID of the subtask that produced this citation (for provenance).
        skip_fetch_url: If True, do not persist fetch_url result into citations.

    Returns:
        Updated citations list.
    """
    if not result.is_success:
        return citations

    if tool_name == "web_search" and isinstance(result.result, dict):
        search_results = result.result.get("results", [])
        for sr in search_results:
            url = sr.get("url", "")
            if url and not any(c.get("url") == url for c in citations):
                # Sanitize snippet before storing — unsafe content here
                # propagates to cross-subtask context, critic, and reporter.
                raw_snippet = sr.get("snippet", "")[:500]
                safe_snippet = sanitize_text_for_llm(raw_snippet)
                safe_title = sanitize_text_for_llm(sr.get("title", ""))
                citations.append({
                    "id": len(citations) + 1,
                    "title": safe_title,
                    "url": url,
                    "snippet": safe_snippet,
                    "source_tool": "web_search",
                    "accessed_at": datetime.now().isoformat(),
                    "verified": False,
                    "fetched_content": "",
                    "source_tier": _classify_source_tier(url),
                    "subtask_id": subtask_id,
                })

    elif (
        tool_name == "fetch_url"
        and not skip_fetch_url
        and isinstance(result.result, dict)
    ):
        url = result.result.get("url", "")
        # fetch_url guarantees "content" is a readable excerpt text now.
        content_excerpt = (result.result.get("excerpt") or result.result.get("content") or "").strip()
        evidence_snippets = result.result.get("evidence_snippets") or []
        is_citable = bool(result.result.get("is_citable", True))
        not_citable_reason = result.result.get("not_citable_reason")
        content_type = result.result.get("content_type", "")
        extracted_text_length = result.result.get("extracted_text_length", 0)

        # Build fetched_content: evidence snippets FIRST (primary evidence),
        # then the excerpt as context.  Evidence snippets are extracted from
        # the FULL text using focus_terms, so they contain the most relevant
        # data even when the excerpt is truncated.
        fetched_content = content_excerpt
        if isinstance(evidence_snippets, list) and evidence_snippets:
            lines = []
            for item in evidence_snippets[:8]:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term") or "").strip()
                snip = str(item.get("snippet") or "").strip()
                if not snip:
                    continue
                if term:
                    lines.append(f"- [{term}] {snip}")
                else:
                    lines.append(f"- {snip}")
            if lines:
                fetched_content = "Evidence snippets:\n" + "\n".join(lines)
                if content_excerpt:
                    fetched_content += "\n\nExcerpt:\n" + content_excerpt
        # Hard cap to keep state manageable.
        if len(fetched_content) > 4000:
            fetched_content = fetched_content[:4000] + f"\n... [truncated, {len(fetched_content)} total chars]"
        # Sanitize fetched content before storing
        fetched_content = sanitize_text_for_llm(fetched_content)

        citation_data = {
            "source_tool": "fetch_url",
            "accessed_at": datetime.now().isoformat(),
            "verified": is_citable,
            "is_citable": is_citable,
            "fetched_content": fetched_content,
            "content_type": content_type,
            "not_citable_reason": not_citable_reason,
            "extracted_text_length": extracted_text_length,
            "source_tier": _classify_source_tier(url),
        }

        if url and not any(c.get("url") == url for c in citations):
            citations.append({
                "id": len(citations) + 1,
                "title": url,
                "url": url,
                "snippet": sanitize_text_for_llm(content_excerpt[:500]),
                "subtask_id": subtask_id,
                **citation_data,
            })
        else:
            # Update existing citation with fetched content
            for c in citations:
                if c.get("url") == url and not c.get("fetched_content"):
                    c.update(citation_data)
                    break

    elif tool_name == "search_document" and isinstance(result.result, dict):
        # search_document fetches a URL and returns semantically relevant chunks.
        # Add as verified citation so the document can be cited in the report.
        res = result.result
        if res.get("error"):
            return citations
        url = res.get("url", "")
        combined_text = (res.get("combined_text") or "").strip()
        if not url:
            return citations
        if any(c.get("url") == url for c in citations):
            # Update existing citation with search_document content if empty
            for c in citations:
                if c.get("url") == url and not c.get("fetched_content"):
                    fetched = sanitize_text_for_llm(combined_text[:4000])
                    if len(combined_text) > 4000:
                        fetched += "\n... [truncated]"
                    c["fetched_content"] = fetched
                    c["verified"] = True
                    c["source_tool"] = "search_document"
                    break
            return citations
        fetched_content = sanitize_text_for_llm(combined_text[:4000])
        if len(combined_text) > 4000:
            fetched_content += "\n... [truncated]"
        citations.append({
            "id": len(citations) + 1,
            "title": url,
            "url": url,
            "snippet": sanitize_text_for_llm(combined_text[:500]),
            "source_tool": "search_document",
            "accessed_at": datetime.now().isoformat(),
            "verified": True,
            "fetched_content": fetched_content,
            "source_tier": _classify_source_tier(url),
            "subtask_id": subtask_id,
        })

    return citations


# ═══════════════════════════════════════════════════════════════
# Helper: Quality validation
# ═══════════════════════════════════════════════════════════════


def _validate_tool_quality(tool_results: list) -> list[str]:
    """Validate quality of tool results from a single round.

    Args:
        tool_results: List of tool result dicts from one round.

    Returns:
        List of quality issue descriptions.
    """
    quality_issues = []
    for tr in tool_results:
        if tr.get("success") and tr.get("result"):
            if tr["tool_name"] == "web_search" and isinstance(tr["result"], dict):
                search_results = tr["result"].get("results", [])
                if not search_results:
                    quality_issues.append("web_search returned no results")
                else:
                    query = tr.get("args", {}).get("query", "").lower()
                    irrelevant_count = 0
                    for sr in search_results:
                        title = sr.get("title", "").lower()
                        snippet = sr.get("snippet", "").lower()
                        if query and len(query) > 3:
                            query_words = set(query.split()[:3])
                            content = (title + " " + snippet).lower()
                            if not any(
                                word in content
                                for word in query_words
                                if len(word) > 2
                            ):
                                irrelevant_count += 1
                    if irrelevant_count == len(search_results) and len(search_results) > 0:
                        quality_issues.append(
                            f"web_search results may be irrelevant to query: {query[:50]}"
                        )
    return quality_issues


# ═══════════════════════════════════════════════════════════════
# Helper: URL and text extraction
# ═══════════════════════════════════════════════════════════════


def _extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from text content.

    Args:
        text: Text that may contain URLs.

    Returns:
        List of unique URLs found.
    """
    if not text:
        return []
    url_pattern = re.compile(
        r'https?://[^\s<>\"\'\)\]\}，。、；：！？】）》\u3000]+'
    )
    urls = url_pattern.findall(text)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        url = url.rstrip('.,;:!?')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _is_low_quality_content(
    url: str,
    content: str,
    *,
    domain_text: str = "",
) -> bool:
    """Check if fetched content is low-quality and should be excluded from
    auto-synthesized summaries.

    Detects stock quote pages, ticker dashboards, and other content that is
    primarily raw numbers with no analytical value.

    The list of known low-quality domains is sourced from the active
    DomainProfile so it can be extended without touching this function.

    Args:
        url: The source URL.
        content: The fetched text content.
        domain_text: Context text used to detect the active domain profile.

    Returns:
        True if the content should be excluded from summaries.
    """
    url_lower = url.lower()

    # Domain-profile-driven domain list (replaces hardcoded _quote_domains)
    profile = detect_domain_profile(domain_text)
    if profile.is_low_quality_url(url):
        return True

    # Universal heuristic: content that is mostly numbers with quote-page markers
    _quote_markers = [
        "前收市", "開市", "買入價", "賣出價", "今日波幅",
        "成交量", "市值", "Chart Range",
        "前收盘", "开盘", "买入", "卖出",
        "Prev Close", "Open", "Bid", "Ask", "Day's Range",
        "Volume", "Market Cap",
    ]
    if content and len(content) > 100:
        digit_chars = sum(1 for c in content[:1000] if c.isdigit())
        ratio = digit_chars / min(len(content), 1000)
        if ratio > 0.25 and any(marker in content for marker in _quote_markers):
            return True

    return False


def _extract_text_excerpt(html_or_text: str, max_len: int = 2000) -> str:
    """Extract meaningful text from HTML or plain text content.

    Strips HTML tags and collapses whitespace to get clean text.

    Args:
        html_or_text: Raw HTML or text content.
        max_len: Maximum excerpt length.

    Returns:
        Clean text excerpt.
    """
    if not html_or_text:
        return ""
    # Strip HTML tags
    text = re.sub(
        r'<script[^>]*>.*?</script>', '', html_or_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r'<style[^>]*>.*?</style>', '', text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&nbsp;', ' ')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]


def _extract_context_around_url(text: str, url: str, window: int = 200) -> str:
    """Extract text context around a URL in content.

    Args:
        text: Full text content.
        url: URL to find context around.
        window: Characters of context to include on each side.

    Returns:
        Context string around the URL.
    """
    if not text or not url:
        return ""
    idx = text.find(url)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(url) + window)
    context = text[start:end].strip()
    context = context.replace(url, "").strip()
    context = re.sub(r'\s+', ' ', context)
    return context[:500]


# ═══════════════════════════════════════════════════════════════
# Helper: Reuse results from previous iterations
# ═══════════════════════════════════════════════════════════════


def _find_reusable_result(
    subtask: dict,
    execution_results: list[dict],
    reuse_mode: str = "fuzzy",
) -> dict | None:
    """Find a completed result from a previous iteration that matches this subtask.

    Matching strategy:
    1. Exact match on subtask_id (e.g., "subtask_001")
    2. High overlap in description keywords (for renamed subtasks with same intent)

    Only returns results that had successful tool calls (not "LLM direct answer" only),
    to avoid reusing low-quality results.

    Args:
        subtask: Current subtask dict.
        execution_results: All execution results from previous iterations.

    Returns:
        Matching result dict, or None if no reusable result found.
    """
    if not execution_results:
        return None

    subtask_id = subtask.get("id", "")
    subtask_desc_raw = subtask.get("description", "")
    subtask_desc = subtask_desc_raw.lower()
    subtask_fp = _subtask_fingerprint(subtask_desc_raw)

    for result in execution_results:
        result_id = result.get("subtask_id", "")
        result_fp = result.get("subtask_fingerprint")

        # Skip results that had no tool calls (LLM-only responses are unreliable)
        if not result.get("tool_results") and not result.get("response"):
            continue

        # Strategy 1: Exact ID match
        if result_id == subtask_id:
            # In strict mode, only reuse when the subtask fingerprint matches.
            # This prevents "replan with stricter constraints" from silently reusing old work.
            if reuse_mode == "strict":
                if not result_fp:
                    continue
                if result_fp != subtask_fp:
                    continue

            # Verify the result has meaningful content
            response = result.get("response", "")
            if response and len(str(response)) > 50:
                logger.info(
                    f"[Executor] Found reusable result by ID match: {subtask_id}"
                )
                return result

        # Strategy 2: Description similarity (for replanned subtasks with new IDs)
        if reuse_mode == "strict":
            continue
        # Extract keywords from both descriptions and check overlap
        result_desc = ""
        # Try to find the description from the summary
        summary = result.get("summary", "").lower()
        # We need at least 60% keyword overlap to consider it a match
        if subtask_desc and summary:
            desc_words = set(subtask_desc.split())
            # Remove very common words
            common = {"的", "了", "在", "是", "和", "与", "及", "对", "将", "从",
                       "the", "a", "an", "is", "are", "for", "to", "of", "in", "and"}
            desc_words -= common
            if len(desc_words) >= 3:
                matched = sum(1 for w in desc_words if w in summary)
                overlap = matched / len(desc_words) if desc_words else 0
                if overlap >= 0.6:
                    response = result.get("response", "")
                    if response and len(str(response)) > 50:
                        logger.info(
                            f"[Executor] Found reusable result by description overlap "
                            f"({overlap:.0%}): {subtask_id} ← {result_id}"
                        )
                        return result

    return None


def _subtask_fingerprint(description: str) -> str:
    """Stable fingerprint of a subtask description.

    Used to make result reuse constraint-aware: if the planner changes constraints
    (e.g., "only use SEC 20-F"), the description changes → fingerprint changes →
    executor will re-run instead of reusing old results.
    """
    desc = (description or "").strip().lower()
    # Normalize whitespace to reduce accidental diffs
    desc = re.sub(r"\s+", " ", desc)
    # Short, stable identifier (keep small to avoid bloating state/logs)
    return hashlib.sha256(desc.encode("utf-8")).hexdigest()[:12]


def _validate_search_query(
    query: str,
    *,
    required_aliases: list[str] | None = None,
) -> str | None:
    """Validate that a search query is meaningful enough to send to DuckDuckGo.

    Returns None if the query is acceptable, or a rejection reason string if it
    should be rejected (returned as an error to the LLM so it can reformulate).

    Catches the common LLM failure modes:
      - Bare domain names (``sec.gov``, ``businesswire.com``)
      - Single generic words (``pdf``, ``ir``, ``html``)
      - Pure URLs that should use ``fetch_url`` instead
    """
    q = (query or "").strip()

    if not q:
        return "Query is empty."

    # Reject pure URLs — LLM should use fetch_url
    if re.match(r"^https?://", q, re.IGNORECASE):
        return (
            "This is a URL, not a search query. "
            "Use fetch_url to retrieve a known URL directly."
        )

    # Reject bare domain names (e.g. "sec.gov", "businesswire.com", "www.hkexnews.hk")
    if re.match(
        r"^(www\.)?\S+\.(com|gov|org|net|cn|hk|eu|uk|io|co|info|edu)(/\S*)?$",
        q, re.IGNORECASE,
    ):
        return (
            "This looks like a domain name, not a search query. "
            "Add specific keywords describing what you are looking for, e.g. "
            "\"ENTITY_NAME YEAR report\" instead of a bare domain."
        )

    # Reject too-short queries (< 6 chars of actual content after stripping)
    content_chars = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf]", "", q)
    if len(content_chars) < 4:
        return (
            f"Query too short ({len(content_chars)} meaningful characters). "
            f"Provide more specific keywords."
        )

    # Reject queries that are just common file-format words
    generic_words = {
        "pdf", "html", "xml", "json", "csv", "doc", "xls", "xlsx",
        "ir", "url", "api", "www", "http", "https", "ftp",
    }
    q_lower = q.lower().strip()
    if q_lower in generic_words:
        return (
            f"\"{q}\" is too generic. Add the company name, topic, "
            f"and year to make a useful search query."
        )

    # Reject overly long queries — each query should target ONE factual question.
    # Queries > 50 chars almost always contain multiple topics or instruction text,
    # which confuse search engines and return relevant=0.
    if len(q) > 50:
        return (
            f"Query too long ({len(q)} chars). A good search query targets ONE "
            f"specific fact and uses 3-7 keywords (max 50 chars). Split your "
            f"question into multiple shorter queries. Example: instead of "
            f"'获取XX公司官方文件编号并查询最新报告', use "
            f"'公司名 报告类型 年份' for the first search."
        )

    if required_aliases:
        if not is_entity_anchored(q, required_aliases):
            return (
                "Query is missing the target entity name/ticker. "
                "Add one alias (e.g., company name, ticker, or identifier) so results stay relevant."
            )

    return None  # query is acceptable


# ═══════════════════════════════════════════════════════════════
# Helper: Topic-level query deduplication
# ═══════════════════════════════════════════════════════════════

_TOPIC_STOP_WORDS = {
    "use", "the", "and", "for", "from", "with", "this", "that", "its",
    "not", "via", "api", "url", "fetch", "search", "query",
    "based", "using", "through", "including", "related",
}


def _extract_topic_keys(query: str) -> frozenset[str]:
    """Extract core topic keywords from a query for topic-level dedup.

    Strips language differences, word order, and filler words.
    Returns a frozenset of normalized tokens (2+ chars).
    """
    q = (query or "").lower()
    q = re.sub(r"\b(the|and|for|from|with|of|in|on|at|to|a|an|is|are)\b", " ", q)
    tokens: set[str] = set()
    # CJK character runs (2+)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,}", q):
        tokens.add(m.group())
    # Alphanumeric tokens (2+ chars)
    for m in re.finditer(r"[a-z0-9][a-z0-9.-]+", q):
        word = m.group()
        if len(word) >= 2 and word not in _TOPIC_STOP_WORDS:
            tokens.add(word)
    return frozenset(tokens)


def _is_topic_searched(query: str) -> str | None:
    """Check if a query's topic has already been searched.

    Returns None if the topic is new, or a message string if it's a repeat.
    Uses Jaccard-like overlap: if overlap >= 85% of the smaller set, it's a repeat.
    Threshold relaxed to allow same-entity different-dimension searches
    (e.g. "company revenue" vs "company R&D expenses").
    """
    keys = _extract_topic_keys(query)
    if len(keys) < 2:
        return None

    for prev_keys in _searched_topics:
        if not prev_keys:
            continue
        overlap = keys & prev_keys
        smaller = min(len(keys), len(prev_keys))
        if smaller > 0 and len(overlap) / smaller >= 0.85:
            return (
                f"This topic was already searched (overlapping keywords: "
                f"{', '.join(sorted(overlap)[:5])}). "
                f"Search for a DIFFERENT topic or use fetch_url on a specific URL."
            )
    return None


# ═══════════════════════════════════════════════════════════════
# Helper: Split subtask description into atomic search queries
# ═══════════════════════════════════════════════════════════════


def _xbrl_to_human_readable(xbrl_data: dict) -> str:
    """Convert XBRL/SEC API JSON integers to human-readable strings for P4 matching.

    SEC EDGAR XBRL returns raw integers like {"revenue": 3783241000} but the
    LLM writes "3.78 billion" or "3.8 billion" or "3,783 million". This function
    adds ALL common representations (1-, 2-, 3-decimal) to the verified corpus.

    Handles: billions, millions, millions of yuan (亿), percentages.
    """
    parts: list[str] = []

    def _walk(obj: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:20]:
                _walk(item, depth + 1)
        elif isinstance(obj, (int, float)):
            val = float(obj)
            # Skip tiny values (counts, IDs, etc.)
            if abs(val) < 1000:
                return
            # Billions — add 1-, 2-, 3-decimal variants so "3.8 billion" and "3.81 billion"
            # are both matched regardless of LLM rounding preference.
            if abs(val) >= 1e9:
                b = val / 1e9
                parts.append(f"{b:.1f}B")
                parts.append(f"{b:.2f}B")
                parts.append(f"{b:.1f} billion")
                parts.append(f"{b:.2f} billion")
                parts.append(f"{b:.3f} billion")
                parts.append(f"{val/1e6:,.0f} million")
                parts.append(f"{val/1e6:,.1f} million")
            # Millions
            elif abs(val) >= 1e6:
                m = val / 1e6
                parts.append(f"{m:.1f}M")
                parts.append(f"{m:.2f}M")
                parts.append(f"{m:.1f} million")
                parts.append(f"{m:.2f} million")
                parts.append(f"{m:,.1f} million")
            # 亿 (Chinese unit) — 1e8 threshold
            if abs(val) >= 1e8:
                yi = val / 1e8
                parts.append(f"{yi:.1f}亿")
                parts.append(f"{yi:.2f}亿")
                parts.append(f"{yi:.3f}亿")
            # 万 (Chinese unit)
            if 1e4 <= abs(val) < 1e8:
                parts.append(f"{val/1e4:.2f}万")
            # Raw integer string (already in JSON but add stripped version)
            parts.append(str(int(val)))

    _walk(xbrl_data)
    return " ".join(parts)


def _constrain_output_to_tool_evidence(
    response_text: str,
    tool_results: list[dict[str, Any]],
) -> tuple[str, int]:
    """P4: Replace ungrounded numerical claims with [DATA NOT VERIFIED] markers.

    Scans the response for significant numbers (monetary, percentages, large integers).
    Uses a two-tier corpus:
      • high_trust_corpus — numbers from authoritative APIs (sec_edgar_*).  A number
        found here is ALWAYS allowed through, regardless of other sources.
      • standard_corpus   — numbers from fetch_url / web_search snippets.

    If a number cannot be matched in either corpus it is replaced with [DATA NOT VERIFIED].

    Returns:
        (constrained_text, replacement_count)
    """
    if not tool_results:
        return response_text, 0

    # ── Build two-tier verified corpora ──────────────────────────────────────
    # high_trust: official government/API data — always authoritative
    high_trust_corpus = ""
    # standard: web content — trust but verify
    standard_corpus = ""

    for tr in tool_results:
        if not tr.get("success") or not isinstance(tr.get("result"), dict):
            continue
        r = tr["result"]
        tool_name = tr.get("tool_name", "")

        if tool_name in ("sec_edgar_financials", "sec_edgar_filings"):
            # Include full JSON (no truncation for key XBRL integers) and all
            # human-readable renderings (1-, 2-, 3-decimal variants).
            high_trust_corpus += " " + json.dumps(r)[:8000]
            high_trust_corpus += " " + _xbrl_to_human_readable(r)
            # Also include the structured summary field if present
            high_trust_corpus += " " + r.get("summary", "")
        elif tool_name == "fetch_url":
            content = r.get("excerpt") or r.get("content") or ""
            standard_corpus += " " + content[:15000]
        elif tool_name in ("search_document", "sec_edgar_search"):
            content = r.get("content") or r.get("excerpt") or ""
            standard_corpus += " " + content[:10000]
        elif tool_name == "web_search":
            for sr in r.get("results", [])[:5]:
                standard_corpus += " " + sr.get("snippet", "")

    if not high_trust_corpus.strip() and not standard_corpus.strip():
        return response_text, 0

    # ── Numeric matching helpers ──────────────────────────────────────────────

    # Regex for explicit scale suffixes — used to decide if a number is "large"
    _SCALE_SUFFIX = re.compile(
        r"(billion|million|trillion|亿|万|[Bb]\b|[Mm]\b)", re.IGNORECASE
    )
    # Currency tokens that may be appended to the pattern match but not to the corpus
    _CURRENCY_SUFFIX = re.compile(
        r"(USD|CNY|EUR|GBP|AUD|美元|元|人民币)$", re.IGNORECASE
    )

    def _num_in_corpus(num_str: str, corpus: str) -> bool:
        """Check if a number string appears verbatim in the corpus.

        "Verbatim" means the digit+scale sequence (after stripping spaces, thousands
        commas, and trailing currency tokens) must appear as a substring of the corpus.

        No tolerance rounding, no scale conversion — the LLM is required to copy
        values exactly as the tool returned them.

        Passes through unconditionally:
          • Numbers whose leading digit sequence is < 1 000 AND that carry no explicit
            scale suffix (e.g. bare percentages like "38.5%", IDs, ratios).
        """
        if not corpus.strip():
            return False

        # Normalise: strip spaces and thousands-separating commas, keep decimal dots
        core = re.sub(r"[\s,]", "", num_str)
        corpus_norm = re.sub(r"[\s,]", "", corpus)

        if not core or len(core) < 2:
            return True

        # ── 1. Direct verbatim substring match ────────────────────────────────
        if core in corpus_norm:
            return True

        # ── 2. Try without trailing currency token ────────────────────────────
        # Pattern may capture "3.81 billion USD" → core = "3.81billionUSD"
        # but corpus may have "3.81billion" (no currency suffix) from xbrl renderer.
        core_no_currency = _CURRENCY_SUFFIX.sub("", core)
        if core_no_currency != core and core_no_currency in corpus_norm:
            return True

        # Also try stripping currency-symbol prefix ($ ¥ € etc.)
        core_stripped = re.sub(r"^[$¥€£￥]", "", core_no_currency)
        if core_stripped != core_no_currency and core_stripped in corpus_norm:
            return True

        # ── 3. Small-number bypass (no scale suffix only) ─────────────────────
        # Numbers like "3.5" (ratio), "38.5%" (pct) — too small to meaningfully
        # verify against financial corpora.  BUT if the number carries an explicit
        # scale word (billion, million, 亿, etc.) it is NOT small regardless of
        # the leading digit value, so do NOT bypass.
        has_scale = bool(_SCALE_SUFFIX.search(num_str))
        if not has_scale:
            leading = re.search(r"[\d\.]+", core)
            if leading:
                try:
                    small_val = float(leading.group())
                    if small_val < 1000:
                        return True
                except ValueError:
                    pass

        return False

    # ── Number pattern to scan in response ───────────────────────────────────
    _NUMBER_PATTERN = re.compile(
        r"""
        (?:
            # Monetary: $10B, ¥3.78亿, 18.9亿美元, USD 3.8 billion
            (?:[$¥€£￥]\s*)?
            \d[\d,\.]*\s*
            (?:亿|万|billion|million|B\b|M\b|K\b|trillion|百亿|千亿)?
            \s*(?:美元|元|人民币|USD|CNY|EUR|GBP|AUD)?
            |
            # Percentages
            \d[\d,\.]*\s*%
            |
            # Large standalone integers (not years)
            (?<!\b20)\d{5,}(?!\d)
        )
        """,
        re.VERBOSE,
    )

    _YEAR_PATTERN = re.compile(r"\b20[12]\d\b")

    replacement_count = 0

    def replace_ungrounded(match: re.Match) -> str:
        nonlocal replacement_count
        num_str = match.group(0)

        # Always allow year numbers
        if _YEAR_PATTERN.fullmatch(num_str.strip()):
            return num_str
        # Always allow citation markers [N]
        if re.fullmatch(r"\[\d+\]", num_str.strip()):
            return num_str
        # Allow very small leading-digit values ONLY when no scale suffix is present.
        # "3.5" (a ratio), "0.28" (a fraction) → safe to pass through without corpus check.
        # "3.8 billion" → has_scale=True → must be verified in corpus; do NOT bypass.
        if not _SCALE_SUFFIX.search(num_str):
            try:
                core_digits = re.search(r"[\d\.]+", num_str)
                if core_digits and float(core_digits.group()) <= 10:
                    return num_str
            except ValueError:
                pass

        # ── Tier 1: high-trust sources (SEC EDGAR API) ───────────────────────
        # If the number appears in official API data, always pass through.
        if high_trust_corpus and _num_in_corpus(num_str, high_trust_corpus):
            return num_str

        # ── Tier 2: standard web sources ────────────────────────────────────
        if standard_corpus and _num_in_corpus(num_str, standard_corpus):
            return num_str

        # ── No source confirms this number → redact ──────────────────────────
        # Only redact when at least one corpus exists (avoid false positives on
        # subtasks with no tool calls or exclusively high-trust-only results).
        if not high_trust_corpus and not standard_corpus:
            return num_str

        replacement_count += 1
        return "[DATA NOT VERIFIED]"

    result_text = _NUMBER_PATTERN.sub(replace_ungrounded, response_text)

    if replacement_count > 0:
        logger.info(
            f"[Executor] P4: Replaced {replacement_count} ungrounded numerical claims "
            f"with [DATA NOT VERIFIED]"
        )

    return result_text, replacement_count


async def _llm_generate_search_queries(
    subtask: dict,
    user_request: str,
    entity_profile: dict[str, Any],
) -> list[str]:
    """Use LLM (MiniMax) to generate entity-anchored search queries for a subtask.

    Replaces the regex-based `_split_to_atomic_queries` to produce semantically
    correct, entity-anchored queries that avoid the description-as-query anti-pattern.

    Returns up to 3 search queries. Falls back to regex approach on error.
    """
    from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

    raw_desc = subtask.get("description", "")
    canonical = entity_profile.get("canonical_name", "")
    aliases = entity_profile.get("aliases", [])[:3]
    alias_str = "、".join(aliases) if aliases else canonical

    system_content = (
        "You are a search query generator. Given a research subtask description, "
        "generate 1-3 concise web search queries (in the same language as the description). "
        "Rules:\n"
        "1. Each query MUST be anchored to the main entity (company/product/subject name).\n"
        "2. Each query should target ONE specific fact (e.g., annual report, regulatory approval, etc.).\n"
        "3. Keep each query under 60 characters. Use keywords, not full sentences.\n"
        "4. Do NOT use instruction words like '请', '确保', '注意' in queries.\n"
        "5. Do NOT generate queries that repeat the same topic.\n"
        "6. If the subtask is a synthesis/summary task (no new data needed), return an EMPTY list.\n"
        "Respond with a JSON object: {\"queries\": [\"q1\", \"q2\", \"q3\"]}"
    )
    user_content = (
        f"Main entity: {canonical} (also known as: {alias_str})\n"
        f"Original user request context: {user_request[:200]}\n\n"
        f"Subtask description:\n{raw_desc}\n\n"
        f"Generate search queries for this subtask."
    )

    try:
        settings = get_settings()
        # MiniMax-M2.1 is hosted on DashScope — use qwen provider with the model name.
        # This is the same pattern the Planner uses for high-quality task decomposition.
        provider = get_provider(provider="qwen", model=settings.planner_model)
        resp = await provider.invoke(
            [_SM(content=system_content), _HM(content=user_content)],
        )
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content)
        raw_queries = parsed.get("queries", [])
        queries = [q.strip() for q in raw_queries if q and isinstance(q, str) and len(q) >= 5]
        logger.info(
            f"[Executor] LLM-generated search queries for {subtask.get('id')}: {queries}"
        )
        return queries[:3]
    except Exception as e:
        logger.warning(
            f"[Executor] LLM query generation failed ({e}); falling back to regex"
        )
        return []


async def _llm_retry_search_queries(
    subtask: dict,
    user_request: str,
    entity_profile: dict[str, Any],
    failed_queries: list[str],
    irrelevant_snippets: list[str],
) -> list[str]:
    """Ask LLM to regenerate search queries after previous queries returned irrelevant results.

    Provides the LLM with what was tried and what came back, so it can
    produce fundamentally different queries instead of minor variations.

    Args:
        subtask: Current subtask dict.
        user_request: Original user request.
        entity_profile: Entity profile with aliases.
        failed_queries: Queries that returned 0 relevant results.
        irrelevant_snippets: Sample titles/snippets from the irrelevant results.

    Returns:
        Up to 3 new search queries, or empty list on failure.
    """
    from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

    canonical = entity_profile.get("canonical_name", "")
    aliases = entity_profile.get("aliases", [])[:3]
    alias_str = ", ".join(aliases) if aliases else canonical
    raw_desc = subtask.get("description", "")

    failed_summary = "\n".join(f"  - \"{q}\"" for q in failed_queries[:3])
    noise_summary = "\n".join(f"  - {s}" for s in irrelevant_snippets[:5])

    system_content = (
        "You are a search query optimizer. Previous search queries returned IRRELEVANT results. "
        "Your job is to generate COMPLETELY DIFFERENT queries that will find the right information.\n\n"
        "Rules:\n"
        "1. Each query MUST contain the entity name or a well-known alias.\n"
        "2. Each query should target ONE specific fact.\n"
        "3. Keep each query under 50 characters.\n"
        "4. Try a DIFFERENT language or angle than the failed queries.\n"
        "5. If the failed queries were in Chinese, try English (and vice versa).\n"
        "6. Use specific product names, ticker symbols, or industry terms to improve precision.\n"
        "7. Do NOT repeat or minimally rephrase the failed queries.\n"
        "Respond with a JSON object: {\"queries\": [\"q1\", \"q2\", \"q3\"]}"
    )
    user_content = (
        f"Entity: {canonical} (aliases: {alias_str})\n"
        f"Task: {raw_desc[:200]}\n\n"
        f"FAILED queries (returned irrelevant results):\n{failed_summary}\n\n"
        f"Irrelevant results that came back:\n{noise_summary}\n\n"
        f"Generate 2-3 NEW search queries using a different strategy."
    )

    try:
        settings = get_settings()
        provider = get_provider(provider="qwen", model=settings.planner_model)
        resp = await provider.invoke(
            [_SM(content=system_content), _HM(content=user_content)],
        )
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content)
        raw_queries = parsed.get("queries", [])
        new_queries = [q.strip() for q in raw_queries if q and isinstance(q, str) and len(q) >= 5]
        logger.info(
            f"[Executor] LLM retry queries for {subtask.get('id')}: {new_queries}"
        )
        return new_queries[:3]
    except Exception as e:
        logger.warning(
            f"[Executor] LLM retry query generation failed ({e})"
        )
        return []


def _split_to_atomic_queries(description: str) -> list[str]:
    """Split a subtask description into atomic search queries.

    Each query targets ONE factual question, suitable for a single search.

    Strategy:
    1. Split description by sentence/clause boundaries
    2. Discard instruction-only segments (no entity nouns)
    3. For each remaining clause, normalize into a short keyword query
    4. Deduplicate and return ordered by specificity

    Example:
        Input:  "获取目标公司的官方编号，并查询最新年度报告信息。
                 确认公司的披露状态"
        Output: ["公司名 官方编号 监管机构",
                 "公司名 年度报告 类型",
                 "公司名 披露状态"]
    """
    desc = (description or "").strip()
    if not desc:
        return []

    # Step 1: Split by sentence boundaries, then by commas for long segments
    segments = re.split(r"[。；\n]+", desc)

    expanded: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > 60:
            sub_parts = re.split(r"[，,]+", seg)
            for sp in sub_parts:
                sp = sp.strip()
                if sp and len(sp) >= 6:
                    expanded.append(sp)
        else:
            if len(seg) >= 6:
                expanded.append(seg)

    if not expanded:
        expanded = [desc]

    # Step 1.5: Filter out instruction-only segments
    # A segment is instruction-only if it contains no entity-like keywords
    # (proper nouns, technical terms, company names, form types).
    filtered: list[str] = []
    for seg in expanded:
        if _is_instruction_only(seg):
            continue
        filtered.append(seg)

    # If everything was filtered, fall back to the original description
    if not filtered:
        filtered = [desc]

    # Step 2: Normalize each segment into a keyword query
    queries: list[str] = []
    seen_topics: list[frozenset[str]] = []

    for seg in filtered:
        q = _normalize_search_query(seg)
        if not q or len(q) < 8:
            continue

        # Deduplicate by topic overlap
        q_keys = _extract_topic_keys(q)
        is_dup = False
        for prev in seen_topics:
            if prev and q_keys:
                overlap = q_keys & prev
                smaller = min(len(q_keys), len(prev))
                if smaller > 0 and len(overlap) / smaller >= 0.6:
                    is_dup = True
                    break
        if is_dup:
            continue

        seen_topics.append(q_keys)
        queries.append(q)

    # Step 3: Sort by length (shorter = more focused = better for search)
    queries.sort(key=lambda q: len(q))

    return queries[:3]


def _segment_has_domain_entity_marker(seg: str) -> bool:
    """Check if segment matches any entity marker pattern from loaded domain profiles.

    Replaces the hardcoded form-type regex (20-F, 10-K, FDA, SEC, etc.).
    All domain-specific patterns live in domains/*.md entity_marker_patterns.
    """
    try:
        from agent_engine.agents.domain_profile import _PROFILES as _dp
        for profile in _dp:
            for pattern in profile.entity_marker_patterns:
                if re.search(pattern, seg, re.IGNORECASE):
                    return True
    except Exception:
        pass
    return False


def _is_instruction_only(segment: str) -> bool:
    """Check if a text segment is pure instruction/constraint with no searchable entity.

    Returns True for segments like:
      - "使用IR网站或官方新闻稿作为来源"
      - "注意：只使用静态页面或PDF中的数据，避免JS渲染的实时行情页"
      - "每个结论都必须附上对应的数据来源URL"
      - "行业报告或第三方分析作为参考"

    Returns False for segments containing entity names, tickers, form types, etc.:
      - "目标公司的核心产品线和研发管线"
      - "监管机构数据库中目标公司的最新年度报告"
    """
    seg = segment.strip()
    if not seg:
        return True

    # Instruction-start patterns (CN + EN)
    instruction_starts = re.compile(
        r"^(?:注意|要求|确保|务必|请|必须|需要|使用|利用|优先|"
        r"重点提取|重点关注|"
        r"avoid|ensure|must|note|use|only|do not|make sure|remember)",
        re.IGNORECASE,
    )
    if instruction_starts.match(seg):
        return True

    # Instruction-keyword density check: if the segment is mostly filler
    # words and contains no proper noun / technical term, it's instruction.
    _instruction_phrases = [
        "作为来源", "作为参考", "来源优先", "优先选择",
        "数据来源", "只使用", "避免", "不要使用",
        "附上", "引用", "标注", "必须",
        "JS渲染", "静态页面", "实时行情",
        "每个结论", "每个关键", "每一个",
        "as a source", "as reference", "do not use",
    ]
    instruction_count = sum(1 for p in _instruction_phrases if p in seg)
    if instruction_count >= 2:
        return True

    # Check for entity presence: at least one of these must exist
    has_entity = False
    # English proper nouns / acronyms (2+ uppercase letters)
    if re.search(r"\b[A-Z]{2,}\b", seg):
        has_entity = True
    # CJK entity-like names (2+ CJK chars that are not common verbs/filler)
    _cjk_filler = set("的了和与及等是在从到通过使用获取提取确认查询查找"
                      "搜索分析评估包括覆盖关键核心最新近期相关重大主要"
                      "具体详细信息数据内容文件链接状态方面维度指标报告"
                      "结果公告更新注意只避免确保务必重点提取")
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", seg)
    for run in cjk_runs:
        if not all(c in _cjk_filler for c in run):
            has_entity = True
            break
    # Domain entity marker patterns (loaded from domain profile MD files)
    if _segment_has_domain_entity_marker(seg):
        has_entity = True
    # Year patterns
    if re.search(r"\b20\d{2}\b", seg):
        has_entity = True

    return not has_entity


def _normalize_search_query(query: str) -> str:
    """Normalize a search query to be keyword-oriented, not instruction-oriented.

    Prevents the common failure mode where an analysis instruction becomes the
    literal ``web_search`` query (tool usage becomes instruction-driven instead
    of reason-driven).

    Design principles (after the "ir" and "pdf" truncation bugs):
      1. **Never** blindly split on colons --- ``site:xxx``, ``Query: '...'``, and
         full-width colons all appear in legitimate subtask descriptions and
         splitting on the last segment is catastrophic.
      2. Stripping should only remove *known boilerplate* (tool-instruction
         prefixes, subtask references, common verbs) --- never domain-relevant
         nouns like company names, tickers, or filing types.
      3. After all cleaning, enforce a *minimum length* safety net: if the
         cleaned result is shorter than 6 characters, fall back to extracting
         capitalized keywords + CJK segments from the original text.
    """
    q = (query or "").strip()
    if not q:
        return ""
    original = q  # keep for fallback
    q = re.sub(r"\s+", " ", q)

    # -- 1. Drop raw URLs (often embedded in instructions) --
    q = re.sub(r"https?://\S+", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()

    # -- 2. Drop search-engine operators (site:xxx, filetype:xxx) --
    # These are useful for the *actual* search, but the pre-search builds its
    # own query; carrying them through would limit results.
    q = re.sub(r"\bsite:\S+", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\bfiletype:\S+", " ", q, flags=re.IGNORECASE)

    # -- 3. Drop tool-instruction prefixes (CN + EN) --
    q = re.sub(
        r"^\s*(?:\u901a\u8fc7|\u4f7f\u7528|\u7528)\s*web_search\s*"
        r"(?:\u67e5\u627e|\u641c\u7d22|\u68c0\u7d22|\u6765)?\s*",
        "", q, flags=re.IGNORECASE,
    )
    q = re.sub(
        r"^\s*web_search\s*(?:\u67e5\u627e|\u641c\u7d22|\u68c0\u7d22)\s*",
        "", q, flags=re.IGNORECASE,
    )
    q = re.sub(
        r"^\s*Use\s+web_search\s+to\s+(?:locate|find|search|search\s+for|look\s+for)\s*",
        "", q, flags=re.IGNORECASE,
    )

    # -- 4. Remove instructional label prefixes like "Query:" --
    # Only remove the *label* itself, keep everything after it.
    q = re.sub(
        r"\b(?:Query|Search|"
        r"\u641c\u7d22\u67e5\u8be2|\u67e5\u8be2|\u641c\u7d22\u5173\u952e\u8bcd"
        r")\s*[\uff1a:]\s*",
        "", q, flags=re.IGNORECASE,
    )

    # -- 5. Drop instruction boilerplate phrases --
    drop_phrases = [
        "\u901a\u8fc7 web_search", "\u901a\u8fc7web_search",
        "web_search \u67e5\u627e", "web_search\u641c\u7d22",
        "web_search \u68c0\u7d22", "web_search",
        "\u6c47\u603b\u77e5\u8bc6\u5e93", "\u77e5\u8bc6\u5e93\u5df2\u6709\u4fe1\u606f",
        "\u5df2\u6709\u4fe1\u606f", "\u4e0a\u8ff0\u5b50\u4efb\u52a1",
        "\u4ee5\u4e0a\u5b50\u4efb\u52a1",
        "\u4ece subtask", "\u4f9d\u8d56", "dependencies",
        "\u6839\u636e\u4e0a\u8ff0", "\u8bf7\u6839\u636e", "\u8bf7\u4f60",
        "\u5e76\u8f93\u51fa", "\u8f93\u51fa",
        "as above", "above subtask", "based on above", "write a report",
        "use the information", "instruction", "requirements",
        # Additional instruction phrases found in real subtask descriptions
        "\u91cd\u70b9\u63d0\u53d6\uff1a",  # 重点提取：
        "\u91cd\u70b9\u5173\u6ce8\uff1a",  # 重点关注：
        "\u6ce8\u610f\uff1a",  # 注意：
        "\u4f5c\u4e3a\u6765\u6e90",  # 作为来源
        "\u4f5c\u4e3a\u53c2\u8003",  # 作为参考
        "\u6765\u6e90\u4f18\u5148\u9009\u62e9",  # 来源优先选择
        "\u4f18\u5148\u4f7f\u7528",  # 优先使用
        "\u6570\u636e\u6765\u6e90",  # 数据来源
        "\u7528\u4e8e\u8865\u5145",  # 用于补充
        "\u4e4b\u5916\u7684\u5b9a\u6027\u5206\u6790",  # 之外的定性分析
        "\u5e76\u901a\u8fc7",  # 并通过
        "\u6838\u5bf9\u4e0e",  # 核对与
        "\u7684\u4e00\u81f4\u6027",  # 的一致性
    ]
    for p in drop_phrases:
        q = q.replace(p, " ")

    # Drop subtask references like "subtask_001"
    q = re.sub(r"\bsubtask_\d+\w*\b", " ", q, flags=re.IGNORECASE)
    # Drop "子任务" only when followed by a number
    q = re.sub(r"\u5b50\u4efb\u52a1\s*\d+", " ", q)

    # -- 6. Strip leading verbs (only at the very beginning) --
    leading_verbs = [
        "\u641c\u7d22", "\u67e5\u627e", "\u6536\u96c6", "\u8c03\u7814",
        "\u67e5\u8be2", "\u83b7\u53d6", "\u63d0\u53d6", "\u68b3\u7406",
        "\u6574\u7406", "\u6c47\u603b", "\u786e\u8ba4", "\u5f52\u7eb3",
        "\u603b\u7ed3", "\u64b0\u5199", "\u5206\u6790",
        "search for", "find", "collect", "research", "analyze", "summarize",
        "write", "locate",
    ]
    for prefix in sorted(leading_verbs, key=len, reverse=True):
        pat = rf"^\s*{re.escape(prefix)}\s*"
        q = re.sub(pat, "", q, count=1, flags=re.IGNORECASE)

    # -- 7. Clean up quotes and punctuation --
    q = re.sub(r'[\u201c\u201d"\'`*]+', " ", q)
    q = re.sub(r"\s+", " ", q).strip(" \uff0c,\u3002.;\uff1b:\uff1a-\u2014()\uff08\uff09")

    # -- 8. Truncate overly long queries --
    if len(q) > 120:
        for sep in ["\u3002", "\uff1b", "; ", ". "]:
            if sep in q:
                first_sentence = q.split(sep)[0].strip()
                if len(first_sentence) >= 15:
                    q = first_sentence
                    break
    if len(q) > 120:
        for sep in ["\uff0c", ", ", ","]:
            if sep in q:
                first_clause = q.split(sep)[0].strip()
                if len(first_clause) >= 20:
                    q = first_clause
                    break
    if len(q) > 120:
        q = q[:120].rstrip()

    # -- 9. Safety net: minimum-length fallback --
    if len(q) < 6:
        q = _extract_keywords_fallback(original)

    return q


def _extract_keywords_fallback(text: str) -> str:
    """Extract meaningful keywords from text when normalization over-strips.

    Pulls out capitalized words (likely proper nouns / tickers / acronyms),
    CJK character runs, numbers with context, and known financial terms.
    """
    tokens: list[str] = []

    # Capitalized words (2+ chars) -- likely entity names, tickers
    stop_words = {
        "use", "the", "and", "for", "from", "with", "this", "that",
        "query", "search", "find", "site", "filter", "html", "pdf",
        "parse", "raw", "fetch", "url", "via", "not", "its", "restrict",
        "published", "official", "website", "investor", "relations",
    }
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9]{1,}\b", text):
        word = m.group()
        if word.lower() not in stop_words:
            tokens.append(word)

    # CJK runs (Chinese/Japanese/Korean characters)
    for m in re.finditer(r"[\u4e00-\u9fff\u3400-\u4dbf]{2,}", text):
        tokens.append(m.group())

    # Year and quarter patterns
    for m in re.finditer(r"Q[1-4]\s*\d{4}|\b20\d{2}\b", text):
        tokens.append(m.group())

    # Domain-profile vocabulary (replaces hardcoded financial_terms list)
    text_lower = text.lower()
    for term in get_domain_search_vocabulary(text):
        if term.lower() in text_lower:
            tokens.append(term)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    result = " ".join(unique[:12])
    return result if len(result) >= 4 else text[:80]

def _derive_focus_terms(subtask_description: str, *, user_request: str = "") -> list[str]:
    """Derive keyword-oriented focus terms for fetch_url evidence extraction.

    Ensures that after fetching an official filing, we can point to exact passages
    for the fields the subtask is trying to verify (R&D, operating cash flow, pipeline, etc.).

    These terms drive `_collect_evidence_snippets()` which scans the FULL
    extracted_text (not just the excerpt) and picks the best 220-char window
    around each match.  The evidence snippets are the primary citation evidence,
    so getting the right focus terms is critical.
    """
    text = f"{user_request}\n{subtask_description}".lower()
    terms: list[str] = []

    def add(*xs: str) -> None:
        for x in xs:
            x = (x or "").strip()
            if x and x not in terms:
                terms.append(x)

    # ── Domain-profile-driven focus terms ──────────────────────────────────
    # Replaces the hardcoded financial if-branches.
    # Each DomainProfile defines FocusTermRules (trigger → terms).
    # detect_domain_profile() selects the right profile at runtime, so adding
    # a new domain only requires editing domain_profile.py.
    profile = detect_domain_profile(text)
    for term in profile.get_focus_terms(text):
        add(term)

    # Dynamic: extract capitalized product/brand names from the subtask description
    # so we don't hardcode any specific drug/company names.
    import re as _re
    for m in _re.finditer(r"\b([A-Z][A-Za-z0-9]{2,}(?:®)?)\b", subtask_description):
        candidate = m.group(1)
        if candidate.lower() in {
            "the", "and", "for", "with", "from", "not", "use", "search",
            "fetch", "web", "url", "sec", "pdf", "html", "api", "get",
            "phase", "step", "note", "item", "form", "key", "new",
        }:
            continue
        add(candidate)

    # Keep list reasonable — evidence_snippets selects top 8 terms anyway
    return terms[:15]


# ═══════════════════════════════════════════════════════════════
# Graph routing
# ═══════════════════════════════════════════════════════════════


def should_continue_executing(state: GraphState) -> str:
    """Determine if execution should continue.

    Args:
        state: Current graph state.

    Returns:
        Next node: "executor" to continue, "critic" to review, "end" to stop.
    """
    status = state.get("status", "")
    subtasks = state.get("subtasks", [])
    current_idx = state.get("current_subtask_index", 0)
    metrics = state.get("metrics", {})

    _route_msg = (
        f"[ROUTE] should_continue_executing: status={status!r}, "
        f"current_idx={current_idx}, total_subtasks={len(subtasks)}, "
        f"tokens={metrics.get('total_tokens', 0)}, "
        f"steps={metrics.get('step_count', 0)}, "
        f"tools={metrics.get('tool_call_count', 0)}"
    )
    logger.info(_route_msg)
    _safe_debug_print(_route_msg)

    if status == "reviewing":
        dest = "[ROUTE] → critic (status=reviewing)"
        logger.info(dest); _safe_debug_print(dest)
        return "critic"

    if status in ("failed", "completed", "cancelled"):
        dest = f"[ROUTE] → end (status={status})"
        logger.info(dest); _safe_debug_print(dest)
        return "end"

    # Check if there are more subtasks
    if current_idx < len(subtasks):
        dest = f"[ROUTE] → executor (subtask {current_idx}/{len(subtasks)} pending)"
        logger.info(dest); _safe_debug_print(dest)
        return "executor"

    dest = "[ROUTE] → critic (all subtasks done)"
    logger.info(dest); _safe_debug_print(dest)
    return "critic"
