"""Reporter agent for generating final reports with verified citations."""

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_engine.agents.state import GraphState
from agent_engine.agents.entity_resolver import resolve_entity_profile
from agent_engine.config import get_settings
from agent_engine.llm import get_provider
from agent_engine.tools.builtin.content_extract import (
    extract_main_text_from_html,
    extract_pdf_text_from_bytes,
)
from agent_engine.tools.builtin.web_search import (
    is_text_safe_for_llm,
    sanitize_text_for_llm,
)

logger = logging.getLogger(__name__)

# Maximum number of citations to verify in parallel
MAX_VERIFY_CONCURRENCY = 8
# Timeout for URL verification/fetch
VERIFY_TIMEOUT = 15.0
# Maximum content to fetch per URL.
# Financial tables in SEC 10-K / HKEX annual reports appear at 5k-20k+ chars.
# 10000 covers most balance sheets and income statements without blowing token budget.
MAX_FETCH_CONTENT = 10000
# Maximum citations to pass to LLM to avoid confusion
MAX_CITATIONS_FOR_LLM = 30
# Minimum citations to reserve per subtask (ensures each subtask has representation)
MIN_CITATIONS_PER_SUBTASK = 1
# Maximum non-numeric case/event citation checks to run via LLM
MAX_CASE_AUDITS = 25
# Paragraph chunk size for one LLM citation-audit call
CASE_AUDIT_PARAGRAPHS_PER_CALL = 3
# Max source chars per citation in batch audit prompt
MAX_BATCH_AUDIT_SOURCE_CHARS = 1800
# Max chars of per-source evidence text passed into report-generation prompt
MAX_SOURCE_EVIDENCE_CHARS = 1800
# Max bullet lines retained from evidence snippets section
MAX_SOURCE_EVIDENCE_LINES = 8
# Reporter generation retry policy (LLM-only; no fallback report synthesis)
REPORT_LLM_MAX_ATTEMPTS = 3
REPORT_RETRY_BACKOFF_SECONDS = 0.8


REPORTER_SYSTEM_PROMPT = """You are a professional report writer with strict academic integrity.

## Your Task
Generate a final report based ONLY on the verified source content provided below.

## Original Request
{user_request}

## Entity Resolution
{entity_resolution}

## PRIMARY DATA SOURCE (use ONLY this for numbers and facts)
## Verified Sources (evidence snippets/summaries)
{verified_sources_text}

## STRUCTURAL GUIDE ONLY (do NOT extract numbers from this section)
## Execution Results Summary
The following summary describes WHAT subtasks were completed and their overall status.
It is provided ONLY to help you understand the report structure and topic coverage.
>>> ALL NUMBERS HAVE BEEN REDACTED FROM THIS SECTION. <<<
>>> You MUST look up every number, statistic, and data point in the Verified Sources above. <<<
{results_summary}

## STRICT ANTI-HALLUCINATION RULES:

1. **ZERO TOLERANCE for fabricated data**: Every number, percentage, statistic, date,
   ranking, ratio, and quantitative claim MUST be directly copy-pasted from the
   "Verified Sources" section above. If you cannot find a number in any Verified Source,
   you MUST NOT include it in the report, not even approximately.

2. **Citation format**: Use the SAME source IDs [N] as shown in the Verified Sources
   list above. These IDs are stable and will NOT be re-numbered.

3. **CRITICAL Citation-content consistency**:
   - Before writing [N] after a claim, you MUST verify that source [N]'s evidence text
     actually contains the EXACT data or fact you are citing.
   - DO NOT guess citation numbers. If you are unsure which source supports a claim,
     DO NOT add a citation number.
   - Example of WRONG citation: writing "revenue was $10B [5]" when source [5]
     does not mention "$10B" or "revenue" at all.
   - Example of CORRECT citation: writing "revenue was $10B [5]" when source [5]'s
     evidence text explicitly states "$10B in revenue".

4. **No unverifiable claims**: If a piece of information cannot be traced to a specific
   verified source, either:
   - Omit it entirely, OR
   - Clearly mark it as "(based on public information)" WITHOUT a citation number

5. **References section**: At the end, include a "## References" section. Format:
   [N] Title - URL
   Only list sources you actually cited in the report body. Do NOT list sources you didn't cite.

6. **Language**: Write in the same language as the user's original request.

7. **Inaccessible sources**: Sources marked as UNVERIFIED should NOT be cited.
   Only cite sources marked as VERIFIED.

8. **Fewer but accurate citations**: It is MUCH BETTER to cite 5 correct sources
   than to cite 30 sources with mismatches. Quality over quantity.

9. **Official-source priority for core conclusions**:
   - Core quantitative conclusions MUST prioritize regulator/company-official sources.
   - Third-party media can be used only as supplementary context.

10. **CRITICAL QUOTE ORIGINAL VALUES WITHOUT CONVERSION**:
   - Quote data in the EXACT unit and currency as written in the source.
   - NEVER convert between currencies or measurement scales.
   - Always state the ORIGINAL value and unit first.

{domain_reporter_hints}

11. **Cross-check rule**: For every sentence that contains a number:
    - Identify which Verified Source [N] contains that number.
    - Confirm the number AND its unit/currency appear in that source's evidence text verbatim.
    - If you cannot confirm, DELETE the sentence or replace the number with
      a qualitative description (e.g., "significant growth" instead of "45% growth").

Output the report in Markdown format."""


def _redact_numbers(text: str) -> str:
    """Redact significant numbers from text to prevent the Reporter from
    copying ungrounded numbers from the Executor's raw analysis.

    Targets monetary values, percentages, and large standalone numbers.
    Preserves stock codes (e.g., 6160.HK), small identifiers, and
    contextual references that are not data claims.

    Args:
        text: Raw text that may contain unverified numbers.

    Returns:
        Text with significant numerical values replaced by [REDACTED].
    """
    # Monetary: $1,234.56 million / $1.3B / USD 100
    # Note: include both currency symbols and common ISO currency tags.
    text = re.sub(
        r'(?:US\$|U\$|\$|USD|CNY|RMB|HKD|EUR|GBP|JPY)\s*[\d,.]+(?:\s*(?:million|billion|trillion|M|B|K|thousand))?',
        '[REDACTED]',
        text,
        flags=re.IGNORECASE,
    )
    # Percentages: 45.2%, 45%
    text = re.sub(r'[\d,]+(?:\.\d+)?\s*%', '[REDACTED]%', text)
    # Numbers with common CJK units (for example values using Chinese unit markers)
    text = re.sub(
        r'[\d,]+(?:\.\d+)?\s*(?:\u4e07\u4ebf|\u4ebf|\u4e07|\u5343|\u767e)',
        '[REDACTED]',
        text,
    )
    # Standalone large numbers (3+ significant digits, not part of stock codes/IDs)
    # Preserve: 6160.HK, subtask_1, Source [3], NCT04993390, bgne-20241231
    # Target:  1234, 2,458,779, 10.5 (as standalone claims)
    text = re.sub(
        r'(?<!\[)(?<![.\w])\d{3}[\d,]*(?:\.\d+)?(?![.\w\-])(?!\])',
        '[REDACTED_NUM]',
        text,
    )
    return text


async def reporter_node(state: GraphState) -> dict[str, Any]:
    """Reporter node that generates report and performs post citation checks.

    Steps:
    1. Use executor-provided citation summaries/snippets directly (no URL re-verify)
    2. Build report from prioritized citations
    3. Post-check report claim-source consistency and fix mismatches

    Args:
        state: Current graph state.

    Returns:
        Updated state with final_report and updated citations.
    """
    citations = list(state.get("citations", []))
    subtasks = state.get("subtasks", [])

    # Step 1: Use citations as produced by executor/previous stages.
    # Policy: do not re-verify URL accessibility at report stage.
    prepared_citations = citations
    evidence_ready_count = sum(
        1
        for c in prepared_citations
        if str(c.get("fetched_content", "") or c.get("snippet", "")).strip()
    )
    logger.info(
        "[Reporter] Using %d citations directly; evidence-ready=%d",
        len(prepared_citations),
        evidence_ready_count,
    )

    # Step 2: Select and prioritize citations, then build text
    prioritized_citations = _prioritize_citations(prepared_citations, MAX_CITATIONS_FOR_LLM)
    prioritized_evidence_ready_count = sum(
        1
        for c in prioritized_citations
        if str(c.get("fetched_content", "") or c.get("snippet", "")).strip()
    )

    # Step 4: Build results summary (STRUCTURAL GUIDE ONLY)
    # Anti-hallucination: we REDACT all numbers from the results_summary so
    # the Reporter LLM cannot copy ungrounded numbers from the Executor's
    # raw analysis text.  The Reporter must look up every data point in the
    # Verified Sources section instead.
    summary_parts = []
    for subtask in subtasks:
        summary_parts.append(f"### Subtask: {subtask.get('description', '')}")
        summary_parts.append(f"Status: {subtask.get('status', 'unknown')}")
        result = subtask.get("result")
        if result:
            if isinstance(result, dict):
                # Only include topic/structure info, not raw analysis with numbers
                analysis = result.get("analysis", "")
                if analysis:
                    result_str = _redact_numbers(str(analysis)[:1500])
                else:
                    result_str = "(analysis empty)"
            else:
                result_str = _redact_numbers(str(result)[:1500])
            summary_parts.append(f"Result (numbers redacted):\n{result_str}")
        summary_parts.append("")

    # Sanitize results summary to prevent content moderation triggers
    results_summary = sanitize_text_for_llm("\n".join(summary_parts))

    # Step 5: Generate report with LLM
    # Uses the reporter-specific model (typically the best available model,
    # since the report is the user-facing deliverable).
    settings = get_settings()
    provider = get_provider(model=settings.reporter_model)

    # Resolve entity profile and domain profile for hints injection
    from agent_engine.agents.domain_profile import detect_domain_profile as _detect_dp
    _dp = _detect_dp(state.get("user_request", ""))
    domain_reporter_hints = _dp.reporter_hints or ""

    profile = resolve_entity_profile(state.get("user_request", ""))
    if profile.get("canonical_name"):
        entity_resolution = (
            f"Canonical entity: {profile['canonical_name']}\n"
            f"Aliases in historical sources: {', '.join(profile.get('aliases', [])[:5])}\n"
            f"Timeline note: {profile.get('timeline_note', '')}\n"
            "When sources use old names, treat them as the same entity but make the "
            "name transition explicit in the report."
        )
    else:
        entity_resolution = "No special entity alias mapping detected."

    # P8: Quality gate - inject data-quality-aware credibility instructions
    # Three-tier system:
    #   good    (grounding >=65%): minimal disclaimer, write normally
    #   partial (35-65%): note specific ungrounded claims, add short caveat
    #   poor    (< 35%): generate a "data gap" report: be explicit about what
    #                    could NOT be verified and focus on what IS available
    data_quality_level = state.get("data_quality_level", "good")
    critic_feedback = state.get("critic_feedback") or {}
    critic_confidence = critic_feedback.get("confidence", 1.0)
    critic_passed = (
        critic_feedback.get("is_correct", True)
        and critic_feedback.get("is_complete", True)
    )

    # Collect specific ungrounded claims from Validator (P2/P3)
    ungrounded_claims: list[str] = []
    for ec in state.get("evidence_claims", []):
        if not ec.get("grounded") and ec.get("claim"):
            ungrounded_claims.append(ec["claim"])
    for st in subtasks:
        val = st.get("validation") or {}
        for det in val.get("details", []):
            if not det.get("grounded") and det.get("claim"):
                ungrounded_claims.append(det["claim"])
    # Deduplicate
    seen_claims: set[str] = set()
    ungrounded_claims = [
        c for c in ungrounded_claims
        if not (c in seen_claims or seen_claims.add(c))  # type: ignore[func-returns-value]
    ]

    credibility_instruction = ""

    if data_quality_level == "poor":
        # Graceful degradation: tell reporter to focus on what IS available
        credibility_instruction = (
            f"\n\n## [Data Quality Warning - Degraded Report Mode]\n"
            f"System verification indicates many requested data points could not be grounded in fetched sources "
            f"(overall grounding < 35%, critic confidence={critic_confidence:.0%}).\n\n"
            "Rules in degraded mode:\n"
            "1. Do not fill in unverified numbers from memory. Use [DATA PENDING VERIFICATION] when needed.\n"
            "2. Add a dedicated [Data Gaps] section listing missing key datapoints and official channels to verify.\n"
            "3. Keep writing fully grounded facts with citations where evidence exists.\n"
            "4. Do not include any quantitative claim that does not appear in Verified Sources.\n"
        )
    elif data_quality_level == "partial" or not critic_passed or critic_confidence < 0.35:
        # Partial mode: only flag specific ungrounded claims
        ungrounded_str = ""
        if ungrounded_claims:
            items = "\n".join(f"- {c[:120]}" for c in ungrounded_claims[:12])
            ungrounded_str = (
                "\n\n## The following claims are currently ungrounded in fetched sources:\n"
                f"{items}\n"
                "Mark only these items as [CANNOT VERIFY FROM SOURCE]. "
                "Other claims with evidence in Verified Sources can be cited normally."
            )
        if ungrounded_claims or critic_confidence < 0.35:
            credibility_instruction = (
                f"\n\n## Note: system confidence={critic_confidence:.2f}. "
                "Add a brief source-quality note at the beginning (1-2 sentences)."
                f"{ungrounded_str}"
            )
    # data_quality_level == "good": no extra instruction needed
    base_user_prompt = (
        "Generate the final report using ONLY the verified sources above.\n"
        "Every claim and case in the report must be traceable to source evidence text.\n"
        "Do not fabricate unsupported facts.\n"
        "For quantitative values, keep original units/currency as written in sources.\n"
        f"{credibility_instruction}"
    )

    # Update metrics
    metrics = state.get("metrics", {})

    # LLM-only generation policy: no local fallback report synthesis.
    # If full prompt fails, retry with progressively lighter prompt budgets.
    if prioritized_citations:
        attempt_plans = [
            {
                "name": "full",
                "citations": prioritized_citations[: min(MAX_CITATIONS_FOR_LLM, len(prioritized_citations))],
                "summary": results_summary,
                "title_only": False,
                "extra_user_instruction": "",
            },
            {
                "name": "compact",
                "citations": prioritized_citations[: min(16, len(prioritized_citations))],
                "summary": results_summary[:3000],
                "title_only": False,
                "extra_user_instruction": (
                    "\nPlease keep the report concise and focus on core conclusions only."
                ),
            },
            {
                "name": "safe_mode",
                "citations": prioritized_citations[: min(8, len(prioritized_citations))],
                "summary": results_summary[:1600],
                "title_only": True,
                "extra_user_instruction": (
                    "\nIf some data cannot be verified from provided sources, omit it instead of guessing."
                ),
            },
        ]
    else:
        attempt_plans = [
            {
                "name": "safe_mode_no_sources",
                "citations": [],
                "summary": results_summary[:1200],
                "title_only": True,
                "extra_user_instruction": (
                    "\nNo verified sources are available. Provide a transparent data-gap report."
                ),
            }
        ]

    attempt_plans = attempt_plans[:REPORT_LLM_MAX_ATTEMPTS]
    final_report = ""
    generation_errors: list[str] = []

    for idx, plan in enumerate(attempt_plans, start=1):
        attempt_sources_text = (
            _build_title_only_sources_text(plan["citations"])
            if plan["title_only"]
            else _build_verified_sources_text(plan["citations"])
        )
        attempt_system_msg = SystemMessage(
            content=REPORTER_SYSTEM_PROMPT.format(
                user_request=state.get("user_request", ""),
                entity_resolution=entity_resolution,
                verified_sources_text=attempt_sources_text,
                results_summary=plan["summary"],
                domain_reporter_hints=domain_reporter_hints,
            )
        )
        attempt_user_msg = HumanMessage(
            content=base_user_prompt + plan["extra_user_instruction"]
        )
        try:
            response = await provider.invoke([attempt_system_msg, attempt_user_msg])
            metrics["input_tokens"] = metrics.get("input_tokens", 0) + response.input_tokens
            metrics["output_tokens"] = metrics.get("output_tokens", 0) + response.output_tokens
            metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
            metrics["step_count"] = metrics.get("step_count", 0) + 1

            candidate = str(response.content or "").strip()
            if candidate:
                final_report = candidate
                if idx > 1:
                    logger.warning(
                        "[Reporter] Report generated on retry attempt %d/%d (%s)",
                        idx,
                        len(attempt_plans),
                        plan["name"],
                    )
                break

            generation_errors.append(
                f"attempt {idx} ({plan['name']}): empty LLM response"
            )
            logger.warning(
                "[Reporter] Empty response on attempt %d/%d (%s)",
                idx,
                len(attempt_plans),
                plan["name"],
            )
        except Exception as e:
            metrics["step_count"] = metrics.get("step_count", 0) + 1
            generation_errors.append(
                f"attempt {idx} ({plan['name']}): {str(e)[:240]}"
            )
            logger.warning(
                "[Reporter] LLM generation attempt %d/%d failed (%s): %s",
                idx,
                len(attempt_plans),
                plan["name"],
                str(e)[:240],
            )

        if idx < len(attempt_plans):
            await asyncio.sleep(REPORT_RETRY_BACKOFF_SECONDS * idx)

    if not str(final_report).strip():
        raise RuntimeError(
            "[Reporter] LLM report generation failed after retries; "
            f"errors={generation_errors[:3]}"
        )

    # Step 6: Verify citation-content consistency
    citation_map = {
        c["id"]: c
        for c in prioritized_citations
        if _build_case_audit_source(c).strip()
    }
    mismatch_count = 0
    try:
        final_report, mismatch_count = await _verify_and_fix_citations(
            final_report,
            citation_map,
            provider,
            metrics,
        )
    except Exception as e:
        logger.error(
            f"[Reporter] Citation consistency check failed, keeping report unchanged: {e}",
            exc_info=True,
        )

    if mismatch_count > 0:
        logger.warning(
            f"[Reporter] Fixed {mismatch_count} citation-content mismatches in report"
        )

    # Step 7: Ensure references section is consistent
    # Extract actually-used citation IDs from the report body (exclude references section)
    # Only match true inline citations like "[1]" at the end of a sentence/clause,
    # not things like "Source[9]" or "[N] Title - URL" in references
    try:
        report_body_for_ids = final_report
        ref_markers = [
            "## References", "## Reference", "## Sources", "## Bibliography",
            "## 参考资料", "## 参考文献",
        ]
        for marker in ref_markers:
            idx = report_body_for_ids.find(marker)
            if idx > 0:
                report_body_for_ids = report_body_for_ids[:idx]
                break

        inline_citation_ids = set(
            int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", report_body_for_ids)
        )

        # Rebuild references section to only include actually cited sources
        user_request = state.get("user_request", "")
        final_report = _rebuild_references_section(
            final_report, citation_map, inline_citation_ids, user_request
        )
    except Exception as e:
        logger.error(
            f"[Reporter] Reference rebuild failed, returning report without rebuild: {e}",
            exc_info=True,
        )

    if not str(final_report).strip():
        raise RuntimeError(
            "[Reporter] Final report became empty after post-processing."
        )

    ai_msg = AIMessage(
        content=f"Final report generated. {prioritized_evidence_ready_count} evidence-ready sources, "
        f"{len(prioritized_citations) - prioritized_evidence_ready_count} sources with limited/empty evidence, "
        f"{mismatch_count} citation mismatches fixed."
    )

    # Return the prioritized citations with their original Executor-assigned IDs.
    # IDs are stable across the pipeline to prevent citation mismatches.
    return {
        "final_report": final_report,
        "citations": prioritized_citations,
        "messages": [ai_msg],
        "metrics": metrics,
    }


# Citation verification helpers


async def _verify_and_fetch_citations(
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify each citation URL and fetch content.

    For each citation:
    - If already verified (e.g. from fetch_url), keep it
    - Otherwise, attempt HTTP GET to verify accessibility
    - Fetch text content excerpt for verified URLs
    - Mark inaccessible URLs as unverified

    Args:
        citations: List of citation dicts.

    Returns:
        Updated citations with verified flag and fetched_content.
    """
    if not citations:
        return []

    semaphore = asyncio.Semaphore(MAX_VERIFY_CONCURRENCY)

    async def verify_one(citation: dict[str, Any]) -> dict[str, Any]:
        c = dict(citation)  # copy
        c["source_tier"] = c.get("source_tier") or _source_tier(c.get("url", ""))

        # Already verified (e.g. came from fetch_url tool with content)
        if (
            c.get("verified")
            and c.get("fetched_content")
            and c.get("is_citable", True)
            and not c.get("not_citable_reason")
            and not str(c.get("fetched_content", "")).lstrip().startswith("%PDF")
        ):
            # Safety check even for pre-verified content
            if not is_text_safe_for_llm(c["fetched_content"]):
                logger.warning(
                    f"[Reporter] Unsafe content in pre-verified citation "
                    f"{c.get('url', '')} - discarding"
                )
                c["fetched_content"] = ""
                c["verified"] = False
                c["is_citable"] = False
                c["not_citable_reason"] = "unsafe content detected"
                return c
            return c

        url = c.get("url", "")
        if not url:
            c["verified"] = False
            return c

        # Normalize URL: add protocol if missing
        url = _normalize_url(url)
        c["url"] = url  # Update the citation with normalized URL
        c["source_tier"] = c.get("source_tier") or _source_tier(url)

        async with semaphore:
            try:
                async with httpx.AsyncClient(
                    timeout=VERIFY_TIMEOUT,
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                ) as client:
                    resp = await client.get(url)

                    if resp.status_code < 400:
                        c["http_status"] = resp.status_code
                        content_type = (resp.headers.get("content-type", "") or "").lower()
                        c["content_type"] = content_type

                        # Extract readable content for citation
                        # - HTML: extract main text
                        # - PDF: extract text via pymupdf when available
                        # - Other binary: mark non-citable
                        is_pdf = "application/pdf" in content_type or url.lower().endswith(".pdf")
                        is_html = "text/html" in content_type
                        is_text = content_type.startswith("text/") or "charset=" in content_type

                        if not c.get("fetched_content"):
                            if is_pdf:
                                pdf_res = extract_pdf_text_from_bytes(resp.content, max_excerpt_len=MAX_FETCH_CONTENT)
                                c["fetched_content"] = pdf_res.excerpt
                                c["is_citable"] = pdf_res.is_citable
                                c["not_citable_reason"] = pdf_res.not_citable_reason
                                c["quality_score"] = pdf_res.quality_score
                                c["quality_issues"] = pdf_res.quality_issues
                                c["verified"] = bool(pdf_res.is_citable)
                            elif is_html or is_text:
                                html_res = extract_main_text_from_html(
                                    resp.text,
                                    url=str(resp.url),
                                    max_excerpt_len=MAX_FETCH_CONTENT,
                                )
                                c["fetched_content"] = html_res.excerpt
                                c["is_citable"] = html_res.is_citable
                                c["not_citable_reason"] = html_res.not_citable_reason
                                c["quality_score"] = html_res.quality_score
                                c["quality_issues"] = html_res.quality_issues
                                c["verified"] = bool(html_res.is_citable)
                            else:
                                c["fetched_content"] = ""
                                c["is_citable"] = False
                                c["not_citable_reason"] = f"non-text content-type: {content_type}"
                                c["quality_score"] = 0.1
                                c["quality_issues"] = [c["not_citable_reason"]]
                                c["verified"] = False
                        else:
                            # If caller supplied fetched_content, still respect is_citable/not_citable flags
                            c["verified"] = bool(c.get("is_citable", True)) and not c.get("not_citable_reason")

                        # Try to extract a better title from HTML if missing
                        if not c.get("title") and "html" in content_type:
                            title_match = re.search(
                                r"<title[^>]*>(.*?)</title>",
                                resp.text[:5000],
                                re.IGNORECASE | re.DOTALL,
                            )
                            if title_match:
                                c["title"] = title_match.group(1).strip()[:200]

                        # Content safety check
                        # Fetched content may contain unsafe material (porn, violence)
                        # that triggers DashScope content moderation (400 error) when
                        # included in the Reporter prompt.  Check and discard.
                        fetched = c.get("fetched_content", "")
                        if fetched and not is_text_safe_for_llm(fetched):
                            logger.warning(
                                f"[Reporter] Unsafe content detected in {url} - "
                                f"marking as not citable"
                            )
                            c["fetched_content"] = ""
                            c["verified"] = False
                            c["is_citable"] = False
                            c["not_citable_reason"] = "unsafe content detected"
                            c["quality_score"] = 0.0

                    else:
                        c["verified"] = False
                        c["http_status"] = resp.status_code
                        c["fetched_content"] = ""
                        logger.warning(f"[Reporter] URL inaccessible ({resp.status_code}): {url}")

            except httpx.TimeoutException:
                c["verified"] = False
                c["http_status"] = None
                c["fetched_content"] = ""
                logger.warning(f"[Reporter] URL timeout: {url}")
            except Exception as e:
                c["verified"] = False
                c["http_status"] = None
                c["fetched_content"] = ""
                logger.warning(f"[Reporter] URL error: {url} - {e}")

        return c

    tasks = [verify_one(c) for c in citations]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified_citations = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            c = dict(citations[i])
            c["verified"] = False
            c["fetched_content"] = ""
            verified_citations.append(c)
        else:
            verified_citations.append(r)

    # Preserve original citation IDs assigned by Executor.
    # Re-numbering would cause the Reporter LLM to use stale IDs from
    # the Execution Results Summary, leading to citation mismatches.
    return verified_citations


def _html_to_text(html: str, max_len: int = 3000) -> str:
    """Convert HTML to clean text.

    Args:
        html: Raw HTML string.
        max_len: Maximum output length.

    Returns:
        Clean text content.
    """
    if not html:
        return ""
    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove nav/header/footer blocks (common noise)
    text = re.sub(r"<(nav|header|footer)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r"<(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
    )
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()
    return text[:max_len]


def _build_verified_sources_text(citations: list[dict[str, Any]]) -> str:
    """Build the verified sources text block for the LLM prompt.

    Args:
        citations: List of citation dicts with verification info.

    Returns:
        Formatted text with source content for each citation.
    """
    if not citations:
        return "No sources available. Generate the report based on the execution results only."

    lines = []
    for c in citations:
        has_evidence = bool(
            str(c.get("fetched_content", "") or c.get("snippet", "")).strip()
        )
        status = "EVIDENCE_READY" if has_evidence else "NO_EVIDENCE"
        title = sanitize_text_for_llm(c.get("title", "") or "(no title)")
        url = c.get("url", "")
        content = c.get("fetched_content", "")
        snippet = c.get("snippet", "")

        lines.append(f"--- Source [{c['id']}] - {status} ---")
        lines.append(f"Title: {title}")
        lines.append(f"URL: {url}")
        lines.append(_compact_source_evidence_text(content, snippet))
        lines.append("")

    return "\n".join(lines)


def _build_title_only_sources_text(citations: list[dict[str, Any]]) -> str:
    """Build a minimal source block with title/url only for safe-mode retries."""
    if not citations:
        return "No sources available."

    lines = []
    for c in citations:
        title = sanitize_text_for_llm(c.get("title", "") or "(no title)")
        url = c.get("url", "")
        lines.append(f"--- Source [{c['id']}] ---")
        lines.append(f"Title: {title}")
        lines.append(f"URL: {url}")
        lines.append("Evidence: (omitted in safe mode)")
        lines.append("")
    return "\n".join(lines)


def _compact_source_evidence_text(content: str, snippet: str) -> str:
    """Compact a source to evidence snippets/summary for prompt efficiency."""
    safe_content = sanitize_text_for_llm(content or "")
    safe_snippet = sanitize_text_for_llm(snippet or "")

    if safe_content:
        # Executor-generated fetched_content often starts with:
        # "Evidence snippets:\n- ...\n\nExcerpt:\n..."
        m = re.search(
            r"Evidence snippets:\s*(.*?)(?:\n\nExcerpt:|\Z)",
            safe_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            raw_lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
            evidence_lines = []
            for ln in raw_lines:
                if len(evidence_lines) >= MAX_SOURCE_EVIDENCE_LINES:
                    break
                line = ln if ln.startswith("-") else f"- {ln}"
                evidence_lines.append(line[:320])
            if evidence_lines:
                return "Evidence Snippets:\n" + "\n".join(evidence_lines)

        return "Source Summary:\n" + safe_content[:MAX_SOURCE_EVIDENCE_CHARS]

    if safe_snippet:
        return "Search Summary:\n" + safe_snippet[:600]

    return "Content: (not available)"

def _normalize_url(url: str) -> str:
    """Normalize URL by adding protocol if missing.

    Args:
        url: Raw URL string.

    Returns:
        Normalized URL with protocol.
    """
    if not url:
        return url

    url = url.strip()

    # Already has protocol
    if url.startswith(("http://", "https://")):
        return url

    # Try https first (most common), fallback to http if needed
    # For domains starting with www. or common TLDs, assume https
    if url.startswith("www.") or "." in url:
        return f"https://{url}"

    return url


def _source_tier(url: str) -> str:
    """Classify citation source tier from URL domain.

    Official domains are read from all loaded domain profiles; no hardcoding here.
    """
    domain = ""
    try:
        domain = (urlparse(url or "").netloc or "").lower()
    except Exception:
        domain = ""

    if not domain:
        return "unknown"

    # Read official domains from all loaded profiles (avoids hardcoding)
    try:
        from agent_engine.agents.domain_profile import _PROFILES as _dp
        official = tuple(d for p in _dp for d in p.official_domains)
    except Exception:
        official = ()

    if any(domain == d or domain.endswith(f".{d}") for d in official):
        return "regulator"
    if "investor" in domain or domain.endswith("ir"):
        return "company_ir"
    # Generic quality-signal media (not domain-specific)
    _trusted = ("reuters.com", "bloomberg.com", "wsj.com", "ft.com")
    if any(domain == d or domain.endswith(f".{d}") for d in _trusted):
        return "trusted_media"
    return "unknown"


def _prioritize_citations(
    citations: list[dict[str, Any]],
    max_count: int,
) -> list[dict[str, Any]]:
    """Prioritize citations with per-subtask quota to ensure coverage.

    Two-phase selection:
    1. Guarantee at least MIN_CITATIONS_PER_SUBTASK per subtask (by quality).
    2. Fill remaining slots by quality score.

    Quality scoring: verified + source_tier + content length + snippet + fetch_url.

    Args:
        citations: All citations (may include subtask_id for provenance).
        max_count: Maximum citations to include.

    Returns:
        Prioritized and limited citation list, preserving original IDs.
    """
    if len(citations) <= max_count:
        return citations

    def _score(c: dict[str, Any]) -> float:
        score = 0.0
        if c.get("verified"):
            score += 10
        tier = c.get("source_tier") or _source_tier(c.get("url", ""))
        score += {
            "regulator": 15,
            "company_ir": 12,
            "trusted_media": 5,
            "unknown": 0,
        }.get(tier, 0)
        if c.get("fetched_content"):
            content_len = len(c.get("fetched_content", ""))
            score += min(content_len / 100, 10)
        if c.get("snippet"):
            score += 3
        if c.get("source_tool") == "fetch_url":
            score += 5
        return score

    scored = [(_score(c), c) for c in citations]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: Guarantee at least MIN_CITATIONS_PER_SUBTASK per subtask
    guaranteed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    by_subtask: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for s, c in scored:
        sid = c.get("subtask_id") or ""
        by_subtask.setdefault(sid, []).append((s, c))

    for sid, items in by_subtask.items():
        for _, c in items[:MIN_CITATIONS_PER_SUBTASK]:
            cid = c.get("id")
            if cid is not None and cid not in seen_ids:
                guaranteed.append(dict(c))
                seen_ids.add(cid)

    # Phase 2: Fill remaining slots by quality (excluding already guaranteed)
    remaining = [c for _, c in scored if c.get("id") not in seen_ids]
    slots_left = max_count - len(guaranteed)
    extra = remaining[:slots_left]
    selected = guaranteed + [dict(c) for c in extra]

    logger.info(
        f"[Reporter] Reduced citations from {len(citations)} to {len(selected)} "
        f"(max={max_count}, guaranteed={len(guaranteed)} from {len(by_subtask)} subtasks)"
    )
    return selected


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Extract first JSON object from model output."""
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    if "```json" in text:
        try:
            block = text.split("```json", 1)[1].split("```", 1)[0]
            return json.loads(block.strip())
        except Exception:
            pass
    if "```" in text:
        try:
            block = text.split("```", 1)[1].split("```", 1)[0]
            if block.strip().startswith("{"):
                return json.loads(block.strip())
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _is_numeric_focused_claim(text: str) -> bool:
    """Heuristic: skip LLM case-check for number-heavy claims."""
    claim = str(text or "")
    if not claim.strip():
        return True
    digits = len(re.findall(r"\d", claim))
    letters = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", claim))
    number_tokens = len(re.findall(r"\d[\d,]*(?:\.\d+)?", claim))
    has_numeric_unit = bool(
        re.search(
            r"(%|USD|CNY|RMB|HKD|EUR|GBP|JPY|\$|million|billion|trillion|thousand|\u4e07|\u4ebf|\u5143)",
            claim,
            re.IGNORECASE,
        )
    )
    if number_tokens >= 3:
        return True
    if has_numeric_unit and digits >= max(3, letters // 2):
        return True
    return False


def _build_case_audit_source(citation: dict[str, Any], max_chars: int = 4000) -> str:
    """Build source text for case/event consistency auditing."""
    parts = []
    title = str(citation.get("title", "") or "").strip()
    if title:
        parts.append(f"Title: {title}")
    snippet = str(citation.get("snippet", "") or "").strip()
    if snippet:
        parts.append(f"Snippet: {snippet}")
    fetched = str(citation.get("fetched_content", "") or "").strip()
    if fetched:
        parts.append(f"Page Content: {fetched}")
    source_text = "\n".join(parts).strip()
    return sanitize_text_for_llm(source_text[:max_chars])


async def _llm_case_claim_supported(
    claim_text: str,
    citation_id: int,
    source_text: str,
    provider,
) -> tuple[str, float, str, int, int]:
    """LLM check for event/case consistency (ignores numeric exactness)."""
    if not source_text.strip():
        return "uncertain", 0.0, "empty source text", 0, 0

    system_msg = SystemMessage(
        content=(
            "You are a citation auditor. Verify ONLY qualitative event/case consistency "
            "between a claim and a cited source.\n"
            "IMPORTANT:\n"
            "- Ignore exact numbers, percentages, and unit/currency conversion differences.\n"
            "- Focus on whether the source supports the underlying event/fact/case statement.\n"
            "- If unclear, output uncertain (do NOT guess).\n"
            "Output JSON only: "
            '{"verdict":"supported|unsupported|uncertain","confidence":0.0,"reason":"..."}'
        )
    )
    user_msg = HumanMessage(
        content=(
            f"Claim: \"{claim_text}\"\n\n"
            f"Source [{citation_id}]:\n{source_text}\n\n"
            "Is the claim's underlying event/fact/case supported by this source? "
            "Output JSON only."
        )
    )
    try:
        response = await asyncio.wait_for(
            provider.invoke([system_msg, user_msg]),
            timeout=30,
        )
        parsed = _extract_json_object(response.content)
        verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
        if verdict not in {"supported", "unsupported", "uncertain"}:
            verdict = "uncertain"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        reason = str(parsed.get("reason", "") or "").strip()[:200]
        return (
            verdict,
            max(0.0, min(confidence, 1.0)),
            reason,
            int(getattr(response, "input_tokens", 0) or 0),
            int(getattr(response, "output_tokens", 0) or 0),
        )
    except Exception as e:
        logger.warning(
            f"[Reporter] LLM case audit failed for citation [{citation_id}]: {e}"
        )
        return "uncertain", 0.0, "llm audit failed", 0, 0


def _build_paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for paragraphs split by blank lines."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n", text or ""):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text or "")))
    return spans


def _paragraph_index_for_pos(spans: list[tuple[int, int]], pos: int) -> int:
    """Find paragraph index for absolute character position."""
    for i, (s, e) in enumerate(spans):
        if s <= pos <= e:
            return i
    return max(0, len(spans) - 1)


async def _llm_batch_case_claims_supported(
    *,
    chunk_text: str,
    claims: list[dict[str, Any]],
    source_by_id: dict[int, str],
    provider,
) -> tuple[dict[int, tuple[str, float, str]], int, int]:
    """Batch audit multiple claim-citation pairs in one LLM call.

    Returns:
        mapping: claim_idx -> (verdict, confidence, reason)
        input_tokens, output_tokens
    """
    if not claims:
        return {}, 0, 0

    sources_lines: list[str] = []
    for cid in sorted(source_by_id):
        src = sanitize_text_for_llm(str(source_by_id.get(cid, "") or ""))
        sources_lines.append(
            f"[{cid}] {src[:MAX_BATCH_AUDIT_SOURCE_CHARS]}"
        )

    claim_lines = []
    for item in claims:
        claim_lines.append(
            f'- idx={item["idx"]}, citation=[{item["citation_id"]}], claim="{item["claim_text"]}"'
        )

    system_msg = SystemMessage(
        content=(
            "You are a citation auditor. Audit claim-source consistency in batch.\n"
            "Focus on qualitative event/fact/case consistency.\n"
            "Ignore exact numeric/unit conversion differences.\n"
            "Output JSON only with this schema:\n"
            '{"decisions":[{"idx":0,"verdict":"supported|unsupported|uncertain","confidence":0.0,"reason":"..."}]}'
        )
    )
    user_msg = HumanMessage(
        content=(
            f"Paragraph Chunk:\n{sanitize_text_for_llm(chunk_text)[:2500]}\n\n"
            f"Claims:\n" + "\n".join(claim_lines) + "\n\n"
            f"Sources:\n" + "\n\n".join(sources_lines) + "\n\n"
            "Judge each claim against its cited source id and return JSON only."
        )
    )

    try:
        response = await asyncio.wait_for(
            provider.invoke([system_msg, user_msg]),
            timeout=35,
        )
        parsed = _extract_json_object(response.content)
        decisions = parsed.get("decisions", [])
        out: dict[int, tuple[str, float, str]] = {}
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("idx", -1))
                except Exception:
                    continue
                verdict = str(item.get("verdict", "uncertain")).strip().lower()
                if verdict not in {"supported", "unsupported", "uncertain"}:
                    verdict = "uncertain"
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                reason = str(item.get("reason", "") or "").strip()[:200]
                out[idx] = (verdict, max(0.0, min(confidence, 1.0)), reason)
        return (
            out,
            int(getattr(response, "input_tokens", 0) or 0),
            int(getattr(response, "output_tokens", 0) or 0),
        )
    except Exception as e:
        logger.warning(f"[Reporter] LLM batch case audit failed: {e}")
        return {}, 0, 0


async def _verify_and_fix_citations(
    report: str,
    citation_map: dict[int, dict[str, Any]],
    provider,
    metrics: dict[str, Any],
) -> tuple[str, int]:
    """Post-check citation consistency with hard checks + LLM case audit.

    Policy:
    - Hard checks: citation ID exists and source content is available.
    - Number-heavy claims: skip numeric locate/unit checks (trust executor stage).
    - Event/case claims: ask LLM to verify claim-source consistency.
    """
    if not citation_map:
        return report, 0

    citation_pattern = re.compile(r'([^.\]]{10,}?)\s*(\[(\d+)\])')
    matches = list(citation_pattern.finditer(report))
    if not matches:
        return report, 0

    mismatch_count = 0
    audited = 0
    remove_flags = [False] * len(matches)
    paragraph_spans = _build_paragraph_spans(report)
    claims_by_chunk: dict[int, list[dict[str, Any]]] = {}

    for midx, match in enumerate(matches):
        context_text = match.group(1).strip()
        citation_id = int(match.group(3))

        citation = citation_map.get(citation_id)
        if not citation:
            mismatch_count += 1
            remove_flags[midx] = True
            logger.warning(
                f"[Reporter] Citation [{citation_id}] not found in sources, removing"
            )
            continue

        source_text = _build_case_audit_source(citation)
        if not source_text.strip():
            mismatch_count += 1
            remove_flags[midx] = True
            logger.warning(
                f"[Reporter] Citation [{citation_id}] has no content to verify, removing"
            )
            continue

        if _is_numeric_focused_claim(context_text):
            continue

        if audited >= MAX_CASE_AUDITS:
            continue

        para_idx = _paragraph_index_for_pos(paragraph_spans, match.start())
        chunk_idx = para_idx // CASE_AUDIT_PARAGRAPHS_PER_CALL
        claims_by_chunk.setdefault(chunk_idx, []).append(
            {
                "idx": midx,
                "paragraph_idx": para_idx,
                "citation_id": citation_id,
                "claim_text": context_text,
                "source_text": source_text,
            }
        )

    for chunk_idx in sorted(claims_by_chunk):
        if audited >= MAX_CASE_AUDITS:
            break
        candidates = claims_by_chunk.get(chunk_idx, [])
        if not candidates:
            continue

        remaining = MAX_CASE_AUDITS - audited
        batch_candidates = candidates[:remaining]
        if not batch_candidates:
            continue

        chunk_para_start = chunk_idx * CASE_AUDIT_PARAGRAPHS_PER_CALL
        chunk_para_end = min(
            len(paragraph_spans) - 1,
            (chunk_idx + 1) * CASE_AUDIT_PARAGRAPHS_PER_CALL - 1,
        )
        chunk_start = paragraph_spans[chunk_para_start][0]
        chunk_end = paragraph_spans[chunk_para_end][1]
        chunk_text = report[chunk_start:chunk_end]

        batch_claims = [
            {
                "idx": item["idx"],
                "citation_id": item["citation_id"],
                "claim_text": item["claim_text"],
            }
            for item in batch_candidates
        ]
        source_by_id = {}
        for item in batch_candidates:
            cid = int(item["citation_id"])
            if cid not in source_by_id:
                source_by_id[cid] = item["source_text"]

        decisions, in_tok, out_tok = await _llm_batch_case_claims_supported(
            chunk_text=chunk_text,
            claims=batch_claims,
            source_by_id=source_by_id,
            provider=provider,
        )
        metrics["input_tokens"] = metrics.get("input_tokens", 0) + in_tok
        metrics["output_tokens"] = metrics.get("output_tokens", 0) + out_tok
        metrics["total_tokens"] = (
            metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
        )
        audited += len(batch_candidates)

        for item in batch_candidates:
            midx = int(item["idx"])
            citation_id = int(item["citation_id"])
            verdict, confidence, reason = decisions.get(
                midx,
                ("uncertain", 0.0, "no decision"),
            )
            if verdict == "unsupported" and confidence >= 0.55:
                remove_flags[midx] = True
                mismatch_count += 1
                logger.warning(
                    "[Reporter] LLM batch-audit removed citation [%d] (confidence=%.2f, reason=%s)",
                    citation_id,
                    confidence,
                    reason or "n/a",
                )

    rebuilt_parts: list[str] = []
    cursor = 0
    for midx, match in enumerate(matches):
        rebuilt_parts.append(report[cursor : match.start()])
        cursor = match.end()
        context_text = match.group(1).strip()
        full_citation = match.group(2)
        if remove_flags[midx]:
            rebuilt_parts.append(context_text)
        else:
            rebuilt_parts.append(context_text + " " + full_citation)
    rebuilt_parts.append(report[cursor:])

    if audited >= MAX_CASE_AUDITS:
        logger.info(
            f"[Reporter] Case-audit limit reached ({MAX_CASE_AUDITS}); remaining citations kept unchanged"
        )
    return "".join(rebuilt_parts), mismatch_count


def _extract_claim_keywords(text: str) -> list[str]:
    """Extract simple keywords from a claim for lightweight checks."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "and", "or", "but", "not", "this", "that", "these", "those",
    }
    words = re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", text or "")
    out: list[str] = []
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        if w.lower() in stop_words:
            continue
        out.append(w)
        if len(out) >= 10:
            break
    return out


_CURRENCY_TOKENS: list[tuple[str, list[str]]] = [
    ("USD", ["$", "usd", "us$", "dollar"]),
    ("CNY", ["cny", "rmb", "yuan", "renminbi"]),
    ("EUR", ["eur", "euro"]),
    ("GBP", ["gbp", "pound"]),
    ("JPY", ["jpy", "yen"]),
    ("HKD", ["hkd", "hk$"]),
]


def _detect_currency(text: str) -> str | None:
    """Detect which currency family a text uses."""
    text_lower = (text or "").lower()
    for family, tokens in _CURRENCY_TOKENS:
        for token in tokens:
            if token in text_lower:
                return family
    return None


def _check_claim_unit_vs_source(
    claim_text: str,
    source_content: str,
) -> str | None:
    """Lightweight currency mismatch checker (best-effort)."""
    claim_currency = _detect_currency(claim_text)
    if not claim_currency:
        return None
    source_currency = _detect_currency(source_content)
    if source_currency and source_currency != claim_currency:
        return f"claim uses {claim_currency} but source uses {source_currency}"
    return None

def _rebuild_references_section(
    report: str,
    citation_map: dict[int, dict[str, Any]],
    used_citation_ids: set[int],
    user_request: str = "",
) -> str:
    """Rebuild references section to include only body-used citations."""
    # Remove existing references section variants (English + Chinese).
    # Use [\s\S]* to match content on same line or following lines (LLM may output
    # "## References [4]..." without newline, causing duplicate References/参考资料).
    ref_patterns = [
        r"\n## References\s*[\s\S]*",
        r"\n## Reference\s*[\s\S]*",
        r"\n## Sources\s*[\s\S]*",
        r"\n## Bibliography\s*[\s\S]*",
        r"\n## 参考资料\s*[\s\S]*",
        r"\n## 参考文献\s*[\s\S]*",
    ]
    report_body = report
    for pattern in ref_patterns:
        report_body = re.sub(pattern, "", report_body, flags=re.DOTALL | re.IGNORECASE)

    body_citation_ids = set(int(m) for m in re.findall(r"\[(\d+)\]", report_body))
    if not body_citation_ids or not citation_map:
        return report_body

    # Use Chinese section title when user request is in Chinese.
    use_chinese = bool(re.search(r"[\u4e00-\u9fff]", user_request or ""))
    section_title = "## 参考资料\n" if use_chinese else "## References\n"
    ref_lines = ["\n\n" + section_title]
    for cid in sorted(body_citation_ids):
        citation = citation_map.get(cid)
        if not citation:
            continue
        title = citation.get("title", "") or citation.get("url", "")
        url = citation.get("url", "")
        ref_lines.append(f"[{cid}] {title} - {url}")

    return report_body + "\n".join(ref_lines)

def _generate_fallback_report(
    user_request: str,
    subtasks: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
    verified_citations: list[dict[str, Any]],
) -> str:
    """Generate a fallback report when LLM fails."""
    lines = [f"# {user_request}\n"]
    lines.append("## Execution Summary\n")
    lines.append(
        f"This report is generated from {len(subtasks)} subtasks in fallback mode. "
        "Numeric values are redacted in fallback mode; please rely on cited sources.\n"
    )

    lines.append("## Subtask Results\n")
    for subtask in subtasks:
        status = str(subtask.get("status", "unknown"))
        lines.append(f"### [{status}] {subtask.get('description', '')}")
        result = subtask.get("result")
        if result:
            if isinstance(result, dict):
                result_str = str(result.get("analysis", ""))
            else:
                result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "...[truncated]"
            result_str = _redact_numbers(result_str)
            lines.append(f"\n{result_str}\n")
        if subtask.get("error"):
            lines.append(f"\nError: {subtask['error']}\n")

    verified_only = [c for c in verified_citations if c.get("verified")]
    if verified_only:
        lines.append("\n## References\n")
        for c in verified_only:
            title = c.get("title", "") or c.get("url", "")
            url = c.get("url", "")
            lines.append(f"- [{c['id']}] {title} - {url}")

    return "\n".join(lines)
