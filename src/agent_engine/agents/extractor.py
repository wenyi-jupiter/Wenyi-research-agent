"""Data extractor — a lightweight LLM call for precise field extraction.

The Extractor is a separate role from the Executor. While the Executor decides
*which tools to call*, the Extractor focuses on *precisely locating and copying
specific data points* from already-fetched text.

This separation reduces cognitive load: the tool-calling LLM no longer needs to
simultaneously do "macro strategy" and "micro string matching".

The extractor uses MiniMax-M2.1 (configurable via settings.extractor_model)
via DashScope for precise number/fact extraction and validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_engine.config import get_settings
from agent_engine.llm import get_provider

logger = logging.getLogger(__name__)

# Maximum text length to send to the extractor LLM in one call.
# Longer texts are truncated (we already have evidence_snippets for targeting).
_MAX_INPUT_CHARS = 30_000

EXTRACTOR_SYSTEM_PROMPT = """You are a precise data extraction assistant.

You will be given a block of text fetched from an official document or report,
along with a list of fields to extract.

RULES:
1. For each field, find the EXACT text in the document that contains the answer.
2. Copy the value VERBATIM — do not paraphrase, round, convert units, or infer.
3. Include a short "context" quote (the sentence or table row where the value appears).
4. If a field cannot be found in the given text, set its value to null and context to "NOT FOUND".
5. Do NOT use any external knowledge — extract ONLY from the provided text.
6. **CRITICAL — Preserve original units**: Copy the value exactly as it appears in
   the source, including its original currency and unit.
   NEVER convert currencies or units — the "unit" field must reflect the ORIGINAL source unit.
7. The "verbatim_quote" field must be a DIRECT copy-paste of 10-50 words from the
   source text surrounding the data point. This will be used for automated verification.

Respond with a JSON object:
{
  "extractions": [
    {
      "field": "the field name",
      "value": "the exact value copied from text, or null",
      "unit": "the original unit as written in source (e.g., 'billion yuan', 'USD million', '%', '亿元')",
      "verbatim_quote": "the exact 10-50 word quote from source containing this value",
      "source_url": "the URL this was extracted from (copied from Source URL above)",
      "context": "the sentence/row where it appears, or NOT FOUND",
      "confidence": 0.0-1.0
    }
  ]
}
"""


async def extract_fields(
    text: str,
    fields: list[str],
    *,
    source_url: str = "",
    subtask_description: str = "",
) -> list[dict[str, Any]]:
    """Extract specific fields from text using a dedicated LLM call.

    Args:
        text: The document text to extract from (e.g., from fetch_url extracted_text).
        fields: List of field names/descriptions to extract
                (e.g., ["total revenue", "R&D expenses", "net loss"]).
        source_url: URL of the source document (for context only).
        subtask_description: Description of the subtask (for context only).

    Returns:
        List of extraction dicts, each with: field, value, context, confidence.
        Returns empty list on failure.
    """
    if not text or not fields:
        return []

    settings = get_settings()

    # Use a dedicated provider instance for extraction.
    # The extractor uses MiniMax-M2.1 via DashScope for precise data extraction.
    try:
        extractor_provider = get_provider(
            provider="qwen",
            model=settings.extractor_model,
        )
    except Exception as e:
        logger.warning(f"[Extractor] Failed to get provider for model={settings.extractor_model}: {e}")
        return []

    # Truncate text to fit context window
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS] + f"\n... [truncated, {len(text)} total chars]"

    # Build the user message
    fields_str = "\n".join(f"- {f}" for f in fields)
    user_content = (
        f"Source URL: {source_url}\n"
        f"Subtask: {subtask_description}\n\n"
        f"Fields to extract:\n{fields_str}\n\n"
        f"--- DOCUMENT TEXT ---\n{text}\n--- END ---\n\n"
        f"Extract each field. Copy values VERBATIM from the text above."
    )

    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await extractor_provider.invoke(messages)
        content = response.content or ""

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content.strip())
        extractions = parsed.get("extractions", [])

        logger.info(
            f"[Extractor] Extracted {len(extractions)} fields from {source_url or 'text'} "
            f"(tokens: {response.input_tokens}+{response.output_tokens})"
        )
        return extractions

    except json.JSONDecodeError as e:
        logger.warning(f"[Extractor] Failed to parse JSON response: {e}")
        return []
    except Exception as e:
        logger.warning(f"[Extractor] Extraction failed: {e}")
        return []


async def validate_subtask_result(
    subtask_result: str,
    evidence_snippets: list[dict[str, Any]],
    fetched_contents: dict[str, str],
) -> dict[str, Any]:
    """Validate whether claims in a subtask result are grounded in evidence.

    This is the "per-subtask verification" step. It checks whether key numbers
    and facts in the executor's output can be found verbatim in the fetched content.

    Args:
        subtask_result: The executor's final analysis text for this subtask.
        evidence_snippets: Evidence snippets collected during fetch_url calls.
        fetched_contents: Dict of {url: excerpt_text} from fetch_url results.

    Returns:
        Dict with:
          - verified: bool — overall pass/fail
          - grounded_claims: int — number of claims found in evidence
          - ungrounded_claims: int — number of claims NOT found in evidence
          - details: list of {claim, found_in, grounded} dicts
    """
    if not subtask_result:
        return {"verified": True, "grounded_claims": 0, "ungrounded_claims": 0, "details": []}

    # Extract ALL quantitative claims from the subtask result.
    # This must be comprehensive to catch every possible hallucinated data point.
    number_patterns = [
        # Monetary: $1,234.56 million, ¥100亿, USD 50M
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|k|thousand))?",
        r"[\d,]+(?:\.\d+)?\s*(?:亿|万|千|百万|billion|million|万元|亿元|美元|USD|RMB|CNY)",
        r"(?:US\$|USD|¥|€|£|CNY|RMB)\s*[\d,]+(?:\.\d+)?",
        # Percentages: 45.2%, 45 percent, increased 37%
        r"[\d,]+(?:\.\d+)?\s*(?:%|percent|百分之|个百分点)",
        r"(?:增长|下降|增加|减少|涨|跌|同比|环比)\s*[\d,]+(?:\.\d+)?\s*%?",
        # Ratios and multiples: 3:1, 3x, 2.5倍
        r"[\d,]+(?:\.\d+)?\s*(?:x|倍|fold)",
        r"[\d,]+(?:\.\d+)?\s*:\s*[\d,]+(?:\.\d+)?",
        # Dates with specific day/month: March 15 2024, 2024年3月
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?",
        # Rankings: ranked #3, 第3位, top 10
        r"(?:ranked?\s*#?\s*|第)\d+(?:位|名|st|nd|rd|th)?",
        r"(?:top|前)\s*\d+",
        # Plain large numbers with commas (5+ chars to avoid noise): 12,345
        r"(?<!\w)[\d,]{5,}(?:\.\d+)?(?!\w)",
    ]
    claims = []
    for pat in number_patterns:
        for m in re.finditer(pat, subtask_result, flags=re.IGNORECASE):
            claim = m.group(0).strip()
            if claim and len(claim) >= 3:
                claims.append(claim)

    if not claims:
        return {"verified": True, "grounded_claims": 0, "ungrounded_claims": 0, "details": []}

    # Deduplicate
    seen = set()
    unique_claims = []
    for c in claims:
        key = re.sub(r"[\s,]", "", c.lower())
        if key not in seen:
            seen.add(key)
            unique_claims.append(c)

    # Build combined evidence text
    evidence_text = ""
    for snip in evidence_snippets:
        s = snip.get("snippet", "")
        if s:
            evidence_text += s + "\n"
    for url, content in fetched_contents.items():
        evidence_text += content + "\n"
    evidence_lower = evidence_text.lower()

    # Check each claim with multi-strategy matching
    details = []
    grounded = 0
    ungrounded = 0

    # Pre-compute normalized evidence (strip all whitespace/commas)
    evidence_normalized = re.sub(r"[\s,]", "", evidence_lower)

    for claim in unique_claims[:20]:
        found_in = None

        # Strategy 1: Exact match
        if claim.lower() in evidence_lower:
            found_in = "evidence (exact match)"

        # Strategy 2: Normalized match (strip whitespace/commas)
        if not found_in:
            normalized = re.sub(r"[\s,]", "", claim.lower())
            if normalized in evidence_normalized:
                found_in = "evidence (normalized match)"

        # Strategy 3: Core number extraction — handles format loss from
        # HTML table→text conversion where "$2,458,779" becomes
        # "2458779" or "2,458,779" scattered across broken table cells.
        if not found_in:
            core_digits = re.sub(r"[^\d.]", "", claim)
            if core_digits and len(core_digits) >= 3:
                if core_digits in re.sub(r"[^\d.]", " ", evidence_text):
                    found_in = "evidence (core number match)"
                else:
                    if "." in core_digits:
                        int_part = core_digits.split(".")[0]
                        if int_part and len(int_part) >= 3:
                            if int_part in evidence_text:
                                found_in = "evidence (integer part match)"

        # Strategy 4: Unit consistency check — prevents currency/magnitude
        # hallucination where e.g. "18.859 billion yuan" becomes "18.86亿美元".
        # If the claim contains a unit, verify the SAME unit family exists
        # near the matched number in the evidence.
        if found_in:
            unit_mismatch = _check_unit_mismatch(claim, evidence_text)
            if unit_mismatch:
                found_in = None  # Revoke grounding — unit doesn't match
                logger.warning(
                    f"[Validator] Unit mismatch for claim '{claim}': {unit_mismatch}"
                )

        if found_in:
            grounded += 1
        else:
            ungrounded += 1

        details.append({
            "claim": claim,
            "found_in": found_in,
            "grounded": found_in is not None,
        })

    # Stricter threshold: 80% of claims must be grounded (was 60%).
    # A subtask with 4 grounded + 1 fabricated claim should NOT pass.
    verified = ungrounded == 0 or (grounded / max(grounded + ungrounded, 1)) >= 0.8

    return {
        "verified": verified,
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
        "details": details,
    }


# ════════════════════════════════════════════════
# Unit-consistency checking helpers
# ════════════════════════════════════════════════

# Currency/unit families — tokens that belong to the same conceptual group.
# If a claim uses a unit from one family but the evidence uses a different family,
# it's a unit mismatch (e.g. "美元" in claim but "人民币/yuan" in evidence).
_UNIT_FAMILIES: list[tuple[str, list[str]]] = [
    ("USD", ["$", "usd", "us$", "美元", "美金", "u.s. dollar", "us dollar"]),
    ("CNY", ["¥", "cny", "rmb", "人民币", "元", "yuan", "renminbi", "万元", "亿元"]),
    ("EUR", ["€", "eur", "euro", "欧元"]),
    ("GBP", ["£", "gbp", "pound", "英镑"]),
    ("JPY", ["jpy", "yen", "日元", "円"]),
    ("HKD", ["hk$", "hkd", "港币", "港元"]),
]

# Magnitude families — if source says "billion" but claim says "万", mismatch.
_MAGNITUDE_FAMILIES: list[tuple[str, list[str]]] = [
    ("billion", ["billion", "b", "十亿"]),
    ("yi", ["亿", "亿元", "亿美元", "亿人民币"]),
    ("million", ["million", "m", "百万"]),
    ("wan", ["万", "万元", "万美元"]),
    ("thousand", ["thousand", "k", "千"]),
    ("trillion", ["trillion", "t", "万亿"]),
]


def _detect_unit_family(
    text: str, families: list[tuple[str, list[str]]]
) -> str | None:
    """Detect which unit/magnitude family a text snippet belongs to.

    Returns the family name (e.g. "USD", "CNY") or None if no match.
    """
    text_lower = text.lower()
    for family_name, tokens in families:
        for token in tokens:
            if token in text_lower:
                return family_name
    return None


def _check_unit_mismatch(claim: str, evidence_text: str) -> str | None:
    """Check if the units in a claim are consistent with evidence.

    Returns a description of the mismatch, or None if consistent.

    Example: claim="18.86亿美元" but evidence says "18.859 billion yuan"
    → returns "claim uses USD but evidence uses CNY"
    """
    # Extract the core number from the claim to locate it in evidence
    core_digits = re.sub(r"[^\d.]", "", claim)
    if not core_digits or len(core_digits) < 3:
        return None  # Too short to verify meaningfully

    # Find the claim's currency family
    claim_currency = _detect_unit_family(claim, _UNIT_FAMILIES)
    if not claim_currency:
        return None  # No currency detected in claim, skip check

    # Find where the number appears in the evidence and extract surrounding context
    # Look for the number (with flexible formatting) in evidence
    search_num = core_digits
    if "." in search_num:
        search_num = search_num.split(".")[0]

    if not search_num or len(search_num) < 3:
        return None

    # Find all positions of this number in the evidence text
    evidence_lower = evidence_text.lower()
    positions = [m.start() for m in re.finditer(re.escape(search_num), evidence_lower)]

    if not positions:
        return None  # Number not found in evidence — will be caught by other checks

    # For each position, extract a window around it and check the currency
    for pos in positions:
        window_start = max(0, pos - 80)
        window_end = min(len(evidence_lower), pos + len(search_num) + 80)
        window = evidence_lower[window_start:window_end]

        evidence_currency = _detect_unit_family(window, _UNIT_FAMILIES)
        if evidence_currency and evidence_currency != claim_currency:
            return (
                f"claim uses {claim_currency} but evidence uses {evidence_currency} "
                f"near '{evidence_text[window_start:window_end][:60]}...'"
            )

    # Also check magnitude consistency
    claim_magnitude = _detect_unit_family(claim, _MAGNITUDE_FAMILIES)
    if claim_magnitude:
        for pos in positions:
            window_start = max(0, pos - 80)
            window_end = min(len(evidence_lower), pos + len(search_num) + 80)
            window = evidence_lower[window_start:window_end]

            evidence_magnitude = _detect_unit_family(window, _MAGNITUDE_FAMILIES)
            if evidence_magnitude and evidence_magnitude != claim_magnitude:
                return (
                    f"claim uses magnitude '{claim_magnitude}' but evidence "
                    f"uses '{evidence_magnitude}'"
                )

    return None
