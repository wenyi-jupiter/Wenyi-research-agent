"""Entity resolution helpers for search/query normalization.

Delegates ALL domain-specific entity knowledge to DomainProfile.
No company names, tickers, or aliases are hardcoded here —
they live in domain_profile.py and can be extended without touching this file.
"""

from __future__ import annotations

import re
from typing import Any

from agent_engine.agents.domain_profile import detect_domain_profile


def _sanitize_aliases(aliases: list[str]) -> list[str]:
    """Normalize and filter noisy aliases before retrieval anchoring."""
    clean: list[str] = []
    seen: set[str] = set()

    for raw in aliases or []:
        alias = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not alias:
            continue
        if len(alias) > 80:
            continue
        if len(alias) < 2 and not re.fullmatch(r"[A-Za-z0-9]{1,2}", alias):
            continue
        if "\ufffd" in alias:
            continue
        if re.search(r"(?:%[0-9A-Fa-f]{2}){3,}", alias):
            continue
        if re.search(r"(?:\\x[0-9A-Fa-f]{2}){3,}", alias):
            continue
        if any(ord(ch) < 32 for ch in alias):
            continue

        # Require at least two word-like/CJK characters to avoid punctuation noise.
        word_like = re.findall(r"[\w\u3400-\u4dbf\u4e00-\u9fff]", alias, flags=re.UNICODE)
        if len(word_like) < 2:
            continue

        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        clean.append(alias)

    return clean[:8]


def resolve_entity_profile(text: str) -> dict[str, Any]:
    """Resolve canonical entity name and aliases from request text.

    Uses the active domain profile to detect entities dynamically.
    All entity knowledge (aliases, ticker symbols, historical names)
    lives in the domain profile's alias_sets and entity_patterns.

    Args:
        text: Combined user request + subtask description text.

    Returns:
        Dict with keys: canonical_name, aliases, timeline_note.
        Returns empty strings/lists when no entity can be resolved.
    """
    profile = detect_domain_profile(text)
    raw_profile = profile.detect_entities(text) or {}
    timeline_note = str(raw_profile.get("timeline_note") or "")

    canonical_raw = str(raw_profile.get("canonical_name") or "").strip()
    canonical_list = _sanitize_aliases([canonical_raw]) if canonical_raw else []
    canonical = canonical_list[0] if canonical_list else ""

    aliases_raw = list(raw_profile.get("aliases") or [])
    merged_aliases = _sanitize_aliases(([canonical] if canonical else []) + aliases_raw)
    if not canonical and merged_aliases:
        canonical = merged_aliases[0]
    if canonical:
        merged_aliases = [canonical] + [
            a for a in merged_aliases if a.casefold() != canonical.casefold()
        ]

    return {
        "canonical_name": canonical,
        "aliases": merged_aliases[:6],
        "timeline_note": timeline_note,
    }


def is_entity_anchored(text: str, aliases: list[str]) -> bool:
    """Return True when text contains at least one entity alias.

    Args:
        text: Query or search string to check.
        aliases: List of known aliases for the target entity.

    Returns:
        True if any alias appears in the text (case-insensitive).
    """
    if not text or not aliases:
        return False
    aliases = _sanitize_aliases(aliases)
    if not aliases:
        return False
    t = text.lower()
    for alias in aliases:
        a = (alias or "").strip().lower()
        if a and a in t:
            return True
    return False


def expand_query_with_alias_anchor(
    query: str,
    aliases: list[str],
    *,
    max_len: int = 50,
) -> str:
    """Ensure a search query includes at least one entity alias anchor.

    Prevents topic drift by guaranteeing the entity name appears in the query.
    Prefers the shortest alias to stay within the max_len budget.

    Args:
        query: The base search query.
        aliases: Known aliases for the target entity.
        max_len: Maximum allowed query length after anchoring.

    Returns:
        Query with an entity anchor prepended if it was missing.
    """
    q = (query or "").strip()
    aliases = _sanitize_aliases(aliases)
    if not q or not aliases:
        return q
    if is_entity_anchored(q, aliases):
        return q

    short_aliases = sorted(
        [a for a in aliases if a and len(a) <= 8],
        key=len,
    )
    anchor = short_aliases[0] if short_aliases else aliases[0]
    anchored = f"{anchor} {q}".strip()
    return anchored[:max_len].strip()


def get_domain_search_vocabulary(text: str) -> list[str]:
    """Return domain-specific search vocabulary for keyword extraction.

    Used by _extract_keywords_fallback() to boost domain-relevant terms
    in query normalization. Replaces the hardcoded financial_terms list.

    Args:
        text: Context text for domain detection.

    Returns:
        List of domain-specific keywords/terms.
    """
    profile = detect_domain_profile(text)
    return profile.search_vocabulary


def get_domain_noise_patterns(text: str) -> list[str]:
    """Return noise patterns for search result filtering.

    Used by _is_search_result_relevant() to drop off-topic results.

    Args:
        text: Context text for domain detection.

    Returns:
        List of noise topic patterns (lowercase strings).
    """
    profile = detect_domain_profile(text)
    return profile.noise_patterns
