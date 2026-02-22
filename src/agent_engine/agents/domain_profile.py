"""Pluggable domain knowledge profiles — loaded from Markdown files.

Domain knowledge lives in agents/domains/*.md files (YAML frontmatter +
Markdown body), NOT in Python code. To add a new domain:
    1. Create agents/domains/<name>.md following the existing templates.
    2. Set a unique `priority` (lower = matched first).
    3. Restart the process (profiles are loaded once at import time).
    No changes to executor.py, planner.py, or entity_resolver.py are needed.

MD file format
--------------
---
name: my_domain
priority: 20          # lower wins; generic = 999 (always last)
signal_threshold: 2   # min signal hits to activate this profile
signals: [kw1, kw2]
alias_sets:
  - signals: [...]
    canonical: "Official Name"
    aliases: [...]
    note: "..."
entity_patterns: ['\bREGEX\b']
focus_term_rules:
  - triggers: [trigger1, trigger2]
    terms: [term1, term2]
low_quality_domains: [domain.com/path]
official_domains: [official.gov]
noise_patterns: [irrelevant topic]
search_vocabulary: [KEYWORD]
min_official_citation_ratio: 0.0
---

## Data Source Strategy

Free-text Markdown body injected into the planner prompt.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DOMAINS_DIR = Path(__file__).parent / "domains"


# ══════════════════════════════════════════════════════════════════
# Data structures  (unchanged public API — callers need no updates)
# ══════════════════════════════════════════════════════════════════


@dataclass
class FocusTermRule:
    """Conditional rule: if *any* trigger keyword appears in context text,
    add these focus terms to the evidence extraction pass."""

    triggers: list[str]
    terms: list[str]


@dataclass
class AliasSet:
    """A set of name aliases for a specific entity within this domain."""

    signals: list[str]
    canonical: str
    aliases: list[str]
    note: str = ""


@dataclass
class DomainProfile:
    """All domain-specific knowledge for one research domain.

    Loaded from a domains/*.md file; never instantiated directly by callers.
    """

    name: str

    # Detection
    signals: list[str]
    signal_threshold: int = 2

    # Entity resolution
    alias_sets: list[AliasSet] = field(default_factory=list)
    entity_patterns: list[str] = field(default_factory=list)

    # Planner prompt injection
    data_source_strategy: str = ""

    # Executor: evidence extraction
    focus_term_rules: list[FocusTermRule] = field(default_factory=list)

    # Executor: URL quality control
    low_quality_domains: list[str] = field(default_factory=list)
    official_domains: list[str] = field(default_factory=list)

    # Executor: noise filter
    noise_patterns: list[str] = field(default_factory=list)

    # Executor: keyword fallback vocabulary
    search_vocabulary: list[str] = field(default_factory=list)

    # Executor: entity marker regex patterns for _is_instruction_only()
    # Patterns that indicate a query segment contains a searchable entity
    # (e.g., form types like 20-F, regulatory identifiers like FDA/SEC).
    entity_marker_patterns: list[str] = field(default_factory=list)

    # Executor system prompt domain hints — injected as {domain_executor_hints}
    executor_hints: str = ""

    # Reporter system prompt domain hints — injected as {domain_reporter_hints}
    reporter_hints: str = ""

    # Critic gate
    min_official_citation_ratio: float = 0.0

    # Internal
    _priority: int = field(default=500, repr=False)
    _is_fallback: bool = field(default=False, repr=False)

    # ══ Public API ══

    def detect(self, text: str) -> bool:
        """Return True if this domain applies to the given text."""
        if self._is_fallback:
            return True
        t = (text or "").lower()
        count = sum(1 for sig in self.signals if sig.lower() in t)
        return count >= self.signal_threshold

    def detect_entities(self, text: str) -> dict[str, Any]:
        """Extract an entity profile from request text."""
        t = text.lower()
        for alias_set in self.alias_sets:
            if any(sig.lower() in t for sig in alias_set.signals):
                return {
                    "canonical_name": alias_set.canonical,
                    "aliases": list(alias_set.aliases),
                    "timeline_note": alias_set.note,
                }
        return self._extract_dynamic_entity(text)

    def _extract_dynamic_entity(self, text: str) -> dict[str, Any]:
        aliases: list[str] = []
        for pattern in self.entity_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                candidate = m.group(0).strip()
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
        if aliases:
            return {
                "canonical_name": aliases[0],
                "aliases": aliases[:6],
                "timeline_note": "",
            }
        return {"canonical_name": "", "aliases": [], "timeline_note": ""}

    def get_focus_terms(self, text: str) -> list[str]:
        """Return focus terms applicable to this context text."""
        t = (text or "").lower()
        terms: list[str] = []
        for rule in self.focus_term_rules:
            if any(trigger.lower() in t for trigger in rule.triggers):
                for term in rule.terms:
                    if term not in terms:
                        terms.append(term)
        return terms[:15]

    def is_low_quality_url(self, url: str) -> bool:
        """Return True if URL matches a known low-quality domain."""
        u = (url or "").lower()
        return any(domain in u for domain in self.low_quality_domains)

    def is_noise_result(self, title: str, snippet: str) -> bool:
        """Return True if a search result looks like domain noise."""
        blob = f"{title} {snippet}".lower()
        return any(p.lower() in blob for p in self.noise_patterns)


# ══════════════════════════════════════════════════════════════════
# MD file loader
# ══════════════════════════════════════════════════════════════════


def _parse_md(path: Path) -> DomainProfile:
    """Parse a domains/*.md file into a DomainProfile.

    Expected format:
        ---
        <YAML frontmatter>
        ---

        <Markdown body used as data_source_strategy>

    Raises:
        ValueError: if the frontmatter is missing or malformed.
    """
    text = path.read_text(encoding="utf-8")

    # Split off YAML frontmatter (between the first two "---" delimiters)
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(
            f"Domain profile {path.name} is missing YAML frontmatter delimiters (---)."
        )
    _, fm_text, body = parts
    meta: dict = yaml.safe_load(fm_text) or {}

    # ── Parse body into named sections using "## Section Title" markers ──
    # Any text before the first "##" becomes "data_source_strategy".
    # Each "## Section Title" starts a new section; the name is snake_cased.
    sections: dict[str, list[str]] = {}
    current_key = "data_source_strategy"
    sections[current_key] = []
    for line in body.splitlines():
        if line.startswith("## "):
            current_key = line[3:].strip().lower().replace(" ", "_")
            sections.setdefault(current_key, [])
        else:
            sections.setdefault(current_key, []).append(line)

    def _section(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    # ── Structured YAML fields ──
    alias_sets: list[AliasSet] = []
    for raw in meta.get("alias_sets") or []:
        alias_sets.append(
            AliasSet(
                signals=list(raw.get("signals") or []),
                canonical=str(raw.get("canonical") or ""),
                aliases=list(raw.get("aliases") or []),
                note=str(raw.get("note") or "").strip(),
            )
        )

    focus_term_rules: list[FocusTermRule] = []
    for raw in meta.get("focus_term_rules") or []:
        focus_term_rules.append(
            FocusTermRule(
                triggers=[str(t) for t in (raw.get("triggers") or [])],
                terms=[str(t) for t in (raw.get("terms") or [])],
            )
        )

    return DomainProfile(
        name=str(meta.get("name") or path.stem),
        signals=[str(s) for s in (meta.get("signals") or [])],
        signal_threshold=int(meta.get("signal_threshold") or 2),
        alias_sets=alias_sets,
        entity_patterns=[str(p) for p in (meta.get("entity_patterns") or [])],
        entity_marker_patterns=[str(p) for p in (meta.get("entity_marker_patterns") or [])],
        data_source_strategy=_section("data_source_strategy"),
        executor_hints=_section("executor_hints"),
        reporter_hints=_section("reporter_hints"),
        focus_term_rules=focus_term_rules,
        low_quality_domains=[str(d) for d in (meta.get("low_quality_domains") or [])],
        official_domains=[str(d) for d in (meta.get("official_domains") or [])],
        noise_patterns=[str(n) for n in (meta.get("noise_patterns") or [])],
        search_vocabulary=[str(v) for v in (meta.get("search_vocabulary") or [])],
        min_official_citation_ratio=float(
            meta.get("min_official_citation_ratio") or 0.0
        ),
        _priority=int(meta.get("priority") or 500),
        _is_fallback=bool(meta.get("is_fallback") or False),
    )


def _load_profiles(domains_dir: Path) -> list[DomainProfile]:
    """Load all *.md files from domains_dir, sorted by priority."""
    profiles: list[DomainProfile] = []
    if not domains_dir.exists():
        logger.warning(
            f"[DomainProfile] domains directory not found: {domains_dir}. "
            "Using empty profile list — GENERIC_FALLBACK will be used for all requests."
        )
        return profiles

    for md_path in sorted(domains_dir.glob("*.md")):
        try:
            profile = _parse_md(md_path)
            profiles.append(profile)
            logger.debug(
                f"[DomainProfile] Loaded '{profile.name}' "
                f"(priority={profile._priority}, "
                f"signals={len(profile.signals)}, "
                f"fallback={profile._is_fallback})"
            )
        except Exception as exc:
            logger.error(
                f"[DomainProfile] Failed to load {md_path.name}: {exc}. "
                "This domain profile will be skipped."
            )

    # Sort: non-fallback profiles by priority (ascending), fallbacks last
    profiles.sort(key=lambda p: (p._is_fallback, p._priority))
    return profiles


# ══════════════════════════════════════════════════════════════════
# Module-level registry (loaded once at import)
# ══════════════════════════════════════════════════════════════════

_PROFILES: list[DomainProfile] = _load_profiles(_DOMAINS_DIR)

# Hard-coded emergency fallback in case the domains/ directory is missing
# or generic.md fails to load. This guarantees detect_domain_profile()
# always returns something.
_EMERGENCY_FALLBACK = DomainProfile(
    name="generic_builtin",
    signals=[],
    signal_threshold=0,
    data_source_strategy=(
        "Prefer official sources. Avoid JavaScript-rendered pages."
    ),
    _priority=9999,
    _is_fallback=True,
)

if not any(p._is_fallback for p in _PROFILES):
    logger.warning(
        "[DomainProfile] No fallback profile found in domains/. "
        "Using built-in emergency fallback."
    )
    _PROFILES.append(_EMERGENCY_FALLBACK)

logger.info(
    f"[DomainProfile] Registry ready: "
    + ", ".join(f"{p.name}(p={p._priority})" for p in _PROFILES)
)


# ══════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════


def detect_domain_profile(text: str) -> DomainProfile:
    """Return the best-matching domain profile for the given text.

    Profiles are checked in priority order (lowest priority number first).
    The fallback profile (is_fallback: true) always matches last.

    Args:
        text: User request or combined context text.

    Returns:
        The first matching DomainProfile.
    """
    for profile in _PROFILES:
        if profile.detect(text):
            return profile
    return _EMERGENCY_FALLBACK
