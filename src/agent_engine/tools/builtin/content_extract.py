"""Content extraction helpers for fetch_url and citation verification.

Goals:
- Always produce *readable text* for citations (no binary/PDF gibberish).
- Provide a short excerpt for prompt inclusion.
- Detect non-citable pages (paywalls/login/directory templates) early.

This module is intentionally dependency-light:
- PDF extraction uses PyMuPDF (pymupdf) when installed, otherwise returns failure.
- HTML extraction uses heuristics + optional trafilatura if installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractResult:
    extracted_text: str
    excerpt: str
    quality_score: float
    quality_issues: list[str]
    is_citable: bool
    not_citable_reason: str | None = None


def _make_excerpt(text: str, max_len: int) -> str:
    """Create a length-limited excerpt via simple head truncation.

    The excerpt is a *preview* — NOT the primary evidence source.
    The real evidence chain works through `evidence_snippets`, which are
    extracted from the FULL `extracted_text` by `_collect_evidence_snippets()`
    using keyword-targeted windows.  Those snippets are stored alongside the
    excerpt in `fetched_content` and survive the truncation.

    So here we keep the simple head-truncation: it gives context (title,
    intro) while the precision work is done by evidence_snippets.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated, {len(text)} total chars]"


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _html_to_text_basic(html: str) -> str:
    """Best-effort HTML → plain text without external libs.

    Preserves table structure by converting <td>/<th> into pipe-delimited columns
    so financial data remains machine-readable after conversion.
    """
    if not html:
        return ""

    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove common chrome blocks (heuristic)
    text = re.sub(
        r"<(nav|header|footer|aside)[^>]*>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ── Table-aware conversion ──
    # Convert <td>/<th> to pipe-delimited cells to preserve table structure.
    # This is critical for financial filings where numbers live in table cells.
    # First close any <td>/<th> with " | " (pipe-delimited columns).
    text = re.sub(r"</t[dh]>\s*", " | ", text, flags=re.IGNORECASE)
    # Opening <td>/<th> just becomes empty (the pipe after closing handles separation)
    text = re.sub(r"<t[dh][^>]*>", "", text, flags=re.IGNORECASE)
    # <tr> becomes a newline (new row)
    text = re.sub(r"</?tr[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove <table>/<thead>/<tbody>/<tfoot> wrappers
    text = re.sub(r"</?(?:table|thead|tbody|tfoot|colgroup|col|caption)[^>]*>", "", text, flags=re.IGNORECASE)

    # Replace other block-ish tags with newlines
    text = re.sub(r"<(p|div|br|h[1-6]|li)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode a few common entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
    )
    # Clean up pipe-delimited rows: remove trailing/leading pipes per line
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            # Remove leading/trailing " | "
            line = line.strip("| ").strip()
            # Collapse multiple pipes
            line = re.sub(r'\|\s*\|', '|', line)
            lines.append(line)
    text = "\n".join(lines)

    # Collapse whitespace
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return _normalize_whitespace(text)


def _detect_paywall_or_login(text: str, html: str, url: str = "") -> str | None:
    """Return a non-citable reason string if the page looks like paywall/login/directory."""
    t = (text or "").lower()
    h = (html or "").lower()
    u = (url or "").lower()

    # ── Whitelists / strong signals (avoid false positives) ──
    # SEC EDGAR Archives filings sometimes contain the words "captcha"/"access denied"
    # in boilerplate/scripts, but the page can still be a full, citable filing.
    if ("sec.gov/archives/edgar/data/" in u or "data.sec.gov/submissions/" in u) and len(text) > 5000:
        if (
            "securities and exchange commission" in t
            or "united states securities and exchange commission" in t
            or "form 10-k" in t
            or "form 20-f" in t
        ):
            return None

    # Paywall / login indicators
    indicators = [
        "log in",
        "login",
        "sign in",
        "subscribe",
        "subscription",
        "register",
        "paywall",
        "please enable javascript",
        "access denied",
        "captcha",
        "verify you are human",
        "验证码",
        "登录",
        "注册",
        "订阅",
        "付费",
        "购买",
        "请登录后查看",
        "无权限",
        "访问受限",
        "人机验证",
    ]
    if any(s in t for s in indicators):
        return "paywall/login/captcha page (not citable)"

    # JS app shells with minimal pre-rendered content
    if len(text) < 200 and ("__next_data__" in h or "window.__initial_state__" in h or "webpack" in h):
        return "js-rendered app shell with minimal content (not citable)"

    # Directory/listing pages: lots of links, little readable body text
    link_count = h.count("<a ")
    if link_count >= 60 and len(text) < 500:
        return "directory/listing page with many links and little body text (not citable)"

    # Obvious “empty template” markers
    if re.search(r"\b(--|—|…)\b", text) and len(text) < 800:
        return "template-like page with placeholders and little content (not citable)"

    return None


def extract_main_text_from_html(
    html: str,
    *,
    url: str = "",
    max_excerpt_len: int = 5000,
) -> ExtractResult:
    """Extract main readable text from HTML (best effort).

    Prefers trafilatura when available; otherwise uses a basic heuristic stripper.
    For financial filing pages (SEC, HKEX, etc.), automatically increases excerpt
    budget and forces table inclusion to preserve financial data.
    """
    issues: list[str] = []
    extracted = ""

    url_lower = (url or "").lower()

    # Detect financial filing pages that need special handling
    is_financial_filing = any(kw in url_lower for kw in [
        "sec.gov/archives/edgar/data/",
        "hkexnews.hk/listedco/",
        "sse.com.cn/disclosure/",
        "cninfo.com.cn/",
        "form20-f", "form10-k", "20-f", "10-k",
    ])

    # For financial filings, always include tables and use a larger excerpt budget
    include_tables = is_financial_filing
    if is_financial_filing:
        max_excerpt_len = max(max_excerpt_len, 8000)
        issues.append("financial filing detected: tables included, excerpt budget increased")

    # ── SEC EDGAR SGML envelope stripping ──────────────────────────────────
    # SEC EDGAR filings sometimes arrive wrapped in SGML headers:
    #   <DOCUMENT>\n<TYPE>EX-99.1\n...<TEXT>\n<html>...</html>
    # The outer SGML tags are not valid HTML. lxml / trafilatura see the whole
    # document as a single tree node (parsed tree length: 1) and discard it.
    # Solution: detect the SGML envelope and extract only the inner <html>…</html>.
    html_for_extraction = html
    _sgml_match = re.search(r"<TEXT>\s*(<html[\s\S]*</html>)", html, re.IGNORECASE)
    if _sgml_match:
        html_for_extraction = _sgml_match.group(1)
        issues.append("SEC SGML envelope stripped — inner HTML extracted")

    # Try trafilatura if installed (better boilerplate removal)
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html_for_extraction,
            include_comments=False,
            include_tables=include_tables,
            favor_recall=is_financial_filing,  # favor_recall for filings: get more content
            url=url or None,
        ) or ""
        extracted = _normalize_whitespace(extracted)
        if extracted:
            issues.append("extracted via trafilatura")
    except Exception:
        extracted = ""

    if not extracted:
        extracted = _html_to_text_basic(html_for_extraction)
        if extracted:
            issues.append("extracted via basic html stripper")

    not_citable_reason = _detect_paywall_or_login(extracted, html, url=url)

    excerpt = _make_excerpt(extracted, max_excerpt_len)

    # Quality score: primarily based on extracted length and citability
    if not extracted:
        quality_score = 0.1
        issues.append("no readable text extracted")
    elif not_citable_reason:
        quality_score = 0.1
        issues.append(not_citable_reason)
    elif len(extracted) < 800:
        quality_score = 0.4
        issues.append("limited extracted text")
    else:
        quality_score = 0.9

    is_citable = bool(extracted) and not bool(not_citable_reason) and quality_score >= 0.3

    return ExtractResult(
        extracted_text=extracted,
        excerpt=excerpt,
        quality_score=quality_score,
        quality_issues=issues,
        is_citable=is_citable,
        not_citable_reason=not_citable_reason,
    )


def extract_pdf_text_from_bytes(
    pdf_bytes: bytes,
    *,
    max_excerpt_len: int = 5000,
    max_pages: int = 25,
    max_chars: int = 120_000,
) -> ExtractResult:
    """Extract readable text from a PDF.

    Uses PyMuPDF (pymupdf) when installed. If missing, returns non-citable result.
    """
    issues: list[str] = []
    if not pdf_bytes:
        return ExtractResult(
            extracted_text="",
            excerpt="",
            quality_score=0.1,
            quality_issues=["empty pdf bytes"],
            is_citable=False,
            not_citable_reason="empty pdf",
        )

    try:
        import fitz  # PyMuPDF  # type: ignore
    except Exception:
        return ExtractResult(
            extracted_text="",
            excerpt="",
            quality_score=0.1,
            quality_issues=["pymupdf not installed"],
            is_citable=False,
            not_citable_reason="pdf text extraction unavailable (missing pymupdf)",
        )

    text_parts: list[str] = []
    used_pages = 0
    truncated = False
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = min(doc.page_count, max_pages)
        for i in range(page_count):
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            t = _normalize_whitespace(t)
            if t:
                text_parts.append(t)
            used_pages = i + 1
            # Early-stop: prevent very large PDFs from timing out or bloating downstream context.
            if max_chars and sum(len(p) for p in text_parts) >= max_chars:
                truncated = True
                break
        extracted = _normalize_whitespace("\n\n".join(text_parts))
        issues.append(f"extracted via pymupdf (pages={used_pages}/{doc.page_count})")
        if truncated:
            issues.append(f"truncated pdf extraction at ~{max_chars} chars")
    except Exception as e:
        extracted = ""
        issues.append(f"pymupdf extraction error: {e}")
    finally:
        try:
            doc.close()  # type: ignore[name-defined]
        except Exception:
            pass

    not_citable_reason = None
    if not extracted:
        not_citable_reason = "pdf text extraction failed (no text)"

    excerpt = _make_excerpt(extracted, max_excerpt_len)

    if not extracted:
        quality_score = 0.1
    elif len(extracted) < 1500:
        # Short but readable: still citable, just lower confidence
        quality_score = 0.6 if len(extracted) >= 100 else 0.4
        issues.append("pdf text extracted but short")
    else:
        quality_score = 0.9

    is_citable = bool(extracted) and not bool(not_citable_reason) and quality_score >= 0.3

    return ExtractResult(
        extracted_text=extracted,
        excerpt=excerpt,
        quality_score=quality_score,
        quality_issues=issues,
        is_citable=is_citable,
        not_citable_reason=not_citable_reason,
    )


# ═══════════════════════════════════════════════════════════════
# Financial statement table extraction
# ═══════════════════════════════════════════════════════════════

# Headings that mark the start of key financial statement sections.
_FINANCIAL_SECTION_HEADINGS = [
    # Income Statement
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated statements of comprehensive income",
    "consolidated income statement",
    "consolidated profit and loss",
    # Balance Sheet
    "consolidated balance sheets",
    "consolidated statements of financial position",
    # Cash Flow
    "consolidated statements of cash flows",
    "consolidated cash flow statement",
    # Stockholders' Equity
    "consolidated statements of stockholders' equity",
    "consolidated statements of changes in equity",
]


def extract_financial_tables(text: str, max_chars_per_section: int = 4000) -> str:
    """Extract financial statement sections from a long document.

    Scans the text for known financial statement headings (income statement,
    balance sheet, cash flow statement) and extracts the surrounding content.
    This is critical for SEC filings where the text can be 100k+ chars but
    the financial tables only occupy a small portion deep in the document.

    Args:
        text: Full extracted text from a financial filing (HTML or PDF).
        max_chars_per_section: Max chars to extract per section.

    Returns:
        Concatenated financial statement sections as formatted text.
        Returns empty string if no sections are found.
    """
    if not text or len(text) < 500:
        return ""

    text_lower = text.lower()
    found_sections: list[tuple[int, str, str]] = []

    for heading in _FINANCIAL_SECTION_HEADINGS:
        pos = text_lower.find(heading)
        if pos < 0:
            continue

        # Already found a section at a nearby position — skip duplicate
        if any(abs(pos - existing_pos) < 200 for existing_pos, _, _ in found_sections):
            continue

        # Extract a window around the heading
        start = max(0, pos - 100)
        end = min(len(text), pos + max_chars_per_section)
        section_text = text[start:end].strip()

        # Find the original-case heading for display
        heading_display = text[pos:pos + len(heading)]

        found_sections.append((pos, heading_display, section_text))

    if not found_sections:
        return ""

    # Sort by position in document
    found_sections.sort(key=lambda x: x[0])

    parts = ["## Extracted Financial Statement Sections\n"]
    for _, heading_display, section_text in found_sections:
        parts.append(f"### {heading_display.strip()}")
        parts.append(section_text)
        parts.append("")

    return "\n".join(parts)
