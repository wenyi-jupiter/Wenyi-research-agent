"""Web search tool implementation using DuckDuckGo (ddgs package)."""

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx
from ddgs import DDGS

from agent_engine.config import get_settings
from agent_engine.tools.registry import tool
from agent_engine.tools.builtin.content_extract import (
    extract_financial_tables,
    extract_main_text_from_html,
    extract_pdf_text_from_bytes,
)
from agent_engine.tools.builtin.http_client import (
    SmartHttpClient,
    fetch_with_playwright,
)

logger = logging.getLogger(__name__)

# Thread pool for running sync DuckDuckGo search
_executor = ThreadPoolExecutor(max_workers=4)

# ── Rate limiting state ──
# Minimum interval (seconds) between consecutive DuckDuckGo requests to avoid
# triggering anti-bot throttling that returns garbage/irrelevant results.
_MIN_SEARCH_INTERVAL = 2.0
_last_search_time = 0.0

# ── Spam / irrelevant domain blocklist ──
# Domains that frequently appear in degraded DuckDuckGo results and are never
# useful for research tasks.  Results from these domains are silently dropped.
_SPAM_DOMAINS = {
    "51chigua", "chigua", "91chigua", "heiliaonet", "17heilian",
    "lllll.lol", "wqlcupe.cc", "openclash.cc", "favcomic.com",
    "ihrrv.", "hwmrz", "cccgg", "lrfzahq",
}

# ── Spam TLD patterns ──
# Random-looking domains on cheap TLDs are almost always spam/porn aggregators.
# Pattern: short random-looking alphanumeric string + cheap TLD
_SPAM_TLD_RE = re.compile(
    r"^https?://(?:www\.)?[a-z0-9]{4,12}\."
    r"(?:cc|xyz|top|work|click|buzz|surf|fun|icu|club|site|online|live|lol|"
    r"tech|bid|win|stream|racing|download|review|date|accountant|science|"
    r"cricket|party|trade|faith|loan|gq|cf|ga|ml|tk)(?:/|$)",
    re.IGNORECASE,
)

# ── Content safety patterns ──
# Regex patterns that detect pornographic, violent, or politically sensitive
# content in search result titles/snippets.  These are checked as a second
# layer *after* domain filtering, so even results from legitimate-looking
# domains get caught if their snippet is toxic.
#
# The patterns are intentionally broad — a false positive just means we drop
# one search result, which is far less costly than letting the content through
# and triggering a DashScope 400 that crashes the entire task.
_UNSAFE_CONTENT_PATTERNS = re.compile(
    r"|".join([
        # Pornographic / sexually explicit (Chinese)
        r"\u7fa4[Pp\u6253]\u8f6e",          # 群P轮
        r"\u8f6e\u64cd",                     # 轮操
        r"\u5c4c\u591a\u903c\u5c11",         # 屌多逼少
        r"\u4e00\u654c\u591a",               # 一敌多 (sexual context)
        r"\u5403\u74dc\u7f51",               # 吃瓜网
        r"\u6210\u4eba\u7f51",               # 成人网
        r"\u8272\u60c5",                     # 色情
        r"\u6deb\u79fd",                     # 淫秽
        r"\u5077\u62cd",                     # 偷拍
        r"\u88f8\u7167",                     # 裸照
        r"\u7ea6\u70ae",                     # 约炮
        r"\u5728\u7ebf\u89c2\u770b.*\u65e0\u7801",  # 在线观看.*无码
        r"AV\u5728\u7ebf",                  # AV在线
        r"\u4e1d\u889c.*\u7f8e\u817f.*\u5185\u8863", # 丝袜.*美腿.*内衣
        # Pornographic (English)
        r"\bxxx\b",
        r"\bporn",
        r"\bhentai\b",
        r"\bnude[sd]?\b",
        r"\bsex\s*(?:tape|video|movie)",
        r"\berotic",
        r"\badult\s+video",
        # Gambling
        r"\u535a\u5f69",                     # 博彩
        r"\u8d4c\u573a",                     # 赌场
        r"\u7f51\u8d4c",                     # 网赌
        r"\u5f00\u5956\u7ed3\u679c",         # 开奖结果
        r"\u5f69\u7968\u9884\u6d4b",         # 彩票预测
        # Violence / graphic
        r"\u6b8b\u5fcd\u5904\u51b3",         # 残忍处决
        r"\u8840\u8165\u89c6\u9891",         # 血腥视频
    ]),
    re.IGNORECASE,
)

# ── Known JS-heavy data portal domains ──
# Sites that render data client-side; static fetch gets empty templates.
# Keep this list as a module-level constant so it's easy to extend.
_JS_HEAVY_DOMAINS = [
    "eastmoney.com", "10jqka.com.cn", "xueqiu.com", "investing.com",
    "finance.sina.com.cn", "stockpage.10jqka.com.cn", "emweb.eastmoney.com",
    "basic.10jqka.com.cn", "quote.eastmoney.com", "data.eastmoney.com",
    "finance.yahoo.com", "tradingview.com",
]


def _detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English.

    Returns:
        'zh' for Chinese, 'en' for English/other.
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return "zh" if chinese_chars > len(text) * 0.15 else "en"


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query for relevance checking.

    Splits on whitespace and common Chinese punctuation, then keeps tokens
    that are >= 2 chars (Chinese) or >= 3 chars (Latin).
    """
    # Split on whitespace and punctuation
    tokens = re.split(r'[\s,，、；;：:。.！!？?()（）\[\]【】""\"]+', query)
    keywords = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        # Skip very short tokens and common stop words
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', t))
        if has_cjk and len(t) >= 2:
            keywords.append(t)
        elif not has_cjk and len(t) >= 3:
            keywords.append(t.lower())
    return keywords


def _is_result_relevant(result: dict, keywords: list[str]) -> bool:
    """Check if a search result is relevant to the query keywords.

    A result is considered relevant if its title or snippet contains at least
    one of the extracted keywords.  This filters out spam/garbage results that
    DuckDuckGo sometimes returns under rate limiting.
    """
    if not keywords:
        return True  # No keywords extracted → can't filter

    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()
    url = (result.get("url") or "").lower()
    combined = title + " " + snippet + " " + url

    for kw in keywords:
        if kw.lower() in combined:
            return True
    return False


def sanitize_text_for_llm(text: str) -> str:
    """Remove or replace unsafe content from text before sending to an LLM.

    This is the last line of defense: called on any text (snippets, fetched
    content excerpts, cross-subtask context) before it's injected into an LLM
    prompt.  If the text contains unsafe patterns, the offending segments are
    replaced with a placeholder so the surrounding context is preserved.

    This function is intentionally conservative — it replaces only the matched
    pattern, not the entire text.
    """
    if not text:
        return text
    return _UNSAFE_CONTENT_PATTERNS.sub("[content filtered]", text)


def is_text_safe_for_llm(text: str) -> bool:
    """Check if text is safe to include in an LLM prompt.

    Returns False if the text contains patterns that are likely to trigger
    content moderation by LLM providers like DashScope.
    """
    if not text:
        return True
    return not bool(_UNSAFE_CONTENT_PATTERNS.search(text))


def _is_spam_url(url: str) -> bool:
    """Check if a URL belongs to a known spam/irrelevant domain."""
    url_lower = url.lower()
    # Exact domain substring match
    if any(domain in url_lower for domain in _SPAM_DOMAINS):
        return True
    # Random-looking domain on cheap TLD (pattern-based catch-all)
    if _SPAM_TLD_RE.match(url_lower):
        return True
    return False


def _is_unsafe_content(result: dict) -> bool:
    """Check if a search result contains unsafe content (porn, violence, etc.).

    This is the critical second layer of defense.  Even if a URL passes domain
    filtering, its title or snippet may contain content that will trigger the
    LLM provider's content moderation (e.g. DashScope 400 Bad Request).

    We intentionally cast a wide net: a false positive only drops one search
    result, but a false negative can crash the entire multi-hour task.
    """
    title = result.get("title", "") or ""
    snippet = result.get("snippet", "") or ""
    combined = title + " " + snippet

    return bool(_UNSAFE_CONTENT_PATTERNS.search(combined))


def _sync_ddgs_search(query: str, num_results: int = 5) -> list[dict]:
    """Perform a DuckDuckGo search using the ddgs library.

    Includes rate limiting and spam/unsafe content filtering.
    Relevance is judged by LLM after fetch_url (not by regex here).

    Args:
        query: Search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of search result dictionaries.
    """
    global _last_search_time

    # ── Rate limiting: wait if too soon after last search ──
    now = time.monotonic()
    elapsed = now - _last_search_time
    if elapsed < _MIN_SEARCH_INTERVAL:
        sleep_time = _MIN_SEARCH_INTERVAL - elapsed
        logger.debug(f"[web_search] Rate limit: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)

    # Limit query length — very long queries return 0 results
    search_query = query[:120] if len(query) > 120 else query

    # Use cn-zh region for Chinese queries to get better Chinese results
    lang = _detect_language(search_query)
    region = "cn-zh" if lang == "zh" else "wt-wt"

    max_attempts = 3
    for attempt in range(max_attempts):
        _last_search_time = time.monotonic()

        raw_results = []
        try:
            search_results = DDGS().text(
                search_query,
                max_results=num_results + 3,  # Request extra to compensate for filtering
                region=region,
                backend="duckduckgo,google",  # Fixed backends; avoid Brave 429 / Mojeek 403
            )
            for r in search_results:
                raw_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
        except Exception as e:
            logger.warning(f"[web_search] DuckDuckGo search error (attempt {attempt+1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(_MIN_SEARCH_INTERVAL * (attempt + 1))
                continue
            break

        # ── Filter: remove spam domains + unsafe content only ──
        # Relevance is judged by LLM after fetch_url, not by regex here.
        filtered = []
        spam_count = 0
        unsafe_count = 0
        for r in raw_results:
            if _is_spam_url(r.get("url", "")):
                spam_count += 1
            elif _is_unsafe_content(r):
                unsafe_count += 1
                logger.warning(
                    f"[web_search] Blocked unsafe content: "
                    f"{r.get('url', '')[:60]}"
                )
            else:
                filtered.append(r)
        if spam_count > 0 or unsafe_count > 0:
            logger.info(
                f"[web_search] Filtered {spam_count} spam + "
                f"{unsafe_count} unsafe results"
            )

        logger.info(
            f"[web_search] query={search_query[:60]}... region={region} "
            f"raw={len(raw_results)} filtered={len(filtered)}"
        )

        if filtered:
            return filtered[:num_results]

        # No results after filtering; retry if API error might have occurred
        if attempt < max_attempts - 1:
            retry_delay = _MIN_SEARCH_INTERVAL * (attempt + 2)
            logger.warning(
                f"[web_search] No usable results — retrying in {retry_delay:.0f}s "
                f"(attempt {attempt+2}/{max_attempts})"
            )
            time.sleep(retry_delay)

    return raw_results[:num_results] if raw_results else []


async def _async_ddgs_search(query: str, num_results: int = 5) -> list[dict]:
    """Async wrapper for DuckDuckGo search.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of search result dictionaries.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_ddgs_search,
        query,
        num_results,
    )


@tool(
    name="web_search",
    description=(
        "Search the web for information using DuckDuckGo and Google. Returns relevant "
        "search results with titles, snippets, and URLs. "
        "IMPORTANT: The query MUST be a topical keyword search "
        "(e.g. 'BeiGene 2023 annual report SEC filing'), "
        "NOT a bare URL or domain name (e.g. NOT 'sec.gov' or "
        "'businesswire.com'). To navigate to a known URL, use fetch_url "
        "instead. Good queries combine: entity name + topic + year/details."
    ),
    tags=["search", "web", "information"],
)
async def web_search(
    query: str,
    num_results: int = 5,
) -> dict:
    """Search the web for information.

    Args:
        query: The search query.
        num_results: Number of results to return (default 5, max 10).

    Returns:
        Dictionary with search results.
    """
    num_results = min(num_results, 10)

    try:
        results = await _async_ddgs_search(query, num_results)

        return {
            "query": query,
            "num_results": len(results),
            "results": results,
            "success": True,
        }
    except httpx.TimeoutException:
        return {
            "query": query,
            "error": "Search request timed out",
            "results": [],
            "success": False,
        }
    except Exception as e:
        return {
            "query": query,
            "error": f"Search failed: {str(e)}",
            "results": [],
            "success": False,
        }


def _assess_content_quality(text: str, url: str) -> dict:
    """Assess quality of fetched content to detect JS-rendered templates and empty data.

    Many data portal sites use JavaScript to render content dynamically.
    A static HTTP fetch only gets the HTML template with placeholders
    like '@pe@', '--', empty table cells, etc.

    Args:
        text: The fetched text content.
        url: The source URL.

    Returns:
        Dict with quality assessment: score (0-1), issues list, is_js_rendered flag.
    """
    issues = []
    is_js_rendered = False

    # Strip HTML for analysis
    clean = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<[^>]+>', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Check 1: Template placeholder variables (strong JS indicator)
    placeholder_patterns = [
        r'@\w+@',           # @pe@, @price@, etc.
        r'\{\{\s*\w+\s*\}\}',  # {{ variable }}
        r'\$\{\w+\}',       # ${variable}
    ]
    placeholder_count = 0
    for pattern in placeholder_patterns:
        matches = re.findall(pattern, text)
        placeholder_count += len(matches)
    if placeholder_count >= 3:
        is_js_rendered = True
        issues.append(f"Contains {placeholder_count} template placeholders (e.g., @pe@) — page requires JavaScript rendering")

    # Check 2: Financial data fields all empty/dash
    empty_data_patterns = [
        r'(?:市盈率|PE|P/E|市净率|PB|P/B|股息率|换手率|成交量|市值)\s*[:：]?\s*[-—–]+',
        r'(?:市盈率|PE|P/E|市净率|PB|P/B|股息率|换手率|成交量|市值)\s*[:：]?\s*$',
    ]
    empty_field_count = 0
    for pattern in empty_data_patterns:
        empty_field_count += len(re.findall(pattern, clean, re.MULTILINE | re.IGNORECASE))
    if empty_field_count >= 2:
        is_js_rendered = True
        issues.append(f"{empty_field_count} financial data fields are empty/dashes — data loaded via JavaScript")

    # Check 3: Very little text content relative to HTML size
    text_ratio = len(clean) / max(len(text), 1)
    if len(text) > 5000 and text_ratio < 0.05:
        is_js_rendered = True
        issues.append(f"Text content is only {text_ratio:.1%} of HTML — likely a JavaScript-rendered page")

    # Check 4: Common JS framework indicators with no real content
    js_indicators = ['React.createElement', 'Vue.component', 'angular.module',
                     '__NEXT_DATA__', 'window.__INITIAL_STATE__', 'var defined=',
                     'webpack', 'chunk.js', 'bundle.js']
    js_indicator_count = sum(1 for ind in js_indicators if ind in text)
    if js_indicator_count >= 2 and len(clean) < 500:
        is_js_rendered = True
        issues.append("Page uses JavaScript frameworks with minimal pre-rendered content")

    # Check 5: Known JS-heavy data portal domains
    # These sites render data client-side; static fetch gets empty templates.
    # Loaded from a configurable list so the codebase stays task-agnostic.
    js_heavy_domains = _JS_HEAVY_DOMAINS
    is_known_js_domain = any(domain in url for domain in js_heavy_domains)
    if is_known_js_domain and len(clean) < 2000:
        is_js_rendered = True
        issues.append("Known JavaScript-heavy portal site with little static content")

    # Calculate quality score
    if is_js_rendered:
        score = 0.1  # Very low — data is not usable
    elif len(clean) < 200:
        score = 0.2
        issues.append("Very little text content extracted")
    elif len(clean) < 500:
        score = 0.4
        issues.append("Limited text content")
    elif issues:
        score = 0.5
    else:
        score = 0.9  # Good content

    return {
        "quality_score": score,
        "issues": issues,
        "is_js_rendered": is_js_rendered,
        "extracted_text_length": len(clean),
    }


@tool(
    name="fetch_url",
    description=(
        "Fetch the content of a URL and return its text content. "
        "Includes a quality assessment — if 'is_js_rendered' is True, the page "
        "requires JavaScript and the fetched content is likely incomplete or empty. "
        "In that case, try a different source (e.g., news articles, annual report PDFs, "
        "official press releases) instead of financial portal pages."
    ),
    tags=["web", "fetch", "http"],
)
async def fetch_url(
    url: str,
    timeout: float = 10.0,
    focus_terms: list[str] | None = None,
) -> dict:
    """Fetch content from a URL with quality assessment.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with URL content, quality assessment, or error.
    """
    try:
        settings = get_settings()

        base_headers = {
            "User-Agent": settings.http_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        sec_headers = dict(base_headers)
        sec_headers["User-Agent"] = settings.sec_user_agent or settings.http_user_agent
        sec_headers["Accept"] = "application/json,text/html,*/*"

        def _host(u: str) -> str:
            try:
                return (urlparse(u).netloc or "").lower()
            except Exception:
                return ""

        def _is_sec_host(h: str) -> bool:
            return h.endswith("sec.gov") or h.endswith("data.sec.gov")

        def _extract_cik(u: str) -> str | None:
            # Try query param CIK=...
            try:
                q = parse_qs(urlparse(u).query or "")
                cik_list = q.get("CIK") or q.get("cik") or []
                if cik_list:
                    m = re.search(r"\d{6,10}", cik_list[0])
                    if m:
                        return m.group(0).zfill(10)
            except Exception:
                pass
            # Try paths like /CIK0001540699.json
            m2 = re.search(r"CIK(\d{6,10})", u, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).zfill(10)
            return None

        def _format_sec_submissions(sub: dict) -> tuple[str, str]:
            cik = str(sub.get("cik", "") or "")
            name = str(sub.get("name", "") or "")
            filings = (sub.get("filings") or {}).get("recent") or {}
            forms = filings.get("form") or []
            accs = filings.get("accessionNumber") or []
            dates = filings.get("filingDate") or []
            prims = filings.get("primaryDocument") or []

            lines = []
            header = f"SEC EDGAR submissions for {name} (CIK {cik}):"
            lines.append(header)
            kept = 0
            for i, form in enumerate(forms):
                if kept >= 10:
                    break
                if form not in ("20-F", "6-K", "10-K"):
                    continue
                try:
                    acc = str(accs[i])
                    acc_nodash = acc.replace("-", "")
                    primary = str(prims[i])
                    filed = str(dates[i]) if i < len(dates) else ""
                    cik_int = int(re.sub(r"\D", "", cik) or "0")
                    archive_url = (
                        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary}"
                    )
                    lines.append(f"- {form} filed {filed} accession {acc} URL: {archive_url}")
                    kept += 1
                except Exception:
                    continue

            text = "\n".join(lines).strip()
            excerpt = text[:5000]
            return text, excerpt

        def _extract_nct(u: str) -> str | None:
            m = re.search(r"(NCT\d{8})", u, flags=re.IGNORECASE)
            return m.group(1).upper() if m else None

        def _get_path(d: dict, path: list[str]):
            cur = d
            for k in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            return cur

        def _format_clinicaltrials_api(data: dict) -> tuple[str, str]:
            # API v2 structure: protocolSection / identificationModule / statusModule
            proto = data.get("protocolSection") or {}
            ident = proto.get("identificationModule") or {}
            status = proto.get("statusModule") or {}

            nct = str(data.get("nctId") or ident.get("nctId") or "")
            title = str(ident.get("briefTitle") or ident.get("officialTitle") or "")
            overall = str(status.get("overallStatus") or "")

            def _date(struct_key: str) -> str:
                v = status.get(struct_key) or {}
                if isinstance(v, dict):
                    return str(v.get("date") or "")
                return ""

            start = _date("startDateStruct")
            primary = _date("primaryCompletionDateStruct")
            completion = _date("completionDateStruct")
            last_update = _date("lastUpdatePostDateStruct")

            lines = [
                f"ClinicalTrials.gov API v2 study {nct}",
                f"Title: {title}",
                f"Overall status: {overall}",
            ]
            if start:
                lines.append(f"Start date: {start}")
            if primary:
                lines.append(f"Primary completion date: {primary}")
            if completion:
                lines.append(f"Completion date: {completion}")
            if last_update:
                lines.append(f"Last update posted: {last_update}")

            text = "\n".join([l for l in lines if l.strip()]).strip()
            excerpt = text[:5000]
            return text, excerpt

        async with SmartHttpClient(
            timeout=timeout,
            max_retries=settings.http_max_retries,
            default_headers=base_headers,
        ) as client:
            response = await client.get(url)

            requested_url = url
            final_url = str(response.url)
            host = _host(final_url) or _host(requested_url)

            # SEC: if we can infer CIK, prefer data.sec.gov submissions JSON
            # over HTML pages (more stable + less JS/WAF issues).
            if _is_sec_host(host):
                cik = _extract_cik(requested_url) or _extract_cik(final_url)
                is_archives = "/archives/" in (urlparse(final_url).path or "").lower()
                is_data_sec = host.endswith("data.sec.gov")
                if cik and (not is_archives) and (not is_data_sec):
                    api_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                    api_resp = await client.get(
                        api_url,
                        headers={**sec_headers, "Accept": "application/json"},
                        max_retries=1,
                    )
                    if api_resp.status_code == 200:
                        sub = api_resp.json()
                        extracted_text, excerpt = _format_sec_submissions(sub)
                        return {
                            "url": api_url,
                            "requested_url": requested_url,
                            "status_code": 200,
                            "content_type": "application/json; source=data.sec.gov",
                            "is_html": False,
                            "is_pdf": False,
                            "content": excerpt,
                            "extracted_text": extracted_text,
                            "excerpt": excerpt,
                            "is_citable": True,
                            "not_citable_reason": None,
                            "quality_score": 0.9,
                            "is_js_rendered": False,
                            "quality_issues": [],
                            "extracted_text_length": len(extracted_text),
                        }

            # ── Domain-specific anti-bot handling ──
            if response.status_code in (403, 429):
                # SEC: retry with SEC-compliant headers; if we can infer CIK, use data.sec.gov JSON
                if _is_sec_host(host):
                    retry = await client.get(
                        requested_url, headers=sec_headers, max_retries=1,
                    )
                    if retry.status_code < 400:
                        response = retry
                        final_url = str(response.url)
                        host = _host(final_url) or host
                    else:
                        cik = _extract_cik(requested_url) or _extract_cik(final_url)
                        if cik:
                            api_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                            api_resp = await client.get(
                                api_url,
                                headers={**sec_headers, "Accept": "application/json"},
                                max_retries=1,
                            )
                            if api_resp.status_code == 200:
                                sub = api_resp.json()
                                extracted_text, excerpt = _format_sec_submissions(sub)
                                return {
                                    "url": api_url,
                                    "requested_url": requested_url,
                                    "status_code": 200,
                                    "content_type": "application/json; source=data.sec.gov",
                                    "is_html": False,
                                    "is_pdf": False,
                                    "content": excerpt,
                                    "extracted_text": extracted_text,
                                    "excerpt": excerpt,
                                    "is_citable": True,
                                    "not_citable_reason": None,
                                    "quality_score": 0.9,
                                    "is_js_rendered": False,
                                    "quality_issues": [],
                                    "extracted_text_length": len(extracted_text),
                                }
                            # If JSON fetch failed, fall through to error handling below

                # ClinicalTrials: if HTML blocked, use official API v2 JSON
                if host.endswith("clinicaltrials.gov"):
                    nct = _extract_nct(requested_url) or _extract_nct(final_url)
                    if nct:
                        api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct}"
                        api_resp = await client.get(
                            api_url,
                            headers={**base_headers, "Accept": "application/json"},
                            max_retries=1,
                        )
                        if api_resp.status_code == 200:
                            data = api_resp.json()
                            extracted_text, excerpt = _format_clinicaltrials_api(data)
                            return {
                                "url": api_url,
                                "requested_url": requested_url,
                                "status_code": 200,
                                "content_type": "application/json; source=clinicaltrials.gov api v2",
                                "is_html": False,
                                "is_pdf": False,
                                "content": excerpt,
                                "extracted_text": extracted_text,
                                "excerpt": excerpt,
                                "is_citable": True,
                                "not_citable_reason": None,
                                "quality_score": 0.9,
                                "is_js_rendered": False,
                                "quality_issues": [],
                                "extracted_text_length": len(extracted_text),
                            }
                        else:
                            return {
                                "url": final_url,
                                "requested_url": requested_url,
                                "status_code": response.status_code,
                                "content_type": None,
                                "is_html": False,
                                "is_pdf": False,
                                "content": "",
                                "extracted_text": "",
                                "excerpt": "",
                                "is_citable": False,
                                "not_citable_reason": (
                                    "clinicaltrials.gov blocked (403/429) from this environment; "
                                    "use official company IR/pipeline disclosures or other accessible registries"
                                ),
                                "quality_score": 0.1,
                                "is_js_rendered": False,
                                "quality_issues": ["clinicaltrials.gov blocked"],
                                "extracted_text_length": 0,
                                "error": f"HTTP error: {response.status_code}",
                            }

            # If still an error after domain-specific handling, return a structured error
            if response.status_code >= 400:
                return {
                    "url": final_url,
                    "requested_url": requested_url,
                    "error": f"HTTP error: {response.status_code}",
                    "status_code": response.status_code,
                }
            content_type = (response.headers.get("content-type", "") or "").lower()

            # ── Content-type routing ──
            is_pdf = "application/pdf" in content_type or final_url.lower().endswith(".pdf")
            is_html = "text/html" in content_type
            is_text = content_type.startswith("text/") or "charset=" in content_type

            extracted_text = ""
            excerpt = ""
            evidence_snippets: list[dict] = []
            not_citable_reason = None
            quality_issues: list[str] = []
            is_js_rendered = False
            extracted_len = 0

            def _collect_evidence_snippets(text: str) -> list[dict]:
                terms = [t.strip() for t in (focus_terms or []) if str(t or "").strip()]
                if not terms or not text:
                    return []
                snippets: list[dict] = []
                seen = set()
                max_snippets = 8
                window = 220
                for term in terms:
                    if len(snippets) >= max_snippets:
                        break
                    key = term.lower()
                    if key in seen or len(key) < 2:
                        continue
                    seen.add(key)
                    # Choose the "best" occurrence: prefer snippets with numbers/$ which
                    # often indicate table rows for financial fields.
                    best = None
                    best_score = -1
                    pat = re.compile(re.escape(term), flags=re.IGNORECASE)
                    for j, m in enumerate(pat.finditer(text)):
                        if j > 200:
                            break
                        idx = m.start()
                        s = max(0, idx - window)
                        e = min(len(text), idx + (m.end() - m.start()) + window)
                        raw = text[s:e].replace("\n", " ").strip()
                        raw = re.sub(r"\s+", " ", raw)
                        # Score: digits + currency markers
                        score = sum(ch.isdigit() for ch in raw)
                        score += raw.count("$") * 10
                        score += raw.count(",")  # thousands separators
                        if score > best_score:
                            best_score = score
                            best = raw
                    if not best:
                        continue
                    snip = best
                    if len(snip) > 600:
                        snip = snip[:600] + f" ... [truncated, {len(snip)} chars]"
                    snippets.append({"term": term, "snippet": snip, "score": best_score})
                return snippets

            if is_pdf:
                pdf_bytes = response.content
                # Limit PDF extraction to avoid tool timeouts on huge PDFs
                pdf_res = extract_pdf_text_from_bytes(
                    pdf_bytes,
                    max_excerpt_len=5000,
                    max_pages=25,
                    max_chars=120_000,
                )
                extracted_text = pdf_res.extracted_text
                excerpt = pdf_res.excerpt
                not_citable_reason = pdf_res.not_citable_reason
                quality_issues = pdf_res.quality_issues
                quality_score = pdf_res.quality_score
                is_citable = pdf_res.is_citable
                extracted_len = len(extracted_text)
                evidence_snippets = _collect_evidence_snippets(extracted_text)
            elif is_html or is_text:
                # For HTML/text-ish content: keep raw HTML for JS-render detection, but only return excerpt as "content"
                raw_html = response.text
                # Assess JS-template quality on raw HTML (pre-extraction)
                js_quality = _assess_content_quality(raw_html[:50000], final_url)
                is_js_rendered = bool(js_quality["is_js_rendered"])

                html_res = extract_main_text_from_html(
                    raw_html,
                    url=final_url,
                    max_excerpt_len=5000,
                )

                extracted_text = html_res.extracted_text
                excerpt = html_res.excerpt
                not_citable_reason = html_res.not_citable_reason
                quality_issues = list(html_res.quality_issues)

                # If JS-rendered template, try Playwright fallback before giving up
                if is_js_rendered or js_quality["quality_score"] < 0.3:
                    pw_ok = False
                    # Only try Playwright for non-portal domains (portals are
                    # intentionally skipped even with JS rendering).
                    pw_enabled = settings.playwright_enabled
                    if pw_enabled and not any(
                        d in final_url for d in _JS_HEAVY_DOMAINS
                    ):
                        pw_resp = await fetch_with_playwright(
                            final_url, timeout=30.0,
                        )
                        if pw_resp and pw_resp.status_code == 200:
                            pw_html = extract_main_text_from_html(
                                pw_resp.text,
                                url=final_url,
                                max_excerpt_len=5000,
                            )
                            if (
                                pw_html.extracted_text
                                and len(pw_html.extracted_text) > 200
                            ):
                                extracted_text = pw_html.extracted_text
                                excerpt = pw_html.excerpt
                                not_citable_reason = pw_html.not_citable_reason
                                quality_issues = list(pw_html.quality_issues)
                                quality_issues.append(
                                    "content rendered via headless browser "
                                    "(Playwright)"
                                )
                                quality_score = pw_html.quality_score
                                is_citable = pw_html.is_citable
                                is_js_rendered = False
                                pw_ok = True

                    if not pw_ok:
                        not_citable_reason = (
                            not_citable_reason
                            or "js-rendered template/empty placeholders "
                            "(not citable)"
                        )
                        quality_issues.append(
                            "⚠ LOW QUALITY: page requires JavaScript "
                            "rendering; fetched HTML is a template"
                        )
                        quality_score = 0.1
                        is_citable = False
                else:
                    quality_score = html_res.quality_score
                    is_citable = html_res.is_citable

                extracted_len = len(extracted_text)
                evidence_snippets = _collect_evidence_snippets(extracted_text)
            else:
                # Non-text/binary payloads are not citable by this tool
                extracted_text = ""
                excerpt = ""
                not_citable_reason = f"non-text content-type: {content_type}"
                quality_issues = [not_citable_reason]
                quality_score = 0.1
                is_citable = False
                extracted_len = 0

            # ── Financial table extraction for SEC / regulatory filings ──
            # When the content is from a large filing, extract financial statement
            # sections and append them.  This ensures key tables (income statement,
            # balance sheet, cash flow) are captured even if truncated from the excerpt.
            financial_tables = ""
            _url_l = final_url.lower()
            _is_filing_url = any(kw in _url_l for kw in [
                "sec.gov/archives/edgar/data/",
                "hkexnews.hk/listedco/",
                "form20-f", "form10-k", "20-f", "10-k",
            ])
            if _is_filing_url and extracted_text and len(extracted_text) > 5000:
                financial_tables = extract_financial_tables(extracted_text)
                if financial_tables:
                    quality_issues.append(
                        f"financial tables extracted ({len(financial_tables)} chars)"
                    )
                    # Append financial tables to evidence snippets as a special entry
                    evidence_snippets.append({
                        "term": "financial_statements",
                        "snippet": financial_tables[:3000],
                    })

            result = {
                "url": final_url,
                "requested_url": requested_url,
                "status_code": response.status_code,
                "content_type": content_type,
                "is_html": bool(is_html),
                "is_pdf": bool(is_pdf),
                # Backward-compatible field: "content" is ALWAYS readable plain text excerpt now.
                "content": excerpt,
                # New fields
                "extracted_text": extracted_text,
                "excerpt": excerpt,
                "focus_terms": focus_terms or [],
                "evidence_snippets": evidence_snippets,
                "is_citable": bool(is_citable),
                "not_citable_reason": not_citable_reason,
                "quality_score": float(quality_score),
                "is_js_rendered": bool(is_js_rendered),
                "quality_issues": quality_issues,
                "extracted_text_length": extracted_len,
                "financial_tables": financial_tables or None,
            }

            if not is_citable:
                result["warning"] = (
                    "⚠ NOT CITABLE: The fetched page is not suitable as a verified source. "
                    "Reason: "
                    + (not_citable_reason or "unknown")
                    + ". Prefer official filings, press releases, or other accessible sources."
                )

            return result

    except TimeoutError:
        return {
            "url": url,
            "requested_url": url,
            "error": "Request timed out",
            "status_code": None,
        }
    except Exception as e:
        return {
            "url": url,
            "requested_url": url,
            "error": str(e),
            "status_code": None,
        }
