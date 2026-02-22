"""RAG-based semantic document search: chunked indexing + vector retrieval.

Solves the "information blackhole" problem where fetch_url truncates large
documents (SEC 10-K/20-F, HKEX annual reports) to a fixed number of chars,
missing financial tables and data buried deep in the file.

Three-phase progressive retrieval:
  Phase 0 — Structure scan (PDF only, zero API cost):
    Use PyMuPDF's built-in get_toc() to read the document's table of contents.
    Match TOC entries against query keywords → identify relevant page ranges.
    Extract ONLY those pages (typically 10-30 pages out of 200+) instead of
    the entire document.  Falls back gracefully when no TOC is available.

  Phase 1 — TOC chunk scan (HTML / PDFs without built-in TOC):
    After chunking, scan the first 30% of chunks for table-of-contents patterns.
    Extract relevant section titles as "anchor queries".

  Phase 2 — Multi-query vector retrieval:
    Embed original query + anchor queries; score every chunk against ALL queries;
    return top-k by max score.  This union-of-relevance approach ensures chunks
    that are specifically relevant to any identified section are boosted.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import numpy as np
import tiktoken

from agent_engine.config import get_settings
from agent_engine.tools.registry import tool
from agent_engine.tools.builtin.content_extract import (
    extract_main_text_from_html,
    extract_pdf_text_from_bytes,
)

logger = logging.getLogger(__name__)

# ── DashScope embedding endpoint ──────────────────────────────────────────────
_DASHSCOPE_EMBED_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)

# ── Chunking parameters ───────────────────────────────────────────────────────
_CHUNK_TOKENS = 400        # tokens per chunk
_CHUNK_OVERLAP = 64        # overlap between consecutive chunks
_MIN_CHUNK_CHARS = 80      # discard whitespace-only micro-chunks
_MAX_DOC_CHARS = 600_000   # ~150k tokens – enough for a full SEC 10-K
_EMBED_BATCH = 8           # texts per embedding API call (DashScope max is 10)

# ── Progressive PDF extraction limits ─────────────────────────────────────────
# Maximum pages to extract per matched TOC section (safety cap)
_MAX_SECTION_PAGES = 60
# Minimum TOC match score to use a section (keyword overlap ratio)
_MIN_TOC_MATCH_SCORE = 0.10
# Maximum total chars from targeted section extraction (feeds into chunker)
_MAX_TARGETED_CHARS = 150_000

# ── TOC detection (text-based fallback for HTML / PDFs without embedded TOC) ──
_TOC_PATTERNS = [
    re.compile(r'(?:目\s*录|table\s+of\s+contents|contents|index)\s*[\n:]', re.I),
    re.compile(r'(?:第[一二三四五六七八九十\d]+[章节部分]|chapter\s+\d|section\s+\d)', re.I),
]
_SECTION_LINE_RE = re.compile(
    r'(?:'
    r'(?:\d+[\.\、])+\s*[\u4e00-\u9fff\w].{2,60}'
    r'|第[一二三四五六七八九十\d]+[章节部分]\s*.{2,40}'
    r'|(?:Part|Chapter|Section|Item)\s+[IVX\d]+\.?\s*.{2,40}'
    r')',
    re.I | re.MULTILINE,
)

# ── Financial section keywords for TOC matching ────────────────────────────────
# Used to score each TOC entry against a query
_FINANCIAL_KEYWORDS = {
    # English — individual terms (used for single-token matching)
    "revenue", "revenues", "sales", "income", "loss", "losses", "profit", "profits",
    "earnings", "ebitda", "r&d", "research", "development",
    "expense", "expenses", "cost", "costs", "margin", "margins",
    "cash", "flow", "flows", "balance", "debt", "equity", "asset", "assets",
    "liability", "liabilities", "financial", "statement", "statements",
    "management", "discussion", "analysis", "md&a", "notes", "supplementary",
    # Chinese — individual terms
    "营收", "收入", "利润", "亏损", "研发", "费用", "现金", "资产", "负债",
    "财务", "报表", "业绩", "毛利", "经营", "管理", "讨论", "分析",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _embed_texts(
    texts: list[str],
    api_key: str,
    model: str,
    dimension: int,
) -> list[list[float]]:
    """Call DashScope text-embedding API in batches, return list of vectors."""
    all_vecs: list[list[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            resp = await client.post(
                _DASHSCOPE_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": {"texts": batch},
                    "parameters": {"dimension": dimension},
                },
            )
            if resp.status_code != 200:
                logger.warning(
                    "[RAG] Embedding error %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                all_vecs.extend([[0.0] * dimension] * len(batch))
                continue
            data = resp.json()
            for emb_obj in data.get("output", {}).get("embeddings", []):
                all_vecs.append(emb_obj.get("embedding", [0.0] * dimension))
    return all_vecs


def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping token-based chunks."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + _CHUNK_TOKENS, len(tokens))
            chunk = enc.decode(tokens[start:end]).strip()
            if len(chunk) >= _MIN_CHUNK_CHARS:
                chunks.append(chunk)
            start += _CHUNK_TOKENS - _CHUNK_OVERLAP
        return chunks
    except Exception:
        chunk_chars = _CHUNK_TOKENS * 4
        overlap_chars = _CHUNK_OVERLAP * 4
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_chars].strip()
            if len(chunk) >= _MIN_CHUNK_CHARS:
                chunks.append(chunk)
            start += chunk_chars - overlap_chars
        return chunks


def _score_toc_entry(title: str, query: str) -> float:
    """Compute keyword-overlap relevance between a TOC entry title and the query.

    Uses three scoring signals and returns their max:
    1. Direct word overlap between title and query
    2. Financial-keyword topic overlap (both sides must have finance keywords)
    3. Substring containment: any query word (≥4 chars) contained in the title
       — handles "Financial" matching "financials", "expense" matching "expenses"

    Returns a float 0.0–1.0.
    """
    t_lower = title.lower()
    q_lower = query.lower()

    # Tokenise: keep compound tokens like "r&d", "non-gaap" intact
    _tok = lambda s: set(re.sub(r"[^\w&\-\u4e00-\u9fff]", " ", s.lower()).split())
    q_words = _tok(query)
    t_words = _tok(title)

    # Remove trivial stop-words from comparison
    _stops = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "and",
               "or", "is", "are", "was", "were", "this", "that", "from", "with", "its"}
    q_sig = q_words - _stops
    t_sig = t_words - _stops

    # 1. Direct word overlap
    direct_overlap = len(q_sig & t_sig) / max(len(q_sig), 1) if q_sig else 0.0

    # 2. Financial keyword topic overlap
    q_fin = q_sig & _FINANCIAL_KEYWORDS
    t_fin = t_sig & _FINANCIAL_KEYWORDS
    if q_fin and t_fin:
        fin_overlap = len(q_fin & t_fin) / max(len(q_fin), 1)
        # Partial topic match: if EITHER side has finance keywords and the other
        # also has finance keywords (but different ones), score is still positive.
        # E.g. query "r&d expenses" vs title "Financial Statements"
        topic_match = min(len(q_fin), len(t_fin)) / max(len(q_fin | t_fin), 1) * 0.5
    else:
        fin_overlap = 0.0
        topic_match = 0.0

    # 3. Substring containment — handles plurals, compound phrases, partial stems
    # e.g. query word "expenses" matches title word "expense" or "financial expenses"
    contains_score = 0.0
    long_q_words = [w for w in q_sig if len(w) >= 4 and w not in _stops]
    if long_q_words:
        matched = 0
        for w in long_q_words:
            # exact substring match
            if w in t_lower:
                matched += 1
            # prefix/stem match: "expenses" starts with "expense" (≥5 chars stem)
            elif len(w) >= 5 and any(t_w.startswith(w[:5]) or w.startswith(t_w[:5])
                                     for t_w in t_sig if len(t_w) >= 4):
                matched += 0.6
        contains_score = matched / len(long_q_words)

    return max(direct_overlap, fin_overlap, topic_match, contains_score * 0.75)


# ── Phase 0: PDF structure extraction via PyMuPDF ─────────────────────────────

def _pdf_progressive_extract(
    pdf_bytes: bytes,
    query: str,
) -> tuple[str, list[dict], str]:
    """Attempt structured TOC-guided extraction from a PDF.

    Returns:
        (text, toc_entries, strategy) where:
        - text: extracted text from targeted pages (empty string on failure)
        - toc_entries: list of {title, page, score} for matched TOC entries
        - strategy: "toc_targeted" | "full_fallback"

    Uses PyMuPDF's get_toc() to read the embedded table of contents, then
    extracts only the pages belonging to the highest-scoring sections.
    Falls back to full-document extraction if no usable TOC is found.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.info("[RAG] PyMuPDF not available — using full-text extraction")
        return "", [], "full_fallback"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("[RAG] fitz.open failed: %s", exc)
        return "", [], "full_fallback"

    total_pages = len(doc)
    toc_raw = doc.get_toc()  # [[level, title, page_num], ...]

    if not toc_raw:
        logger.info("[RAG] PDF has no embedded TOC (%d pages) — full extraction", total_pages)
        doc.close()
        return "", [], "full_fallback"

    logger.info("[RAG] PDF TOC: %d entries, %d total pages", len(toc_raw), total_pages)

    # Score every TOC entry
    scored_entries: list[dict] = []
    for level, title, page_num in toc_raw:
        score = _score_toc_entry(title, query)
        scored_entries.append({
            "level": level,
            "title": title,
            "page": page_num,       # 1-indexed in PyMuPDF
            "score": score,
        })

    # Keep entries above threshold, plus always include high-level entries that
    # are parents of matched entries (to preserve financial section context)
    threshold = max(_MIN_TOC_MATCH_SCORE, sorted([e["score"] for e in scored_entries], reverse=True)[0] * 0.4)
    matched = [e for e in scored_entries if e["score"] >= threshold]

    if not matched:
        logger.info("[RAG] No TOC entries matched query %r — full extraction", query[:50])
        doc.close()
        return "", [], "full_fallback"

    logger.info(
        "[RAG] Matched %d TOC sections: %s",
        len(matched),
        [f"{e['title'][:30]}(p{e['page']},s={e['score']:.2f})" for e in matched[:5]],
    )

    # Determine page ranges for matched sections:
    # Each section spans from its start page to (next sibling's start page - 1)
    pages_to_extract: set[int] = set()
    toc_pages = [e["page"] for e in scored_entries]

    for entry in matched:
        start_page = entry["page"] - 1  # convert to 0-indexed
        # Find the next entry at the same or higher level (same-level sibling / parent)
        entry_idx = next(
            (i for i, e in enumerate(scored_entries) if e["page"] == entry["page"] and e["title"] == entry["title"]),
            None,
        )
        end_page = total_pages  # default: to end of document
        if entry_idx is not None:
            for sibling in scored_entries[entry_idx + 1:]:
                if sibling["level"] <= entry["level"]:
                    end_page = sibling["page"] - 1  # exclusive, 0-indexed
                    break

        # Cap per-section page count
        section_page_count = end_page - start_page
        if section_page_count > _MAX_SECTION_PAGES:
            end_page = start_page + _MAX_SECTION_PAGES
            logger.info(
                "[RAG] Section '%s' capped at %d pages (was %d)",
                entry["title"][:40], _MAX_SECTION_PAGES, section_page_count,
            )

        for p in range(max(0, start_page), min(total_pages, end_page)):
            pages_to_extract.add(p)

    # Extract text from targeted pages only
    text_parts: list[str] = []
    total_chars = 0
    for page_num in sorted(pages_to_extract):
        page = doc[page_num]
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{page_text.strip()}")
            total_chars += len(page_text)
        if total_chars >= _MAX_TARGETED_CHARS:
            logger.info("[RAG] Targeted extraction hit char limit (%d chars)", total_chars)
            break

    doc.close()

    targeted_text = "\n\n".join(text_parts)
    if not targeted_text.strip():
        return "", [], "full_fallback"

    logger.info(
        "[RAG] Targeted PDF extraction: %d pages → %d chars (vs full doc ~%d pages)",
        len(pages_to_extract), len(targeted_text), total_pages,
    )
    return targeted_text, matched, "toc_targeted"


# ── Phase 1: Text-based TOC anchor extraction (HTML / PDF without embedded TOC) ──

def _is_toc_chunk(chunk: str) -> bool:
    return any(pat.search(chunk) for pat in _TOC_PATTERNS)


def _extract_toc_anchors(chunks: list[str], query: str, max_anchors: int = 3) -> list[str]:
    """Scan first 30% of chunks for TOC patterns; extract relevant section titles."""
    scan_limit = max(5, len(chunks) // 3)
    toc_sections: list[str] = []

    for chunk in chunks[:scan_limit]:
        if not _is_toc_chunk(chunk):
            continue
        for m in _SECTION_LINE_RE.finditer(chunk):
            title = m.group(0).strip()
            if title and len(title) > 4:
                toc_sections.append(title)

    if not toc_sections:
        return []

    scored = [
        (_score_toc_entry(title, query), title)
        for title in toc_sections
        if _score_toc_entry(title, query) > 0
    ]
    scored.sort(reverse=True)
    anchors = [t for _, t in scored[:max_anchors]]

    if anchors:
        logger.info("[RAG] Text-TOC anchors for %r: %s", query[:40], anchors)
    return anchors


# ── Phase 2: Multi-query vector retrieval ─────────────────────────────────────

def _cosine_scores(
    chunk_vecs: np.ndarray,
    query_vec: np.ndarray,
) -> np.ndarray:
    chunk_norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True).clip(min=1e-9)
    query_norm = float(np.linalg.norm(query_vec)) or 1e-9
    return (chunk_vecs / chunk_norms) @ (query_vec / query_norm)


# ── Public tool ───────────────────────────────────────────────────────────────

@tool(
    name="search_document",
    description=(
        "Semantic search inside a LONG document (SEC 10-K/20-F, HKEX annual report, PDF). "
        "Unlike fetch_url which truncates to the first N characters, search_document uses "
        "a THREE-PHASE progressive approach: "
        "(1) Structure scan — for PDFs, reads the built-in table of contents (TOC) to "
        "identify which pages contain the target data, then extracts ONLY those pages "
        "(typically 10-30 pages from a 200-page filing) — zero embedding cost for this step; "
        "(2) TOC anchor extraction — for HTML or PDFs without embedded TOC, scans the "
        "document's text TOC to find relevant section titles as anchor queries; "
        "(3) Multi-query vector search — embeds the original query plus TOC anchors, "
        "scores all chunks against all queries, returns top-k by max relevance score. "
        "Use this when: (1) fetch_url shows a TRUNCATED warning; "
        "(2) you need specific financial figures buried deep in a filing; "
        "(3) the document is >30 pages. "
        "Args: url (str) — document URL; query (str) — what to find "
        "(e.g. 'R&D expenses 2024 full year'); top_k (int, default 5) — chunks to return."
    ),
)
async def search_document(
    url: str,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Three-phase progressive document search.

    Phase 0 (PDF + PyMuPDF): Read built-in TOC → identify relevant page ranges
        → extract only those pages.  This is zero-API-cost and very precise.
        Falls back to full extraction if no embedded TOC exists.

    Phase 1 (text TOC scan): Chunk the (possibly pre-narrowed) text → scan
        early chunks for TOC-like patterns → extract section title anchors.

    Phase 2 (multi-query vector search): Embed original query + anchors → score
        all chunks against all queries → return top-k by max score.

    Args:
        url:   URL of the document (PDF, HTML page, or SEC EDGAR filing).
        query: What to look for (e.g. "R&D expenses full year 2024").
        top_k: Chunks to return (default 5, max 10).

    Returns:
        dict with:
          - relevant_chunks: list of {rank, score, text}
          - combined_text: top chunks joined with separators
          - total_chunks: chunks in the (possibly pre-narrowed) text
          - document_length: chars before chunking
          - extraction_strategy: "toc_targeted" | "full_fallback"
          - toc_matched_sections: TOC entries used for page targeting (PDF only)
          - toc_anchors: text-TOC section titles used as extra queries
          - url: source URL
    """
    settings = get_settings()
    api_key = settings.dashscope_api_key
    if not api_key:
        return {"error": "DashScope API key not configured (DASHSCOPE_API_KEY in .env)"}

    top_k = min(max(1, top_k), 10)

    # ── Fetch document ─────────────────────────────────────────────────────────
    from agent_engine.tools.builtin.http_client import SmartHttpClient

    # PDFs are often large (annual reports, filings); use longer timeout and more retries
    is_pdf_url = url.lower().split("?")[0].endswith(".pdf")
    fetch_timeout = 120 if is_pdf_url else 60
    fetch_retries = 4 if is_pdf_url else 2

    logger.info("[RAG] Fetching: %s (timeout=%ds, retries=%d)", url, fetch_timeout, fetch_retries)
    try:
        async with SmartHttpClient(timeout=fetch_timeout, max_retries=fetch_retries) as client:
            resp = await client.get(url)
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code} fetching {url}"}
    except Exception as exc:
        return {"error": f"Failed to fetch {url}: {exc}"}

    content_type = resp.headers.get("content-type", "").lower()
    is_pdf = "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")

    # ── Phase 0: PDF TOC-guided targeted extraction ────────────────────────────
    extraction_strategy = "full_fallback"
    toc_matched_sections: list[dict] = []
    text = ""

    if is_pdf:
        targeted_text, toc_matched_sections, extraction_strategy = _pdf_progressive_extract(
            resp.content, query
        )
        if targeted_text:
            text = targeted_text
            if len(text) > _MAX_TARGETED_CHARS:
                text = text[:_MAX_TARGETED_CHARS]

    # Full extraction fallback (HTML, or PDF without usable TOC)
    if not text:
        if is_pdf:
            extract_result = extract_pdf_text_from_bytes(
                resp.content, max_pages=100, max_chars=_MAX_DOC_CHARS
            )
            text = extract_result.extracted_text
        else:
            extract_result = extract_main_text_from_html(resp.text, url=url)
            text = extract_result.extracted_text

        if len(text) > _MAX_DOC_CHARS:
            logger.warning("[RAG] Document truncated from %d to %d chars", len(text), _MAX_DOC_CHARS)
            text = text[:_MAX_DOC_CHARS]

    if not text or len(text) < 200:
        return {"error": f"No meaningful text extracted from {url} (got {len(text)} chars)"}

    logger.info("[RAG] Text ready: %d chars (strategy=%s)", len(text), extraction_strategy)

    # ── Chunk ──────────────────────────────────────────────────────────────────
    chunks = _chunk_text(text)
    if not chunks:
        return {"error": "Document produced no valid chunks"}

    logger.info("[RAG] %d chunks", len(chunks))

    # ── Phase 1: Text-based TOC anchor extraction ──────────────────────────────
    toc_anchors = _extract_toc_anchors(chunks, query)

    # ── Phase 2: Multi-query vector retrieval ──────────────────────────────────
    queries = [query] + toc_anchors
    all_texts = chunks + queries

    try:
        embeddings = await _embed_texts(
            all_texts,
            api_key=api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
    except Exception as exc:
        return {"error": f"Embedding API failed: {exc}"}

    n_chunks = len(chunks)
    n_queries = len(queries)
    if len(embeddings) < n_chunks + 1:
        return {"error": "Embedding API returned insufficient vectors"}

    chunk_vecs = np.array(embeddings[:n_chunks], dtype=np.float32)
    query_vecs = np.array(embeddings[n_chunks:n_chunks + n_queries], dtype=np.float32)

    # Score every chunk against ALL queries; take the max (union of relevance)
    all_scores = np.stack(
        [_cosine_scores(chunk_vecs, query_vecs[i]) for i in range(n_queries)],
        axis=1,
    )  # (n_chunks, n_queries)
    max_scores = all_scores.max(axis=1)

    # Return top-k
    actual_k = min(top_k, len(chunks))
    top_idx = np.argsort(max_scores)[-actual_k:][::-1]
    relevant = [
        {"rank": i + 1, "score": float(max_scores[idx]), "text": chunks[idx]}
        for i, idx in enumerate(top_idx)
    ]
    combined = "\n\n---\n\n".join(
        f"[Rank {c['rank']}, relevance={c['score']:.3f}]\n{c['text']}"
        for c in relevant
    )

    logger.info(
        "[RAG] Done. strategy=%s scores=%s anchors=%d",
        extraction_strategy,
        [f"{c['score']:.3f}" for c in relevant],
        len(toc_anchors),
    )

    return {
        "url": url,
        "query": query,
        "extraction_strategy": extraction_strategy,
        "toc_matched_sections": [
            {"title": e["title"], "page": e["page"], "score": round(e["score"], 3)}
            for e in toc_matched_sections[:10]
        ],
        "toc_anchors": toc_anchors,
        "relevant_chunks": relevant,
        "combined_text": combined,
        "total_chunks": len(chunks),
        "document_length": len(text),
    }
