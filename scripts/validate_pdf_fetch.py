#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate that fetch_url extracts readable text from PDFs.

This script verifies two things:
1) Direct PDF fetch via fetch_url returns readable excerpt (not %PDF binary).
2) web_search can find a PDF URL and fetch_url can extract it (search->fetch flow).
"""

import asyncio

from agent_engine.tools.builtin.web_search import fetch_url, web_search


async def main() -> None:
    pdf_candidates = [
        # Common public sample PDFs (try several in case of network blocks)
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "https://www.africau.edu/images/default/sample.pdf",
        "https://www.orimi.com/pdf-test.pdf",
    ]

    r = None
    for pdf_url in pdf_candidates:
        r = await fetch_url(pdf_url)
        if r.get("error") is None and r.get("status_code") == 200:
            break
    assert r is not None

    print("=== fetch_url(PDF) ===")
    print("url:", r.get("url"))
    print("status_code:", r.get("status_code"))
    print("error:", r.get("error"))
    print("content_type:", r.get("content_type"))
    print("is_pdf:", r.get("is_pdf"))
    print("is_citable:", r.get("is_citable"))
    print("quality_score:", r.get("quality_score"))
    print("not_citable_reason:", r.get("not_citable_reason"))
    content_preview = (r.get("content") or "").replace("\n", " ")[:200]
    extracted_preview = (r.get("extracted_text") or "").replace("\n", " ")[:200]
    print("content_preview:", content_preview)
    print("extracted_preview:", extracted_preview)
    assert r.get("error") is None and r.get("status_code") == 200, "fetch_url must succeed"
    assert r.get("is_pdf") is True, "should detect PDF content"
    assert (r.get("extracted_text") or "").strip(), "extracted_text should not be empty"
    assert not content_preview.lstrip().startswith("%PDF"), "content should be readable text excerpt"

    print("\n=== web_search -> fetch_url ===")
    # Use a query that tends to yield direct .pdf URLs
    s = await web_search("orimi pdf-test.pdf", num_results=5)
    results = s.get("results", []) or []
    print("search_results:", len(results))
    pdfs = [
        x for x in results
        if (x.get("url", "").lower().endswith(".pdf") or ".pdf" in x.get("url", "").lower())
    ]
    assert pdfs, "Expected at least one PDF URL in search results"

    r2 = None
    picked = None
    for item in pdfs:
        picked = item.get("url")
        if not picked:
            continue
        print("trying:", picked)
        r2 = await fetch_url(picked)
        if r2.get("error") is None and r2.get("status_code") == 200:
            break

    assert r2 is not None and picked is not None
    print("picked:", picked)
    print("status_code:", r2.get("status_code"))
    print("error:", r2.get("error"))
    print("is_pdf:", r2.get("is_pdf"))
    print("is_citable:", r2.get("is_citable"))
    print("quality_score:", r2.get("quality_score"))
    content_preview2 = (r2.get("content") or "").replace("\n", " ")[:200]
    print("content_preview:", content_preview2)
    assert r2.get("error") is None and r2.get("status_code") == 200, "fetch_url must succeed for at least one search PDF"
    assert r2.get("is_pdf") is True, "should detect PDF content"
    assert (r2.get("extracted_text") or "").strip(), "extracted_text should not be empty"
    assert not content_preview2.lstrip().startswith("%PDF"), "content should be readable text excerpt"

    print("\nOK: PDF content is extracted as readable text.")


if __name__ == "__main__":
    asyncio.run(main())

