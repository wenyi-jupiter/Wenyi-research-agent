"""SEC EDGAR structured data tool.

Fetches financial data directly from SEC's XBRL/JSON APIs, bypassing the
need to parse multi-MB HTML filings.  This solves the "truncation problem"
where key financial tables appear deep in a filing and get cut off by
content extraction limits.

Data sources:
- Company Facts API: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
- Submissions API:   https://data.sec.gov/submissions/CIK{cik}.json

SEC Fair Access policy requires a User-Agent with contact info.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from agent_engine.tools.registry import tool

logger = logging.getLogger(__name__)

_SEC_UA = (
    "AgentEngine/1.0 (research-agent; contact@example.com)"
)
_TIMEOUT = 20.0

# Common XBRL taxonomy concepts for financial statement line items.
# These are US-GAAP standard tags that appear in most 10-K / 20-F filings.
_COMMON_CONCEPTS = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:SalesRevenueNet",
    ],
    "net_income": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
    ],
    "total_assets": [
        "us-gaap:Assets",
    ],
    "total_liabilities": [
        "us-gaap:Liabilities",
    ],
    "stockholders_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsAndShortTermInvestments",
    ],
    "operating_cash_flow": [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    ],
    "rd_expenses": [
        "us-gaap:ResearchAndDevelopmentExpense",
    ],
    "operating_income": [
        "us-gaap:OperatingIncomeLoss",
    ],
    "eps_basic": [
        "us-gaap:EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "us-gaap:EarningsPerShareDiluted",
    ],
}


def _normalize_cik(cik_or_ticker: str) -> str | None:
    """Normalize a CIK string to 10-digit zero-padded format.

    Accepts:
      - Pure digits: "1651308" -> "0001651308"
      - CIK prefix:  "CIK0001651308" -> "0001651308"
    """
    s = cik_or_ticker.strip().upper()
    s = re.sub(r"^CIK", "", s)
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return None
    return s.zfill(10)


@tool(
    name="sec_edgar_financials",
    description=(
        "Fetch structured financial data from SEC EDGAR's XBRL API for a given "
        "company CIK number. Returns key financial metrics (revenue, net income, "
        "cash, R&D expenses, etc.) as structured data — no HTML/PDF parsing needed. "
        "This is the PREFERRED way to get US-listed company financials. "
        "You need the company's CIK number (e.g., '1651308' for BeiGene). "
        "Use web_search to find a company's CIK if you don't know it."
    ),
    tags=["sec", "edgar", "financial", "xbrl"],
)
async def sec_edgar_financials(
    cik: str,
    metrics: list[str] | None = None,
    fiscal_year: int | None = None,
) -> dict[str, Any]:
    """Fetch structured financial data from SEC EDGAR XBRL API.

    Args:
        cik: SEC CIK number (e.g., "1651308"). Can include "CIK" prefix.
        metrics: Optional list of metric names to retrieve. If None, returns
                 all common metrics. Available: revenue, net_income, total_assets,
                 total_liabilities, stockholders_equity, cash_and_equivalents,
                 operating_cash_flow, rd_expenses, operating_income, eps_basic,
                 eps_diluted.
        fiscal_year: Optional fiscal year to filter results (e.g., 2023).
                     If None, returns the most recent 5 years.

    Returns:
        Dict with structured financial data, source URL, and metadata.
    """
    normalized_cik = _normalize_cik(cik)
    if not normalized_cik:
        return {
            "success": False,
            "error": f"Invalid CIK: {cik!r}. Must be a numeric CIK number.",
        }

    api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                api_url,
                headers={"User-Agent": _SEC_UA, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {
                "success": False,
                "error": f"CIK {normalized_cik} not found on SEC EDGAR.",
                "url": api_url,
            }
        return {
            "success": False,
            "error": f"SEC EDGAR API error: {e.response.status_code}",
            "url": api_url,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch SEC data: {e}"}

    # Extract company info
    entity_name = data.get("entityName", "")
    cik_str = data.get("cik", normalized_cik)

    # Get the US-GAAP facts
    us_gaap = data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        # Try ifrs-full for foreign private issuers
        us_gaap = data.get("facts", {}).get("ifrs-full", {})

    if not us_gaap:
        return {
            "success": True,
            "entity_name": entity_name,
            "cik": cik_str,
            "url": api_url,
            "warning": "No US-GAAP or IFRS data found. Company may not have XBRL filings.",
            "financials": {},
        }

    # Determine which metrics to extract
    requested = metrics or list(_COMMON_CONCEPTS.keys())
    financials: dict[str, list[dict]] = {}

    for metric_name in requested:
        concepts = _COMMON_CONCEPTS.get(metric_name, [])
        if not concepts:
            continue

        for concept_tag in concepts:
            # Strip namespace prefix for lookup
            bare_concept = concept_tag.split(":")[-1] if ":" in concept_tag else concept_tag
            concept_data = us_gaap.get(bare_concept, {})
            if not concept_data:
                continue

            units_data = concept_data.get("units", {})
            label = concept_data.get("label", bare_concept)

            # Try USD first, then shares, then pure
            for unit_key in ["USD", "USD/shares", "shares", "pure"]:
                entries = units_data.get(unit_key, [])
                if entries:
                    # Filter to annual filings (10-K / 20-F) only
                    annual_entries = [
                        e for e in entries
                        if e.get("form") in ("10-K", "20-F", "10-K/A", "20-F/A")
                        and "frame" in e  # Only entries with fiscal year frame
                    ]

                    # Apply fiscal year filter
                    if fiscal_year:
                        annual_entries = [
                            e for e in annual_entries
                            if str(fiscal_year) in (e.get("frame", "") or "")
                        ]

                    if not annual_entries:
                        # Fallback: also try entries without frame but with fy
                        annual_entries = [
                            e for e in entries
                            if e.get("form") in ("10-K", "20-F", "10-K/A", "20-F/A")
                            and (
                                not fiscal_year
                                or str(fiscal_year) in str(e.get("fy", ""))
                            )
                        ]

                    # Sort by date descending, take last 5 years
                    annual_entries.sort(key=lambda e: e.get("end", ""), reverse=True)
                    if not fiscal_year:
                        annual_entries = annual_entries[:5]

                    if annual_entries:
                        financials[metric_name] = [
                            {
                                "value": e.get("val"),
                                "unit": unit_key,
                                "period_end": e.get("end", ""),
                                "period_start": e.get("start", ""),
                                "fiscal_year": e.get("fy"),
                                "form": e.get("form", ""),
                                "filed": e.get("filed", ""),
                                "accession": e.get("accn", ""),
                                "label": label,
                                "concept": bare_concept,
                            }
                            for e in annual_entries
                        ]
                        break  # Found data for this unit

            if metric_name in financials:
                break  # Found data for this concept

    # Build a human-readable summary for the LLM
    summary_lines = [
        f"## {entity_name} (CIK: {cik_str}) — SEC EDGAR Financial Data",
        "",
    ]
    for metric_name, entries in financials.items():
        summary_lines.append(f"### {metric_name.replace('_', ' ').title()}")
        for e in entries[:3]:
            val = e["value"]
            unit = e["unit"]
            fy = e.get("fiscal_year", "?")
            end = e.get("period_end", "?")
            # Build a human-readable label AND keep the raw integer so that P4
            # can do exact substring matching against what the LLM copies verbatim.
            if unit == "USD" and isinstance(val, (int, float)):
                if abs(val) >= 1e9:
                    val_str = f"${val/1e9:,.2f} billion (raw: {int(val):,} USD)"
                elif abs(val) >= 1e6:
                    val_str = f"${val/1e6:,.2f} million (raw: {int(val):,} USD)"
                else:
                    val_str = f"${val:,.0f} USD"
            elif unit == "USD/shares" and isinstance(val, (int, float)):
                val_str = f"${val:.2f}/share"
            else:
                val_str = f"{val} {unit}"
            summary_lines.append(f"  FY{fy} (ending {end}): {val_str}")
        summary_lines.append("")

    return {
        "success": True,
        "entity_name": entity_name,
        "cik": cik_str,
        "url": api_url,
        "financials": financials,
        "summary": "\n".join(summary_lines),
        "metrics_found": list(financials.keys()),
        "metrics_missing": [m for m in requested if m not in financials],
    }


@tool(
    name="sec_edgar_filings",
    description=(
        "Look up a company's recent SEC filings (10-K, 20-F, 10-Q, DEF 14A, etc.) "
        "by CIK number. Returns filing dates, accession numbers, and direct links "
        "to the filing documents. Use this to find the correct filing URL before "
        "calling fetch_url."
    ),
    tags=["sec", "edgar", "filings"],
)
async def sec_edgar_filings(
    cik: str,
    filing_type: str | None = None,
    count: int = 5,
) -> dict[str, Any]:
    """Look up recent SEC filings for a company.

    Args:
        cik: SEC CIK number (e.g., "1651308").
        filing_type: Optional filter by filing type (e.g., "10-K", "20-F", "DEF 14A").
        count: Number of recent filings to return (default 5, max 20).

    Returns:
        Dict with recent filings and direct URLs.
    """
    normalized_cik = _normalize_cik(cik)
    if not normalized_cik:
        return {"success": False, "error": f"Invalid CIK: {cik!r}"}

    api_url = f"https://data.sec.gov/submissions/CIK{normalized_cik}.json"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                api_url,
                headers={"User-Agent": _SEC_UA, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"SEC EDGAR API error: {e.response.status_code}",
            "url": api_url,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch SEC data: {e}"}

    entity_name = data.get("name", "")
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    count = min(count, 20)
    filings = []
    for i in range(len(forms)):
        form = forms[i] if i < len(forms) else ""
        if filing_type and form.upper() != filing_type.upper():
            continue

        accession = accessions[i] if i < len(accessions) else ""
        accession_nodash = accession.replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(normalized_cik)}/{accession_nodash}/{primary_doc}"
        ) if primary_doc else ""

        filings.append({
            "form": form,
            "filing_date": dates[i] if i < len(dates) else "",
            "accession_number": accession,
            "primary_document": primary_doc,
            "description": descriptions[i] if i < len(descriptions) else "",
            "url": filing_url,
            "index_url": (
                f"https://www.sec.gov/cgi-bin/browse-edgar?"
                f"action=getcompany&CIK={normalized_cik}&type={form}"
                f"&dateb=&owner=include&count=5"
            ),
        })

        if len(filings) >= count:
            break

    return {
        "success": True,
        "entity_name": entity_name,
        "cik": normalized_cik,
        "url": api_url,
        "filings": filings,
        "total_filings_found": len(filings),
    }
