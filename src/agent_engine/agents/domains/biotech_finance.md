---
name: biotech_finance
priority: 10
signal_threshold: 2

signals:
  # Company / sector indicators
  - beone
  - beigene
  - 百济神州
  - bgne
  - biotech
  - biopharma
  - pharma
  # Regulatory / filing indicators
  - sec
  - edgar
  - hkex
  - 20-f
  - 10-k
  - ipo
  - annual report
  - 披露
  - 港交所
  # Clinical indicators
  - pipeline
  - 管线
  - 临床
  - clinical
  - fda
  - nmpa
  - ema
  # Investment intent
  - stock
  - invest
  - 投资
  - long-term
  - 长期持有
  # Financial data keywords
  - revenue
  - 营收
  - 研发
  - r&d

alias_sets:
  - signals: [beone, beigene, 百济神州, bgne, "688235"]
    canonical: "BeOne Medicines Ltd."
    aliases:
      - BeOne Medicines
      - BeiGene
      - 百济神州
      - BGNE
      - "688235"
    note: |
      BeOne Medicines is the post-2025 brand name. Historical SEC filings
      and older disclosures are filed under BeiGene Ltd.
      Search both names to ensure full coverage.

entity_patterns:
  - '\b[A-Z]{2,5}(?=\s+(?:stock|shares|nasdaq|nasdaq:|nyse:|hkex:))'
  - '\b\d{6}\.(?:SH|SZ|HK)\b'
  - '(?:nasdaq|nyse|hkex):\s*([A-Z0-9.]+)'

# Regex patterns that indicate a query segment contains searchable entity markers.
# Used by _is_instruction_only() to decide whether a clause has retrieval value.
entity_marker_patterns:
  - '\b(?:10-[KQ]|20-F|8-K|CIK|HKEX|FDA|EMA|NMPA|SEC|NDA|BLA|IND|ANDA|MAA)\b'
  - '\bPhase\s+[123I]+\b'
  - '\b(?:BGNE|6160|688235)\b'

focus_term_rules:
  - triggers:
      - 20-f
      - 10-k
      - annual report
      - 财报
      - 年报
      - 营收
      - revenue
      - financial result
      - earnings
      - sec filing
      - r&d
      - 研发
      - cash flow
      - 现金流
      - balance sheet
      - income statement
      - 利润
      - 亏损
      - 费用
    terms:
      - total revenue
      - net product revenues
      - product revenue
      - research and development expenses
      - R&D expenses
      - 研发
      - net loss
      - net income
      - loss from operations
      - operating loss
      - net cash used in operating activities

  - triggers: [cash, 现金, balance sheet, liquidity]
    terms: [cash and cash equivalents, restricted cash]

  - triggers: [gross, 毛利, margin]
    terms: [gross profit, gross margin]

  - triggers:
      - pipeline
      - 管线
      - 临床
      - clinical
      - phase
      - iii期
      - phase iii
      - phase 3
      - ii期
      - phase ii
      - phase 2
    terms:
      - pipeline
      - Phase 3
      - Phase III
      - Phase 2
      - Phase II
      - clinical trial
      - IND
      - NDA
      - BLA

  - triggers: [orr, pfs, efficacy, 疗效, 响应率, survival, 生存]
    terms:
      - overall response rate
      - progression-free survival
      - overall survival
      - ORR
      - PFS
      - OS
      - median PFS

low_quality_domains:
  - finance.yahoo.com/quote
  - google.com/finance/quote
  - laohu8.com/m/hq
  - xueqiu.com/S/
  - money.finance.sina.com.cn
  - quote.eastmoney.com
  - stockpage.10jqka.com.cn
  - investing.com/equities
  - tradingview.com/chart

official_domains:
  - sec.gov
  - hkexnews.hkex.com.hk
  - clinicaltrials.gov
  - fda.gov
  - nmpa.gov.cn

noise_patterns:
  - luigi mangione
  - crime
  - murder
  - celebrity gossip
  - entertainment
  - sports news

search_vocabulary:
  - SEC
  - EDGAR
  - HKEX
  - FDA
  - EMA
  - NMPA
  - 10-K
  - 10-Q
  - 20-F
  - 8-K
  - CIK
  - annual report
  - earnings
  - revenue
  - filing
  - disclosure
  - 年报
  - 季报
  - 财报
  - 披露
  - 临床
  - 管线

min_official_citation_ratio: 0.4
---

## Data Source Strategy

When planning research subtasks for a publicly listed biotech / pharma company:

**Priority order for authoritative data:**
1. **Regulatory filings (highest priority)**:
   - US: SEC EDGAR (sec.gov) — 20-F, 10-K, 8-K filings
   - HK: HKEX News (hkexnews.hkex.com.hk) — HK exchange announcements
   - Use `sec_edgar_filings` / `sec_edgar_financials` tools when available.
2. **Official press releases**: company IR pages, Businesswire, PRNewswire, GlobeNewswire.
3. **Reputable financial media**: Reuters, Bloomberg, WSJ, FT (as supplementary context only).
4. **Clinical data**: ClinicalTrials.gov, company pipeline pages, conference abstracts.

**AVOID these sources for core conclusions:**
- JavaScript-rendered real-time quote pages (Yahoo Finance quote tab,
  East Money real-time pages) — static fetch_url returns empty templates.
- Individual social media posts, blogs, or anonymous forums.
- Sources older than 24 months for financial/clinical data claims.

**Entity naming**: the company may have been renamed. Plan alias-aware searches
covering both current and historical names to avoid data gaps.

## Executor Hints

### Domain-specific query examples
- GOOD: `"ENTITY_NAME 2024 annual report SEC 20-F filing"` (entity + filing type + year)
- GOOD: `"ENTITY_NAME FDA approval 2024 press release"` (entity + regulatory event + year)
- BAD: `"sec.gov"` (bare domain — use `fetch_url` with a full SEC URL instead)

### Specialized data tools
- **For US-listed company financials**: Use `sec_edgar_financials` with the company CIK number.
  Returns structured XBRL data (revenue, net income, R&D spend, cash) directly from SEC filings.
  Use `sec_edgar_filings` first to get the CIK and accession numbers.

### Derived calculations
When you have raw financial data, compute derived metrics rather than leaving them unknown:
- Gross margin = (Revenue − COGS) / Revenue
- YoY growth = (Current − Prior) / Prior × 100
- Debt-to-equity = Total liabilities / Stockholders' equity
- End-of-period cash = Opening cash + Operating CF + Investing CF + Financing CF + FX

### Unit and currency rules
- Quote all values in the EXACT currency and unit as written in the source.
- WRONG: source says "18.859 billion yuan" → you write "18.86亿美元"
- RIGHT: source says "18.859 billion yuan" → you write "18.859 billion yuan" or "188.59亿元"
- NEVER convert CNY↔USD, EUR↔GBP, or 万↔million without explicit source support.

## Reporter Hints

### Core conclusion citation priority
- Financial figures, regulatory approval dates, and clinical data MUST cite
  regulator/company-official sources (SEC EDGAR, HKEX, company IR pages).
- Third-party media may support qualitative context only, not core numbers.

### Currency and unit rules
- Quote values exactly as in the verified source. NEVER convert currencies.
- If the source says "188.59亿元人民币", write "188.59亿元人民币" — NOT "26.2亿美元".
- If you want to add a converted figure, state the original first:
  "188.59亿元人民币（按7.2汇率折合约26.2亿美元，仅供参考）"
- WRONG: Source says "18.859 billion yuan" → you write "18.86亿美元"
- RIGHT: Source says "18.859 billion yuan" → you write "188.59亿元"
