---
name: generic
priority: 999
is_fallback: true
signal_threshold: 0
signals: []
alias_sets: []
entity_patterns: []
entity_marker_patterns: []
focus_term_rules: []
low_quality_domains: []
official_domains: []
noise_patterns:
  - luigi mangione
  - crime
  - murder
search_vocabulary: []
min_official_citation_ratio: 0.0
---

## Data Source Strategy

When fetching data from the web:
1. Prefer official sources (government sites, company IR pages, academic journals).
2. Use reputable news agencies for recent events.
3. Avoid JavaScript-rendered pages that return empty templates — search for
   static HTML alternatives (news articles, press releases, official filings).

## Executor Hints

## Reporter Hints
