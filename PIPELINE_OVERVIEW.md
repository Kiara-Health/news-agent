# Fertility News Pipeline Overview

## What this pipeline does

This pipeline turns a large stream of fertility and reproductive-medicine news into a small, publication-ready weekly package.

It is designed to answer one core question:

**"What are the most important, timely, and trustworthy updates this week?"**

The process is automated, but intentionally conservative. It prefers fewer high-quality stories over a fuller but weaker issue.

---

## End-to-end flow (from raw feeds to final content)

## 1) Source collection (news ingestion)

The system starts by pulling content from a curated list of medical and scientific feeds (configured in `feed-ingestor/news-sources.yaml`).

For each run, it:
- fetches entries from enabled sources,
- captures article metadata (title, URL, authors, dates, source),
- parses and standardizes feed fields into a consistent format.

At this stage, the pipeline may see a large number of raw entries because feeds often return more than one week of items.

---

## 2) Date handling and freshness attribution

After ingestion, each item is assigned freshness information:
- actual publication date (when available),
- fallback freshness date (for example, updated timestamp),
- date source used,
- freshness confidence level.

This is important because some journal feeds provide incomplete publication dates. The pipeline now tracks this explicitly instead of treating all missing-date items the same way.

---

## 3) Quality normalization and hygiene filters

Before editorial selection, the system removes low-value or unsuitable items, such as:
- corrections,
- errata,
- retractions,
- entries with insufficient metadata.

It also deduplicates near-identical stories so the same update is not counted multiple times.

---

## 4) Time-window filtering (for example, past 7 days)

When run with a setting like `--days 7`, the pipeline keeps only items that fall within that requested window (based on the best available date signal).

This is the stage that usually causes the largest drop in volume.

---

## 5) Novelty and repeat suppression across runs

A run-history database tracks what has been:
- seen previously,
- selected previously,
- published previously.

This prevents recycling the same stories issue after issue, unless configured to allow resurfacing after a set number of days.

---

## 6) Relevance, science, and publishability scoring

Each candidate gets scored across several dimensions:
- topic relevance to fertility audience,
- scientific signal,
- overall impact/reportability.

Then hard quality gates are applied (not just soft scoring). For example:
- main stories require stronger evidence and freshness,
- quick hits allow lighter evidence but still must meet minimum quality.

The result: high-ranking but weakly supported items can still be blocked.

---

## 7) Selection policy and issue-state decision

The selector does not "fill quotas at all costs."

Instead, it enforces:
- evidence thresholds,
- date/freshness requirements,
- article-type eligibility,
- source diversity caps,
- novelty rules.

The run is labeled with an issue state:
- `full_issue`
- `underfilled_issue`
- `quick_hits_only`
- `no_publishable_items`

This gives a transparent explanation when a week is data-light.

---

## 8) Article summarization (evidence-bounded)

For selected items, the summarization layer:
- extracts what can be supported by available text,
- generates cautious summaries,
- avoids unsupported claims and invented numbers.

Low-information items are downgraded to short formats rather than padded with generic copy.

---

## 9) Newsletter composition

A separate newsletter composer transforms the validated article set into a reader-facing newsletter:
- title + date,
- editor note,
- featured stories,
- briefs/watchlist,
- closing takeaway.

It uses only validated structured outputs from the report layer and keeps language professional, concise, and non-hyped.

---

## 10) Additional channel outputs

From the same selected set, the pipeline also produces:
- podcast script,
- LinkedIn post (standard + compact),
- optional banner prompt.

This keeps cross-channel messaging aligned with the same editorial source of truth.

---

## Why this is useful for marketing and editorial teams

- **Consistency:** one unified content engine across newsletter, podcast, and social.
- **Trustworthiness:** stronger controls against over-claiming and weak evidence.
- **Transparency:** every rejection/demotion has a reason in diagnostics.
- **Quality-first behavior:** the pipeline can intentionally publish fewer stories when the week is thin.
- **Operational speed:** automated weekly production while preserving editorial safeguards.

---

## What to expect week to week

- Some weeks produce a fuller issue.
- Some weeks may produce quick-hits-only or underfilled issues if:
  - source updates are sparse,
  - many items are outside the date window,
  - evidence quality is too low for headline treatment.

This is expected behavior by design and protects brand trust.

---

## Key files (for orientation)

- Source config: `feed-ingestor/news-sources.yaml`
- Orchestrator: `news-agent/pipeline_fertility.py`
- Selection policy: `news-agent/selection_policy.py`
- Summarization engine: `news-agent/summarizer.py`
- Newsletter composer: `news-agent/newsletter_composer.py`
- Run outputs: `news-agent/output/<run_id>/...`

