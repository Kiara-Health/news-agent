# Year-End Podcast Configuration Guide

## Overview

This guide explains how to configure `config.json` for generating a year-end podcast that summarizes the main achievements of 2025 with smooth temporal distribution across the entire year.

## Recommended Configuration: `config_year_end.json`

A pre-configured file `config_year_end.json` has been created with optimal settings for year-end summaries. Use it like this:

```bash
python pipeline.py --start-date 2025-01-01 --end-date 2025-12-31 --config config_year_end.json
```

## Key Configuration Changes Explained

### 1. **Article Selection Parameters**

```json
"article_selection": {
  "max_main_stories": 12,           // Increased from 6 to cover ~1 per month
  "max_quick_hits": 18,             // Increased from 12 for broader coverage
  "main_stories_candidates": 30,    // Increased from 10 for better selection pool
  "quick_hits_candidates": 40,      // Increased from 15 for more options
  "max_main_stories_per_topic": 3,  // Relaxed from 2 to allow more coverage
  "max_quick_hits_per_topic": 4,    // Relaxed from 3
  "main_story_duration_seconds": 150, // Slightly reduced from 180 for more stories
  "quick_hit_duration_seconds": 20,
  "enable_temporal_distribution": true,  // NEW: Ensures articles spread across year
  "temporal_periods": 12              // NEW: Divide year into 12 periods (months)
}
```

**Rationale:**
- **12 main stories**: Roughly one major achievement per month
- **Larger candidate pools**: More options to choose from when balancing impact, topic, and temporal diversity
- **Relaxed topic limits**: Allows more comprehensive coverage while still maintaining diversity
- **Temporal distribution**: NEW feature ensures articles are spread across the entire year, not clustered in one period

### 2. **Time Allocation**

```json
"time_allocation": {
  "main_stories_percent": 0.65,    // Increased from 0.6 (more focus on main stories)
  "quick_hits_percent": 0.25,      // Reduced from 0.3 (less time on quick hits)
  "analysis_percent": 0.1          // Same (10% for trends/insights)
}
```

**Rationale:** Year-end summaries should focus more on major achievements (main stories) rather than quick updates.

### 3. **Target Duration**

```json
"target_duration_seconds": 1800    // Increased from 600 (30 minutes vs 10 minutes)
```

**Rationale:** A year-end summary needs more time to cover 12 months of achievements. 30 minutes allows ~2.5 minutes per main story plus quick hits and analysis.

### 4. **Impact Scoring - Recency Bonus**

```json
"recency_bonus": {
  "day1": 0,    // Disabled (was 2)
  "day3": 0     // Disabled (was 1)
}
```

**Rationale:** For year-end summaries, we want articles from throughout the year, not just recent ones. Disabling recency bonus ensures older articles have equal opportunity.

### 5. **Temporal Distribution (NEW FEATURE)**

The algorithm now supports temporal distribution to ensure articles are spread across the year:

- **`enable_temporal_distribution: true`**: Activates temporal diversity checking
- **`temporal_periods: 12`**: Divides the year into 12 periods (months)
- The algorithm ensures main stories are distributed across different time periods
- Quick hits also consider temporal diversity but with more lenient limits

## How Temporal Distribution Works

1. The algorithm calculates the date range of all articles
2. Divides the range into `temporal_periods` equal time segments
3. Assigns each article to a time period based on its publication date
4. When selecting articles, ensures no single time period dominates
5. For main stories: max 1-2 articles per period (depending on total)
6. For quick hits: more lenient (2-3 per period)

This ensures your year-end podcast covers achievements from:
- Early 2025 (Q1)
- Mid 2025 (Q2-Q3)
- Late 2025 (Q4)

## Usage Examples

### Basic Year-End Summary
```bash
python pipeline.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --config config_year_end.json
```

### Custom Duration
If you want a shorter summary (e.g., 20 minutes), you can override:
```bash
python podcast_generator.py \
  --input filtered_articles.txt \
  --config config_year_end.json \
  --duration 1200
```

### Adjusting Temporal Distribution
If you want quarterly coverage instead of monthly:
1. Edit `config_year_end.json`
2. Set `"temporal_periods": 4`
3. Adjust `max_main_stories` to 4-6 accordingly

## Fine-Tuning Recommendations

### If you get too many articles from one period:
- Increase `temporal_periods` (e.g., 24 for bi-weekly distribution)
- Decrease `max_main_stories_per_topic` to force more topic diversity

### If you want more comprehensive coverage:
- Increase `max_main_stories` to 15-18
- Increase `main_stories_candidates` to 40-50
- Increase `target_duration_seconds` to 2400 (40 minutes)

### If you want a more focused summary:
- Decrease `max_main_stories` to 8-10
- Decrease `max_quick_hits` to 12-15
- Keep `target_duration_seconds` at 1200-1500 (20-25 minutes)

## Expected Output

With `config_year_end.json`, you should get:
- **12 main stories** covering major achievements (~2.5 min each = 30 min)
- **18 quick hits** for additional coverage (~20 sec each = 6 min)
- **Temporal distribution** across all 12 months
- **Topic diversity** across therapeutics, research, industry, etc.
- **Total duration**: ~30 minutes

## Troubleshooting

**Problem**: All articles from one time period
- **Solution**: Ensure `enable_temporal_distribution: true` and check that articles span the full year

**Problem**: Not enough articles selected
- **Solution**: Increase `main_stories_candidates` and `quick_hits_candidates`

**Problem**: Too many articles from one topic
- **Solution**: Decrease `max_main_stories_per_topic` and `max_quick_hits_per_topic`

**Problem**: Podcast too long/short
- **Solution**: Adjust `target_duration_seconds` or `main_story_duration_seconds`
