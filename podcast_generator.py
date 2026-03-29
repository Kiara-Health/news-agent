#!/usr/bin/env python3
"""
Podcast Generator Script

This script implements a hybrid approach to generate a weekly fertility-medicine
newsletter from filtered articles.  It uses occurrence-based selection, impact
scoring, topic diversity, and the evidence-bounded two-stage summarizer.

Summarization is handled by summarizer.py, which:
  - Extracts structured evidence from each article (Stage A)
  - Generates conservative, constrained prose from that evidence (Stage B)
  - Validates the output and blocks unsupported numerical/design claims
  - Detects contradictions across reruns via an SQLite evidence cache
"""

import argparse
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys
import os
from collections import defaultdict, Counter
import random
from llm_caller import call_openai as _call_openai

import summarizer as _summarizer
import selection_policy as _policy_mod

class PodcastGenerator:
    def __init__(
        self,
        articles_file: str = "filtered_articles.txt",
        config_file: str = None,
        evidence_db: str = None,
    ):
        self.articles_file = articles_file
        self.articles = []
        self.selected_articles = []

        # Path to the SQLite evidence cache used by the two-stage summarizer.
        # Contradiction detection is skipped when None.
        self.evidence_db = evidence_db

        # Load configuration
        self.config = self.load_config(config_file)

        # Selection policy (novelty, freshness, evidence, source diversity, etc.)
        self._selection_policy = _policy_mod.SelectionPolicy(self.config)
        
        # Impact keywords and their weights
        self.impact_keywords = self.config.get('impact_scoring', {}).get('keywords', {
            'clinical trial': 5,
            'fda': 5,
            'approval': 5,
            'breakthrough': 4,
            'discovery': 4,
            'first': 4,
            'novel': 4,
            'treatment': 3,
            'cure': 3,
            'drug': 3,
            'therapy': 3,
            'funding': 2,
            'investment': 2,
            'partnership': 2,
            'collaboration': 2,
            'study': 2,
            'research': 2,
            'development': 2
        })
        
        # Topic categories for diversity
        self.topic_categories = self.config.get('topic_categories', {
            'therapeutics': ['treatment', 'therapy', 'drug', 'cure', 'clinical trial', 'fda', 'approval'],
            'diagnostics': ['diagnostic', 'detection', 'screening', 'test', 'biomarker'],
            'research': ['research', 'study', 'discovery', 'breakthrough', 'novel'],
            'industry': ['funding', 'investment', 'partnership', 'collaboration', 'company'],
            'technology': ['technology', 'platform', 'tool', 'device', 'ai', 'machine learning'],
            'genetics': ['gene', 'genetic', 'dna', 'rna', 'genome', 'crispr'],
            'microbiome': ['microbiome', 'bacteria', 'microbial', 'gut', 'microbiome'],
            'cancer': ['cancer', 'oncology', 'tumor', 'carcinoma', 'leukemia'],
            'rare_disease': ['rare disease', 'orphan', 'genetic disorder'],
            'infectious_disease': ['infection', 'virus', 'bacterial', 'pathogen', 'vaccine']
        })
    
    def load_config(self, config_file: str = None) -> Dict:
        """Load configuration from JSON file."""
        default_config_file = "config.json"
        
        # Determine which config file to use
        if config_file is None:
            config_file = default_config_file
        
        # Try to load the config file
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    config["_config_path"] = os.path.abspath(config_file)
                    print(f"Loaded configuration from: {config_file}")
                    return config
            except Exception as e:
                print(f"Warning: Could not load config file '{config_file}': {e}")
                print("Using default configuration values.")
        else:
            if config_file != default_config_file:
                print(f"Warning: Config file '{config_file}' not found. Using default configuration values.")
        
        # Return empty dict to use defaults in __init__
        return {}
        
    def parse_articles_file(self) -> List[Dict]:
        """Parse the filtered articles file and extract article information."""
        if not os.path.exists(self.articles_file):
            print(f"Error: Articles file '{self.articles_file}' not found.")
            return []
        
        articles = []
        current_article = {}
        in_metadata = True
        
        try:
            with open(self.articles_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip metadata section
                if line.startswith("ARTICLE DETAILS"):
                    in_metadata = False
                    i += 1
                    continue
                
                if in_metadata:
                    i += 1
                    continue
                
                # Look for article start pattern
                if line.startswith("Article ") and "------------------------------" in lines[i+1]:
                    # Save previous article if exists
                    if current_article:
                        articles.append(current_article)
                    
                    # Start new article
                    current_article = {}
                    article_num = line.split()[1]
                    current_article['number'] = int(article_num)
                    i += 2  # Skip the separator line
                    continue
                
                # Parse article fields
                if line.startswith("Title: "):
                    current_article['title'] = line[7:]
                elif line.startswith("URL: "):
                    current_article['url'] = line[5:]
                elif line.startswith("Source: "):
                    current_article['source'] = line[8:]
                elif line.startswith("Authors: "):
                    raw_authors = line[9:].strip()
                    current_article['authors'] = [a.strip() for a in raw_authors.split(",") if a.strip()]
                elif line.startswith("Published: "):
                    date_str = line[11:]
                    try:
                        current_article['published_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['published_date'] = None
                elif line.startswith("Updated: "):
                    date_str = line[9:]
                    try:
                        current_article['updated_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['updated_date'] = None
                elif line.startswith("Freshness Date: "):
                    date_str = line[16:]
                    try:
                        current_article['freshness_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['freshness_date'] = None
                elif line.startswith("Fetched At: "):
                    date_str = line[11:]
                    try:
                        current_article['fetched_at'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['fetched_at'] = None
                elif line.startswith("First Seen At: "):
                    date_str = line[15:]
                    try:
                        current_article['first_seen_at'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['first_seen_at'] = None
                elif line.startswith("Date Source: "):
                    current_article['date_source'] = line[13:].strip() or "unknown"
                elif line.startswith("Used Fallback Date: "):
                    current_article['used_fallback_date'] = line[20:].strip().lower() == "true"
                elif line.startswith("Freshness Confidence: "):
                    current_article['freshness_confidence'] = line[22:].strip() or "low"
                elif line.startswith("Occurrences: "):
                    current_article['occurrences'] = int(line[13:])
                elif line.startswith("Content Length: "):
                    content_length_str = line[16:]
                    # Remove "characters" suffix if present
                    content_length_str = content_length_str.replace(" characters", "")
                    try:
                        current_article['content_length'] = int(content_length_str)
                    except ValueError:
                        current_article['content_length'] = 0
                elif line.startswith("Content: "):
                    content = line[9:]
                    # Continue reading content if it spans multiple lines
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith("="):
                        content += " " + lines[j].strip()
                        j += 1
                    current_article['content'] = content
                    i = j - 1  # Adjust index
                
                i += 1
            
            # Add the last article
            if current_article:
                articles.append(current_article)
            
            print(f"Successfully parsed {len(articles)} articles from {self.articles_file}")
            return articles
            
        except Exception as e:
            print(f"Error parsing articles file: {e}")
            return []
    
    # Keywords that suggest methodological rigour — used for scientific_importance.
    _SCIENCE_QUALITY_TERMS = [
        "randomized", "randomised", "clinical trial", "prospective", "cohort",
        "systematic review", "meta-analysis", "longitudinal", "peer-reviewed",
        "placebo", "double-blind", "multicenter", "statistical significance",
        "odds ratio", "relative risk", "hazard ratio", "confidence interval",
        "p-value", "p <", "pubmed", "doi:",
    ]

    # Topic buckets for balanced weekly mix.
    BALANCED_TOPIC_BUCKETS = {
        "art_ivf":            ["ivf", "in vitro fertilization", "embryo transfer", "egg retrieval",
                               "oocyte", "cryopreservation", "frozen embryo", "blastocyst",
                               "icsi", "egg freezing", "sperm injection"],
        "fertility_optimisation": ["preconception", "trying to conceive", "ttc", "ovulation",
                                   "fertility diet", "supplements", "coq10", "dhea",
                                   "antioxidant", "lifestyle", "bmi", "weight"],
        "menstrual_cycle":    ["menstrual", "cycle", "period", "pcos", "endometriosis",
                               "luteal phase", "follicular", "anovulation", "irregular cycle"],
        "male_factor":        ["male factor", "semen analysis", "sperm", "azoospermia",
                               "varicocele", "testosterone", "motility", "morphology"],
        "mental_health":      ["mental health", "anxiety", "depression", "stress", "grief",
                               "infertility distress", "psychological", "counseling",
                               "quality of life", "patient experience"],
        "genetics_genomics":  ["preimplantation genetic", "pgt", "carrier screening",
                               "aneuploidy", "chromosome", "genetic testing", "gene",
                               "hereditary", "mosaic"],
    }

    def calculate_scientific_importance(self, article: Dict) -> float:
        """
        Score 0–10 reflecting methodological rigour and evidence quality.
        Higher for RCTs, systematic reviews, and peer-reviewed studies.
        """
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        score = sum(1.5 for term in self._SCIENCE_QUALITY_TERMS if term in text)
        content_length = article.get('content_length', 0)
        if content_length > 1000:
            score += 2.0
        elif content_length > 400:
            score += 0.5
        return min(score, 10.0)

    def calculate_audience_relevance(self, article: Dict) -> float:
        """
        Score 0–10 reflecting relevance to a fertility-medicine patient/clinician audience.
        Uses domain-specific keywords from config plus balanced-bucket matching.
        """
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        score = 0.0
        for keyword, weight in self.impact_keywords.items():
            if keyword.lower() in text:
                score += weight * 0.3
        # Balanced-bucket bonus
        for bucket_kws in self.BALANCED_TOPIC_BUCKETS.values():
            if any(kw in text for kw in bucket_kws):
                score += 1.0
        return min(score, 10.0)

    def calculate_impact_score(self, article: Dict) -> float:
        """
        Combined impact score = weighted blend of scientific_importance and
        audience_relevance, plus occurrence and recency bonuses.

        Both sub-scores are also stored on the article dict for transparency.
        """
        impact_config = self.config.get('impact_scoring', {})
        occurrence_multiplier = impact_config.get('occurrence_multiplier', 2)
        content_length_threshold = impact_config.get('content_length_threshold', 300)
        content_length_bonus = impact_config.get('content_length_bonus', 1)
        recency_bonus = impact_config.get('recency_bonus', {'day1': 2, 'day3': 1})

        sci = self.calculate_scientific_importance(article)
        rel = self.calculate_audience_relevance(article)
        article['scientific_importance'] = round(sci, 2)
        article['audience_relevance'] = round(rel, 2)

        # Weighted blend (40 % science quality, 60 % audience relevance)
        score = 0.4 * sci + 0.6 * rel

        # Occurrence bonus
        score += article.get('occurrences', 1) * occurrence_multiplier

        # Content length bonus
        if article.get('content_length', 0) > content_length_threshold:
            score += content_length_bonus

        # Recency bonus
        if article.get('published_date'):
            days_old = (datetime.now() - article['published_date']).days
            if days_old <= 1:
                score += recency_bonus.get('day1', 2)
            elif days_old <= 3:
                score += recency_bonus.get('day3', 1)

        # Evidence-tier penalty (pre-LLM heuristic)
        # This ensures low-evidence articles rank below abstract-backed articles.
        tier = _summarizer.estimate_evidence_tier(article)
        article['evidence_tier_estimate'] = tier
        score += _summarizer._TIER_SCORE_PENALTY.get(tier, 0.0)

        return score
    
    def classify_topic(self, article: Dict) -> str:
        """Classify article into topic categories."""
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        topic_scores = defaultdict(int)
        
        for topic, keywords in self.topic_categories.items():
            for keyword in keywords:
                if keyword in text:
                    topic_scores[topic] += 1
        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def select_articles_hybrid(self, articles: List[Dict], target_duration: int = None) -> List[Dict]:
        """Select articles using hybrid approach for target duration (in seconds)."""
        if not articles:
            return []
        
        # Get config values with defaults
        selection_config = self.config.get('article_selection', {})
        time_config = self.config.get('time_allocation', {})
        
        # Use target_duration from config if not provided
        if target_duration is None:
            target_duration = self.config.get('target_duration_seconds', 600)
        
        max_main_stories = selection_config.get('max_main_stories', 6)
        max_quick_hits = selection_config.get('max_quick_hits', 12)
        main_stories_candidates = selection_config.get('main_stories_candidates', 10)
        quick_hits_candidates = selection_config.get('quick_hits_candidates', 15)
        max_main_stories_per_topic = selection_config.get('max_main_stories_per_topic', 2)
        max_quick_hits_per_topic = selection_config.get('max_quick_hits_per_topic', 3)
        main_story_duration = selection_config.get('main_story_duration_seconds', 180)
        quick_hit_duration = selection_config.get('quick_hit_duration_seconds', 20)
        enable_temporal_distribution = selection_config.get('enable_temporal_distribution', False)
        temporal_periods = selection_config.get('temporal_periods', 12)  # Default: 12 months
        
        main_stories_percent = time_config.get('main_stories_percent', 0.6)
        quick_hits_percent = time_config.get('quick_hits_percent', 0.3)
        analysis_percent = time_config.get('analysis_percent', 0.1)
        
        # Calculate impact scores and classify topics
        for article in articles:
            article['impact_score'] = self.calculate_impact_score(article)
            article['topic'] = self.classify_topic(article)

        # ── Selection policy (novelty / freshness / evidence / type / negscoring) ──
        import os, time
        _run_id = f"run_{int(time.time())}"
        _db = self.evidence_db or os.path.join(
            os.path.dirname(self.config.get("_config_path", ".") or "."),
            "selection_history.db",
        )
        self._selection_policy.apply_all(articles, db_path=_db, run_id=_run_id)

        # Remove hard-excluded articles (corrections, errata, retractions)
        articles = [a for a in articles if not a.get("_policy", {}).get("suppressed")]

        # Sort by adjusted impact score
        articles.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # If temporal distribution is enabled, organize articles by time periods
        if enable_temporal_distribution:
            # Calculate date range
            dates = [a['published_date'] for a in articles if a.get('published_date')]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                date_range_days = (max_date - min_date).days
                
                if date_range_days > 0:
                    period_duration = date_range_days / temporal_periods
                    
                    # Assign each article to a time period
                    for article in articles:
                        if article.get('published_date'):
                            days_from_start = (article['published_date'] - min_date).days
                            article['temporal_period'] = int(days_from_start / period_duration) if period_duration > 0 else 0
                        else:
                            article['temporal_period'] = -1  # No date
                    
                    print(f"Temporal distribution enabled: {temporal_periods} periods over {date_range_days} days")
        
        selected_articles = []
        topic_coverage = defaultdict(int)
        temporal_coverage = defaultdict(int) if enable_temporal_distribution else None
        estimated_duration = 0
        
        # Target time allocation
        main_stories_time = target_duration * main_stories_percent
        quick_hits_time = target_duration * quick_hits_percent
        analysis_time = target_duration * analysis_percent
        
        # ── Gate-first selection: single pass, no quota filling ───────────────────
        # max_main_stories and max_quick_hits are CEILINGS, not targets.
        # An article is tried for main story first; if it fails the gate it is
        # tried for quick hit; if that fails too it is rejected with diagnostics.
        main_stories:  List[Dict] = []
        quick_hits:    List[Dict] = []
        rejected:      List[Dict] = []

        main_src_counts: Dict[str, int] = defaultdict(int)
        qh_src_counts:   Dict[str, int] = defaultdict(int)

        main_cap_per_src = self._selection_policy._sd.get("max_main_stories_per_source", 2)
        qh_cap_per_src   = self._selection_policy._sd.get("max_quick_hits_per_source",   3)
        min_distinct_main = self._selection_policy._sd.get("min_distinct_sources_main", 1)
        min_distinct_total = self._selection_policy._sd.get("min_distinct_sources_total", 1)

        publishable_pool = [
            a for a in articles
            if (a.get("_policy", {}) or {}).get("publishable_candidate", False)
        ]
        distinct_publishable_sources = {
            _policy_mod._normalize_source(a.get("source", "")) for a in publishable_pool
        }
        self._last_degraded_mode_reason = None
        if len(distinct_publishable_sources) < min_distinct_total:
            self._last_degraded_mode_reason = (
                f"publishable pool has only {len(distinct_publishable_sources)} distinct "
                f"source(s), below min_distinct_sources_total={min_distinct_total}"
            )
        elif len(distinct_publishable_sources) < min_distinct_main:
            self._last_degraded_mode_reason = (
                f"publishable pool has only {len(distinct_publishable_sources)} distinct "
                f"source(s), below min_distinct_sources_main={min_distinct_main}"
            )

        def _simulate_with_caps(cap_main: int, cap_qh: int) -> Dict[str, int]:
            sm, sq = 0, 0
            main_src = defaultdict(int)
            qh_src = defaultdict(int)
            for art in publishable_pool:
                src = _policy_mod._normalize_source(art.get("source", ""))
                if sm < max_main_stories and (art.get("_policy", {}) or {}).get("main_story_eligible", False):
                    if main_src[src] < cap_main:
                        main_src[src] += 1
                        sm += 1
                        continue
                if sq < max_quick_hits and (art.get("_policy", {}) or {}).get("quick_hit_eligible", False):
                    if qh_src[src] < cap_qh:
                        qh_src[src] += 1
                        sq += 1
            return {"main": sm, "quick": sq}

        self._last_source_cap_audit = {
            "publishable_pool_count": len(publishable_pool),
            "publishable_distinct_sources": len(distinct_publishable_sources),
            "strict_caps": _simulate_with_caps(main_cap_per_src, qh_cap_per_src),
            "relaxed_caps": _simulate_with_caps(max(main_cap_per_src, 1) * 2, max(qh_cap_per_src, 1) * 2),
            "caps_off": _simulate_with_caps(9999, 9999),
        }

        for article in articles:
            _pol = article.setdefault("_policy", {})

            # ── Try main story ──────────────────────────────────────────────
            tried_main = False
            if len(main_stories) < max_main_stories:
                tried_main = True
                eligible, reason = self._selection_policy.is_main_story_eligible(
                    article, current_main_stories=main_stories
                )
                _pol["main_story_ineligible_reason"] = reason

                if eligible:
                    src = _policy_mod._normalize_source(article.get("source", ""))
                    if main_src_counts[src] >= main_cap_per_src:
                        _pol["source_diversity_status"] = (
                            f"capped: '{src}' already has "
                            f"{main_src_counts[src]}/{main_cap_per_src} main stories"
                        )
                        eligible = False

                if eligible:
                    topic_ok = (
                        topic_coverage[article["topic"]] < max_main_stories_per_topic
                    )
                    temporal_ok = True
                    if enable_temporal_distribution and temporal_coverage is not None:
                        period = article.get("temporal_period", -1)
                        if period >= 0:
                            max_per_period = max(1, max_main_stories // temporal_periods)
                            temporal_ok = temporal_coverage[period] < max_per_period
                        else:
                            temporal_ok = sum(temporal_coverage.values()) < max_main_stories * 0.8

                    if topic_ok and temporal_ok:
                        main_stories.append(article)
                        _pol["selected_as"] = "main_story"
                        src = _policy_mod._normalize_source(article.get("source", ""))
                        main_src_counts[src] += 1
                        topic_coverage[article["topic"]] += 1
                        if enable_temporal_distribution and temporal_coverage is not None:
                            period = article.get("temporal_period", -1)
                            if period >= 0:
                                temporal_coverage[period] += 1
                        estimated_duration += main_story_duration
                        continue  # no need to try quick hit
                    else:
                        _pol["main_story_ineligible_reason"] = (
                            "topic/temporal coverage cap reached"
                        )

            # ── Try quick hit ───────────────────────────────────────────────
            if len(quick_hits) < max_quick_hits:
                qh_eligible, qh_reason = self._selection_policy.is_quick_hit_eligible(article)
                _pol["quick_hit_ineligible_reason"] = qh_reason if not qh_eligible else None

                if qh_eligible:
                    src = _policy_mod._normalize_source(article.get("source", ""))
                    if qh_src_counts[src] >= qh_cap_per_src:
                        _pol["source_diversity_status"] = (
                            f"capped: '{src}' already has "
                            f"{qh_src_counts[src]}/{qh_cap_per_src} quick hits"
                        )
                        qh_eligible = False

                if qh_eligible and topic_coverage[article["topic"]] < max_quick_hits_per_topic:
                    quick_hits.append(article)
                    _pol["selected_as"] = "quick_hit"
                    src = _policy_mod._normalize_source(article.get("source", ""))
                    qh_src_counts[src] += 1
                    topic_coverage[article["topic"]] += 1
                    if enable_temporal_distribution and temporal_coverage is not None:
                        period = article.get("temporal_period", -1)
                        if period >= 0:
                            temporal_coverage[period] += 1
                    estimated_duration += quick_hit_duration
                    continue

            rejected.append(article)

        selected_articles = main_stories + quick_hits

        # ── Issue state ───────────────────────────────────────────────────────
        issue_state = _policy_mod.compute_issue_state(
            main_stories, quick_hits, max_main_stories, max_quick_hits
        )
        self._last_issue_state    = issue_state
        self._last_main_stories   = main_stories
        self._last_quick_hits     = quick_hits
        self._last_rejected       = rejected
        self._last_max_main       = max_main_stories
        self._last_max_qh         = max_quick_hits

        print(
            f"[SELECTION] issue_state={issue_state} | "
            f"main={len(main_stories)}/{max_main_stories} "
            f"quick={len(quick_hits)}/{max_quick_hits} "
            f"rejected={len(rejected)}"
        )
        if self._last_degraded_mode_reason:
            print(f"[SELECTION] degraded_mode_reason={self._last_degraded_mode_reason}")
        print(f"[SELECTION] source_cap_audit={self._last_source_cap_audit}")
        if issue_state in ("underfilled_issue", "quick_hits_only", "no_publishable_items"):
            print(
                f"[SELECTION] ⚠ {issue_state.upper()} — "
                f"{len(rejected)} articles failed all eligibility gates."
            )
            # Emit first few rejection reasons for debugging
            for art in rejected[:5]:
                p = art.get("_policy", {})
                main_reason = p.get("main_story_ineligible_reason", "—")
                qh_reason   = p.get("quick_hit_ineligible_reason",  "—")
                title_short = (art.get("title") or "")[:60]
                print(
                    f"  REJECTED: {title_short!r}\n"
                    f"    main: {main_reason}\n"
                    f"    qh:   {qh_reason}"
                )

        if enable_temporal_distribution and temporal_coverage:
            periods_covered = len([p for p, c in temporal_coverage.items() if c > 0])
            print(f"[SELECTION] Temporal coverage: {periods_covered}/{temporal_periods} periods")
        print(f"[SELECTION] Estimated duration: {estimated_duration/60:.1f} minutes")

        return selected_articles
    
    def _make_llm_caller(self):
        """Return the OpenAI caller bound to this generator's config."""
        cfg = self.config

        def caller(prompt: str, _cfg: Dict = None, timeout: int = 60) -> Optional[str]:
            return _call_openai(prompt, cfg, timeout=timeout)

        return caller

    def generate_article_summary(self, article: Dict, detailed: bool = False) -> str:
        """
        Evidence-bounded two-stage summarizer (Stage A + Stage B from summarizer.py).

        The result is stored on the article dict as:
          - article['generated_summary']   – formatted prose (4-section string)
          - article['structured_summary']  – StructuredSummary object
        Returns the formatted prose string.
        """
        title = article.get('title', '')[:80]
        print(f"\n  Summarising: {title}...")

        llm_caller = self._make_llm_caller()
        structured = _summarizer.summarize_article(
            article=article,
            config=self.config,
            db_path=self.evidence_db,
            llm_caller=llm_caller,
        )

        prose = structured.to_prose()
        article['generated_summary'] = prose
        article['structured_summary'] = structured

        confidence = structured.evidence.confidence if structured.evidence else "low"
        quality = (
            structured.evidence.source_text_quality if structured.evidence else "snippet_only"
        )
        fallback_flag = " [fallback]" if structured.is_fallback else ""
        print(
            f"  ✅ Summary generated ({len(prose)} chars, "
            f"confidence={confidence}, quality={quality}{fallback_flag})"
        )

        # Log any hard-validator failures for visibility
        for vr in structured.validation_results:
            if not vr.passed:
                print(f"  ⚠️  Validator [{vr.rule}]: {vr.details}")

        # Log contradiction warnings
        if structured.contradiction and structured.contradiction.has_contradiction:
            print(f"  🔄 Contradiction detected: {structured.contradiction.fields_changed}")

        return prose

    def generate_podcast_script(self, articles: List[Dict]) -> str:
        """Generate the weekly report script from selected articles."""
        if not articles:
            return "No articles selected for report generation."

        pub = self.config.get('publication_settings', {})
        podcast_title = pub.get('podcast_title', 'Weekly News Report')
        podcast_intro = pub.get(
            'podcast_intro',
            "This week's fertility and reproductive medicine news roundup covers "
            "the latest research findings, clinical developments, and emerging evidence."
        )
        podcast_closing = pub.get(
            'podcast_closing',
            "This report was compiled from peer-reviewed journals, clinical publications, "
            "and specialist news sources."
        )
        topics_label = pub.get('topics_label', 'the field')

        selection_config = self.config.get('article_selection', {})
        max_main_stories = selection_config.get('max_main_stories', 6)

        main_stories = articles[:max_main_stories]
        quick_hits = articles[max_main_stories:]

        script = []

        # Header
        script.append(f"=== {podcast_title.upper()} ===")
        script.append("")
        script.append(podcast_intro)
        script.append("")

        # Main stories
        script.append("=== MAIN STORIES ===")
        script.append("")

        for i, article in enumerate(main_stories, 1):
            script.append(f"Story {i}: {article['title']}")
            script.append("")
            summary = self.generate_article_summary(article, detailed=True)
            script.append(summary)
            script.append("")
            # Append evidence quality note so readers can assess reliability
            ev = article.get('structured_summary')
            if ev and ev.evidence:
                script.append(
                    f"  [Source quality: {ev.evidence.source_text_quality} | "
                    f"Confidence: {ev.evidence.confidence} | "
                    f"Article type: {ev.evidence.article_type}]"
                )
            script.append("---")
            script.append("")

        # Quick hits — short_blurb and full-tier items not selected as main stories
        quick_hit_articles = [a for a in quick_hits
                               if a.get('evidence_tier_estimate', 'short_blurb')
                               != 'titles_to_watch']
        titles_to_watch = [a for a in quick_hits
                           if a.get('evidence_tier_estimate', 'short_blurb')
                           == 'titles_to_watch']

        if quick_hit_articles:
            script.append("=== QUICK HITS ===")
            script.append("")
            script.append(f"Brief updates from across {topics_label}:")
            script.append("")

            for article in quick_hit_articles:
                script.append(f"• {article['title']}")
                summary = self.generate_article_summary(article, detailed=False)
                structured = article.get('structured_summary')
                if structured and not structured.is_fallback and structured.tier == "full":
                    compact = " ".join(filter(None, [
                        structured.what_it_found,
                        structured.why_it_matters,
                    ]))
                    script.append(f"  {compact}")
                elif structured and structured.tier == "short_blurb":
                    script.append(f"  {summary}")
                else:
                    script.append(f"  {summary}")
                script.append("")

        # Titles to Watch — very low evidence; no padded summary
        if titles_to_watch:
            # Generate minimal summaries for these (no LLM, just metadata)
            for article in titles_to_watch:
                self.generate_article_summary(article, detailed=False)

            script.append("=== TITLES TO WATCH ===")
            script.append("")
            script.append(
                "The following items were identified as potentially relevant "
                "but had insufficient retrieved text to produce a reliable summary. "
                "Consult primary sources directly."
            )
            script.append("")
            for article in titles_to_watch:
                source = article.get('source', '')
                pub_date = (
                    article['published_date'].strftime('%Y-%m-%d')
                    if article.get('published_date') else ''
                )
                date_str = f" ({pub_date})" if pub_date else ""
                script.append(f"• {article['title']}{date_str}")
                if source:
                    script.append(f"  Source: {source}")
                script.append("")

        # Repeated fallback language check
        all_selected = main_stories + quick_hit_articles + titles_to_watch
        repeated = _summarizer.check_repeated_fallbacks(all_selected)
        if repeated:
            logger.warning(
                "Repeated generic fallback phrases detected across summaries: %s", repeated
            )

        # Trends & insights
        script.append("=== TRENDS & INSIGHTS ===")
        script.append("")
        trends = self.analyze_trends(articles)
        script.append(trends)
        script.append("")
        script.append(podcast_closing)
        script.append("")

        source_summary = self.generate_source_summary(articles)
        script.append(source_summary)

        return "\n".join(script)
    
    def generate_source_summary(self, articles: List[Dict]) -> str:
        """Generate a summary of all sources used in the podcast."""
        sources = [article.get('source', '') for article in articles]
        source_counts = Counter(sources)
        
        summary = []
        summary.append("=== SOURCES SUMMARY ===")
        summary.append("")
        summary.append("This podcast was compiled from the following sources:")
        summary.append("")
        
        # Sort sources by article count
        sorted_sources = source_counts.most_common()
        
        for source, count in sorted_sources:
            summary.append(f"• {source}: {count} article{'s' if count > 1 else ''}")
        
        summary.append("")
        summary.append(f"Total sources: {len(source_counts)}")
        summary.append(f"Total articles: {len(articles)}")
        
        return "\n".join(summary)
    
    def analyze_trends(self, articles: List[Dict]) -> str:
        """Analyze trends in the selected articles."""
        topics = [article.get('topic', 'general') for article in articles]
        topic_counts = Counter(topics)
        
        # Find most common topics
        top_topics = topic_counts.most_common(3)
        
        # Analyze sources
        sources = [article.get('source', '') for article in articles]
        source_counts = Counter(sources)
        
        trends = "Looking at this week's developments, "
        
        if top_topics:
            main_topic = top_topics[0][0]
            trends += f"the focus has been on {main_topic.replace('_', ' ')} research, "
            
            if len(top_topics) > 1:
                second_topic = top_topics[1][0]
                trends += f"followed by {second_topic.replace('_', ' ')}. "
            else:
                trends += "showing a concentrated effort in this area. "
        
        # Add source diversity comment
        unique_sources = len(source_counts)
        trends += f"We're seeing coverage from {unique_sources} different sources, "
        trends += "indicating broad industry interest in these developments."
        
        return trends
    
    def save_podcast_script(self, script: str, output_file: str):
        """Save the podcast script to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(script)
            
            print(f"Podcast script saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving podcast script: {e}")

    def save_consolidated_report(self, articles: List[Dict], txt_path: str, json_path: str):
        """
        Save a consolidated report of the selected articles in both .txt and .json formats.

        Each record contains: title, source, authors, published_date, url, topic,
        impact_score, section (main_story | quick_hit), and the generated summary.
        """
        selection_config = self.config.get('article_selection', {})
        max_main = selection_config.get('max_main_stories', 6)
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        pub = self.config.get('publication_settings', {})
        podcast_title = pub.get('podcast_title', 'Weekly Newsletter')

        # ── JSON output ──────────────────────────────────────────────────────
        records = []
        for i, art in enumerate(articles):
            _pol_art = art.get("_policy") or {}
            # Use the gate-assigned role; fall back to positional for backward compat
            section = _pol_art.get("selected_as") or ("main_story" if i < max_main else "quick_hit")
            pub_date = art['published_date'].strftime('%Y-%m-%d') if art.get('published_date') else None
            structured: Optional[_summarizer.StructuredSummary] = art.get('structured_summary')
            evidence_meta: Dict = {}
            summary_sections: Dict = {}
            if structured:
                # Prefer the policy-based article type (title-pattern driven) over the
                # LLM's classification, which can mislabel reviews as original_research.
                _llm_art_type = structured.evidence.article_type if structured.evidence else "unknown"
                _pol_art_type = _pol_art.get("article_type") or _llm_art_type
                evidence_meta = {
                    "confidence": structured.evidence.confidence if structured.evidence else "low",
                    "source_text_quality": (
                        structured.evidence.source_text_quality if structured.evidence else "snippet_only"
                    ),
                    "article_type": _pol_art_type,
                    "llm_article_type": _llm_art_type,
                    "is_fallback": structured.is_fallback,
                    "validation_passed": all(v.passed for v in structured.validation_results),
                    "unsupported_sentence_count": sum(
                        1 for lbl in structured.sentence_labels if lbl.label == "unsupported"
                    ),
                }
                summary_sections = {
                    "what_it_studied": structured.what_it_studied,
                    "what_it_found": structured.what_it_found,
                    "why_it_matters": structured.why_it_matters,
                    "caveats": structured.caveats,
                }
            impact = round(art.get('impact_score', 0.0), 2)
            ev_suf = evidence_meta.get("evidence_sufficiency",
                                        structured.evidence_sufficiency if structured else 0.0)
            tier_str = evidence_meta.get("article_type",
                                          structured.tier if structured else "unknown")
            actual_tier = structured.tier if structured else art.get('evidence_tier_estimate', 'unknown')
            _tier_mult = {"full": 1.0, "short_blurb": 0.7, "titles_to_watch": 0.3}
            reportability = round(impact * _tier_mult.get(actual_tier, 1.0), 2)

            _pol = art.get("_policy") or {}
            _fresh = _pol.get("freshness_status", {}) or {}
            _ev = _pol.get("evidence_status", {}) or {}
            # Canonical fields for debugging consistency
            canonical_article_type = _pol.get("article_type_canonical", _pol.get("article_type", "unknown"))
            canonical_suff = _pol.get("evidence_sufficiency_canonical", _ev.get("sufficiency", ev_suf or 0.0))
            canonical_novelty = _pol.get("novelty_state_canonical", (_pol.get("novelty_status", {}) or {}).get("status", "new"))
            records.append({
                "rank": i + 1,
                "section": section,
                "title": art.get('title', ''),
                "source": art.get('source', ''),
                "authors": art.get('authors', []),
                "published_date": pub_date,
                "url": art.get('url', ''),
                "topic": art.get('topic', ''),
                "secondary_topics": art.get('secondary_topics', []),
                "impact_score": impact,
                "article_type": canonical_article_type,
                "novelty_state": canonical_novelty,
                "scientific_importance": round(art.get('scientific_importance', 0.0), 2),
                "audience_relevance": round(art.get('audience_relevance', 0.0), 2),
                "reportability_score": reportability,
                "evidence_tier": actual_tier,
                "evidence_sufficiency": round(float(canonical_suff), 3) if canonical_suff else 0.0,
                "date_source": _fresh.get("date_source", art.get("date_source", "unknown")),
                "used_fallback_date": bool(_fresh.get("used_fallback_date", art.get("used_fallback_date", False))),
                "freshness_confidence": _fresh.get("freshness_confidence", art.get("freshness_confidence", "low")),
                "summary": art.get('generated_summary', art.get('content', '')[:300]),
                "summary_sections": summary_sections,
                "evidence_quality": evidence_meta,
                "selection_diagnostics": {
                    "article_type":                   _pol.get("article_type", "unknown"),
                    "article_type_canonical":         canonical_article_type,
                    "primary_topic":                  _pol.get("primary_topic", ""),
                    "secondary_topics":               _pol.get("secondary_topics", []),
                    "selected_as":                    _pol.get("selected_as"),
                    "coverage_candidate":             _pol.get("coverage_candidate", True),
                    "publishable_candidate":          _pol.get("publishable_candidate", False),
                    "suppressed":                     _pol.get("suppressed", False),
                    "suppressed_reason":              _pol.get("suppressed_reason"),
                    "rejection_reasons":              _pol.get("rejection_reasons", []),
                    "downgrade_reasons":              _pol.get("downgrade_reasons", []),
                    "novelty_status":                 _pol.get("novelty_status", {}),
                    "novelty_state_canonical":        canonical_novelty,
                    "evidence_status":                _pol.get("evidence_status", {}),
                    "evidence_sufficiency_canonical": round(float(canonical_suff), 3),
                    "freshness_status":               _pol.get("freshness_status", {}),
                    "source_diversity_status":        _pol.get("source_diversity_status"),
                    "main_story_eligible":            _pol.get("main_story_eligible", True),
                    "main_story_ineligible_reason":   _pol.get("main_story_ineligible_reason"),
                    "quick_hit_eligible":             _pol.get("quick_hit_eligible", False),
                    "quick_hit_ineligible_reason":    _pol.get("quick_hit_ineligible_reason"),
                    "selection_stage_scores":         _pol.get("selection_stage_scores", {}),
                },
            })

        # Use per-run metadata stored by select_articles_hybrid when available
        _issue_state = getattr(self, "_last_issue_state", "unknown")
        _max_main    = getattr(self, "_last_max_main", 0)
        _max_qh      = getattr(self, "_last_max_qh",   0)
        _all_arts    = getattr(self, "_last_main_stories", []) + \
                       getattr(self, "_last_quick_hits",   []) + \
                       getattr(self, "_last_rejected",     [])
        if not _all_arts:
            _all_arts = articles

        diag_summary = _policy_mod.build_selection_diagnostics_summary(
            all_articles=_all_arts,
            selected=articles,
            issue_state=_issue_state,
            max_main=_max_main,
            max_qh=_max_qh,
        )
        diag_summary["degraded_mode_reason"] = getattr(self, "_last_degraded_mode_reason", None)
        diag_summary["source_cap_audit"] = getattr(self, "_last_source_cap_audit", {})

        _sel_main = sum(1 for a in articles if (a.get("_policy") or {}).get("selected_as") == "main_story")
        _sel_qh   = sum(1 for a in articles if (a.get("_policy") or {}).get("selected_as") == "quick_hit")
        json_payload = {
            "title": podcast_title,
            "generated_at": generated_at,
            "issue_state": _issue_state,
            "total_articles": len(articles),
            "main_stories_count": _sel_main,
            "quick_hits_count": _sel_qh,
            "selection_diagnostics_summary": diag_summary,
            "articles": records,
        }

        try:
            with open(json_path, 'w', encoding='utf-8') as fh:
                json.dump(json_payload, fh, indent=2, ensure_ascii=False)
            print(f"Consolidated report (JSON) saved to: {json_path}")
        except Exception as e:
            print(f"Error saving JSON report: {e}")

        # ── TXT output ───────────────────────────────────────────────────────
        SEP = "=" * 70
        sep = "-" * 70

        lines = [
            SEP,
            f"  {podcast_title.upper()} — SELECTED ARTICLES REPORT",
            SEP,
            f"  Generated : {generated_at}",
            f"  Articles  : {len(articles)} total "
            f"({json_payload['main_stories_count']} main, "
            f"{json_payload['quick_hits_count']} quick hits)",
            SEP,
            "",
        ]

        main_recs = [r for r in records if r['section'] == 'main_story']
        quick_recs = [r for r in records if r['section'] == 'quick_hit']

        if main_recs:
            lines += ["MAIN STORIES", sep, ""]
            for r in main_recs:
                authors_str = ", ".join(r['authors']) if r['authors'] else "—"
                eq = r.get('evidence_quality', {})
                lines += [
                    f"  [{r['rank']}] {r['title']}",
                    f"      Source    : {r['source']}",
                    f"      Authors   : {authors_str}",
                    f"      Published : {r['published_date'] or '—'}",
                    f"      URL       : {r['url']}",
                    f"      Topic     : {r['topic']}  |  Impact score: {r['impact_score']}",
                    f"      Science   : {r.get('scientific_importance', '—')}  |  "
                    f"Relevance: {r.get('audience_relevance', '—')}",
                    f"      Evidence  : confidence={eq.get('confidence','—')}  "
                    f"quality={eq.get('source_text_quality','—')}  "
                    f"type={eq.get('article_type','—')}  "
                    f"tier={r.get('evidence_tier','—')}  "
                    f"sufficiency={r.get('evidence_sufficiency','—')}",
                    f"      Reportability: {r.get('reportability_score','—')}",
                    "",
                ]
                # Wrap summary at 80 chars
                summary = r['summary']
                for para in summary.split('\n\n'):
                    wrapped = []
                    words = para.split()
                    line_ = "      "
                    for word in words:
                        if len(line_) + len(word) + 1 > 78:
                            wrapped.append(line_.rstrip())
                            line_ = "      " + word + " "
                        else:
                            line_ += word + " "
                    if line_.strip():
                        wrapped.append(line_.rstrip())
                    lines += wrapped + [""]
                lines += [sep, ""]

        if quick_recs:
            lines += ["QUICK HITS", sep, ""]
            for r in quick_recs:
                authors_str = ", ".join(r['authors']) if r['authors'] else "—"
                lines += [
                    f"  [{r['rank']}] {r['title']}",
                    f"      Source    : {r['source']}",
                    f"      Authors   : {authors_str}",
                    f"      Published : {r['published_date'] or '—'}",
                    f"      URL       : {r['url']}",
                    "",
                ]
                summary = r['summary']
                for para in summary.split('\n\n'):
                    wrapped = []
                    words = para.split()
                    line_ = "      "
                    for word in words:
                        if len(line_) + len(word) + 1 > 78:
                            wrapped.append(line_.rstrip())
                            line_ = "      " + word + " "
                        else:
                            line_ += word + " "
                    if line_.strip():
                        wrapped.append(line_.rstrip())
                    lines += wrapped + [""]
                lines += [sep, ""]

        lines += [SEP]

        try:
            with open(txt_path, 'w', encoding='utf-8') as fh:
                fh.write("\n".join(lines))
            print(f"Consolidated report (TXT) saved to: {txt_path}")
        except Exception as e:
            print(f"Error saving TXT report: {e}")

    def generate_podcast(self, target_duration: int = None) -> str:
        """Generate a complete podcast from articles."""
        # Parse articles
        articles = self.parse_articles_file()
        
        if not articles:
            return "No articles found to generate podcast."
        
        # Select articles using hybrid approach (uses config if target_duration is None)
        selected_articles = self.select_articles_hybrid(articles, target_duration)
        
        if not selected_articles:
            return "No articles selected for podcast generation."
        
        # Store for caller access (e.g. consolidated report)
        self.selected_articles = selected_articles

        # Generate script
        script = self.generate_podcast_script(selected_articles)
        
        return script

def main():
    """Main function to handle command line arguments and execute podcast generation."""
    parser = argparse.ArgumentParser(
        description="Generate a podcast script from filtered articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python podcast_generator.py
  python podcast_generator.py --duration 480 --output podcast_script.txt
  python podcast_generator.py --input my_articles.txt --duration 720
        """
    )
    
    parser.add_argument('--input', '-i', default='filtered_articles.txt', 
                       help='Input articles file (default: filtered_articles.txt)')
    parser.add_argument('--output', '-o', default='podcast_script.txt',
                       help='Output script file (default: podcast_script.txt)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Target duration in seconds (overrides config, default: use config value)')
    parser.add_argument('--config', '-c', default=None,
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--report-txt', default=None,
                       help='Path for the consolidated TXT report '
                            '(default: same dir as --output, named selected_articles_report.txt)')
    parser.add_argument('--report-json', default=None,
                       help='Path for the consolidated JSON report '
                            '(default: same dir as --output, named selected_articles_report.json)')
    parser.add_argument('--evidence-db', default=None,
                       help='Path to the SQLite evidence cache used for contradiction detection '
                            '(default: evidence_cache.db in the same dir as --output)')

    args = parser.parse_args()

    # Initialize generator
    generator = PodcastGenerator(args.input, args.config, evidence_db=args.evidence_db)
    
    # Use duration from args or config
    duration = args.duration
    if duration is None:
        duration = generator.config.get('target_duration_seconds', 600)
    
    print(f"Generating {duration/60:.1f}-minute podcast from {args.input}")
    
    # Generate podcast
    script = generator.generate_podcast(duration)
    
    # Save script
    generator.save_podcast_script(script, args.output)

    # Derive report / cache paths from output path when not explicitly specified
    from pathlib import Path as _Path
    out_dir = _Path(args.output).parent
    report_txt  = args.report_txt  or str(out_dir / "selected_articles_report.txt")
    report_json = args.report_json or str(out_dir / "selected_articles_report.json")
    if generator.evidence_db is None:
        generator.evidence_db = str(out_dir / "evidence_cache.db")

    # Save consolidated report (uses selected_articles stored by generate_podcast)
    if hasattr(generator, 'selected_articles') and generator.selected_articles:
        generator.save_consolidated_report(
            generator.selected_articles, report_txt, report_json
        )
    
    print("Podcast generation complete!")

if __name__ == "__main__":
    main()
