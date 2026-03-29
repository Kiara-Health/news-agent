"""
Selection Policy
================
Post-scoring article filter and enrichment layer for the fertility newsletter
pipeline.  Called after impact-score calculation, before final selection.

Responsibilities
----------------
1. Article-type classification   — multi-signal (title, content, source, summarizer)
2. Topic assignment              — primary + secondary topics with priority rules
3. Novelty suppression           — SQLite-backed cross-run seen/emitted tracking
4. Freshness checks              — age limits, missing-date penalties
5. Evidence thresholds           — minimum sufficiency per tier
6. Negative scoring              — downweight editorials, missing metadata, etc.
7. Main-story eligibility        — hard gates beyond raw score
8. Source-diversity caps         — per-source quotas applied during selection
9. Per-item diagnostics          — every article carries a ``_policy`` dict

Integration
-----------
``SelectionPolicy.apply_all(articles, db_path)`` is the single entry point called
from ``podcast_generator.py`` after scoring.  It mutates articles in-place (adds
``_policy`` and adjusts ``impact_score``) then returns the list unchanged so the
existing selection loop can filter/sort as before.

``SelectionPolicy.select_with_diversity(pool, n, tier, source_counts)`` is called
inside the selection loops to enforce source caps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config values (used when a section or key is absent from config)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict] = {
    "novelty_control": {
        "history_path": "selection_history.db",
        "suppress_previously_emitted": True,
        "suppress_previously_seen": False,
        "allow_repeat_after_days": 90,
        "lookback_days": 14,
        "identity_priority": ["doi", "pmid", "url", "fingerprint"],
        "update_history_on_selection": True,
        "update_history_on_publish": False,
    },
    "source_diversity": {
        "max_main_stories_per_source": 2,
        "max_quick_hits_per_source": 3,
        "min_distinct_sources_main": 1,
        "min_distinct_sources_total": 1,
        "source_penalty_after_threshold": 0.3,
    },
    "evidence_thresholds": {
        "min_main_story_sufficiency": 0.55,
        "min_quick_hit_sufficiency": 0.20,
        "min_main_story_confidence": "low",
        "allow_abstract_only_main_story": True,
        "allow_abstract_only_quick_hit": True,
        "drop_if_no_findings_extracted": False,
        "downgrade_if_findings_missing": True,
    },
    "freshness_rules": {
        # Legacy single flag — kept for backward compat
        "require_published_at": False,
        # Per-tier date requirements (take precedence over legacy flag)
        "require_published_at_for_main_story": True,
        "require_published_at_for_quick_hit": False,
        "max_age_days_main_story": 21,
        "max_age_days_quick_hit": 30,
        "fallback_date_sources": ["freshness_date", "updated_date", "fetched_at", "feed_timestamp"],   # empty list = no fallback
        "penalize_missing_date": True,
        "missing_date_penalty": 2.0,
    },
    "quick_hit_rules": {
        "min_evidence_sufficiency": 0.20,
        "allow_abstract_only": True,
        "require_title_and_url": True,
        "max_age_days": 30,
        "drop_if_explicitly_no_findings": False,
        "disallow_if_findings_contain": [],
        "ineligible_article_types": ["correction", "erratum", "retraction"],
    },
    "article_type_rules": {
        "explicit_patterns": {},
        "default_type": "unknown",
        "demote_types_for_main_story": [
            "editorial", "letter", "correction", "erratum",
            "retraction", "commentary", "guideline",
        ],
        "exclude_types": ["correction", "erratum", "retraction"],
        "review_type_labels": ["review", "systematic_review", "meta_analysis"],
    },
    "negative_scoring": {
        "keywords": {},
        "phrases": [],
        "article_types": {
            "editorial":   -2.0,
            "letter":      -3.0,
            "correction":  -10.0,
            "erratum":     -10.0,
            "retraction":  -10.0,
            "commentary":  -1.5,
        },
        "missing_metadata_penalties": {
            "missing_published_date": -1.0,
            "missing_findings":       -2.0,
            "low_sufficiency":        -1.5,
        },
    },
    "main_story_rules": {
        "min_reportability_score": 0.0,
        "min_impact_score": 3.0,
        "require_nontrivial_findings": True,
        "min_findings_length": 80,
        "require_distinct_angle_from_other_main_stories": False,
        "max_main_stories_same_source_family": 2,
        "require_evidence_above_quick_hit_floor": True,
        # Allowlist: only these types may become main stories.
        # Empty list = all types allowed (backward compat).
        "eligible_article_types": [],
        "ineligible_article_types": [
            "editorial", "letter", "correction", "erratum",
            "retraction", "commentary",
        ],
        # Any findings text matching one of these phrases disqualifies the article.
        "disallow_if_findings_contain": [
            "Detailed findings were not available",
            "consult the primary source for methods and results",
            "not available in the retrieved source text",
        ],
    },
    "topic_assignment": {
        "primary_topic_required": True,
        "max_topics_per_article": 3,
        "topic_priority": [],
        "min_keyword_hits_for_topic": 1,
        "allow_secondary_topics": True,
    },
    "selection_diagnostics": {
        "enabled": True,
    },
}

# ---------------------------------------------------------------------------
# Built-in article-type title patterns  (ordered: most specific first)
# ---------------------------------------------------------------------------

_BUILTIN_TYPE_PATTERNS: List[Tuple[str, str]] = [
    # Exclusion-worthy
    (r"\bretraction\b|\bretracted\b",                        "retraction"),
    (r"\bcorrigendum\b|\berratum\b|\berrata\b",             "erratum"),
    (r"\bcorrection\b(?! of)",                              "correction"),
    (r"\bexpression of concern\b",                          "retraction"),
    # Design labels
    (r"\bsystematic review\b.*\bmeta.?analysis\b"
     r"|\bmeta.?analysis\b.*\bsystematic review\b",         "meta_analysis"),
    (r"\bsystematic review\b",                              "systematic_review"),
    (r"\bmeta.?analysis\b",                                 "meta_analysis"),
    (r"\brandomized controlled trial\b|\bRCT\b",            "rct"),
    (r"\bprospective cohort\b|\bcohort study\b",            "cohort_study"),
    (r"\bretrospective\b.*\bstudy\b|\bcase.control\b",      "observational_study"),
    (r"\bcase report\b|\bcase series\b",                    "case_report"),
    (r"\bguideline[s]?\b|\bclinical practice\b",            "guideline"),
    # Non-research
    (r"\bletter to the editor\b|\bletter:\s",               "letter"),
    (r"\beditorial\b|\beditor.?s note\b",                   "editorial"),
    (r"\bcommentary\b|\bperspective\b|\bviewpoint\b",       "commentary"),
    (r"\breview article\b|\bnarrative review\b",            "review"),
    # Generic review (least specific — must come last)
    (r"\breview\b",                                         "review"),
]

_CONFIDENCE_ORDER = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge(user: Dict, defaults: Dict) -> Dict:
    """Shallow-merge user config over defaults."""
    result = dict(defaults)
    result.update(user)
    return result


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _content_fingerprint(article: Dict) -> str:
    """Short SHA-256 fingerprint of title + url for dedup."""
    raw = (article.get("title") or "") + "|" + (article.get("url") or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalize_source(source: str) -> str:
    """Collapse source variants to a canonical label for source-family grouping."""
    return re.sub(r"\s+[-–—]\s+.*", "", source or "").strip().lower()


def _estimate_evidence_sufficiency_heuristic(article: Dict) -> float:
    """
    Pre-LLM estimate of evidence sufficiency.

    Called by ``check_evidence`` when the article has not yet been through the
    LLM summariser (``evidence_sufficiency`` is absent or 0.0).

    Signals used (in order of reliability):
    1. ``evidence_tier_estimate`` — set by ``calculate_impact_score`` in podcast_generator
    2. Raw content length — proxy for abstract vs. snippet
    3. ``evidence_quality.source_text_quality`` — set by prior runs of the summariser

    Calibration
    -----------
    * snippet_only  / titles_to_watch: ~0.08  (title + no abstract)
    * short abstract / short_blurb:    ~0.30  (partial abstract)
    * full abstract  / full tier:       ~0.60  (structured abstract)
    * full text:                        ~0.80  (complete paper)
    """
    tier = article.get("evidence_tier_estimate")
    content_len = len(article.get("content") or "")

    # Derive tier from content length if not already set
    if not tier:
        if content_len < 300:
            tier = "titles_to_watch"
        elif content_len < 800:
            tier = "short_blurb"
        else:
            tier = "full"

    base = {"full": 0.60, "short_blurb": 0.30, "titles_to_watch": 0.08}.get(tier, 0.30)

    # Refine with source_text_quality if available from a previous run
    eq = article.get("evidence_quality") or {}
    squal = (eq.get("source_text_quality") or "").strip()
    if squal == "snippet_only":
        base = min(base, 0.12)
    elif squal == "full_text":
        base = max(base, 0.70)
    elif squal == "abstract":
        # Keep tier-derived estimate; abstract is already captured by content length
        pass

    return round(base, 3)


# ---------------------------------------------------------------------------
# SQLite novelty store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS selection_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT    NOT NULL,
    doi            TEXT,
    pmid           TEXT,
    fingerprint    TEXT,
    title          TEXT,
    source         TEXT,
    published_date TEXT,
    first_seen_at  TEXT    NOT NULL,
    last_seen_at   TEXT    NOT NULL,
    emitted_at     TEXT,
    run_id         TEXT,
    UNIQUE(url)
);
CREATE INDEX IF NOT EXISTS sh_doi         ON selection_history(doi);
CREATE INDEX IF NOT EXISTS sh_fingerprint ON selection_history(fingerprint);
"""


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _lookup_article(conn: sqlite3.Connection, article: Dict,
                    identity_priority: List[str]) -> Optional[sqlite3.Row]:
    """
    Return the first matching row using the configured identity fields,
    tried in priority order.
    """
    doi         = (article.get("doi") or "").strip() or None
    pmid        = (article.get("pmid") or "").strip() or None
    url         = (article.get("url") or "").strip() or None
    fingerprint = _content_fingerprint(article)

    field_map = {
        "doi":         ("doi",         doi),
        "pmid":        ("pmid",        pmid),
        "url":         ("url",         url),
        "fingerprint": ("fingerprint", fingerprint),
    }
    conn.row_factory = sqlite3.Row
    for key in identity_priority:
        col, val = field_map.get(key, (None, None))
        if val:
            row = conn.execute(
                f"SELECT * FROM selection_history WHERE {col} = ?", (val,)
            ).fetchone()
            if row:
                return row
    return None


def _upsert_article(conn: sqlite3.Connection, article: Dict,
                    run_id: str, emit: bool = False) -> None:
    now        = _now_utc().isoformat()
    url        = (article.get("url") or "").strip()
    doi        = (article.get("doi") or "").strip() or None
    pmid       = (article.get("pmid") or "").strip() or None
    fingerprint = _content_fingerprint(article)
    pub_date   = (
        article["published_date"].strftime("%Y-%m-%d")
        if article.get("published_date") else None
    )
    emitted_at = now if emit else None

    conn.execute(
        """
        INSERT INTO selection_history
            (url, doi, pmid, fingerprint, title, source, published_date,
             first_seen_at, last_seen_at, emitted_at, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            last_seen_at  = excluded.last_seen_at,
            emitted_at    = COALESCE(excluded.emitted_at, emitted_at),
            run_id        = excluded.run_id
        """,
        (url, doi, pmid, fingerprint,
         article.get("title"), article.get("source"), pub_date,
         now, now, emitted_at, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# SelectionPolicy
# ---------------------------------------------------------------------------

class SelectionPolicy:
    """
    Stateless (per-run) policy engine.  State across runs is stored in SQLite.

    Typical usage in podcast_generator.py
    --------------------------------------
    ::

        policy = SelectionPolicy(self.config)
        articles = policy.apply_all(articles, db_path=self.evidence_db)
        # articles now have ._policy dicts; impact_score may be adjusted

        # during selection — source diversity
        main_stories = policy.select_with_diversity(
            pool=candidates, n=max_main, tier="main_story"
        )
    """

    def __init__(self, config: Dict):
        self.config = config
        self._nc   = _merge(config.get("novelty_control",       {}), _DEFAULTS["novelty_control"])
        self._sd   = _merge(config.get("source_diversity",      {}), _DEFAULTS["source_diversity"])
        self._et   = _merge(config.get("evidence_thresholds",   {}), _DEFAULTS["evidence_thresholds"])
        self._fr   = _merge(config.get("freshness_rules",       {}), _DEFAULTS["freshness_rules"])
        self._atr  = _merge(config.get("article_type_rules",    {}), _DEFAULTS["article_type_rules"])
        self._ns   = _merge(config.get("negative_scoring",      {}), _DEFAULTS["negative_scoring"])
        self._msr  = _merge(config.get("main_story_rules",      {}), _DEFAULTS["main_story_rules"])
        self._qhr  = _merge(config.get("quick_hit_rules",       {}), _DEFAULTS["quick_hit_rules"])
        self._ta   = _merge(config.get("topic_assignment",      {}), _DEFAULTS["topic_assignment"])
        self._topic_cats: Dict[str, List[str]] = config.get("topic_categories", {})
        self._topic_priority: List[str] = (
            self._ta.get("topic_priority") or list(self._topic_cats.keys())
        )

    # ------------------------------------------------------------------
    # 1. Article-type classification
    # ------------------------------------------------------------------

    def classify_article_type(self, article: Dict) -> str:
        """
        Assign an article type using multiple signals in priority order:
        1. Title patterns (most reliable)
        2. Config explicit_patterns
        3. Content/abstract text patterns
        4. Existing summarizer output (weakest — may be wrong)

        Returns one of: original_research, review, systematic_review,
        meta_analysis, rct, cohort_study, observational_study, case_report,
        editorial, commentary, letter, guideline, correction, erratum,
        retraction, unknown.
        """
        title   = (article.get("title")   or "").lower()
        content = (article.get("content") or "").lower()
        probe   = title + " " + content[:500]  # title gets extra weight via position

        # 1. User-configured explicit patterns (highest authority)
        for pattern, label in (self._atr.get("explicit_patterns") or {}).items():
            if re.search(pattern, probe, re.IGNORECASE):
                return label

        # 2. Title-only check against built-ins (strong signal)
        for pattern, label in _BUILTIN_TYPE_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                return label

        # 3. Full probe (title + content) against built-ins
        for pattern, label in _BUILTIN_TYPE_PATTERNS:
            if re.search(pattern, probe, re.IGNORECASE):
                return label

        # 4. Summarizer evidence_quality.article_type (weakest signal)
        eq = (article.get("evidence_quality") or {})
        existing = (eq.get("article_type") or "").lower().strip()
        if existing and existing not in ("unknown", ""):
            return existing

        return self._atr.get("default_type", "unknown")

    # ------------------------------------------------------------------
    # 2. Topic assignment
    # ------------------------------------------------------------------

    def assign_topics(self, article: Dict) -> Tuple[str, List[str]]:
        """
        Return ``(primary_topic, secondary_topics)`` using keyword hit counts.
        Falls back to the article's existing ``topic`` field if no hits.
        """
        min_hits    = self._ta.get("min_keyword_hits_for_topic", 1)
        max_topics  = self._ta.get("max_topics_per_article", 3)
        allow_sec   = self._ta.get("allow_secondary_topics", True)

        probe = (
            (article.get("title") or "")
            + " " + (article.get("content") or "")[:1000]
        ).lower()

        hit_counts: Dict[str, int] = {}
        for topic, keywords in self._topic_cats.items():
            hits = sum(
                1 for kw in keywords
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", probe)
            )
            if hits >= min_hits:
                hit_counts[topic] = hits

        if not hit_counts:
            existing = article.get("topic") or "general"
            return existing, []

        # Sort by hit count descending; break ties by priority order
        priority_idx = {t: i for i, t in enumerate(self._topic_priority)}
        ranked = sorted(
            hit_counts.items(),
            key=lambda kv: (-kv[1], priority_idx.get(kv[0], 9999)),
        )

        primary = ranked[0][0]
        if allow_sec:
            secondary = [t for t, _ in ranked[1:max_topics]]
        else:
            secondary = []
        return primary, secondary

    # ------------------------------------------------------------------
    # 3. Novelty check
    # ------------------------------------------------------------------

    def check_novelty(
        self, article: Dict, db_path: Optional[str]
    ) -> Dict:
        """
        Returns a novelty dict:
        ``{status, first_seen_days_ago, emitted_days_ago, suppressed,
           suppress_reason, identity_matched_by}``
        """
        result = {
            "status":               "new",
            "first_seen_days_ago":  None,
            "emitted_days_ago":     None,
            "suppressed":           False,
            "suppress_reason":      None,
            "identity_matched_by":  None,
        }
        if not db_path:
            return result

        allow_repeat_after = self._nc.get("allow_repeat_after_days", 90)
        suppress_emitted   = self._nc.get("suppress_previously_emitted", True)
        suppress_seen      = self._nc.get("suppress_previously_seen", False)
        priority           = self._nc.get("identity_priority",
                                          ["doi", "pmid", "url", "fingerprint"])

        try:
            conn = _open_db(db_path)
            row  = _lookup_article(conn, article, priority)
            conn.close()
        except Exception as exc:
            logger.warning("Novelty check DB error: %s", exc)
            return result

        if row is None:
            return result

        now = _now_utc()
        first_seen = datetime.fromisoformat(row["first_seen_at"]).replace(tzinfo=timezone.utc)
        first_seen_days = (now - first_seen).days
        result["first_seen_days_ago"] = first_seen_days

        if row["emitted_at"]:
            emitted_dt = datetime.fromisoformat(row["emitted_at"]).replace(tzinfo=timezone.utc)
            emitted_days = (now - emitted_dt).days
            result["emitted_days_ago"] = emitted_days
            result["status"] = "emitted"

            if emitted_days < allow_repeat_after and suppress_emitted:
                result["suppressed"]      = True
                result["suppress_reason"] = (
                    f"previously emitted {emitted_days}d ago "
                    f"(allow_repeat_after={allow_repeat_after}d)"
                )
                return result
        else:
            result["status"] = "seen"
            if first_seen_days < allow_repeat_after and suppress_seen:
                result["suppressed"]      = True
                result["suppress_reason"] = (
                    f"previously seen {first_seen_days}d ago "
                    f"(suppress_previously_seen=true)"
                )
                return result

        return result

    # ------------------------------------------------------------------
    # 4. Freshness check
    # ------------------------------------------------------------------

    def check_freshness(self, article: Dict) -> Dict:
        """
        Returns freshness dict:
        ``{status, age_days, has_published_date, date_source,
           eligible_for_main, eligible_for_quick_hit, suppressed, suppress_reason}``

        Per-tier requirements (require_published_at_for_main_story /
        require_published_at_for_quick_hit) take precedence over the legacy
        ``require_published_at`` flag.
        """
        max_main  = self._fr.get("max_age_days_main_story", 21)
        max_quick = self._fr.get("max_age_days_quick_hit",  30)

        # Per-tier date requirements (new) — fall back to legacy flag
        require_main  = self._fr.get(
            "require_published_at_for_main_story",
            self._fr.get("require_published_at", False),
        )
        require_quick = self._fr.get(
            "require_published_at_for_quick_hit",
            self._fr.get("require_published_at", False),
        )

        # Allowed fallback date sources (empty list = no fallback)
        fallback_sources = self._fr.get(
            "fallback_date_sources",
            # legacy: if old flag was True, default to ["feed_timestamp"]
            ["feed_timestamp"] if self._fr.get("fallback_to_feed_timestamp_if_missing", True) else [],
        )

        result = {
            "status":                  "unknown",
            "age_days":                None,
            "has_published_date":      False,
            "has_actual_publication_date": False,
            "date_source":             "none",
            "used_fallback_date":      False,
            "freshness_confidence":    "low",
            "eligible_for_main":       True,
            "eligible_for_quick_hit":  True,
            "suppressed":              False,
            "suppress_reason":         None,
        }

        pub_dt = _to_utc(article.get("published_date"))
        if pub_dt:
            result["date_source"] = "published_date"
            result["has_actual_publication_date"] = True
            result["freshness_confidence"] = "high"
        else:
            for src in fallback_sources:
                fallback_dt = _to_utc(article.get(src))
                if fallback_dt:
                    pub_dt = fallback_dt
                    result["date_source"] = src
                    result["used_fallback_date"] = True
                    result["freshness_confidence"] = (
                        "medium" if src in ("freshness_date", "updated_date") else "low"
                    )
                    break

        if pub_dt is None:
            result["has_published_date"] = False
            result["status"] = "missing_freshness_date"
            if require_main:
                result["eligible_for_main"] = False
            if require_quick:
                result["eligible_for_quick_hit"] = False
            # Suppress (hard-exclude) when either per-tier require flag is set,
            # or the legacy require_published_at flag is set (backward compat).
            legacy_require = self._fr.get("require_published_at", False)
            if require_quick or (legacy_require and require_main):
                result["suppressed"]      = True
                result["suppress_reason"] = "missing usable freshness date"
            return result

        result["has_published_date"] = bool(result["has_actual_publication_date"])
        age_days = (datetime.now(tz=timezone.utc) - pub_dt).total_seconds() / 86400
        result["age_days"] = round(age_days, 1)

        if age_days > max_main:
            result["eligible_for_main"] = False
        if age_days > max_quick:
            result["eligible_for_quick_hit"] = False
            result["status"]     = "stale"
            result["suppressed"] = True
            result["suppress_reason"] = (
                f"article is {int(age_days)}d old (max_age_days_quick_hit={max_quick})"
            )
        else:
            result["status"] = "fresh"

        return result

    # ------------------------------------------------------------------
    # 5. Evidence status
    # ------------------------------------------------------------------

    def check_evidence(self, article: Dict) -> Dict:
        """
        Returns evidence dict:
        ``{sufficiency, estimated, confidence, has_findings, source_quality,
           eligible_for_main, eligible_for_quick_hit,
           suppressed, suppress_reason}``

        When ``evidence_sufficiency`` is absent or 0.0 (pre-LLM), the
        heuristic ``_estimate_evidence_sufficiency_heuristic`` is used so that
        the eligibility gates are meaningful before the summariser runs.
        """
        min_main  = self._et.get("min_main_story_sufficiency",  0.55)
        min_quick = self._et.get("min_quick_hit_sufficiency",   0.20)
        allow_abs_main  = self._et.get("allow_abstract_only_main_story", True)
        allow_abs_quick = self._et.get("allow_abstract_only_quick_hit",  True)
        drop_no_findings = self._et.get("drop_if_no_findings_extracted", False)
        conf_map = {"high": 2, "medium": 1, "low": 0}
        min_conf_str = self._et.get("min_main_story_confidence", "low")
        min_conf_int = conf_map.get(min_conf_str, 0)

        # Use actual post-LLM value when available; fall back to heuristic
        ev_suf_actual = float(article.get("evidence_sufficiency") or 0.0)
        if ev_suf_actual > 0.0:
            ev_suf    = ev_suf_actual
            estimated = False
        else:
            ev_suf    = _estimate_evidence_sufficiency_heuristic(article)
            estimated = True

        eq = article.get("evidence_quality") or {}
        confidence   = eq.get("confidence", "low")
        src_quality  = eq.get("source_text_quality", "snippet_only")
        has_findings = bool(
            (article.get("summary_sections") or {}).get("what_it_found") or
            (article.get("generated_summary") and len(article.get("generated_summary", "")) > 50)
        )

        result = {
            "sufficiency":          round(ev_suf, 3),
            "estimated":            estimated,
            "confidence":           confidence,
            "has_findings":         has_findings,
            "source_quality":       src_quality,
            "eligible_for_main":    True,
            "eligible_for_quick_hit": True,
            "suppressed":           False,
            "suppress_reason":      None,
        }

        if drop_no_findings and not has_findings:
            result["suppressed"]         = True
            result["suppress_reason"]    = "no findings extracted (drop_if_no_findings_extracted=true)"
            result["eligible_for_main"]  = False
            result["eligible_for_quick_hit"] = False
            return result

        if ev_suf < min_quick:
            result["eligible_for_quick_hit"] = False
            result["eligible_for_main"]      = False
        elif ev_suf < min_main:
            result["eligible_for_main"] = False

        if not allow_abs_main and src_quality == "abstract":
            result["eligible_for_main"] = False
        if not allow_abs_quick and src_quality == "abstract":
            result["eligible_for_quick_hit"] = False

        conf_int = conf_map.get(confidence, 0)
        if conf_int < min_conf_int:
            result["eligible_for_main"] = False

        return result

    # ------------------------------------------------------------------
    # 6. Negative scoring
    # ------------------------------------------------------------------

    def negative_score_adjustment(self, article: Dict) -> float:
        """
        Return a negative float to subtract from (or add to) the impact score.
        """
        adj: float = 0.0
        title   = (article.get("title")   or "").lower()
        content = (article.get("content") or "").lower()
        probe   = title + " " + content[:300]

        # Keyword penalties
        for kw, penalty in (self._ns.get("keywords") or {}).items():
            if kw.lower() in probe:
                adj += float(penalty)

        # Phrase penalties
        for phrase in (self._ns.get("phrases") or []):
            if phrase.lower() in probe:
                adj -= 1.0

        # Article-type penalties
        art_type  = article.get("_policy", {}).get("article_type") or "unknown"
        type_pen  = (self._ns.get("article_types") or {}).get(art_type, 0.0)
        adj += float(type_pen)

        # Missing-metadata penalties
        meta_pen = self._ns.get("missing_metadata_penalties") or {}
        if not article.get("published_date"):
            adj += float(meta_pen.get("missing_published_date", 0.0))

        ev_status = article.get("_policy", {}).get("evidence_status") or {}
        if not ev_status.get("has_findings", True):
            adj += float(meta_pen.get("missing_findings", 0.0))
        if ev_status.get("sufficiency", 1.0) < self._et.get("min_main_story_sufficiency", 0.40):
            adj += float(meta_pen.get("low_sufficiency", 0.0))

        return adj

    # ------------------------------------------------------------------
    # 7. Main-story eligibility gate
    # ------------------------------------------------------------------

    def is_main_story_eligible(self, article: Dict,
                                current_main_stories: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Hard eligibility gate before promoting an article to a main story.

        Checks (in order):
        1. Evidence sufficiency  — must meet threshold (uses heuristic pre-LLM)
        2. Freshness / date      — missing date fails when require_published_at_for_main_story=true
        3. Article-type allowlist / ineligible types
        4. Score gates
        5. Findings quality      — min length and banned phrases
        6. Source family cap

        Returns ``(eligible: bool, reason: str)``.
        """
        impact        = article.get("impact_score", 0.0)
        reportability = article.get("reportability_score", impact)
        policy        = article.get("_policy", {})
        art_type      = policy.get("article_type", "unknown")
        evidence      = policy.get("evidence_status", {})
        freshness     = policy.get("freshness_status", {})

        min_impact   = self._msr.get("min_impact_score", 3.0)
        min_rep      = self._msr.get("min_reportability_score", 0.0)
        req_findings = self._msr.get("require_nontrivial_findings", True)
        req_ev_floor = self._msr.get("require_evidence_above_quick_hit_floor", True)
        min_findings_len = self._msr.get("min_findings_length", 80)

        # ── 1. Evidence threshold ──────────────────────────────────────────
        if not evidence.get("eligible_for_main", False):
            suf = evidence.get("sufficiency", 0.0)
            est = " (estimated)" if evidence.get("estimated") else ""
            return False, (
                f"evidence_sufficiency {suf:.2f}{est} < "
                f"min_main_story_sufficiency {self._et.get('min_main_story_sufficiency', 0.55)}"
            )

        # ── 2. Date / freshness ────────────────────────────────────────────
        if not freshness.get("eligible_for_main", True):
            detail = freshness.get("status", "missing_date")
            return False, f"freshness gate failed: {detail}"

        # ── 3. Article-type allowlist / ineligible list ────────────────────
        ineligible_types = (
            self._msr.get("ineligible_article_types") or
            self._atr.get("demote_types_for_main_story") or []
        )
        if art_type in ineligible_types:
            return False, f"article type '{art_type}' is ineligible for main stories"

        eligible_types = self._msr.get("eligible_article_types") or []
        if eligible_types and art_type not in eligible_types:
            return False, (
                f"article type '{art_type}' not in eligible_article_types "
                f"({', '.join(eligible_types)})"
            )

        # ── 4. Score gates ─────────────────────────────────────────────────
        if impact < min_impact:
            return False, f"impact_score {impact:.1f} < min_impact_score {min_impact}"
        if reportability < min_rep:
            return False, f"reportability {reportability:.2f} < min_reportability_score {min_rep}"

        # ── 5. Findings quality ────────────────────────────────────────────
        if req_findings and not evidence.get("has_findings", True):
            return False, "require_nontrivial_findings: no findings extracted"

        # Findings must meet minimum length
        findings_text = (
            (article.get("summary_sections") or {}).get("what_it_found") or
            article.get("generated_summary") or ""
        )
        if len(findings_text.strip()) < min_findings_len:
            return False, (
                f"findings text length {len(findings_text.strip())} "
                f"< min_findings_length {min_findings_len}"
            )

        # Findings must not contain disqualifying boilerplate
        disallow = self._msr.get("disallow_if_findings_contain") or []
        for phrase in disallow:
            if phrase.lower() in findings_text.lower():
                return False, f"findings contain disqualifying phrase: '{phrase[:60]}'"

        # Evidence above quick-hit floor (belt-and-suspenders)
        if req_ev_floor:
            min_quick = self._et.get("min_quick_hit_sufficiency", 0.20)
            suf = evidence.get("sufficiency", 0.0)
            if suf < min_quick:
                return False, (
                    f"evidence_sufficiency {suf:.2f} below quick-hit floor {min_quick}"
                )

        # ── 6. Source family cap ───────────────────────────────────────────
        if current_main_stories:
            src_family  = _normalize_source(article.get("source", ""))
            same_family = sum(
                1 for a in current_main_stories
                if _normalize_source(a.get("source", "")) == src_family
            )
            max_family  = self._msr.get("max_main_stories_same_source_family", 2)
            if same_family >= max_family:
                return False, (
                    f"source family '{src_family}' already has {same_family}/{max_family} main stories"
                )

        return True, "ok"

    # ------------------------------------------------------------------
    # 7b. Quick-hit eligibility gate
    # ------------------------------------------------------------------

    def is_quick_hit_eligible(self, article: Dict) -> Tuple[bool, str]:
        """
        Eligibility gate for quick hits (more permissive than main stories).

        Returns ``(eligible: bool, reason: str)``.
        """
        policy   = article.get("_policy", {})
        art_type = policy.get("article_type", "unknown")
        evidence = policy.get("evidence_status", {})
        freshness = policy.get("freshness_status", {})

        # Already hard-suppressed (excluded type, stale, novelty)
        if policy.get("suppressed"):
            return False, policy.get("suppressed_reason") or "suppressed"

        # Ineligible article types for quick hits
        ineligible = self._qhr.get("ineligible_article_types") or ["correction", "erratum", "retraction"]
        if art_type in ineligible:
            return False, f"article type '{art_type}' is ineligible for quick hits"

        # Evidence threshold
        if not evidence.get("eligible_for_quick_hit", False):
            suf = evidence.get("sufficiency", 0.0)
            return False, (
                f"evidence_sufficiency {suf:.2f} < "
                f"min_quick_hit_sufficiency {self._qhr.get('min_evidence_sufficiency', 0.20)}"
            )

        # Freshness — quick hits not suppressed by missing date (only stale)
        if freshness.get("status") == "stale":
            return False, freshness.get("suppress_reason", "article is stale")

        # Age limit for quick hits
        max_age = self._qhr.get("max_age_days", 30)
        age = freshness.get("age_days")
        if age is not None and age > max_age:
            return False, f"article is {int(age)}d old > max_age_days {max_age}"

        # Require title and URL
        if self._qhr.get("require_title_and_url", True):
            if not (article.get("title") or "").strip():
                return False, "missing title"
            if not (article.get("url") or "").strip():
                return False, "missing url"

        # Optional: disallow specific fallback phrases in findings
        if self._qhr.get("drop_if_explicitly_no_findings", False):
            findings = (
                (article.get("summary_sections") or {}).get("what_it_found") or
                article.get("generated_summary") or ""
            )
            for phrase in (self._qhr.get("disallow_if_findings_contain") or []):
                if phrase.lower() in findings.lower():
                    return False, f"findings contain disqualifying phrase: '{phrase[:60]}'"

        return True, "ok"

    # ------------------------------------------------------------------
    # 8. Source-diversity selection
    # ------------------------------------------------------------------

    def select_with_diversity(
        self,
        pool: List[Dict],
        n: int,
        tier: str = "main_story",
        existing_source_counts: Optional[Dict[str, int]] = None,
    ) -> List[Dict]:
        """
        Greedy selection of up to ``n`` articles from ``pool`` while enforcing
        per-source caps defined in ``source_diversity`` config.

        ``tier`` is "main_story" or "quick_hit".
        ``existing_source_counts`` lets callers pass source tallies from prior
        selections so main-story + quick-hit caps are tracked jointly.
        """
        if tier == "main_story":
            cap = self._sd.get("max_main_stories_per_source", 2)
        else:
            cap = self._sd.get("max_quick_hits_per_source", 3)

        source_counts: Dict[str, int] = dict(existing_source_counts or {})
        selected: List[Dict] = []

        for article in pool:
            if len(selected) >= n:
                break
            src = _normalize_source(article.get("source", ""))
            count = source_counts.get(src, 0)
            if count >= cap:
                diag = article.setdefault("_policy", {})
                if not diag.get("source_diversity_status"):
                    diag["source_diversity_status"] = (
                        f"capped: source '{src}' already has {count}/{cap} {tier}s"
                    )
                continue
            source_counts[src] = count + 1
            article.setdefault("_policy", {})["source_diversity_status"] = "ok"
            selected.append(article)

        return selected

    # ------------------------------------------------------------------
    # 9. Master apply_all
    # ------------------------------------------------------------------

    def apply_all(
        self,
        articles: List[Dict],
        db_path: Optional[str] = None,
        run_id: str = "default",
    ) -> List[Dict]:
        """
        Run all policy checks on every article in-place.

        For each article, attaches ``article['_policy']``:
        ::

            {
              "article_type":           str,
              "primary_topic":          str,
              "secondary_topics":       List[str],
              "novelty_status":         Dict,
              "freshness_status":       Dict,
              "evidence_status":        Dict,
              "negative_adjustment":    float,
              "source_diversity_status": str | None,
              "main_story_eligible":    bool,
              "main_story_ineligible_reason": str,
              "suppressed":             bool,
              "suppressed_reason":      str | None,
              "selected_as":            None,
              "selection_stage_scores": Dict,
            }

        Also adjusts ``article['impact_score']`` by applying the negative
        adjustment.  Excluded article types (correction/erratum/retraction) are
        flagged with ``suppressed=True`` so the selection loop can skip them.
        """
        db_path_eff = db_path or self._nc.get("history_path", "selection_history.db")

        for article in articles:
            # --- Classification ----------------------------------------
            art_type = self.classify_article_type(article)
            primary_topic, secondary_topics = self.assign_topics(article)

            # Override the article's topic field with the improved assignment
            article["topic"]           = primary_topic
            article["secondary_topics"] = secondary_topics

            # --- Initialise _policy before negative_score_adjustment ----
            policy: Dict = {
                "article_type":                art_type,
                "article_type_canonical":      art_type,
                "primary_topic":               primary_topic,
                "secondary_topics":            secondary_topics,
                "novelty_status":              {},
                "freshness_status":            {},
                "evidence_status":             {},
                "evidence_sufficiency_canonical": 0.0,
                "novelty_state_canonical":     "new",
                "negative_adjustment":         0.0,
                "source_diversity_status":     None,
                "main_story_eligible":         True,
                "main_story_ineligible_reason": "not yet evaluated",
                "quick_hit_eligible":          True,
                "quick_hit_ineligible_reason": None,
                "coverage_candidate":          True,
                "publishable_candidate":       False,
                "rejection_reasons":           [],
                "downgrade_reasons":           [],
                "suppressed":                  False,
                "suppressed_reason":           None,
                "selected_as":                 None,
                "selection_stage_scores":      {
                    "base_impact":        round(article.get("impact_score", 0.0), 3),
                    "negative_adjustment": 0.0,
                    "final_impact":       round(article.get("impact_score", 0.0), 3),
                    "reportability":      round(article.get("reportability_score", 0.0), 3),
                },
            }
            article["_policy"] = policy

            # --- Novelty -----------------------------------------------
            novelty = self.check_novelty(article, db_path_eff)
            policy["novelty_status"] = novelty
            policy["novelty_state_canonical"] = novelty.get("status", "new")

            # --- Freshness ---------------------------------------------
            freshness = self.check_freshness(article)
            policy["freshness_status"] = freshness

            # --- Evidence ----------------------------------------------
            evidence = self.check_evidence(article)
            policy["evidence_status"] = evidence
            policy["evidence_sufficiency_canonical"] = round(
                float(evidence.get("sufficiency", 0.0)), 3
            )

            # --- Negative scoring --------------------------------------
            neg_adj = self.negative_score_adjustment(article)
            policy["negative_adjustment"] = round(neg_adj, 3)
            base = article.get("impact_score", 0.0)
            article["impact_score"] = round(base + neg_adj, 3)
            policy["selection_stage_scores"].update({
                "negative_adjustment": round(neg_adj, 3),
                "final_impact":        round(base + neg_adj, 3),
            })

            # --- Hard suppression --------------------------------------
            exclude_types = self._atr.get("exclude_types", [])
            if art_type in exclude_types:
                policy["suppressed"]       = True
                policy["suppressed_reason"] = f"article type '{art_type}' is excluded"
                policy["rejection_reasons"].append(policy["suppressed_reason"])

            elif novelty.get("suppressed"):
                policy["suppressed"]       = True
                policy["suppressed_reason"] = novelty["suppress_reason"]
                policy["rejection_reasons"].append(policy["suppressed_reason"])

            elif freshness.get("suppressed"):
                policy["suppressed"]       = True
                policy["suppressed_reason"] = freshness["suppress_reason"]
                policy["rejection_reasons"].append(policy["suppressed_reason"])

            elif evidence.get("suppressed"):
                policy["suppressed"]       = True
                policy["suppressed_reason"] = evidence["suppress_reason"]
                policy["rejection_reasons"].append(policy["suppressed_reason"])

            # --- Main-story eligibility (informational at this stage) --
            eligible, reason = self.is_main_story_eligible(article)
            policy["main_story_eligible"]         = eligible
            policy["main_story_ineligible_reason"] = reason
            if not eligible:
                policy["downgrade_reasons"].append(reason)

            # --- Quick-hit eligibility (informational at this stage) ---
            qh_eligible, qh_reason = self.is_quick_hit_eligible(article)
            policy["quick_hit_eligible"] = qh_eligible
            policy["quick_hit_ineligible_reason"] = None if qh_eligible else qh_reason
            if not qh_eligible and qh_reason:
                policy["rejection_reasons"].append(qh_reason)

            # Coverage includes all non-suppressed candidates from retrieval window.
            policy["coverage_candidate"] = not policy.get("suppressed", False)
            # Publishable means eligible for at least one output tier.
            policy["publishable_candidate"] = bool(
                (not policy.get("suppressed", False))
                and (policy["main_story_eligible"] or policy["quick_hit_eligible"])
            )

        # --- Novelty DB: record as seen --------------------------------
        if self._nc.get("update_history_on_selection", True) and db_path_eff:
            try:
                conn = _open_db(db_path_eff)
                for article in articles:
                    _upsert_article(conn, article, run_id=run_id, emit=False)
                conn.close()
            except Exception as exc:
                logger.warning("Error writing novelty history: %s", exc)

        return articles

    # ------------------------------------------------------------------
    # Mark emitted (called after publication)
    # ------------------------------------------------------------------

    def mark_emitted(
        self, articles: List[Dict], db_path: Optional[str] = None, run_id: str = "default"
    ) -> None:
        """Mark a list of selected articles as emitted in the novelty DB."""
        if not self._nc.get("update_history_on_publish", False):
            return
        db_path_eff = db_path or self._nc.get("history_path", "selection_history.db")
        try:
            conn = _open_db(db_path_eff)
            for article in articles:
                _upsert_article(conn, article, run_id=run_id, emit=True)
            conn.close()
        except Exception as exc:
            logger.warning("Error marking articles as emitted: %s", exc)


# ---------------------------------------------------------------------------
# Issue-state helpers
# ---------------------------------------------------------------------------

def compute_issue_state(
    main_stories: List[Dict],
    quick_hits: List[Dict],
    max_main: int,
    max_qh: int,
) -> str:
    """
    Return a string describing the quality state of the current selection run.

    ``full_issue``           — main stories and quick hits are both at capacity
    ``underfilled_issue``    — some main-story slots are empty but at least one filled
    ``quick_hits_only``      — no main stories passed eligibility gates
    ``no_publishable_items`` — nothing passed any gate
    """
    n_main = len(main_stories)
    n_qh   = len(quick_hits)

    if n_main == 0 and n_qh == 0:
        return "no_publishable_items"
    if n_main == 0:
        return "quick_hits_only"
    if n_main < max_main or n_qh < max_qh:
        return "underfilled_issue"
    return "full_issue"


# ---------------------------------------------------------------------------
# Diagnostics summary (for report)
# ---------------------------------------------------------------------------

def build_selection_diagnostics_summary(
    all_articles: List[Dict],
    selected: List[Dict],
    issue_state: str = "unknown",
    max_main: int = 0,
    max_qh: int = 0,
) -> Dict:
    """
    Build a machine-readable diagnostics dict covering all filtering stages.
    Suitable for inclusion in the consolidated report JSON.
    """
    total = len(all_articles)

    def _count(predicate) -> int:
        return sum(1 for a in all_articles if predicate(a))

    def _policy(a: Dict) -> Dict:
        return a.get("_policy") or {}

    suppressed_novelty   = _count(lambda a: (_policy(a).get("novelty_status")   or {}).get("suppressed"))
    suppressed_freshness = _count(lambda a: (_policy(a).get("freshness_status") or {}).get("suppressed"))
    suppressed_evidence  = _count(lambda a: (_policy(a).get("evidence_status")  or {}).get("suppressed"))
    suppressed_type      = _count(lambda a: (_policy(a).get("suppressed_reason") or "").startswith("article type"))
    suppressed_diversity = _count(lambda a: (_policy(a).get("source_diversity_status") or "").startswith("capped"))
    total_suppressed     = _count(lambda a: _policy(a).get("suppressed"))
    previously_seen      = _count(lambda a: (_policy(a).get("novelty_status") or {}).get("status") == "seen")
    previously_emitted   = _count(lambda a: (_policy(a).get("novelty_status") or {}).get("status") == "emitted")
    missing_date         = _count(
        lambda a: not ((_policy(a).get("freshness_status") or {}).get("has_actual_publication_date", False))
    )
    fallback_freshness   = _count(
        lambda a: bool((_policy(a).get("freshness_status") or {}).get("used_fallback_date"))
    )

    # Articles rejected for specific reasons in the selection loop
    rejected_main  = _count(lambda a: not _policy(a).get("main_story_eligible", True)
                                       and not _policy(a).get("suppressed"))
    main_eligible  = _count(lambda a: _policy(a).get("main_story_eligible", False))
    publishable    = _count(lambda a: _policy(a).get("publishable_candidate", False))
    coverage       = _count(lambda a: _policy(a).get("coverage_candidate", False))

    after_novelty  = _count(lambda a: not ((_policy(a).get("novelty_status") or {}).get("suppressed")))
    after_fresh    = _count(lambda a: not ((_policy(a).get("freshness_status") or {}).get("suppressed")))
    after_evidence = _count(lambda a: not ((_policy(a).get("evidence_status") or {}).get("suppressed")))
    after_source_caps = total - suppressed_diversity

    main_count  = sum(1 for a in selected if (_policy(a).get("selected_as") == "main_story"))
    quick_count = sum(1 for a in selected if (_policy(a).get("selected_as") == "quick_hit"))
    selected_total = len(selected)
    role_mismatch = selected_total != (main_count + quick_count)

    underfilled_main = max(0, max_main - main_count) if max_main else 0
    underfilled_qh   = max(0, max_qh   - quick_count) if max_qh  else 0

    # Rejection reasons histogram
    rejection_counts: Dict[str, int] = defaultdict(int)
    for a in all_articles:
        p = _policy(a)
        if p.get("suppressed_reason"):
            key = p["suppressed_reason"].split(" ")[0:4]  # first 4 words
            rejection_counts[" ".join(key)] += 1
        elif not p.get("main_story_eligible", True):
            reason = p.get("main_story_ineligible_reason", "unknown gate")
            key = reason.split(" ")[0:4]
            rejection_counts[" ".join(key)] += 1

    # Source breakdown for selected
    source_freq: Dict[str, int] = defaultdict(int)
    for a in selected:
        source_freq[a.get("source", "unknown")] += 1

    return {
        "issue_state":                 issue_state,
        "candidate_count":             total,
        "coverage_candidates":         coverage,
        "publishable_candidates":      publishable,
        "candidates_after_novelty_filter": after_novelty,
        "candidates_after_freshness_filter": after_fresh,
        "candidates_after_evidence_filter": after_evidence,
        "candidates_after_source_caps": after_source_caps,
        "candidates_main_story_eligible": main_eligible,
        "previously_seen":             previously_seen,
        "previously_emitted":          previously_emitted,
        "suppressed_total":            total_suppressed,
        "suppressed_novelty":          suppressed_novelty,
        "suppressed_freshness":        suppressed_freshness,
        "suppressed_low_evidence":     suppressed_evidence,
        "suppressed_bad_type":         suppressed_type,
        "suppressed_source_cap":       suppressed_diversity,
        "missing_publication_date":    missing_date,
        "used_fallback_freshness_date": fallback_freshness,
        "rejected_from_main_story":    rejected_main,
        "selected_main_stories":       main_count,
        "selected_quick_hits":         quick_count,
        "selected_role_mismatch":      role_mismatch,
        "underfilled_slots_main":      underfilled_main,
        "underfilled_slots_quick_hits": underfilled_qh,
        "rejection_counts_by_reason":  dict(rejection_counts),
        "source_distribution":         dict(source_freq),
    }
