"""
Tests for selection_policy.py.

Required test classes
---------------------
1. TestRepeatSuppression        — cross-run emitted/seen suppression
2. TestSourceCapEnforcement     — per-source quota during selection
3. TestEvidenceThresholds       — low-sufficiency / no-findings gates
4. TestFreshnessRules           — missing/stale date handling
5. TestArticleTypeClassification — multi-signal type assignment

Bonus classes
-------------
6. TestTopicAssignment          — primary/secondary topic logic
7. TestNegativeScoring          — score adjustments
8. TestMainStoryEligibility     — eligibility gate combinations
9. TestDiagnosticsSummary       — build_selection_diagnostics_summary output
"""

import os
import sqlite3
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import selection_policy as SP
from selection_policy import SelectionPolicy, build_selection_diagnostics_summary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TOPIC_CATS = {
    "art_ivf": ["IVF", "ICSI", "embryo transfer", "blastocyst", "oocyte retrieval"],
    "male_factor": ["sperm", "semen", "azoospermia", "male infertility"],
    "endometriosis": ["endometriosis", "endometrial"],
    "pcos": ["PCOS", "polycystic ovary", "polycystic ovarian"],
}

_BASE_CONFIG = {
    "topic_categories": _TOPIC_CATS,
    "openai_model": "gpt-4.1",
}


def _policy(extra_cfg: dict = None) -> SelectionPolicy:
    cfg = dict(_BASE_CONFIG)
    if extra_cfg:
        cfg.update(extra_cfg)
    return SelectionPolicy(cfg)


def _article(
    title="IVF live birth study",
    source="Fertility and Sterility",
    url="https://example.com/ivf-study",
    pub_days_ago=10,
    evidence_sufficiency=0.65,
    confidence="medium",
    source_quality="abstract",
    is_fallback=False,
    what_it_found="Women in the treatment group had higher live birth rates.",
    art_type=None,
    content="",
) -> dict:
    pub_date = (
        datetime.now(tz=timezone.utc) - timedelta(days=pub_days_ago)
        if pub_days_ago is not None else None
    )
    return {
        "title": title,
        "source": source,
        "url": url,
        "published_date": pub_date,
        "content": content or f"This study examined {title.lower()}.",
        "evidence_sufficiency": evidence_sufficiency,
        "evidence_quality": {
            "confidence": confidence,
            "source_text_quality": source_quality,
            "article_type": art_type or "original_research",
            "is_fallback": is_fallback,
        },
        "summary_sections": {
            "what_it_studied": f"The effect of treatment on outcomes in {title}.",
            "what_it_found": what_it_found,
            "why_it_matters": "This may inform clinical practice.",
            "caveats": "",
        },
        "impact_score": 8.0,
        "reportability_score": 6.5,
    }


# ---------------------------------------------------------------------------
# 1. TestRepeatSuppression
# ---------------------------------------------------------------------------

class TestRepeatSuppression:

    def test_new_article_not_suppressed(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p = _policy({"novelty_control": {
                "suppress_previously_emitted": True,
                "suppress_previously_seen": False,
                "allow_repeat_after_days": 90,
                "identity_priority": ["url"],
                "update_history_on_selection": False,
            }})
            art = _article()
            result = p.check_novelty(art, db)
            assert result["status"] == "new"
            assert result["suppressed"] is False
        finally:
            os.unlink(db)

    def test_emitted_article_suppressed_within_window(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            conn = SP._open_db(db)
            art = _article(url="https://example.com/emitted-article")
            SP._upsert_article(conn, art, run_id="run_prior", emit=True)
            conn.close()

            p = _policy({"novelty_control": {
                "suppress_previously_emitted": True,
                "allow_repeat_after_days": 90,
                "identity_priority": ["url"],
            }})
            result = p.check_novelty(art, db)
            assert result["status"] == "emitted"
            assert result["suppressed"] is True
            assert "emitted" in (result["suppress_reason"] or "").lower()
        finally:
            os.unlink(db)

    def test_emitted_article_allowed_after_repeat_window(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            conn = SP._open_db(db)
            art = _article(url="https://example.com/old-emitted")
            # Manually insert with old emitted_at
            old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=100)).isoformat()
            conn.execute(
                """INSERT INTO selection_history
                   (url, fingerprint, title, source, first_seen_at, last_seen_at, emitted_at, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (art["url"], SP._content_fingerprint(art),
                 art["title"], art["source"], old_ts, old_ts, old_ts, "run_old"),
            )
            conn.commit()
            conn.close()

            p = _policy({"novelty_control": {
                "suppress_previously_emitted": True,
                "allow_repeat_after_days": 90,
                "identity_priority": ["url"],
            }})
            result = p.check_novelty(art, db)
            assert result["suppressed"] is False, (
                "Article emitted >90 days ago should not be suppressed"
            )
        finally:
            os.unlink(db)

    def test_seen_but_not_emitted_not_suppressed_by_default(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            conn = SP._open_db(db)
            art = _article(url="https://example.com/seen-only")
            SP._upsert_article(conn, art, run_id="run_prior", emit=False)
            conn.close()

            p = _policy({"novelty_control": {
                "suppress_previously_seen": False,  # default
                "identity_priority": ["url"],
            }})
            result = p.check_novelty(art, db)
            assert result["status"] == "seen"
            assert result["suppressed"] is False

        finally:
            os.unlink(db)

    def test_apply_all_marks_articles_seen(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p = _policy({"novelty_control": {
                "update_history_on_selection": True,
                "identity_priority": ["url"],
            }})
            arts = [_article(url=f"https://example.com/art{i}") for i in range(3)]
            p.apply_all(arts, db_path=db, run_id="run1")

            conn = SP._open_db(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT url FROM selection_history").fetchall()
            conn.close()
            urls = {r["url"] for r in rows}
            assert all(f"https://example.com/art{i}" in urls for i in range(3))
        finally:
            os.unlink(db)


# ---------------------------------------------------------------------------
# 2. TestSourceCapEnforcement
# ---------------------------------------------------------------------------

class TestSourceCapEnforcement:

    def test_source_cap_limits_main_stories(self):
        p = _policy({"source_diversity": {"max_main_stories_per_source": 2}})
        arts = [
            _article(title=f"FS Article {i}", source="Fertility and Sterility", url=f"https://fs.com/{i}")
            for i in range(5)
        ]
        selected = p.select_with_diversity(arts, n=5, tier="main_story")
        assert len(selected) <= 2, (
            f"Source cap of 2 should allow at most 2, got {len(selected)}"
        )

    def test_source_cap_allows_different_sources(self):
        p = _policy({"source_diversity": {"max_main_stories_per_source": 2}})
        arts = [
            _article(title="FS 1",   source="Fertility and Sterility",   url="https://fs.com/1"),
            _article(title="HR 1",   source="Human Reproduction",        url="https://hr.com/1"),
            _article(title="FS 2",   source="Fertility and Sterility",   url="https://fs.com/2"),
            _article(title="HR 2",   source="Human Reproduction",        url="https://hr.com/2"),
            _article(title="FS 3",   source="Fertility and Sterility",   url="https://fs.com/3"),
        ]
        selected = p.select_with_diversity(arts, n=5, tier="main_story")
        sources = [a["source"] for a in selected]
        fs_count = sources.count("Fertility and Sterility")
        hr_count = sources.count("Human Reproduction")
        assert fs_count <= 2
        assert hr_count <= 2
        assert len(selected) == 4  # 2 FS + 2 HR

    def test_capped_article_gets_diagnostic(self):
        p = _policy({"source_diversity": {"max_main_stories_per_source": 1}})
        arts = [
            _article(title="FS 1", source="Fertility and Sterility", url="https://fs.com/1"),
            _article(title="FS 2", source="Fertility and Sterility", url="https://fs.com/2"),
        ]
        p.select_with_diversity(arts, n=2, tier="main_story")
        capped = [a for a in arts if (a.get("_policy") or {}).get("source_diversity_status", "").startswith("capped")]
        assert len(capped) >= 1

    def test_quick_hit_cap_independent_of_main_story_cap(self):
        p = _policy({"source_diversity": {
            "max_main_stories_per_source": 1,
            "max_quick_hits_per_source": 3,
        }})
        arts = [_article(title=f"A{i}", url=f"https://x.com/{i}") for i in range(5)]
        qh = p.select_with_diversity(arts, n=5, tier="quick_hit")
        assert len(qh) == 3  # capped at 3


# ---------------------------------------------------------------------------
# 3. TestEvidenceThresholds
# ---------------------------------------------------------------------------

class TestEvidenceThresholds:

    def test_high_sufficiency_eligible_for_main(self):
        p = _policy({"evidence_thresholds": {"min_main_story_sufficiency": 0.40}})
        art = _article(evidence_sufficiency=0.75)
        result = p.check_evidence(art)
        assert result["eligible_for_main"] is True

    def test_low_sufficiency_ineligible_for_main(self):
        p = _policy({"evidence_thresholds": {"min_main_story_sufficiency": 0.40}})
        art = _article(evidence_sufficiency=0.20)
        result = p.check_evidence(art)
        assert result["eligible_for_main"] is False

    def test_below_quick_hit_floor_ineligible_for_both(self):
        p = _policy({"evidence_thresholds": {
            "min_main_story_sufficiency": 0.40,
            "min_quick_hit_sufficiency": 0.15,
        }})
        art = _article(evidence_sufficiency=0.05)
        result = p.check_evidence(art)
        assert result["eligible_for_main"]     is False
        assert result["eligible_for_quick_hit"] is False

    def test_no_findings_flagged(self):
        p = _policy({"evidence_thresholds": {"drop_if_no_findings_extracted": True}})
        art = _article(what_it_found="")
        art["summary_sections"]["what_it_found"] = ""
        result = p.check_evidence(art)
        assert result["suppressed"] is True
        assert "findings" in (result["suppress_reason"] or "").lower()

    def test_no_findings_flagged_when_drop_false_still_marks_has_findings_false(self):
        p = _policy({"evidence_thresholds": {
            "drop_if_no_findings_extracted": False,
            "downgrade_if_findings_missing": True,
        }})
        art = _article(what_it_found="")
        art["summary_sections"]["what_it_found"] = ""
        result = p.check_evidence(art)
        assert result["suppressed"] is False  # not dropped
        assert result["has_findings"] is False

    def test_main_story_ineligible_when_evidence_threshold_not_met(self):
        p = _policy({
            "evidence_thresholds": {"min_main_story_sufficiency": 0.60},
            "main_story_rules": {"min_impact_score": 0.0, "require_nontrivial_findings": False},
        })
        art = _article(evidence_sufficiency=0.30)
        art["_policy"] = {
            "article_type": "original_research",
            "evidence_status": p.check_evidence(art),
            "freshness_status": p.check_freshness(art),
        }
        eligible, reason = p.is_main_story_eligible(art)
        assert eligible is False
        assert "evidence" in reason.lower() or "sufficiency" in reason.lower()


# ---------------------------------------------------------------------------
# 4. TestFreshnessRules
# ---------------------------------------------------------------------------

class TestFreshnessRules:

    def test_fresh_article_eligible(self):
        p = _policy({"freshness_rules": {"max_age_days_main_story": 30}})
        art = _article(pub_days_ago=5)
        result = p.check_freshness(art)
        assert result["status"] == "fresh"
        assert result["eligible_for_main"] is True

    def test_old_article_ineligible_for_main_not_quick_hit(self):
        p = _policy({"freshness_rules": {
            "max_age_days_main_story": 30,
            "max_age_days_quick_hit": 90,
        }})
        art = _article(pub_days_ago=40)
        result = p.check_freshness(art)
        assert result["eligible_for_main"]     is False
        assert result["eligible_for_quick_hit"] is True

    def test_very_old_article_suppressed(self):
        p = _policy({"freshness_rules": {
            "max_age_days_main_story": 30,
            "max_age_days_quick_hit": 60,
        }})
        art = _article(pub_days_ago=100)
        result = p.check_freshness(art)
        assert result["suppressed"] is True
        assert "stale" in result["status"]

    def test_missing_date_with_require_false_not_suppressed(self):
        p = _policy({"freshness_rules": {
            "require_published_at": False,
            "fallback_to_feed_timestamp_if_missing": False,
        }})
        art = _article(pub_days_ago=None)
        result = p.check_freshness(art)
        assert result["suppressed"] is False
        assert result["status"] == "missing_freshness_date"

    def test_missing_date_with_require_true_suppressed(self):
        p = _policy({"freshness_rules": {
            "require_published_at": True,
            "fallback_to_feed_timestamp_if_missing": False,
        }})
        art = _article(pub_days_ago=None)
        result = p.check_freshness(art)
        assert result["suppressed"] is True

    def test_fallback_to_feed_timestamp(self):
        p = _policy({"freshness_rules": {
            "require_published_at": True,
            "fallback_to_feed_timestamp_if_missing": True,
            "max_age_days_main_story": 30,
        }})
        art = _article(pub_days_ago=None)
        # Provide a recent feed_timestamp
        art["feed_timestamp"] = datetime.now(tz=timezone.utc) - timedelta(days=3)
        result = p.check_freshness(art)
        assert result["date_source"] == "feed_timestamp"
        assert result["suppressed"] is False

    def test_fallback_to_updated_date_marks_fallback(self):
        p = _policy({"freshness_rules": {
            "require_published_at_for_main_story": True,
            "fallback_date_sources": ["updated_date"],
        }})
        art = _article(pub_days_ago=None)
        art["updated_date"] = datetime.now(tz=timezone.utc) - timedelta(days=2)
        result = p.check_freshness(art)
        assert result["date_source"] == "updated_date"
        assert result["used_fallback_date"] is True
        assert result["freshness_confidence"] in ("medium", "low")
        assert result["eligible_for_main"] is True


# ---------------------------------------------------------------------------
# 5. TestArticleTypeClassification
# ---------------------------------------------------------------------------

class TestArticleTypeClassification:

    def test_systematic_review_in_title(self):
        p = _policy({})
        art = _article(title="A systematic review of IVF outcomes")
        assert p.classify_article_type(art) == "systematic_review"

    def test_meta_analysis_in_title(self):
        p = _policy({})
        art = _article(title="Sperm DNA fragmentation: a meta-analysis")
        assert p.classify_article_type(art) == "meta_analysis"

    def test_rct_in_title(self):
        p = _policy({})
        art = _article(title="A randomized controlled trial of melatonin in IVF")
        assert p.classify_article_type(art) == "rct"

    def test_editorial_in_title(self):
        p = _policy({})
        art = _article(title="Editorial: new directions in reproductive medicine")
        assert p.classify_article_type(art) == "editorial"

    def test_erratum_classified_and_excluded(self):
        p = _policy({})
        art = _article(title="Corrigendum to 'Effect of DHEA on IVF outcomes'")
        art_type = p.classify_article_type(art)
        assert art_type in ("erratum", "correction")

    def test_retraction_classified(self):
        p = _policy({})
        art = _article(title="Retraction: prior study on oocyte quality")
        assert p.classify_article_type(art) == "retraction"

    def test_exclude_types_suppressed_in_apply_all(self):
        p = _policy({
            "article_type_rules": {"exclude_types": ["retraction", "erratum"]},
        })
        arts = [
            _article(title="Normal IVF study"),
            _article(title="Corrigendum to prior paper", url="https://x.com/corr"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all(arts, db_path=db)
            suppressed = [a for a in arts if a.get("_policy", {}).get("suppressed")]
            assert len(suppressed) >= 1
            assert any("erratum" in (a["_policy"].get("suppressed_reason") or "")
                       or "correction" in (a["_policy"].get("suppressed_reason") or "")
                       for a in suppressed)
        finally:
            os.unlink(db)

    def test_explicit_pattern_overrides_builtin(self):
        p = _policy({
            "article_type_rules": {
                "explicit_patterns": {"best practice": "guideline"},
                "default_type": "unknown",
            }
        })
        art = _article(title="Best practice recommendations for IUI")
        assert p.classify_article_type(art) == "guideline"

    def test_fallback_to_summarizer_type(self):
        p = _policy({})
        art = _article(title="Some ambiguous title")
        art["evidence_quality"]["article_type"] = "review"
        result = p.classify_article_type(art)
        assert result == "review"

    def test_demoted_type_not_main_story(self):
        p = _policy({
            "article_type_rules": {
                "demote_types_for_main_story": ["editorial", "commentary"],
            },
            "main_story_rules": {"min_impact_score": 0.0},
        })
        art = _article(title="Commentary on recent IVF advances")
        art["_policy"] = {
            "article_type": "editorial",
            "evidence_status": {"eligible_for_main": True, "has_findings": True, "sufficiency": 0.7},
            "freshness_status": {"eligible_for_main": True, "suppressed": False},
        }
        eligible, reason = p.is_main_story_eligible(art)
        assert eligible is False
        assert "editorial" in reason.lower()


# ---------------------------------------------------------------------------
# 6. TestTopicAssignment
# ---------------------------------------------------------------------------

class TestTopicAssignment:

    def test_primary_topic_assigned_from_title(self):
        p = _policy({})
        art = _article(title="IVF outcomes in women with endometriosis")
        primary, secondary = p.assign_topics(art)
        assert primary in ("art_ivf", "endometriosis")

    def test_secondary_topics_limited_by_max(self):
        p = _policy({"topic_assignment": {"max_topics_per_article": 2}})
        art = _article(
            title="IVF and sperm quality in PCOS",
            content="Semen analysis, ICSI, polycystic ovary syndrome, blastocyst culture.",
        )
        primary, secondary = p.assign_topics(art)
        assert len(secondary) <= 1  # max 2 total → at most 1 secondary

    def test_no_match_returns_existing_topic(self):
        p = _policy({})
        art = _article(title="Unrelated article about general medicine")
        art["topic"] = "general"
        primary, secondary = p.assign_topics(art)
        assert primary == "general"

    def test_min_keyword_hits_filters_weak_matches(self):
        p = _policy({"topic_assignment": {"min_keyword_hits_for_topic": 3}})
        art = _article(title="IVF outcomes", content="One mention of IVF only.")
        # Only 1 keyword hit — below the min_keyword_hits_for_topic=3 threshold
        art["topic"] = "fallback"
        primary, secondary = p.assign_topics(art)
        assert primary == "fallback"  # no topic meets threshold → return existing


# ---------------------------------------------------------------------------
# 7. TestNegativeScoring
# ---------------------------------------------------------------------------

class TestNegativeScoring:

    def test_editorial_type_penalised(self):
        p = _policy({
            "negative_scoring": {
                "article_types": {"editorial": -5.0},
                "missing_metadata_penalties": {},
            }
        })
        art = _article()
        art["_policy"] = {"article_type": "editorial", "evidence_status": {"has_findings": True, "sufficiency": 0.7}}
        adj = p.negative_score_adjustment(art)
        assert adj == -5.0

    def test_missing_published_date_penalised(self):
        p = _policy({
            "negative_scoring": {
                "article_types": {},
                "missing_metadata_penalties": {"missing_published_date": -2.0},
            }
        })
        art = _article(pub_days_ago=None)
        art["_policy"] = {"article_type": "unknown", "evidence_status": {"has_findings": True, "sufficiency": 0.7}}
        adj = p.negative_score_adjustment(art)
        assert adj == -2.0

    def test_no_penalty_for_clean_article(self):
        p = _policy({
            "negative_scoring": {
                "article_types": {"editorial": -2.0},
                "missing_metadata_penalties": {"missing_published_date": -1.0},
            }
        })
        art = _article()
        art["_policy"] = {"article_type": "original_research", "evidence_status": {"has_findings": True, "sufficiency": 0.7}}
        adj = p.negative_score_adjustment(art)
        assert adj == 0.0


# ---------------------------------------------------------------------------
# 8. TestMainStoryEligibility
# ---------------------------------------------------------------------------

class TestMainStoryEligibility:

    def _ready_article(self, **kwargs):
        art = _article(**kwargs)
        # Use permissive main_story_rules so this helper doesn't fail on ancillary gates
        p = _policy({
            "main_story_rules": {
                "min_findings_length": 0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
            },
            "freshness_rules": {"require_published_at_for_main_story": False},
        })
        art["_policy"] = {
            "article_type": "original_research",
            "evidence_status": p.check_evidence(art),
            "freshness_status": p.check_freshness(art),
        }
        return art, p

    def test_good_article_is_eligible(self):
        art, p = self._ready_article(evidence_sufficiency=0.70)
        eligible, reason = p.is_main_story_eligible(art)
        assert eligible is True, f"Expected eligible, got reason: {reason}"

    def test_below_impact_threshold_ineligible(self):
        p = _policy({"main_story_rules": {"min_impact_score": 10.0}})
        art = _article(evidence_sufficiency=0.80)
        art["impact_score"] = 3.0
        art["_policy"] = {
            "article_type": "original_research",
            "evidence_status": p.check_evidence(art),
            "freshness_status": p.check_freshness(art),
        }
        eligible, reason = p.is_main_story_eligible(art)
        assert eligible is False
        assert "impact_score" in reason

    def test_source_family_cap_blocks_excess(self):
        p = _policy({
            "main_story_rules": {
                "max_main_stories_same_source_family": 1,
                "min_findings_length": 0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
            },
            "freshness_rules": {"require_published_at_for_main_story": False},
        })
        existing = [_article(source="Fertility and Sterility", url="https://fs.com/1")]
        new_art = _article(source="Fertility and Sterility — Articles in Press",
                           url="https://fs.com/2", evidence_sufficiency=0.80)
        new_art["impact_score"] = 15.0
        new_art["_policy"] = {
            "article_type": "original_research",
            "evidence_status": p.check_evidence(new_art),
            "freshness_status": p.check_freshness(new_art),
        }
        eligible, reason = p.is_main_story_eligible(new_art, current_main_stories=existing)
        assert eligible is False
        assert "source family" in reason.lower()


# ---------------------------------------------------------------------------
# 9. TestDiagnosticsSummary
# ---------------------------------------------------------------------------

class TestDiagnosticsSummary:

    def test_summary_counts_suppressed(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p = _policy({"article_type_rules": {"exclude_types": ["retraction"]}})
            arts = [
                _article(title="Good article",   url="https://x.com/1"),
                _article(title="Retracted paper", url="https://x.com/2",
                         content="retraction notice of a prior paper"),
            ]
            arts[1]["title"] = "Retraction: prior paper"
            p.apply_all(arts, db_path=db)
            summary = build_selection_diagnostics_summary(arts, selected=arts[:1])
            assert summary["candidate_count"] == 2
            assert summary["suppressed_total"] >= 1
        finally:
            os.unlink(db)

    def test_summary_tracks_source_distribution(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p = _policy({})
            arts = [
                _article(source="Fertility and Sterility", url="https://fs.com/1"),
                _article(source="Human Reproduction",      url="https://hr.com/1"),
            ]
            for a in arts:
                a.setdefault("_policy", {})["selected_as"] = "main_story"
            summary = build_selection_diagnostics_summary(arts, selected=arts)
            assert "Fertility and Sterility" in summary["source_distribution"]
            assert "Human Reproduction"      in summary["source_distribution"]
        finally:
            os.unlink(db)

    def test_summary_has_all_required_keys(self):
        summary = build_selection_diagnostics_summary([], [])
        required = [
            "issue_state", "candidate_count", "previously_seen", "previously_emitted",
            "suppressed_total", "suppressed_novelty", "suppressed_freshness",
            "suppressed_low_evidence", "suppressed_bad_type", "suppressed_source_cap",
            "coverage_candidates", "publishable_candidates",
            "candidates_after_novelty_filter", "candidates_after_freshness_filter",
            "candidates_after_evidence_filter", "candidates_after_source_caps",
            "missing_publication_date", "selected_main_stories", "selected_quick_hits",
            "underfilled_slots_main", "underfilled_slots_quick_hits",
            "rejection_counts_by_reason", "source_distribution",
        ]
        for key in required:
            assert key in summary, f"Missing key in diagnostics summary: {key}"

    def test_summary_selected_counts_match_selected_roles(self):
        arts = [
            _article(url="https://x.com/1"),
            _article(url="https://x.com/2"),
            _article(url="https://x.com/3"),
        ]
        arts[0].setdefault("_policy", {})["selected_as"] = "main_story"
        arts[1].setdefault("_policy", {})["selected_as"] = "quick_hit"
        # third left unselected
        summary = build_selection_diagnostics_summary(arts, selected=arts[:2])
        assert summary["selected_main_stories"] == 1
        assert summary["selected_quick_hits"] == 1
        assert summary["selected_role_mismatch"] is False


# ---------------------------------------------------------------------------
# 10. TestHardGates — gate-first selection behavior
# ---------------------------------------------------------------------------

class TestHardGates:
    """New tests covering the gate-first selection requirements."""

    # ── Test 1: underfill when quality is low ────────────────────────────────

    def test_main_stories_underfilled_when_all_evidence_weak(self):
        """Articles with heuristic evidence below threshold must not be promoted."""
        p = _policy({
            "evidence_thresholds": {"min_main_story_sufficiency": 0.55},
            "freshness_rules": {"require_published_at_for_main_story": False},
            "main_story_rules": {
                "min_impact_score": 0.0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
                "require_nontrivial_findings": False,
                "min_findings_length": 0,
            },
        })
        # Short content → heuristic gives ~0.30, below 0.55 threshold
        arts = [
            _article(title=f"Short study {i}", url=f"https://x.com/{i}",
                     content="Short abstract.", evidence_sufficiency=0.0)
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all(arts, db_path=db)
            main = [a for a in arts if p.is_main_story_eligible(a)[0]]
            assert len(main) == 0, "All weak articles should fail main-story gate"
        finally:
            os.unlink(db)

    # ── Test 2: abstract-only low-sufficiency rejected from main stories ─────

    def test_abstract_only_below_threshold_rejected_from_main(self):
        """abstract-only article with heuristic sufficiency < threshold fails gate."""
        p = _policy({
            "evidence_thresholds": {"min_main_story_sufficiency": 0.55},
            "freshness_rules": {"require_published_at_for_main_story": False},
            "main_story_rules": {
                "min_impact_score": 0.0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
                "require_nontrivial_findings": False,
                "min_findings_length": 0,
            },
        })
        # 500-char content → heuristic short_blurb → ~0.30 < 0.55
        art = _article(
            content="A" * 500,
            evidence_sufficiency=0.0,  # Pre-LLM: not yet set
        )
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all([art], db_path=db)
            eligible, reason = p.is_main_story_eligible(art)
            assert eligible is False
            assert "sufficiency" in reason.lower()
        finally:
            os.unlink(db)

    # ── Test 3: missing publication date blocks main story ───────────────────

    def test_missing_date_blocks_main_story_when_required(self):
        """Missing publication date must block main story when configured."""
        p = _policy({
            "freshness_rules": {
                "require_published_at_for_main_story": True,
                "fallback_date_sources": [],
            },
            "evidence_thresholds": {"min_main_story_sufficiency": 0.0},
            "main_story_rules": {
                "min_impact_score": 0.0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
                "require_nontrivial_findings": False,
                "min_findings_length": 0,
            },
        })
        art = _article(pub_days_ago=None, evidence_sufficiency=0.80)
        art["published_date"] = None  # ensure no date
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all([art], db_path=db)
            eligible, reason = p.is_main_story_eligible(art)
            assert eligible is False
            assert "missing_date" in reason.lower() or "freshness" in reason.lower()
        finally:
            os.unlink(db)

    # ── Test 4: source cap prevents single-source domination ─────────────────

    def test_source_cap_prevents_single_source_domination(self):
        """max_main_stories_per_source=1 must cap one source to one main story."""
        p = _policy({
            "source_diversity": {"max_main_stories_per_source": 1},
            "freshness_rules": {"require_published_at_for_main_story": False},
            "evidence_thresholds": {"min_main_story_sufficiency": 0.0},
            "main_story_rules": {
                "min_impact_score": 0.0,
                "max_main_stories_same_source_family": 1,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [],
                "require_nontrivial_findings": False,
                "min_findings_length": 0,
            },
        })
        arts = [
            _article(title=f"F&S study {i}",
                     source="Fertility and Sterility",
                     url=f"https://fs.com/{i}",
                     evidence_sufficiency=0.80)
            for i in range(4)
        ]
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all(arts, db_path=db)
            selected = []
            for art in arts:
                ok, _ = p.is_main_story_eligible(art, current_main_stories=selected)
                if ok:
                    selected.append(art)

            sources = [_article_source(a) for a in selected]
            from collections import Counter
            counts = Counter(sources)
            assert max(counts.values()) <= 1, (
                f"Source cap violated: {dict(counts)}"
            )
        finally:
            os.unlink(db)

    # ── Test 5: systematic review / meta-analysis classified correctly ────────

    def test_systematic_review_meta_analysis_classified_correctly(self):
        """Title containing 'systematic review and meta-analysis' must not be original_research."""
        p = _policy({})
        art = _article(
            title="Impact of Male Genital Tract Infections on Semen Quality: "
                  "A Systematic Review and Meta-Analysis",
        )
        art_type = p.classify_article_type(art)
        assert art_type in ("systematic_review", "meta_analysis"), (
            f"Expected systematic_review or meta_analysis, got '{art_type}'"
        )

    # ── Test 6: no-publishable-items state when all fail ─────────────────────

    def test_no_publishable_items_state_when_all_fail(self):
        """compute_issue_state must return no_publishable_items when both lists empty."""
        from selection_policy import compute_issue_state
        state = compute_issue_state([], [], max_main=5, max_qh=10)
        assert state == "no_publishable_items"

    def test_quick_hits_only_state_when_no_main(self):
        """compute_issue_state must return quick_hits_only when main is empty but qh non-empty."""
        from selection_policy import compute_issue_state
        dummy_qh = [_article()]
        state = compute_issue_state([], dummy_qh, max_main=5, max_qh=10)
        assert state == "quick_hits_only"

    def test_underfilled_issue_state_partial_main(self):
        """compute_issue_state must return underfilled_issue when main < max."""
        from selection_policy import compute_issue_state
        one_main = [_article()]
        five_qh  = [_article(url=f"https://x.com/{i}") for i in range(5)]
        state = compute_issue_state(one_main, five_qh, max_main=5, max_qh=5)
        assert state == "underfilled_issue"

    def test_disallow_if_findings_contain_blocks_main_story(self):
        """Articles whose findings text contains a banned phrase must fail main-story gate."""
        p = _policy({
            "freshness_rules": {"require_published_at_for_main_story": False},
            "evidence_thresholds": {"min_main_story_sufficiency": 0.0},
            "main_story_rules": {
                "min_impact_score": 0.0,
                "eligible_article_types": [],
                "ineligible_article_types": [],
                "disallow_if_findings_contain": [
                    "Detailed findings were not available",
                ],
                "require_nontrivial_findings": False,
                "min_findings_length": 0,
            },
        })
        art = _article(
            what_it_found="Detailed findings were not available in the retrieved text.",
            evidence_sufficiency=0.80,
        )
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            p.apply_all([art], db_path=db)
            eligible, reason = p.is_main_story_eligible(art)
            assert eligible is False
            assert "disqualifying phrase" in reason.lower()
        finally:
            os.unlink(db)


def _article_source(art: dict) -> str:
    """Normalized source label for comparison."""
    import selection_policy as SP
    return SP._normalize_source(art.get("source", ""))
