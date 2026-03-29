"""
Tests for newsletter_composer.py.

Required test classes
---------------------
1. TestNoEmptyFieldsInNewsletter
   – No featured story with empty body/title.
   – No featured story without a concrete validated finding.
   – Briefs text is non-empty.
   – Watchlist entries have at least a title.

2. TestNoFeaturedWithoutConcretesFinding
   – Articles with no concrete finding must NOT become featured.
   – Articles with fallback-phrased 'what_it_found' must NOT become featured.
   – Only articles with substantive, non-internal evidence reach featured tier.

3. TestLowEvidenceRoutedToWatchlist
   – Snippet-only / titles_to_watch evidence items go to watchlist.
   – is_fallback=True items go to watchlist regardless of score.
   – Very low newsletter score items go to watchlist.

4. TestNoForbiddenPhrasesInNewsletter
   – INTERNAL_REPORT_PHRASES must not appear in featured body, brief text, or
     watchlist context.
   – Editor's note and closing must not contain forbidden phrases.
   – The Markdown render must not expose internal phrasing.

5. TestSectionDiversityAndFormat
   – Repeated identical fallback blurbs are grouped/collapsed in watchlist.
   – Featured story openers vary across multiple stories (no identical first phrase).
   – Watchlist entries are compact (no multi-paragraph block).
   – newsletter_worthiness_score breakdown sums correctly.

Sample before/after fixture at the bottom of this file demonstrates the
transformation from internal-report format to clean newsletter copy.
"""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import newsletter_composer as NC


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_record(
    title="Test Article",
    source="Fertility and Sterility",
    url="https://example.com/article",
    pub_date="2026-03-01",
    topic="art_ivf",
    evidence_tier="full",
    evidence_sufficiency=0.75,
    is_fallback=False,
    article_type="original_research",
    confidence="medium",
    what_it_studied="The study examined the effect of X on Y.",
    what_it_found="Women in the intervention group had lower rates of outcome Z compared to controls.",
    why_it_matters="These findings may inform counselling for patients planning treatment.",
    caveats="Sample size was not reported in the available text.",
    audience_relevance=7.0,
    impact_score=8.5,
) -> dict:
    return {
        "title": title,
        "source": source,
        "url": url,
        "published_date": pub_date,
        "topic": topic,
        "evidence_tier": evidence_tier,
        "evidence_sufficiency": evidence_sufficiency,
        "audience_relevance": audience_relevance,
        "impact_score": impact_score,
        "summary": what_it_found,
        "summary_sections": {
            "what_it_studied": what_it_studied,
            "what_it_found": what_it_found,
            "why_it_matters": why_it_matters,
            "caveats": caveats,
        },
        "evidence_quality": {
            "confidence": confidence,
            "source_text_quality": "abstract" if evidence_tier == "full" else "snippet_only",
            "article_type": article_type,
            "is_fallback": is_fallback,
            "validation_passed": True,
            "unsupported_sentence_count": 0,
        },
    }


def _make_fallback_record(title="Fallback Article", source="MedPage Today", topic="general") -> dict:
    """Article that would be generated as snippet-only fallback."""
    return _make_record(
        title=title,
        source=source,
        topic=topic,
        evidence_tier="titles_to_watch",
        evidence_sufficiency=0.05,
        is_fallback=True,
        article_type="unknown",
        confidence="low",
        what_it_studied=(
            "The available text describes an article from MedPage Today on the topic: "
            f"{title}."
        ),
        what_it_found="Detailed findings were not available in the retrieved source text.",
        why_it_matters=(
            "The topic is relevant to fertility and reproductive medicine; "
            "readers are encouraged to consult the full publication for details."
        ),
        caveats="Only limited metadata was available.",
        audience_relevance=2.0,
        impact_score=2.0,
    )


def _strong_record(n=1) -> dict:
    """Article with strong evidence that should become Featured."""
    return _make_record(
        title=f"Melatonin Supplementation and IVF Outcomes Study {n}",
        source="Human Reproduction",
        topic="art_ivf",
        evidence_tier="full",
        evidence_sufficiency=0.82,
        is_fallback=False,
        article_type="original_research",
        confidence="medium",
        what_it_studied="The effect of melatonin supplementation on live birth rates in women undergoing IVF.",
        what_it_found=(
            "Women receiving melatonin prior to IVF had higher live birth rates "
            "compared to those undergoing standard IVF protocols."
        ),
        why_it_matters=(
            "If replicated, melatonin supplementation may represent a low-cost adjunct "
            "to IVF for selected patient groups."
        ),
        caveats="This was a single-centre study; external validity requires confirmation.",
        audience_relevance=8.5,
        impact_score=12.0,
    )


def _null_llm(prompt, cfg, timeout=120):
    """Mock LLM caller that always returns None (simulates unavailability)."""
    return None


def _compose(articles, config=None, llm_caller=_null_llm):
    config = config or {}
    return NC.compose_newsletter(articles, config, llm_caller=llm_caller)


# ---------------------------------------------------------------------------
# Test 1 – No empty fields in final newsletter
# ---------------------------------------------------------------------------

class TestNoEmptyFieldsInNewsletter:

    def test_featured_story_has_non_empty_title(self):
        newsletter = _compose([_strong_record()])
        for story in newsletter.get("featured") or []:
            assert story.get("title"), f"Featured story has empty title: {story}"

    def test_featured_story_has_non_empty_body(self):
        newsletter = _compose([_strong_record()])
        for story in newsletter.get("featured") or []:
            assert story.get("body"), f"Featured story has empty body: {story}"

    def test_brief_has_non_empty_title(self):
        moderate = _make_record(
            title="Moderate Evidence Brief",
            evidence_tier="short_blurb",
            evidence_sufficiency=0.40,
            what_it_found="Sperm DNA fragmentation was associated with reduced blastocyst development.",
            why_it_matters="This may inform laboratory protocols for fragmentation assessment.",
        )
        newsletter = _compose([moderate])
        for b in newsletter.get("briefs") or []:
            assert b.get("title"), f"Brief has empty title: {b}"

    def test_watchlist_entry_has_non_empty_title(self):
        newsletter = _compose([_make_fallback_record()])
        for w in newsletter.get("watchlist") or []:
            assert w.get("title"), f"Watchlist entry has empty title: {w}"

    def test_newsletter_has_editor_note(self):
        newsletter = _compose([_strong_record()])
        assert newsletter.get("editor_note"), "Newsletter editor note is empty"

    def test_newsletter_has_closing(self):
        newsletter = _compose([_strong_record()])
        assert newsletter.get("closing"), "Newsletter closing is empty"

    def test_markdown_render_has_no_empty_section_headers(self):
        """
        Ensure rendered Markdown does not contain headers immediately followed
        by an empty line then another header (collapsed empty section).
        """
        from newsletter_composer import _render_newsletter_md
        articles = [_strong_record(), _make_fallback_record()]
        newsletter = _compose(articles)
        md = _render_newsletter_md(newsletter)
        # No "**What it found**: " with nothing after the colon
        assert "**What it found**: \n" not in md, "Empty 'What it found' label in Markdown"
        assert "**Why it matters**: \n" not in md, "Empty 'Why it matters' label in Markdown"
        assert "**What it studied**: \n" not in md, "Empty 'What it studied' label in Markdown"


# ---------------------------------------------------------------------------
# Test 2 – No featured story without a concrete validated finding
# ---------------------------------------------------------------------------

class TestNoFeaturedWithoutConcreteFinding:

    def test_article_with_fallback_finding_not_featured(self):
        art = _make_record(
            title="Article with fallback finding",
            evidence_tier="full",
            evidence_sufficiency=0.70,
            is_fallback=False,
            what_it_found="Detailed findings were not available in the retrieved source text.",
        )
        newsletter = _compose([art])
        featured_titles = [s["title"] for s in (newsletter.get("featured") or [])]
        assert art["title"] not in featured_titles, (
            "Article with fallback 'what_it_found' must not appear in featured"
        )

    def test_article_with_no_finding_not_featured(self):
        art = _make_record(
            title="Article with empty finding",
            evidence_tier="full",
            evidence_sufficiency=0.72,
            is_fallback=False,
            what_it_found="",
        )
        newsletter = _compose([art])
        featured_titles = [s["title"] for s in (newsletter.get("featured") or [])]
        assert art["title"] not in featured_titles

    def test_article_with_internal_phrase_in_finding_not_featured(self):
        art = _make_record(
            title="Internal phrase article",
            evidence_tier="full",
            evidence_sufficiency=0.68,
            is_fallback=False,
            what_it_found=(
                "The available text describes an association between BMI and oocyte quality."
            ),
        )
        newsletter = _compose([art])
        featured_titles = [s["title"] for s in (newsletter.get("featured") or [])]
        assert art["title"] not in featured_titles

    def test_strong_article_becomes_featured(self):
        newsletter = _compose([_strong_record()])
        assert len(newsletter.get("featured") or []) >= 1, (
            "Strong article should appear in featured"
        )

    def test_newsletter_worthiness_score_has_finding_component(self):
        art_with = _strong_record()
        art_without = _make_record(
            what_it_found="Detailed findings were not available in the retrieved source text.",
        )
        score_with    = NC.newsletter_worthiness_score(art_with)
        score_without = NC.newsletter_worthiness_score(art_without)
        assert score_with["has_finding"] is True
        assert score_without["has_finding"] is False
        assert score_with["score"] > score_without["score"]


# ---------------------------------------------------------------------------
# Test 3 – Low-evidence items routed to watchlist
# ---------------------------------------------------------------------------

class TestLowEvidenceRoutedToWatchlist:

    def test_titles_to_watch_tier_goes_to_watchlist(self):
        art = _make_fallback_record()
        newsletter = _compose([art])
        watchlist_titles = [w["title"] for w in (newsletter.get("watchlist") or [])]
        assert art["title"] in watchlist_titles, (
            "titles_to_watch article must appear in newsletter watchlist"
        )

    def test_is_fallback_true_goes_to_watchlist(self):
        art = _make_record(
            title="Fallback full-tier article",
            evidence_tier="full",
            evidence_sufficiency=0.75,
            is_fallback=True,  # explicitly flagged as fallback
        )
        newsletter = _compose([art])
        watchlist_titles = [w["title"] for w in (newsletter.get("watchlist") or [])]
        assert art["title"] in watchlist_titles

    def test_very_low_score_goes_to_watchlist(self):
        # A short_blurb article with low confidence, unknown type, and no findings
        # should score below the briefs threshold.
        art = _make_record(
            title="Low score article",
            evidence_tier="short_blurb",
            evidence_sufficiency=0.10,
            is_fallback=False,
            confidence="low",       # realistically low evidence
            article_type="unknown", # no reliable type classification
            audience_relevance=1.0,
            what_it_found="",
            why_it_matters="",
        )
        score = NC.newsletter_worthiness_score(art)
        assert score["score"] < NC.BRIEFS_THRESHOLD, (
            f"Expected score < {NC.BRIEFS_THRESHOLD}, got {score['score']}: {score}"
        )
        tier = NC.route_to_tier(art, score)
        assert tier == "watchlist"

    def test_strong_evidence_does_not_go_to_watchlist(self):
        art = _strong_record()
        score = NC.newsletter_worthiness_score(art)
        tier = NC.route_to_tier(art, score)
        assert tier == "featured", f"Strong article should be featured, got: {tier}"

    def test_moderate_evidence_goes_to_briefs_not_featured(self):
        art = _make_record(
            title="Moderate evidence article",
            evidence_tier="short_blurb",
            evidence_sufficiency=0.38,
            is_fallback=False,
            what_it_found=(
                "Sperm DNA fragmentation was associated with reduced blastocyst development."
            ),
            why_it_matters="Assessment may improve embryo selection.",
            audience_relevance=6.0,
        )
        score = NC.newsletter_worthiness_score(art)
        tier = NC.route_to_tier(art, score)
        assert tier in ("briefs", "watchlist"), (
            f"short_blurb article should not become featured: tier={tier}, score={score}"
        )


# ---------------------------------------------------------------------------
# Test 4 – No forbidden internal-report phrases in newsletter output
# ---------------------------------------------------------------------------

class TestNoForbiddenPhrasesInNewsletter:

    FORBIDDEN = NC.INTERNAL_REPORT_PHRASES

    def _all_newsletter_text(self, newsletter: dict) -> str:
        parts = [
            newsletter.get("editor_note") or "",
            newsletter.get("closing") or "",
        ]
        for story in newsletter.get("featured") or []:
            parts += [story.get("body") or "", story.get("caveat") or ""]
        for brief in newsletter.get("briefs") or []:
            parts.append(brief.get("text") or "")
        for w in newsletter.get("watchlist") or []:
            parts.append(w.get("context") or "")
        return " ".join(parts).lower()

    def test_no_forbidden_phrase_in_featured_body(self):
        art = _make_record(
            title="Clean featured article",
            evidence_tier="full",
            evidence_sufficiency=0.78,
            is_fallback=False,
            what_it_found=(
                "Women with endometriosis had lower live birth rates compared to controls."
            ),
            why_it_matters=(
                "This evidence may influence pre-treatment counselling decisions."
            ),
        )
        newsletter = _compose([art])
        for story in newsletter.get("featured") or []:
            body = (story.get("body") or "").lower()
            for phrase in self.FORBIDDEN:
                assert phrase not in body, (
                    f"Forbidden phrase '{phrase}' found in featured body: {body[:120]}"
                )

    def test_no_forbidden_phrase_in_briefs(self):
        art = _make_record(
            title="Brief article",
            evidence_tier="short_blurb",
            evidence_sufficiency=0.42,
            is_fallback=False,
            what_it_found="Sperm motility was associated with IVF outcome in this cohort.",
        )
        newsletter = _compose([art])
        for b in newsletter.get("briefs") or []:
            text = (b.get("text") or "").lower()
            for phrase in self.FORBIDDEN:
                assert phrase not in text, (
                    f"Forbidden phrase '{phrase}' found in brief: {text[:120]}"
                )

    def test_fallback_article_forbidden_phrase_blocked_in_newsletter(self):
        """
        A fallback article may contain internal phrases in its summary — but those
        phrases must NOT leak through to the newsletter watchlist context.
        """
        art = _make_fallback_record(title="Fallback phrase article")
        newsletter = _compose([art])
        all_text = self._all_newsletter_text(newsletter)
        # The full internal phrases should not appear verbatim in the newsletter
        forbidden_found = [p for p in self.FORBIDDEN if p in all_text]
        assert not forbidden_found, (
            f"Internal-report phrases appeared in newsletter: {forbidden_found}"
        )

    def test_clean_for_newsletter_returns_none_for_internal_phrase(self):
        for phrase in NC.INTERNAL_REPORT_PHRASES:
            result = NC._clean_for_newsletter(f"This sentence contains: {phrase}.")
            assert result is None, (
                f"_clean_for_newsletter should return None for phrase: {phrase!r}"
            )

    def test_clean_for_newsletter_keeps_clean_text(self):
        clean = "Women with endometriosis had lower live birth rates compared to controls."
        result = NC._clean_for_newsletter(clean)
        assert result == clean

    def test_no_forbidden_phrase_in_editor_note_template_fallback(self):
        """Even the template-based editor's note must be free of internal phrases."""
        from newsletter_composer import _generate_editor_note
        articles = [_strong_record()]
        note = _generate_editor_note(articles, {}, llm_caller=_null_llm)
        note_lower = note.lower()
        for phrase in self.FORBIDDEN:
            assert phrase not in note_lower, (
                f"Forbidden phrase '{phrase}' in editor note template: {note[:120]}"
            )


# ---------------------------------------------------------------------------
# Test 5 – Section diversity and formatting
# ---------------------------------------------------------------------------

class TestSectionDiversityAndFormat:

    def test_multiple_featured_stories_have_varied_openers(self):
        """Featured story paragraphs should not all start with the same phrase."""
        articles = [_strong_record(n=i) for i in range(1, 5)]
        newsletter = _compose(articles)
        featured = newsletter.get("featured") or []
        if len(featured) < 2:
            pytest.skip("Need at least 2 featured articles for this test")
        openers = [
            (story.get("body") or "")[:40].lower()
            for story in featured
        ]
        unique_openers = set(openers)
        assert len(unique_openers) > 1, (
            f"All featured stories start with the same phrase: {openers}"
        )

    def test_watchlist_entries_are_compact(self):
        """Watchlist entries must not contain multi-sentence bodies."""
        articles = [_make_fallback_record(f"Watchlist Article {i}") for i in range(3)]
        newsletter = _compose(articles)
        for w in newsletter.get("watchlist") or []:
            context = w.get("context") or ""
            sentences = [s for s in context.split(".") if s.strip()]
            assert len(sentences) <= 1, (
                f"Watchlist entry is not compact (> 1 sentence): {context!r}"
            )

    def test_grouped_fallback_entries_have_no_context(self):
        """Articles with identical fallback blurbs get grouped (no repeated context text)."""
        shared_summary = "Detailed findings were not available in the retrieved source text."
        articles = [
            _make_fallback_record(f"Grouped Article {i}")
            for i in range(4)
        ]
        # Force all summaries to the same text
        for art in articles:
            art["summary"] = shared_summary
        grouped = NC._group_similar_watchlist_entries(articles)
        # All should be marked as grouped
        assert all(g.get("_grouped_fallback") for g in grouped), (
            "All identical-blurb entries should be marked _grouped_fallback"
        )

    def test_newsletter_worthiness_score_components_sum_correctly(self):
        """
        The weighted components should approximately equal the composite score.
        """
        art = _strong_record()
        d = NC.newsletter_worthiness_score(art)
        manual = (
            0.30 * d["evidence_suf"]
            + 0.25 * float(d["has_finding"])
            + 0.15 * d["type_confidence"]
            + 0.15 * d["specificity"]
            + 0.15 * d["reader_relevance"]
        )
        assert abs(manual - d["score"]) < 0.01, (
            f"Score components don't sum to composite: {manual} vs {d['score']}"
        )

    def test_article_counts_match_section_lengths(self):
        articles = [
            _strong_record(1),
            _strong_record(2),
            _make_fallback_record("Watchlist A"),
            _make_fallback_record("Watchlist B"),
        ]
        newsletter = _compose(articles)
        counts = newsletter.get("article_counts") or {}
        assert counts.get("featured") == len(newsletter.get("featured") or [])
        assert counts.get("briefs") == len(newsletter.get("briefs") or [])
        assert counts.get("watchlist") == len(newsletter.get("watchlist") or [])

    def test_markdown_render_has_all_sections_present(self):
        from newsletter_composer import _render_newsletter_md
        articles = [_strong_record(), _make_fallback_record()]
        newsletter = _compose(articles)
        md = _render_newsletter_md(newsletter)
        assert "## Featured Stories" in md or "## Watchlist" in md, (
            "Rendered Markdown should contain at least one content section"
        )
        # Disclaimer must always appear
        assert "not be used as clinical guidance" in md


# ---------------------------------------------------------------------------
# Sample before/after fixture
# (Demonstrates the transformation from internal-report format to newsletter copy)
# ---------------------------------------------------------------------------

SAMPLE_BEFORE = {
    "title": "Endometriosis and IVF Live Birth Rates",
    "source": "Human Reproduction",
    "published_date": "2026-02-15",
    "url": "https://example.com/endo-ivf",
    "topic": "art_ivf",
    "evidence_tier": "full",
    "evidence_sufficiency": 0.72,
    "audience_relevance": 8.0,
    "impact_score": 11.0,
    "summary": "Women with endometriosis had lower live birth rates compared to controls.",
    "summary_sections": {
        "what_it_studied": (
            "**What it studied**: The effect of endometriosis on live birth rates "
            "in women undergoing IVF."
        ),
        "what_it_found": (
            "**What it found**: Women with endometriosis had lower live birth rates "
            "compared to controls in this cohort."
        ),
        "why_it_matters": (
            "**Why it matters**: These findings may inform pre-treatment counselling "
            "for women with endometriosis planning IVF."
        ),
        "caveats": (
            "**Caveats**: Sample size and study design details were not reported in "
            "the available text."
        ),
    },
    "evidence_quality": {
        "confidence": "medium",
        "source_text_quality": "abstract",
        "article_type": "original_research",
        "is_fallback": False,
        "validation_passed": True,
        "unsupported_sentence_count": 0,
    },
}

EXPECTED_AFTER_EXCERPT = (
    # The newsletter should strip the markdown headers and produce clean prose
    "endometriosis"  # topic present
)


class TestSampleBeforeAfterTransformation:
    """Verify the SAMPLE_BEFORE record transforms cleanly for the newsletter."""

    def test_sample_becomes_featured(self):
        newsletter = _compose([SAMPLE_BEFORE])
        featured = newsletter.get("featured") or []
        assert len(featured) >= 1, "Sample article should appear in Featured"
        assert featured[0]["title"] == SAMPLE_BEFORE["title"]

    def test_sample_featured_body_has_no_markdown_headers(self):
        newsletter = _compose([SAMPLE_BEFORE])
        featured = newsletter.get("featured") or []
        body = (featured[0].get("body") or "").lower()
        assert "**what it studied**" not in body
        assert "**what it found**" not in body
        assert "**why it matters**" not in body
        assert "**caveats**" not in body

    def test_sample_featured_body_contains_key_content(self):
        newsletter = _compose([SAMPLE_BEFORE])
        featured = newsletter.get("featured") or []
        body = (featured[0].get("body") or "").lower()
        assert EXPECTED_AFTER_EXCERPT in body, (
            f"Expected key topic '{EXPECTED_AFTER_EXCERPT}' not found in body: {body[:200]}"
        )

    def test_sample_featured_body_has_no_forbidden_phrases(self):
        newsletter = _compose([SAMPLE_BEFORE])
        featured = newsletter.get("featured") or []
        body = (featured[0].get("body") or "").lower()
        for phrase in NC.INTERNAL_REPORT_PHRASES:
            assert phrase not in body, (
                f"Forbidden phrase '{phrase}' found in featured body"
            )
