"""
Automated tests for summarizer.py.

All 7 required test cases are covered:
  1. No unsupported numbers in generated summary.
  2. Missing sample size must not be invented.
  3. Review/editorial article must not be framed as original research.
  4. Low-evidence (snippet_only) produces a short cautious blurb.
  5. Contradiction detection flags opposing effect directions across reruns.
  6. Study-design blocking: unsupported design terms rejected.
  7. Sentence verifier: unsupported sentences labelled and filtered.

No network calls are made — LLM calls are injected via mocks.
"""

import json
import sqlite3
import sys
import os

import pytest

# Allow importing summarizer from the parent news-agent directory when tests
# are run from within the tests/ subdirectory or from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import summarizer as S

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(
    title="Test Article",
    content="",
    url="https://example.com/test",
    source="Test Journal",
    published_date=None,
):
    return {
        "title": title,
        "content": content,
        "url": url,
        "source": source,
        "published_date": published_date,
        "authors": [],
    }


def _make_evidence(
    article_type="original_research",
    source_text_quality="abstract",
    confidence="medium",
    sample_size_value=None,
    study_design_value=None,
    key_findings=None,
    missing_fields=None,
):
    ev = S.EvidenceObject(
        article_id="https://example.com/test",
        url="https://example.com/test",
        title="Test Article",
        article_type=article_type,
        source_text_quality=source_text_quality,
        confidence=confidence,
    )
    ev.sample_size = S.SupportedField(value=sample_size_value, support=[sample_size_value] if sample_size_value else [])
    ev.study_design = S.SupportedField(value=study_design_value, support=[study_design_value] if study_design_value else [])
    ev.key_findings = key_findings or []
    ev.missing_fields = missing_fields or []
    return ev


def _in_memory_conn():
    """Open an in-memory SQLite connection with the evidence_cache table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(S._DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Test 1 – No unsupported numbers
# ---------------------------------------------------------------------------

class TestNoUnsupportedNumbers:
    """
    Given evidence with NO numeric findings, the generated summary must contain
    no numbers.  validate_numbers should fail when numbers are present.
    """

    def test_summary_without_numbers_passes(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="lower live birth rates compared to controls",
                support=["lower live birth rates compared to controls"],
                contains_numeric_claim=False,
            )]
        )
        summary = "Live birth rates were lower in women with endometriosis compared to controls."
        result = S.validate_numbers(summary, ev)
        assert result.passed, f"Expected pass but got: {result.details}"

    def test_summary_with_unsupported_number_fails(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="lower live birth rates compared to controls",
                support=["lower live birth rates"],
                contains_numeric_claim=False,
            )]
        )
        # "23" does not appear anywhere in the evidence
        summary = "Live birth rates were 23% lower in women with endometriosis."
        result = S.validate_numbers(summary, ev)
        assert not result.passed, "Expected failure for unsupported number '23'"
        assert "23" in result.offending_items

    def test_number_present_in_evidence_passes(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="42% reduction in live birth rate",
                support=["42% reduction in live birth rate"],
                contains_numeric_claim=True,
            )]
        )
        summary = "The study observed a 42% reduction in live birth rate."
        result = S.validate_numbers(summary, ev)
        assert result.passed, f"Expected pass; evidence supports 42: {result.details}"


# ---------------------------------------------------------------------------
# Test 2 – Missing sample size must not be invented
# ---------------------------------------------------------------------------

class TestMissingSampleSize:
    """
    If sample_size is absent in source text, the summary must not introduce
    a specific patient count or similar number.
    """

    def test_invented_sample_size_triggers_validator(self):
        ev = _make_evidence(
            sample_size_value=None,
            missing_fields=["sample_size"],
            key_findings=[S.KeyFinding(
                value="improved pregnancy rates observed",
                support=["improved pregnancy rates observed"],
                contains_numeric_claim=False,
            )],
        )
        # "120" is not present in evidence — should be caught
        summary = "The study enrolled 120 women and found improved pregnancy rates."
        result = S.validate_numbers(summary, ev)
        assert not result.passed
        assert "120" in result.offending_items

    def test_no_sample_size_no_number_passes(self):
        ev = _make_evidence(
            sample_size_value=None,
            missing_fields=["sample_size"],
        )
        summary = (
            "Sample size was not reported in the available text. "
            "The study examined pregnancy outcomes in women with PCOS."
        )
        result = S.validate_numbers(summary, ev)
        assert result.passed

    def test_produce_fallback_does_not_invent_sample_size(self):
        ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
            sample_size_value=None,
        )
        fallback = S.produce_fallback(ev)
        prose = fallback.to_prose()
        # No number should appear at all
        numbers = S._extract_numbers(prose)
        assert numbers == [], f"Fallback invented number(s): {numbers}"


# ---------------------------------------------------------------------------
# Test 3 – Review/editorial article must not be framed as original research
# ---------------------------------------------------------------------------

class TestReviewArticleHandling:
    """
    A review or editorial must not be summarised as if it were original research.
    The study-design validator blocks RCT/trial language absent from evidence.
    """

    def test_review_article_rejects_rct_language(self):
        ev = _make_evidence(
            article_type="review",
            study_design_value=None,    # no design stated
        )
        summary = "This randomized trial enrolled women with PCOS and found significant improvement."
        result = S.validate_design_claims(summary, ev)
        assert not result.passed
        assert "randomized" in result.offending_items

    def test_review_article_rejects_prospective_language(self):
        ev = _make_evidence(
            article_type="review",
            study_design_value=None,
        )
        summary = "This prospective cohort study followed 400 patients over 12 months."
        result = S.validate_design_claims(summary, ev)
        assert not result.passed

    def test_editorial_does_not_pass_design_check_with_rct_claim(self):
        ev = _make_evidence(
            article_type="editorial",
            study_design_value=None,
        )
        summary = "This multicenter study provides important evidence."
        result = S.validate_design_claims(summary, ev)
        assert not result.passed, "multicenter not in evidence; should fail"

    def test_review_without_design_language_passes(self):
        ev = _make_evidence(
            article_type="review",
            study_design_value=None,
        )
        summary = (
            "This review examined evidence on preconception supplementation. "
            "The authors found consistent associations across multiple studies."
        )
        result_design = S.validate_design_claims(summary, ev)
        assert result_design.passed


# ---------------------------------------------------------------------------
# Test 4 – Low-evidence fallback is short and cautious
# ---------------------------------------------------------------------------

class TestLowEvidenceFallback:
    """
    snippet_only / low-confidence retrieval should produce a brief, cautious
    blurb — not a detailed results section.
    """

    def test_fallback_is_flagged(self):
        ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
        )
        result = S.produce_fallback(ev)
        assert result.is_fallback

    def test_fallback_contains_no_invented_numbers(self):
        ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
        )
        result = S.produce_fallback(ev)
        prose = result.to_prose()
        assert S._extract_numbers(prose) == [], f"Fallback contains numbers: {prose}"

    def test_fallback_does_not_contain_detailed_findings_header(self):
        ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
        )
        result = S.produce_fallback(ev)
        prose = result.to_prose().lower()
        # Ensure it is not claiming specific study results
        assert "found that" not in prose or "not available" in prose

    def test_full_pipeline_uses_fallback_for_low_confidence(self):
        """
        When the LLM caller returns None (simulating unavailability), the pipeline
        must fall back gracefully for a snippet_only article.
        """
        article = _make_article(
            title="Brief news item",
            content="Short snippet.",  # < 300 chars → snippet_only
        )

        def null_ollama(prompt, cfg, timeout=300):
            return None

        result = S.summarize_article(
            article=article,
            config={},
            db_path=None,
            llm_caller=null_ollama,
        )
        assert result.is_fallback
        prose = result.to_prose()
        assert len(prose) > 0


# ---------------------------------------------------------------------------
# Test 5 – Contradiction detection flags opposing effect directions
# ---------------------------------------------------------------------------

class TestContradictionDetection:

    def test_no_contradiction_first_run(self):
        conn = _in_memory_conn()
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="increased pregnancy rates in IVF group",
                support=["increased pregnancy rates"],
            )]
        )
        ev.article_id = "https://example.com/article-1"
        result = S.check_contradictions(ev, conn)
        assert not result.has_contradiction
        conn.close()

    def test_contradiction_detected_on_second_run(self):
        conn = _in_memory_conn()

        ev_v1 = _make_evidence(
            key_findings=[S.KeyFinding(
                value="increased pregnancy rates in IVF group",
                support=["increased pregnancy rates"],
            )],
            sample_size_value="200 patients",
        )
        ev_v1.article_id = "https://example.com/article-1"
        summary_v1 = S.StructuredSummary(
            what_it_studied="IVF outcomes",
            what_it_found="Increased rates.",
            why_it_matters="Relevant.",
            evidence=ev_v1,
        )
        S.persist_evidence(ev_v1, summary_v1, conn)

        # Second run: opposite effect direction, different sample size
        ev_v2 = _make_evidence(
            key_findings=[S.KeyFinding(
                value="no significant change in pregnancy rates",
                support=["no significant change"],
            )],
            sample_size_value="350 patients",  # changed
        )
        ev_v2.article_id = "https://example.com/article-1"

        result = S.check_contradictions(ev_v2, conn)
        assert result.has_contradiction
        assert any("key_finding" in f for f in result.fields_changed)
        assert any("sample_size" in f for f in result.fields_changed)
        conn.close()

    def test_no_contradiction_when_evidence_unchanged(self):
        conn = _in_memory_conn()

        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="improved outcomes with melatonin supplementation",
                support=["improved outcomes"],
            )],
            sample_size_value="50 women",
            study_design_value="randomized controlled trial",
        )
        ev.article_id = "https://example.com/article-2"
        summary = S.StructuredSummary(
            what_it_studied="Melatonin and IVF.",
            what_it_found="Improved outcomes.",
            why_it_matters="Important.",
            evidence=ev,
        )
        S.persist_evidence(ev, summary, conn)

        # Identical second run
        ev2 = _make_evidence(
            key_findings=[S.KeyFinding(
                value="improved outcomes with melatonin supplementation",
                support=["improved outcomes"],
            )],
            sample_size_value="50 women",
            study_design_value="randomized controlled trial",
        )
        ev2.article_id = "https://example.com/article-2"
        result = S.check_contradictions(ev2, conn)
        assert not result.has_contradiction
        conn.close()


# ---------------------------------------------------------------------------
# Test 6 – Study-design blocking
# ---------------------------------------------------------------------------

class TestStudyDesignBlocking:
    """
    Summary must not say 'randomized', 'prospective', 'multicenter', etc.
    unless those terms appear explicitly in evidence.study_design.
    """

    DESIGN_TERM_CASES = [
        ("randomized", "This randomized trial showed improvement."),
        ("prospective", "A prospective study followed patients for 6 months."),
        ("multicenter", "This multicenter analysis included three sites."),
        ("double-blind", "The double-blind design reduced bias."),
        ("retrospective", "A retrospective chart review was conducted."),
    ]

    @pytest.mark.parametrize("term,summary", DESIGN_TERM_CASES)
    def test_unsupported_design_term_fails(self, term, summary):
        ev = _make_evidence(study_design_value=None)  # no design in evidence
        result = S.validate_design_claims(summary, ev)
        assert not result.passed, (
            f"Expected failure for term '{term}' absent from evidence"
        )
        assert term in result.offending_items, (
            f"Expected '{term}' in offending_items: {result.offending_items}"
        )

    def test_supported_design_term_passes(self):
        ev = _make_evidence(study_design_value="randomized controlled trial")
        summary = "This randomized controlled trial enrolled women undergoing IVF."
        result = S.validate_design_claims(summary, ev)
        assert result.passed, f"Expected pass; evidence contains 'randomized': {result.details}"

    def test_validate_summary_collects_all_validators(self):
        ev = _make_evidence(
            study_design_value=None,
            key_findings=[],
        )
        # Both a number and a design term are unsupported
        summary = "This randomized study enrolled 150 patients and found significant results."
        results = S.validate_summary(summary, ev)
        assert len(results) >= 3  # number + design + causal (+ inference_language_validator)
        by_rule = {r.rule: r for r in results}
        assert not by_rule["number_validator"].passed
        assert not by_rule["design_validator"].passed


# ---------------------------------------------------------------------------
# Test 7 – Sentence verifier
# ---------------------------------------------------------------------------

class TestSentenceVerifier:
    """
    The rule-based verifier must label unsupported sentences correctly and
    apply_sentence_filter must remove them from the structured summary.
    """

    def test_sentence_with_unsupported_number_is_unsupported(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="lower live birth rates compared to controls",
                support=["lower live birth rates"],
                contains_numeric_claim=False,
            )]
        )
        summary = (
            "Live birth rates were lower in women with endometriosis. "
            "Rates were 23% lower than in controls. "
            "This finding is clinically relevant."
        )
        labels = S.verify_sentences_ruleset(summary, ev)
        by_sentence = {lbl.sentence: lbl for lbl in labels}
        assert "Rates were 23% lower than in controls." in by_sentence
        assert by_sentence["Rates were 23% lower than in controls."].label == "unsupported"

    def test_sentence_matching_key_finding_is_supported(self):
        """
        A sentence that is a near-paraphrase of a key finding should receive
        at least 'supported' when token overlap is ≥ 60 %, or 'weakly_supported'
        when the match is partial.  It must NOT be 'unsupported'.
        """
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="lower live birth rates compared to controls",
                support=["lower live birth rates compared to controls"],
                contains_numeric_claim=False,
            )]
        )
        summary = "Live birth rates were lower compared to controls in this population."
        labels = S.verify_sentences_ruleset(summary, ev)
        assert len(labels) == 1
        assert labels[0].label in ("supported", "weakly_supported"), (
            f"Expected supported or weakly_supported, got: {labels[0].label} "
            f"(reason: {labels[0].reason})"
        )
        assert labels[0].label != "unsupported"

    def test_sentence_with_unsupported_design_term_is_unsupported(self):
        ev = _make_evidence(study_design_value=None)
        summary = "This multicenter randomized study found no significant difference."
        labels = S.verify_sentences_ruleset(summary, ev)
        assert labels[0].label == "unsupported"
        assert "multicenter" in labels[0].reason or "randomized" in labels[0].reason

    def test_apply_sentence_filter_removes_unsupported_sentences(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="lower live birth rates compared to controls",
                support=["lower live birth rates compared to controls"],
            )]
        )
        # "38%" is not in evidence — that sentence should be removed
        structured = S.StructuredSummary(
            what_it_studied="IVF outcomes in endometriosis patients.",
            what_it_found=(
                "Live birth rates were lower compared to controls. "
                "The reduction was 38%."
            ),
            why_it_matters="These findings are relevant to clinical practice.",
            evidence=ev,
        )
        prose = structured.to_prose()
        structured.sentence_labels = S.verify_sentences_ruleset(prose, ev)
        filtered = S.apply_sentence_filter(structured)
        assert "38%" not in filtered.what_it_found, (
            f"Unsupported sentence not removed: {filtered.what_it_found}"
        )

    def test_all_supported_sentences_survive_filter(self):
        ev = _make_evidence(
            key_findings=[S.KeyFinding(
                value="endometriosis was associated with lower live birth rates",
                support=["endometriosis was associated with lower live birth rates"],
            )]
        )
        structured = S.StructuredSummary(
            what_it_studied="The relationship between endometriosis and IVF outcomes.",
            what_it_found="Endometriosis was associated with lower live birth rates.",
            why_it_matters="This evidence may inform pre-treatment counselling.",
            evidence=ev,
        )
        prose = structured.to_prose()
        structured.sentence_labels = S.verify_sentences_ruleset(prose, ev)
        filtered = S.apply_sentence_filter(structured)
        # Core finding should survive
        assert "lower live birth rates" in filtered.what_it_found


# ---------------------------------------------------------------------------
# Additional integration-style test: full pipeline with injected LLM responses
# ---------------------------------------------------------------------------

class TestFullPipelineWithMockLLM:
    """
    Verify the full summarize_article() pipeline produces valid output when
    Mock LLM returns well-formed (and intentionally malformed) responses.
    """

    def _mock_good_extraction(self) -> str:
        return json.dumps({
            "article_type": "original_research",
            "source_text_quality": "abstract",
            "study_design": {"value": "prospective cohort", "support": ["prospective cohort study"]},
            "population": {"value": "women undergoing IVF", "support": ["women undergoing IVF"]},
            "sample_size": {"value": "N=120", "support": ["120 women were enrolled"]},
            "intervention_or_exposure": {"value": "IVF with melatonin supplementation",
                                          "support": ["melatonin supplementation prior to IVF"]},
            "comparison": {"value": "standard IVF", "support": ["compared to standard IVF"]},
            "primary_outcomes": [{"value": "live birth rate", "support": ["live birth rate"]}],
            "key_findings": [{"value": "higher live birth rate in melatonin group",
                               "support": ["higher live birth rate in melatonin group"],
                               "contains_numeric_claim": False}],
            "limitations": [{"value": "single-centre study", "support": ["single-centre design"]}],
            "missing_fields": [],
            "confidence": "medium",
        })

    def _mock_good_generation(self) -> str:
        return json.dumps({
            "what_it_studied": "The effect of melatonin supplementation on IVF live birth rates.",
            "what_it_found": (
                "Women receiving melatonin prior to IVF had a higher live birth rate "
                "compared to those undergoing standard IVF."
            ),
            "why_it_matters": (
                "If confirmed in larger trials, melatonin supplementation may represent "
                "a low-cost adjunct to IVF protocols."
            ),
            "caveats": (
                "This was a single-centre prospective cohort study; "
                "findings require replication in larger multi-site studies before clinical adoption."
            ),
        })

    def test_pipeline_produces_structured_summary(self):
        call_count = {"n": 0}

        def mock_caller(prompt, cfg, timeout=300):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return self._mock_good_extraction()
            return self._mock_good_generation()

        article = _make_article(
            title="Melatonin supplementation and IVF outcomes",
            content=(
                "A prospective cohort study enrolled 120 women undergoing IVF with melatonin "
                "supplementation prior to IVF compared to standard IVF. "
                "Results showed higher live birth rate in melatonin group. "
                "The study was conducted at a single centre."
            ),
        )

        result = S.summarize_article(
            article=article,
            config={},
            db_path=None,
            llm_caller=mock_caller,
        )

        assert not result.is_fallback
        assert result.what_it_studied
        assert result.what_it_found
        assert result.why_it_matters
        # The number "120" appears in the evidence so should not block validation
        vr_map = {v.rule: v for v in result.validation_results}
        # prospective cohort is in evidence.study_design so design validator should pass
        assert vr_map["design_validator"].passed, vr_map["design_validator"].details

    def test_pipeline_falls_back_on_json_parse_failure(self):
        """If the LLM returns garbage, the pipeline must not raise and must fall back."""
        def garbage_caller(prompt, cfg, timeout=300):
            return "This is not valid JSON at all!!!"

        article = _make_article(
            title="A relevant fertility study",
            content="Abstract: " + ("word " * 80),  # > 300 chars → "abstract" quality
        )
        result = S.summarize_article(
            article=article,
            config={},
            db_path=None,
            llm_caller=garbage_caller,
        )
        # Should not raise; must produce a safe output
        assert result is not None
        prose = result.to_prose()
        assert len(prose) > 0


# ---------------------------------------------------------------------------
# Test 8 – Evidence sufficiency scoring & tiering
# ---------------------------------------------------------------------------

class TestEvidenceSufficiencyAndTiering:
    """
    Snippet-only fallback items must score lower than abstract-backed items,
    and the tier system must correctly classify evidence levels.
    """

    def test_abstract_evidence_scores_higher_than_snippet(self):
        snippet_ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
        )
        abstract_ev = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            sample_size_value="N=80",
            key_findings=[S.KeyFinding(value="improved outcomes", support=["improved outcomes"])],
        )
        assert S.evidence_sufficiency_score(abstract_ev) > S.evidence_sufficiency_score(snippet_ev)

    def test_key_findings_increase_score(self):
        base = _make_evidence(source_text_quality="abstract")
        with_findings = _make_evidence(
            source_text_quality="abstract",
            key_findings=[S.KeyFinding(value="lower birth rates", support=["lower birth rates"])],
        )
        assert S.evidence_sufficiency_score(with_findings) > S.evidence_sufficiency_score(base)

    def test_full_evidence_is_full_tier(self):
        ev = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            sample_size_value="50 women",
            key_findings=[S.KeyFinding(value="positive association", support=["positive association"])],
        )
        assert S.evidence_tier_from_evidence(ev) == "full"

    def test_sparse_abstract_is_short_blurb_tier(self):
        ev = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            # No sample size, no key findings → low sufficiency
        )
        # score = 0.30 (abstract) → "short_blurb" (0.20 ≤ score < 0.50)
        assert S.evidence_tier_from_evidence(ev) in ("short_blurb", "titles_to_watch")

    def test_snippet_only_is_titles_to_watch_tier(self):
        ev = _make_evidence(
            source_text_quality="snippet_only",
            confidence="low",
        )
        assert S.evidence_tier_from_evidence(ev) == "titles_to_watch"

    def test_estimate_evidence_tier_short_content_is_not_full(self):
        article = _make_article(title="Brief update", content="Short.")
        tier = S.estimate_evidence_tier(article)
        assert tier in ("short_blurb", "titles_to_watch")

    def test_estimate_evidence_tier_long_content_is_full(self):
        article = _make_article(
            title="A prospective cohort study",
            content=(
                "Background: Endometriosis affects fertility outcomes. "
                "We conducted a prospective cohort study of 120 women. "
                "Methods: Women undergoing IVF were enrolled and followed. "
                "Results: Live birth rates were lower in the endometriosis group. "
                "Conclusion: Endometriosis is associated with reduced IVF success. "
            ) * 3,  # > 300 chars
        )
        tier = S.estimate_evidence_tier(article)
        assert tier == "full"

    def test_tier_penalty_applied_in_impact_calculation(self):
        """
        Verify the penalty constants exist and that titles_to_watch has the
        largest penalty (most negative).
        """
        assert S._TIER_SCORE_PENALTY["titles_to_watch"] < S._TIER_SCORE_PENALTY["short_blurb"]
        assert S._TIER_SCORE_PENALTY["short_blurb"] < S._TIER_SCORE_PENALTY["full"]


# ---------------------------------------------------------------------------
# Test 9 – Genericness detection
# ---------------------------------------------------------------------------

class TestGenericnessDetection:

    def _make_summary(
        self,
        what_it_studied="The study examined X.",
        what_it_found="The study found improvements in outcomes.",
        why_it_matters="This is relevant to fertility and reproductive medicine.",
        caveats="",
        evidence=None,
        tier="full",
    ) -> S.StructuredSummary:
        if evidence is None:
            evidence = _make_evidence()
        return S.StructuredSummary(
            what_it_studied=what_it_studied,
            what_it_found=what_it_found,
            why_it_matters=why_it_matters,
            caveats=caveats,
            evidence=evidence,
            tier=tier,
        )

    def test_generic_why_it_matters_detected(self):
        summary = self._make_summary(
            why_it_matters="The topic is relevant to fertility and reproductive medicine."
        )
        article = _make_article(title="IVF outcomes study")
        result = S.detect_genericness(summary, article)
        assert result.is_generic
        assert any("Why it matters" in r for r in result.reasons)

    def test_title_restatement_in_what_it_found_detected(self):
        title = "melatonin supplementation improves ivf outcomes in pcos patients"
        article = _make_article(title=title)
        summary = self._make_summary(
            what_it_found="melatonin supplementation improves ivf outcomes in pcos patients"
        )
        result = S.detect_genericness(summary, article)
        assert result.is_generic
        assert any("restatement" in r.lower() or "overlaps" in r.lower() for r in result.reasons)

    def test_fallback_language_in_what_it_found_detected(self):
        summary = self._make_summary(
            what_it_found="Detailed findings were not available in the retrieved source text."
        )
        article = _make_article(title="A fertility study")
        result = S.detect_genericness(summary, article)
        assert result.is_generic
        assert any("fallback" in r.lower() or "concrete" in r.lower() for r in result.reasons)

    def test_specific_non_generic_summary_not_flagged(self):
        summary = self._make_summary(
            what_it_studied="The effect of endometriosis on IVF live birth rates.",
            what_it_found="Women with endometriosis had lower live birth rates compared to controls.",
            why_it_matters=(
                "The association between endometriosis and reduced IVF success "
                "may influence pre-treatment counselling for women planning IVF."
            ),
            caveats="Sample size was not reported in the available text.",
        )
        article = _make_article(title="Endometriosis and IVF outcomes")
        result = S.detect_genericness(summary, article)
        assert not result.is_generic, f"Should not be generic: {result.reasons}"

    def test_can_regenerate_flag_requires_sufficient_evidence(self):
        ev_poor = _make_evidence(source_text_quality="snippet_only", confidence="low")
        summary_poor = self._make_summary(evidence=ev_poor, why_it_matters="The topic is relevant.")
        result_poor = S.detect_genericness(summary_poor, _make_article())
        assert not result_poor.can_regenerate

        ev_good = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            key_findings=[S.KeyFinding(value="improved outcomes", support=["improved"])],
        )
        summary_good = self._make_summary(evidence=ev_good, why_it_matters="The topic is relevant.")
        result_good = S.detect_genericness(summary_good, _make_article())
        assert result_good.can_regenerate


# ---------------------------------------------------------------------------
# Test 10 – Repeated fallback language detection
# ---------------------------------------------------------------------------

class TestRepeatedFallbackLanguage:

    def test_repeated_phrase_across_majority_of_articles_flagged(self):
        articles = []
        for i in range(6):
            articles.append({
                "title": f"Article {i}",
                "generated_summary": (
                    "The topic is relevant to fertility and reproductive medicine; "
                    "readers are encouraged to consult the full publication for details."
                ),
            })
        repeated = S.check_repeated_fallbacks(articles, threshold_pct=0.30)
        assert len(repeated) > 0, "Should detect repeated generic phrases"
        assert any("topic is relevant" in p for p in repeated)

    def test_non_repeated_phrases_not_flagged(self):
        articles = [
            {"title": "A", "generated_summary": "Women with endometriosis had lower live birth rates."},
            {"title": "B", "generated_summary": "Melatonin supplementation was associated with improved IVF outcomes."},
            {"title": "C", "generated_summary": "Sperm DNA fragmentation was linked to embryo quality."},
        ]
        repeated = S.check_repeated_fallbacks(articles, threshold_pct=0.30)
        assert repeated == [], f"Should not flag unique summaries: {repeated}"

    def test_small_collection_threshold_is_at_least_2(self):
        """Threshold is max(2, n * pct) so even a single repeated phrase in 2/3 articles
        should be caught at 30% threshold when n ≥ 2."""
        articles = [
            {"title": "A", "generated_summary": "Detailed findings were not available in the retrieved source text."},
            {"title": "B", "generated_summary": "Detailed findings were not available in the retrieved source text."},
            {"title": "C", "generated_summary": "Embryo quality was associated with live birth rate."},
        ]
        repeated = S.check_repeated_fallbacks(articles, threshold_pct=0.30)
        assert any("detailed findings were not" in p for p in repeated)


# ---------------------------------------------------------------------------
# Test 11 – Inference language blocking
# ---------------------------------------------------------------------------

class TestInferenceLanguageBlocking:

    def test_speculative_phrase_blocked_for_medium_confidence(self):
        ev = _make_evidence(confidence="medium", source_text_quality="abstract")
        summary = "The study found a strong inverse relationship between BMI and fertility."
        result = S.validate_inference_language(summary, ev)
        assert not result.passed
        assert result.rule == "inference_language_validator"
        assert "strong inverse relationship" in result.offending_items

    def test_speculative_phrase_blocked_for_low_confidence(self):
        ev = _make_evidence(confidence="low", source_text_quality="snippet_only")
        summary = "Oxidative stress could negatively impact sperm DNA integrity."
        result = S.validate_inference_language(summary, ev)
        assert not result.passed

    def test_speculative_phrase_passes_when_in_evidence_corpus(self):
        ev = _make_evidence(
            confidence="medium",
            key_findings=[S.KeyFinding(
                value="strong inverse relationship between BMI and oocyte quality",
                support=["strong inverse relationship between BMI and oocyte quality"],
                contains_numeric_claim=False,
            )],
        )
        summary = "The study observed a strong inverse relationship between BMI and oocyte quality."
        result = S.validate_inference_language(summary, ev)
        assert result.passed, f"Should pass when phrase is in evidence: {result.details}"

    def test_no_speculative_phrase_passes(self):
        ev = _make_evidence(confidence="medium")
        summary = "The retrieved abstract describes an association between PCOS and reduced fertility."
        result = S.validate_inference_language(summary, ev)
        assert result.passed

    def test_inference_validator_included_in_validate_summary(self):
        ev = _make_evidence(confidence="medium")
        summary = "The study clearly demonstrates that IVF significantly reduces infertility burden."
        results = S.validate_summary(summary, ev)
        rules = {r.rule for r in results}
        assert "inference_language_validator" in rules
        failed_rules = {r.rule for r in results if not r.passed}
        assert "inference_language_validator" in failed_rules


# ---------------------------------------------------------------------------
# Test 12 – Multi-tier fallback output format
# ---------------------------------------------------------------------------

class TestMultiTierFallbackFormat:

    def test_titles_to_watch_tier_produces_single_line(self):
        ev = _make_evidence(source_text_quality="snippet_only", confidence="low")
        result = S.produce_fallback(ev)
        assert result.tier == "titles_to_watch"
        prose = result.to_prose()
        # Should be a single compact line, not a multi-section template
        assert "**What it found**" not in prose
        assert "**Why it matters**" not in prose
        assert len(prose.split("\n")) == 1

    def test_short_blurb_tier_produces_compact_output(self):
        ev = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            # No key findings → score ~0.30 → "short_blurb"
        )
        result = S.produce_fallback(ev)
        assert result.tier == "short_blurb"
        prose = result.to_prose()
        assert "**What it found**" not in prose
        assert "**Why it matters**" not in prose

    def test_no_numbers_in_any_tier(self):
        for sqt, conf in [("snippet_only", "low"), ("abstract", "medium")]:
            ev = _make_evidence(source_text_quality=sqt, confidence=conf)
            result = S.produce_fallback(ev)
            prose = result.to_prose()
            numbers = S._extract_numbers(prose)
            assert numbers == [], f"Numbers found in {result.tier} fallback: {numbers}"

    def test_full_tier_uses_4_section_template(self):
        """A StructuredSummary with tier='full' should use the 4-section template."""
        ev = _make_evidence(
            source_text_quality="abstract",
            confidence="medium",
            key_findings=[S.KeyFinding(value="improved pregnancy rates", support=["improved"])],
        )
        summary = S.StructuredSummary(
            what_it_studied="The effect of X on Y.",
            what_it_found="Improved pregnancy rates were observed.",
            why_it_matters="These findings are relevant to clinical practice.",
            caveats="Sample size was not reported.",
            evidence=ev,
            tier="full",
        )
        prose = summary.to_prose()
        assert "**What it studied**" in prose
        assert "**What it found**" in prose
        assert "**Why it matters**" in prose
        assert "**Caveats**" in prose

    def test_titles_to_watch_uses_title_from_evidence(self):
        ev = _make_evidence(source_text_quality="snippet_only", confidence="low")
        ev.title = "Novel biomarker for embryo quality"
        ev.journal = "Fertility and Sterility"
        result = S.produce_fallback(ev)
        assert result.tier == "titles_to_watch"
        prose = result.to_prose()
        assert "Novel biomarker for embryo quality" in prose
