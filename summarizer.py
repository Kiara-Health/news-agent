"""
Evidence-bounded two-stage summarizer for fertility/reproductive medicine articles.

Architecture
------------
Stage A  — evidence_extraction
    A constrained LLM prompt extracts only claims explicitly present in the
    source text into a structured EvidenceObject.  Missing fields stay null;
    no values are inferred from model background knowledge.

Stage B  — prose_generation
    A second constrained prompt generates a 4-section summary solely from the
    EvidenceObject.  It may not introduce any values absent from that object.
    If source quality is "snippet_only" (< 300 chars) the stage is skipped
    entirely and a safe template-based fallback is used instead.

Validators  (post Stage B, pure Python)
    * number_validator   – rejects summaries containing numbers absent from
                           the evidence object.
    * design_validator   – rejects unsupported study-design labels.
    * causal_validator   – flags unsupported causal / recommendation language.

Sentence verifier  (rule-based)
    Labels each sentence supported / weakly_supported / unsupported and
    removes or softens the latter two before the summary ships.

Contradiction detector
    Persists evidence objects to a SQLite cache (evidence_cache.db) and
    compares field-level values across reruns, flagging material changes in
    sample size, effect direction, study design, or key outcomes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_caller import call_openai as _default_llm_caller

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNIPPET_ONLY_THRESHOLD = 300    # chars
ABSTRACT_THRESHOLD = 2_000      # chars

DESIGN_TERMS: List[str] = [
    "randomized", "randomised", "prospective", "retrospective",
    "multicenter", "multicentre", "multi-center", "multi-centre",
    "single-center", "single-centre", "double-blind", "double blind",
    "placebo-controlled", "placebo controlled", "crossover", "cross-over",
    "case-control", "case control",
]

CAUSAL_PHRASES: List[str] = [
    "causes ", "caused by", "leads to", "results in",
    "we recommend", "clinicians should", "patients should",
    "the guideline", "standard of care", "should be offered",
    "should be considered first-line", "proves that", "demonstrates that",
    "confirms that",
]

# Speculative inference phrases that go beyond what sparse evidence supports.
# These are blocked for low/medium-confidence summaries unless the phrase
# appears verbatim in the evidence corpus.
SPECULATIVE_PHRASES: List[str] = [
    "strong inverse relationship",
    "strong positive relationship",
    "strong association",
    "could negatively impact",
    "could positively impact",
    "resulting in fewer transferable",
    "resulting in reduced",
    "significantly impair",
    "significantly reduce",
    "significantly increase",
    "dramatically reduc",
    "dramatically increas",
    "markedly improv",
    "markedly reduc",
    "clearly demonstrat",
    "unequivocally",
    "directly causes",
    "definitively shows",
]

# Generic filler phrases that characterise low-information fallback summaries.
# Used by the genericness detector.
GENERIC_FALLBACK_PHRASES: List[str] = [
    "the topic is relevant to fertility and reproductive medicine",
    "readers are encouraged to consult the full publication",
    "detailed findings were not available in the retrieved source text",
    "only limited metadata was available",
    "this blurb does not reflect the study",
    "this appears to be a research article",
    "this appears to be a review article",
    "this appears to be an article",
    "this appears to be an editorial",
    "this appears to be a commentary",
]

_ARTICLE_TYPE_HINTS: Dict[str, List[str]] = {
    "editorial":         ["editorial", "editor's note", "letter to the editor",
                          "letter:", "in this issue:", "invited commentary"],
    "commentary":        ["commentary", "comment on", "perspective:", "viewpoint",
                          "expert opinion", "opinion piece", "clinical perspective"],
    "review":            ["systematic review", "meta-analysis", "narrative review",
                          "scoping review", "literature review", "review article",
                          "review of the literature", "pooled analysis",
                          "meta analysis", "cochrane", "evidence synthesis"],
    "original_research": ["randomized", "randomised", "cohort study", "clinical trial",
                          "prospective study", "retrospective study", "enrolled", "recruited",
                          "participants were", "patients underwent", "n =", "n=",
                          "case-control", "case series", "cross-sectional study",
                          "we conducted", "the aim of this study", "the purpose of this study"],
}

# Source-level cues: if the article source matches a known news / commentary site,
# treat as commentary by default.  Journal names map to original_research.
_SOURCE_TYPE_MAP: Dict[str, str] = {
    # News / commentary sites → commentary
    "medpage today": "commentary",
    "medscape": "commentary",
    "healthline": "commentary",
    "webmd": "commentary",
    "everyday health": "commentary",
    "verywell health": "commentary",
    "medical news today": "commentary",
    "the guardian": "commentary",
    "bbc health": "commentary",
    "science daily": "commentary",
    # Specialist fertility / reproductive journals → original_research (conservative default)
    "fertility and sterility": "original_research",
    "human reproduction": "original_research",
    "journal of assisted reproduction and genetics": "original_research",
    "reproductive biomedicine online": "original_research",
    "reproductive sciences": "original_research",
    "journal of reproductive medicine": "original_research",
    "archives of gynecology and obstetrics": "original_research",
    "gynecological endocrinology": "original_research",
    "andrology": "original_research",
    "human reproduction update": "review",
    "cochrane database": "review",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SupportedField:
    value: Optional[str] = None
    support: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"value": self.value, "support": self.support}

    @classmethod
    def from_dict(cls, d: dict) -> "SupportedField":
        if not isinstance(d, dict):
            return cls()
        return cls(value=d.get("value"), support=d.get("support") or [])


@dataclass
class KeyFinding:
    value: str = ""
    support: List[str] = field(default_factory=list)
    contains_numeric_claim: bool = False

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "support": self.support,
            "contains_numeric_claim": self.contains_numeric_claim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyFinding":
        if not isinstance(d, dict):
            return cls()
        return cls(
            value=d.get("value") or "",
            support=d.get("support") or [],
            contains_numeric_claim=bool(d.get("contains_numeric_claim", False)),
        )


@dataclass
class EvidenceObject:
    article_id: str = ""          # canonical URL or DOI
    url: str = ""
    title: str = ""
    journal: str = ""
    published_date: Optional[str] = None
    article_type: str = "unknown"           # original_research|review|editorial|commentary|unknown
    source_text_quality: str = "snippet_only"  # full_text|abstract|snippet_only
    study_design: SupportedField = field(default_factory=SupportedField)
    population: SupportedField = field(default_factory=SupportedField)
    sample_size: SupportedField = field(default_factory=SupportedField)
    intervention_or_exposure: SupportedField = field(default_factory=SupportedField)
    comparison: SupportedField = field(default_factory=SupportedField)
    primary_outcomes: List[SupportedField] = field(default_factory=list)
    key_findings: List[KeyFinding] = field(default_factory=list)
    limitations: List[SupportedField] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    confidence: str = "low"                  # high|medium|low

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "journal": self.journal,
            "published_date": self.published_date,
            "article_type": self.article_type,
            "source_text_quality": self.source_text_quality,
            "study_design": self.study_design.to_dict(),
            "population": self.population.to_dict(),
            "sample_size": self.sample_size.to_dict(),
            "intervention_or_exposure": self.intervention_or_exposure.to_dict(),
            "comparison": self.comparison.to_dict(),
            "primary_outcomes": [p.to_dict() for p in self.primary_outcomes],
            "key_findings": [f.to_dict() for f in self.key_findings],
            "limitations": [l.to_dict() for l in self.limitations],
            "missing_fields": self.missing_fields,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceObject":
        if not isinstance(d, dict):
            return cls()
        obj = cls(
            article_id=d.get("article_id") or "",
            url=d.get("url") or "",
            title=d.get("title") or "",
            journal=d.get("journal") or "",
            published_date=d.get("published_date"),
            article_type=d.get("article_type") or "unknown",
            source_text_quality=d.get("source_text_quality") or "snippet_only",
            missing_fields=d.get("missing_fields") or [],
            confidence=d.get("confidence") or "low",
        )
        obj.study_design = SupportedField.from_dict(d.get("study_design") or {})
        obj.population = SupportedField.from_dict(d.get("population") or {})
        obj.sample_size = SupportedField.from_dict(d.get("sample_size") or {})
        obj.intervention_or_exposure = SupportedField.from_dict(
            d.get("intervention_or_exposure") or {}
        )
        obj.comparison = SupportedField.from_dict(d.get("comparison") or {})
        obj.primary_outcomes = [
            SupportedField.from_dict(p) for p in (d.get("primary_outcomes") or [])
        ]
        obj.key_findings = [
            KeyFinding.from_dict(f) for f in (d.get("key_findings") or [])
        ]
        obj.limitations = [
            SupportedField.from_dict(l) for l in (d.get("limitations") or [])
        ]
        return obj


@dataclass
class ValidationResult:
    passed: bool
    rule: str
    details: str
    offending_items: List[str] = field(default_factory=list)


@dataclass
class SentenceLabel:
    sentence: str
    label: str      # supported | weakly_supported | unsupported
    reason: str


@dataclass
class ContradictionResult:
    has_contradiction: bool
    fields_changed: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class StructuredSummary:
    what_it_studied: str = ""
    what_it_found: str = ""
    why_it_matters: str = ""
    caveats: str = ""
    evidence: Optional[EvidenceObject] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    sentence_labels: List[SentenceLabel] = field(default_factory=list)
    contradiction: Optional[ContradictionResult] = None
    is_fallback: bool = False
    # Evidence tier controls output format: "full" | "short_blurb" | "titles_to_watch"
    tier: str = "full"
    evidence_sufficiency: float = 0.0

    def to_prose(self) -> str:
        """Return formatted prose appropriate to the evidence tier."""
        if self.tier == "titles_to_watch":
            return self.what_it_studied

        if self.tier == "short_blurb":
            parts = [p for p in [self.what_it_studied, self.caveats] if p]
            return " ".join(parts)

        # Full 4-section format — skip any empty section to avoid blank template labels
        lines = []
        if self.what_it_studied:
            lines.append(f"**What it studied**: {self.what_it_studied}")
        if self.what_it_found:
            lines.append(f"**What it found**: {self.what_it_found}")
        if self.why_it_matters:
            lines.append(f"**Why it matters**: {self.why_it_matters}")
        if self.caveats:
            lines.append(f"**Caveats**: {self.caveats}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "what_it_studied": self.what_it_studied,
            "what_it_found": self.what_it_found,
            "why_it_matters": self.why_it_matters,
            "caveats": self.caveats,
            "is_fallback": self.is_fallback,
            "tier": self.tier,
            "evidence_sufficiency": round(self.evidence_sufficiency, 3),
            "confidence": self.evidence.confidence if self.evidence else "low",
            "source_text_quality": (
                self.evidence.source_text_quality if self.evidence else "snippet_only"
            ),
            "validation_passed": all(v.passed for v in self.validation_results),
            "validation_details": [
                {"rule": v.rule, "passed": v.passed, "details": v.details}
                for v in self.validation_results
            ],
        }


# ---------------------------------------------------------------------------
# Genericness detection
# ---------------------------------------------------------------------------

@dataclass
class GenericnessResult:
    is_generic: bool
    reasons: List[str]
    can_regenerate: bool   # evidence supports a retry attempt
    suggested_tier: str    # tier to downgrade to if can't regenerate


def detect_genericness(
    summary: StructuredSummary, article: Dict
) -> GenericnessResult:
    """
    Detect if a summary is too generic to be useful as a main story entry.

    Checks
    ------
    1. "What it found" mostly restates the article title (≥ 65 % token overlap).
    2. "Why it matters" contains only a generic relevance phrase.
    3. "What it found" contains only fallback language (no concrete finding).
    4. Non-fallback main-story entry has an empty "What it found".

    Returns a GenericnessResult with downgrade recommendation.
    """
    reasons: List[str] = []

    title_tokens = set(re.findall(r"\b\w{4,}\b", (article.get("title") or "").lower()))
    found_lower = (summary.what_it_found or "").lower()
    found_tokens = set(re.findall(r"\b\w{4,}\b", found_lower))
    matters_lower = (summary.why_it_matters or "").lower()

    # Check 1 – "What it found" essentially restates the title
    if title_tokens and found_tokens:
        overlap = len(title_tokens & found_tokens) / max(len(title_tokens), len(found_tokens))
        if overlap >= 0.65:
            reasons.append(
                f"'What it found' overlaps {overlap:.0%} with the title (mostly a restatement)"
            )

    # Check 2 – "Why it matters" is purely generic
    generic_matters = [
        "the topic is relevant",
        "readers are encouraged to consult",
        "relevant to fertility and reproductive medicine",
        "relevant to fertility",
    ]
    if any(p in matters_lower for p in generic_matters):
        reasons.append("'Why it matters' contains only a generic relevance phrase")

    # Check 3 – "What it found" is entirely fallback language
    fallback_found = [
        "not available in the retrieved",
        "not reported in available text",
        "findings were not available",
        "could not be extracted",
        "detailed findings were not",
    ]
    if found_lower and any(p in found_lower for p in fallback_found):
        reasons.append("'What it found' contains only fallback language; no concrete finding")

    # Check 4 – Non-fallback summary has no "What it found" at all
    if not summary.is_fallback and not (summary.what_it_found or "").strip():
        reasons.append("Non-fallback summary has an empty 'What it found' section")

    is_generic = bool(reasons)

    # Can we retry? Only when evidence is substantive enough to expect improvement.
    ev = summary.evidence
    can_regen = bool(
        is_generic
        and ev is not None
        and ev.confidence in ("medium", "high")
        and ev.source_text_quality != "snippet_only"
        and ev.key_findings
    )

    if is_generic:
        suf = evidence_sufficiency_score(ev) if ev else 0.0
        if suf >= 0.50:
            suggested_tier = "full"    # retry, keep tier
        elif suf >= 0.20:
            suggested_tier = "short_blurb"
        else:
            suggested_tier = "titles_to_watch"
    else:
        suggested_tier = summary.tier

    return GenericnessResult(
        is_generic=is_generic,
        reasons=reasons,
        can_regenerate=can_regen,
        suggested_tier=suggested_tier,
    )


def check_repeated_fallbacks(
    articles: List[Dict], threshold_pct: float = 0.35
) -> List[str]:
    """
    Scan all generated summaries and return any generic fallback phrase that
    appears in more than *threshold_pct* of the article summaries.

    Call this after all articles have been summarised to detect systematic
    generic output that would make the report feel repetitive.
    """
    counts: Dict[str, int] = {}
    for art in articles:
        summary = (art.get("generated_summary") or "").lower()
        for phrase in GENERIC_FALLBACK_PHRASES:
            if phrase in summary:
                counts[phrase] = counts.get(phrase, 0) + 1

    n = max(len(articles), 1)
    threshold = max(2, int(n * threshold_pct))
    return [phrase for phrase, cnt in counts.items() if cnt >= threshold]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a medical information extraction system.

TASK: Extract structured evidence from the ARTICLE TEXT below.

STRICT CONSTRAINTS:
- Extract ONLY claims explicitly present in the provided text.
- For every extracted value, include a direct quote from the source text as "support".
- If a field has no support in the text, set value to null and support to [].
- Do not infer sample sizes, percentages, or study designs from context.
- Do not use background knowledge to fill any field.
- Classify article_type conservatively; use "unknown" if not clearly stated.
- source_text_quality: "snippet_only" if < 300 chars, "abstract" if 300-2000 chars, \
"full_text" if > 2000 chars.

ARTICLE TEXT:
<source>
{source_text}
</source>

Respond ONLY with a valid JSON object — no markdown, no explanation:
{{
  "article_type": "original_research|review|editorial|commentary|unknown",
  "source_text_quality": "full_text|abstract|snippet_only",
  "study_design": {{"value": null, "support": []}},
  "population": {{"value": null, "support": []}},
  "sample_size": {{"value": null, "support": []}},
  "intervention_or_exposure": {{"value": null, "support": []}},
  "comparison": {{"value": null, "support": []}},
  "primary_outcomes": [{{"value": null, "support": []}}],
  "key_findings": [{{"value": "...", "support": ["exact quote"], "contains_numeric_claim": false}}],
  "limitations": [{{"value": null, "support": []}}],
  "missing_fields": ["list of field names with no source support"],
  "confidence": "high|medium|low"
}}

For missing_fields, list every field that could not be supported by the source text.
Set confidence: "high" = full text with specific data; "medium" = structured abstract; \
"low" = title/snippet only.
"""

_GENERATION_PROMPT = """\
You are a conservative medical writer for a professional fertility medicine newsletter.

EVIDENCE OBJECT:
{evidence_json}

ARTICLE TYPE: {article_type}
SOURCE QUALITY: {source_text_quality}
CONFIDENCE: {confidence}

STRICT RULES:
1. Use ONLY information present in the evidence object. Do not add background knowledge.
2. Do not introduce any numbers, percentages, or statistics absent from the evidence object.
3. Do not use any study-design term (randomized, prospective, multicenter, etc.) unless it
   appears verbatim in the evidence object.
4. Do not make causal claims beyond what evidence explicitly states.
5. Do not frame a review, editorial, or commentary as original research.
6. If a field is null or listed in missing_fields, say so plainly: "Sample size was not
   reported in the available text."
7. Keep clinical implications modest and hedged ("may suggest", "consistent with",
   "warrants further research").
8. FORBIDDEN PHRASES — never write these unless they are exact quotes from the evidence:
   - "strong inverse relationship" / "strong positive relationship" / "strong association"
   - "could negatively impact" / "could positively impact"
   - "resulting in fewer transferable embryos" / "resulting in reduced"
   - "significantly impair/reduce/increase"
   - "dramatically reduce/increase"
   - "markedly improve/reduce"
   - "clearly demonstrates" / "definitively shows" / "unequivocally"
   If you cannot describe a finding without these phrases, instead use:
   - "the retrieved abstract describes an association between …"
   - "the available text suggests the article examines …"
   - "the relationship between X and Y is not fully characterised in the available text"

TONE: Professional, clinical, conservative.
- No first person ("I", "we", "our").
- No podcast/editorial voice ("today we're diving into", "spotlighting", "clarion call").
- No hype or overconfident clinical assertions.
- Do NOT restate the article title as if it were a finding.

FORMAT — respond with ONLY a JSON object, no markdown:
{{
  "what_it_studied": "One sentence describing the research question, evidence-bounded.",
  "what_it_found": "One to two sentences on key findings. Numbers only if in evidence. If findings are not available, write: 'Specific findings were not available in the retrieved text.'",
  "why_it_matters": "One cautious sentence. Avoid generic statements like 'the topic is relevant'; be specific to the article's population or exposure.",
  "caveats": "One sentence if evidence is incomplete, article type is not original_research, or limitations are known. Empty string otherwise."
}}
"""

_VERIFIER_PROMPT = """\
You are a fact-checking assistant for medical summaries.

EVIDENCE OBJECT:
{evidence_json}

SUMMARY TO VERIFY:
{summary}

For each sentence in the summary, determine if it is:
- "supported": directly backed by the evidence object
- "weakly_supported": plausible from evidence but not directly stated
- "unsupported": introduces content not found in, or contradicting, evidence

Respond ONLY with a valid JSON array, no markdown:
[
  {{"sentence": "...", "label": "supported|weakly_supported|unsupported", "reason": "..."}}
]
"""


# ---------------------------------------------------------------------------
# Heuristic helpers (no LLM needed)
# ---------------------------------------------------------------------------

def classify_source_quality(text: str) -> str:
    """Classify source text quality by length."""
    n = len(text or "")
    if n < SNIPPET_ONLY_THRESHOLD:
        return "snippet_only"
    if n < ABSTRACT_THRESHOLD:
        return "abstract"
    return "full_text"


def classify_article_type_heuristic(
    title: str, content: str, source: str = ""
) -> str:
    """
    Fast heuristic classification before sending to LLM.

    Applies three layers of evidence, in order of reliability:
    1. Source-name lookup (most reliable for known news sites / journals).
    2. Content/title keyword matching.
    3. Falls back to "unknown".
    """
    source_lower = (source or "").lower()
    for pattern, atype in _SOURCE_TYPE_MAP.items():
        if pattern in source_lower:
            return atype

    combined = (title + " " + (content or "")).lower()
    for atype, hints in _ARTICLE_TYPE_HINTS.items():
        if any(h in combined for h in hints):
            return atype
    return "unknown"


def _extract_numbers(text: str) -> List[str]:
    """Extract all number tokens (integers, decimals, percentages) from text."""
    return re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)


def _collect_evidence_numbers(evidence: EvidenceObject) -> set:
    """Collect all numbers mentioned anywhere in the evidence object."""
    corpus_parts = []
    for sf in [
        evidence.study_design, evidence.population, evidence.sample_size,
        evidence.intervention_or_exposure, evidence.comparison,
    ]:
        if sf.value:
            corpus_parts.append(sf.value)
        corpus_parts.extend(sf.support)
    for sf in evidence.primary_outcomes + evidence.limitations:
        if sf.value:
            corpus_parts.append(sf.value)
        corpus_parts.extend(sf.support)
    for kf in evidence.key_findings:
        corpus_parts.append(kf.value)
        corpus_parts.extend(kf.support)
    corpus = " ".join(corpus_parts)
    return set(_extract_numbers(corpus))


def _collect_evidence_design_terms(evidence: EvidenceObject) -> set:
    """
    Collect study-design terms explicitly present anywhere in the evidence object.

    We scan all text fields — study_design, key_findings, limitations,
    population, etc. — so that design-related language mentioned in limitations
    (e.g. "single-centre design") is correctly treated as evidenced.
    """
    parts: List[str] = []
    for sf in [
        evidence.study_design, evidence.population, evidence.sample_size,
        evidence.intervention_or_exposure, evidence.comparison,
    ]:
        if sf.value:
            parts.append(sf.value)
        parts.extend(sf.support)
    for sf in evidence.primary_outcomes + evidence.limitations:
        if sf.value:
            parts.append(sf.value)
        parts.extend(sf.support)
    for kf in evidence.key_findings:
        parts.append(kf.value)
        parts.extend(kf.support)
    all_text = " ".join(parts).lower()
    return {term for term in DESIGN_TERMS if term in all_text}


def _collect_all_evidence_text(evidence: EvidenceObject) -> str:
    """Return all text from every evidence field as a single string."""
    parts: List[str] = []
    for sf in [
        evidence.study_design, evidence.population, evidence.sample_size,
        evidence.intervention_or_exposure, evidence.comparison,
    ]:
        if sf.value:
            parts.append(sf.value)
        parts.extend(sf.support)
    for sf in evidence.primary_outcomes + evidence.limitations:
        if sf.value:
            parts.append(sf.value)
        parts.extend(sf.support)
    for kf in evidence.key_findings:
        parts.append(kf.value)
        parts.extend(kf.support)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Evidence sufficiency & tiering
# ---------------------------------------------------------------------------

def evidence_sufficiency_score(evidence: EvidenceObject) -> float:
    """
    Score 0.0 – 1.0 reflecting how much concrete, citable evidence the
    EvidenceObject contains.

    Scoring breakdown
    -----------------
    source_text_quality:  full_text +0.50 | abstract +0.30 | snippet_only 0
    population populated:              +0.05
    sample_size populated:             +0.10
    study_design populated:            +0.05
    intervention_or_exposure:          +0.05
    primary_outcomes non-empty:        +0.05
    key_findings non-empty:            +0.10
    key_findings contain numeric:      +0.05   (extra for concrete numbers)
    limitations non-empty:             +0.05
    """
    score = 0.0
    if evidence.source_text_quality == "full_text":
        score += 0.50
    elif evidence.source_text_quality == "abstract":
        score += 0.30
    # snippet_only → 0

    if evidence.population.value:
        score += 0.05
    if evidence.sample_size.value:
        score += 0.10
    if evidence.study_design.value:
        score += 0.05
    if evidence.intervention_or_exposure.value:
        score += 0.05
    if evidence.primary_outcomes:
        score += 0.05
    if evidence.key_findings:
        score += 0.10
        if any(kf.contains_numeric_claim for kf in evidence.key_findings):
            score += 0.05
    if evidence.limitations:
        score += 0.05

    return min(score, 1.0)


def evidence_tier_from_evidence(evidence: EvidenceObject) -> str:
    """
    Determine reporting tier from an already-extracted EvidenceObject.

    "full"           score ≥ 0.50 → 4-section template is appropriate.
    "short_blurb"    score ≥ 0.20 → 2-sentence compact blurb.
    "titles_to_watch" score < 0.20 → title + source only.
    """
    score = evidence_sufficiency_score(evidence)
    if score >= 0.50:
        return "full"
    if score >= 0.20:
        return "short_blurb"
    return "titles_to_watch"


def estimate_evidence_tier(article: Dict) -> str:
    """
    Pre-LLM heuristic tier estimate based on content length and title signals.
    Used during article scoring/ranking *before* OpenAI is called.

    "full"            content ≥ 300 chars AND heuristic type is research/review
    "short_blurb"     content 100-300 chars OR type is commentary/editorial
    "titles_to_watch" content < 100 chars
    """
    content = article.get("content") or ""
    title = article.get("title") or ""
    source = article.get("source") or ""
    n = len(content)

    if n < 100:
        return "titles_to_watch"

    atype = classify_article_type_heuristic(title, content, source)

    if n < SNIPPET_ONLY_THRESHOLD:
        # Very short content — only original_research with structure gets short_blurb
        return "short_blurb"

    # ≥ 300 chars
    if atype in ("editorial", "commentary"):
        return "short_blurb"

    return "full"


# Ranking penalty applied to impact_score based on pre-LLM tier estimate.
_TIER_SCORE_PENALTY: Dict[str, float] = {
    "full": 0.0,
    "short_blurb": -2.0,
    "titles_to_watch": -6.0,
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_numbers(summary: str, evidence: EvidenceObject) -> ValidationResult:
    """
    Reject summary if it contains any numbers absent from the evidence object.
    Pure Python — no LLM needed.
    """
    summary_numbers = set(_extract_numbers(summary))
    evidence_numbers = _collect_evidence_numbers(evidence)
    unsupported = summary_numbers - evidence_numbers
    if unsupported:
        return ValidationResult(
            passed=False,
            rule="number_validator",
            details=f"Summary contains {len(unsupported)} number(s) not found in evidence.",
            offending_items=sorted(unsupported),
        )
    return ValidationResult(passed=True, rule="number_validator", details="All numbers supported.")


def validate_design_claims(summary: str, evidence: EvidenceObject) -> ValidationResult:
    """
    Reject summary if it uses study-design terms absent from the evidence object.
    """
    supported_terms = _collect_evidence_design_terms(evidence)
    summary_lower = summary.lower()
    offending = [
        term for term in DESIGN_TERMS
        if term in summary_lower and term not in supported_terms
    ]
    if offending:
        return ValidationResult(
            passed=False,
            rule="design_validator",
            details=(
                f"Summary uses {len(offending)} study-design term(s) not present in evidence: "
                f"{offending}"
            ),
            offending_items=offending,
        )
    return ValidationResult(
        passed=True, rule="design_validator", details="No unsupported design terms."
    )


def validate_causal_claims(
    summary: str, evidence: EvidenceObject
) -> ValidationResult:
    """
    Warn when summary uses causal/recommendation language that is not warranted
    given article type and confidence.  Fails for low-confidence non-RCT articles.
    """
    summary_lower = summary.lower()
    found = [p for p in CAUSAL_PHRASES if p in summary_lower]
    if not found:
        return ValidationResult(
            passed=True, rule="causal_validator", details="No causal/recommendation phrases detected."
        )
    # Only hard-fail for low confidence or non-original-research
    should_fail = (
        evidence.confidence == "low"
        or evidence.article_type not in ("original_research", "unknown")
    )
    msg = (
        f"Causal/recommendation language detected: {found}. "
        f"Article type={evidence.article_type}, confidence={evidence.confidence}."
    )
    return ValidationResult(
        passed=not should_fail,
        rule="causal_validator",
        details=msg,
        offending_items=found,
    )


def validate_inference_language(
    summary: str, evidence: EvidenceObject
) -> ValidationResult:
    """
    Block speculative/over-interpreting phrases that are NOT present verbatim
    in the evidence corpus.

    Hard-fails for low and medium confidence.  For high confidence (full text)
    we still flag but do not block, since the phrase may be a direct quote.
    """
    summary_lower = summary.lower()
    ev_corpus = _collect_all_evidence_text(evidence).lower()

    offending = [
        phrase for phrase in SPECULATIVE_PHRASES
        if phrase.lower() in summary_lower
        and phrase.lower() not in ev_corpus
    ]

    if not offending:
        return ValidationResult(
            passed=True,
            rule="inference_language_validator",
            details="No unsupported speculative phrases detected.",
        )

    should_fail = evidence.confidence in ("low", "medium")
    return ValidationResult(
        passed=not should_fail,
        rule="inference_language_validator",
        details=(
            f"Summary uses {len(offending)} speculative phrase(s) not in evidence corpus: "
            f"{offending}"
        ),
        offending_items=offending,
    )


def validate_summary(summary: str, evidence: EvidenceObject) -> List[ValidationResult]:
    """Run all validators and return the full list of results."""
    return [
        validate_numbers(summary, evidence),
        validate_design_claims(summary, evidence),
        validate_causal_claims(summary, evidence),
        validate_inference_language(summary, evidence),
    ]


# ---------------------------------------------------------------------------
# Rule-based sentence verifier
# ---------------------------------------------------------------------------

def _sentences(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _strip_md_headers(text: str) -> str:
    """
    Remove Markdown section headers of the form '**Label**: ' from each line
    so that sentence-level matching works on content, not formatting.
    """
    return re.sub(r"\*\*[^*]+\*\*:\s*", "", text)


def _token_overlap_ratio(a: str, b: str, min_len: int = 3) -> float:
    """
    Return what fraction of *b*'s tokens of length ≥ min_len appear in *a*.
    Used for fuzzy key-finding matching without requiring exact substring.
    """
    a_tokens = set(re.findall(rf"\b\w{{{min_len},}}\b", a.lower()))
    b_tokens = set(re.findall(rf"\b\w{{{min_len},}}\b", b.lower()))
    if not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(b_tokens)


def verify_sentences_ruleset(
    summary: str, evidence: EvidenceObject
) -> List[SentenceLabel]:
    """
    Label each sentence as supported / weakly_supported / unsupported using
    pure-Python rules.  No LLM call needed.

    Markdown section headers (``**Label**: ``) are stripped before processing
    so that sentence text matches the plain field content stored in
    StructuredSummary, enabling apply_sentence_filter to work correctly.

    Rules (applied in order, first match wins):
    1. Sentence contains a number not in evidence → unsupported.
    2. Sentence uses a design term not in evidence → unsupported.
    3. Sentence has ≥ 60% token overlap with a key_finding.value → supported.
    4. Sentence mentions evidence population/intervention values → weakly_supported.
    5. Otherwise → weakly_supported (can't verify qualitative claims without LLM).
    """
    evidence_numbers = _collect_evidence_numbers(evidence)
    evidence_design_terms = _collect_evidence_design_terms(evidence)
    kf_texts = [kf.value for kf in evidence.key_findings if kf.value]
    pop_val = (evidence.population.value or "").lower()
    ivx_val = (evidence.intervention_or_exposure.value or "").lower()

    # Strip markdown headers so sentence text matches raw field content
    clean_summary = _strip_md_headers(summary)

    labels: List[SentenceLabel] = []
    for sent in _sentences(clean_summary):
        sent_lower = sent.lower()

        # Rule 1 – unsupported number
        sent_numbers = set(_extract_numbers(sent))
        unsupported_nums = sent_numbers - evidence_numbers
        if unsupported_nums:
            labels.append(SentenceLabel(
                sentence=sent,
                label="unsupported",
                reason=f"Contains number(s) not in evidence: {sorted(unsupported_nums)}",
            ))
            continue

        # Rule 2 – unsupported design term
        bad_terms = [
            t for t in DESIGN_TERMS
            if t in sent_lower and t not in evidence_design_terms
        ]
        if bad_terms:
            labels.append(SentenceLabel(
                sentence=sent,
                label="unsupported",
                reason=f"Uses design term(s) not in evidence: {bad_terms}",
            ))
            continue

        # Rule 3 – ≥ 60 % token overlap with a key finding
        if any(_token_overlap_ratio(sent_lower, kf) >= 0.60 for kf in kf_texts):
            labels.append(SentenceLabel(
                sentence=sent, label="supported",
                reason="Matches a key finding in the evidence object (≥ 60% token overlap).",
            ))
            continue

        # Rule 4 – mentions known population/intervention
        if (pop_val and pop_val in sent_lower) or (ivx_val and ivx_val in sent_lower):
            labels.append(SentenceLabel(
                sentence=sent, label="weakly_supported",
                reason="References population or intervention from evidence.",
            ))
            continue

        labels.append(SentenceLabel(
            sentence=sent, label="weakly_supported",
            reason="Qualitative claim; could not verify without full text.",
        ))

    return labels


def apply_sentence_filter(
    structured: StructuredSummary,
) -> StructuredSummary:
    """
    Remove unsupported sentences from the 4-section summary in-place.
    Returns the (possibly modified) StructuredSummary.
    """
    if not structured.sentence_labels:
        return structured

    supported_sentences = {
        lbl.sentence
        for lbl in structured.sentence_labels
        if lbl.label != "unsupported"
    }

    def _filter_field(text: str) -> str:
        sents = _sentences(text)
        kept = [s for s in sents if s in supported_sentences]
        return " ".join(kept).strip() if kept else text  # keep original if all would be removed

    structured.what_it_studied = _filter_field(structured.what_it_studied)
    structured.what_it_found = _filter_field(structured.what_it_found)
    structured.why_it_matters = _filter_field(structured.why_it_matters)
    if structured.caveats:
        structured.caveats = _filter_field(structured.caveats)

    return structured


# ---------------------------------------------------------------------------
# Contradiction detection & persistence
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS evidence_cache (
    article_id    TEXT PRIMARY KEY,
    url           TEXT,
    title         TEXT,
    evidence_json TEXT NOT NULL,
    summary_json  TEXT,
    sample_size   TEXT,
    study_design  TEXT,
    article_type  TEXT,
    confidence    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_DDL)
    conn.commit()
    return conn


def _normalize_for_comparison(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def check_contradictions(
    new_evidence: EvidenceObject, conn: sqlite3.Connection
) -> ContradictionResult:
    """
    Compare new_evidence against the last-stored version for the same article.
    Returns a ContradictionResult flagging material field-level changes.
    """
    if not new_evidence.article_id:
        return ContradictionResult(has_contradiction=False, details="No article_id; skipping.")

    row = conn.execute(
        "SELECT * FROM evidence_cache WHERE article_id = ?",
        (new_evidence.article_id,),
    ).fetchone()

    if row is None:
        return ContradictionResult(has_contradiction=False, details="No prior record found.")

    prior = EvidenceObject.from_dict(json.loads(row["evidence_json"]))

    changed: List[str] = []

    # Compare sample size
    if _normalize_for_comparison(prior.sample_size.value) != _normalize_for_comparison(
        new_evidence.sample_size.value
    ):
        changed.append(
            f"sample_size: '{prior.sample_size.value}' → '{new_evidence.sample_size.value}'"
        )

    # Compare study design
    if _normalize_for_comparison(prior.study_design.value) != _normalize_for_comparison(
        new_evidence.study_design.value
    ):
        changed.append(
            f"study_design: '{prior.study_design.value}' → '{new_evidence.study_design.value}'"
        )

    # Compare article type
    if prior.article_type != new_evidence.article_type:
        changed.append(
            f"article_type: '{prior.article_type}' → '{new_evidence.article_type}'"
        )

    # Compare first key finding (primary finding direction)
    prior_kf = prior.key_findings[0].value if prior.key_findings else ""
    new_kf = new_evidence.key_findings[0].value if new_evidence.key_findings else ""
    if _normalize_for_comparison(prior_kf) != _normalize_for_comparison(new_kf):
        changed.append(f"key_finding[0]: '{prior_kf}' → '{new_kf}'")

    # Compare numbers — flag if numeric evidence changed materially
    prior_nums = _collect_evidence_numbers(prior)
    new_nums = _collect_evidence_numbers(new_evidence)
    gained = new_nums - prior_nums
    lost = prior_nums - new_nums
    if gained or lost:
        changed.append(
            f"numeric_evidence: gained={sorted(gained)}, lost={sorted(lost)}"
        )

    if changed:
        return ContradictionResult(
            has_contradiction=True,
            fields_changed=changed,
            details=f"{len(changed)} field(s) changed across reruns.",
        )
    return ContradictionResult(has_contradiction=False, details="No material changes detected.")


def persist_evidence(
    evidence: EvidenceObject,
    summary: StructuredSummary,
    conn: sqlite3.Connection,
) -> None:
    """Upsert the evidence object and summary into the cache."""
    now = datetime.now(tz=timezone.utc).isoformat()
    evidence_json = json.dumps(evidence.to_dict(), ensure_ascii=False)
    summary_json = json.dumps(summary.to_dict(), ensure_ascii=False)

    existing = conn.execute(
        "SELECT article_id FROM evidence_cache WHERE article_id = ?",
        (evidence.article_id,),
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE evidence_cache
               SET url=?, title=?, evidence_json=?, summary_json=?,
                   sample_size=?, study_design=?, article_type=?,
                   confidence=?, updated_at=?
               WHERE article_id=?""",
            (
                evidence.url,
                evidence.title,
                evidence_json,
                summary_json,
                evidence.sample_size.value,
                evidence.study_design.value,
                evidence.article_type,
                evidence.confidence,
                now,
                evidence.article_id,
            ),
        )
    else:
        conn.execute(
            """INSERT INTO evidence_cache
               (article_id, url, title, evidence_json, summary_json,
                sample_size, study_design, article_type, confidence, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                evidence.article_id,
                evidence.url,
                evidence.title,
                evidence_json,
                summary_json,
                evidence.sample_size.value,
                evidence.study_design.value,
                evidence.article_type,
                evidence.confidence,
                now,
                now,
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Fallback blurb (template-based, no LLM)
# ---------------------------------------------------------------------------

_ARTICLE_TYPE_PHRASES = {
    "original_research": "a research article",
    "review":            "a review article",
    "editorial":         "an editorial",
    "commentary":        "a commentary piece",
    "unknown":           "an article",
}


def produce_fallback(evidence: EvidenceObject) -> StructuredSummary:
    """
    Generate a safe, template-based summary without calling the LLM.

    The output format depends on the evidence tier:

    "full" / "short_blurb"
        Two-sentence blurb naming the population/intervention when available,
        plus an honest caveat about missing detail.

    "titles_to_watch"
        A single compact line: title — article type from source.
        No findings section; the article belongs in a "Titles to Watch" list.
    """
    suf = evidence_sufficiency_score(evidence)
    tier = evidence_tier_from_evidence(evidence)

    type_phrase = _ARTICLE_TYPE_PHRASES.get(evidence.article_type, "an article")
    pop = evidence.population.value
    ivx = evidence.intervention_or_exposure.value
    journal = evidence.journal or "an unspecified source"

    # Build the "what it studied" sentence, using available metadata
    if pop and ivx:
        studied = (
            f"The available text describes {type_phrase} examining {ivx} "
            f"in relation to {pop}."
        )
    elif pop:
        studied = (
            f"The available text describes {type_phrase} involving {pop}."
        )
    elif ivx:
        studied = (
            f"The available text describes {type_phrase} examining {ivx}."
        )
    else:
        topic = evidence.title or "a fertility or reproductive medicine topic"
        studied = (
            f"The available text describes {type_phrase} from {journal} "
            f"on the topic: {topic}."
        )

    if tier == "titles_to_watch":
        title_line = evidence.title or "Untitled article"
        compact = (
            f"{title_line} — {type_phrase} from {journal}."
        )
        return StructuredSummary(
            what_it_studied=compact,
            tier="titles_to_watch",
            evidence=evidence,
            is_fallback=True,
            evidence_sufficiency=suf,
        )

    # short_blurb (or full-tier where LLM failed and we fall back)
    caveats = (
        "Detailed findings were not available in the retrieved text; "
        "consult the primary source for methods and results."
    )
    return StructuredSummary(
        what_it_studied=studied,
        what_it_found="",
        why_it_matters="",
        caveats=caveats,
        tier="short_blurb",
        evidence=evidence,
        is_fallback=True,
        evidence_sufficiency=suf,
    )


# ---------------------------------------------------------------------------
# LLM caller (OpenAI) — imported from llm_caller.py
# ---------------------------------------------------------------------------
# _default_llm_caller is already imported at the top of this file as:
#   from llm_caller import call_openai as _default_llm_caller


def _parse_json_response(raw: Optional[str], context: str) -> Optional[dict]:
    """
    Parse a JSON object from a potentially noisy LLM response.
    Strips markdown fences, leading/trailing text.
    """
    if not raw:
        return None
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in %s response", context)
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in %s response: %s", context, exc)
        return None


# ---------------------------------------------------------------------------
# Stage A – Evidence extraction
# ---------------------------------------------------------------------------

def extract_evidence(
    article: Dict,
    config: Dict,
    llm_caller: Callable = _default_llm_caller,
) -> EvidenceObject:
    """
    Stage A: send a constrained extraction prompt to the LLM and return a
    structured EvidenceObject.  Falls back to a heuristic-only object if the
    LLM call fails or returns unparseable output.
    """
    title = article.get("title") or ""
    content = article.get("content") or ""
    url = article.get("url") or ""
    source = article.get("source") or ""
    pub_date = (
        article["published_date"].strftime("%Y-%m-%d")
        if article.get("published_date")
        else None
    )

    # Heuristic pre-fill (no LLM needed)
    heuristic_type = classify_article_type_heuristic(title, content, source)
    source_quality = classify_source_quality(content)

    # Build article_id deterministically
    article_id = url or title

    base = EvidenceObject(
        article_id=article_id,
        url=url,
        title=title,
        journal=source,
        published_date=pub_date,
        article_type=heuristic_type,
        source_text_quality=source_quality,
        confidence="low" if source_quality == "snippet_only" else "medium",
    )

    # If source quality is too low, skip LLM extraction entirely
    if source_quality == "snippet_only" and len(content) < 100:
        base.missing_fields = [
            "study_design", "population", "sample_size",
            "intervention_or_exposure", "comparison",
            "primary_outcomes", "key_findings", "limitations",
        ]
        return base

    source_text = f"Title: {title}\n\n{content}"
    prompt = _EXTRACTION_PROMPT.format(source_text=source_text)
    raw = llm_caller(prompt, config)
    parsed = _parse_json_response(raw, "evidence extraction")

    if parsed is None:
        logger.warning("Evidence extraction failed for '%s'; using heuristic.", title[:60])
        base.missing_fields = [
            "study_design", "population", "sample_size",
            "key_findings", "limitations",
        ]
        return base

    # Merge LLM output onto base object
    obj = EvidenceObject.from_dict(parsed)
    obj.article_id = article_id
    obj.url = url
    obj.title = title
    obj.journal = source
    obj.published_date = pub_date
    # Trust heuristic type when LLM returns "unknown" and heuristic is confident
    if obj.article_type == "unknown" and heuristic_type != "unknown":
        obj.article_type = heuristic_type

    return obj


# ---------------------------------------------------------------------------
# Stage B – Prose generation
# ---------------------------------------------------------------------------

def generate_from_evidence(
    evidence: EvidenceObject,
    config: Dict,
    llm_caller: Callable = _default_llm_caller,
    hint: str = "",
) -> StructuredSummary:
    """
    Stage B: generate a StructuredSummary from the EvidenceObject.

    Evidence tier determines whether a full 4-section template is generated
    or a template-based fallback is used instead:

    "full"             → LLM generation with constrained prompt.
    "short_blurb"      → produce_fallback() (short, no LLM).
    "titles_to_watch"  → produce_fallback() (single-line, no LLM).

    Parameters
    ----------
    hint:
        Optional extra instruction appended to the prompt, used on regeneration
        attempts when the first pass produced a generic summary.
    """
    suf = evidence_sufficiency_score(evidence)
    tier = evidence_tier_from_evidence(evidence)

    if tier != "full":
        return produce_fallback(evidence)

    evidence_json = json.dumps(evidence.to_dict(), indent=2, ensure_ascii=False)
    prompt = _GENERATION_PROMPT.format(
        evidence_json=evidence_json,
        article_type=evidence.article_type,
        source_text_quality=evidence.source_text_quality,
        confidence=evidence.confidence,
    )
    if hint:
        prompt += f"\n\nADDITIONAL INSTRUCTION: {hint}"

    raw = llm_caller(prompt, config)
    parsed = _parse_json_response(raw, "prose generation")

    if parsed is None:
        logger.warning("Prose generation failed for '%s'; using fallback.", evidence.title[:60])
        return produce_fallback(evidence)

    return StructuredSummary(
        what_it_studied=parsed.get("what_it_studied") or "",
        what_it_found=parsed.get("what_it_found") or "",
        why_it_matters=parsed.get("why_it_matters") or "",
        caveats=parsed.get("caveats") or "",
        evidence=evidence,
        tier="full",
        evidence_sufficiency=suf,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def summarize_article(
    article: Dict,
    config: Dict,
    db_path: Optional[str] = None,
    llm_caller: Callable = _default_llm_caller,
) -> StructuredSummary:
    """
    Full evidence-bounded summarization pipeline:

    1. Extract evidence (Stage A)
    2. Generate prose from evidence (Stage B) — or fallback
    3. Validate the generated summary
    4. Verify sentences and filter unsupported ones
    5. Check for contradictions against prior runs
    6. Persist evidence and summary to cache
    7. Return StructuredSummary

    Parameters
    ----------
    article:
        Article dict with keys: title, url, content, source, published_date, authors.
    config:
        Config dict (from config.fertility.json).
    db_path:
        Path to the SQLite evidence cache.  Contradiction checks are skipped when None.
    llm_caller:
        Injectable LLM caller for testability.  Defaults to the shared OpenAI caller.
    """
    title = article.get("title", "")[:80]

    # Stage A
    logger.debug("Stage A: extracting evidence for '%s'", title)
    evidence = extract_evidence(article, config, llm_caller)

    # Stage B (or fallback)
    logger.debug("Stage B: generating prose for '%s' (confidence=%s)", title, evidence.confidence)
    summary = generate_from_evidence(evidence, config, llm_caller)

    if not summary.is_fallback:
        prose = summary.to_prose()

        # Validators
        summary.validation_results = validate_summary(prose, evidence)
        failed = [v for v in summary.validation_results if not v.passed]
        if failed:
            logger.warning(
                "Validation failed for '%s': %s",
                title,
                [v.rule for v in failed],
            )
            # Hard-block validators: number, design, inference language
            hard_fails = [
                v for v in failed
                if v.rule in (
                    "number_validator", "design_validator", "inference_language_validator"
                )
            ]
            if hard_fails:
                logger.warning("Hard validator failure → switching to fallback for '%s'", title)
                summary = produce_fallback(evidence)
                summary.validation_results = validate_summary(summary.to_prose(), evidence)

        # Sentence verifier + filter
        if not summary.is_fallback:
            summary.sentence_labels = verify_sentences_ruleset(prose, evidence)
            summary = apply_sentence_filter(summary)

    # Genericness check — try to regenerate once if evidence supports it
    if not summary.is_fallback:
        gen_result = detect_genericness(summary, article)
        if gen_result.is_generic:
            logger.warning(
                "Generic summary detected for '%s': %s",
                title, gen_result.reasons,
            )
            if gen_result.can_regenerate:
                logger.info("Attempting targeted regeneration for '%s'", title)
                retry = generate_from_evidence(
                    evidence, config, llm_caller,
                    hint=(
                        "The previous attempt was too generic. "
                        "Focus on specific findings extracted in the evidence object. "
                        "Do not just restate the title. "
                        "If 'what_it_found' cannot be made specific, write "
                        "'Specific findings were not available in the retrieved text.'"
                    ),
                )
                if not detect_genericness(retry, article).is_generic:
                    summary = retry
                    logger.info("Regeneration improved specificity for '%s'", title)
                else:
                    logger.warning("Regeneration still generic; downgrading '%s'", title)
                    summary = produce_fallback(evidence)
                    summary.tier = gen_result.suggested_tier
            else:
                summary = produce_fallback(evidence)
                summary.tier = gen_result.suggested_tier

    # Contradiction detection + persistence
    if db_path:
        try:
            conn = _get_conn(db_path)
            summary.contradiction = check_contradictions(evidence, conn)
            if summary.contradiction.has_contradiction:
                logger.warning(
                    "Contradiction detected for '%s': %s",
                    title,
                    summary.contradiction.fields_changed,
                )
            persist_evidence(evidence, summary, conn)
            conn.close()
        except Exception as exc:
            logger.warning("Evidence cache error for '%s': %s", title, exc)

    return summary
