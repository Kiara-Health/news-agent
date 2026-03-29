"""
Newsletter Composer
===================
Second-stage pipeline that consumes the validated article report and transforms
it into a curated, readable newsletter.

Data flow
---------
    [consolidated_report.json]  (output of podcast_generator / summarizer)
           ↓
    newsletter_worthiness_score()   — evidence + specificity + relevance
           ↓
    route_to_tier()                 — featured / briefs / watchlist
           ↓
    section builders                — clean prose, no internal-report phrases
           ↓
    _generate_editor_note()         — constrained LLM (topic-only, no claims)
    _generate_closing_takeaway()    — constrained LLM (topic-only, no claims)
           ↓
    [newsletter.md / newsletter.json]

Rules enforced throughout
-------------------------
* Newsletter copy never contains internal-report phrasing ("The available text
  describes…", "This appears to be…", "Detailed findings were not available…").
* Featured stories must have at least one concrete validated finding.
* Snippet-only / titles_to_watch evidence stays in Watchlist.
* The LLM editorial passes receive only article titles and topic labels —
  they may not access raw article text or add new claims.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from llm_caller import call_openai as _call_openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURED_THRESHOLD = 0.60    # newsletter_worthiness_score required for featured
BRIEFS_THRESHOLD   = 0.35    # minimum score for briefs section

# Phrases acceptable in the internal evidence report but FORBIDDEN in newsletter copy.
INTERNAL_REPORT_PHRASES: List[str] = [
    "the available text describes",
    "the available text suggests",
    "this appears to be",
    "detailed findings were not available",
    "only limited metadata was available",
    "this blurb does not reflect",
    "not available in the retrieved",
    "not reported in available text",
    "readers are encouraged to consult the full publication",
    "the topic is relevant to fertility and reproductive medicine",
    "the topic is relevant to fertility",
    "consult the primary source",
    "specific findings were not available",
    "findings were not available in",
    "consult the full publication",
]

_TYPE_LABELS: Dict[str, str] = {
    "original_research": "study",
    "review":            "review",
    "editorial":         "editorial",
    "commentary":        "commentary piece",
    "unknown":           "article",
}

# Source attributions varied by position (verb comes from the original sentence).
_SOURCE_PHRASES: List[str] = [
    "published in {source}",
    "appearing in {source}",
    "from {source}",
    "reported in {source}",
    "in {source}",
    "out of {source}",
]

# ---------------------------------------------------------------------------
# Newsletter-worthiness scoring
# ---------------------------------------------------------------------------

def _has_concrete_finding(summary_sections: Dict) -> bool:
    """
    Return True if 'what_it_found' contains a real finding that is not a
    fallback phrase and is not just a restatement of the title.
    """
    found = (summary_sections.get("what_it_found") or "").strip()
    if not found or len(found) < 30:
        return False
    found_lower = found.lower()
    for phrase in INTERNAL_REPORT_PHRASES:
        if phrase in found_lower:
            return False
    return True


def _summary_specificity(summary_sections: Dict, title: str = "") -> float:
    """
    0.0 = entirely generic or fallback.
    1.0 = specific, varied, non-repetitive.
    """
    texts = [
        summary_sections.get("what_it_studied") or "",
        summary_sections.get("what_it_found") or "",
        summary_sections.get("why_it_matters") or "",
    ]
    combined = " ".join(texts).lower()
    total_chars = sum(len(t) for t in texts)

    if not combined.strip():
        return 0.0

    # Penalise internal-report phrases
    if any(phrase in combined for phrase in INTERNAL_REPORT_PHRASES):
        return 0.15

    # Penalise if 'what_it_found' closely overlaps with the title
    title_tokens = set(re.findall(r"\b\w{4,}\b", title.lower()))
    found_tokens = set(re.findall(r"\b\w{4,}\b",
                                  (summary_sections.get("what_it_found") or "").lower()))
    if title_tokens and found_tokens:
        overlap = len(title_tokens & found_tokens) / max(len(title_tokens), len(found_tokens))
        if overlap >= 0.70:
            return 0.25

    # Length proxy for richness
    if total_chars >= 300:
        return 1.0
    if total_chars >= 150:
        return 0.70
    return 0.45


def newsletter_worthiness_score(record: Dict) -> Dict:
    """
    Composite newsletter-worthiness score (0.0 – 1.0) with sub-scores.

    Weights
    -------
    evidence_sufficiency   0.30
    has_concrete_finding   0.25
    article_type_conf      0.15
    summary_specificity    0.15
    reader_relevance       0.15

    Returns a dict with 'score' and all sub-scores for diagnostics.
    """
    eq = record.get("evidence_quality") or {}
    ss = record.get("summary_sections") or {}

    # 1. Evidence sufficiency
    ev_suf_raw = record.get("evidence_sufficiency", 0.0)
    confidence_map = {"high": 1.0, "medium": 0.65, "low": 0.15}
    confidence = eq.get("confidence", "low")
    conf_score = confidence_map.get(confidence, 0.15)
    suf_score = max(float(ev_suf_raw), conf_score)

    # 2. Concrete finding
    has_finding = _has_concrete_finding(ss)

    # 3. Article-type confidence
    type_conf_map = {
        "original_research": 1.0,
        "review":            0.80,
        "unknown":           0.50,
        "commentary":        0.35,
        "editorial":         0.30,
    }
    art_type = eq.get("article_type") or record.get("evidence_tier") or "unknown"
    type_score = type_conf_map.get(art_type, 0.50)

    # 4. Summary specificity
    spec_score = _summary_specificity(ss, record.get("title") or "")

    # 5. Reader relevance (audience_relevance from report, normalised to 0–1)
    rel_raw = record.get("audience_relevance", 0.0)
    rel_score = min(float(rel_raw) / 10.0, 1.0)

    composite = (
        0.30 * suf_score
        + 0.25 * float(has_finding)
        + 0.15 * type_score
        + 0.15 * spec_score
        + 0.15 * rel_score
    )
    composite = round(min(composite, 1.0), 3)

    return {
        "score":              composite,
        "evidence_suf":       round(suf_score, 3),
        "has_finding":        has_finding,
        "type_confidence":    round(type_score, 3),
        "specificity":        round(spec_score, 3),
        "reader_relevance":   round(rel_score, 3),
    }


# ---------------------------------------------------------------------------
# Tier routing
# ---------------------------------------------------------------------------

def route_to_tier(record: Dict, score_details: Dict) -> str:
    """
    Route an article record to "featured", "briefs", or "watchlist".

    Rules (applied in order)
    ------------------------
    1. featured requires: score ≥ FEATURED_THRESHOLD AND concrete finding
       AND evidence_tier == "full" AND not is_fallback.
    2. briefs requires: score ≥ BRIEFS_THRESHOLD AND not is_fallback.
    3. Everything else → watchlist.
    """
    nw_score = score_details.get("score", 0.0)
    has_finding = score_details.get("has_finding", False)
    is_fallback = (record.get("evidence_quality") or {}).get("is_fallback", True)
    evidence_tier = record.get("evidence_tier", "titles_to_watch")

    if (
        nw_score >= FEATURED_THRESHOLD
        and has_finding
        and evidence_tier == "full"
        and not is_fallback
    ):
        return "featured"

    if nw_score >= BRIEFS_THRESHOLD and not is_fallback:
        return "briefs"

    return "watchlist"


# ---------------------------------------------------------------------------
# Newsletter prose cleaners
# ---------------------------------------------------------------------------

def _clean_for_newsletter(text: str) -> Optional[str]:
    """
    Clean a validated summary field for newsletter use.

    Returns None if the text is empty, entirely fallback phrasing, or
    consists only of markdown structural formatting.
    """
    if not text or not text.strip():
        return None
    text_lower = text.lower()
    for phrase in INTERNAL_REPORT_PHRASES:
        if phrase in text_lower:
            return None
    # Strip markdown bold section headers (**Label**: )
    cleaned = re.sub(r"\*\*[^*]+\*\*:\s*", "", text).strip()
    return cleaned if cleaned else None


def _normalize_opening_sentence(
    studied: str, source: str, art_type: str, position: int
) -> str:
    """
    Replace a generic subject ("The study", "This paper", "Investigators") with a
    varied, source-attributed equivalent.  The original verb is kept so the predicate
    reads naturally — no duplicate verbs are introduced.

    Examples
    --------
    "The study examined X in Y."
        → "A study published in <source> examined X in Y."
    "This review investigated the relationship between A and B."
        → "A review appearing in <source> investigated the relationship between A and B."
    "Investigators identified 21 ClinGen‑curated genes…"
        → "Investigators reported in <source> identified 21 ClinGen‑curated genes…"
    """
    type_label = _TYPE_LABELS.get(art_type, "study")
    src = source or "the journal"
    src_phrase = _SOURCE_PHRASES[(position - 1) % len(_SOURCE_PHRASES)].format(source=src)

    # Map of (pattern, replacement_subject) — verb is kept from the original sentence.
    _SUBJECT_MAP = [
        (r"^The study\b",     f"A study {src_phrase}"),
        (r"^This study\b",    f"A new study {src_phrase}"),
        (r"^The paper\b",     f"A paper {src_phrase}"),
        (r"^This paper\b",    f"A {type_label} {src_phrase}"),
        (r"^The article\b",   f"An article {src_phrase}"),
        (r"^This article\b",  f"A {type_label} {src_phrase}"),
        (r"^The research\b",  f"Research {src_phrase}"),
        (r"^This research\b", f"Research {src_phrase}"),
        (r"^The analysis\b",  f"An analysis {src_phrase}"),
        (r"^This analysis\b", f"An analysis {src_phrase}"),
        (r"^The review\b",    f"A review {src_phrase}"),
        (r"^This review\b",   f"A review {src_phrase}"),
        (r"^The trial\b",     f"A trial {src_phrase}"),
        (r"^A study\b",       f"A study {src_phrase}"),
        (r"^A paper\b",       f"A {type_label} {src_phrase}"),
        (r"^Investigators\b", f"Investigators reporting {src_phrase}"),
        (r"^Researchers\b",   f"Researchers {src_phrase}"),
        (r"^Authors\b",       f"Authors reporting {src_phrase}"),
    ]
    for pattern, replacement in _SUBJECT_MAP:
        new_text = re.sub(pattern, replacement, studied, flags=re.IGNORECASE)
        if new_text != studied:
            return new_text

    # No generic subject found — prepend a minimal attribution before the sentence
    return f"A {type_label} {src_phrase}: {studied[0].lower()}{studied[1:]}"


def _group_similar_watchlist_entries(records: List[Dict]) -> List[Dict | str]:
    """
    Detect records with nearly identical fallback blurbs and return them as-is,
    but annotate heavily repeated ones so the watchlist formatter can group them.
    This avoids the newsletter showing the same sentence five times.
    """
    # Count how many records share the same first 60 chars of their summary
    snippets: Counter = Counter()
    for rec in records:
        snippet = (rec.get("summary") or "")[:60].strip().lower()
        if snippet:
            snippets[snippet] += 1

    # Mark records whose opening is shared by ≥ 3 others
    marked: List[Dict] = []
    for rec in records:
        snippet = (rec.get("summary") or "")[:60].strip().lower()
        rec = dict(rec)
        rec["_grouped_fallback"] = snippets.get(snippet, 1) >= 3
        marked.append(rec)
    return marked


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_featured_prose(record: Dict, position: int) -> Dict:
    """
    Build a featured story dict ready for newsletter rendering.

    Returns
    -------
    {title, body, caveat, source_line, url, nw_score}
    """
    ss = record.get("summary_sections") or {}
    eq = record.get("evidence_quality") or {}
    title = record.get("title") or ""
    source = record.get("source") or ""
    pub_date = record.get("published_date") or ""
    url = record.get("url") or ""
    art_type = eq.get("article_type") or "unknown"

    studied_raw = _clean_for_newsletter(ss.get("what_it_studied") or "")
    found_raw   = _clean_for_newsletter(ss.get("what_it_found") or "")
    matters_raw = _clean_for_newsletter(ss.get("why_it_matters") or "")
    caveats_raw = _clean_for_newsletter(ss.get("caveats") or "")

    # Build the opening sentence with source attribution and varied structure
    if studied_raw:
        opening = _normalize_opening_sentence(studied_raw, source, art_type, position)
    else:
        opening = ""

    # Assemble body — only include non-None parts
    parts = [p for p in [opening, found_raw, matters_raw] if p]
    body = " ".join(parts).strip()

    # Caveat note, formatted as italics in Markdown
    caveat = f"*Note: {caveats_raw}*" if caveats_raw else ""

    # Source attribution line
    source_parts = [p for p in [source, pub_date] if p]
    source_line = " · ".join(source_parts)

    return {
        "position":    position,
        "title":       title,
        "body":        body,
        "caveat":      caveat,
        "source_line": source_line,
        "url":         url,
        "nw_score":    record.get("_nw_score", 0.0),
    }


def _build_brief_text(record: Dict) -> Dict:
    """
    Build a brief item (one or two clean sentences).

    Prefers 'what_it_found'; falls back to 'what_it_studied'.
    """
    ss = record.get("summary_sections") or {}
    title  = record.get("title") or ""
    source = record.get("source") or ""
    url    = record.get("url") or ""

    text = (
        _clean_for_newsletter(ss.get("what_it_found") or "")
        or _clean_for_newsletter(ss.get("what_it_studied") or "")
        or ""
    )

    # Trim to two sentences max for briefs
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    brief_text = " ".join(sentences[:2]).strip()

    return {
        "title":  title,
        "text":   brief_text,
        "source": source,
        "url":    url,
    }


def _build_watchlist_entry(record: Dict) -> Dict:
    """
    Build a watchlist entry (title + source + optional single-sentence context).
    """
    eq       = record.get("evidence_quality") or {}
    title    = record.get("title") or ""
    source   = record.get("source") or ""
    pub_date = record.get("published_date") or ""
    url      = record.get("url") or ""
    art_type = eq.get("article_type") or "unknown"
    grouped  = record.get("_grouped_fallback", False)

    type_label = _TYPE_LABELS.get(art_type, "article")
    date_str   = f" ({pub_date})" if pub_date else ""

    # For grouped fallbacks (identical blurbs), provide no sentence at all
    context = "" if grouped else _build_watchlist_context(record)

    return {
        "title":      title,
        "type_label": type_label,
        "source":     source,
        "date":       pub_date,
        "url":        url,
        "context":    context,
        "grouped":    grouped,
    }


def _build_watchlist_context(record: Dict) -> str:
    """
    Generate a single-sentence context line for a watchlist entry, using only
    the article type and topic — no findings, no fallback phrases.
    """
    eq      = record.get("evidence_quality") or {}
    topic   = record.get("topic") or ""
    art_type = eq.get("article_type") or "unknown"
    type_label = _TYPE_LABELS.get(art_type, "article")

    if topic and topic not in ("general", "unknown"):
        topic_clean = topic.replace("_", " ")
        return f"A {type_label} on {topic_clean}."
    return ""


# ---------------------------------------------------------------------------
# LLM editorial helpers
# ---------------------------------------------------------------------------

_EDITOR_NOTE_PROMPT = """\
You are writing the editor's note for a weekly fertility medicine newsletter.

This week's highlighted topics:
{topics}

Featured article titles:
{titles}

Write 2-3 sentences that:
- Introduce the main themes across this week's selected articles.
- Use a professional, editorial tone — not "I", no hype phrases.
- Do NOT introduce specific statistics, study findings, or clinical claims.
- Do NOT begin with "This week" or "Welcome".
- Address an audience of fertility clinicians, researchers, and informed patients.

Respond with ONLY the paragraph text, no labels, no preamble.
"""

_CLOSING_PROMPT = """\
You are writing the closing paragraph for a weekly fertility medicine newsletter.

This week's covered topics:
{topics}

Write 2-3 sentences that:
- Reflect on cross-cutting themes or open questions in the evidence this week.
- Use a forward-looking, professionally measured tone.
- Do NOT introduce specific statistics or study conclusions.
- Invite readers to follow developments or share feedback.

Respond with ONLY the paragraph text, no labels, no preamble.
"""


def _call_llm_editorial(
    prompt: str, config: Dict, timeout: int = 60
) -> Optional[str]:
    """Call OpenAI for short editorial text (editor's note, closing paragraph)."""
    return _call_openai(prompt, config, timeout=timeout)


def _build_topic_summary(articles: List[Dict]) -> str:
    """Build a concise bullet list of topics from the selected articles."""
    topics = []
    seen: set = set()
    for art in articles:
        topic = (art.get("topic") or "").replace("_", " ")
        title = (art.get("title") or "")[:80]
        if topic and topic not in seen and topic not in ("general", "unknown"):
            topics.append(f"- {topic}")
            seen.add(topic)
        elif title and title not in seen:
            topics.append(f"- {title}")
            seen.add(title)
    return "\n".join(topics[:8])


def _generate_editor_note(
    articles: List[Dict],
    config: Dict,
    llm_caller: Optional[Callable] = None,
) -> str:
    """
    Generate the editor's note.

    Uses the provided LLM caller (or the default OpenAI HTTP caller) if available.
    Falls back to a clean template if LLM is unreachable.
    """
    topics = _build_topic_summary(articles)
    titles = "\n".join(
        f"- {(a.get('title') or '')[:80]}" for a in articles[:6]
    )
    prompt = _EDITOR_NOTE_PROMPT.format(topics=topics, titles=titles)

    caller = llm_caller or _call_llm_editorial
    raw = caller(prompt, config)
    if raw and _is_acceptable_editorial(raw):
        return raw.strip()

    # Template fallback — does not mention any specific findings
    topic_list = ", ".join(
        (a.get("topic") or "").replace("_", " ")
        for a in articles[:4]
        if a.get("topic") and a.get("topic") not in ("general", "unknown")
    )
    pub = config.get("publication_settings") or {}
    audience = pub.get("audience_description", "clinicians and patients")
    area = pub.get("topics_label", "fertility and reproductive medicine")
    intro = (
        f"This week's edition covers recent developments across {topic_list or area}, "
        f"drawing on published research and specialist sources. "
        f"As always, the summaries reflect retrieved evidence and are intended for "
        f"{audience}."
    )
    return intro


def _generate_closing_takeaway(
    articles: List[Dict],
    config: Dict,
    llm_caller: Optional[Callable] = None,
) -> str:
    """
    Generate the closing paragraph.

    Uses OpenAI if available; otherwise returns a template closing.
    """
    topics = _build_topic_summary(articles)
    prompt = _CLOSING_PROMPT.format(topics=topics)

    caller = llm_caller or _call_llm_editorial
    raw = caller(prompt, config)
    if raw and _is_acceptable_editorial(raw):
        return raw.strip()

    # Template fallback
    pub = config.get("publication_settings") or {}
    area = pub.get("topics_label", "fertility and reproductive medicine")
    return (
        f"The research landscape in {area} continues to evolve, with emerging "
        f"findings across multiple fronts. "
        f"Readers are encouraged to follow primary publications for full methodology "
        f"and to consider findings in the context of individual clinical situations. "
        f"Feedback and topic suggestions are always welcome."
    )


def _is_acceptable_editorial(text: str) -> bool:
    """
    Basic sanity check: reject LLM editorial output that slipped through with
    forbidden phrases or is clearly malformed.
    """
    if len(text) < 30:
        return False
    text_lower = text.lower()
    hard_rejects = [
        "the available text", "this appears to be", "detailed findings",
        "not available in the retrieved", "{topics}", "{titles}",
    ]
    return not any(p in text_lower for p in hard_rejects)


# ---------------------------------------------------------------------------
# Newsletter renderer (Markdown)
# ---------------------------------------------------------------------------

def _render_newsletter_md(newsletter: Dict) -> str:
    """Render the structured newsletter dict as a Markdown string."""
    lines: List[str] = []
    title = newsletter.get("title") or "Weekly Fertility Medicine Newsletter"
    date  = newsletter.get("date") or ""
    sep   = "---"

    lines += [f"# {title}", f"*{date}*", "", sep, ""]

    # Editor's note
    editor_note = newsletter.get("editor_note") or ""
    if editor_note:
        lines += ["## Editor's Note", "", editor_note, "", sep, ""]

    # Featured stories
    featured = newsletter.get("featured") or []
    if featured:
        lines += ["## Featured Stories", ""]
        for story in featured:
            n     = story.get("position", "")
            ftitle = story.get("title") or ""
            body  = story.get("body") or ""
            caveat = story.get("caveat") or ""
            src   = story.get("source_line") or ""
            url   = story.get("url") or ""

            if url:
                lines.append(f"### {n}. [{ftitle}]({url})")
            else:
                lines.append(f"### {n}. {ftitle}")
            lines.append("")
            if body:
                lines.append(body)
            if caveat:
                lines += ["", caveat]
            if src:
                lines += ["", f"*{src}*"]
            lines += ["", sep, ""]

    # Briefs
    briefs = newsletter.get("briefs") or []
    if briefs:
        lines += ["## Briefs", ""]
        for b in briefs:
            btitle = b.get("title") or ""
            btext  = b.get("text") or ""
            bsrc   = b.get("source") or ""
            burl   = b.get("url") or ""

            title_part = f"[{btitle}]({burl})" if burl else btitle
            src_part   = f" *({bsrc})*" if bsrc else ""
            text_part  = f" {btext}" if btext else ""
            lines.append(f"**{title_part}**{src_part}:{text_part}")
            lines.append("")
        lines += [sep, ""]

    # Watchlist
    watchlist = newsletter.get("watchlist") or []
    if watchlist:
        lines += [
            "## Watchlist",
            "",
            "*Topics that surfaced this week with limited available detail. "
            "Consult primary sources for methods and results.*",
            "",
        ]
        for w in watchlist:
            wtitle  = w.get("title") or ""
            wsource = w.get("source") or ""
            wdate   = w.get("date") or ""
            wurl    = w.get("url") or ""
            context = w.get("context") or ""

            title_part = f"[{wtitle}]({wurl})" if wurl else wtitle
            meta_parts = [p for p in [wsource, wdate] if p]
            meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
            context_part = f" — {context}" if context else ""
            lines.append(f"- {title_part}{meta}{context_part}")
        lines += ["", sep, ""]

    # Closing
    closing = newsletter.get("closing") or ""
    if closing:
        lines += ["## Closing", "", closing, "", sep, ""]

    # Disclaimer footer
    lines += [
        "",
        "*This newsletter is compiled from published research and specialist sources. "
        "Summaries reflect retrieved evidence and should not be used as clinical guidance.*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main composition entry point
# ---------------------------------------------------------------------------

def compose_newsletter(
    articles: List[Dict],
    config: Dict,
    output_txt: Optional[str] = None,
    output_json: Optional[str] = None,
    llm_caller: Optional[Callable] = None,
) -> Dict:
    """
    Compose a curated newsletter from validated article records.

    Parameters
    ----------
    articles:
        List of article dicts as produced by ``PodcastGenerator.save_consolidated_report``
        (i.e. each dict contains ``summary_sections``, ``evidence_quality``,
        ``evidence_tier``, ``evidence_sufficiency``, etc.).
    config:
        Pipeline config dict (used for publication_settings and OpenAI config).
    output_txt:
        If provided, the rendered Markdown is written to this path.
    output_json:
        If provided, the structured newsletter dict is written to this path.
    llm_caller:
        Optional injectable LLM caller (for testing or model switching).
        Defaults to the shared OpenAI caller.

    Returns
    -------
    dict with keys: title, date, editor_note, featured, briefs, watchlist,
    closing, article_counts, repeated_warnings.
    """
    if not articles:
        logger.warning("compose_newsletter: no articles provided.")
        return {}

    # Score every article
    scored: List[Tuple[Dict, Dict]] = []
    for art in articles:
        details = newsletter_worthiness_score(art)
        art = dict(art)
        art["_nw_score"]   = details["score"]
        art["_nw_details"] = details
        scored.append((art, details))

    # Sort descending by newsletter score so we fill featured/briefs greedily
    scored.sort(key=lambda x: x[0]["_nw_score"], reverse=True)

    featured_records: List[Dict] = []
    briefs_records:   List[Dict] = []
    watchlist_records: List[Dict] = []

    for art, details in scored:
        tier = route_to_tier(art, details)
        if tier == "featured" and len(featured_records) < 5:
            featured_records.append(art)
        elif tier == "briefs":
            briefs_records.append(art)
        else:
            watchlist_records.append(art)

    # Apply grouped-fallback detection on watchlist
    watchlist_records = _group_similar_watchlist_entries(watchlist_records)

    # Repeated fallback warning (reportable but not blocking)
    all_selected = featured_records + briefs_records + watchlist_records
    repeated_warnings = _detect_repeated_phrases(all_selected)

    # Build section content
    pub = config.get("publication_settings") or {}
    newsletter_title = pub.get("newsletter_title") or pub.get("podcast_title") or \
                       "Weekly Fertility Medicine Newsletter"
    today = datetime.now().strftime("%B %d, %Y")

    editor_note = _generate_editor_note(
        featured_records + briefs_records, config, llm_caller
    )

    featured_sections = [
        _build_featured_prose(art, i + 1)
        for i, art in enumerate(featured_records)
    ]
    brief_sections    = [_build_brief_text(art)       for art in briefs_records]
    watchlist_sections = [_build_watchlist_entry(art) for art in watchlist_records]

    closing = _generate_closing_takeaway(
        featured_records + briefs_records, config, llm_caller
    )

    newsletter = {
        "title":        newsletter_title,
        "date":         today,
        "editor_note":  editor_note,
        "featured":     featured_sections,
        "briefs":       brief_sections,
        "watchlist":    watchlist_sections,
        "closing":      closing,
        "article_counts": {
            "featured":  len(featured_sections),
            "briefs":    len(brief_sections),
            "watchlist": len(watchlist_sections),
        },
        "repeated_phrase_warnings": repeated_warnings,
    }

    rendered_md = _render_newsletter_md(newsletter)

    if output_txt:
        try:
            with open(output_txt, "w", encoding="utf-8") as fh:
                fh.write(rendered_md)
            print(f"Newsletter (Markdown) saved to: {output_txt}")
        except Exception as exc:
            logger.error("Error saving newsletter TXT: %s", exc)

    if output_json:
        try:
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump(newsletter, fh, indent=2, ensure_ascii=False)
            print(f"Newsletter (JSON) saved to: {output_json}")
        except Exception as exc:
            logger.error("Error saving newsletter JSON: %s", exc)

    return newsletter


def compose_newsletter_from_report(
    report_json_path: str,
    config: Dict,
    output_txt: Optional[str] = None,
    output_json: Optional[str] = None,
    llm_caller: Optional[Callable] = None,
) -> Dict:
    """
    Convenience wrapper: load articles from a consolidated report JSON file,
    then call ``compose_newsletter``.
    """
    with open(report_json_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    articles = report.get("articles") or []
    return compose_newsletter(articles, config, output_txt, output_json, llm_caller)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_repeated_phrases(articles: List[Dict]) -> List[str]:
    """
    Detect internal-report phrases that appear in the newsletter-visible fields of
    more than 40 % of processed articles (signals systematic fallback pollution).

    Only scans ``summary_sections`` (the source for newsletter prose), not the raw
    ``summary`` field which intentionally carries the internal evidence text.
    """
    n = max(len(articles), 1)
    threshold = max(2, int(n * 0.40))
    counts: Counter = Counter()
    for art in articles:
        ss = art.get("summary_sections") or {}
        text = " ".join(str(v) for v in ss.values() if v).lower()
        for phrase in INTERNAL_REPORT_PHRASES:
            if phrase in text:
                counts[phrase] += 1
    return [p for p, c in counts.items() if c >= threshold]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compose a curated newsletter from a validated article report."
    )
    parser.add_argument(
        "--report", "-r", required=True,
        help="Path to the consolidated report JSON (from podcast_generator)."
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to the pipeline config JSON (default: config.fertility.json)."
    )
    parser.add_argument(
        "--output-md", "-o", default=None,
        help="Output Markdown newsletter file."
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Output structured newsletter JSON file."
    )
    args = parser.parse_args()

    config_path = args.config or str(
        Path(__file__).parent / "config.fertility.json"
    )
    config: Dict = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

    report_path = args.report
    out_dir     = Path(report_path).parent
    output_md   = args.output_md   or str(out_dir / "newsletter.md")
    output_json = args.output_json or str(out_dir / "newsletter.json")

    newsletter = compose_newsletter_from_report(
        report_path, config, output_md, output_json
    )

    counts = newsletter.get("article_counts") or {}
    print(
        f"Newsletter composed: {counts.get('featured',0)} featured, "
        f"{counts.get('briefs',0)} briefs, "
        f"{counts.get('watchlist',0)} watchlist."
    )
    warnings = newsletter.get("repeated_phrase_warnings") or []
    if warnings:
        print(f"⚠️  Repeated generic phrase warnings: {len(warnings)}")
    print(f"Markdown output: {output_md}")
    print(f"JSON output:     {output_json}")


if __name__ == "__main__":
    main()
