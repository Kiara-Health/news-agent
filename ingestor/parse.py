"""
Low-level RSS/Atom entry parsing and identifier extraction.

Responsibilities:
- Convert raw ``feedparser`` entry objects to plain dicts suitable for
  normalization.
- Extract DOI and PMID from multiple possible locations in an entry.
- Strip HTML noise from summaries while preserving meaningful text.
- Parse and convert feed dates to Python datetime objects (UTC).

Assumptions (documented inline):
- feedparser always returns ``published_parsed`` / ``updated_parsed`` in UTC.
- DOIs always start with "10." followed by 4+ digits and a slash.
- PubMed article URLs contain the PMID as the last path segment.
- ``dc:identifier`` values may use the prefix "doi:" or "pmid:".
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# DOI pattern: starts with "10." followed by 4–9 digits, a slash, and at
# least one non-whitespace character.  Trailing punctuation is stripped.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)

# PubMed URLs: https://pubmed.ncbi.nlm.nih.gov/12345678/
#          or  https://www.ncbi.nlm.nih.gov/pubmed/12345678
_PMID_URL_RE = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d{1,8})",
    re.IGNORECASE,
)

# "pmid:12345678" inside dc:identifier or similar fields
_PMID_PREFIX_RE = re.compile(r"\bpmid:(\d{1,8})\b", re.IGNORECASE)

# HTML tag stripper
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Collapse consecutive whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Trailing punctuation to trim from DOIs
_DOI_TRAIL_RE = re.compile(r"[.,;:)>\]'\"]+$")


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------


def struct_time_to_utc(t: Optional[struct_time]) -> Optional[datetime]:
    """
    Convert a feedparser ``*_parsed`` struct_time (always UTC) to an
    aware ``datetime`` in UTC.

    Returns ``None`` when the input is ``None``, invalid, or pre-epoch (< 1970).
    """
    if t is None:
        return None
    try:
        dt = datetime(*t[:6], tzinfo=timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.year < 1970:
        return None
    return dt


def parse_datetime_text(raw: Any) -> Optional[datetime]:
    """
    Parse free-form feed date strings to UTC.

    Supports RFC2822 (common in RSS), ISO-8601, and simple ``YYYY-MM-DD``.
    Returns None when parsing fails.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # RFC2822 / email-date, e.g. "Sun, 30 Mar 2026 12:00:00 GMT"
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # ISO / common datetime strings
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue

    # Date-only fallback
    try:
        dt = datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
    try:
        dt = datetime(*t[:6], tzinfo=timezone.utc)
        # feedparser uses (0,0,0,0,0,0,…) as a sentinel for missing dates
        if dt.year < 1970:
            return None
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------


def strip_html(text: str, max_length: Optional[int] = None) -> str:
    """
    Remove HTML tags and decode entities from *text*.

    Consecutive whitespace is collapsed to a single space.  The result is
    optionally truncated to *max_length* characters.
    """
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = html.unescape(cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(" ", 1)[0]
    return cleaned


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------


def extract_doi(text: str) -> Optional[str]:
    """
    Extract the first DOI found in *text*.

    Searches raw text (not HTML); callers should strip HTML first when
    searching summary/content fields.
    """
    if not text:
        return None
    match = _DOI_RE.search(text)
    if match:
        doi = _DOI_TRAIL_RE.sub("", match.group(1))
        return doi
    return None


def extract_pmid(text: str) -> Optional[str]:
    """
    Extract a PubMed ID from *text*.

    Matches both URL-embedded PMIDs and explicit ``pmid:NNNNN`` prefixes.
    Returns the PMID as a string (digits only) or ``None``.
    """
    if not text:
        return None
    m = _PMID_URL_RE.search(text)
    if m:
        return m.group(1)
    m = _PMID_PREFIX_RE.search(text)
    if m:
        return m.group(1)
    return None


def _search_identifiers_in_entry(entry: Any) -> tuple[Optional[str], Optional[str]]:
    """
    Search all plausible locations in a feedparser entry for DOI and PMID.

    Locations checked (in priority order):
    1. ``entry.link`` — often a DOI resolver URL or PubMed URL
    2. ``entry.id`` — some publishers put the DOI here
    3. ``entry.tags`` — keyword tags sometimes carry DOI URIs
    4. ``dc:identifier`` fields in ``entry.dc_identifier``
    5. Summary / content text (fallback; may produce false positives)
    """
    doi: Optional[str] = None
    pmid: Optional[str] = None

    candidate_texts: List[str] = []

    # Link
    if hasattr(entry, "link") and entry.link:
        candidate_texts.append(entry.link)

    # Entry ID
    if hasattr(entry, "id") and entry.id:
        candidate_texts.append(entry.id)

    # dc:identifier (appears as entry.dc_identifier in feedparser)
    for attr in ("dc_identifier", "prism_doi", "dc_source"):
        val = getattr(entry, attr, None)
        if val:
            candidate_texts.append(val)

    # Tags / categories
    if hasattr(entry, "tags"):
        for tag in entry.tags or []:
            term = getattr(tag, "term", "") or ""
            scheme = getattr(tag, "scheme", "") or ""
            candidate_texts.extend([term, scheme])

    # Summary and content (last resort)
    if hasattr(entry, "summary") and entry.summary:
        candidate_texts.append(entry.summary[:500])
    if hasattr(entry, "content") and entry.content:
        for block in entry.content:
            candidate_texts.append(getattr(block, "value", "")[:500])

    for text in candidate_texts:
        if not doi:
            doi = extract_doi(text)
        if not pmid:
            pmid = extract_pmid(text)
        if doi and pmid:
            break

    return doi, pmid


# ---------------------------------------------------------------------------
# Canonical URL resolution
# ---------------------------------------------------------------------------


def canonical_url_from_entry(entry: Any) -> str:
    """
    Determine the best canonical URL for a feed entry.

    Priority:
    1. ``rel="alternate"`` links (the article landing page)
    2. DOI resolver URL (https://doi.org/…) if a DOI was found
    3. ``entry.link``
    """
    # Prefer rel=alternate links
    if hasattr(entry, "links"):
        for link in entry.links or []:
            rel = getattr(link, "rel", "")
            href = getattr(link, "href", "")
            if rel == "alternate" and href:
                return href

    # doi.org link is as canonical as it gets
    doi, _ = _search_identifiers_in_entry(entry)
    if doi:
        doi_url = f"https://doi.org/{doi}"
        # Only use if no link at all; prefer the publisher landing page
        if not getattr(entry, "link", ""):
            return doi_url

    return getattr(entry, "link", "") or ""


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------


def extract_authors(entry: Any) -> List[str]:
    """
    Return a list of author name strings from a feedparser entry.

    feedparser exposes authors as ``entry.authors`` (list of dicts with
    ``name`` / ``email`` keys) or the simpler ``entry.author`` string.
    """
    authors: List[str] = []

    if hasattr(entry, "authors") and entry.authors:
        for a in entry.authors:
            name = getattr(a, "name", "") or ""
            if name.strip():
                authors.append(name.strip())

    if not authors and hasattr(entry, "author") and entry.author:
        authors.append(entry.author.strip())

    return authors


# ---------------------------------------------------------------------------
# Tag / keyword extraction
# ---------------------------------------------------------------------------


def extract_tags(entry: Any, feed_tags: Optional[List[str]] = None) -> List[str]:
    """
    Collect category / keyword tags from a feedparser entry plus any
    feed-level tags from the config.

    Deduplicates case-insensitively.
    """
    seen: set[str] = set()
    tags: List[str] = []

    def _add(t: str) -> None:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            tags.append(t.strip())

    if hasattr(entry, "tags"):
        for tag in entry.tags or []:
            term = getattr(tag, "term", "") or ""
            if term:
                _add(term)

    for t in feed_tags or []:
        _add(t)

    return tags


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def extract_content_blocks(entry: Any) -> tuple[str, str]:
    """
    Return ``(summary, content_snippet)`` as plain text.

    ``summary`` is the feed-provided abstract / description (up to 1 000
    chars after HTML stripping).

    ``content_snippet`` is the first 500 chars of the full ``<content>``
    block, useful when the full text is embedded in the feed.
    """
    summary_raw = ""
    content_raw = ""

    if hasattr(entry, "summary") and entry.summary:
        summary_raw = entry.summary
    elif hasattr(entry, "description") and entry.description:
        summary_raw = entry.description

    if hasattr(entry, "content") and entry.content:
        for block in entry.content:
            val = getattr(block, "value", "")
            if val:
                content_raw = val
                break

    summary = strip_html(summary_raw, max_length=1000)
    snippet = strip_html(content_raw or summary_raw, max_length=500)

    return summary, snippet


# ---------------------------------------------------------------------------
# Top-level entry parsing
# ---------------------------------------------------------------------------


def parse_entry(
    entry: Any,
    feed_url: str,
    feed_name: str,
    feed_tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single feedparser ``entry`` object to a plain dict.

    Returns ``None`` if the entry lacks both a title and a usable URL
    (the minimum viable record per requirements).
    """
    try:
        title = strip_html(getattr(entry, "title", "") or "")
        url = canonical_url_from_entry(entry)

        if not title and not url:
            logger.debug("Dropping entry with no title and no URL from %s.", feed_name)
            return None

        doi, pmid = _search_identifiers_in_entry(entry)
        published_at = (
            struct_time_to_utc(getattr(entry, "published_parsed", None))
            or parse_datetime_text(getattr(entry, "published", None))
            or parse_datetime_text(getattr(entry, "dc_date", None))
            or parse_datetime_text(getattr(entry, "prism_publicationdate", None))
        )
        updated_at = (
            struct_time_to_utc(getattr(entry, "updated_parsed", None))
            or parse_datetime_text(getattr(entry, "updated", None))
            or parse_datetime_text(getattr(entry, "modified", None))
        )
        authors = extract_authors(entry)
        tags = extract_tags(entry, feed_tags)
        summary, content_snippet = extract_content_blocks(entry)

        # Detect language from dc:language; default to "en"
        language = getattr(entry, "dc_language", None) or getattr(
            entry, "language", "en"
        ) or "en"

        # raw_item_id: prefer entry.id; fall back to link
        raw_item_id = getattr(entry, "id", "") or url

        # Preserve source-specific fields verbatim in raw_metadata
        raw_metadata: Dict[str, Any] = {}
        for attr in (
            "id",
            "guidislink",
            "published",
            "updated",
            "dc_publisher",
            "dc_date",
            "prism_publicationdate",
            "prism_volume",
            "prism_number",
            "prism_startingpage",
            "prism_doi",
            "dc_format",
            "dc_type",
            "dc_rights",
        ):
            val = getattr(entry, attr, None)
            if val is not None:
                raw_metadata[attr] = val

        return {
            "title": title,
            "canonical_url": url,
            "published_at": published_at,
            "updated_at": updated_at,
            "authors": authors,
            "summary": summary,
            "content_snippet": content_snippet,
            "tags": tags,
            "doi": doi,
            "pmid": pmid,
            "language": language,
            "raw_item_id": raw_item_id,
            "feed_url": feed_url,
            "feed_name": feed_name,
            "raw_metadata": raw_metadata,
        }
    except Exception as exc:
        logger.error("Error parsing entry from %s: %s", feed_name, exc, exc_info=True)
        return None


def parse_feed(
    parsed_feed: Any,
    feed_url: str,
    feed_name: str,
    feed_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse all entries from a feedparser result object.

    Entries that fail to produce a valid dict are silently skipped (logged at
    DEBUG level).
    """
    entries = []
    for entry in parsed_feed.entries or []:
        record = parse_entry(entry, feed_url, feed_name, feed_tags)
        if record is not None:
            entries.append(record)
    logger.debug(
        "Parsed %d valid entries from feed '%s'.", len(entries), feed_name
    )
    return entries
