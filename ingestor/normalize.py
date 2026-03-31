"""
Normalize raw parsed feed dicts into :class:`NormalizedItem` instances.

Responsibilities:
- Map parse.py output dicts to the canonical output schema.
- Build stable ``dedupe_key`` and ``content_fingerprint`` fields.
- Apply basic quality filters (drop records with no title+URL, drop
  corrections/errata if configured).
- Record the ``fetched_at`` timestamp.

No editorial judgment is applied here beyond the hygiene rules described
in the requirements (dropping clearly invalid records, optionally dropping
errata).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from .models import DedupeConfig, FeedConfig, NormalizedItem, NoveltyStatus, RetrievalStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords that identify corrections / errata — case-insensitive, whole word
# ---------------------------------------------------------------------------

_CORRECTION_KEYWORDS = re.compile(
    r"\b(correction|erratum|errata|retraction|retracted|corrigendum|"
    r"expression\s+of\s+concern|publisher.s\s+note)\b",
    re.IGNORECASE,
)

# Tracking/noise query parameters to strip from URLs before hashing
_STRIP_PARAMS = frozenset(
    [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
        "ref",
        "source",
        "rss",
        "cmp",
        "mc_cid",
        "mc_eid",
    ]
)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """
    Return a stable, lowercase URL suitable for deduplication.

    Strips tracking query parameters, fragments, and trailing slashes.
    Does NOT follow redirects (that would require network I/O).
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        # Rebuild without fragment; strip tracking params from query string
        if parsed.query:
            kept = "&".join(
                part
                for part in parsed.query.split("&")
                if "=" in part and part.split("=")[0] not in _STRIP_PARAMS
            )
        else:
            kept = ""
        clean = urlunparse(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", kept, "")
        )
        return clean
    except Exception:
        return url.lower().rstrip("/")


# ---------------------------------------------------------------------------
# Dedupe key and content fingerprint
# ---------------------------------------------------------------------------


def build_dedupe_key(
    doi: Optional[str],
    pmid: Optional[str],
    canonical_url: str,
    title: str,
    published_at: Optional[datetime],
) -> str:
    """
    Build a stable, human-readable identifier for cross-run deduplication.

    Priority: DOI > PMID > normalized URL hash > title+date hash.
    The DOI and PMID forms are stored verbatim so they remain human-readable
    in the history database.
    """
    if doi:
        return f"doi:{doi.strip().lower()}"
    if pmid:
        return f"pmid:{pmid.strip()}"
    norm_url = normalize_url(canonical_url)
    if norm_url:
        h = hashlib.sha256(norm_url.encode()).hexdigest()[:16]
        return f"url:{h}"
    # Last resort: hash of normalized title + date string
    norm_title = re.sub(r"\s+", " ", title.lower().strip())
    date_str = published_at.date().isoformat() if published_at else "nodate"
    h = hashlib.sha256(f"{norm_title}|{date_str}".encode()).hexdigest()[:16]
    return f"title:{h}"


def build_content_fingerprint(title: str, summary: str) -> str:
    """
    A short, stable fingerprint used as a *fuzzy* cross-run match signal.

    Based on normalized title + first 200 chars of summary.  Minor edits
    (e.g. typographic corrections) may change this fingerprint, which is
    intentional: we want the DOI / PMID / URL keys to be the primary match
    signal, and the fingerprint only as a last resort.
    """
    norm_title = re.sub(r"\s+", " ", title.lower().strip())
    norm_snippet = re.sub(r"\s+", " ", summary[:200].lower().strip())
    combined = f"{norm_title}|{norm_snippet}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Correction / erratum detection
# ---------------------------------------------------------------------------


def is_correction(title: str) -> bool:
    """Return True if the title appears to describe a correction or retraction."""
    return bool(_CORRECTION_KEYWORDS.search(title))


# ---------------------------------------------------------------------------
# Richness score (used by deduplication to pick the best version)
# ---------------------------------------------------------------------------


def richness_score(item: NormalizedItem) -> int:
    """
    Return a simple integer richness score for an item.

    Higher is richer.  Used when two items are detected as duplicates to
    decide which version to keep.
    """
    score = 0
    if item.doi:
        score += 10
    if item.pmid:
        score += 8
    if item.authors:
        score += len(item.authors)
    if item.summary:
        score += min(len(item.summary) // 50, 10)
    if item.published_at:
        score += 5
    if item.tags:
        score += len(item.tags)
    return score


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------


def normalize_item(
    raw: Dict[str, Any],
    feed_cfg: FeedConfig,
    fetched_at: datetime,
    dedupe_cfg: Optional[DedupeConfig] = None,
) -> Optional[NormalizedItem]:
    """
    Convert a raw parsed dict (from :func:`parse.parse_entry`) into a
    :class:`NormalizedItem`.

    Returns ``None`` if the record fails the minimum quality bar:
    - Must have a non-empty title **or** a non-empty canonical URL.
    - Optionally drops corrections/errata when ``dedupe_cfg.drop_corrections``
      is True.
    """
    title: str = raw.get("title", "") or ""
    canonical_url: str = raw.get("canonical_url", "") or ""

    # Minimum quality gate
    if not title and not canonical_url:
        logger.debug("Dropping record with no title and no URL.")
        return None

    # Optional errata filter
    if dedupe_cfg and dedupe_cfg.drop_corrections and title and is_correction(title):
        logger.debug("Dropping correction/erratum: %s", title[:80])
        return None

    doi: Optional[str] = raw.get("doi")
    pmid: Optional[str] = raw.get("pmid")
    published_at: Optional[datetime] = raw.get("published_at")
    updated_at: Optional[datetime] = raw.get("updated_at")
    if published_at is not None:
        effective_freshness_at = published_at
        date_source = "published_at"
        used_fallback_date = False
        freshness_confidence = "high"
    elif updated_at is not None:
        effective_freshness_at = updated_at
        date_source = "updated_at"
        used_fallback_date = True
        freshness_confidence = "medium"
    else:
        effective_freshness_at = fetched_at
        date_source = "fetched_at"
        used_fallback_date = True
        freshness_confidence = "low"
    summary: str = raw.get("summary", "") or ""
    content_snippet: str = raw.get("content_snippet", "") or ""

    dedupe_key = build_dedupe_key(doi, pmid, canonical_url, title, published_at)
    content_fingerprint = build_content_fingerprint(title, summary)

    return NormalizedItem(
        source_name=feed_cfg.name,
        source_type=feed_cfg.source_type,
        feed_url=feed_cfg.url,
        title=title,
        canonical_url=canonical_url,
        published_at=published_at,
        updated_at=updated_at,
        effective_freshness_at=effective_freshness_at,
        date_source=date_source,
        used_fallback_date=used_fallback_date,
        freshness_confidence=freshness_confidence,
        authors=raw.get("authors", []),
        summary=summary,
        content_snippet=content_snippet,
        tags=raw.get("tags", []) + feed_cfg.tags,
        doi=doi,
        pmid=pmid,
        language=raw.get("language", "en"),
        raw_item_id=raw.get("raw_item_id", ""),
        dedupe_key=dedupe_key,
        content_fingerprint=content_fingerprint,
        fetched_at=fetched_at,
        retrieval_status=RetrievalStatus.OK,
        raw_metadata=raw.get("raw_metadata", {}),
        novelty_status=NoveltyStatus.NEW,
    )


def normalize_feed_items(
    raw_items: List[Dict[str, Any]],
    feed_cfg: FeedConfig,
    fetched_at: Optional[datetime] = None,
    dedupe_cfg: Optional[DedupeConfig] = None,
    since_days: Optional[int] = None,
) -> List[NormalizedItem]:
    """
    Normalize a list of raw dicts from a single feed.

    Args:
        raw_items:  Output of :func:`parse.parse_feed`.
        feed_cfg:   The feed's config entry (used for source_name, tags, etc.).
        fetched_at: Timestamp to stamp on every item (defaults to UTC now).
        dedupe_cfg: Dedup settings (controls errata filtering, thresholds).
        since_days: When set, discard items older than this many days.

    Returns:
        List of valid :class:`NormalizedItem` instances.
    """
    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)

    cutoff: Optional[datetime] = None
    if since_days is not None:
        from datetime import timedelta
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=since_days)

    items: List[NormalizedItem] = []
    dropped_quality = 0
    dropped_age = 0

    for raw in raw_items:
        item = normalize_item(raw, feed_cfg, fetched_at, dedupe_cfg)
        if item is None:
            dropped_quality += 1
            continue

        # Age filter: prefer published_at, otherwise use effective freshness date.
        if cutoff is not None and item.effective_freshness_at is not None:
            if item.effective_freshness_at < cutoff:
                dropped_age += 1
                continue

        items.append(item)

    if dropped_quality or dropped_age:
        logger.debug(
            "Feed '%s': dropped %d for quality, %d outside date window.",
            feed_cfg.name,
            dropped_quality,
            dropped_age,
        )
    return items
