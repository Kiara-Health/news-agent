"""
HTTP fetching layer for the fertility feed ingestor.

Design decisions:
- Uses ``requests`` (already in the news-agent dependency tree) rather than
  ``httpx`` to avoid an extra dependency.
- Retry logic is implemented with ``tenacity`` for readable exponential
  backoff with jitter; it retries only on connection/timeout errors and
  HTTP 5xx responses.
- ETag / Last-Modified headers are supported and persisted in a lightweight
  JSON sidecar file next to the history database.
- Each feed is fetched independently.  Failures are caught, logged, and
  reported in FeedDiagnostics without aborting the rest of the run.
- ``feedparser`` receives the raw bytes/text rather than the URL, so our
  custom headers and retry logic are always in effect.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import feedparser  # type: ignore[import-untyped]
import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import FeedConfig, FeedDiagnostics, IngestConfig, RetrievalStatus

logger = logging.getLogger(__name__)

# feedparser attribute that indicates a hard parse error (not a warning)
_FATAL_BOZO_TYPES = (
    "urllib.error.URLError",
    "socket.error",
)

# ---------------------------------------------------------------------------
# ETag / Last-Modified sidecar cache
# ---------------------------------------------------------------------------


def _cache_path(history_db_path: str) -> Path:
    """Return the path of the ETag cache JSON file alongside the history DB."""
    base = Path(history_db_path)
    return base.with_suffix(".etag_cache.json")


def load_etag_cache(history_db_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load the per-feed ETag / Last-Modified cache from disk.

    Returns a dict keyed by feed URL.
    """
    path = _cache_path(history_db_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read ETag cache %s: %s", path, exc)
        return {}


def save_etag_cache(
    history_db_path: str, cache: Dict[str, Dict[str, str]]
) -> None:
    """Persist the updated ETag / Last-Modified cache to disk."""
    path = _cache_path(history_db_path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not write ETag cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Retry-decorated HTTP GET
# ---------------------------------------------------------------------------


def _build_retry_get(max_retries: int, backoff_factor: float):
    """
    Return a tenacity-decorated function that does a single GET with retries.

    We create the decorated function dynamically so that ``max_retries`` and
    ``backoff_factor`` come from config rather than being hard-coded.
    """

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff_factor, min=2, max=60),
        retry=retry_if_exception_type((ConnectionError, ReadTimeout, RequestException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(url: str, headers: Dict[str, str], timeout: int) -> requests.Response:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code >= 500:
            # Force tenacity to retry on server-side errors
            raise RequestException(
                f"Server error {response.status_code} for {url}",
                response=response,
            )
        return response

    return _get


# ---------------------------------------------------------------------------
# Single-feed fetch
# ---------------------------------------------------------------------------


def fetch_feed(
    feed_cfg: FeedConfig,
    ingest_cfg: IngestConfig,
    etag_cache: Optional[Dict[str, Dict[str, str]]] = None,
    force_refresh: bool = False,
) -> Tuple[Optional[feedparser.FeedParserDict], FeedDiagnostics, Dict[str, str]]:
    """
    Fetch and parse one RSS/Atom feed.

    Returns:
        A 3-tuple of:
        - ``feedparser.FeedParserDict`` if the fetch and parse succeeded,
          ``None`` otherwise.
        - :class:`FeedDiagnostics` describing the outcome.
        - A dict of updated caching headers (``etag`` / ``modified``) to
          persist back to the ETag cache.  Empty if nothing changed.
    """
    diag = FeedDiagnostics(
        feed_name=feed_cfg.name,
        feed_url=feed_cfg.url,
        status="failed",
    )
    updated_cache_entry: Dict[str, str] = {}

    headers: Dict[str, str] = {
        "User-Agent": ingest_cfg.user_agent,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }

    # Attach conditional-GET headers from cache unless force-refresh is enabled.
    cache_entry = (etag_cache or {}).get(feed_cfg.url, {})
    if not force_refresh:
        if cache_entry.get("etag"):
            headers["If-None-Match"] = cache_entry["etag"]
        if cache_entry.get("modified"):
            headers["If-Modified-Since"] = cache_entry["modified"]
    else:
        logger.debug("Force-refresh enabled for %s: skipping ETag/Last-Modified headers.", feed_cfg.name)

    _get = _build_retry_get(ingest_cfg.max_retries, ingest_cfg.backoff_factor)

    try:
        response = _get(feed_cfg.url, headers, ingest_cfg.timeout_seconds)
    except RetryError as exc:
        diag.error = f"All retries exhausted: {exc}"
        logger.error("Feed %s failed after all retries: %s", feed_cfg.name, exc)
        return None, diag, updated_cache_entry
    except RequestException as exc:
        diag.error = str(exc)
        logger.error("Feed %s fetch error: %s", feed_cfg.name, exc)
        return None, diag, updated_cache_entry

    diag.http_status = response.status_code

    # 304 Not Modified — no new content
    if response.status_code == 304:
        diag.status = RetrievalStatus.NOT_MODIFIED.value
        diag.etag_hit = True
        diag.items_fetched = 0
        logger.info("Feed %s: 304 Not Modified (ETag hit).", feed_cfg.name)
        return None, diag, updated_cache_entry

    if not response.ok:
        diag.error = f"HTTP {response.status_code}"
        logger.warning("Feed %s returned HTTP %d.", feed_cfg.name, response.status_code)
        return None, diag, updated_cache_entry

    # Persist updated caching headers for next run
    if response.headers.get("ETag"):
        updated_cache_entry["etag"] = response.headers["ETag"]
    if response.headers.get("Last-Modified"):
        updated_cache_entry["modified"] = response.headers["Last-Modified"]

    # Parse with feedparser (offline, from content bytes)
    try:
        parsed = feedparser.parse(
            response.content,
            response_headers=dict(response.headers),
        )
    except Exception as exc:
        diag.error = f"feedparser exception: {exc}"
        logger.error("Feed %s parse exception: %s", feed_cfg.name, exc)
        return None, diag, updated_cache_entry

    # bozo == True means feedparser encountered a malformation; it usually
    # still returns partial entries, so we continue with a warning.
    if parsed.bozo:
        exc_type = type(parsed.bozo_exception).__name__
        logger.warning(
            "Feed %s is malformed (%s: %s). Attempting partial parse.",
            feed_cfg.name,
            exc_type,
            parsed.bozo_exception,
        )
        diag.status = RetrievalStatus.PARTIAL.value
    else:
        diag.status = RetrievalStatus.OK.value

    diag.items_fetched = len(parsed.entries)
    logger.info(
        "Feed %s: fetched %d entries (HTTP %d).",
        feed_cfg.name,
        diag.items_fetched,
        response.status_code,
    )
    return parsed, diag, updated_cache_entry


# ---------------------------------------------------------------------------
# Multi-feed orchestration
# ---------------------------------------------------------------------------


def fetch_all_feeds(
    feeds: List[FeedConfig],
    ingest_cfg: IngestConfig,
    etag_cache: Optional[Dict[str, Dict[str, str]]] = None,
    courtesy_delay: float = 0.5,
    force_refresh: bool = False,
) -> Tuple[Dict[str, feedparser.FeedParserDict], List[FeedDiagnostics], Dict[str, Dict[str, str]]]:
    """
    Fetch all enabled feeds, up to ``ingest_cfg.max_workers`` in parallel.

    A per-feed ``courtesy_delay`` (seconds) is inserted between submissions
    to avoid hammering the same origin server when multiple feeds share a
    domain.

    Returns:
        A 3-tuple of:
        - Dict mapping feed URL → parsed FeedParserDict (excludes failures).
        - List of :class:`FeedDiagnostics` for every attempted feed.
        - Updated ETag cache dict (merge into the existing cache before saving).
    """
    enabled = [f for f in feeds if f.enabled]
    logger.info(
        "Fetching %d enabled feeds (max_workers=%d, force_refresh=%s).",
        len(enabled), ingest_cfg.max_workers, force_refresh
    )

    results: Dict[str, feedparser.FeedParserDict] = {}
    all_diags: List[FeedDiagnostics] = []
    updated_cache: Dict[str, Dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=ingest_cfg.max_workers) as pool:
        future_to_feed = {}
        for feed_cfg in enabled:
            future = pool.submit(fetch_feed, feed_cfg, ingest_cfg, etag_cache, force_refresh)
            future_to_feed[future] = feed_cfg
            time.sleep(courtesy_delay)

        for future in as_completed(future_to_feed):
            feed_cfg = future_to_feed[future]
            try:
                parsed, diag, cache_entry = future.result()
            except Exception as exc:
                logger.error(
                    "Unhandled error fetching feed %s: %s", feed_cfg.name, exc
                )
                diag = FeedDiagnostics(
                    feed_name=feed_cfg.name,
                    feed_url=feed_cfg.url,
                    status="failed",
                    error=str(exc),
                )
                parsed = None
                cache_entry = {}

            all_diags.append(diag)
            if cache_entry:
                updated_cache[feed_cfg.url] = cache_entry
            if parsed is not None:
                results[feed_cfg.url] = parsed

    logger.info(
        "Fetch complete: %d/%d feeds returned data.",
        len(results),
        len(enabled),
    )
    return results, all_diags, updated_cache
