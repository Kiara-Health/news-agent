"""
Pydantic v2 models for the fertility feed ingestor.

All public schemas are defined here so downstream consumers have a single
import target.  Only ingestion-level concerns live here; scoring, editorial
ranking, and newsletter-formatting logic belong in other modules.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NoveltyStatus(str, Enum):
    """Describes whether an item is new relative to prior pipeline runs."""

    NEW = "new"
    SEEN_NOT_EMITTED = "seen_not_emitted"
    PREVIOUSLY_EMITTED = "previously_emitted"
    REAPPEARED = "reappeared"
    UPDATED_EXISTING_ITEM = "updated_existing_item"


class RetrievalStatus(str, Enum):
    """HTTP / parsing status for a feed fetch attempt."""

    OK = "ok"
    NOT_MODIFIED = "not_modified"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core output schema
# ---------------------------------------------------------------------------


class NormalizedItem(BaseModel):
    """
    The canonical output record for a single ingested article.

    Downstream systems (scoring, filtering, summarization) consume lists of
    these records serialized as JSON or JSONL.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # --- Provenance ---------------------------------------------------------
    source_name: str = Field(description="Human-readable feed name from config.")
    source_type: str = Field(
        default="journal",
        description="Feed category: journal | pubmed | preprint | news.",
    )
    feed_url: str = Field(description="The RSS/Atom URL this item came from.")

    # --- Core bibliographic fields -----------------------------------------
    title: str
    canonical_url: str
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    effective_freshness_at: Optional[datetime] = None
    date_source: str = "unknown"
    used_fallback_date: bool = False
    freshness_confidence: str = "low"
    authors: List[str] = Field(default_factory=list)
    summary: str = ""
    content_snippet: str = Field(
        default="",
        description="First ~500 chars of full content, HTML stripped.",
    )
    tags: List[str] = Field(default_factory=list)

    # --- Persistent identifiers --------------------------------------------
    doi: Optional[str] = None
    pmid: Optional[str] = None
    language: str = "en"

    # --- Pipeline keys ------------------------------------------------------
    raw_item_id: str = Field(
        default="",
        description="Original entry.id from the feed, unmodified.",
    )
    dedupe_key: str = Field(
        default="",
        description=(
            "Stable identifier used for cross-run deduplication. "
            "Derived from DOI > PMID > canonical URL hash > title hash."
        ),
    )
    content_fingerprint: str = Field(
        default="",
        description="Short SHA-256 of normalised title + summary snippet.",
    )

    # --- Retrieval metadata ------------------------------------------------
    fetched_at: datetime
    retrieval_status: RetrievalStatus = RetrievalStatus.OK
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific fields preserved verbatim.",
    )

    # --- Cross-run novelty fields ------------------------------------------
    is_previously_seen: bool = False
    is_previously_emitted: bool = False
    previous_first_seen_at: Optional[datetime] = None
    previous_last_seen_at: Optional[datetime] = None
    previous_last_emitted_at: Optional[datetime] = None
    novelty_status: NoveltyStatus = NoveltyStatus.NEW


# ---------------------------------------------------------------------------
# History / persistence schema
# ---------------------------------------------------------------------------


class HistoryRecord(BaseModel):
    """
    One row in the persistent history store (SQLite).

    Tracks every item the pipeline has ever seen or emitted so subsequent
    runs can suppress repeats.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    dedupe_key: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    canonical_url: str = ""
    title: str = ""
    source_name: str = ""
    content_fingerprint: str = ""

    first_seen_at: datetime
    last_seen_at: datetime
    times_seen: int = 1
    times_emitted: int = 0
    last_emitted_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class FeedConfig(BaseModel):
    """Per-feed configuration entry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    url: str
    source_type: str = "journal"
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)

    # ETag / Last-Modified caching (populated at runtime, persisted per run)
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class IngestConfig(BaseModel):
    """HTTP-layer tuning parameters."""

    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    user_agent: str = "KiaraHealth-FeedIngestor/1.0 (+https://kiara.health)"
    lookback_days: int = 14
    max_workers: int = 4


class DedupeConfig(BaseModel):
    """Within-run deduplication settings."""

    title_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "SequenceMatcher ratio above which two titles are treated as the "
            "same article.  0.85 balances precision vs. over-merging."
        ),
    )
    drop_corrections: bool = Field(
        default=True,
        description=(
            "Drop items whose title contains correction / erratum / "
            "retraction / corrigendum keywords."
        ),
    )


class HistoryConfig(BaseModel):
    """Cross-run persistence settings."""

    path: str = "ingestor_history.db"
    allow_repeat_after_days: Optional[int] = Field(
        default=None,
        description=(
            "Re-allow a previously-emitted item after this many days. "
            "None means suppress indefinitely."
        ),
    )
    suppress_emitted: bool = True
    suppress_seen: bool = False


class IngestorConfig(BaseModel):
    """Top-level configuration container loaded from YAML/JSON."""

    feeds: List[FeedConfig] = Field(default_factory=list)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)


# ---------------------------------------------------------------------------
# Diagnostics / run summary
# ---------------------------------------------------------------------------


class FeedDiagnostics(BaseModel):
    """Per-feed outcome reported in the run summary."""

    feed_name: str
    feed_url: str
    status: str
    items_fetched: int = 0
    error: Optional[str] = None
    http_status: Optional[int] = None
    etag_hit: bool = False


class RunSummary(BaseModel):
    """Aggregate statistics for a single ingestor run."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    feeds_attempted: int = 0
    feeds_succeeded: int = 0
    feeds_failed: int = 0

    raw_items_fetched: int = 0
    items_after_normalization: int = 0
    items_after_deduplication: int = 0
    items_dropped_for_quality: int = 0

    previously_seen_items: int = 0
    previously_emitted_items: int = 0
    novel_items: int = 0
    suppressed_as_repeat: int = 0
    updated_existing_items: int = 0

    per_feed_diagnostics: List[FeedDiagnostics] = Field(default_factory=list)
