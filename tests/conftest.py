"""
Shared pytest fixtures for the feed ingestor test suite.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from ingestor.models import (
    DedupeConfig,
    FeedConfig,
    HistoryConfig,
    NormalizedItem,
    NoveltyStatus,
    RetrievalStatus,
)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def utc(*args: int) -> datetime:
    """Convenience: UTC datetime from year, month, day[, hour, min, sec]."""
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Minimal feedparser entry mock
# ---------------------------------------------------------------------------


class MockEntry(SimpleNamespace):
    """
    Minimal stand-in for a feedparser entry object.

    feedparser entries expose attributes directly (not dict keys), so
    SimpleNamespace is a clean approximation for testing parse.py.
    """

    def __init__(self, **kwargs: Any) -> None:
        defaults: Dict[str, Any] = {
            "title": "Test Article",
            "link": "https://example.com/article/1",
            "id": "https://example.com/article/1",
            "summary": "This is a test summary.",
            "content": None,
            "authors": [],
            "author": "",
            "tags": [],
            "published_parsed": None,
            "updated_parsed": None,
            "links": [],
            "dc_identifier": None,
            "prism_doi": None,
            "dc_source": None,
            "dc_language": None,
            "language": "en",
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class MockFeed(SimpleNamespace):
    """Stand-in for a feedparser FeedParserDict."""

    def __init__(self, entries=None, **kwargs: Any) -> None:
        defaults: Dict[str, Any] = {
            "entries": entries or [],
            "bozo": False,
            "bozo_exception": None,
            "feed": SimpleNamespace(
                title="Test Feed",
                description="A test feed",
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def feed_cfg() -> FeedConfig:
    return FeedConfig(
        name="Test Journal",
        url="https://example.com/feed.rss",
        source_type="journal",
        tags=["fertility"],
    )


@pytest.fixture
def dedupe_cfg() -> DedupeConfig:
    return DedupeConfig(title_similarity_threshold=0.85, drop_corrections=True)


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Return a path to a temporary SQLite history database."""
    return str(tmp_path / "history.db")


@pytest.fixture
def history_cfg(tmp_db: str) -> HistoryConfig:
    return HistoryConfig(
        path=tmp_db,
        suppress_emitted=True,
        suppress_seen=False,
    )


@pytest.fixture
def sample_item(feed_cfg: FeedConfig) -> NormalizedItem:
    """A fully-populated sample NormalizedItem."""
    return NormalizedItem(
        source_name=feed_cfg.name,
        source_type=feed_cfg.source_type,
        feed_url=feed_cfg.url,
        title="IVF outcomes in patients with endometriosis: a systematic review",
        canonical_url="https://academic.oup.com/humrep/article/39/1/1/7000001",
        published_at=utc(2024, 3, 1, 10, 0, 0),
        authors=["Smith J", "Jones A"],
        summary="We reviewed 42 RCTs examining IVF outcomes in endometriosis patients.",
        content_snippet="Background: Endometriosis affects up to 10% of reproductive-age women.",
        tags=["fertility", "ivf", "endometriosis"],
        doi="10.1093/humrep/dead001",
        pmid="38500001",
        fetched_at=utc(2024, 3, 1, 12, 0, 0),
        dedupe_key="doi:10.1093/humrep/dead001",
        content_fingerprint="abcdef1234567890",
        novelty_status=NoveltyStatus.NEW,
    )
