"""
Tests for ingestor/history.py

Covers:
- History DB initialization
- New item insertion (first_seen_at set, times_seen=1)
- Existing item update (times_seen incremented, last_seen_at updated)
- Novelty status: new / seen_not_emitted / previously_emitted / reappeared
- Suppression logic: suppress_emitted=True, suppress_seen=False
- mark_emitted updates times_emitted and last_emitted_at
- annotate_items_with_history batch round-trip
- History matching priority: DOI > PMID > URL > dedupe_key > fingerprint
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ingestor.history import (
    _connect,
    annotate_items_with_history,
    get_history_stats,
    init_db,
    lookup_item,
    mark_emitted,
    resolve_novelty,
    should_suppress,
    upsert_item,
)
from ingestor.models import HistoryConfig, NormalizedItem, NoveltyStatus
from tests.conftest import utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    title: str = "IVF outcomes in endometriosis",
    doi: str | None = "10.1093/humrep/dead001",
    pmid: str | None = "38500001",
    canonical_url: str = "https://academic.oup.com/humrep/article/1",
    dedupe_key: str = "doi:10.1093/humrep/dead001",
    content_fingerprint: str = "fp001",
    published_at: datetime | None = None,
) -> NormalizedItem:
    return NormalizedItem(
        source_name="Test Journal",
        source_type="journal",
        feed_url="https://example.com/feed.rss",
        title=title,
        canonical_url=canonical_url,
        doi=doi,
        pmid=pmid,
        published_at=published_at or utc(2024, 3, 1),
        fetched_at=utc(2024, 3, 1, 12),
        dedupe_key=dedupe_key,
        content_fingerprint=content_fingerprint,
        novelty_status=NoveltyStatus.NEW,
    )


# ---------------------------------------------------------------------------
# DB initialization
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_table(self, tmp_db):
        init_db(tmp_db)
        import sqlite3
        with sqlite3.connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            ).fetchone()
        assert row is not None

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # should not raise


# ---------------------------------------------------------------------------
# Insert and lookup
# ---------------------------------------------------------------------------


class TestInsertAndLookup:
    def test_new_item_inserted(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            record = upsert_item(conn, item, existing=None, now=utc(2024, 3, 1, 12))
        assert record.times_seen == 1
        assert record.times_emitted == 0
        assert record.first_seen_at == utc(2024, 3, 1, 12)

    def test_lookup_by_doi(self, tmp_db):
        item = _make_item(doi="10.1093/humrep/dead001")
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1, 12))
            found = lookup_item(conn, item)
        assert found is not None
        assert found.doi == "10.1093/humrep/dead001"

    def test_lookup_by_pmid_when_no_doi(self, tmp_db):
        item = _make_item(doi=None, pmid="38500001", dedupe_key="pmid:38500001")
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1, 12))
            found = lookup_item(conn, item)
        assert found is not None

    def test_lookup_by_dedupe_key(self, tmp_db):
        item = _make_item(doi=None, pmid=None, dedupe_key="url:abc123def456")
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1, 12))
            found = lookup_item(conn, item)
        assert found is not None

    def test_lookup_by_fingerprint_only(self, tmp_db):
        item = _make_item(doi=None, pmid=None, canonical_url="", dedupe_key="", content_fingerprint="fp_unique_999")
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1, 12))
            found = lookup_item(conn, item)
        assert found is not None

    def test_lookup_returns_none_for_unknown(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            found = lookup_item(conn, item)
        assert found is None


# ---------------------------------------------------------------------------
# Update on re-appearance
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_times_seen_incremented(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            existing = upsert_item(conn, item, None, utc(2024, 3, 1))
            updated = upsert_item(conn, item, existing, utc(2024, 3, 8))
        assert updated.times_seen == 2

    def test_last_seen_at_updated(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            existing = upsert_item(conn, item, None, utc(2024, 3, 1))
            updated = upsert_item(conn, item, existing, utc(2024, 3, 8))
        assert updated.last_seen_at == utc(2024, 3, 8)

    def test_first_seen_preserved(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            existing = upsert_item(conn, item, None, utc(2024, 3, 1))
            updated = upsert_item(conn, item, existing, utc(2024, 3, 8))
        assert updated.first_seen_at == utc(2024, 3, 1)


# ---------------------------------------------------------------------------
# Novelty status resolution
# ---------------------------------------------------------------------------


class TestResolveNovelty:
    def test_new_item_is_new(self, history_cfg):
        item = _make_item()
        status = resolve_novelty(item, record=None, cfg=history_cfg)
        assert status == NoveltyStatus.NEW

    def test_seen_never_emitted(self, tmp_db, history_cfg):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            record = upsert_item(conn, item, None, utc(2024, 3, 1))
        status = resolve_novelty(item, record, history_cfg)
        assert status == NoveltyStatus.SEEN_NOT_EMITTED

    def test_previously_emitted(self, tmp_db, history_cfg):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            record = upsert_item(conn, item, None, utc(2024, 3, 1))
        mark_emitted([item.dedupe_key], tmp_db, emitted_at=utc(2024, 3, 5))
        # Reload record to get updated times_emitted
        with _connect(tmp_db) as conn:
            record = lookup_item(conn, item)
        status = resolve_novelty(item, record, history_cfg, now=utc(2024, 3, 8))
        assert status == NoveltyStatus.PREVIOUSLY_EMITTED

    def test_reappeared_after_allow_repeat(self, tmp_db):
        cfg = HistoryConfig(path=tmp_db, suppress_emitted=True, allow_repeat_after_days=30)
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            record = upsert_item(conn, item, None, utc(2024, 1, 1))
        mark_emitted([item.dedupe_key], tmp_db, emitted_at=utc(2024, 1, 5))
        with _connect(tmp_db) as conn:
            record = lookup_item(conn, item)
        # 60 days later → exceeds allow_repeat_after_days=30 → REAPPEARED
        status = resolve_novelty(item, record, cfg, now=utc(2024, 3, 5))
        assert status == NoveltyStatus.REAPPEARED


# ---------------------------------------------------------------------------
# Suppression logic
# ---------------------------------------------------------------------------


class TestShouldSuppress:
    def test_suppress_previously_emitted_when_configured(self, history_cfg):
        assert should_suppress(NoveltyStatus.PREVIOUSLY_EMITTED, history_cfg) is True

    def test_do_not_suppress_new(self, history_cfg):
        assert should_suppress(NoveltyStatus.NEW, history_cfg) is False

    def test_do_not_suppress_seen_not_emitted_by_default(self, history_cfg):
        assert should_suppress(NoveltyStatus.SEEN_NOT_EMITTED, history_cfg) is False

    def test_suppress_seen_when_configured(self, tmp_db):
        cfg = HistoryConfig(path=tmp_db, suppress_emitted=False, suppress_seen=True)
        assert should_suppress(NoveltyStatus.SEEN_NOT_EMITTED, cfg) is True

    def test_reappeared_not_suppressed(self, history_cfg):
        assert should_suppress(NoveltyStatus.REAPPEARED, history_cfg) is False


# ---------------------------------------------------------------------------
# mark_emitted
# ---------------------------------------------------------------------------


class TestMarkEmitted:
    def test_updates_times_emitted(self, tmp_db):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1))
        count = mark_emitted([item.dedupe_key], tmp_db, emitted_at=utc(2024, 3, 5))
        assert count == 1
        with _connect(tmp_db) as conn:
            record = lookup_item(conn, item)
        assert record is not None
        assert record.times_emitted == 1
        assert record.last_emitted_at == utc(2024, 3, 5)

    def test_unknown_key_does_not_error(self, tmp_db):
        init_db(tmp_db)
        count = mark_emitted(["nonexistent:key"], tmp_db)
        assert count == 0

    def test_empty_list(self, tmp_db):
        count = mark_emitted([], tmp_db)
        assert count == 0


# ---------------------------------------------------------------------------
# annotate_items_with_history batch round-trip
# ---------------------------------------------------------------------------


class TestAnnotateBatch:
    def test_new_items_are_eligible(self, history_cfg):
        items = [
            _make_item(
                doi=f"10.1234/test{i:03d}",
                pmid=f"3850000{i}",
                dedupe_key=f"doi:10.1234/test{i:03d}",
                content_fingerprint=f"fp{i:03d}",
                canonical_url=f"https://example.com/article/{i}",
            )
            for i in range(3)
        ]
        eligible, suppressed, updated, new_count = annotate_items_with_history(
            items, history_cfg, now=utc(2024, 3, 1, 12)
        )
        assert len(eligible) == 3
        assert len(suppressed) == 0
        assert new_count == 3

    def test_previously_emitted_suppressed(self, tmp_db, history_cfg):
        item = _make_item()
        init_db(tmp_db)
        # Pre-seed history: seen and emitted
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1))
        mark_emitted([item.dedupe_key], tmp_db, emitted_at=utc(2024, 3, 5))

        eligible, suppressed, _, _ = annotate_items_with_history(
            [item], history_cfg, now=utc(2024, 3, 8)
        )
        assert len(suppressed) == 1
        assert len(eligible) == 0
        assert suppressed[0].novelty_status == NoveltyStatus.PREVIOUSLY_EMITTED

    def test_seen_not_emitted_remains_eligible(self, tmp_db, history_cfg):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1))
        # NOT marking as emitted

        eligible, suppressed, _, _ = annotate_items_with_history(
            [item], history_cfg, now=utc(2024, 3, 8)
        )
        assert len(eligible) == 1
        assert eligible[0].novelty_status == NoveltyStatus.SEEN_NOT_EMITTED

    def test_novelty_fields_populated(self, tmp_db, history_cfg):
        item = _make_item()
        init_db(tmp_db)
        with _connect(tmp_db) as conn:
            upsert_item(conn, item, None, utc(2024, 3, 1))

        eligible, _, _, _ = annotate_items_with_history(
            [item], history_cfg, now=utc(2024, 3, 8)
        )
        assert eligible[0].is_previously_seen is True
        assert eligible[0].previous_first_seen_at == utc(2024, 3, 1)
