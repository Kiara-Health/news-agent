"""
Cross-run novelty tracking backed by SQLite.

Design decisions:
- Uses stdlib ``sqlite3`` (no external ORM).  The schema is intentionally
  minimal: one ``history`` table with indexes on every match column.
- All datetimes are stored as ISO-8601 UTC strings (SQLite has no native
  datetime type); they are round-tripped through Python ``datetime`` objects.
- Matching priority: DOI > PMID > normalized canonical URL > dedupe_key >
  content_fingerprint.  The first match wins.
- ``mark_emitted()`` is a separate operation so downstream pipeline steps can
  call it after a newsletter is actually published, keeping history honest.

Thread safety: the module is designed for single-process, single-threaded
use (one ingestor run at a time).  The SQLite connection is opened and closed
within each public function call via a context manager.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Optional

from .models import HistoryConfig, HistoryRecord, NormalizedItem, NoveltyStatus
from .normalize import normalize_url

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key          TEXT    UNIQUE NOT NULL,
    doi                 TEXT,
    pmid                TEXT,
    canonical_url       TEXT    DEFAULT '',
    title               TEXT    DEFAULT '',
    source_name         TEXT    DEFAULT '',
    content_fingerprint TEXT    DEFAULT '',
    first_seen_at       TEXT    NOT NULL,
    last_seen_at        TEXT    NOT NULL,
    times_seen          INTEGER DEFAULT 1,
    times_emitted       INTEGER DEFAULT 0,
    last_emitted_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_hist_doi          ON history(doi);
CREATE INDEX IF NOT EXISTS idx_hist_pmid         ON history(pmid);
CREATE INDEX IF NOT EXISTS idx_hist_url          ON history(canonical_url);
CREATE INDEX IF NOT EXISTS idx_hist_fingerprint  ON history(content_fingerprint);
"""

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(_TS_FORMAT)


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: fromisoformat handles many formats
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection with WAL mode; commit on exit, rollback on error."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("BEGIN;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


def init_db(db_path: str) -> None:
    """Create the history table and indexes if they don't already exist."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.debug("History DB initialized at %s", db_path)


# ---------------------------------------------------------------------------
# Row ↔ HistoryRecord conversion
# ---------------------------------------------------------------------------


def _row_to_record(row: sqlite3.Row) -> HistoryRecord:
    return HistoryRecord(
        dedupe_key=row["dedupe_key"],
        doi=row["doi"],
        pmid=row["pmid"],
        canonical_url=row["canonical_url"] or "",
        title=row["title"] or "",
        source_name=row["source_name"] or "",
        content_fingerprint=row["content_fingerprint"] or "",
        first_seen_at=_str_to_dt(row["first_seen_at"]) or datetime.now(tz=timezone.utc),
        last_seen_at=_str_to_dt(row["last_seen_at"]) or datetime.now(tz=timezone.utc),
        times_seen=row["times_seen"] or 1,
        times_emitted=row["times_emitted"] or 0,
        last_emitted_at=_str_to_dt(row["last_emitted_at"]),
    )


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def lookup_item(
    conn: sqlite3.Connection,
    item: NormalizedItem,
) -> Optional[HistoryRecord]:
    """
    Find an existing history record matching *item*.

    Match order: DOI → PMID → normalized canonical URL → dedupe_key →
    content_fingerprint.  Returns the first match found.
    """
    row: Optional[sqlite3.Row] = None

    if item.doi:
        row = conn.execute(
            "SELECT * FROM history WHERE doi = ? LIMIT 1", (item.doi.lower(),)
        ).fetchone()

    if row is None and item.pmid:
        row = conn.execute(
            "SELECT * FROM history WHERE pmid = ? LIMIT 1", (item.pmid,)
        ).fetchone()

    if row is None and item.canonical_url:
        norm = normalize_url(item.canonical_url)
        row = conn.execute(
            "SELECT * FROM history WHERE canonical_url = ? LIMIT 1", (norm,)
        ).fetchone()

    if row is None and item.dedupe_key:
        row = conn.execute(
            "SELECT * FROM history WHERE dedupe_key = ? LIMIT 1", (item.dedupe_key,)
        ).fetchone()

    if row is None and item.content_fingerprint:
        row = conn.execute(
            "SELECT * FROM history WHERE content_fingerprint = ? LIMIT 1",
            (item.content_fingerprint,),
        ).fetchone()

    return _row_to_record(row) if row else None


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_item(
    conn: sqlite3.Connection,
    item: NormalizedItem,
    existing: Optional[HistoryRecord],
    now: Optional[datetime] = None,
) -> HistoryRecord:
    """
    Insert a new history record for *item* or update an existing one.

    When *existing* is not None (item was seen before), the following fields
    are updated: ``last_seen_at``, ``times_seen``.  The ``dedupe_key`` for the
    row remains stable even if the item reappears with a slightly different URL.

    Returns the final :class:`HistoryRecord`.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    norm_url = normalize_url(item.canonical_url)

    if existing is None:
        # New record
        conn.execute(
            """
            INSERT OR IGNORE INTO history
                (dedupe_key, doi, pmid, canonical_url, title, source_name,
                 content_fingerprint, first_seen_at, last_seen_at,
                 times_seen, times_emitted, last_emitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, NULL)
            """,
            (
                item.dedupe_key,
                item.doi.lower() if item.doi else None,
                item.pmid,
                norm_url,
                item.title[:500],
                item.source_name,
                item.content_fingerprint,
                _dt_to_str(now),
                _dt_to_str(now),
            ),
        )
        return HistoryRecord(
            dedupe_key=item.dedupe_key,
            doi=item.doi,
            pmid=item.pmid,
            canonical_url=norm_url,
            title=item.title[:500],
            source_name=item.source_name,
            content_fingerprint=item.content_fingerprint,
            first_seen_at=now,
            last_seen_at=now,
            times_seen=1,
            times_emitted=0,
        )
    else:
        # Update existing record
        conn.execute(
            """
            UPDATE history
            SET last_seen_at = ?,
                times_seen   = times_seen + 1
            WHERE dedupe_key = ?
            """,
            (_dt_to_str(now), existing.dedupe_key),
        )
        return HistoryRecord(
            dedupe_key=existing.dedupe_key,
            doi=existing.doi or item.doi,
            pmid=existing.pmid or item.pmid,
            canonical_url=existing.canonical_url or norm_url,
            title=existing.title or item.title[:500],
            source_name=existing.source_name or item.source_name,
            content_fingerprint=existing.content_fingerprint or item.content_fingerprint,
            first_seen_at=existing.first_seen_at,
            last_seen_at=now,
            times_seen=existing.times_seen + 1,
            times_emitted=existing.times_emitted,
            last_emitted_at=existing.last_emitted_at,
        )


# ---------------------------------------------------------------------------
# Novelty resolution
# ---------------------------------------------------------------------------


def resolve_novelty(
    item: NormalizedItem,
    record: Optional[HistoryRecord],
    cfg: HistoryConfig,
    now: Optional[datetime] = None,
) -> NoveltyStatus:
    """
    Determine the :class:`NoveltyStatus` for *item* given its history *record*.

    Rules (evaluated in order):
    1. No record → ``new``.
    2. Record exists, never emitted → ``seen_not_emitted``.
    3. Record exists, was emitted, but ``allow_repeat_after_days`` is set and
       the last emission is older than that threshold → ``reappeared``.
    4. Record exists, was emitted → ``previously_emitted``.
    5. (Future use) Significant content change detected → ``updated_existing_item``.
    """
    if record is None:
        return NoveltyStatus.NEW

    if now is None:
        now = datetime.now(tz=timezone.utc)

    if record.times_emitted == 0:
        return NoveltyStatus.SEEN_NOT_EMITTED

    # Check repeat-after policy
    if cfg.allow_repeat_after_days is not None and record.last_emitted_at:
        age_days = (now - record.last_emitted_at).days
        if age_days >= cfg.allow_repeat_after_days:
            return NoveltyStatus.REAPPEARED

    return NoveltyStatus.PREVIOUSLY_EMITTED


def should_suppress(novelty: NoveltyStatus, cfg: HistoryConfig) -> bool:
    """
    Return True if an item with *novelty* status should be suppressed given
    the current history configuration.
    """
    if novelty == NoveltyStatus.PREVIOUSLY_EMITTED and cfg.suppress_emitted:
        return True
    if novelty == NoveltyStatus.SEEN_NOT_EMITTED and cfg.suppress_seen:
        return True
    return False


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------


def annotate_items_with_history(
    items: List[NormalizedItem],
    cfg: HistoryConfig,
    now: Optional[datetime] = None,
) -> tuple[List[NormalizedItem], List[NormalizedItem], int, int]:
    """
    Look up every item in history, annotate novelty fields, upsert records,
    and return suppressed vs. eligible items.

    Returns a 4-tuple of:
    - ``eligible``: items not suppressed (pass to downstream).
    - ``suppressed``: items suppressed by history policy.
    - ``updated_count``: items whose content fingerprint changed (future).
    - ``new_count``: items that were truly new this run.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    init_db(cfg.path)

    eligible: List[NormalizedItem] = []
    suppressed: List[NormalizedItem] = []
    updated_count = 0
    new_count = 0

    with _connect(cfg.path) as conn:
        for item in items:
            existing = lookup_item(conn, item)
            novelty = resolve_novelty(item, existing, cfg, now)

            # Detect content changes on previously-seen items
            if (
                existing is not None
                and existing.content_fingerprint
                and item.content_fingerprint
                and existing.content_fingerprint != item.content_fingerprint
            ):
                novelty = NoveltyStatus.UPDATED_EXISTING_ITEM
                updated_count += 1

            # Upsert history regardless of suppression
            record = upsert_item(conn, item, existing, now)

            # Annotate the item in-place (model is mutable via model_copy)
            item = item.model_copy(
                update={
                    "novelty_status": novelty,
                    "is_previously_seen": existing is not None,
                    "is_previously_emitted": (existing is not None and existing.times_emitted > 0),
                    "previous_first_seen_at": record.first_seen_at if existing else None,
                    "previous_last_seen_at": record.last_seen_at if existing else None,
                    "previous_last_emitted_at": record.last_emitted_at if existing else None,
                }
            )

            if novelty == NoveltyStatus.NEW:
                new_count += 1

            if should_suppress(novelty, cfg):
                suppressed.append(item)
            else:
                eligible.append(item)

    logger.info(
        "History: %d eligible, %d suppressed, %d new, %d updated.",
        len(eligible),
        len(suppressed),
        new_count,
        updated_count,
    )
    return eligible, suppressed, updated_count, new_count


# ---------------------------------------------------------------------------
# Mark-emitted operation (called by downstream after newsletter publish)
# ---------------------------------------------------------------------------


def mark_emitted(
    dedupe_keys: List[str],
    db_path: str,
    emitted_at: Optional[datetime] = None,
) -> int:
    """
    Record that the items identified by *dedupe_keys* were actually emitted
    (published in a newsletter issue).

    Updates ``times_emitted`` and ``last_emitted_at`` in the history table.
    Returns the number of rows updated.
    """
    if not dedupe_keys:
        return 0

    if emitted_at is None:
        emitted_at = datetime.now(tz=timezone.utc)

    ts = _dt_to_str(emitted_at)
    updated = 0

    with _connect(db_path) as conn:
        for key in dedupe_keys:
            cursor = conn.execute(
                """
                UPDATE history
                SET times_emitted   = times_emitted + 1,
                    last_emitted_at = ?
                WHERE dedupe_key = ?
                """,
                (ts, key),
            )
            updated += cursor.rowcount

    logger.info("Marked %d items as emitted (at %s).", updated, ts)
    return updated


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------


def get_history_stats(db_path: str) -> dict:
    """Return aggregate stats from the history DB for logging/reporting."""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                          AS total_records,
                    SUM(times_emitted > 0)            AS ever_emitted,
                    SUM(times_emitted = 0)            AS seen_never_emitted,
                    MAX(last_seen_at)                 AS most_recent_seen
                FROM history
                """
            ).fetchone()
            if row:
                return {
                    "total_records": row[0],
                    "ever_emitted": row[1],
                    "seen_never_emitted": row[2],
                    "most_recent_seen": row[3],
                }
    except Exception as exc:
        logger.warning("Could not query history stats: %s", exc)
    return {}
