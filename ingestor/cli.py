"""
Command-line interface for the fertility feed ingestor.

Subcommands
-----------
ingest
    Fetch, parse, normalise, deduplicate, and annotate articles from all
    configured feeds.  Outputs JSON or JSONL to a file or stdout.

mark-emitted
    Read a JSON/JSONL file of previously emitted :class:`NormalizedItem`
    records and update the history DB to reflect that they were published.
    Call this after each newsletter is sent.

Usage examples
--------------
# Standard twice-weekly run
feed-ingestor ingest --config config.yaml --since-days 7 --output items.jsonl --format jsonl

# Dry-run without suppression (review what would be emitted)
feed-ingestor ingest --config config.yaml --suppress-emitted false --verbose

# After publishing, mark the emitted items so history stays accurate
feed-ingestor mark-emitted --items items.jsonl --history-path history.db

# Force re-allow items older than 60 days
feed-ingestor ingest --config config.yaml --allow-repeat-after-days 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional, TextIO

from .config import load_config, merge_cli_overrides
from .dedupe import deduplicate
from .fetch import fetch_all_feeds, load_etag_cache, save_etag_cache
from .history import (
    annotate_items_with_history,
    get_history_stats,
    mark_emitted as _mark_emitted,
)
from .models import NormalizedItem, RunSummary
from .normalize import normalize_feed_items
from .parse import parse_feed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _serialize_item(item: NormalizedItem) -> dict:
    """Serialize a NormalizedItem to a JSON-safe dict."""
    return item.model_dump(mode="json")


def _write_output(
    items: List[NormalizedItem],
    fmt: str,
    dest: Optional[str],
) -> None:
    """Write *items* to *dest* (file path or stdout) in *fmt* format."""
    fh: TextIO
    should_close = False

    if dest:
        fh = open(dest, "w", encoding="utf-8")
        should_close = True
    else:
        fh = sys.stdout

    try:
        if fmt == "jsonl":
            for item in items:
                fh.write(json.dumps(_serialize_item(item), ensure_ascii=False))
                fh.write("\n")
        else:  # json
            payload = [_serialize_item(i) for i in items]
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    finally:
        if should_close:
            fh.close()


def _write_summary(summary: RunSummary, dest: Optional[str], fmt: str) -> None:
    """Write the run summary alongside the items output."""
    if not dest:
        # When writing to stdout don't pollute the data stream
        return
    import os

    base, ext = os.path.splitext(dest)
    summary_path = f"{base}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary.model_dump(mode="json"), fh, indent=2)
    logger.info("Run summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# `ingest` subcommand
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the full ingest pipeline.  Returns an exit code."""
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(tz=timezone.utc)

    # --- Config loading -----------------------------------------------------
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(
        cfg,
        history_path=args.history_path,
        suppress_emitted=_parse_bool(args.suppress_emitted),
        suppress_seen=_parse_bool(args.suppress_seen),
        lookback_days=args.lookback_days,
        allow_repeat_after_days=args.allow_repeat_after_days,
    )

    # Respect --limit early to avoid unnecessary fetching when testing
    limit: Optional[int] = args.limit

    if not cfg.feeds:
        logger.error("No feeds configured. Add feeds to your config file and retry.")
        return 1

    logger.info(
        "Run %s: %d feeds configured, lookback_days=%d",
        run_id,
        len(cfg.feeds),
        cfg.ingest.lookback_days,
    )

    # --- ETag cache ---------------------------------------------------------
    etag_cache = load_etag_cache(cfg.history.path)

    # --- Fetch --------------------------------------------------------------
    parsed_feeds, all_diags, updated_etags = fetch_all_feeds(
        cfg.feeds, cfg.ingest, etag_cache, force_refresh=bool(args.force_refresh)
    )

    # Persist updated ETags immediately so they survive a mid-run crash
    if updated_etags:
        merged = {**etag_cache, **updated_etags}
        save_etag_cache(cfg.history.path, merged)

    feeds_succeeded = sum(
        1 for d in all_diags if d.status in ("ok", "partial", "not_modified")
    )
    feeds_failed = sum(1 for d in all_diags if d.status == "failed")

    summary = RunSummary(
        run_id=run_id,
        started_at=started_at,
        feeds_attempted=len(all_diags),
        feeds_succeeded=feeds_succeeded,
        feeds_failed=feeds_failed,
        per_feed_diagnostics=all_diags,
    )

    # --- Parse + Normalize --------------------------------------------------
    all_items: List[NormalizedItem] = []
    since_days: Optional[int] = args.since_days or cfg.ingest.lookback_days

    feed_cfg_map = {f.url: f for f in cfg.feeds}

    for feed_url, parsed in parsed_feeds.items():
        feed_cfg = feed_cfg_map.get(feed_url)
        if feed_cfg is None:
            continue
        raw_items = parse_feed(parsed, feed_url, feed_cfg.name, feed_cfg.tags)
        summary.raw_items_fetched += len(raw_items)
        normalized = normalize_feed_items(
            raw_items,
            feed_cfg,
            fetched_at=started_at,
            dedupe_cfg=cfg.dedupe,
            since_days=since_days,
        )
        all_items.extend(normalized)

    summary.items_after_normalization = len(all_items)
    dropped_for_quality = summary.raw_items_fetched - summary.items_after_normalization
    summary.items_dropped_for_quality = max(dropped_for_quality, 0)

    # --- Within-run deduplication ------------------------------------------
    deduped_items, removed = deduplicate(all_items, cfg.dedupe)
    summary.items_after_deduplication = len(deduped_items)

    # --- History annotation + suppression -----------------------------------
    eligible, suppressed_items, updated_count, new_count = annotate_items_with_history(
        deduped_items, cfg.history, now=started_at
    )

    summary.previously_seen_items = sum(1 for i in deduped_items if i.is_previously_seen)
    summary.previously_emitted_items = sum(1 for i in deduped_items if i.is_previously_emitted)
    summary.novel_items = new_count
    summary.suppressed_as_repeat = len(suppressed_items)
    summary.updated_existing_items = updated_count
    summary.completed_at = datetime.now(tz=timezone.utc)

    # --- Apply --limit ------------------------------------------------------
    output_items = eligible[:limit] if limit else eligible

    # --- Output -------------------------------------------------------------
    _write_output(output_items, args.format, args.output)
    _write_summary(summary, args.output, args.format)

    # --- Console run summary ------------------------------------------------
    _print_run_summary(summary, verbose=args.verbose)

    return 0


# ---------------------------------------------------------------------------
# `mark-emitted` subcommand
# ---------------------------------------------------------------------------


def cmd_mark_emitted(args: argparse.Namespace) -> int:
    """
    Read a JSON/JSONL file of items and mark them as emitted in history.

    This should be called after each newsletter send so the history DB
    accurately reflects real editorial decisions rather than just retrieval.
    """
    db_path: str = args.history_path or "ingestor_history.db"
    items_path: str = args.items

    dedupe_keys: List[str] = []
    try:
        with open(items_path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.splitlines() if line.strip()]

        for record in records:
            key = record.get("dedupe_key", "")
            if key:
                dedupe_keys.append(key)
    except Exception as exc:
        logger.error("Could not read items file %s: %s", items_path, exc)
        return 1

    if not dedupe_keys:
        logger.warning("No dedupe_keys found in %s. Nothing to mark.", items_path)
        return 0

    updated = _mark_emitted(dedupe_keys, db_path)
    print(f"Marked {updated} items as emitted in {db_path}.")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    """Convert CLI string 'true'/'false' to bool, or None if not provided."""
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes")


def _print_run_summary(summary: RunSummary, verbose: bool = False) -> None:
    """Print a human-readable run summary to stderr."""
    elapsed = ""
    if summary.completed_at and summary.started_at:
        secs = (summary.completed_at - summary.started_at).total_seconds()
        elapsed = f"  elapsed: {secs:.1f}s"

    lines = [
        "",
        f"=== Feed Ingestor Run {summary.run_id} ==={elapsed}",
        f"  Feeds:         {summary.feeds_succeeded}/{summary.feeds_attempted} succeeded"
        + (f", {summary.feeds_failed} failed" if summary.feeds_failed else ""),
        f"  Raw items:     {summary.raw_items_fetched}",
        f"  After norm:    {summary.items_after_normalization}",
        f"  After dedupe:  {summary.items_after_deduplication}",
        f"  Novel:         {summary.novel_items}",
        f"  Seen/not-emtd: {summary.previously_seen_items - summary.previously_emitted_items}",
        f"  Prev emitted:  {summary.previously_emitted_items}",
        f"  Suppressed:    {summary.suppressed_as_repeat}",
        f"  Updated items: {summary.updated_existing_items}",
        f"  Quality drops: {summary.items_dropped_for_quality}",
    ]

    if verbose:
        lines.append("")
        lines.append("  Per-feed diagnostics:")
        for d in summary.per_feed_diagnostics:
            status_str = d.status.upper()
            err_str = f" [{d.error}]" if d.error else ""
            etag_str = " (ETag hit)" if d.etag_hit else ""
            lines.append(
                f"    {d.feed_name}: {status_str}{etag_str} — {d.items_fetched} items{err_str}"
            )

    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="feed-ingestor",
        description="Fertility feed ingestor — fetch, normalize, deduplicate RSS/Atom articles.",
    )
    root.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-feed diagnostics."
    )

    sub = root.add_subparsers(dest="command", required=True)

    # -- ingest --------------------------------------------------------------
    ingest = sub.add_parser("ingest", help="Run the full ingestion pipeline.")
    ingest.add_argument("--config", metavar="PATH", help="Path to YAML/JSON config file.")
    ingest.add_argument(
        "--since-days",
        type=int,
        default=None,
        metavar="N",
        dest="since_days",
        help="Only include items published within the last N days.",
    )
    ingest.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        metavar="N",
        dest="lookback_days",
        help="Override ingest.lookback_days from config.",
    )
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of output items (for testing).",
    )
    ingest.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output file path.  Writes to stdout when omitted.",
    )
    ingest.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (default: jsonl).",
    )
    ingest.add_argument(
        "--history-path",
        metavar="PATH",
        default=None,
        dest="history_path",
        help="Path to the SQLite history database.",
    )
    ingest.add_argument(
        "--suppress-emitted",
        metavar="true|false",
        default=None,
        dest="suppress_emitted",
        help="Suppress items that were previously emitted (default: true).",
    )
    ingest.add_argument(
        "--suppress-seen",
        metavar="true|false",
        default=None,
        dest="suppress_seen",
        help="Suppress items that were seen but never emitted (default: false).",
    )
    ingest.add_argument(
        "--allow-repeat-after-days",
        type=int,
        default=None,
        metavar="N",
        dest="allow_repeat_after_days",
        help="Re-allow emitted items after N days.",
    )
    ingest.add_argument(
        "--force-refresh",
        action="store_true",
        dest="force_refresh",
        help=(
            "Bypass conditional GET (ETag/If-Modified-Since) and fetch feeds "
            "as full refresh for this run."
        ),
    )

    # -- mark-emitted --------------------------------------------------------
    emit = sub.add_parser(
        "mark-emitted",
        help="Mark items from a JSON/JSONL file as emitted in history.",
    )
    emit.add_argument(
        "--items",
        required=True,
        metavar="PATH",
        help="Path to JSON or JSONL file of NormalizedItem records.",
    )
    emit.add_argument(
        "--history-path",
        metavar="PATH",
        default=None,
        dest="history_path",
        help="Path to the SQLite history database.",
    )

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    parser = build_parser()
    args = parser.parse_args()

    # Logging setup — verbose goes to DEBUG, default to INFO
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    if args.command == "ingest":
        sys.exit(cmd_ingest(args))
    elif args.command == "mark-emitted":
        sys.exit(cmd_mark_emitted(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
