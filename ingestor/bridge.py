"""
Bridge: convert feed-ingestor NormalizedItem records into the text file
format consumed by the news-agent downstream tools.

The downstream tools (query_articles.py, podcast_generator.py,
linkedin_extractor.py) all read a custom fixed-width text format.
This module generates that format directly from NormalizedItem records,
bypassing the need to run rss_parser.py + query_articles.py when the
feed-ingestor is used as the ingestion layer.

The output is the "filtered_articles.txt" format (the POST-query format
that podcast_generator.py reads), because the ingestor already handles
date filtering. query_articles.py is therefore not needed in the
integrated pipeline.

CLI usage (standalone):
    python -m ingestor.bridge --input items.jsonl --output filtered_articles.txt

Programmatic usage:
    from ingestor.bridge import jsonl_to_filtered_articles_txt, write_filtered_articles
    text = jsonl_to_filtered_articles_txt(items)
    write_filtered_articles(items, "filtered_articles.txt")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from .models import NormalizedItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Return the netloc portion of a URL, or 'unknown'."""
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def _fmt_dt(dt: Optional[datetime]) -> str:
    """Format a datetime for the legacy text format."""
    if dt is None:
        return "Unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _article_content(item: NormalizedItem) -> str:
    """Return the richest available text content for an item."""
    return item.summary or item.content_snippet or item.title


# ---------------------------------------------------------------------------
# METADATA HEADER (mirrors query_articles.py save_filtered_articles output)
# ---------------------------------------------------------------------------


def _build_metadata_header(items: List[NormalizedItem], generated_at: str) -> str:
    """Build the metadata section that podcast_generator.py skips."""
    dates = [i.published_at for i in items if i.published_at is not None]
    if dates:
        date_range = f"{min(dates).date()} to {max(dates).date()}"
    else:
        date_range = "N/A"

    domains = [_extract_domain(i.canonical_url) for i in items if i.canonical_url]
    domain_counts = Counter(domains)
    unique_sources = len(domain_counts)
    top_sources = domain_counts.most_common(5)

    content_lengths = [len(_article_content(i)) for i in items]
    avg_len = int(sum(content_lengths) / len(content_lengths)) if content_lengths else 0
    missing_pub = sum(1 for i in items if i.published_at is None)
    fallback_used = sum(
        1 for i in items
        if (i.used_fallback_date or (i.published_at is None and (i.updated_at or i.fetched_at)))
    )

    occurrence_distribution = Counter([1] * len(items))  # always 1 per unique article

    lines = [
        "FILTERED ARTICLES - METADATA REPORT",
        "=" * 60,
        "",
        "QUERY INFORMATION:",
        "-" * 20,
        f"Date Range Requested: (feed-ingestor managed)",
        f"Actual Date Range: {date_range}",
        f"Total Articles Found: {len(items)}",
        f"Generated At: {generated_at}",
        f"Source File: feed-ingestor JSONL",
        "",
        "SOURCE ANALYSIS:",
        "-" * 15,
        f"Unique Sources: {unique_sources}",
        "Top Sources:",
    ]
    for domain, count in top_sources:
        lines.append(f"  • {domain}: {count} article{'s' if count > 1 else ''}")
    lines.append("")

    lines += [
        "CONTENT ANALYSIS:",
        "-" * 16,
        f"Average Content Length: {avg_len} characters",
        f"Missing Published Date: {missing_pub}",
        f"Used Fallback Freshness Date: {fallback_used}",
        "",
        "OCCURRENCE DISTRIBUTION:",
        "-" * 23,
        f"  1 occurrence(s): {len(items)} articles",
        "",
        "=" * 60,
        "ARTICLE DETAILS",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Article body (mirrors query_articles.py save_filtered_articles article block)
# ---------------------------------------------------------------------------


def _build_article_block(item: NormalizedItem, number: int) -> str:
    """Render a single article in the filtered_articles.txt block format."""
    content = _article_content(item)
    source = item.source_name or _extract_domain(item.canonical_url)
    authors_str = ", ".join(item.authors) if item.authors else ""
    # Backward compatibility: older JSONL files may not include the new
    # freshness fields yet, so derive them from existing timestamps.
    freshness_dt = (
        item.effective_freshness_at
        or item.published_at
        or item.updated_at
        or item.fetched_at
    )
    date_source = item.date_source or "unknown"
    if date_source in ("", "unknown", "none"):
        if item.published_at:
            date_source = "published_at"
        elif item.updated_at:
            date_source = "updated_at"
        else:
            date_source = "fetched_at"
    used_fallback = bool(item.used_fallback_date or (not item.published_at and freshness_dt is not None))
    freshness_conf = item.freshness_confidence or (
        "high" if date_source == "published_at" else "medium" if date_source == "updated_at" else "low"
    )
    lines = [
        f"Article {number}",
        "-" * 30,
        f"Title: {item.title}",
        f"URL: {item.canonical_url}",
        f"Source: {source}",
        f"Authors: {authors_str}",
        f"Published: {_fmt_dt(item.published_at)}",
        f"Updated: {_fmt_dt(item.updated_at)}",
        f"Freshness Date: {_fmt_dt(freshness_dt)}",
        f"Fetched At: {_fmt_dt(item.fetched_at)}",
        f"First Seen At: {_fmt_dt(item.previous_first_seen_at)}",
        f"Date Source: {date_source}",
        f"Used Fallback Date: {'true' if used_fallback else 'false'}",
        f"Freshness Confidence: {freshness_conf}",
        "Occurrences: 1",
        f"Content Length: {len(content)} characters",
        f"Content: {content}",
        "",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def items_to_filtered_articles_txt(items: List[NormalizedItem]) -> str:
    """
    Convert a list of :class:`NormalizedItem` records to the
    ``filtered_articles.txt`` string format consumed by ``podcast_generator.py``
    and ``linkedin_extractor.py``.

    The output is identical in structure to what ``query_articles.py`` produces,
    so no changes to downstream tools are required.
    """
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    parts = [_build_metadata_header(items, generated_at)]
    for i, item in enumerate(items, 1):
        parts.append(_build_article_block(item, i))
    return "\n".join(parts)


def write_filtered_articles(items: List[NormalizedItem], output_path: str) -> int:
    """
    Write *items* to *output_path* in ``filtered_articles.txt`` format.

    Returns the number of articles written.
    """
    text = items_to_filtered_articles_txt(items)
    Path(output_path).write_text(text, encoding="utf-8")
    return len(items)


def load_jsonl(path: str) -> List[NormalizedItem]:
    """Load NormalizedItem records from a JSON or JSONL file."""
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        records = json.loads(raw_text)
    else:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    return [NormalizedItem.model_validate(r) for r in records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert feed-ingestor JSONL output to filtered_articles.txt format.",
        epilog=(
            "Example:\n"
            "  python -m ingestor.bridge "
            "--input items.jsonl --output filtered_articles.txt"
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="JSONL file from feed-ingestor.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path. Writes to stdout when omitted.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.input)
    text = items_to_filtered_articles_txt(items)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {len(items)} articles to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
