"""
Tests for ingestor/parse.py

Covers:
- DOI extraction from various text formats
- PMID extraction from URL and prefix patterns
- Date parsing and UTC conversion
- HTML stripping
- Malformed / minimal feed entry handling
- Full parse_entry round-trip
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from time import struct_time

import pytest

from ingestor.parse import (
    canonical_url_from_entry,
    extract_authors,
    extract_content_blocks,
    extract_doi,
    extract_pmid,
    parse_entry,
    parse_feed,
    strip_html,
    struct_time_to_utc,
)
from tests.conftest import MockEntry, MockFeed


# ---------------------------------------------------------------------------
# DOI extraction
# ---------------------------------------------------------------------------


class TestExtractDoi:
    def test_plain_doi(self):
        assert extract_doi("10.1093/humrep/dead001") == "10.1093/humrep/dead001"

    def test_doi_in_url(self):
        text = "https://doi.org/10.1016/j.fertnstert.2024.01.001"
        assert extract_doi(text) == "10.1016/j.fertnstert.2024.01.001"

    def test_doi_with_trailing_punctuation(self):
        text = "See 10.1093/humrep/dead001."
        assert extract_doi(text) == "10.1093/humrep/dead001"

    def test_doi_in_sentence(self):
        text = "Published as doi:10.1038/s41591-024-01234-5 in Nature Medicine."
        assert extract_doi(text) == "10.1038/s41591-024-01234-5"

    def test_no_doi(self):
        assert extract_doi("No DOI here.") is None

    def test_empty_string(self):
        assert extract_doi("") is None

    def test_doi_in_dc_identifier(self):
        text = "doi:10.1093/humrep/dead999"
        assert extract_doi(text) == "10.1093/humrep/dead999"

    def test_nature_doi(self):
        text = "https://www.nature.com/articles/s41591-024-02873-3"
        # This is NOT a DOI (no 10. prefix)
        assert extract_doi(text) is None

    def test_doi_minimum_registrant(self):
        # DOI registrants have at least 4 digits
        assert extract_doi("10.123/short") is None
        assert extract_doi("10.1234/valid") == "10.1234/valid"


# ---------------------------------------------------------------------------
# PMID extraction
# ---------------------------------------------------------------------------


class TestExtractPmid:
    def test_pubmed_url(self):
        text = "https://pubmed.ncbi.nlm.nih.gov/38500001/"
        assert extract_pmid(text) == "38500001"

    def test_ncbi_pubmed_url(self):
        text = "https://www.ncbi.nlm.nih.gov/pubmed/12345678"
        assert extract_pmid(text) == "12345678"

    def test_pmid_prefix(self):
        text = "pmid:38500001"
        assert extract_pmid(text) == "38500001"

    def test_pmid_prefix_case_insensitive(self):
        text = "PMID:38500001"
        assert extract_pmid(text) == "38500001"

    def test_no_pmid(self):
        assert extract_pmid("No PMID here.") is None

    def test_empty(self):
        assert extract_pmid("") is None


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


class TestStructTimeToUtc:
    def _make_struct(self, *args: int) -> struct_time:
        # struct_time fields: (year, month, day, hour, min, sec, wday, yday, isdst)
        padded = list(args) + [0] * (9 - len(args))
        return struct_time(padded)

    def test_valid_date(self):
        st = self._make_struct(2024, 3, 1, 10, 0, 0)
        dt = struct_time_to_utc(st)
        assert dt == datetime(2024, 3, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_none_input(self):
        assert struct_time_to_utc(None) is None

    def test_pre_epoch_returns_none(self):
        st = self._make_struct(1900, 1, 1, 0, 0, 0)
        assert struct_time_to_utc(st) is None

    def test_timezone_is_utc(self):
        st = self._make_struct(2024, 6, 15, 8, 30, 0)
        dt = struct_time_to_utc(st)
        assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class TestStripHtml:
    def test_removes_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self):
        assert "&amp;" not in strip_html("Fish &amp; chips")
        assert "Fish & chips" == strip_html("Fish &amp; chips")

    def test_collapses_whitespace(self):
        result = strip_html("  lots   of   spaces  ")
        assert result == "lots of spaces"

    def test_max_length_truncates(self):
        long_text = "word " * 200
        result = strip_html(long_text, max_length=50)
        assert len(result) <= 50

    def test_empty_string(self):
        assert strip_html("") == ""


# ---------------------------------------------------------------------------
# Malformed / empty entry handling
# ---------------------------------------------------------------------------


class TestParseEntry:
    def test_minimal_entry_with_title_and_url(self):
        entry = MockEntry(title="A title", link="https://example.com/1")
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is not None
        assert result["title"] == "A title"
        assert result["canonical_url"] == "https://example.com/1"

    def test_entry_without_title_and_url_returns_none(self):
        entry = MockEntry(title="", link="")
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is None

    def test_entry_with_only_title_is_valid(self):
        entry = MockEntry(title="Just a title", link="")
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is not None

    def test_doi_extracted_from_link(self):
        entry = MockEntry(
            link="https://doi.org/10.1093/humrep/dead001",
            title="Endometriosis and IVF",
        )
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is not None
        assert result["doi"] == "10.1093/humrep/dead001"

    def test_pmid_extracted_from_link(self):
        entry = MockEntry(
            link="https://pubmed.ncbi.nlm.nih.gov/38500001/",
            title="IVF outcomes",
        )
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is not None
        assert result["pmid"] == "38500001"

    def test_summary_html_stripped(self):
        entry = MockEntry(
            title="Test",
            link="https://example.com/1",
            summary="<p>Abstract: <b>important</b> findings.</p>",
        )
        result = parse_entry(entry, "https://example.com/feed.rss", "Test Feed")
        assert result is not None
        assert "<" not in result["summary"]
        assert "important" in result["summary"]

    def test_feed_tags_appended(self):
        entry = MockEntry(title="Test", link="https://example.com/1")
        result = parse_entry(
            entry,
            "https://example.com/feed.rss",
            "Test Feed",
            feed_tags=["fertility", "ivf"],
        )
        assert result is not None
        assert "fertility" in result["tags"]

    def test_exception_in_entry_returns_none(self):
        """An entry that raises on attribute access should not crash the pipeline."""
        # Simulate a malformed entry by passing something that's not an entry
        result = parse_entry(None, "https://example.com/feed.rss", "Test Feed")
        assert result is None


class TestParseFeed:
    def test_parses_multiple_entries(self):
        entries = [
            MockEntry(title=f"Article {i}", link=f"https://example.com/{i}")
            for i in range(5)
        ]
        feed = MockFeed(entries=entries)
        results = parse_feed(feed, "https://example.com/feed.rss", "Test Feed")
        assert len(results) == 5

    def test_empty_feed(self):
        feed = MockFeed(entries=[])
        results = parse_feed(feed, "https://example.com/feed.rss", "Test Feed")
        assert results == []

    def test_invalid_entries_skipped(self):
        entries = [
            MockEntry(title="", link=""),  # will be dropped
            MockEntry(title="Valid", link="https://example.com/1"),
        ]
        feed = MockFeed(entries=entries)
        results = parse_feed(feed, "https://example.com/feed.rss", "Test Feed")
        assert len(results) == 1
        assert results[0]["title"] == "Valid"
