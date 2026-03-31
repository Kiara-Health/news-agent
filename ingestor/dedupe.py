"""
Within-run deduplication for the fertility feed ingestor.

Strategy (mirrors the requirements spec):

1. **DOI match** — exact equality after lower-casing.
2. **PMID match** — exact equality.
3. **Canonical URL match** — normalized URL equality.
4. **Fuzzy title + date proximity** — SequenceMatcher ratio ≥ threshold AND
   publication dates within 7 days of each other.

When multiple items are judged to be the same article the *richest* version
is kept (highest richness score from :func:`normalize.richness_score`).
Distinct articles on similar *topics* are not merged — only items that share
a strong identifier OR whose titles are nearly identical AND dates are close
are considered duplicates.

Design tradeoff: using Python's stdlib ``difflib.SequenceMatcher`` avoids an
extra dependency (``rapidfuzz``).  For the small batch sizes expected in a
twice-weekly newsletter pipeline (≤ 200 items / run) the O(n²) title
comparison is fast enough.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from .models import DedupeConfig, NormalizedItem
from .normalize import normalize_url, richness_score

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.85
_DATE_PROXIMITY_DAYS = 7


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _normalize_title_for_cmp(title: str) -> str:
    """Lower-case, collapse whitespace, remove leading news-wire prefixes."""
    import re

    t = title.lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Strip common prefixes that differ across aggregators
    t = re.sub(r"^(breaking|exclusive|update|correction|retraction):\s*", "", t)
    return t


def title_similarity(a: str, b: str) -> float:
    """
    Return a SequenceMatcher similarity ratio in [0, 1] between two titles.

    Input strings are normalized before comparison.
    """
    na = _normalize_title_for_cmp(a)
    nb = _normalize_title_for_cmp(b)
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _dates_close(a: Optional[object], b: Optional[object], days: int = _DATE_PROXIMITY_DAYS) -> bool:
    """Return True when both dates are present and within *days* of each other."""
    if a is None or b is None:
        # If one date is missing we accept the title match alone
        return True
    try:
        diff = abs((a - b).total_seconds())  # type: ignore[operator]
        return diff <= days * 86400
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Cluster building
# ---------------------------------------------------------------------------


def _group_items(
    items: List[NormalizedItem],
    threshold: float,
) -> List[List[int]]:
    """
    Return a list of index-clusters.  Each cluster contains the indices of
    items that are considered duplicates of each other.

    Uses union-find (disjoint set) internally for O(n α(n)) merging.
    """
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Exact identifier lookups — O(n) with hash maps
    doi_map: Dict[str, int] = {}
    pmid_map: Dict[str, int] = {}
    url_map: Dict[str, int] = {}

    for i, item in enumerate(items):
        if item.doi:
            key = item.doi.lower()
            if key in doi_map:
                union(i, doi_map[key])
            else:
                doi_map[key] = i

        if item.pmid:
            if item.pmid in pmid_map:
                union(i, pmid_map[item.pmid])
            else:
                pmid_map[item.pmid] = i

        norm_url = normalize_url(item.canonical_url)
        if norm_url:
            if norm_url in url_map:
                union(i, url_map[norm_url])
            else:
                url_map[norm_url] = i

    # Fuzzy title comparison — O(n²) but n is small per run
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue  # already in same cluster
            sim = title_similarity(items[i].title, items[j].title)
            if sim >= threshold and _dates_close(
                items[i].published_at, items[j].published_at
            ):
                logger.debug(
                    "Fuzzy title match (%.2f): '%s' ≈ '%s'",
                    sim,
                    items[i].title[:60],
                    items[j].title[:60],
                )
                union(i, j)

    # Collect clusters
    cluster_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        cluster_map.setdefault(root, []).append(i)

    return list(cluster_map.values())


# ---------------------------------------------------------------------------
# Keep-richest selection
# ---------------------------------------------------------------------------


def _pick_best(candidates: List[NormalizedItem]) -> NormalizedItem:
    """
    From a cluster of duplicate items, return the richest one.

    When scores are tied the item with the earliest position in the input
    list is preferred (stable ordering).
    """
    return max(candidates, key=richness_score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deduplicate(
    items: List[NormalizedItem],
    cfg: Optional[DedupeConfig] = None,
) -> Tuple[List[NormalizedItem], int]:
    """
    Remove within-run duplicates from *items*.

    Args:
        items: List of normalized items from all feeds combined.
        cfg:   Deduplication settings.  Uses defaults when ``None``.

    Returns:
        A 2-tuple of:
        - Deduplicated list of :class:`NormalizedItem` (one per cluster,
          keeping the richest version).
        - Number of items removed.
    """
    if not items:
        return [], 0

    threshold = cfg.title_similarity_threshold if cfg else _DEFAULT_THRESHOLD
    clusters = _group_items(items, threshold)

    kept: List[NormalizedItem] = []
    removed = 0

    for cluster in clusters:
        if len(cluster) == 1:
            kept.append(items[cluster[0]])
        else:
            candidates = [items[i] for i in cluster]
            best = _pick_best(candidates)
            kept.append(best)
            removed += len(cluster) - 1
            logger.debug(
                "Cluster of %d duplicates → kept '%s'.",
                len(cluster),
                best.title[:80],
            )

    logger.info(
        "Deduplication: %d items in → %d out (%d removed).",
        len(items),
        len(kept),
        removed,
    )
    return kept, removed
