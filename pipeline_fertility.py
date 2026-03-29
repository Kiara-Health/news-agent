#!/usr/bin/env python3
"""
Fertility & Reproductive Medicine Newsletter Pipeline
=====================================================

End-to-end orchestrator that connects:

  1. feed-ingestor  — fetches fertility RSS/Atom feeds, normalises,
                      deduplicates, and annotates novelty
  2. bridge         — converts NormalizedItem JSONL → filtered_articles.txt
                      (the text format expected by downstream tools)
  3. podcast_generator.py   — scores, selects, summarises → podcast script
  4. linkedin_extractor.py  — podcast + articles → LinkedIn posts
  5. banner_prompt_generator.py — podcast → image-generation prompt (OpenAI)

Prerequisites
-------------
* feed-ingestor package installed  (pip install -e ../feed-ingestor)
* OPENAI_API_KEY set in the environment  (article summarisation and newsletter composition)
* OPENAI_API_KEY set in environment or .env  (banner prompt generation)
* ../feed-ingestor/news-sources.yaml configured with fertility feed URLs

Usage examples
--------------
# Standard twice-weekly run (last 7 days)
python pipeline_fertility.py

# Custom lookback and explicit output directory
python pipeline_fertility.py --days 14 --output my_output

# Skip banner generation (no OpenAI key required)
python pipeline_fertility.py --skip-banner

# Dry-run: ingest only, no LLM steps
python pipeline_fertility.py --ingest-only

# Re-run from existing JSONL (skip live fetching)
python pipeline_fertility.py --from-jsonl output/run_XYZ/raw/items.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths — adjust if the repo layout changes
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.absolute()
_INGESTOR_DIR = _SCRIPT_DIR.parent / "feed-ingestor"
_DEFAULT_INGESTOR_CONFIG = _INGESTOR_DIR / "news-sources.yaml"
_DEFAULT_FERTILITY_CONFIG = _SCRIPT_DIR / "config.fertility.json"
_DEFAULT_HISTORY_DB = _SCRIPT_DIR / "ingestor_history.db"

# Load .env from news-agent directory so banner pre-check sees OPENAI_API_KEY.
load_dotenv(_SCRIPT_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{_ts()}] {level:7s} {msg}", flush=True)


def log_step(step: str, success: bool = True) -> None:
    icon = "✅" if success else "❌"
    log(f"{icon} {step}", level="STEP")


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------


def run(cmd: List[str], step: str, cwd: Optional[Path] = None) -> bool:
    """Run *cmd* as a subprocess, stream output, and return success flag."""
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd or _SCRIPT_DIR),
        capture_output=False,   # stream directly to terminal
        text=True,
    )
    ok = result.returncode == 0
    log_step(step, ok)
    return ok


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_ingest(
    run_dir: Path,
    ingestor_config: Path,
    history_db: Path,
    since_days: int,
    jsonl_out: Path,
    force_refresh: bool = False,
) -> bool:
    """Run feed-ingestor to produce a JSONL of NormalizedItem records."""
    cmd = [
        sys.executable, "-m", "ingestor.cli",
        "--verbose",
        "ingest",
        "--config", str(ingestor_config.resolve()),
        "--since-days", str(since_days),
        "--history-path", str(history_db.resolve()),
        "--output", str(jsonl_out.resolve()),
        "--format", "jsonl",
    ]
    if force_refresh:
        cmd.append("--force-refresh")
    return run(cmd, "Feed ingestion", cwd=_INGESTOR_DIR)


def step_bridge(jsonl_path: Path, filtered_out: Path) -> bool:
    """Convert ingestor JSONL to filtered_articles.txt for downstream tools."""
    cmd = [
        sys.executable, "-m", "ingestor.bridge",
        "--input", str(jsonl_path.resolve()),
        "--output", str(filtered_out.resolve()),
    ]
    ok = run(cmd, "Bridge: JSONL → filtered_articles.txt", cwd=_INGESTOR_DIR)
    if ok:
        log(f"Filtered articles written to: {filtered_out}")
    return ok


def step_podcast(
    filtered_articles: Path,
    podcast_out: Path,
    fertility_config: Path,
    duration: Optional[int],
) -> bool:
    """Run podcast_generator.py to produce a podcast script."""
    cmd = [
        sys.executable, "podcast_generator.py",
        "--input", str(filtered_articles),
        "--output", str(podcast_out),
        "--config", str(fertility_config),
    ]
    if duration:
        cmd += ["--duration", str(duration)]
    return run(cmd, "Podcast script generation", cwd=_SCRIPT_DIR)


def step_linkedin(
    podcast_file: Path,
    filtered_articles: Path,
    linkedin_out: Path,
    linkedin_compact_out: Path,
    fertility_config: Path,
) -> bool:
    """Generate standard and compact LinkedIn posts."""
    ok1 = run(
        [
            sys.executable, "linkedin_extractor.py",
            "--podcast", str(podcast_file),
            "--articles", str(filtered_articles),
            "--config", str(fertility_config),
            "--output", str(linkedin_out),
        ],
        "LinkedIn post (standard)",
        cwd=_SCRIPT_DIR,
    )
    ok2 = run(
        [
            sys.executable, "linkedin_extractor.py",
            "--podcast", str(podcast_file),
            "--articles", str(filtered_articles),
            "--config", str(fertility_config),
            "--compact",
            "--output", str(linkedin_compact_out),
        ],
        "LinkedIn post (compact)",
        cwd=_SCRIPT_DIR,
    )
    return ok1 and ok2


def step_banner(
    podcast_file: Path,
    banner_out: Path,
    domain: str = "fertility and reproductive medicine",
) -> bool:
    """Generate a LinkedIn banner image prompt via OpenAI."""
    if not os.environ.get("OPENAI_API_KEY"):
        log("OPENAI_API_KEY not set — skipping banner prompt generation.", "WARN")
        return True  # non-fatal

    cmd = [
        sys.executable, "banner_prompt_generator.py",
        "--podcast", str(podcast_file),
        "--output", str(banner_out),
    ]
    return run(cmd, "Banner image prompt generation", cwd=_SCRIPT_DIR)


def step_newsletter(
    report_json: Path,
    newsletter_md: Path,
    newsletter_json: Path,
    fertility_config: Path,
) -> bool:
    """Run newsletter_composer.py to produce the curated newsletter."""
    cmd = [
        sys.executable, "newsletter_composer.py",
        "--report", str(report_json),
        "--config", str(fertility_config),
        "--output-md", str(newsletter_md),
        "--output-json", str(newsletter_json),
    ]
    return run(cmd, "Newsletter composition", cwd=_SCRIPT_DIR)


def step_mark_emitted(jsonl_path: Path, history_db: Path) -> None:
    """Optionally mark the output items as emitted in history."""
    cmd = [
        sys.executable, "-m", "ingestor.cli",
        "mark-emitted",
        "--items", str(jsonl_path.resolve()),
        "--history-path", str(history_db.resolve()),
    ]
    run(cmd, "Mark items as emitted", cwd=_INGESTOR_DIR)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _count_articles(filtered_articles: Path) -> int:
    if not filtered_articles.exists():
        return 0
    text = filtered_articles.read_text(encoding="utf-8")
    return text.count("Article ") - text.count("ARTICLE DETAILS")


def write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    steps_status: dict,
    article_count: int,
) -> None:
    summary = run_dir / "pipeline_summary.txt"
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("FERTILITY & REPRODUCTIVE MEDICINE PIPELINE SUMMARY\n")
        fh.write("=" * 55 + "\n\n")
        fh.write(f"Run directory : {run_dir}\n")
        fh.write(f"Completed at  : {_ts()}\n")
        fh.write(f"Lookback days : {args.days}\n")
        fh.write(f"Articles found: {article_count}\n\n")
        fh.write("STEP STATUS:\n")
        for step, ok in steps_status.items():
            icon = "✅" if ok else "❌"
            fh.write(f"  {icon} {step}\n")
        fh.write("\nGENERATED FILES:\n")
        for f in sorted(run_dir.rglob("*")):
            if f.is_file() and f.suffix in (".txt", ".jsonl", ".json"):
                fh.write(f"  {f.relative_to(run_dir)}\n")
    log(f"Summary written to: {summary}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = Path(args.output) / run_id

    # Directory structure
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    (run_dir / "final").mkdir(parents=True, exist_ok=True)
    log(f"Run ID: {run_id}  |  Output: {run_dir}")

    # File paths
    jsonl_path        = run_dir / "raw" / "items.jsonl"
    filtered_path     = run_dir / "processed" / "filtered_articles.txt"
    podcast_path      = run_dir / "final" / "podcast_script.txt"
    report_json_path  = run_dir / "final" / "selected_articles_report.json"
    newsletter_md     = run_dir / "final" / "newsletter.md"
    newsletter_json   = run_dir / "final" / "newsletter.json"
    linkedin_path     = run_dir / "final" / "linkedin_post.txt"
    linkedin_compact  = run_dir / "final" / "linkedin_post_compact.txt"
    banner_path       = run_dir / "final" / "banner_image_prompt.txt"

    # Config files
    ingestor_cfg   = Path(args.ingestor_config)
    fertility_cfg  = Path(args.fertility_config)
    history_db     = Path(args.history_db)

    steps_status: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Ingest (or reuse existing JSONL)
    # ------------------------------------------------------------------
    if args.from_jsonl:
        src = Path(args.from_jsonl)
        shutil.copy(src, jsonl_path)
        log(f"Using existing JSONL: {src}")
        steps_status["Ingestion (reused)"] = True
    else:
        ok = step_ingest(
            run_dir, ingestor_cfg, history_db, args.days, jsonl_path,
            force_refresh=bool(args.force_refresh),
        )
        steps_status["Ingestion"] = ok
        if not ok:
            log("Ingestion failed — aborting pipeline.", "ERROR")
            write_summary(run_dir, args, steps_status, 0)
            return 1

    if args.ingest_only:
        log("--ingest-only set: stopping after ingestion.")
        write_summary(run_dir, args, steps_status, 0)
        return 0

    # ------------------------------------------------------------------
    # Step 2: Bridge JSONL → filtered_articles.txt
    # ------------------------------------------------------------------
    ok = step_bridge(jsonl_path, filtered_path)
    steps_status["Bridge (JSONL→TXT)"] = ok
    if not ok:
        log("Bridge conversion failed — aborting.", "ERROR")
        write_summary(run_dir, args, steps_status, 0)
        return 1

    article_count = _count_articles(filtered_path)
    log(f"{article_count} articles available for downstream processing.")
    if article_count == 0:
        log("No articles to process after ingestion — exiting.", "WARN")
        write_summary(run_dir, args, steps_status, 0)
        return 0

    # ------------------------------------------------------------------
    # Step 3: Podcast generation
    # ------------------------------------------------------------------
    ok = step_podcast(filtered_path, podcast_path, fertility_cfg, args.duration)
    steps_status["Podcast generation"] = ok
    if not ok:
        log("Podcast generation failed — continuing to remaining steps.", "WARN")

    # ------------------------------------------------------------------
    # Step 4: Newsletter composition (consumes the consolidated report JSON)
    # ------------------------------------------------------------------
    if report_json_path.exists():
        ok = step_newsletter(report_json_path, newsletter_md, newsletter_json, fertility_cfg)
        steps_status["Newsletter composition"] = ok
    else:
        log("No consolidated report JSON found — skipping newsletter composition.", "WARN")
        steps_status["Newsletter composition"] = False

    # ------------------------------------------------------------------
    # Step 5: LinkedIn posts
    # ------------------------------------------------------------------
    if podcast_path.exists():
        ok = step_linkedin(
            podcast_path, filtered_path,
            linkedin_path, linkedin_compact,
            fertility_cfg,
        )
        steps_status["LinkedIn posts"] = ok
    else:
        log("No podcast script found — skipping LinkedIn post generation.", "WARN")
        steps_status["LinkedIn posts"] = False

    # ------------------------------------------------------------------
    # Step 5: Banner image prompt
    # ------------------------------------------------------------------
    if not args.skip_banner and podcast_path.exists():
        # Read domain from fertility config
        domain = "fertility and reproductive medicine"
        try:
            cfg_data = json.loads(fertility_cfg.read_text(encoding="utf-8"))
            domain = cfg_data.get("publication_settings", {}).get("banner_domain", domain)
        except Exception:
            pass
        ok = step_banner(podcast_path, banner_path, domain)
        steps_status["Banner prompt"] = ok
    else:
        log("Skipping banner prompt generation.", "INFO")
        steps_status["Banner prompt"] = None

    # ------------------------------------------------------------------
    # Optional: mark-emitted
    # ------------------------------------------------------------------
    if args.mark_emitted and jsonl_path.exists():
        step_mark_emitted(jsonl_path, history_db)
        steps_status["Mark emitted"] = True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    write_summary(run_dir, args, steps_status, article_count)

    failed = [s for s, ok in steps_status.items() if ok is False]
    if failed:
        log(f"Pipeline completed with failures: {', '.join(failed)}", "WARN")
        print(f"\n{'='*60}")
        print(f"⚠️  Pipeline completed with {len(failed)} failed step(s).")
        print(f"📂  Output: {run_dir}")
        return 1
    else:
        print(f"\n{'='*60}")
        print(f"🎉  Pipeline completed successfully!")
        print(f"📂  Output:     {run_dir}")
        print(f"📝  Podcast:    {podcast_path}")
        if newsletter_md.exists():
            print(f"📰  Newsletter: {newsletter_md}")
        print(f"💼  LinkedIn:   {linkedin_path}")
        if banner_path.exists():
            print(f"🖼️   Banner:     {banner_path}")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline_fertility.py",
        description="Fertility & Reproductive Medicine end-to-end newsletter pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Root output directory (default: output).",
    )
    parser.add_argument(
        "--days", "-d", type=int, default=7,
        help="Look back N days for new articles (default: 7).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        dest="force_refresh",
        help=(
            "Force feed refresh by bypassing ETag/If-Modified-Since checks "
            "during ingestion."
        ),
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Podcast target duration in seconds (overrides config).",
    )
    parser.add_argument(
        "--ingestor-config",
        default=str(_DEFAULT_INGESTOR_CONFIG),
        dest="ingestor_config",
        help="Path to feed-ingestor YAML config (default: feed-ingestor/news-sources.yaml).",
    )
    parser.add_argument(
        "--fertility-config",
        default=str(_DEFAULT_FERTILITY_CONFIG),
        dest="fertility_config",
        help="Path to fertility JSON config (default: config.fertility.json).",
    )
    parser.add_argument(
        "--history-db",
        default=str(_DEFAULT_HISTORY_DB),
        dest="history_db",
        help="Path to feed-ingestor history SQLite DB.",
    )
    parser.add_argument(
        "--from-jsonl",
        default=None,
        dest="from_jsonl",
        metavar="PATH",
        help="Skip live ingestion and use an existing JSONL file.",
    )
    parser.add_argument(
        "--ingest-only", action="store_true", dest="ingest_only",
        help="Stop after ingestion (no LLM steps).",
    )
    parser.add_argument(
        "--skip-banner", action="store_true", dest="skip_banner",
        help="Skip banner image prompt generation (no OpenAI needed).",
    )
    parser.add_argument(
        "--mark-emitted", action="store_true", dest="mark_emitted",
        help="Mark output items as emitted in history after the run.",
    )

    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
