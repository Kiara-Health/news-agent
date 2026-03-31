"""
Configuration loading for the fertility feed ingestor.

Supports YAML (preferred) and JSON config files.  Falls back to a safe
default IngestorConfig if no path is given.

Feed URLs are never embedded here; they live exclusively in the config file
so operators can add/remove sources without touching code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import IngestorConfig

logger = logging.getLogger(__name__)


def load_config(path: Optional[str] = None) -> IngestorConfig:
    """
    Load :class:`IngestorConfig` from a YAML or JSON file.

    Args:
        path: Filesystem path to a ``.yaml``, ``.yml``, or ``.json`` config
              file.  When *None* the function returns a default
              :class:`IngestorConfig` with no feeds configured.

    Returns:
        Validated :class:`IngestorConfig` instance.

    Raises:
        FileNotFoundError: When *path* is supplied but the file does not exist.
        RuntimeError: When the file cannot be parsed.
    """
    if path is None:
        logger.info("No config path provided; using default IngestorConfig (no feeds).")
        return IngestorConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            suffix = config_path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                # pyyaml is a soft dependency; give a clear error if absent.
                try:
                    import yaml  # type: ignore[import-untyped]
                except ImportError as exc:
                    raise RuntimeError(
                        "PyYAML is required for YAML config files. "
                        "Install it with: pip install pyyaml"
                    ) from exc
                raw = yaml.safe_load(fh) or {}
            elif suffix == ".json":
                raw = json.load(fh)
            else:
                raise ValueError(
                    f"Unsupported config format '{suffix}'. Use .yaml or .json."
                )
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse config file '{config_path}': {exc}"
        ) from exc

    logger.info("Loaded config from %s", config_path)
    return IngestorConfig.model_validate(raw)


def merge_cli_overrides(
    cfg: IngestorConfig,
    *,
    history_path: Optional[str] = None,
    suppress_emitted: Optional[bool] = None,
    suppress_seen: Optional[bool] = None,
    lookback_days: Optional[int] = None,
    allow_repeat_after_days: Optional[int] = None,
) -> IngestorConfig:
    """
    Return a new :class:`IngestorConfig` with CLI flag values overlaid on top
    of file-level settings.

    Only non-None arguments are applied so that callers can pass every
    possible CLI arg without worrying about accidental overwrites.
    """
    # Build updated sub-configs as plain dicts to avoid Pydantic immutability
    history_dict = cfg.history.model_dump()
    ingest_dict = cfg.ingest.model_dump()

    if history_path is not None:
        history_dict["path"] = history_path
    if suppress_emitted is not None:
        history_dict["suppress_emitted"] = suppress_emitted
    if suppress_seen is not None:
        history_dict["suppress_seen"] = suppress_seen
    if allow_repeat_after_days is not None:
        history_dict["allow_repeat_after_days"] = allow_repeat_after_days
    if lookback_days is not None:
        ingest_dict["lookback_days"] = lookback_days

    return IngestorConfig(
        feeds=cfg.feeds,
        ingest=cfg.ingest.model_copy(update=ingest_dict),
        dedupe=cfg.dedupe,
        history=cfg.history.model_copy(update=history_dict),
    )
