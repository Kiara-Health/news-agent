"""
LLM Caller
==========
Single OpenAI wrapper used by every pipeline stage.

Caller signature (kept backward-compatible with prior local-LLM callers):

    call_openai(prompt: str, config: Dict, timeout: int = 60) -> Optional[str]

Config keys read
----------------
openai_model  : str  — model name, defaults to "gpt-4.1"
openai_api_key: str  — API key; OPENAI_API_KEY env var takes precedence

JSON mode
---------
If the prompt contains the phrase "valid JSON object" the call is sent with
``response_format={"type": "json_object"}`` so the model is guaranteed to return
parseable JSON.  All other calls get plain-text output.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI, APIError, APITimeoutError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4.1"
_JSON_TRIGGER = "valid JSON object"

# Load local .env once on module import so subprocess runs pick up OPENAI_API_KEY
# without requiring manual shell export. Environment variables already set in the
# process are preserved (override=False).
_ENV_PATH = Path(__file__).with_name(".env")
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
else:
    load_dotenv(override=False)


def call_openai(
    prompt: str,
    config: Dict,
    timeout: int = 60,
) -> Optional[str]:
    """
    Send *prompt* to the OpenAI Chat Completions API and return the response text.

    Returns ``None`` on any error so callers can fall back gracefully.
    """
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or config.get("openai_api_key")
        or ""
    )
    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set. "
            "Export it as an environment variable or add 'openai_api_key' to config."
        )
        return None

    model = config.get("openai_model") or _DEFAULT_MODEL
    json_mode = _JSON_TRIGGER in prompt

    client = OpenAI(api_key=api_key, timeout=timeout)
    kwargs: Dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except APITimeoutError:
        logger.warning("OpenAI request timed out (model=%s, timeout=%ds).", model, timeout)
    except APIError as exc:
        logger.warning("OpenAI API error: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error calling OpenAI: %s", exc)
    return None
