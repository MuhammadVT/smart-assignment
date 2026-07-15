"""
Model plumbing for the grounded slot-choice call: prompt -> raw dict, routed
through the shared backend under the `slotpick` role model. Injectable so tests
drive the selector with a fake and no network.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def generate_slot_choice(config: "Config", prompt: str) -> dict:
    """Call the configured backend (slotpick role model) and parse the JSON."""
    from smart_assignment.shared.config import ROLE_SLOTPICK
    from smart_assignment.shared.llm import generate_text

    raw = generate_text(config.for_role(ROLE_SLOTPICK), prompt)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        # Log the raw reply (empty, prose, an error string) so a parse failure is
        # diagnosable; the caller still falls back deterministically.
        logger.warning("Slot-pick LLM reply was not JSON (len=%d): %r", len(raw), raw[:500])
        raise
