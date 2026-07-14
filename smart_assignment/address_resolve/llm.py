"""
Model plumbing for the grounded address-choice call: prompt -> raw dict, routed
through the shared backend under the `address_resolve` role model. Injectable so
tests drive the resolver with a fake and no network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config


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


def generate_address_choice(config: "Config", prompt: str) -> dict:
    """Call the configured backend (address_resolve role model) and parse JSON."""
    from smart_assignment.shared.config import ROLE_ADDRESS_RESOLVE
    from smart_assignment.shared.llm import generate_text

    return _extract_json(generate_text(config.for_role(ROLE_ADDRESS_RESOLVE), prompt))
