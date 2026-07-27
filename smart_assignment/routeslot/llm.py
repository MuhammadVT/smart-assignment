"""
Model plumbing for the grounded route-slot call: prompt -> raw choice dict, routed
through the shared backend under the `judgment` role model (the route-slot decision
is the judgment decision).

The reliable channel is a FUNCTION CALL: we offer the model one tool
(`ROUTE_SLOT_DECISION_TOOL`) whose arguments ARE the decision, and use those args
directly. The conversational SAGE agent narrates when asked for a JSON string but
readily emits function calls, so this is what makes the grounded layer actually
fire on that backend. If the model narrates instead of calling the tool, we
salvage/parse JSON out of the prose (strict -> brace-slice -> ``json_repair``) as a
fallback. Injectable so tests drive it with a fake and no network.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Parse a model's TEXTUAL reply into a dict (the narration fallback, when the
    model didn't call the tool): strict JSON, then a brace-slice, then a last-resort
    ``json_repair`` that can lift a JSON object out of surrounding prose. Raises
    ``json.JSONDecodeError`` when nothing parseable is found."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Last resort: json_repair (vendored with the Sage SDK) can recover a JSON
        # object embedded in prose or with minor syntax slips. It returns "" / {}
        # for pure prose with no object -> treat that as unparseable.
        repaired = _repair_json(text)
        if isinstance(repaired, dict) and repaired:
            return repaired
        raise exc


def _repair_json(text: str) -> object:
    """Best-effort JSON recovery via ``json_repair``; returns ``None`` if the
    library isn't importable so the caller falls back cleanly."""
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:  # pragma: no cover - json_repair ships with the SDK
        return None
    return repair_json(text, return_objects=True)


def generate_route_slot_choice(config: "Config", prompt: str) -> dict:
    """Ask the judgment-role model for a route-slot decision via a tool call, and
    return the raw choice dict.

    Prefers the model's structured function-call arguments; on narration, salvages
    JSON from the prose. Raises ``json.JSONDecodeError`` when neither yields a
    parseable object, so the caller falls back deterministically."""
    from smart_assignment.routeslot.prompts import ROUTE_SLOT_DECISION_TOOL
    from smart_assignment.shared.config import ROLE_JUDGMENT
    from smart_assignment.shared.llm import generate_tool_call

    call_args, raw = generate_tool_call(
        config.for_role(ROLE_JUDGMENT), prompt, ROUTE_SLOT_DECISION_TOOL, role=ROLE_JUDGMENT
    )
    if call_args is not None:
        return call_args

    # The model narrated instead of calling the tool -> parse/repair the prose.
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        # Surface WHAT the backend returned (empty, prose, an error string) so a
        # parse failure is diagnosable instead of just "JSONDecodeError". Truncated
        # to keep the log readable; the caller still falls back deterministically.
        logger.warning("Route-slot LLM reply was not JSON (len=%d): %r", len(raw), raw[:500])
        raise
