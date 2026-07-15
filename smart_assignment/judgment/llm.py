"""
Model plumbing for the grounded-judgment call: turn a prompt string into a
raw judgment dict, routed through the same backend toggle everything else uses
(`shared.llm.generate_text`).

`generate_judgment` is the default `judgment_fn` the `GroundedJudge` calls. It
is deliberately thin and swappable — tests inject a fake `judgment_fn` so none
of the judgment logic needs a network or credentials to exercise.

The model is asked (in prompts.py) for JSON only, but real models still
occasionally wrap it in ```json fences or add stray prose; `_extract_json`
tolerates that. Any failure raises, and the caller (`judge.py`) treats a raised
`judgment_fn` as a mechanical failure -> deterministic fallback.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Best-effort: pull the first JSON object out of a model reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        # strip a leading ```json / ``` fence and the trailing ```
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def generate_judgment(config: "Config", prompt: str) -> dict:
    """Call the configured LLM backend and return the parsed JSON dict.

    Uses the judgment role's model (config.for_role), so the decision call can
    run on a different model than the conversational agent. Raises on any
    backend error or unparseable reply; the caller converts that into a
    deterministic fallback rather than surfacing it.
    """
    from smart_assignment.shared.config import ROLE_JUDGMENT
    from smart_assignment.shared.llm import generate_text

    raw = generate_text(config.for_role(ROLE_JUDGMENT), prompt)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        # Log the raw reply (empty, prose, an error string) so a parse failure is
        # diagnosable; the caller still falls back deterministically.
        logger.warning("Judgment LLM reply was not JSON (len=%d): %r", len(raw), raw[:500])
        raise
