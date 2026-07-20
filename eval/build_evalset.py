"""Deterministically build an ADK ``EvalSet`` JSON from the golden cases.

Two modes, by design:

* **structural** (``build_eval_set``) -- no LLM backend needed. Emits each case's
  user message + the expected *tool trajectory* (with ``intake_customer``'s real
  ground-truth args), and an empty final response. This is what Phase 2a commits
  to ``eval/data/slot_recommendation.test.json`` and what the trajectory-only
  ``test_config.json`` scores against. Because it is pure and deterministic, a
  test can assert the committed file stays in sync with this builder.

* **capture** (Phase 2b) -- ``eval/capture.py`` runs the real ``root_agent`` to
  record the actual final responses into ``data/captured_responses.json`` (a
  committed ``{eval_id: text}`` file). This builder reads that file and populates
  ``final_response`` from it, so a captured dataset is still produced purely from
  two committed inputs (``golden_cases.py`` + the captured file) -- byte-stable,
  so the sync test in ``tests/eval/test_build_evalset.py`` still holds. When the
  captured file is absent (as in a fresh Phase-2a checkout), ``final_response``
  stays ``None`` and the structural output is reproduced exactly.

Run ``python3 -m eval.build_evalset`` to (re)generate the committed dataset.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from eval.golden_cases import GOLDEN_CASES, GoldenCase, expected_trajectory

EVAL_SET_ID = "slot_recommendation_golden"
EVAL_SET_NAME = "Slot recommendation: agent trajectory over the mock fixtures"
EVAL_SET_DESCRIPTION = (
    "Golden dataset for the conversational slot-recommendation agent, built "
    "deterministically from smart_assignment.mock_customers by eval/build_evalset.py. "
    "Phase 2a scores TOOL TRAJECTORY only (see eval/data/test_config.json); the "
    "expected final responses are captured with a live backend in Phase 2b. "
    "Regenerate with:  python3 -m eval.build_evalset"
)

_DATASET_PATH = pathlib.Path(__file__).parent / "data" / "slot_recommendation.test.json"
# Committed {eval_id: final_response} source of truth, written by eval/capture.py
# (Phase 2b). Absent on a fresh Phase-2a checkout -> final_response stays None.
_CAPTURED_PATH = pathlib.Path(__file__).parent / "data" / "captured_responses.json"


def load_captured() -> Dict[str, str]:
    """The captured ``{eval_id: final_response}`` text map, or ``{}`` if not yet
    captured.

    Reading a committed file (not a live call) keeps the builder deterministic and
    backend-free -- the non-determinism of a real run is frozen into the committed
    file at capture time (see eval/capture.py). eval/capture.py's on-disk entries
    are ``{"final_response": str, "escalated": bool}`` (it also tracks whether a
    case escalated, for eval/test_response_match.py); this builder only needs the
    text, so it extracts just that -- keeping this function's return shape (and
    thus every downstream dataset field) unchanged regardless of that extra data.
    Tolerates the older plain-``{eval_id: text}`` format too, so a file captured
    before outcome-tracking was added still loads."""
    if not _CAPTURED_PATH.exists():
        return {}
    raw = json.loads(_CAPTURED_PATH.read_text(encoding="utf-8"))
    return {
        eval_id: (entry["final_response"] if isinstance(entry, dict) else entry)
        for eval_id, entry in raw.items()
    }


def _user_content(text: str) -> Dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def _final_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """A captured final response as an ADK ``Content`` dict (model role), or ``None``
    when nothing was captured for this case (structural/Phase-2a)."""
    if not text:
        return None
    return {"role": "model", "parts": [{"text": text}]}


def _tool_uses(case: GoldenCase) -> List[Dict[str, Any]]:
    """The expected tool calls as ADK ``FunctionCall`` dicts (name + args)."""
    return [
        {"name": name, "args": args}
        for name, args in expected_trajectory(case)
    ]


def _invocation(case: GoldenCase, captured: Dict[str, str]) -> Dict[str, Any]:
    """One conversation turn in ADK ``Invocation`` shape. ``final_response`` is the
    captured agent narration when present (Phase 2b), else ``None`` -- trajectory
    scoring ignores it either way."""
    return {
        "invocation_id": case.eval_id,
        "user_content": _user_content(case.query),
        "final_response": _final_response(captured.get(case.eval_id)),
        "intermediate_data": {"tool_uses": _tool_uses(case)},
    }


def build_eval_set(
    cases: List[GoldenCase] = GOLDEN_CASES,
    captured: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return a schema-valid ADK ``EvalSet`` dict for the given golden cases.

    Deterministic and backend-free: ``creation_timestamp`` is fixed at 0.0 so the
    output is byte-stable and a test can assert the committed file matches.
    ``captured`` defaults to the committed capture file (``{}`` if absent); pass an
    explicit map to build a populated set in a hermetic test without the file.
    """
    captured = load_captured() if captured is None else captured
    return {
        "eval_set_id": EVAL_SET_ID,
        "name": EVAL_SET_NAME,
        "description": EVAL_SET_DESCRIPTION,
        "eval_cases": [
            {
                "eval_id": case.eval_id,
                "conversation": [_invocation(case, captured)],
                "session_input": None,
                "creation_timestamp": 0.0,
            }
            for case in cases
        ],
        "creation_timestamp": 0.0,
    }


def render_dataset(
    cases: List[GoldenCase] = GOLDEN_CASES,
    captured: Optional[Dict[str, str]] = None,
) -> str:
    """The committed-file text: pretty JSON with a trailing newline."""
    return json.dumps(build_eval_set(cases, captured), indent=2) + "\n"


def main() -> None:
    _DATASET_PATH.write_text(render_dataset(), encoding="utf-8")
    print(f"Wrote {len(GOLDEN_CASES)} eval case(s) to {_DATASET_PATH}")


if __name__ == "__main__":
    main()
