"""Deterministically build an ADK ``EvalSet`` JSON from the golden cases.

Two modes, by design:

* **structural** (``build_eval_set``) -- no LLM backend needed. Emits each case's
  user message + the expected *tool trajectory* (with ``intake_customer``'s real
  ground-truth args), and an empty final response. This is what Phase 2a commits
  to ``eval/data/slot_recommendation.test.json`` and what the trajectory-only
  ``test_config.json`` scores against. Because it is pure and deterministic, a
  test can assert the committed file stays in sync with this builder.

* **capture** (Phase 2b) -- a follow-up helper that runs the real ``root_agent``
  to record the actual final responses too, so the dataset can additionally score
  final-response quality. Deferred because it needs a live backend to build and
  verify; it will produce the identical structure, only with ``final_response``
  populated. Not shipped here so we don't commit agent-driving code we can't yet
  exercise.

Run ``python3 -m eval.build_evalset`` to (re)generate the committed dataset.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

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


def _user_content(text: str) -> Dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def _tool_uses(case: GoldenCase) -> List[Dict[str, Any]]:
    """The expected tool calls as ADK ``FunctionCall`` dicts (name + args)."""
    return [
        {"name": name, "args": args}
        for name, args in expected_trajectory(case)
    ]


def _invocation(case: GoldenCase) -> Dict[str, Any]:
    """One conversation turn in ADK ``Invocation`` shape. ``final_response`` is left
    empty in structural mode -- trajectory-only scoring ignores it, and Phase 2b
    fills it from a real run."""
    return {
        "invocation_id": case.eval_id,
        "user_content": _user_content(case.query),
        "final_response": None,
        "intermediate_data": {"tool_uses": _tool_uses(case)},
    }


def build_eval_set(cases: List[GoldenCase] = GOLDEN_CASES) -> Dict[str, Any]:
    """Return a schema-valid ADK ``EvalSet`` dict for the given golden cases.

    Deterministic and backend-free: ``creation_timestamp`` is fixed at 0.0 so the
    output is byte-stable and a test can assert the committed file matches.
    """
    return {
        "eval_set_id": EVAL_SET_ID,
        "name": EVAL_SET_NAME,
        "description": EVAL_SET_DESCRIPTION,
        "eval_cases": [
            {
                "eval_id": case.eval_id,
                "conversation": [_invocation(case)],
                "session_input": None,
                "creation_timestamp": 0.0,
            }
            for case in cases
        ],
        "creation_timestamp": 0.0,
    }


def render_dataset(cases: List[GoldenCase] = GOLDEN_CASES) -> str:
    """The committed-file text: pretty JSON with a trailing newline."""
    return json.dumps(build_eval_set(cases), indent=2) + "\n"


def main() -> None:
    _DATASET_PATH.write_text(render_dataset(), encoding="utf-8")
    print(f"Wrote {len(GOLDEN_CASES)} eval case(s) to {_DATASET_PATH}")


if __name__ == "__main__":
    main()
