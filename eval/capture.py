"""
Phase 2b capture -- record the agent's real final responses into the eval dataset.

Runs the live ``root_agent`` (smart_assignment/agent.py) over each golden case, so
it NEEDS a configured LLM backend (e.g. the Sage credentials -- see .env.example),
and records the agent's concluding natural-language response per case. The text is
written to ``eval/data/captured_responses.json`` -- a committed, human-reviewable
``{eval_id: final_response}`` file -- and the dataset is regenerated so
``final_response`` is populated from it (see eval/build_evalset.py).

Why a separate committed file (rather than editing the dataset directly): the
builder stays deterministic and the hermetic sync test
(tests/eval/test_build_evalset.py) keeps holding, because both the committed
dataset and ``render_dataset()`` derive ``final_response`` from this same file.
The non-determinism of a real run is frozen once, at capture time.

This is the deferred Phase 2b step: structural trajectory + intake args stay in
build_evalset.py (no backend); only the natural-language responses -- which can
only come from a real run -- are produced here.

Usage (needs a live backend; run from the repo root):

    python3 -m eval.capture            # capture all cases, rewrite the dataset
    python3 -m eval.capture --check    # dry run: print captures, write nothing

Imports stay credential-free: the heavy ADK/agent imports are lazy inside the
functions, so importing this module never needs a backend and it is safe for the
hermetic suite to import.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
from typing import Dict, List, Optional

from eval.golden_cases import GOLDEN_CASES, GoldenCase

# A distinct app/user id so capture runs are easy to spot in a trace backend
# (e.g. Arize Phoenix) separately from web-app or ad-hoc runs.
_APP_NAME = "smart_assignment_eval_capture"
_USER_ID = "eval_capture"
_CAPTURED_PATH = pathlib.Path(__file__).parent / "data" / "captured_responses.json"


def _require_backend() -> None:
    """Fail fast with an actionable message when no LLM backend is configured, so a
    capture run can't silently record empty or errored responses.

    Mirrors the credential checks the web app uses (webapp/llm_chat.py) without
    importing anything heavy."""
    from smart_assignment.shared.config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    if config.llm_backend == "sage":
        missing = [
            var
            for var in ("SAGE_CLIENT_ID", "SAGE_CLIENT_SECRET", "SAGE_ENVIRONMENT")
            if not os.environ.get(var)
        ]
        if missing:
            raise SystemExit(
                "eval.capture needs the Sage backend credentials, but these are unset: "
                f"{', '.join(missing)}. Set them (see .env.example) and re-run."
            )
        return
    # "standard" backend: a litellm "<provider>/<model>" carries its own key; a
    # bare Gemini model name needs GOOGLE_API_KEY or Vertex.
    if "/" not in config.model and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    ):
        raise SystemExit(
            "eval.capture needs an LLM backend. For SMART_ASSIGNMENT_LLM_BACKEND=standard "
            "with a bare Gemini model, set GOOGLE_API_KEY (see .env.example)."
        )


async def _capture_case(case: GoldenCase) -> str:
    """Run the live agent once on the case's intake message and return the agent's
    final natural-language response.

    Single-turn by design (one user message per case, matching the dataset). For an
    escalation the agent hands off to a human via ADK's ``request_input`` long-running
    tool; that handoff message (the triage brief) IS the agent's final output for the
    turn, so it's what we capture. The run drives the same ADK ``Runner`` the web app
    uses, in non-streaming mode (parity with ``adk web``)."""
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from smart_assignment.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=_APP_NAME, session_service=session_service)
    await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=case.eval_id
    )
    new_message = types.Content(role="user", parts=[types.Part(text=case.query)])

    final_texts: List[str] = []
    escalation_prompt: Optional[str] = None
    async for event in runner.run_async(
        user_id=_USER_ID,
        session_id=case.eval_id,
        new_message=new_message,
        run_config=RunConfig(streaming_mode=StreamingMode.NONE),
    ):
        # Human-in-the-loop escalation: request_input surfaces as a long-running
        # call; its message is the agent's final handoff for this turn.
        if getattr(event, "long_running_tool_ids", None):
            for call in event.get_function_calls():
                if call.id in event.long_running_tool_ids:
                    escalation_prompt = (call.args or {}).get("message") or escalation_prompt
            continue
        # Tool calls / tool return values drive the pipeline; nothing to record.
        if event.get_function_calls() or event.get_function_responses():
            continue
        # Aggregated (non-partial) natural-language text only, so we record each
        # reply once rather than per streamed chunk.
        if event.content and event.content.parts and not getattr(event, "partial", False):
            text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
            if text.strip():
                final_texts.append(text.strip())

    if escalation_prompt:
        return escalation_prompt.strip()
    if not final_texts:
        raise RuntimeError(
            f"{case.eval_id}: the agent produced no final text response and no escalation. "
            "Check the backend/model is actually answering (try `python3 -m eval.capture --check`)."
        )
    # The concluding narration is the last aggregated text event.
    return final_texts[-1]


async def _capture_all(cases: List[GoldenCase]) -> Dict[str, str]:
    captured: Dict[str, str] = {}
    for case in cases:
        print(f"[capture] running {case.eval_id} ...", flush=True)
        captured[case.eval_id] = await _capture_case(case)
    return captured


def _serialize(captured: Dict[str, str]) -> str:
    """Sorted keys + trailing newline so the committed file is stable and diffs are
    readable."""
    ordered = {key: captured[key] for key in sorted(captured)}
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real agent final responses into the eval dataset (Phase 2b)."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry run: capture and print responses without writing any files.",
    )
    args = parser.parse_args()

    _require_backend()
    captured = asyncio.run(_capture_all(GOLDEN_CASES))

    if args.check:
        print(_serialize(captured))
        print(f"[capture] --check: captured {len(captured)} response(s); no files written.")
        return

    _CAPTURED_PATH.write_text(_serialize(captured), encoding="utf-8")
    # Regenerate the dataset so final_response is populated from the captured file.
    from eval.build_evalset import main as build_dataset

    build_dataset()
    print(f"[capture] wrote {len(captured)} response(s) to {_CAPTURED_PATH}")
    print("[capture] regenerated the dataset. Commit BOTH files:")
    print("           eval/data/captured_responses.json")
    print("           eval/data/slot_recommendation.test.json")


if __name__ == "__main__":
    main()
