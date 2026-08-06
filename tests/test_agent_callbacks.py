"""
Tests for the agent error-recovery callbacks (smart_assignment/agent_callbacks.py).

These stay offline: the callbacks are plain functions, and the wiring assertions
inspect the LlmAgent kwargs rather than building an agent (which would resolve the
LLM backend). The point of every test here is that a failure ends up *handled and
labelled* rather than unwinding the Runner -- and that the label is what keeps an
error notice out of the agent's reasoning.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.agent_callbacks import (
    AGENT_MODEL_ERROR,
    MODEL_ERROR_MESSAGE,
    error_callbacks,
    model_error_response,
    tool_error_response,
)
from smart_assignment.shared.config import Config


# --- the model-error hook ---------------------------------------------------


def test_model_error_returns_a_final_text_response():
    """Content is required, not optional: the web app only emits a chat frame for
    an event with content.parts, so an error-code-only response would render a
    blank turn -- worse than the failure it replaces."""
    response = model_error_response(error=RuntimeError("backend exploded"))

    assert response is not None
    text = "".join(p.text or "" for p in response.content.parts)
    assert text == MODEL_ERROR_MESSAGE
    assert response.content.role == "model"


def test_model_error_response_is_labelled_as_an_error():
    response = model_error_response(error=RuntimeError("backend exploded"))
    assert response.error_code == AGENT_MODEL_ERROR
    assert "RuntimeError" in response.error_message
    assert "backend exploded" in response.error_message


def test_model_error_response_terminates_the_flow():
    """ADK loops until an event is_final_response(), which requires no function
    calls. A response that carried one would spin the flow forever."""
    response = model_error_response(error=RuntimeError("boom"))
    assert not response.get_function_calls()


def test_model_error_accepts_adks_keyword_invocation():
    # ADK calls it as callback(callback_context=..., llm_request=..., error=...).
    response = model_error_response(
        callback_context=object(), llm_request=object(), error=ValueError("x")
    )
    assert response is not None


def test_model_error_message_does_not_leak_backend_detail_to_the_user():
    response = model_error_response(error=RuntimeError("api key sk-secret rejected"))
    text = "".join(p.text or "" for p in response.content.parts)
    assert "sk-secret" not in text  # the detail belongs in the log, not the chat


# --- the tool-error hook ----------------------------------------------------


class _Tool:
    name = "escalation_triage"


def test_tool_error_returns_the_pipeline_tool_result_shape():
    """Deliberately the same {"ok": false, "error": ...} contract every pipeline
    tool already returns, so the existing breadcrumb machinery reads it with no
    new frame type."""
    result = tool_error_response(
        tool=_Tool(), args={}, tool_context=None, error=RuntimeError("nope")
    )
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]


def test_tool_error_result_is_read_as_a_failure_by_the_webapp():
    from smart_assignment.webapp.llm_chat import _tool_outcome

    result = tool_error_response(
        tool=_Tool(), args={}, tool_context=None, error=RuntimeError("nope")
    )
    ok, error = _tool_outcome(result)
    assert ok is False
    assert "RuntimeError" in error


def test_tool_error_message_is_bounded():
    # The model sees the tool result, so an unbounded backend error would land
    # verbatim in the next prompt.
    result = tool_error_response(
        tool=_Tool(), args={}, tool_context=None, error=RuntimeError("x" * 5000)
    )
    assert len(result["error"]) <= 300


def test_tool_error_survives_a_tool_without_a_name():
    result = tool_error_response(tool=object(), args=None, tool_context=None, error=OSError("io"))
    assert result["ok"] is False


# --- the flag and the wiring ------------------------------------------------


def test_recovery_is_on_by_default():
    assert Config().recover_from_agent_errors is True


def test_flag_reads_from_env():
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_RECOVER_FROM_AGENT_ERRORS": "false"}):
        assert Config.from_env().recover_from_agent_errors is False
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_RECOVER_FROM_AGENT_ERRORS": "true"}):
        assert Config.from_env().recover_from_agent_errors is True


def test_callbacks_are_installed_when_enabled():
    kwargs = error_callbacks(Config(recover_from_agent_errors=True))
    assert kwargs["on_model_error_callback"] is model_error_response
    assert kwargs["on_tool_error_callback"] is tool_error_response


def test_flag_off_reproduces_the_previous_agent_exactly():
    # No kwargs at all -> LlmAgent is constructed byte-for-byte as before and ADK
    # keeps raising.
    assert error_callbacks(Config(recover_from_agent_errors=False)) == {}


def test_both_agents_wire_the_callbacks():
    """The root and batch agents must both install them -- an unattended batch run
    is exactly where an unhandled exception is least likely to be noticed."""
    import inspect

    from smart_assignment import agent as agent_module

    source = inspect.getsource(agent_module)
    assert source.count("error_callbacks(") >= 2


# --- end-to-end through a real ADK Runner -----------------------------------
#
# The tests above pin the shapes. This one pins the BEHAVIOR: drive a real
# LlmAgent + Runner whose model raises the *actual* pydantic ValidationError that
# killed the Katy Mills turn, and assert the turn survives with the callbacks on
# and still dies with them off. Everything here is ADK's real machinery -- only
# the model is a fake, and it raises a genuine error rather than a stand-in.


def _real_validation_error() -> Exception:
    """The exact exception ADK raises when sage array-wraps a tool call's args."""
    from google.genai import types

    try:
        types.Part.from_function_call(
            name="escalation_triage", args=[{"request": '{"ok": true}'}]
        )
    except Exception as exc:  # noqa: BLE001 - capturing it IS the point
        return exc
    raise AssertionError("expected FunctionCall(args=<list>) to be rejected")


def _agent_whose_model_raises(config: Config):
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm

    boom = _real_validation_error()

    class RaisingLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            raise boom
            yield  # pragma: no cover - makes this an async generator

    return LlmAgent(
        name="raising_agent",
        model=RaisingLlm(model="fake-model"),
        description="an agent whose model always fails",
        instruction="unused",
        **error_callbacks(config),
    )


async def _drive(agent) -> list:
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    service = InMemorySessionService()
    await service.create_session(app_name="t", user_id="u", session_id="s")
    runner = Runner(agent=agent, app_name="t", session_service=service)
    return [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
            run_config=RunConfig(streaming_mode=StreamingMode.NONE),
        )
    ]


def test_validation_error_no_longer_takes_the_turn_down():
    import asyncio

    agent = _agent_whose_model_raises(Config(recover_from_agent_errors=True))
    events = asyncio.run(_drive(agent))

    assert events, "the turn produced no events at all"
    final = events[-1]
    assert final.error_code == AGENT_MODEL_ERROR
    text = "".join(p.text or "" for p in final.content.parts)
    assert text == MODEL_ERROR_MESSAGE
    # The flow must actually stop, not spin looking for another model call.
    assert final.is_final_response()


def test_with_recovery_off_the_turn_still_dies_as_before():
    """The control: flag off reproduces ADK's raw behavior exactly, so this change
    cannot be masking a failure anyone was relying on seeing."""
    import asyncio

    import pytest

    agent = _agent_whose_model_raises(Config(recover_from_agent_errors=False))
    with pytest.raises(Exception) as caught:
        asyncio.run(_drive(agent))
    assert "valid dictionary" in str(caught.value)


def test_recovered_turn_is_not_treated_as_agent_reasoning():
    """The webapp guard: the recovery notice reaches the user as a chat message but
    must never be captured as the result card's 'Why the agent chose this'."""
    import asyncio

    agent = _agent_whose_model_raises(Config(recover_from_agent_errors=True))
    events = asyncio.run(_drive(agent))
    final = events[-1]

    # This mirrors the condition in webapp/llm_chat.stream_turn.
    saw_recommendation = True
    captured_as_reasoning = saw_recommendation and not getattr(final, "error_code", None)
    assert captured_as_reasoning is False


def test_recovery_covers_the_shapes_the_arg_repair_deliberately_leaves_alone():
    """Quantifies what this layer adds on top of Config.repair_tool_call_args.

    That repair is deliberately narrow -- it only unwraps an array holding exactly
    ONE object, because picking between two would be a guess. Every other array
    shape is passed through and still reaches pydantic. Those are precisely the
    cases that used to kill a turn with no remedy at all, and that the model-error
    callback now ends gracefully."""
    from smart_assignment.shared.llm import _coerce_tool_call_args

    unrepairable = [
        [{"request": "a"}, {"request": "b"}],  # two objects: ambiguous
        [["a", "b"]],  # no object at all
        ["just", "strings"],
    ]
    for payload in unrepairable:
        repaired, _ = _coerce_tool_call_args(payload)
        assert repaired is None, f"expected {payload!r} to be left for ADK to reject"

    # One object plus debris IS repaired -- the captured sage shape.
    repaired, discarded = _coerce_tool_call_args([{"request": "a"}, ["debris"]])
    assert repaired == {"request": "a"} and discarded == [["debris"]]


def test_triage_agent_deliberately_has_no_model_error_callback():
    """AgentTool returns the sub-agent's last content as the tool result, and the
    root instruction relays the brief VERBATIM to a specialist. A model-error
    callback on the triage agent would hand that specialist an apology dressed as
    an escalation brief; it must raise into the root's tool-error hook instead."""
    import inspect

    from smart_assignment.triage import agent as triage_agent

    source = inspect.getsource(triage_agent)
    assert "on_model_error_callback" not in source
