"""Hermetic tests for eval/response_extract.py -- the one reader that decides
whether an agent turn ended in narration or a human handoff.

No LLM backend and no live run: the module duck-types ``Content``/``Part``, so
plain stubs exercise it exactly as ADK objects would. One test does import ADK
(a base dependency, not an extra) to pin the handoff tool's real name.
"""

from __future__ import annotations

from typing import Any, Optional

from eval.response_extract import (
    HANDOFF_MESSAGE_ARG,
    HANDOFF_TOOL_NAME,
    aggregated_text,
    extract_final_response,
    handoff_message,
    has_function_activity,
)


class _Call:
    def __init__(self, name: str, args: Optional[dict] = None):
        self.name = name
        self.args = args


class _Part:
    """A genai ``Part`` stub: exactly one of text / function_call /
    function_response, as the real type produces."""

    def __init__(self, text=None, function_call=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _Content:
    def __init__(self, *parts: Any):
        self.parts = list(parts)


def _handoff(message) -> _Content:
    return _Content(_Part(function_call=_Call(HANDOFF_TOOL_NAME, {HANDOFF_MESSAGE_ARG: message})))


def _tool_call(name: str = "recommend_or_escalate") -> _Content:
    return _Content(_Part(function_call=_Call(name, {})))


def _tool_result() -> _Content:
    return _Content(_Part(function_response={"result": "ok"}))


# --------------------------------------------------------------------------
# The name this whole module hinges on
# --------------------------------------------------------------------------


def test_handoff_tool_name_matches_adks_own_tool():
    """The constant is hardcoded to keep this module ADK-free; this is the pin
    that stops it drifting.

    ADK names the tool ``adk_request_input``, NOT ``request_input`` -- the latter
    is only the Python symbol you import (google/adk/tools/_request_input_tool.py
    renames the function to REQUEST_INPUT_FUNCTION_CALL_NAME before wrapping it).
    Matching the wrong string would classify every escalation as a recommend,
    silently, and send the highest-stakes prose to the wrong judge rubric.

    (The other half of the switch from ``Event.long_running_tool_ids`` to
    name-matching -- that ``adk_request_input`` is the ONLY long-running tool the
    agent registers, so the two are the same set -- was verified directly against
    ``root_agent`` and is recorded in this commit. It is not asserted here:
    ``root_agent`` is built lazily and raises without Sage credentials by design
    (see tests/test_offline_import.py), so pinning it would mean a subprocess,
    which is more machinery than that secondary invariant is worth.)
    """
    from google.adk.tools import request_input

    assert request_input.name == HANDOFF_TOOL_NAME


# --------------------------------------------------------------------------
# Reading a single content
# --------------------------------------------------------------------------


def test_handoff_message_is_read_from_the_tool_call_args():
    assert handoff_message(_handoff("Escalating: no route clears the bar.")) == (
        "Escalating: no route clears the bar."
    )


def test_a_different_tool_call_is_not_a_handoff():
    assert handoff_message(_tool_call()) is None


def test_a_blank_handoff_message_is_not_a_handoff():
    # An empty brief is nothing a specialist can act on; treating it as an
    # escalation would record an empty final_response as the turn's answer.
    for blank in (None, "", "   "):
        assert handoff_message(_handoff(blank)) is None


def test_text_parts_are_joined_and_stripped():
    # A model may split one reply across several text parts.
    content = _Content(_Part(text="  Route RTE-4100 "), _Part(text="on Tuesday.  "))
    assert aggregated_text(content) == "Route RTE-4100 on Tuesday."


def test_content_without_parts_reads_as_empty():
    assert aggregated_text(_Content()) == ""
    assert handoff_message(_Content()) is None
    assert has_function_activity(_Content()) is False


def test_function_activity_covers_calls_and_results():
    assert has_function_activity(_tool_call()) is True
    assert has_function_activity(_tool_result()) is True
    assert has_function_activity(_Content(_Part(text="hello"))) is False


# --------------------------------------------------------------------------
# Reading a whole turn
# --------------------------------------------------------------------------


def test_recommend_turn_returns_the_last_narration():
    result = extract_final_response(
        [
            _Content(_Part(text="Let me look at the routes.")),
            _tool_call(),
            _tool_result(),
            _Content(_Part(text="I recommend RTE-4100 on Tuesday 07:00-10:00.")),
        ]
    )
    assert result == ("I recommend RTE-4100 on Tuesday 07:00-10:00.", False)


def test_escalate_turn_returns_the_handoff_brief():
    result = extract_final_response([_tool_call(), _tool_result(), _handoff("SITUATION ...")])
    assert result.final_response == "SITUATION ..."
    assert result.escalated is True


def test_a_handoff_wins_over_narration_in_the_same_turn():
    # Once the agent has handed off, the brief IS the turn's output -- whatever
    # it said around the handoff is not what a specialist acts on.
    result = extract_final_response(
        [
            _Content(_Part(text="This one needs a human.")),
            _handoff("SITUATION ..."),
        ]
    )
    assert result == ("SITUATION ...", True)


def test_a_handoff_before_later_narration_still_wins():
    # Order-independent on purpose: ADK may emit trailing text after the
    # long-running call, and that must not demote an escalation to a recommend.
    result = extract_final_response([_handoff("SITUATION ..."), _Content(_Part(text="Done."))])
    assert result == ("SITUATION ...", True)


def test_a_tool_call_never_contributes_text():
    # Preserves capture.py's long-standing behavior: an event carrying a function
    # call is pipeline machinery, not the answer, even if it also carries text.
    mixed = _Content(_Part(text="calling a tool"), _Part(function_call=_Call("some_tool", {})))
    assert extract_final_response([mixed, _Content(_Part(text="The answer."))]) == (
        "The answer.",
        False,
    )
    assert extract_final_response([mixed]) is None


def test_none_contents_are_skipped():
    # A live ADK event can carry content=None.
    assert extract_final_response([None, _Content(_Part(text="ok")), None]) == ("ok", False)


def test_a_turn_with_neither_returns_none_for_the_caller_to_explain():
    # The caller raises, because only it knows which eval case was running.
    assert extract_final_response([]) is None
    assert extract_final_response([_tool_call(), _tool_result()]) is None
    assert extract_final_response([_Content(_Part(text="   "))]) is None
