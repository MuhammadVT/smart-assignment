"""Read one agent turn's final answer out of its content stream -- once, for
every caller that needs it.

An agent turn ends one of two ways, and telling them apart is the whole job:

* **Recommend** -- the turn ends on ordinary narration, and the agent's answer is
  the last aggregated text it produced.
* **Escalate** -- the turn ends by handing off to a human through ADK's
  long-running ``adk_request_input`` tool. The handoff brief rides in that tool
  call's ``message`` argument, NOT in any text part, so a reader that only looks
  at ``.text`` sees an empty final response for every escalation. (That is
  exactly ADK's own blind spot in ``response_match_score`` -- see
  ``eval/test_response_match.py``'s module docstring.)

``eval/capture.py`` already had this logic inline, and
``smart_assignment/batch/agent_runner.py`` has its own copy. A third was about
to appear in the eval-run harvester, which is what forced the issue: the
harvester reads ADK ``Invocation`` records rather than a live ``Event`` stream,
and an ``InvocationEvent`` carries only ``author`` and ``content`` -- no
``long_running_tool_ids``. So a copy there would have had to detect the handoff
a *different* way than capture does, and two mechanisms for one question drift.
When they drift, an escalation is silently recorded as a recommendation and gets
scored by the wrong rubric.

So this module works on ``Content`` objects, the one shape both paths can
produce, and detects the handoff by TOOL NAME -- which is equally available to
both. That is a real (if invisible) behavior change for capture, which used to
match on ``Event.long_running_tool_ids``: in this app the two are the same set,
because ``adk_request_input`` is the only long-running tool registered (see
``smart_assignment/agent.py``), and ``tests/eval/test_response_extract.py`` pins
the name against ADK's own tool object so a rename upstream fails a test instead
of quietly mislabeling every escalation.

Deliberately import-light: it duck-types ``Content``/``Part`` (only ``.parts``,
``.text``, ``.function_call``, ``.name``, ``.args`` are touched) and imports
nothing from ``google.adk``, so the hermetic suite can exercise it with plain
stubs and importing it never costs an ADK import.
"""

from __future__ import annotations

from typing import Any, Iterable, List, NamedTuple, Optional

# ADK names the long-running human-handoff tool "adk_request_input" -- NOT
# "request_input", which is only the Python symbol you import. [VERIFIED against
# installed google-adk 2.3.0: google/adk/tools/_request_input_tool.py renames the
# function to REQUEST_INPUT_FUNCTION_CALL_NAME before wrapping it.] Hardcoded
# rather than imported so this module stays ADK-free; pinned to ADK's own tool
# object by tests/eval/test_response_extract.py so the two cannot drift apart.
HANDOFF_TOOL_NAME = "adk_request_input"

# The argument carrying the brief a human specialist reads.
HANDOFF_MESSAGE_ARG = "message"


class ExtractedResponse(NamedTuple):
    """A turn's answer plus how it ended. ``escalated`` is what decides which
    scorer may look at the text at all (see ``eval/test_response_match.py``) and
    which rubric judges it (see ``eval/test_quality.py``)."""

    final_response: str
    escalated: bool


def _parts(content: Any) -> List[Any]:
    return list(getattr(content, "parts", None) or [])


def handoff_message(content: Any) -> Optional[str]:
    """The escalation brief in this content's ``adk_request_input`` call, or
    ``None`` if it holds no such call. A call present but carrying a blank
    message reads as ``None``: an empty brief is not a handoff worth recording."""
    for part in _parts(content):
        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None) == HANDOFF_TOOL_NAME:
            message = (getattr(call, "args", None) or {}).get(HANDOFF_MESSAGE_ARG)
            if message and str(message).strip():
                return str(message).strip()
    return None


def has_function_activity(content: Any) -> bool:
    """True when this content is a tool call or a tool result rather than
    narration. Such a content is skipped for TEXT even if it also carries a text
    part -- preserving the behavior ``eval/capture.py`` has always had, where a
    tool-call event never contributes to the final answer."""
    for part in _parts(content):
        if getattr(part, "function_call", None) is not None:
            return True
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def aggregated_text(content: Any) -> str:
    """All text parts of this content joined and stripped ("" when there are
    none). Joined rather than taking the first part because a model may split
    one reply across several text parts."""
    text = "".join(part.text for part in _parts(content) if getattr(part, "text", None))
    return text.strip()


def extract_final_response(contents: Iterable[Any]) -> Optional[ExtractedResponse]:
    """The turn's answer, read from its contents in order.

    An escalation wins over any narration in the same turn: once the agent has
    handed off, the brief IS its output for that turn, and whatever it said
    around the handoff is not the thing a human acts on. Otherwise the answer is
    the LAST aggregated narration, since earlier text is intermediate.

    Returns ``None`` when the turn produced neither -- the caller raises, because
    only it knows which case it was running.
    """
    escalation: Optional[str] = None
    texts: List[str] = []

    for content in contents:
        if content is None:
            continue
        message = handoff_message(content)
        if message:
            escalation = message
            continue
        if has_function_activity(content):
            continue
        text = aggregated_text(content)
        if text:
            texts.append(text)

    if escalation:
        return ExtractedResponse(escalation, escalated=True)
    if texts:
        return ExtractedResponse(texts[-1], escalated=False)
    return None
