"""
Error-recovery callbacks for the agents in ``agent.py``.

ADK re-raises whatever a model or a tool throws (``base_llm_flow`` re-raises the
model error; ``functions`` re-raises the tool error), and that exception unwinds
the whole Runner. One malformed reply therefore destroys an entire turn -- even
when the deterministic pipeline underneath it already **succeeded** and the user
has been shown its breadcrumbs. The web app then discards the agent's real work
and serves a deterministic result that can contradict what it just said.

ADK offers exactly two hooks to stop that, and they need different answers:

``on_model_error_callback``  (the agent's OWN model call failed)
    Returning an ``LlmResponse`` makes ADK yield it instead of raising. It has no
    function calls, so ``Event.is_final_response()`` is True and the flow's
    ``while True`` loop terminates -- there is no retry semantics here and no
    spin risk. The response carries BOTH ``content`` (so the surface actually
    shows something -- the web app only emits a frame for an event with
    ``content.parts``, so an error-code-only response would render a blank turn)
    and ``error_code``/``error_message`` (so logs, telemetry, and the callers
    below can tell an error notice apart from real agent prose).

``on_tool_error_callback``  (a tool raised, including the escalation_triage AgentTool)
    Returning a dict makes ADK use it as the tool result. We return the same
    ``{"ok": False, "error": ...}`` shape every pipeline tool already uses, so
    ``webapp.llm_chat._tool_outcome`` marks that step failed with the real reason
    and the model sees an ordinary failed-tool result it can still act on (e.g.
    escalate with a bare reason when triage broke). The turn continues.

Why the triage sub-agent deliberately gets NEITHER: ``AgentTool`` returns the
sub-agent's last content as the tool result, and the root instruction relays the
brief **verbatim** to a specialist. A model-error callback there would hand an
apology to that specialist as if it were the escalation brief. Letting it raise
into the root agent's tool-error callback is strictly safer.

Everything here is gated by ``Config.recover_from_agent_errors`` (default on) and
is deliberately dependency-light, so it unit-tests offline with no credentials.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Stamped on the LlmResponse this module synthesizes, and therefore on the Event
# ADK builds from it (Event subclasses LlmResponse and the finalizer merges the
# response in). It is the discriminator callers use to tell "the agent said this"
# from "the agent broke" -- see webapp/llm_chat.py and batch/agent_runner.py,
# which both refuse to treat an error notice as the agent's own reasoning.
AGENT_MODEL_ERROR = "AGENT_MODEL_ERROR"

# What the user is told when a turn could not be completed. Deliberately short and
# non-technical: the real cause goes to the log, not to a prospect-facing chat.
# It says what is still true, because the pipeline result usually survived.
MODEL_ERROR_MESSAGE = (
    "Sorry -- something went wrong while I was finishing that step, so I had to "
    "stop partway. Anything already shown above still stands. Please send that "
    "last message again."
)

# Longest error string copied into a tool result or a log line. The model sees the
# tool result, so an unbounded backend error would otherwise land in the prompt.
_MAX_ERROR_CHARS = 300


def _describe(error: BaseException) -> str:
    """A short, single-line ``Type: message`` description of an exception."""
    text = f"{type(error).__name__}: {error}".replace("\n", " ").strip()
    return text[:_MAX_ERROR_CHARS]


def model_error_response(
    *, callback_context: Any = None, llm_request: Any = None, error: BaseException
) -> Optional[Any]:
    """ADK ``on_model_error_callback``: end the turn with a plain reply instead of
    unwinding the Runner.

    Signature matches how ADK invokes it -- by keyword
    (``callback_context=``, ``llm_request=``, ``error=``); the first two are
    unused but must be accepted. Returns ``None`` if the response cannot be built,
    which restores ADK's raw behavior (it re-raises) rather than swallowing the
    failure silently.
    """
    logger.exception(
        "Agent model call failed (%s); ending the turn with a recoverable reply "
        "instead of dropping it.",
        _describe(error),
        exc_info=error,
    )
    try:
        from google.adk.models import LlmResponse
        from google.genai import types

        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=MODEL_ERROR_MESSAGE)]
            ),
            error_code=AGENT_MODEL_ERROR,
            error_message=_describe(error),
        )
    except Exception:  # noqa: BLE001 - a recovery hook must never add a new failure
        logger.warning(
            "Could not build the model-error response; letting ADK raise as before.",
            exc_info=True,
        )
        return None


def tool_error_response(
    *, tool: Any = None, args: Any = None, tool_context: Any = None, error: BaseException
) -> Optional[dict]:
    """ADK ``on_tool_error_callback``: turn a raised tool into the ordinary
    ``{"ok": False, "error": ...}`` result every pipeline tool already returns, so
    the step is reported failed and the conversation carries on.

    Signature matches how ADK invokes it -- by keyword (``tool=``, ``args=``,
    ``tool_context=``, ``error=``).
    """
    name = getattr(tool, "name", None) or "tool"
    logger.exception(
        "Tool %r raised (%s); reporting it as a failed tool result so the turn "
        "continues.",
        name,
        _describe(error),
        exc_info=error,
    )
    return {"ok": False, "error": _describe(error)}


def error_callbacks(config: Any) -> dict:
    """The ``LlmAgent`` keyword arguments that install both hooks, or an empty dict
    when ``Config.recover_from_agent_errors`` is off -- in which case the agent is
    constructed exactly as before and ADK's raw raising behavior is restored."""
    if not getattr(config, "recover_from_agent_errors", True):
        return {}
    return {
        "on_model_error_callback": model_error_response,
        "on_tool_error_callback": tool_error_response,
    }
