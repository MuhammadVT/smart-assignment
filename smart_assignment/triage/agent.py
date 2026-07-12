"""
The escalation-triage sub-agent and its ``AgentTool`` wrapper.

``build_triage_agent`` builds an ``LlmAgent`` with two read-only tools
(``get_escalation_context`` to load the trace, ``check_brief_grounding`` to
verify its own draft) and a deterministic ``after_model_callback`` backstop that
annotates any final brief still containing ungrounded figures. ``build_triage_tool``
wraps the agent as an ``AgentTool`` so ``root_agent`` calls it like any tool.

Both builders call ``get_llm`` (which resolves the configured backend and may
need credentials), so they must be invoked lazily -- ``smart_assignment/agent.py``
does so only while building ``root_agent``, never at import time.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import AgentTool, FunctionTool

from smart_assignment.shared.config import ROLE_TRIAGE, Config
from smart_assignment.shared.llm import get_llm
from smart_assignment.triage.context import (
    _STATE_TRIAGE_GROUNDING_KEY,
    check_brief_grounding,
    get_escalation_context,
)
from smart_assignment.triage.prompts import TRIAGE_INSTRUCTION
from smart_assignment.triage.verifier import verify_brief

logger = logging.getLogger(__name__)

# The sub-agent's name doubles as the tool name root_agent calls it by; the
# root instruction (prompts.ESCALATION_TRIAGE_GUIDANCE) references this exact
# string, so keep them in sync.
TRIAGE_AGENT_NAME = "escalation_triage"


def _content_text(content) -> str:
    """Concatenate the text parts of an ADK Content, tolerating None/empties."""
    if content is None or not getattr(content, "parts", None):
        return ""
    return "".join(getattr(p, "text", "") or "" for p in content.parts)


def _grounding_backstop(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Deterministic after-model backstop: if the triage agent's *final* brief
    (a text response with no tool calls) still cites figures or routes that
    aren't grounded in the escalation context, append a caveat naming them.

    This always runs, so ungrounded figures are flagged even if the agent
    skipped the check_brief_grounding self-check. Defensive by construction --
    any failure leaves the brief unchanged rather than breaking the agent.
    """
    try:
        if llm_response is None or llm_response.get_function_calls():
            return None  # not a final text turn
        text = _content_text(getattr(llm_response, "content", None))
        if not text.strip():
            return None
        grounding = None
        try:
            grounding = callback_context.state.get(_STATE_TRIAGE_GROUNDING_KEY)
        except Exception:  # noqa: BLE001 - state access is best-effort here
            grounding = None
        if not grounding:
            return None
        result = verify_brief(text, grounding)
        if result.ok:
            return None
        logger.warning(
            "Triage brief contained ungrounded figures %s / routes %s; appending a caveat.",
            result.ungrounded_numbers,
            result.ungrounded_routes,
        )
        from google.genai import types

        new_content = types.Content(
            role="model",
            parts=[types.Part(text=f"{text}\n\n{result.caveat()}")],
        )
        return llm_response.model_copy(update={"content": new_content})
    except Exception:  # noqa: BLE001 - never let the backstop break the agent
        logger.warning(
            "Triage grounding backstop failed; leaving the brief unchanged.", exc_info=True
        )
        return None


def build_triage_agent(config: Config) -> LlmAgent:
    """Construct the escalation-triage ``LlmAgent`` (resolves the LLM backend)."""
    return LlmAgent(
        name=TRIAGE_AGENT_NAME,
        model=get_llm(config.for_role(ROLE_TRIAGE)),
        description=(
            "Turns an escalated slot recommendation into a concise specialist "
            "brief -- root cause, concrete remediation options, and the question "
            "to ask -- grounded in the evaluation trace. Never changes the decision."
        ),
        instruction=TRIAGE_INSTRUCTION,
        tools=[
            FunctionTool(get_escalation_context),
            FunctionTool(check_brief_grounding),
        ],
        after_model_callback=_grounding_backstop,
    )


def build_triage_tool(config: Config) -> AgentTool:
    """The triage agent wrapped as an ``AgentTool`` for root_agent's tools list."""
    return AgentTool(agent=build_triage_agent(config))
