"""
The escalation-triage sub-agent and its ``AgentTool`` wrapper.

``build_triage_agent`` builds an ``LlmAgent`` with a single read-only tool
(``get_escalation_context``); ``build_triage_tool`` wraps it as an ``AgentTool``
so ``root_agent`` can call it like any other tool. The sub-agent's ``name`` is
the tool name ``root_agent`` uses (``escalation_triage``).

Both call ``get_llm`` (which resolves the configured backend and may need
credentials), so they must be invoked lazily -- ``smart_assignment/agent.py``
does so only while building ``root_agent``, never at import time.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, FunctionTool

from smart_assignment.shared.config import Config
from smart_assignment.shared.llm import get_llm
from smart_assignment.triage.context import get_escalation_context
from smart_assignment.triage.prompts import TRIAGE_INSTRUCTION

# The sub-agent's name doubles as the tool name root_agent calls it by; the
# root instruction (prompts.ESCALATION_TRIAGE_GUIDANCE) references this exact
# string, so keep them in sync.
TRIAGE_AGENT_NAME = "escalation_triage"


def build_triage_agent(config: Config) -> LlmAgent:
    """Construct the escalation-triage ``LlmAgent`` (resolves the LLM backend)."""
    return LlmAgent(
        name=TRIAGE_AGENT_NAME,
        model=get_llm(config),
        description=(
            "Turns an escalated slot recommendation into a concise specialist "
            "brief -- root cause, concrete remediation options, and the question "
            "to ask -- grounded in the evaluation trace. Never changes the decision."
        ),
        instruction=TRIAGE_INSTRUCTION,
        tools=[FunctionTool(get_escalation_context)],
    )


def build_triage_tool(config: Config) -> AgentTool:
    """The triage agent wrapped as an ``AgentTool`` for root_agent's tools list."""
    return AgentTool(agent=build_triage_agent(config))
