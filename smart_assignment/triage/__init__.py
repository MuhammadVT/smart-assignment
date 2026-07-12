"""
Escalation triage: the first real multi-agent split in this project.

When ``root_agent`` escalates a slot recommendation for human review, it calls
the ``escalation_triage`` sub-agent (an ``LlmAgent`` exposed as an
``AgentTool``) to turn the full evaluation trace into a concise specialist
brief -- the root cause, concrete remediation options, and the question to ask
-- before the ``request_input`` handoff.

Why an AgentTool (consult-and-return), not a peer agent with control transfer:
``root_agent`` stays in control of the conversation and keeps ownership of the
``request_input`` human-in-the-loop pause; triage is a bounded call that
returns a brief. It runs strictly *downstream* of the deterministic decision
and only reads state (via ``get_escalation_context``) -- it never changes the
route, the score, or the decision, so the pipeline's auditability is untouched.

Everything is built lazily by ``root_agent``'s own construction
(``smart_assignment/agent.py``), so importing the package stays credential-free.
"""

from __future__ import annotations

from smart_assignment.triage.agent import (
    TRIAGE_AGENT_NAME,
    build_triage_agent,
    build_triage_tool,
)
from smart_assignment.triage.context import check_brief_grounding, get_escalation_context
from smart_assignment.triage.verifier import BriefVerification, collect_grounding, verify_brief

__all__ = [
    "TRIAGE_AGENT_NAME",
    "build_triage_agent",
    "build_triage_tool",
    "get_escalation_context",
    "check_brief_grounding",
    "BriefVerification",
    "collect_grounding",
    "verify_brief",
]
