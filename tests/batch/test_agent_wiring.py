"""
Wiring tests for the BATCH agent variant (agent.build_batch_agent): the batch
instruction composition, the tool-set contract, and that it stays a distinct
entry point that never disturbs root_agent.

These stay offline. The instruction/tool-assembly tests never resolve the LLM
backend; the one test that builds the LlmAgent patches ``get_llm`` (and the triage
AgentTool) so no credentials are needed -- the same discipline as the triage
wiring tests.
"""

from __future__ import annotations

from unittest.mock import patch

from google.adk.tools import FunctionTool

from smart_assignment import agent as agent_module
from smart_assignment.prompts import build_batch_instruction, build_instruction
from smart_assignment.shared.config import Config
from smart_assignment.triage import TRIAGE_AGENT_NAME


def _tool_names(tools) -> set[str]:
    return {getattr(t, "name", None) for t in tools}


def _fake_triage_tool() -> FunctionTool:
    """A real BaseTool named like the triage AgentTool, so it passes LlmAgent's
    tool validation without building the actual (credential-resolving) sub-agent."""

    def escalation_triage(tool_context) -> str:  # noqa: ARG001 - stub for wiring only
        return ""

    return FunctionTool(escalation_triage)


# --- The batch instruction --------------------------------------------------


def test_batch_instruction_drives_the_consolidated_tool_not_the_step_tools():
    text = build_batch_instruction()
    assert "assign_prospect" in text
    # The four step-by-step tools are deliberately NOT part of the batch flow.
    for step_tool in (
        "find_candidate_routes",
        "evaluate_and_score_routes",
        "recommend_or_escalate",
    ):
        assert step_tool not in text
    # No address-resolution step (batch trusts the CRM address as-is).
    assert "resolve_address" not in text


def test_batch_instruction_mentions_triage_only_when_enabled():
    with_triage = build_batch_instruction(include_triage=True)
    without = build_batch_instruction(include_triage=False)
    assert TRIAGE_AGENT_NAME in with_triage
    assert TRIAGE_AGENT_NAME not in without
    # The consolidated flow is present either way.
    assert "assign_prospect" in with_triage
    assert "assign_prospect" in without


def test_batch_instruction_escalates_via_request_input_without_triage():
    """With triage off, the escalation path is a bare request_input record."""
    text = build_batch_instruction(include_triage=False)
    assert "request_input" in text
    assert "requires_human_review" in text


def test_batch_instruction_is_distinct_from_the_conversational_one():
    """It must not carry the multi-turn/conversational guidance (revisions, the
    step-by-step progress notes) that only makes sense with a human present."""
    batch = build_batch_instruction(include_triage=True)
    conversational = build_instruction(include_triage=True)
    assert batch != conversational
    assert "BATCH mode" in batch
    assert "BATCH mode" not in conversational


# --- The batch tool set -----------------------------------------------------


def test_batch_tools_without_triage_are_just_assign_plus_handoff():
    tools = agent_module._batch_agent_tools(Config(use_escalation_triage=False))
    names = _tool_names(tools)
    assert "assign_prospect" in names
    assert TRIAGE_AGENT_NAME not in names
    # Exactly the consolidated tool + the human-handoff record, nothing else.
    assert len(tools) == 2


def test_batch_tools_include_triage_agenttool_when_enabled():
    fake_triage = object()
    with patch.object(
        agent_module, "_batch_agent_tools", wraps=agent_module._batch_agent_tools
    ), patch("smart_assignment.triage.build_triage_tool", return_value=fake_triage) as build:
        tools = agent_module._batch_agent_tools(Config(use_escalation_triage=True))
    build.assert_called_once()
    assert fake_triage in tools
    assert "assign_prospect" in _tool_names(tools)


# --- Building the agent (offline: backend + triage tool patched) ------------


def test_build_batch_agent_wires_model_role_instruction_and_tools():
    with patch.object(agent_module, "get_llm", return_value="fake-model") as get_llm, patch(
        "smart_assignment.triage.build_triage_tool", return_value=_fake_triage_tool()
    ):
        batch_agent = agent_module.build_batch_agent(Config(use_escalation_triage=True))

    # Model resolved once, through the per-role seam (ROLE_ROOT_AGENT).
    get_llm.assert_called_once()
    assert batch_agent.model == "fake-model"
    assert batch_agent.name == "smart_assignment_batch_agent"
    assert "assign_prospect" in batch_agent.instruction
    assert TRIAGE_AGENT_NAME in batch_agent.instruction
    names = _tool_names(batch_agent.tools)
    assert "assign_prospect" in names
    assert TRIAGE_AGENT_NAME in names


def test_build_batch_agent_omits_triage_when_disabled():
    with patch.object(agent_module, "get_llm", return_value="fake-model"):
        batch_agent = agent_module.build_batch_agent(Config(use_escalation_triage=False))
    assert TRIAGE_AGENT_NAME not in batch_agent.instruction
    assert TRIAGE_AGENT_NAME not in _tool_names(batch_agent.tools)


def test_building_the_batch_agent_does_not_build_root_agent():
    """The batch agent is a separate entry point: constructing it must not touch
    the lazily-built root_agent (which would resolve the backend for the wrong
    surface)."""
    agent_module._root_agent = None  # ensure a clean slate
    with patch.object(agent_module, "get_llm", return_value="fake-model"), patch(
        "smart_assignment.triage.build_triage_tool", return_value=_fake_triage_tool()
    ):
        agent_module.build_batch_agent(Config(use_escalation_triage=True))
    assert agent_module._root_agent is None
