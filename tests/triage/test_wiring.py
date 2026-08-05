"""
Wiring tests for the escalation-triage sub-agent: the config flag, the
instruction composition, and the tool-name/argument contract. These stay offline
-- they never build the real LlmAgent (which would resolve the LLM backend).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.prompts import (
    TRIAGE_REQUEST_LINE,
    build_batch_instruction,
    build_instruction,
)
from smart_assignment.shared.config import Config
from smart_assignment.triage import TRIAGE_AGENT_NAME


def test_triage_enabled_by_default():
    assert Config().use_escalation_triage is True


def test_flag_reads_from_env():
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE": "false"}):
        assert Config.from_env().use_escalation_triage is False
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_USE_ESCALATION_TRIAGE": "true"}):
        assert Config.from_env().use_escalation_triage is True


def test_instruction_mentions_triage_only_when_enabled():
    with_triage = build_instruction(include_triage=True)
    without = build_instruction(include_triage=False)
    assert TRIAGE_AGENT_NAME in with_triage
    assert TRIAGE_AGENT_NAME not in without
    # The base workflow guidance is present either way.
    assert "recommend_or_escalate" in with_triage
    assert "recommend_or_escalate" in without


def test_tool_name_matches_the_instruction_reference():
    # The instruction tells the model to call this exact tool name.
    assert TRIAGE_AGENT_NAME == "escalation_triage"
    assert TRIAGE_AGENT_NAME in build_instruction(include_triage=True)


# --- the ``request`` argument contract --------------------------------------
#
# ADK's AgentTool declares ``request`` as a REQUIRED string, so the model must
# always send one, while the triage agent ignores it and reads session state via
# get_escalation_context. Unspecified, the model filled that hole with a verbatim
# paste of the whole decision result -- the oversized payload behind the
# array-wrapped tool-call arguments that crash ADK's parser. These tests pin the
# fix: one short line, named identically in both instructions, plus an explicit
# ban on pasting tool output.


def test_both_instructions_pin_the_same_request_line():
    expected = f'request="{TRIAGE_REQUEST_LINE}"'
    assert expected in build_instruction(include_triage=True)
    assert expected in build_batch_instruction(include_triage=True)


def test_request_line_stays_short_enough_to_be_harmless():
    # The whole point is that this value is a fixed label, not a payload. A long
    # line here would quietly reintroduce the blob this change removed.
    assert len(TRIAGE_REQUEST_LINE) < 120
    assert "{" not in TRIAGE_REQUEST_LINE and "}" not in TRIAGE_REQUEST_LINE


def test_both_instructions_forbid_pasting_tool_output_into_request():
    for instruction in (
        build_instruction(include_triage=True),
        build_batch_instruction(include_triage=True),
    ):
        assert "NEVER put the" in instruction
        assert "in request" in instruction


def test_request_line_is_absent_when_triage_is_disabled():
    # The instruction must never name an argument for a tool that isn't wired in.
    assert TRIAGE_REQUEST_LINE not in build_instruction(include_triage=False)
    assert TRIAGE_REQUEST_LINE not in build_batch_instruction(include_triage=False)


def test_instruction_argument_name_matches_adk_agenttool_declaration():
    """The prompt tells the model to pass ``request=``; assert that IS the argument
    ADK's AgentTool declares, so an ADK rename fails here instead of silently
    making the guidance wrong. Uses a throwaway LlmAgent with a plain string model,
    which resolves no backend and needs no credentials."""
    from google.adk.agents import LlmAgent
    from google.adk.tools import AgentTool

    stub = LlmAgent(
        name=TRIAGE_AGENT_NAME,
        model="gemini-2.0-flash",
        description="stub for the declaration contract only",
        instruction="unused",
    )
    declaration = AgentTool(agent=stub)._get_declaration()
    schema = declaration.parameters_json_schema
    if schema is None:  # older ADK builds a types.Schema instead
        schema = {
            "properties": dict(declaration.parameters.properties),
            "required": list(declaration.parameters.required),
        }
    assert "request" in schema["properties"]
    assert schema["required"] == ["request"]
