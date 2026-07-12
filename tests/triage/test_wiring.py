"""
Wiring tests for the escalation-triage sub-agent: the config flag, the
instruction composition, and the tool-name contract. These stay offline -- they
never build the LlmAgent (which would resolve the LLM backend).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.prompts import build_instruction
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
