"""
Which tools ``root_agent`` is built with, under each orchestration shape
(``Config.use_consolidated_pipeline_tool``).

This is the seam the flag actually acts on -- one branch in
``agent._build_root_agent`` and one in ``prompts.build_instruction`` -- so it is
the thing worth pinning: flag-off must register exactly today's four step tools,
flag-on exactly one, and both must keep the non-pipeline tools (human handoff,
address resolution, triage) intact.
"""

from __future__ import annotations

import pytest

from smart_assignment.shared.config import Config

_STEP_TOOLS = {
    "intake_customer",
    "find_candidate_routes",
    "evaluate_and_score_routes",
    "recommend_or_escalate",
}


def _tool_names(root) -> set[str]:
    names = set()
    for tool in root.tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name:
            names.add(name)
    return names


def _build(monkeypatch, **overrides):
    import smart_assignment.agent as agent_module

    cfg = Config(llm_backend="standard", model="gemini-2.5-flash", **overrides)
    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG", cfg)
    return agent_module._build_root_agent()


def test_flag_off_registers_the_four_step_tools(monkeypatch):
    names = _tool_names(_build(monkeypatch, use_escalation_triage=False))
    assert _STEP_TOOLS <= names
    assert "assign_delivery_slot" not in names


def test_flag_on_registers_exactly_one_pipeline_tool(monkeypatch):
    names = _tool_names(
        _build(
            monkeypatch,
            use_consolidated_pipeline_tool=True,
            use_escalation_triage=False,
        )
    )
    assert "assign_delivery_slot" in names
    assert not (_STEP_TOOLS & names), "step tools must not be registered in consolidated mode"


@pytest.mark.parametrize("consolidated", [False, True])
def test_non_pipeline_tools_survive_both_shapes(monkeypatch, consolidated):
    """Consolidation collapses the deterministic chain only -- the human handoff,
    address resolution, and triage all run before or after it, so they must be
    present either way."""
    names = _tool_names(
        _build(
            monkeypatch,
            use_consolidated_pipeline_tool=consolidated,
            use_address_resolution=True,
            use_escalation_triage=True,
        )
    )
    assert "resolve_address" in names
    assert "escalation_triage" in names
    # ADK's built-in human-in-the-loop tool (registered under its own name).
    assert any("request_input" in n for n in names)


@pytest.mark.parametrize("consolidated", [False, True])
def test_instruction_matches_the_registered_tools(monkeypatch, consolidated):
    """The instruction must never name a tool the agent wasn't given."""
    root = _build(
        monkeypatch,
        use_consolidated_pipeline_tool=consolidated,
        use_address_resolution=True,
        use_escalation_triage=True,
    )
    names = _tool_names(root)
    for candidate in _STEP_TOOLS | {"assign_delivery_slot"}:
        if candidate in root.instruction:
            assert candidate in names, f"instruction names unregistered tool {candidate}"
