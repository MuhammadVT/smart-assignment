"""
Per-role model selection: Config.for_role / resolved_model and the wiring that
routes each LLM-using surface through its role's model.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.shared.config import (
    ROLE_JUDGMENT,
    ROLE_QUALITY_JUDGE,
    ROLE_ROOT_AGENT,
    ROLE_TRIAGE,
    Config,
)


# --- Config.for_role / resolved_model ---------------------------------------


def test_no_override_returns_the_same_config_object():
    c = Config(llm_backend="standard", model="gemini-2.5-flash")
    # Identity: no override means no copy, so behavior is provably unchanged.
    assert c.for_role(ROLE_ROOT_AGENT) is c
    assert c.resolved_model(ROLE_ROOT_AGENT) == "gemini-2.5-flash"


def test_standard_backend_overrides_model_field_only():
    c = Config(
        llm_backend="standard",
        model="gemini-2.5-flash",
        sage_model="sage-gemini-2.5-flash",
        role_models={ROLE_TRIAGE: "gemini-2.5-flash-lite"},
    )
    scoped = c.for_role(ROLE_TRIAGE)
    assert scoped.model == "gemini-2.5-flash-lite"
    assert scoped.sage_model == "sage-gemini-2.5-flash"  # untouched
    assert c.model == "gemini-2.5-flash"  # original unchanged (a copy was made)
    assert c.resolved_model(ROLE_TRIAGE) == "gemini-2.5-flash-lite"


def test_sage_backend_overrides_sage_model_field_only():
    c = Config(
        llm_backend="sage",
        model="gemini-2.5-flash",
        sage_model="sage-gemini-2.5-flash",
        role_models={ROLE_JUDGMENT: "sage-gemini-2.5-pro"},
    )
    scoped = c.for_role(ROLE_JUDGMENT)
    assert scoped.sage_model == "sage-gemini-2.5-pro"
    assert scoped.model == "gemini-2.5-flash"  # untouched
    assert c.resolved_model(ROLE_JUDGMENT) == "sage-gemini-2.5-pro"


def test_each_role_resolves_independently():
    c = Config(
        llm_backend="standard",
        model="base",
        role_models={ROLE_TRIAGE: "lite", ROLE_JUDGMENT: "pro"},
    )
    assert c.resolved_model(ROLE_ROOT_AGENT) == "base"  # default
    assert c.resolved_model(ROLE_TRIAGE) == "lite"
    assert c.resolved_model(ROLE_JUDGMENT) == "pro"


def test_role_models_read_from_env():
    env = {
        "SMART_ASSIGNMENT_MODEL_ROOT_AGENT": "m-root",
        "SMART_ASSIGNMENT_MODEL_TRIAGE": "m-triage",
        "SMART_ASSIGNMENT_MODEL_JUDGMENT": "m-judge",
    }
    with patch.dict(os.environ, env):
        c = Config.from_env()
    assert c.role_models[ROLE_ROOT_AGENT] == "m-root"
    assert c.role_models[ROLE_TRIAGE] == "m-triage"
    assert c.role_models[ROLE_JUDGMENT] == "m-judge"


def test_quality_judge_role_model_read_from_env():
    # eval/test_quality.py's role, added for Phase 3a -- not a product
    # decision-layer role, but resolved through the same generic mechanism.
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_MODEL_QUALITY_JUDGE": "m-quality-judge"}):
        c = Config.from_env()
    assert c.role_models[ROLE_QUALITY_JUDGE] == "m-quality-judge"
    assert c.resolved_model(ROLE_QUALITY_JUDGE) == "m-quality-judge"


def test_unset_role_env_yields_no_override():
    # With none of the SMART_ASSIGNMENT_MODEL_* set, role_models is empty and
    # every role resolves to the global model.
    keys = [
        "SMART_ASSIGNMENT_MODEL_ROOT_AGENT",
        "SMART_ASSIGNMENT_MODEL_TRIAGE",
        "SMART_ASSIGNMENT_MODEL_JUDGMENT",
        "SMART_ASSIGNMENT_MODEL_ADDRESS_RESOLVE",
    ]
    with patch.dict(os.environ, {k: "" for k in keys}):
        c = Config.from_env()
    assert c.role_models == {}


# --- wiring: the LLM surfaces route through their role's model ---------------


def test_root_agent_and_triage_build_with_distinct_models(monkeypatch):
    import smart_assignment.agent as agent_module

    cfg = Config(
        llm_backend="standard",
        model="gemini-2.5-flash",
        role_models={ROLE_TRIAGE: "gemini-2.5-flash-lite"},
        use_escalation_triage=True,
    )
    # _build_root_agent reads the module-level DEFAULT_CONFIG.
    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG", cfg)
    root = agent_module._build_root_agent()

    assert root.model == "gemini-2.5-flash"  # root_agent role -> default
    triage_tool = next(t for t in root.tools if getattr(t, "name", "") == "escalation_triage")
    assert triage_tool.agent.model == "gemini-2.5-flash-lite"  # triage role -> lite


def test_route_slot_choice_uses_the_judgment_role_model(monkeypatch):
    import smart_assignment.shared.llm as llm_module
    from smart_assignment.routeslot.llm import generate_route_slot_choice

    captured = {}

    # The route-slot choice now comes from a tool call: capture the model the
    # judgment-role-scoped config resolved to, and return the decision as the
    # tool's arguments (call_args, text).
    def fake_generate_tool_call(config, prompt, tool, role=None):
        captured["model"] = config.model
        return {"chosen_index": 0, "decision_summary": "ok", "primary_reasons": ["r"]}, ""

    monkeypatch.setattr(llm_module, "generate_tool_call", fake_generate_tool_call)
    cfg = Config(llm_backend="standard", model="base", role_models={ROLE_JUDGMENT: "judge-model"})
    generate_route_slot_choice(cfg, "prompt")
    assert captured["model"] == "judge-model"
