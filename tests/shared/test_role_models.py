"""
Per-role model selection: Config.for_role / resolved_model and the wiring that
routes each LLM-using surface through its role's model.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.shared.config import (
    ROLE_JUDGMENT,
    ROLE_REASONING,
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
    assert c.resolved_model(ROLE_REASONING) == "base"  # default


def test_role_models_read_from_env():
    env = {
        "SMART_ASSIGNMENT_MODEL_ROOT_AGENT": "m-root",
        "SMART_ASSIGNMENT_MODEL_TRIAGE": "m-triage",
        "SMART_ASSIGNMENT_MODEL_JUDGMENT": "m-judge",
        "SMART_ASSIGNMENT_MODEL_REASONING": "m-reason",
    }
    with patch.dict(os.environ, env):
        c = Config.from_env()
    assert c.role_models[ROLE_ROOT_AGENT] == "m-root"
    assert c.role_models[ROLE_TRIAGE] == "m-triage"
    assert c.role_models[ROLE_JUDGMENT] == "m-judge"
    assert c.role_models[ROLE_REASONING] == "m-reason"


def test_unset_role_env_yields_no_override():
    # With none of the SMART_ASSIGNMENT_MODEL_* set, role_models is empty and
    # every role resolves to the global model.
    keys = [
        "SMART_ASSIGNMENT_MODEL_ROOT_AGENT",
        "SMART_ASSIGNMENT_MODEL_TRIAGE",
        "SMART_ASSIGNMENT_MODEL_JUDGMENT",
        "SMART_ASSIGNMENT_MODEL_REASONING",
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


def test_generate_judgment_uses_the_judgment_role_model(monkeypatch):
    import smart_assignment.shared.llm as llm_module
    from smart_assignment.judgment.llm import generate_judgment

    captured = {}

    def fake_generate_text(config, prompt):
        captured["model"] = config.model
        return '{"decision":"RECOMMEND","confidence":"HIGH",' \
               '"recommended_route_id":"X","rationale":"ok","citations":[]}'

    monkeypatch.setattr(llm_module, "generate_text", fake_generate_text)
    cfg = Config(llm_backend="standard", model="base", role_models={ROLE_JUDGMENT: "judge-model"})
    generate_judgment(cfg, "prompt")
    assert captured["model"] == "judge-model"


def test_llm_reasoner_uses_the_reasoning_role_model(monkeypatch):
    import smart_assignment.shared.llm as llm_module
    from smart_assignment.reasoning import LLMReasoner

    captured = {}

    def fake_generate_text(config, prompt):
        captured["model"] = config.model
        return "narrative"

    monkeypatch.setattr(llm_module, "generate_text", fake_generate_text)
    cfg = Config(llm_backend="standard", model="base", role_models={ROLE_REASONING: "reason-model"})
    reasoner = LLMReasoner(cfg)
    # No feasible candidates -> a short deterministic trace is passed to the LLM.
    reasoner.explain(customer=_Cust(), ranked=[], infeasible=[], total_score=0.0, config=cfg)
    assert captured["model"] == "reason-model"


class _Cust:
    name = "Test Co"
    address = "1 Main St"
