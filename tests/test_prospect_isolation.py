"""
Regression net for the cross-prospect contamination bug, at the surface where it
was worst: ONE ADK session, TWO prospects, NO rotation layer -- exactly what
``adk web`` / ``adk run`` do. Everything is real ADK machinery (Runner, session
state, FunctionTool dispatch) and the real pipeline tools; only the MODEL is
scripted, replaying the observed tool-call sequence deterministically with no
backend or credentials.

The bug this pins (observed live before the fix): prospect A concluded with a
TUE 07:00-10:00 preference; prospect B arrived phrased so no rotation fired and
stated no preference; intake's merge carried A's slot into B, and B was decided
against a delivery preference its customer never expressed.
"""

from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import patch

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.tools import slot_recommendation as tools_module
from smart_assignment.tools.slot_recommendation import (
    _STATE_PROFILE_KEY,
    intake_customer,
    recommend_or_escalate,
    start_new_prospect,
)

_ADDR_A = "1200 McKinney St, Houston, TX 77010"
_ADDR_B = "5000 Katy Mills Cir, Katy, TX 77494"


@pytest.fixture(autouse=True)
def _use_mock_geocoder():
    with patch.object(tools_module, "_GEOCODER", MockGeocoder()):
        yield


def _scripted_llm(steps):
    """A BaseLlm whose every generate call pops the next scripted step: a
    ("call", name, args) tool call or a ("text", message) final reply. Replays
    the model's observed behavior exactly, with zero model nondeterminism."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    queue = deque(steps)

    class ScriptedLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            kind, *rest = queue.popleft()
            if kind == "call":
                name, args = rest
                part = types.Part.from_function_call(name=name, args=args)
            else:
                part = types.Part(text=rest[0])
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    return ScriptedLlm(model="scripted")


def _agent(steps):
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool

    return LlmAgent(
        name="scripted_agent",
        model=_scripted_llm(steps),
        description="scripted replay of the observed tool-call sequence",
        instruction="unused",
        tools=[
            FunctionTool(intake_customer),
            FunctionTool(start_new_prospect),
            FunctionTool(recommend_or_escalate),
        ],
    )


async def _run_turns(agent, turns):
    """Drive each turn through a real Runner against ONE session, then return
    that session's final state -- the profile a next decision would use."""
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    service = InMemorySessionService()
    await service.create_session(app_name="iso", user_id="u", session_id="one")
    runner = Runner(agent=agent, app_name="iso", session_service=service)
    for message in turns:
        async for _ in runner.run_async(
            user_id="u",
            session_id="one",
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            run_config=RunConfig(streaming_mode=StreamingMode.NONE),
        ):
            pass
    session = await service.get_session(app_name="iso", user_id="u", session_id="one")
    return (session.state or {}).get(_STATE_PROFILE_KEY) or {}


_TURN1_STEPS = [
    (
        "call",
        "intake_customer",
        {
            "address": _ADDR_A,
            "order_quantity_cases": 90,
            "preferred_day": "TUE",
            "preferred_window_start": "07:00",
            "preferred_window_end": "10:00",
        },
    ),
    ("call", "recommend_or_escalate", {}),
    ("text", "Decision presented."),
]


def test_adk_web_new_prospect_does_not_inherit_the_previous_slot():
    """THE captured leak, replayed through real ADK with no rotation anywhere:
    prospect B states address+cases only, and must not be decided with A's
    TUE 07:00-10:00. Guarded by intake_customer's deterministic reset."""
    steps = _TURN1_STEPS + [
        ("call", "intake_customer", {"address": _ADDR_B, "order_quantity_cases": 260}),
        ("call", "recommend_or_escalate", {}),
        ("text", "Second decision presented."),
    ]
    profile = asyncio.run(
        _run_turns(_agent(steps), ["prospect A", "new customer at Katy Mills"])
    )
    assert profile["address"] == _ADDR_B
    assert profile["order_quantity_cases"] == 260
    # These three were 'TUE'/'07:00'/'10:00' before the fix.
    assert profile.get("preferred_day") is None
    assert profile.get("preferred_window_start") is None
    assert profile.get("preferred_window_end") is None


def test_adk_web_no_address_switch_is_stopped_by_the_boundary_tool():
    """'Another customer, 40 cases' -- indistinguishable from a revision at the
    tool level, so the scripted model declares the switch (as observed live).
    The new customer's order must not run against A's address."""
    steps = _TURN1_STEPS + [
        ("call", "start_new_prospect", {}),
        ("call", "intake_customer", {"order_quantity_cases": 40}),
        ("text", "I still need the new customer's address."),
    ]
    profile = asyncio.run(
        _run_turns(_agent(steps), ["prospect A", "another customer, 40 cases"])
    )
    assert profile.get("address") is None  # A's address is gone
    assert profile.get("order_quantity_cases") == 40
    assert profile.get("preferred_day") is None


def test_adk_web_revision_still_keeps_the_profile():
    """Control: a real revision must keep merging, or the fix broke the feature
    the merge exists for."""
    steps = _TURN1_STEPS + [
        ("call", "intake_customer", {"order_quantity_cases": 20}),
        ("call", "recommend_or_escalate", {}),
        ("text", "Updated decision presented."),
    ]
    profile = asyncio.run(_run_turns(_agent(steps), ["prospect A", "try 20 cases"]))
    assert profile["address"] == _ADDR_A
    assert profile["order_quantity_cases"] == 20
    assert profile["preferred_day"] == "TUE"
