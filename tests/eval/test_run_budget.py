"""Hermetic tests for eval/run_budget.py (no LLM backend, no live agent).

The budget exists because a hung backend once ran a full eval for 9.8 hours. These
drive it with sleeping coroutines and a tiny budget, so both the behaviour and the
*legibility* of the failure are pinned without a backend.
"""

from __future__ import annotations

import asyncio

import pytest

from eval.run_budget import (
    BUDGET_ENV,
    DEFAULT_BUDGET_SECONDS,
    EvalRunBudgetExceeded,
    resolve_budget_seconds,
    run_budget,
)


async def _under_budget():
    async with run_budget():
        await asyncio.sleep(0)
    return "finished"


async def _over_budget():
    async with run_budget():
        await asyncio.sleep(60)  # far past any budget these tests set


def test_a_run_inside_the_budget_is_untouched(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "5")
    assert asyncio.run(_under_budget()) == "finished"


def test_a_run_past_the_budget_is_aborted(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "0.05")
    with pytest.raises(EvalRunBudgetExceeded):
        asyncio.run(_over_budget())


def test_the_failure_message_is_actionable(monkeypatch):
    """The whole point is that whoever watched the suite go red can tell what
    happened without reading the source."""
    monkeypatch.setenv(BUDGET_ENV, "0.05")
    with pytest.raises(EvalRunBudgetExceeded) as excinfo:
        asyncio.run(_over_budget())

    message = str(excinfo.value)
    assert "BUDGET STOP, not a scoring failure" in message  # what it is not
    assert "timed out" in message and "getaddrinfo" in message  # what to grep for
    assert f"{BUDGET_ENV}=2400" in message  # how to change it
    assert "eval/run_budget.py" in message  # where it lives


def test_a_real_failure_is_not_reshaped_into_a_timeout(monkeypatch):
    # A genuine scoring failure must surface as itself, not as a budget stop.
    monkeypatch.setenv(BUDGET_ENV, "5")

    async def _fails():
        async with run_budget():
            raise AssertionError("tool_trajectory_avg_score Failed")

    with pytest.raises(AssertionError, match="tool_trajectory_avg_score"):
        asyncio.run(_fails())


def test_the_applied_budget_is_yielded(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "7.5")

    async def _peek():
        async with run_budget() as budget:
            return budget

    assert asyncio.run(_peek()) == 7.5


def test_budget_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    assert resolve_budget_seconds() == DEFAULT_BUDGET_SECONDS


def test_budget_reads_the_env_var(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "2400")
    assert resolve_budget_seconds() == 2400.0


def test_the_default_is_loose_enough_not_to_fire_on_a_slow_run():
    # Healthy full runs measured 77-161s. Guard the headroom so nobody tightens
    # this into a flaky failure without reading why it is generous.
    assert DEFAULT_BUDGET_SECONDS >= 900
