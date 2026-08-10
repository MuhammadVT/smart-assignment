"""Hermetic tests for eval/run_config.py's resolve_num_runs() -- no LLM backend
needed, and no ``google-adk[eval]`` either, which is exactly why the resolution
lives in its own module rather than inside eval/test_eval.py (that file imports
AgentEvaluator, from the ``eval`` extra the hermetic suite deliberately lacks).
"""

from __future__ import annotations

import pytest

from eval.run_config import DEFAULT_NUM_RUNS, NUM_RUNS_ENV, resolve_num_runs


def test_default_is_one_run_per_case(monkeypatch):
    # The point of the module: ADK's own default is 2, and this suite overrides
    # it to 1. A change here doubles live LLM cost on every credentialed run.
    monkeypatch.delenv(NUM_RUNS_ENV, raising=False)
    assert DEFAULT_NUM_RUNS == 1
    assert resolve_num_runs() == 1


def test_env_override_raises_the_replay_count(monkeypatch):
    monkeypatch.setenv(NUM_RUNS_ENV, "3")
    assert resolve_num_runs() == 3


def test_blank_or_whitespace_value_falls_back_to_the_default(monkeypatch):
    # An env var set to "" (a commented-out .env line re-added empty, or an
    # unset CI variable expanding to nothing) means "not configured", not "0".
    for blank in ("", "   "):
        monkeypatch.setenv(NUM_RUNS_ENV, blank)
        assert resolve_num_runs() == DEFAULT_NUM_RUNS


def test_non_integer_value_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setenv(NUM_RUNS_ENV, "two")
    with pytest.raises(ValueError):
        resolve_num_runs()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_below_one_is_rejected_with_an_explanation(monkeypatch, value):
    # ADK would take num_runs=0 as "replay nothing" and report success over zero
    # cases -- the silent green this suite exists to prevent (cf. the dropped-case
    # guard in eval/inference_guard.py).
    monkeypatch.setenv(NUM_RUNS_ENV, value)
    with pytest.raises(ValueError) as excinfo:
        resolve_num_runs()
    assert NUM_RUNS_ENV in str(excinfo.value)
    assert "at least 1" in str(excinfo.value)
