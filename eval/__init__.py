"""Agent-level evaluation for smart_assignment.

This package holds the golden dataset + tooling for ADK's ``AgentEvaluator``,
which replays scripted intake conversations against the real ``root_agent`` and
scores them on TRAJECTORY (did the agent drive the deterministic pipeline in the
right order) and, once real reference responses are captured, FINAL-RESPONSE
quality.

It is kept OUT of the hermetic unit suite on purpose: ``pyproject.toml`` sets
``testpaths = ["tests"]``, so ``eval/test_eval.py`` runs only when explicitly
targeted (the advisory ``agent-eval`` CI job), because -- unlike the unit tests --
it needs a live LLM backend to run.
"""
