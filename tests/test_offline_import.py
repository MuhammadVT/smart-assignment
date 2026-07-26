"""
The package must import offline with no LLM credentials.

Under the default ``sage`` backend, building the agent needs credentials. Two
mechanisms keep that off the import path, and these tests pin both with
controlled, credential-free environments in a subprocess:

* ``agent.root_agent`` is constructed lazily (PEP 562), so even importing
  ``smart_assignment.agent`` directly never resolves the LLM backend.
* ``smart_assignment/__init__.py`` does not import the ``agent`` submodule at
  all, so importing the package (or the agent-free ``runtime`` fast path) pulls
  in no Google ADK -- the offline scripts, the eval scorer, and any
  decisions-only service skip that import cost entirely.
"""

from __future__ import annotations

import os
import subprocess
import sys

# A clean environment: no backend override (so it defaults to "sage") and no
# credentials for any backend.
_STRIP_PREFIXES = ("SAGE_", "GOOGLE_", "OPENAI_", "SMART_ASSIGNMENT_LLM", "SMART_ASSIGNMENT_MODEL")
_BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(_STRIP_PREFIXES)}


def _run(code: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_package_imports_without_credentials():
    """Default (sage) backend, no creds -> importing the package must succeed."""
    result = _run(
        "import smart_assignment\n"
        "import smart_assignment.agent\n"
        "import smart_assignment.pipeline\n"
        "from smart_assignment.mock_customers import SAMPLE_CUSTOMERS\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_run_slot_recommendation_runs_offline():
    """The full pipeline runs with no creds (LLMReasoner self-falls-back)."""
    result = _run(
        "from smart_assignment.mock_customers import SAMPLE_CUSTOMERS\n"
        "from smart_assignment.pipeline import run_slot_recommendation\n"
        "r = run_slot_recommendation(SAMPLE_CUSTOMERS[0])\n"
        "assert r.recommendation.reasoning\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_root_agent_builds_under_standard_backend_without_key():
    """Accessing root_agent constructs it; the standard backend needs no key just
    to build (a bare Gemini model resolves to a string)."""
    result = _run(
        "from smart_assignment.agent import root_agent\n"
        "assert root_agent.name == 'smart_assignment_agent'\n"
        "print('ok')\n",
        extra_env={
            "SMART_ASSIGNMENT_LLM_BACKEND": "standard",
            "SMART_ASSIGNMENT_MODEL": "gemini-2.5-flash",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_root_agent_access_still_requires_sage_credentials():
    """Lazy does not weaken the contract: building under sage with no creds still
    raises (only now at access, not at import)."""
    result = _run(
        "from smart_assignment.agent import root_agent\n" "print(root_agent)\n",
        extra_env={"SMART_ASSIGNMENT_LLM_BACKEND": "sage"},
    )
    assert result.returncode != 0
    assert "SAGE" in result.stderr


def test_importing_the_package_does_not_import_google_adk():
    """The package must not drag the ADK in for consumers that never touch the
    agent -- `smart_assignment/__init__.py` deliberately omits the eager
    `from smart_assignment import agent` it used to carry."""
    result = _run(
        "import sys\n"
        "import smart_assignment\n"
        "import smart_assignment.pipeline\n"
        "adk = [m for m in sys.modules if m.startswith('google.adk')]\n"
        "assert not adk, adk\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_agent_submodule_is_still_reachable_without_the_eager_import():
    """Dropping the eager import must not break ADK discovery, which reaches the
    agent as a submodule attribute."""
    result = _run(
        "from smart_assignment import agent\n"
        "assert hasattr(agent, '_build_root_agent')\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
