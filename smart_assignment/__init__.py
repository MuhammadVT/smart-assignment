"""smart_assignment: a conversational agent for automated delivery slot assignment."""

# Load .env (if present) BEFORE importing any submodule, so configuration set
# there -- SMART_ASSIGNMENT_* flags, the LLM backend + credentials, the data
# source and geocoder -- is in os.environ before Config.from_env() (and the ADK
# agent) read it at import time. This is the single place env is loaded, so every
# entry point stays on the same configuration: the web app already called
# load_dotenv() itself, and now `adk run` / `adk web` / `adk deploy` (which import
# this package but never called it) pick up the exact same .env instead of
# silently ignoring it. load_dotenv() does not override variables already exported
# in the shell, and is a no-op when no .env file exists.
from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

# Load the repo-root .env by an absolute, package-relative path first (so it is
# found regardless of the process's working directory), then a CWD search as a
# fallback. load_dotenv does not override variables already in the environment.
_load_dotenv(_Path(__file__).resolve().parent.parent / ".env")
_load_dotenv()

# NOTE: the `agent` submodule is deliberately NOT imported here. It imports
# Google ADK at module level, so an eager import made every consumer of this
# package -- including the agent-free fast path (`smart_assignment.runtime`), the
# offline scripts, and the eval scorer -- pay the ADK import cost for code they
# never touch. Nothing needs the eager binding: ADK, the web app, and the tests
# all reach the agent by importing the submodule explicitly
# (`from smart_assignment.agent import root_agent`), which works exactly the same
# without it. `root_agent` itself stays lazy (PEP 562 in agent.py), so that import
# is still credential-free.

__all__: list[str] = []
