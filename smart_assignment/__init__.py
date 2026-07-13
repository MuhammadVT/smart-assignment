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
from dotenv import load_dotenv as _load_dotenv

_load_dotenv()

from smart_assignment import agent  # noqa: E402  (must follow load_dotenv above)

__all__ = ["agent"]
