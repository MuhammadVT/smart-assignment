"""Pytest config for ds_utils_lite tests (isolated from repo root conftest)."""

import os

# Avoid importing smart_assignment when parent conftest is cut off.
os.environ.setdefault("SMART_ASSIGNMENT_LLM_BACKEND", "standard")
os.environ.setdefault("SMART_ASSIGNMENT_MODEL", "gemini-2.5-flash")
