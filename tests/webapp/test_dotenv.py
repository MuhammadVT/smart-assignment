"""
A .env file (see .env.example) must be picked up automatically -- for both ways
of launching the webapp: `python3 scripts/run_web.py` and
`uvicorn smart_assignment.webapp.app:app` directly. Each subprocess below runs
with a deliberately clean environment (only PATH/HOME) and a temp cwd holding
just a `.env` file, so credentials can only reach the app via load_dotenv().
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_CLEAN_ENV = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root")}

_ENV_FILE = """\
SMART_ASSIGNMENT_WEBAPP_MODE=llm
SMART_ASSIGNMENT_LLM_BACKEND=sage
SAGE_CLIENT_ID=test-client-id
SAGE_CLIENT_SECRET=test-client-secret
SAGE_ENVIRONMENT=test-env
"""


def _run(code: str, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=_CLEAN_ENV, capture_output=True, text=True
    )


def test_direct_app_import_picks_up_dotenv(tmp_path):
    (tmp_path / ".env").write_text(_ENV_FILE)
    result = _run(
        "from smart_assignment.webapp.app import app\n"
        "from smart_assignment.webapp.llm_chat import resolve_mode\n"
        "from smart_assignment.shared.config import DEFAULT_CONFIG\n"
        "assert DEFAULT_CONFIG.llm_backend == 'sage', DEFAULT_CONFIG.llm_backend\n"
        "assert resolve_mode(DEFAULT_CONFIG) == {'mode': 'llm', 'configured': 'llm'}\n"
        "print('ok')\n",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_run_web_module_loads_dotenv_before_its_own_backend_default(tmp_path):
    """run_web.py's `setdefault("SMART_ASSIGNMENT_LLM_BACKEND", "standard")` must
    not clobber a real value from .env -- it only wins when nothing is set."""
    (tmp_path / ".env").write_text(_ENV_FILE)
    run_web_py = str(Path(__file__).parents[2] / "scripts" / "run_web.py")
    result = _run(
        "import runpy, os\n"
        f"runpy.run_path({run_web_py!r}, run_name='not_main')\n"
        "backend = os.environ.get('SMART_ASSIGNMENT_LLM_BACKEND')\n"
        "assert backend == 'sage', backend\n"
        "print('ok')\n",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
