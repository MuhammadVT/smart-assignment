"""
Deploy the smart_assignment agent to Google's Agent Runtime.

[ASSUMPTION] This follows the deployment convention used across Google's
own adk-samples repo (a deploy.py invoked from inside deployment/ so
relative paths resolve correctly). It has NOT been run against a real
GCP project as part of building this repo -- verify project/region
values and IAM permissions before running in your environment.

Usage (from inside the deployment/ directory):
    python3 deploy.py --project YOUR_PROJECT_ID --region us-central1

Equivalent ADK CLI form (run from repo root):
    adk deploy agent_engine \\
        --env_file .env \\
        --region=us-central1 \\
        smart_assignment

See: https://google.github.io/adk-docs/deploy/agent-runtime/deploy/
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
AGENT_PATH = REPO_ROOT / "smart_assignment"
ENV_FILE = REPO_ROOT / ".env"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy smart_assignment to Agent Runtime")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    args = parser.parse_args()

    if not ENV_FILE.exists():
        print(
            f"ERROR: {ENV_FILE} not found. Copy .env.example to .env and "
            f"fill in required values before deploying.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        "adk",
        "deploy",
        "agent_engine",
        "--env_file",
        str(ENV_FILE),
        "--region",
        args.region,
        "--project",
        args.project,
        str(AGENT_PATH),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
