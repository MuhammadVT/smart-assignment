"""Launch the delivery window Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parent / "dlvr_window" / "app.py"
    source_flags = {"--sample", "--sql", "--cache"}
    streamlit_args: list[str] = []
    app_args: list[str] = []

    for arg in sys.argv[1:]:
        if arg in source_flags:
            app_args.append(arg)
        else:
            streamlit_args.append(arg)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *streamlit_args]
    if app_args:
        cmd.extend(["--", *app_args])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
