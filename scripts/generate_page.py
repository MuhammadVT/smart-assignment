"""
Regenerate the GitHub Pages overview site (docs/index.html) from live workflow
output, so the published examples never drift from the code.

Run it whenever the mock data, constraints, scoring, or config change:

    python3 scripts/generate_page.py                 # writes docs/index.html
    python3 scripts/generate_page.py --output /tmp/preview.html

The heavy lifting lives in smart_assignment/reporting/page.py (importable and
unit-tested); this is just the CLI entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smart_assignment.reporting.page import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Smart Assignment overview page")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the HTML (default: docs/index.html at the repo root)",
    )
    args = parser.parse_args()
    out = generate(output_path=args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
