"""
One-shot assignment CLI -- the agent-free fast path from a shell.

Where ``scripts/run_local.py`` is a *demo* that prints a human-readable trace for
the bundled mock customers, this is an **integration entry point**: give it a real
prospect (or a JSONL file of them) and it emits the decision as JSON, ready to
pipe into anything. Both drive the same deterministic pipeline; this one goes
through ``smart_assignment.runtime`` so it shares the API's exact payload shape
and cost profiles.

Runs fully OFFLINE under the default ``economy`` profile -- no credentials, no LLM
calls, no ADK import.

Run:
    # one prospect
    python3 scripts/run_assign.py --address "1200 McKinney St, Houston, TX 77010" \\
        --cases 90 --day TUE --window 07:00-10:00

    # a batch: one JSON object per line, keys matching runtime.assign's arguments
    python3 scripts/run_assign.py --batch prospects.jsonl --profile balanced
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from smart_assignment import runtime


def _split_window(window: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``"07:00-10:00"`` -> ``("07:00", "10:00")``; ``None`` -> ``(None, None)``."""
    if not window:
        return None, None
    parts = window.replace("--", "-").split("-")
    if len(parts) != 2:
        raise SystemExit(f"--window must look like 07:00-10:00, got {window!r}")
    return parts[0].strip(), parts[1].strip()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--address", help="Prospect street address (primary identifier).")
    parser.add_argument("--cases", type=int, help="Order quantity, in cases.")
    parser.add_argument("--day", help="Preferred day: MON/TUE/WED/THU/FRI/SAT.")
    parser.add_argument("--window", help='Preferred window, "HH:MM-HH:MM".')
    parser.add_argument("--name", help="Business/contact name (optional).")
    parser.add_argument("--customer-number", help="Existing Sysco number (optional).")
    parser.add_argument(
        "--batch",
        help="Path to a JSONL file of prospects (one JSON object per line); "
        "'-' reads stdin. Mutually exclusive with --address.",
    )
    parser.add_argument(
        "--profile",
        default=runtime.DEFAULT_PROFILE,
        choices=list(runtime.PROFILES),
        help="Cost profile (default: %(default)s -- no LLM calls).",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 for compact).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    indent = args.indent or None

    if args.batch:
        stream = sys.stdin if args.batch == "-" else open(args.batch, encoding="utf-8")
        try:
            prospects = [json.loads(line) for line in stream if line.strip()]
        finally:
            if stream is not sys.stdin:
                stream.close()
        results = runtime.assign_batch(prospects, profile=args.profile)
        print(json.dumps(results, indent=indent))
        # Non-zero only if EVERY prospect failed -- a partial batch is a success
        # with per-item errors the caller can inspect.
        return 0 if any(r.get("ok") for r in results) else 1

    if not args.address or args.cases is None:
        raise SystemExit("Provide --address and --cases, or --batch <file>.")

    start, end = _split_window(args.window)
    result = runtime.assign(
        args.address,
        args.cases,
        preferred_day=args.day,
        preferred_window_start=start,
        preferred_window_end=end,
        name=args.name,
        customer_number=args.customer_number,
        profile=args.profile,
    )
    print(json.dumps(result, indent=indent))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
