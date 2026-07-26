"""
Run the headless decision service (`smart_assignment.service`) from the command
line — one prospect, or a batch.

This is the non-conversational path: no agent, no chat, no address confirmation.
Deterministic steps 1-4 in plain Python, then the same grounded step-5 decision
the conversational agent makes, under the same configuration. It exists both as a
real batch entry point and as the way to eyeball Mode 3 without a UI.

    # one prospect
    python scripts/run_assign.py --address "1200 McKinney St, Houston, TX 77010" \
        --cases 90 --day TUE --from 07:00 --to 10:00

    # the bundled mock prospects (offline: pair with SMART_ASSIGNMENT_DATA_SOURCE=mock)
    python scripts/run_assign.py --samples

    # a batch: one JSON object per line, the keys from_salesforce_record accepts
    python scripts/run_assign.py --file prospects.jsonl --out results.jsonl

    # add --json for machine-readable output, --html DIR for the SC-facing panel
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable, Optional

from smart_assignment import service
from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import CustomerProfile

_RULE = "=" * 78


def _load_jsonl(path: pathlib.Path) -> list[CustomerProfile]:
    """Read prospect records, one JSON object per line.

    A record that cannot be parsed is reported and skipped rather than aborting
    the file: the same principle the batch itself follows, applied at the door.
    """
    prospects: list[CustomerProfile] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            prospects.append(service.from_salesforce_record(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  ! line {lineno} skipped: {exc}", file=sys.stderr)
    return prospects


def _profile_from_args(args: argparse.Namespace) -> CustomerProfile:
    return service.from_salesforce_record(
        {
            "address": args.address,
            "order_quantity_cases": args.cases,
            "name": args.name,
            "customer_number": args.customer_number,
            "preferred_day": args.day,
            "preferred_window_start": getattr(args, "from"),
            "preferred_window_end": args.to,
        }
    )


def _print_outcome(outcome: service.AssignmentOutcome) -> None:
    """A compact, auditable summary — the decision, why, and the evidence rows."""
    who = outcome.customer
    print(_RULE)
    # ASCII only in printed output: a Windows console is cp1252/cp437 and a
    # non-ASCII character comes out mangled (same convention as run_local.py).
    print(f"PROSPECT  {who['name']} - {who['address']}")
    print(f"          {who['order_quantity_cases']} cases, preferred slot: "
          f"{who['preferred_day'] or 'any'} {who['preferred_window'] or ''}".rstrip())

    if not outcome.ok:
        print(f"  FAILED [{outcome.error_kind}] {outcome.error}")
        return

    decision = outcome.decision or {}
    print(f"  DECISION  {decision.get('decision')}")
    if decision.get("recommended_route_id"):
        print(
            f"  ROUTE     {decision['recommended_route_id']} - "
            f"{decision.get('recommended_route_name')} "
            f"[{decision.get('recommended_day')}] {decision.get('recommended_window')}"
        )
    if decision.get("review_reason"):
        print(f"  REVIEW    {decision['review_reason']}")
    if decision.get("grounded_fallback"):
        # Surfaced because it tells you WHY the reasoning looks thin: the model
        # was unavailable or its answer failed verification, so the deterministic
        # floor produced this decision.
        print(f"  NOTE      deterministic fallback: {decision.get('grounded_fallback_reason')}")

    if decision.get("decision_summary"):
        print(f"  SUMMARY   {decision['decision_summary']}")
    for reason in decision.get("primary_reasons") or []:
        print(f"    - {reason}")
    if decision.get("key_tradeoff"):
        print(f"  TRADE-OFF {decision['key_tradeoff']}")
    if decision.get("runner_up"):
        print(f"  RUNNER-UP {decision['runner_up']}")

    print("  CANDIDATES")
    for cand in outcome.candidates:
        status = "FEASIBLE  " if cand["feasible"] else "infeasible"
        score = f"{cand['total_score']:.2f}" if cand["feasible"] else " n/a"
        print(
            f"    {status} {cand['route_id']:>10} {cand['route_name'][:22]:24} "
            f"[{cand['day']}] {cand['distance_miles']:5.1f} mi  "
            f"util {cand['utilization_after']:.0%}  {cand['chosen_window']:>12}  score {score}"
        )
        for outcome_row in cand["constraints"]:
            if not outcome_row["passed"]:
                print(f"        FAIL {outcome_row['name']}: {outcome_row['detail']}")


def _write_html(outcomes: Iterable[service.AssignmentOutcome], directory: pathlib.Path) -> None:
    """Write each successful decision's SC-facing panel, so Mode 3's output can be
    inspected in a browser without standing up a service. Reuses the same
    `build_workflow_payload` the demo app and the published page use, so what you
    see here cannot drift from what either of those would render."""
    from smart_assignment.reporting.page import build_workflow_payload

    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, outcome in enumerate(outcomes, start=1):
        if not outcome.ok or outcome.result is None:
            continue
        payload = build_workflow_payload(outcome.result, DEFAULT_CONFIG)
        target = directory / f"prospect_{index:02d}.html"
        target.write_text(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{payload.get('name', 'Prospect')}</title>"
            "<body style=\"margin:0;padding:24px;background:#f3f5f9;"
            "font-family:system-ui,sans-serif\">"
            f"{payload.get('frontendHtml', '')}</body>",
            encoding="utf-8",
        )
        written += 1
    print(f"\nWrote {written} SC panel(s) to {directory}")


def _resolve_prospects(args: argparse.Namespace) -> Optional[list[CustomerProfile]]:
    if args.samples:
        return list(SAMPLE_CUSTOMERS)
    if args.file:
        path = pathlib.Path(args.file)
        if not path.exists():
            raise SystemExit(f"No such file: {path}")
        return _load_jsonl(path)
    if not args.address or args.cases is None:
        return None
    return [_profile_from_args(args)]


def _use_utf8_stdout() -> None:
    """Render the decision text faithfully on a legacy console.

    The narrative `routeslot` composes contains typographic characters (an em-dash
    between the route and its day, a middle dot between fields). A Windows console
    defaults to cp1252, which turns those into mangled bytes -- and a line carrying
    one can be dropped entirely when piped. Reconfiguring this process's stdout is
    a local fix that leaves the decision text itself untouched; it is best-effort,
    because stdout may already be a stream that cannot be reconfigured.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        pass


def main() -> None:
    _use_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Decide a delivery route and slot for one prospect, or a batch.",
    )
    source = parser.add_argument_group("prospect source (pick one)")
    source.add_argument("--address", help="street address, city, state, ZIP")
    source.add_argument("--cases", type=int, help="order quantity, in cases")
    source.add_argument("--file", help="JSONL file, one prospect record per line")
    source.add_argument(
        "--samples", action="store_true", help="use the bundled mock prospects"
    )

    optional = parser.add_argument_group("optional prospect details")
    optional.add_argument("--name", help="business/contact name")
    optional.add_argument("--customer-number", dest="customer_number")
    optional.add_argument("--day", help="preferred day: MON/TUE/WED/THU/FRI/SAT")
    optional.add_argument("--from", dest="from", help='preferred window start, "HH:MM"')
    optional.add_argument("--to", help='preferred window end, "HH:MM"')

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true", help="emit JSON instead of a summary")
    output.add_argument("--out", help="write JSONL results to this file")
    output.add_argument("--html", help="write each SC-facing panel into this directory")

    args = parser.parse_args()

    try:
        prospects = _resolve_prospects(args)
    except ValueError as exc:  # a malformed --address/--day/... combination
        raise SystemExit(f"Invalid prospect: {exc}") from None

    if prospects is None:
        parser.error("give --address and --cases, or --file, or --samples")
    if not prospects:
        raise SystemExit("No usable prospect records found.")

    outcomes = service.assign_many(prospects)

    if args.json:
        print(json.dumps([o.to_dict() for o in outcomes], indent=2))
    else:
        for outcome in outcomes:
            _print_outcome(outcome)
        print(_RULE)
        decided = sum(1 for o in outcomes if o.ok)
        review = sum(1 for o in outcomes if o.requires_human_review)
        print(f"{len(outcomes)} prospect(s): {decided} decided, "
              f"{len(outcomes) - decided} failed, {review} need human review")

    if args.out:
        target = pathlib.Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for outcome in outcomes:
                handle.write(json.dumps(outcome.to_dict()) + "\n")
        print(f"\nWrote {len(outcomes)} result(s) to {target}")

    if args.html:
        _write_html(outcomes, pathlib.Path(args.html))


if __name__ == "__main__":
    main()
