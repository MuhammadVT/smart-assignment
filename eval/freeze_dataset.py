"""
Freeze curated candidate cases into a self-contained, anonymized snapshot dataset.

This is the authoring on-ramp for the *human-feedback* path: take the candidate
cases produced by ``scripts/curate_feedback.py`` / ``scripts/phoenix_curate.py``,
run each once against your real data source to capture the *world* it needs (the
candidate routes with capacity/committed-stops/windows/tiers, and the prospect's
geocoded coordinates), **anonymize** it, and write a committed bundle that then
replays fully offline (see integrations/snapshot_data.py + eval/dataset.py).

**Anonymization (the PII line).** Scoring depends on *geometry and capacity*, not
identities. So the freeze keeps the numbers -- coordinates, capacity, windows,
tiers, route codes -- and drops the identifiers: the prospect's name and street
address become synthetic labels (the label keys the geocode map to the real
coordinates, so distance math is unchanged), and committed-stop customer numbers
become ``STOP-*`` placeholders. The result is PII-free *by construction*, like
the ``mock`` world, and safe to commit and run in a shared CI. (Committed-stop
coordinates are kept as bare geometry -- optional jitter is a future hardening.)

**One shared world.** Like ``mock`` (a handful of routes, many prospects), a
snapshot dataset is ONE world: the deduplicated union of every case's candidate
routes, with each prospect evaluated against it. The golden ``expected_outcome`` /
``expected_route_id`` / ``expected_window`` are set from the human's target when
one was given (a promoted thumbs-down), else captured from the decision at freeze
time (a confirming/baseline case). Because coordinates are preserved exactly, the
baseline captured against the real world matches the frozen world.

Run (with your real data source configured -- e.g. cache/census):
    python3 -m eval.freeze_dataset --cases eval/data/feedback_candidates.json \\
        --name feedback_2026_07
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smart_assignment.shared.config import ROLE_ROOT_AGENT, Config
from smart_assignment.shared.models import (
    CustomerProfile,
    Decision,
    GeoPoint,
    Route,
)

logger = logging.getLogger(__name__)


def _window_str(window) -> Optional[str]:
    if not window:
        return None
    return f"{window[0].strftime('%H:%M')}-{window[1].strftime('%H:%M')}"


def _outcome(decision: Decision) -> str:
    return "recommend" if decision == Decision.RECOMMENDED else "escalate"


def _anon_address(eval_id: str, index: int) -> str:
    """A synthetic, obviously-not-real address label for a prospect. It keys the
    geocode map to the real coordinates, so it must be stable and non-empty."""
    return f"{100 + index} Snapshot Way, Anytown (anonymized: {eval_id})"


def _anonymize_route(route: Route) -> Route:
    """A copy of ``route`` with committed-stop customer identifiers replaced by
    ``STOP-*`` placeholders. Geometry, capacity, windows, and tiers are kept
    (they drive the scoring); route id/name are operational codes, not customer PII."""
    stops = [
        replace(stop, customer_number=f"STOP-{route.route_id}-{i:03d}")
        for i, stop in enumerate(route.committed_stops)
    ]
    return replace(route, committed_stops=stops)


def _expected_fields(
    candidate: Dict[str, Any], result: Any
) -> Tuple[str, Optional[str], Optional[str]]:
    """The golden target for a case: the human's stated target when a promoted
    thumbs-down provided one, else the decision captured at freeze time."""
    rec = result.recommendation
    suggested = candidate.get("suggested_expected_outcome")
    if suggested:
        # Human-set target (route/slot may be filled in during review, else null).
        return (
            str(suggested),
            candidate.get("expected_route_id"),
            candidate.get("expected_window"),
        )
    return (
        _outcome(rec.decision),
        rec.recommended_route_id,
        rec.recommended_window,
    )


def build_bundle(
    candidates: List[Dict[str, Any]],
    *,
    config: Config,
    geocoder: Any = None,
    source_label: str = "feedback",
    dataset_name: str = "snapshot",
    created_at: Optional[str] = None,
) -> Tuple[List[Route], Dict[str, GeoPoint], List[Dict[str, Any]], Dict[str, Any]]:
    """Run each candidate once to capture its world, anonymize, and assemble the
    four bundle parts ``(routes, geocode, cases, manifest)``.

    ``geocoder`` (a ``Geocoder``) and ``config`` select the *authoring* world
    (e.g. census + cache). Candidates that can't be reconstructed or geocoded are
    skipped (logged), so a partial batch still yields a valid bundle."""
    from eval.case_source import SkippedCase, candidate_to_case
    from smart_assignment.pipeline import run_slot_recommendation
    from smart_assignment.reasoning import DeterministicReasoner

    world: Dict[Tuple[str, str], Route] = {}
    geocode: Dict[str, GeoPoint] = {}
    cases: List[Dict[str, Any]] = []
    skipped = 0

    for index, candidate in enumerate(candidates):
        try:
            case = candidate_to_case(candidate)
        except SkippedCase as exc:
            logger.warning("Skipping %s: %s", candidate.get("eval_id"), exc)
            skipped += 1
            continue

        customer = case.customer
        run_kwargs: Dict[str, Any] = {"config": config, "reasoner": DeterministicReasoner()}
        if geocoder is not None:
            run_kwargs["geocoder"] = geocoder
        try:
            result = run_slot_recommendation(customer, **run_kwargs)
        except Exception as exc:  # noqa: BLE001 - a bad case shouldn't abort the whole freeze
            logger.warning("Skipping %s: pipeline failed (%s)", case.eval_id, exc)
            skipped += 1
            continue

        coords = getattr(result.customer, "location", None)
        if coords is None:
            logger.warning("Skipping %s: no geocode result", case.eval_id)
            skipped += 1
            continue

        # Capture this case's candidate routes into the shared world (dedup by id+day).
        for evaluation in result.candidates_considered:
            route = evaluation.route
            world.setdefault((route.route_id, route.day.value), _anonymize_route(route))

        anon_address = _anon_address(case.eval_id, index)
        geocode[anon_address] = coords
        expected_outcome, expected_route_id, expected_window = _expected_fields(candidate, result)

        slot = customer.preferred_slot
        cases.append(
            {
                "eval_id": case.eval_id,
                "query": _reanonymized_query(customer, anon_address),
                "context": {
                    "name": f"Prospect {index + 1}",
                    "address": anon_address,
                    "order_quantity_cases": customer.order_quantity_cases,
                    **(
                        {
                            "preferred_day": slot.day.name,
                            "preferred_window": _window_str(slot.window),
                        }
                        if slot is not None
                        else {}
                    ),
                },
                "expected_outcome": expected_outcome,
                "expected_route_id": expected_route_id,
                "expected_window": expected_window,
                "note": candidate.get("note"),
                "provenance": {
                    "source": source_label,
                    "original": candidate.get("provenance") or {},
                },
            }
        )

    manifest = {
        "name": dataset_name,
        "source": source_label,
        "anonymized": True,
        "case_count": len(cases),
        "skipped_count": skipped,
        "route_count": len(world),
        "created_at": created_at,
        "captured_with": {
            "backend": config.llm_backend,
            "model": config.resolved_model(ROLE_ROOT_AGENT),
            "total_score_threshold": config.total_score_threshold,
            "use_route_slot_scoring": config.use_route_slot_scoring,
        },
    }
    return list(world.values()), geocode, cases, manifest


def _reanonymized_query(customer: CustomerProfile, anon_address: str) -> str:
    """An intake message that uses the anonymized address (never the real one)."""
    parts: List[str] = [f"Prospect at {anon_address}", f"{customer.order_quantity_cases} cases"]
    slot = customer.preferred_slot
    if slot is not None:
        parts.append(f"prefers {slot.day.name} {_window_str(slot.window)}")
    return ", ".join(parts)


def freeze_to_dir(
    candidates: List[Dict[str, Any]],
    out_dir: str,
    *,
    config: Config,
    geocoder: Any = None,
    source_label: str = "feedback",
    dataset_name: str = "snapshot",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and write a self-contained bundle to ``out_dir``. Returns the manifest."""
    from smart_assignment.integrations.snapshot_data import write_bundle

    routes, geocode, cases, manifest = build_bundle(
        candidates,
        config=config,
        geocoder=geocoder,
        source_label=source_label,
        dataset_name=dataset_name,
        created_at=created_at,
    )
    write_bundle(out_dir, routes=routes, geocode=geocode, cases=cases, manifest=manifest)
    return manifest


def main() -> None:
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Curated candidate-cases JSON to freeze.")
    parser.add_argument("--name", required=True, help="Dataset name (the bundle directory name).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (defaults to eval/data/snapshots/<name>).",
    )
    parser.add_argument("--source", default="feedback", help="Provenance source label.")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as handle:
        candidates = json.load(handle)

    out_dir = args.out or str(Path("eval") / "data" / "snapshots" / args.name)
    manifest = freeze_to_dir(
        candidates,
        out_dir,
        config=Config.from_env(),
        source_label=args.source,
        dataset_name=args.name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    print(
        f"Froze {manifest['case_count']} case(s) ({manifest['skipped_count']} skipped), "
        f"{manifest['route_count']} route(s) -> {out_dir}"
    )
    print(f"Run it:  SMART_ASSIGNMENT_EVAL_DATASET={args.name} python3 -m pytest eval/ -q")


if __name__ == "__main__":
    main()
