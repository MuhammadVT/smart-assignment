"""
Generate a self-contained *synthetic* snapshot dataset -- the second on-ramp
(alongside the human-feedback freeze path) for curating golden datasets.

A synthetic dataset is authored, not captured: a small designed world (routes
with capacity/committed-stops/windows/tiers) plus a handful of prospects placed to
exercise a mix of outcomes (clean recommend, over-capacity escalate, out-of-range
escalate, preferred-window match). It is PII-free by construction -- every name,
address, and coordinate is invented -- so it needs no anonymization and is safe to
commit and run in CI.

It reuses the same bundle format and the same replay substrate as the freeze path
(``integrations/snapshot_data.write_bundle``), so a synthetic dataset and a
feedback-derived one are indistinguishable to the eval runner: one loader, one
scorer, one CI. The golden ``expected_*`` fields are captured by running the
current pipeline against the designed world (injected directly via
``run_slot_recommendation(routes=..., geocoder=...)``, so no round-trip), locking
today's behavior as the regression baseline.

Run:
    python3 -m eval.synthetic --name synthetic_v1
"""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smart_assignment.shared.config import ROLE_ROOT_AGENT, Config
from smart_assignment.shared.geo import AddressNotFoundError
from smart_assignment.shared.models import (
    CustomerProfile,
    Decision,
    DayOfWeek,
    GeoPoint,
    PreferredSlot,
    Route,
    RouteStop,
)

# A synthetic metro centered on a deliberately fictional coordinate ("Anytown").
_CENTER = GeoPoint(40.00, -100.00)


class _DictGeocoder:
    """A trivial geocoder backed by an in-memory {address: GeoPoint} map -- used
    only to author the synthetic world (never shipped in the bundle)."""

    def __init__(self, mapping: Dict[str, GeoPoint]) -> None:
        self._map = mapping

    def geocode(self, address: str) -> GeoPoint:
        point = self._map.get(address)
        if point is None:
            raise AddressNotFoundError(address, "synthetic geocoder has no such address")
        return point


def _stop(number: str, lat: float, lon: float, window: Tuple[time, time], tier: str) -> RouteStop:
    return RouteStop(
        customer_number=number,
        location=GeoPoint(lat, lon),
        delivery_time_window=window,
        customer_tier=tier,
    )


def synthetic_routes() -> List[Route]:
    """A small, designed, PII-free route world (stable -- no randomness)."""
    morning = (time(8, 0), time(11, 0))
    afternoon = (time(13, 0), time(16, 0))
    return [
        Route(
            route_id="SYN-100",
            name="North Loop",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(40.05, -100.00),
            service_radius_miles=20.0,
            vehicle_capacity_cases=1000.0,
            avg_load_cases=500.0,
            available_windows=[morning],
            committed_stops=[
                _stop("STOP-100-000", 40.045, -100.005, morning, "5"),
                _stop("STOP-100-001", 40.052, -99.995, morning, "4"),
                _stop("STOP-100-002", 40.048, -100.010, morning, "Other"),
            ],
        ),
        Route(
            route_id="SYN-200",
            name="South Loop",
            day=DayOfWeek.WED,
            service_center=GeoPoint(39.95, -100.00),
            service_radius_miles=20.0,
            vehicle_capacity_cases=800.0,
            avg_load_cases=600.0,
            available_windows=[afternoon],
            committed_stops=[
                _stop("STOP-200-000", 39.955, -100.004, afternoon, "5"),
                _stop("STOP-200-001", 39.948, -99.996, afternoon, "Perks"),
            ],
        ),
        Route(
            route_id="SYN-300",
            name="East Loop",
            day=DayOfWeek.THU,
            service_center=GeoPoint(40.00, -99.90),
            service_radius_miles=15.0,
            vehicle_capacity_cases=1200.0,
            avg_load_cases=400.0,
            available_windows=[morning],
            committed_stops=[
                _stop("STOP-300-000", 40.005, -99.905, morning, "4"),
                _stop("STOP-300-001", 39.995, -99.895, morning, "Other"),
            ],
        ),
    ]


def _prospect(
    eval_id: str,
    lat: float,
    lon: float,
    cases: int,
    slot: Optional[PreferredSlot],
    note: str,
) -> Tuple[str, GeoPoint, CustomerProfile, str]:
    address = f"{eval_id} address, Anytown (synthetic)"
    customer = CustomerProfile(
        name=f"Synthetic {eval_id}",
        address=address,
        order_quantity_cases=cases,
        preferred_slot=slot,
    )
    return eval_id, GeoPoint(lat, lon), customer, note


def synthetic_prospects() -> List[Tuple[str, GeoPoint, CustomerProfile, str]]:
    """Prospects placed to exercise a mix of outcomes."""
    morning = (time(8, 0), time(11, 0))
    return [
        _prospect("syn_clean_recommend", 40.045, -100.008, 100,
                  PreferredSlot(DayOfWeek.TUE, morning),
                  "Near North Loop, small order, window match."),
        _prospect("syn_over_capacity", 39.955, -100.002, 900,
                  None, "Near South Loop but a 900-case order overflows the truck."),
        _prospect("syn_out_of_range", 41.60, -100.00, 120,
                  None, "~110 miles from every route -> out of service range."),
        _prospect("syn_east_recommend", 40.005, -99.902, 150,
                  PreferredSlot(DayOfWeek.THU, morning), "Near East Loop, comfortable headroom."),
    ]


def _window_str(window) -> Optional[str]:
    if not window:
        return None
    return f"{window[0].strftime('%H:%M')}-{window[1].strftime('%H:%M')}"


def build_synthetic_bundle(
    config: Config,
    *,
    dataset_name: str = "synthetic_v1",
    created_at: Optional[str] = None,
) -> Tuple[List[Route], Dict[str, GeoPoint], List[Dict[str, Any]], Dict[str, Any]]:
    """Assemble the four bundle parts for the designed world. Expected fields are
    captured by running the current pipeline against the injected world."""
    from smart_assignment.pipeline import run_slot_recommendation
    from smart_assignment.reasoning import DeterministicReasoner

    routes = synthetic_routes()
    prospects = synthetic_prospects()
    geocode = {profile.address: coords for _id, coords, profile, _note in prospects}
    geocoder = _DictGeocoder(geocode)

    cases: List[Dict[str, Any]] = []
    for eval_id, _coords, profile, note in prospects:
        result = run_slot_recommendation(
            profile,
            routes=routes,
            geocoder=geocoder,
            config=config,
            reasoner=DeterministicReasoner(),
        )
        rec = result.recommendation
        outcome = "recommend" if rec.decision == Decision.RECOMMENDED else "escalate"
        slot = profile.preferred_slot
        cases.append(
            {
                "eval_id": eval_id,
                "query": f"{profile.name} at {profile.address}, "
                f"{profile.order_quantity_cases} cases"
                + (f", prefers {slot.day.name} {_window_str(slot.window)}" if slot else ""),
                "context": {
                    "name": profile.name,
                    "address": profile.address,
                    "order_quantity_cases": profile.order_quantity_cases,
                    **(
                        {
                            "preferred_day": slot.day.name,
                            "preferred_window": _window_str(slot.window),
                        }
                        if slot is not None
                        else {}
                    ),
                },
                "expected_outcome": outcome,
                "expected_route_id": rec.recommended_route_id if outcome == "recommend" else None,
                "expected_window": rec.recommended_window if outcome == "recommend" else None,
                "note": note,
                "provenance": {"source": "synthetic"},
            }
        )

    manifest = {
        "name": dataset_name,
        "source": "synthetic",
        "anonymized": True,  # synthetic == PII-free by construction
        "case_count": len(cases),
        "route_count": len(routes),
        "created_at": created_at,
        "captured_with": {
            "backend": config.llm_backend,
            "model": config.resolved_model(ROLE_ROOT_AGENT),
            "total_score_threshold": config.total_score_threshold,
            "use_route_slot_scoring": config.use_route_slot_scoring,
        },
    }
    return routes, geocode, cases, manifest


def generate_to_dir(
    out_dir: str,
    *,
    config: Optional[Config] = None,
    dataset_name: str = "synthetic_v1",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and write the synthetic bundle to ``out_dir``. Returns the manifest."""
    from smart_assignment.integrations.snapshot_data import write_bundle

    config = config or Config()
    routes, geocode, cases, manifest = build_synthetic_bundle(
        config, dataset_name=dataset_name, created_at=created_at
    )
    write_bundle(out_dir, routes=routes, geocode=geocode, cases=cases, manifest=manifest)
    return manifest


def main() -> None:
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="synthetic_v1", help="Dataset name (bundle directory).")
    parser.add_argument(
        "--out", default=None, help="Output dir (default eval/data/snapshots/<name>)."
    )
    args = parser.parse_args()

    out_dir = args.out or str(Path("eval") / "data" / "snapshots" / args.name)
    manifest = generate_to_dir(
        out_dir,
        dataset_name=args.name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    print(
        f"Generated synthetic dataset '{args.name}': {manifest['case_count']} case(s), "
        f"{manifest['route_count']} route(s) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
