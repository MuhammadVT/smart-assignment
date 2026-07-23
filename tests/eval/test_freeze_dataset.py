"""End-to-end test for the freeze authoring path: freeze the mock world into a
self-contained bundle, then replay it and confirm the decisions reproduce and the
bundle is anonymized (PII-free)."""

from __future__ import annotations

from eval import dataset as ds
from eval.case_source import candidate_to_case
from eval.freeze_dataset import _outcome, freeze_to_dir
from smart_assignment.integrations import route_capacity_client as rcc
from smart_assignment.integrations import snapshot_data
from smart_assignment.integrations.geocoding_client import (
    MockGeocoder,
    SnapshotGeocoder,
    resolve_geocoder,
)
from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import Config


def _candidate(index, customer):
    ctx = {
        "name": customer.name,
        "address": customer.address,
        "order_quantity_cases": customer.order_quantity_cases,
    }
    slot = customer.preferred_slot
    if slot is not None:
        ctx["preferred_day"] = slot.day.name
        ctx["preferred_window"] = (
            f"{slot.window[0].strftime('%H:%M')}-{slot.window[1].strftime('%H:%M')}"
        )
    return {"eval_id": f"case_{index}", "context": ctx}


def test_freeze_mock_then_replay_reproduces_and_anonymizes(tmp_path, monkeypatch):
    config = Config()

    # 1) Baseline decisions against the mock world, and the candidate inputs.
    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE", "mock")
    rcc.clear_route_cache()
    baseline = {}
    candidates = []
    for index, customer in enumerate(SAMPLE_CUSTOMERS):
        result = run_slot_recommendation(
            customer, config=config, reasoner=DeterministicReasoner(), geocoder=MockGeocoder()
        )
        baseline[f"case_{index}"] = _outcome(result.recommendation.decision)
        candidates.append(_candidate(index, customer))

    # 2) Freeze into a bundle under a tmp snapshots root.
    root = tmp_path / "snapshots"
    bundle_dir = root / "roundtrip"
    manifest = freeze_to_dir(
        candidates,
        str(bundle_dir),
        config=config,
        geocoder=MockGeocoder(),
        source_label="test",
        dataset_name="roundtrip",
    )
    assert manifest["case_count"] == len(SAMPLE_CUSTOMERS)
    assert manifest["anonymized"] is True
    assert manifest["route_count"] > 0

    # 3) Anonymization: no real customer name or street address leaked into the bundle.
    blob = (
        (bundle_dir / "cases.json").read_text(encoding="utf-8")
        + (bundle_dir / "geocode.json").read_text(encoding="utf-8")
        + (bundle_dir / "routes.json").read_text(encoding="utf-8")
    )
    for customer in SAMPLE_CUSTOMERS:
        assert customer.name not in blob
        assert customer.address not in blob

    # 4) Replay: pin the snapshot dataset and re-run each case fully offline.
    monkeypatch.setenv(ds._SNAPSHOTS_ROOT_ENV, str(root))
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "roundtrip")
    ds.apply_eval_dataset()
    rcc.clear_route_cache()
    assert isinstance(resolve_geocoder(), SnapshotGeocoder)  # snapshot geocoder is pinned

    cases = snapshot_data.load_cases(str(bundle_dir))
    assert len(cases) == len(SAMPLE_CUSTOMERS)
    for case_dict in cases:
        case = candidate_to_case(case_dict)
        result = run_slot_recommendation(
            case.customer, config=config, reasoner=DeterministicReasoner()
        )
        got = _outcome(result.recommendation.decision)
        # The replayed decision matches the frozen expected AND the mock baseline.
        assert got == case_dict["expected_outcome"] == baseline[case.eval_id]

    rcc.clear_route_cache()


def test_freeze_skips_redacted_cases(tmp_path):
    config = Config()
    candidates = [
        {"eval_id": "ok", "context": {"name": "N", "address": "5 Main St, Houston, TX",
                                      "order_quantity_cases": 10}},
        {"eval_id": "redacted", "context": {"name": "X", "address": "[redacted], Houston",
                                            "order_quantity_cases": 20}},
    ]
    manifest = freeze_to_dir(
        candidates, str(tmp_path / "b"), config=config, geocoder=MockGeocoder(),
        dataset_name="b", source_label="test",
    )
    assert manifest["case_count"] == 1
    assert manifest["skipped_count"] == 1
