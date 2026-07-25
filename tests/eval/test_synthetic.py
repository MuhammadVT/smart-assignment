"""Tests for the synthetic dataset generator and its round-trip through the
snapshot substrate."""

from __future__ import annotations

from eval import dataset as ds
from eval.case_source import candidate_to_case
from eval.freeze_dataset import _outcome
from eval.synthetic import build_synthetic_bundle, generate_to_dir
from smart_assignment.integrations import route_capacity_client as rcc
from smart_assignment.integrations import snapshot_data
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import Config


def test_synthetic_bundle_covers_recommend_and_escalate():
    _routes, _geocode, cases, manifest = build_synthetic_bundle(Config())
    by_id = {c["eval_id"]: c for c in cases}
    assert by_id["syn_clean_recommend"]["expected_outcome"] == "recommend"
    assert by_id["syn_clean_recommend"]["expected_route_id"] == "SYN-100"
    assert by_id["syn_over_capacity"]["expected_outcome"] == "escalate"
    assert by_id["syn_out_of_range"]["expected_outcome"] == "escalate"
    assert by_id["syn_east_recommend"]["expected_route_id"] == "SYN-300"
    assert manifest["source"] == "synthetic" and manifest["anonymized"] is True


def test_synthetic_roundtrip_reproduces(tmp_path, monkeypatch):
    config = Config()
    root = tmp_path / "snapshots"
    bundle = root / "synthetic_v1"
    generate_to_dir(str(bundle), config=config, dataset_name="synthetic_v1")

    monkeypatch.setenv(ds._SNAPSHOTS_ROOT_ENV, str(root))
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "synthetic_v1")
    ds.apply_eval_dataset()
    rcc.clear_route_cache()

    for case_dict in snapshot_data.load_cases(str(bundle)):
        case = candidate_to_case(case_dict)
        result = run_slot_recommendation(
            case.customer, config=config, reasoner=DeterministicReasoner()
        )
        assert _outcome(result.recommendation.decision) == case_dict["expected_outcome"]

    rcc.clear_route_cache()
