"""Tests for outcome/route-slot scoring against snapshot datasets, including the
self-contained gate over the committed datasets."""

from __future__ import annotations

import json

from eval import dataset as ds
from eval.outcome_scoring import (
    PATH_DETERMINISTIC,
    score_all_snapshots,
    score_dataset,
)
from eval.synthetic import generate_to_dir
from smart_assignment.shared.config import Config


def test_committed_snapshots_reproduce_golden_deterministically():
    # The self-contained CI gate: the current deterministic model reproduces the
    # golden outcome AND route-slot for every committed snapshot dataset.
    scores = score_all_snapshots(path=PATH_DETERMINISTIC)
    assert scores, "expected at least one committed snapshot dataset (synthetic_v1)"
    for score in scores:
        assert score.all_pass(), score.summary() + " :: " + str(
            [(f.eval_id, f.got_outcome, f.got_route_id) for f in score.failures()]
        )
        # A recommend case must match the full route-slot, not just the outcome.
        assert score.route_slot_total >= 1
        assert score.route_slot_pass == score.route_slot_total


def test_side_effect_free_env(monkeypatch):
    import os

    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE", "mock")
    before = os.environ.get("SMART_ASSIGNMENT_DATA_SOURCE")
    dataset = next(d for d in ds.all_datasets().values() if d.kind == "snapshot")
    score_dataset(dataset, path=PATH_DETERMINISTIC)
    # The pinned snapshot env was restored -- no leak into the rest of the process.
    assert os.environ.get("SMART_ASSIGNMENT_DATA_SOURCE") == before


def test_detects_a_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "snapshots"
    bundle = root / "tampered"
    generate_to_dir(str(bundle), config=Config(), dataset_name="tampered")

    # Corrupt one case's golden outcome so scoring must flag it.
    cases_path = bundle / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    recommend_case = next(c for c in cases if c["expected_outcome"] == "recommend")
    recommend_case["expected_outcome"] = "escalate"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    monkeypatch.setenv(ds._SNAPSHOTS_ROOT_ENV, str(root))
    dataset = ds.all_datasets()["tampered"]
    score = score_dataset(dataset, path=PATH_DETERMINISTIC)
    assert not score.all_pass()
    assert any(not c.outcome_match for c in score.cases)
