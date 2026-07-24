"""Tests for snapshot-dataset discovery, pinning, and content hashing in
eval/dataset.py."""

from __future__ import annotations

import json

import pytest

from eval import dataset as ds
from smart_assignment.integrations import snapshot_data
from smart_assignment.shared.models import GeoPoint


def _make_bundle(root, name, *, routes=None, geocode=None, cases=None, manifest=None):
    bundle = root / name
    snapshot_data.write_bundle(
        str(bundle),
        routes=routes or [],
        geocode=geocode or {"1 Main St": GeoPoint(29.7, -95.3)},
        cases=cases if cases is not None else [{"eval_id": f"{name}_c1"}],
        manifest=manifest or {"source": "test", "count": 1},
    )
    return bundle


@pytest.fixture
def snapshots_root(tmp_path, monkeypatch):
    root = tmp_path / "snapshots"
    root.mkdir()
    monkeypatch.setenv(ds._SNAPSHOTS_ROOT_ENV, str(root))
    return root


def test_discovery_and_all_datasets(snapshots_root, monkeypatch):
    _make_bundle(snapshots_root, "feedback_2026_07")
    # A directory without a manifest is NOT a dataset.
    (snapshots_root / "not_a_dataset").mkdir()
    datasets = ds.all_datasets()
    assert "mock" in datasets  # built-in always present
    assert "feedback_2026_07" in datasets
    assert "not_a_dataset" not in datasets
    assert datasets["feedback_2026_07"].kind == "snapshot"


def test_resolve_and_apply_pins_snapshot(snapshots_root, monkeypatch):
    bundle = _make_bundle(snapshots_root, "synthetic_v1")
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "synthetic_v1")
    dataset = ds.apply_eval_dataset()
    assert dataset.kind == "snapshot"
    import os

    assert os.environ["SMART_ASSIGNMENT_DATA_SOURCE"] == "snapshot"
    assert os.environ["SMART_ASSIGNMENT_GEOCODER"] == "snapshot"
    assert os.environ[snapshot_data.SNAPSHOT_DIR_ENV] == str(bundle)
    assert os.environ["SMART_ASSIGNMENT_DATA_SOURCE_STRICT"] == "1"


def test_apply_mock_clears_snapshot_dir(snapshots_root, monkeypatch):
    monkeypatch.setenv(snapshot_data.SNAPSHOT_DIR_ENV, "/stale/dir")
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "mock")
    ds.apply_eval_dataset()
    import os

    assert snapshot_data.SNAPSHOT_DIR_ENV not in os.environ


def test_unknown_dataset_lists_discovered(snapshots_root, monkeypatch):
    _make_bundle(snapshots_root, "known_snap")
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "does_not_exist")
    with pytest.raises(ValueError, match="known_snap"):
        ds.resolve_eval_dataset()


def test_content_ref_hashes_bundle_and_moves_on_change(snapshots_root, monkeypatch):
    bundle = _make_bundle(snapshots_root, "snap")
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "snap")
    dataset = ds.resolve_eval_dataset()
    ref1 = ds.dataset_content_ref(dataset)
    assert ref1.startswith("sha256:")

    # A manifest-only edit does NOT change the data ref.
    (bundle / "manifest.json").write_text(json.dumps({"source": "changed"}), encoding="utf-8")
    assert ds.dataset_content_ref(dataset) == ref1

    # A cases edit DOES.
    (bundle / "cases.json").write_text(json.dumps([{"eval_id": "new"}]), encoding="utf-8")
    assert ds.dataset_content_ref(dataset) != ref1
