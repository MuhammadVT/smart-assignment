"""Hermetic tests for the eval dataset lock (``eval/dataset.py``): declaration,
pinning, and provenance. No backend -- these only exercise env resolution and
the deterministic content hash of the committed ``mock`` fixtures.
"""

from __future__ import annotations

import pytest

from eval import dataset as ds


def test_default_dataset_is_mock(monkeypatch):
    monkeypatch.delenv(ds.EVAL_DATASET_ENV, raising=False)
    resolved = ds.resolve_eval_dataset()
    assert resolved.name == "mock"
    assert resolved.data_source == "mock"
    assert resolved.geocoder == "mock"
    assert resolved.kind == "code"


def test_unknown_dataset_raises(monkeypatch):
    monkeypatch.setenv(ds.EVAL_DATASET_ENV, "nope")
    with pytest.raises(ValueError, match="not a known eval dataset"):
        ds.resolve_eval_dataset()


def test_apply_pins_source_geocoder_and_strict(monkeypatch):
    monkeypatch.delenv(ds.EVAL_DATASET_ENV, raising=False)
    # Pre-set the target vars so monkeypatch restores them at teardown even though
    # apply_eval_dataset writes os.environ directly.
    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE", "cache")
    monkeypatch.setenv("SMART_ASSIGNMENT_GEOCODER", "census")
    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE_STRICT", "")

    applied = ds.apply_eval_dataset()

    assert applied.name == "mock"
    import os

    assert os.environ["SMART_ASSIGNMENT_DATA_SOURCE"] == "mock"
    assert os.environ["SMART_ASSIGNMENT_GEOCODER"] == "mock"
    assert os.environ["SMART_ASSIGNMENT_DATA_SOURCE_STRICT"] == "1"


def test_content_ref_is_deterministic():
    mock = ds.resolve_eval_dataset()
    ref1 = ds.dataset_content_ref(mock)
    ref2 = ds.dataset_content_ref(mock)
    assert ref1 == ref2  # stable across calls -- a real identity, not a nonce
    assert ref1.startswith("sha256:")


def test_run_provenance_shape(monkeypatch):
    monkeypatch.delenv(ds.EVAL_DATASET_ENV, raising=False)
    mock = ds.resolve_eval_dataset()
    provenance = ds.run_provenance(mock)

    assert provenance["dataset"]["name"] == "mock"
    assert provenance["dataset"]["kind"] == "code"
    assert provenance["dataset"]["ref"].startswith("sha256:")
    assert provenance["backend"]  # resolved from Config, non-empty
    assert provenance["model"]
