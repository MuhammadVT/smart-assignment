"""Enforce that the eval dataset stays LOCKED and every golden case is captured
against it -- the harness side of "an eval result is reproducible from declared,
versioned inputs" (see ``eval/dataset.py`` and ``eval/README.md``).

These are hermetic (no backend): they only inspect the committed
``captured_responses.json`` against ``golden_cases.py``, so they run in the
always-required ``tests/`` suite.

Adoption transition: once the committed captures carry the provenance schema
(a ``captured_with`` block, written by ``eval/capture.py``), the coverage and
identity checks below are HARD gates -- every golden case must have a capture,
all captures must agree on the declared dataset identity, and that identity must
be the declared default. Until the committed file adopts provenance (which needs
a credentialed ``eval.capture`` run, done out-of-band), those checks *skip* with
an actionable message rather than fail -- so the mechanism ships green and starts
biting the moment provenance lands. ``test_no_orphan_captures`` is always on.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from eval.dataset import DEFAULT_EVAL_DATASET
from eval.golden_cases import GOLDEN_CASES

_CAPTURED_PATH = (
    pathlib.Path(__file__).parents[2] / "eval" / "data" / "captured_responses.json"
)

_SKIP_UNADOPTED = (
    "captured_responses.json predates the dataset-provenance schema; run "
    "`python3 -m eval.capture` (needs a backend) to lock captures onto the declared "
    "dataset -- see eval/README.md. This becomes a hard gate once provenance lands."
)


def _load_raw() -> dict:
    if not _CAPTURED_PATH.exists():
        return {}
    return json.loads(_CAPTURED_PATH.read_text(encoding="utf-8"))


def _has_provenance(entry: object) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("captured_with"), dict)


def _provenance_adopted(raw: dict) -> bool:
    """True once any committed capture carries provenance -- the signal that the
    schema is in force and the coverage/identity gates should enforce, not skip."""
    return any(_has_provenance(entry) for entry in raw.values())


def test_no_orphan_captures():
    # Always on: every captured id must be a real golden case, so a renamed or
    # removed case can't leave a stale, silently-scored response behind.
    raw = _load_raw()
    golden_ids = {case.eval_id for case in GOLDEN_CASES}
    orphans = sorted(set(raw) - golden_ids)
    assert not orphans, f"captured_responses.json has entries for unknown eval_ids: {orphans}"


def test_every_golden_case_is_captured():
    raw = _load_raw()
    if not _provenance_adopted(raw):
        pytest.skip(_SKIP_UNADOPTED)
    missing = sorted({case.eval_id for case in GOLDEN_CASES} - set(raw))
    assert not missing, (
        f"these golden cases have no captured response: {missing}. "
        "Run `python3 -m eval.capture` for all cases (no SMART_ASSIGNMENT_EVAL_IDS filter) "
        "and commit both eval/data/ files."
    )


def test_captures_agree_on_the_declared_dataset():
    raw = _load_raw()
    if not _provenance_adopted(raw):
        pytest.skip(_SKIP_UNADOPTED)

    names: set = set()
    refs: set = set()
    for eval_id, entry in raw.items():
        assert _has_provenance(entry), (
            f"{eval_id} has no captured_with provenance; re-run `python3 -m eval.capture` "
            "so every committed capture records the dataset it was made against."
        )
        dataset = entry["captured_with"]["dataset"]
        names.add(dataset["name"])
        refs.add(dataset["ref"])

    assert names == {DEFAULT_EVAL_DATASET}, (
        f"captures were made against unexpected dataset(s) {sorted(names)}; the declared "
        f"default is {DEFAULT_EVAL_DATASET!r}. Re-capture against the declared dataset."
    )
    assert len(refs) == 1, (
        f"captures disagree on dataset content ref {sorted(refs)} -- they were produced "
        "against different versions of the dataset. Re-capture all cases in one run."
    )


def test_provenance_records_backend_and_model():
    raw = _load_raw()
    if not _provenance_adopted(raw):
        pytest.skip(_SKIP_UNADOPTED)
    for eval_id, entry in raw.items():
        captured_with = entry["captured_with"]
        assert captured_with.get("backend"), f"{eval_id} provenance is missing 'backend'"
        assert captured_with.get("model"), f"{eval_id} provenance is missing 'model'"
