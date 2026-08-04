"""Hermetic tests for the durable judge-verdict log (``eval/judge_log.py``).

No LLM and no DeepEval: the metric and test case are duck-typed by the module
under test, so a tiny fake stands in for ``GEval``/``LLMTestCase``. That is the
point of the seam -- the recording path is exercised end to end without the
``eval-quality`` extra installed.
"""

from __future__ import annotations

import json

import pytest

from eval.judge_log import (
    JudgeVerdictRecord,
    append_verdict,
    content_ref,
    excerpt,
    measure_and_record,
    read_verdicts,
)


class _FakeMetric:
    """Stands in for a DeepEval ``GEval``: measured asynchronously, then read
    back off the object -- including the singleton behavior that makes snapshotting
    at measure time necessary."""

    def __init__(self, scores, threshold=0.5):
        self._scores = list(scores)
        self.threshold = threshold
        self.score = None
        self.reason = None

    async def a_measure(self, test_case):  # noqa: ARG002 - signature parity only
        self.score = self._scores.pop(0)
        self.reason = f"judged {self.score}"


class _FakeTestCase:
    def __init__(self, actual_output=""):
        self.actual_output = actual_output


def _record(**overrides) -> JudgeVerdictRecord:
    base = dict(
        eval_id="case_a",
        dimension="response_clarity",
        threshold=0.5,
        passed=True,
        decision_id="case_a",
        score=0.8,
        reason="clear",
    )
    base.update(overrides)
    return JudgeVerdictRecord(**base)


# --- the log itself -------------------------------------------------------


def test_append_and_read_round_trip(tmp_path):
    path = str(tmp_path / "verdicts.jsonl")
    assert append_verdict(_record(), path) is True
    assert append_verdict(_record(eval_id="case_b", score=0.2, passed=False), path) is True

    records = read_verdicts(path)
    assert [r.eval_id for r in records] == ["case_a", "case_b"]
    assert records[0].score == 0.8 and records[0].passed is True
    assert records[1].score == 0.2 and records[1].passed is False
    assert records[1].dimension == "response_clarity"


def test_append_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "deeper" / "verdicts.jsonl")
    assert append_verdict(_record(), path) is True
    assert len(read_verdicts(path)) == 1


def test_empty_path_disables_recording(tmp_path):
    """The path IS the switch -- an empty value records nothing and says so."""
    assert append_verdict(_record(), "") is False
    assert append_verdict(_record(), "   ") is False
    assert read_verdicts("") == []


def test_write_failure_is_swallowed(tmp_path):
    """A verdict that can't be persisted must never fail the eval that made it."""
    directory = tmp_path / "is_a_directory"
    directory.mkdir()
    assert append_verdict(_record(), str(directory)) is False


def test_missing_log_reads_as_empty(tmp_path):
    assert read_verdicts(str(tmp_path / "never_written.jsonl")) == []


def test_malformed_line_is_skipped_and_rest_still_reads(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    append_verdict(_record(), str(path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
        handle.write("\n")
    append_verdict(_record(eval_id="case_c"), str(path))

    records = read_verdicts(str(path))
    assert [r.eval_id for r in records] == ["case_a", "case_c"]


def test_record_is_json_serializable_with_sorted_keys(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    append_verdict(_record(judge={"model": "m"}, run={"backend": "standard"}), str(path))
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    assert raw["judge"] == {"model": "m"}
    assert raw["run"] == {"backend": "standard"}
    assert list(raw) == sorted(raw)


def test_legacy_row_tolerates_missing_fields(tmp_path):
    """A hand-edited or older row still reads, defaulting decision_id to eval_id."""
    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        json.dumps({"eval_id": "case_a", "dimension": "brief_quality"}) + "\n",
        encoding="utf-8",
    )
    record = read_verdicts(str(path))[0]
    assert record.decision_id == "case_a"
    assert record.score is None and record.passed is False and record.threshold == 0.0


def test_failure_line_matches_the_assertion_format():
    assert _record(score=0.4, passed=False, reason="jargon").failure_line() == (
        "case_a: 0.40 < 0.5 -- jargon"
    )
    assert _record(score=None, passed=False, reason="no score").failure_line() == (
        "case_a: n/a < 0.5 -- no score"
    )


# --- content ref / excerpt ------------------------------------------------


def test_content_ref_is_stable_and_change_sensitive():
    assert content_ref("hello") == content_ref("hello")
    assert content_ref("hello") != content_ref("hello!")
    assert content_ref("hello").startswith("sha256:")


def test_excerpt_collapses_whitespace_and_truncates():
    assert excerpt("a\n\n  b") == "a b"
    long_text = "x" * 500
    assert excerpt(long_text).endswith("...")
    assert len(excerpt(long_text)) == 303


# --- measure_and_record ---------------------------------------------------


@pytest.mark.asyncio
async def test_measure_and_record_writes_a_verdict(tmp_path):
    path = str(tmp_path / "verdicts.jsonl")
    metric = _FakeMetric([0.9])

    record = await measure_and_record(
        metric,
        _FakeTestCase("The recommended route is Tuesday morning."),
        eval_id="case_a",
        dimension="response_clarity",
        run={"dataset": {"name": "mock"}},
        path=path,
    )

    assert record.passed is True and record.score == 0.9 and record.threshold == 0.5
    assert record.decision_id == "case_a"  # defaults to eval_id
    assert record.reason == "judged 0.9"
    assert record.judged_at
    assert record.judge["metric"] == "_FakeMetric"
    assert record.run == {"dataset": {"name": "mock"}}
    assert record.output_excerpt == "The recommended route is Tuesday morning."
    assert record.output_ref == content_ref("The recommended route is Tuesday morning.")
    assert read_verdicts(path) == [record]


@pytest.mark.asyncio
async def test_measure_and_record_snapshots_each_score(tmp_path):
    """The metric is a reused singleton: a per-case score must be captured at
    measure time, not read back after the loop."""
    path = str(tmp_path / "verdicts.jsonl")
    metric = _FakeMetric([0.9, 0.1])

    for eval_id in ("case_a", "case_b"):
        await measure_and_record(
            metric, _FakeTestCase("out"), eval_id=eval_id, dimension="d", run={}, path=path
        )

    assert [(r.eval_id, r.score, r.passed) for r in read_verdicts(path)] == [
        ("case_a", 0.9, True),
        ("case_b", 0.1, False),
    ]
    assert metric.score == 0.1  # the singleton only remembers the last case


@pytest.mark.asyncio
async def test_explicit_decision_id_is_kept_for_the_calibration_join(tmp_path):
    path = str(tmp_path / "verdicts.jsonl")
    record = await measure_and_record(
        _FakeMetric([0.7]),
        _FakeTestCase("out"),
        eval_id="feedback_1210bd3e_negative",
        dimension="response_clarity",
        decision_id="1210bd3e4a984ff0bfb72a5426af3ed6",
        run={},
        path=path,
    )
    assert record.decision_id == "1210bd3e4a984ff0bfb72a5426af3ed6"


@pytest.mark.asyncio
async def test_unscorable_judge_does_not_pass(tmp_path):
    """A judge returning no usable number must not wave the case through."""
    record = await measure_and_record(
        _FakeMetric([None]),
        _FakeTestCase("out"),
        eval_id="case_a",
        dimension="d",
        run={},
        path=str(tmp_path / "verdicts.jsonl"),
    )
    assert record.score is None and record.passed is False


@pytest.mark.asyncio
async def test_judge_errors_propagate(tmp_path):
    """Persistence is best-effort; the judge call is not. A judge that cannot
    score is a real failure and must stay loud."""

    class _BrokenMetric(_FakeMetric):
        async def a_measure(self, test_case):
            raise RuntimeError("judge backend unavailable")

    with pytest.raises(RuntimeError, match="judge backend unavailable"):
        await measure_and_record(
            _BrokenMetric([]),
            _FakeTestCase("out"),
            eval_id="case_a",
            dimension="d",
            run={},
            path=str(tmp_path / "verdicts.jsonl"),
        )


# --- run provenance -------------------------------------------------------


def test_run_provenance_is_computed_once_and_reused(monkeypatch):
    """Replaying a case mutates the in-memory mock fixtures, so a freshly
    computed dataset content ref drifts run to run. Provenance must describe the
    DATASET, not the run's mutation state -- so it is resolved once and reused
    (the same trap eval/capture.py sidesteps by snapshotting up front). Proven
    against a changing source so the property holds regardless of test order."""
    from eval import dataset as dataset_module
    from eval.judge_log import current_run_provenance

    calls = []

    def _drifting_provenance(dataset):
        calls.append(dataset)
        return {"dataset": {"name": "mock", "ref": f"sha256:call{len(calls)}"}}

    monkeypatch.setattr(dataset_module, "run_provenance", _drifting_provenance)

    current_run_provenance.cache_clear()
    try:
        first = current_run_provenance()
        assert current_run_provenance() == first
        assert current_run_provenance()["dataset"]["ref"] == "sha256:call1"
        assert len(calls) == 1, "provenance was recomputed; records would disagree on the dataset"
    finally:
        current_run_provenance.cache_clear()


# --- the config knob ------------------------------------------------------


def test_judge_log_path_defaults_beside_the_human_feedback_log(monkeypatch):
    from smart_assignment.shared.config import Config

    monkeypatch.delenv("SMART_ASSIGNMENT_JUDGE_LOG_PATH", raising=False)
    assert Config.from_env().judge_log_path == "feedback_data/judge_verdicts.jsonl"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_judge_log_path_env_disables_recording(monkeypatch, value):
    """Unlike feedback_log_path, an explicitly empty value is meaningful here:
    it turns recording off rather than falling back to the default."""
    from smart_assignment.shared.config import Config

    monkeypatch.setenv("SMART_ASSIGNMENT_JUDGE_LOG_PATH", value)
    assert Config.from_env().judge_log_path == ""


def test_judge_log_path_env_override(monkeypatch):
    from smart_assignment.shared.config import Config

    monkeypatch.setenv("SMART_ASSIGNMENT_JUDGE_LOG_PATH", "/tmp/verdicts.jsonl")
    assert Config.from_env().judge_log_path == "/tmp/verdicts.jsonl"


@pytest.mark.asyncio
async def test_recording_disabled_still_returns_the_verdict(tmp_path):
    """Turning the log off changes what is stored, never what the test sees."""
    record = await measure_and_record(
        _FakeMetric([0.2]),
        _FakeTestCase("out"),
        eval_id="case_a",
        dimension="d",
        run={},
        path="",
    )
    assert record.score == 0.2 and record.passed is False
