"""Keep what the agent actually said during a live eval run, instead of
discarding it.

``eval/test_eval.py`` already replays every case against the real agent, and
already pays for it -- a full conversation per case against a live LLM backend.
It then throws every final response away and scores only the tool trajectory.
Meanwhile ``eval/test_quality.py`` judged prose from ``eval/data/golden_responses.json``,
which is only as fresh as the last time somebody remembered to run
``python3 -m eval.capture`` by hand. So the two jobs in CI could disagree about
what the agent says: one measuring this commit, the other judging text from
whenever.

This module closes that for free. It observes the inference stream
``eval/inference_guard.py`` is already watching (see that module for why this is
an observer and not a second monkeypatch) and writes what each case produced to
``feedback_data/latest_run_responses.json``. Zero extra LLM calls.

**Two files, two jobs -- deliberately not one.** ``eval/data/golden_responses.json``
stays exactly what it was: the committed, human-reviewed *reference*, promoted by
a deliberate act, feeding the reference-BASED metrics (``response_match_score``
and its v2). This file is the *current run's* output, regenerated every run,
never committed, feeding the reference-FREE judges (``brief_quality``,
``response_clarity``). Collapsing them would have cost the approved-reference
semantics and the reviewable diff, and would have let a run grade itself.

It lives under ``feedback_data/`` because that directory is already gitignored
and already holds eval-run output -- ``judge_log.py`` writes its verdicts there
by default -- so this needs no new ignore rule and sits beside the log it pairs
with.

**Replaces rather than merges**, unlike ``eval/capture.py``. This file describes
one run; merging would let a case the current run never touched sit alongside
cases it did, under a filename claiming otherwise.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from eval.capture import CaptureResult, read_records, response_record, serialize_records
from eval.case_set import resolve_case_set
from eval.dataset import resolve_eval_dataset, run_provenance
from eval.judge_log import utc_now_iso
from eval.response_extract import extract_final_response

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).parent.parent
# Beside judge_log.py's verdicts, in the already-gitignored run-output directory.
HARVEST_PATH = _REPO_ROOT / "feedback_data" / "latest_run_responses.json"


def load_latest_run(path: Optional[pathlib.Path] = None) -> Dict[str, CaptureResult]:
    """What the most recent live eval run produced, or ``{}`` when no run has
    written the file. Same record shape as the golden reference, read by the same
    parser (``eval.capture.read_records``)."""
    return read_records(HARVEST_PATH if path is None else path)


def _invocation_contents(invocation: Any) -> List[Any]:
    """Every ``Content`` of one invocation, in order, as
    ``eval/response_extract.py`` wants them.

    [VERIFIED against installed google-adk 2.3.0]
    ``EvaluationGenerator.convert_events_to_eval_invocations`` drops the final
    event from ``intermediate_data.invocation_events`` *unless* it carries
    function calls, and separately exposes it as ``final_response``. So neither
    field alone is the whole turn: on a RECOMMEND the closing narration lives
    only in ``final_response``, while on an ESCALATE the handoff call is in both.
    Concatenating covers both; the duplicate on an escalate is harmless, since
    the reader takes the handoff message either way.
    """
    intermediate = getattr(invocation, "intermediate_data", None)
    events = getattr(intermediate, "invocation_events", None) or []
    contents = [getattr(event, "content", None) for event in events]
    contents.append(getattr(invocation, "final_response", None))
    return contents


class RunHarvester:
    """Collects each eval case's real response, then writes them as one file.

    Constructed BEFORE the run starts, because it snapshots the run's provenance
    then: replaying a case mutates the in-memory mock fixtures, so
    ``eval.dataset.dataset_content_ref`` returns a different hash afterwards and
    a ref taken at write time would describe the mutation state rather than the
    dataset. ``eval/capture.py`` and ``eval/judge_log.py`` both dodge the same
    trap the same way.
    """

    def __init__(self) -> None:
        self._provenance = run_provenance(resolve_eval_dataset())
        self._captured_at = utc_now_iso()
        self._decision_ids = {case.eval_id: case.decision_id for case in resolve_case_set().cases}
        self._results: Dict[str, CaptureResult] = {}

    def observe(self, result: Any) -> None:
        """Record one ``InferenceResult``. Safe to call for a failed inference --
        it has no usable invocations and is skipped, having already been reported
        by ``eval/inference_guard.py``.

        With more than one run per case (``SMART_ASSIGNMENT_EVAL_NUM_RUNS`` above
        its default of 1) ADK produces one result per run per case: the LAST one
        wins, and the file records only that. Judging every run would multiply
        judge cost by the same factor for a question nobody asked."""
        eval_id = getattr(result, "eval_case_id", None)
        if not eval_id:
            return
        for invocation in getattr(result, "inferences", None) or []:
            extracted = extract_final_response(_invocation_contents(invocation))
            if extracted is None:
                continue
            self._results[eval_id] = CaptureResult(
                final_response=extracted.final_response,
                escalated=extracted.escalated,
                decision_id=self._decision_ids.get(eval_id),
            )

    def write(self, path: Optional[pathlib.Path] = None) -> int:
        """Write the harvest and return how many responses it holds.

        Writes even when empty, so a run that produced nothing leaves a truthful
        empty file rather than a stale one from a previous run that
        ``eval/test_quality.py`` would then judge as if it were current."""
        target = HARVEST_PATH if path is None else path
        entries = {
            eval_id: response_record(result, self._provenance, self._captured_at)
            for eval_id, result in self._results.items()
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialize_records(entries), encoding="utf-8")
        logger.info("Harvested %d agent response(s) to %s", len(entries), target)
        return len(entries)
