"""Fail loudly when a live eval case never produced an inference.

ADK's ``LocalEvalService`` deliberately swallows a per-case inference failure so
that one bad case can't take down the rest of the run: it logs
``Inference failed for eval case `X` with error ...``, marks that
``InferenceResult`` ``InferenceStatus.FAILURE``, and returns it (see
``local_eval_service.py``'s ``_perform_inference_single_eval_item``). Nothing
downstream re-raises. ``AgentEvaluator`` then groups results *by eval id* --
a dropped case simply isn't a key -- so it contributes no metric result, no
failure, and no mention in the summary.

The consequence is the dangerous part: **a dropped case looks exactly like a
case that passed.** A run whose escalate cases both crashed reports
``1 passed``, and the score is silently computed over whatever survived. That
is how a real trajectory regression stayed invisible until someone read the
raw log, and it's the wrong default for a suite that is meant to become a
required check.

This module closes that gap without changing ADK's behavior: it wraps
``LocalEvalService.perform_inference`` to observe each ``InferenceResult`` as it
streams past, records the ones marked ``FAILURE``, and raises afterwards. Cases
still don't take each other down -- the run completes and every other case is
scored exactly as before -- but the *test* now fails, naming what was dropped
and why.

Used by both live-eval entry points (``eval/test_eval.py``,
``eval/test_response_match.py``), which run ``AgentEvaluator.evaluate()`` and
otherwise share the blind spot identically.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator, List

logger = logging.getLogger(__name__)


class DroppedEvalCasesError(AssertionError):
    """One or more eval cases never produced an inference, so they were never
    scored. Subclasses ``AssertionError`` so pytest reports it as a plain test
    failure rather than an infrastructure error."""


def _describe(dropped: List[Any]) -> str:
    lines = [
        f"  - {result.eval_case_id}: {result.error_message}" for result in dropped
    ]
    return "\n".join(lines)


@contextlib.contextmanager
def fail_on_dropped_cases() -> Iterator[List[Any]]:
    """Raise ``DroppedEvalCasesError`` if any eval case failed inference inside
    this block.

    Yields the (initially empty) list of failed ``InferenceResult``s, so a caller
    can inspect them before the check fires.

    An exception raised by the block itself is never masked -- ``AgentEvaluator``'s
    own metric assertion is the more specific failure, so it wins, and the dropped
    cases are logged instead. The patch is always undone, on every path.
    """
    dropped: List[Any] = []

    try:
        from google.adk.evaluation.base_eval_service import InferenceStatus
        from google.adk.evaluation.local_eval_service import LocalEvalService
    except ImportError:  # pragma: no cover - exercised only without the eval extra
        # Without google-adk[eval] there is nothing to guard; AgentEvaluator itself
        # raises its own actionable "Eval module is not installed" right after, and
        # swallowing that here would replace a good error with a worse one.
        logger.warning(
            "google-adk[eval] is not installed; dropped-case detection is inactive."
        )
        yield dropped
        return

    original = LocalEvalService.perform_inference

    async def _recording_perform_inference(self, inference_request):
        # Pass every result straight through, unchanged and in order, so ADK's own
        # streaming/aclose semantics are untouched; only observe as they go by.
        async for result in original(self, inference_request=inference_request):
            if result.status == InferenceStatus.FAILURE:
                dropped.append(result)
            yield result

    LocalEvalService.perform_inference = _recording_perform_inference
    try:
        yield dropped
    except BaseException:
        if dropped:
            logger.error(
                "In addition to the failure below, %d eval case(s) were dropped "
                "before scoring:\n%s",
                len(dropped),
                _describe(dropped),
            )
        raise
    finally:
        LocalEvalService.perform_inference = original

    if dropped:
        raise DroppedEvalCasesError(
            f"{len(dropped)} eval case(s) never produced an inference and were "
            "therefore NEVER SCORED -- without this check the run would have "
            "reported a pass over only the surviving cases:\n"
            f"{_describe(dropped)}\n"
            "A dropped case is an infrastructure or agent failure (a backend "
            "timeout, a malformed model reply), not a metric failure. Re-run to "
            "confirm it is transient; if it reproduces, fix the underlying error "
            "rather than ignoring it."
        )
