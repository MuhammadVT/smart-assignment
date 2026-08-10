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

**It is also the one place that patches that method.** Anything else needing to
see the inference stream -- ``eval/capture_harvest.py`` records what the agent
actually said -- registers an ``observer`` here rather than wrapping
``perform_inference`` a second time. Two independent patchers of one method work
only while they happen to be nested correctly, and nothing would enforce that.
An observer is a plain callable, so a consumer needs no monkeypatching at all,
and a runner that does not pass one *cannot* observe -- which is why
``eval/test_response_match.py`` is structurally incapable of harvesting into the
very reference file it scores against.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Iterator, List, Sequence

logger = logging.getLogger(__name__)


class DroppedEvalCasesError(AssertionError):
    """One or more eval cases never produced an inference, so they were never
    scored. Subclasses ``AssertionError`` so pytest reports it as a plain test
    failure rather than an infrastructure error."""


def _describe(dropped: List[Any]) -> str:
    lines = [f"  - {result.eval_case_id}: {result.error_message}" for result in dropped]
    return "\n".join(lines)


@contextlib.contextmanager
def fail_on_dropped_cases(
    observers: Sequence[Callable[[Any], None]] = (),
) -> Iterator[List[Any]]:
    """Raise ``DroppedEvalCasesError`` if any eval case failed inference inside
    this block.

    Yields the (initially empty) list of failed ``InferenceResult``s, so a caller
    can inspect them before the check fires.

    ``observers`` are called with every ``InferenceResult`` as it streams past,
    successes included -- the seam described in the module docstring. An observer
    that raises is logged and skipped, never propagated: it runs inside ADK's own
    async generator, so letting it escape would abort the whole eval run and turn
    an additive, advisory concern into a total failure. (A consumer that needs its
    absence noticed should check for its own missing output afterwards, which is
    what ``eval/test_quality.py`` does.)

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
        logger.warning("google-adk[eval] is not installed; dropped-case detection is inactive.")
        yield dropped
        return

    original = LocalEvalService.perform_inference

    async def _recording_perform_inference(self, inference_request):
        # Pass every result straight through, unchanged and in order, so ADK's own
        # streaming/aclose semantics are untouched; only observe as they go by.
        async for result in original(self, inference_request=inference_request):
            if result.status == InferenceStatus.FAILURE:
                dropped.append(result)
            for observe in observers:
                try:
                    observe(result)
                except Exception:  # noqa: BLE001 - see the docstring: never abort the run
                    logger.warning(
                        "An inference observer raised on eval case %s; continuing.",
                        getattr(result, "eval_case_id", "<unknown>"),
                        exc_info=True,
                    )
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
