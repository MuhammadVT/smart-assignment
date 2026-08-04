"""Control and observe ADK's inference stage from the live-eval entry points.

Both concerns here hang off the same seam -- wrapping
``LocalEvalService.perform_inference``, the one method every eval case's
inference streams through -- which is why they share a module rather than
reimplementing ADK's evaluate flow twice:

* :func:`fail_on_dropped_cases` -- a dropped case must fail the run, not vanish.
* :func:`pinned_parallelism` -- how many cases ADK infers concurrently.

--- Dropped cases ---

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

--- Parallelism ---

``AgentEvaluator`` builds its ``InferenceConfig()`` with defaults and exposes no
override, so every eval case runs concurrently (ADK's default is 4). Against a
single Sage endpoint that concurrency inflates per-call latency until brief
generation crosses the request timeout, and the case is dropped -- an
infrastructure failure that says nothing about the agent.
:func:`pinned_parallelism` overrides that value on the request as it goes past,
driven by ``SMART_ASSIGNMENT_EVAL_PARALLELISM``. Nothing else about ADK's
scheduling changes.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)

PARALLELISM_ENV = "SMART_ASSIGNMENT_EVAL_PARALLELISM"
# Cases inferred concurrently when the env var is unset.
#
# 4 is ADK's own default, kept deliberately: a sweep of the full golden set at
# parallelism 1 / 2 / 4 (3 runs each) came out 1/3 green at EVERY setting, while
# median wall clock went 226s / 183s / 92s. Contention is not what drops cases --
# the drops span recommend AND escalate cases at every setting, which is the
# signature of backend latency variance, not of concurrency. Lowering this by
# default would buy nothing measurable and cost 2.5x the wall clock.
#
# The knob still earns its place: it is the only control over this that exists
# (AgentEvaluator exposes none), so a contended environment can dial it down
# without a code change. Revisit the default if a larger sample says otherwise.
DEFAULT_PARALLELISM = 4


class DroppedEvalCasesError(AssertionError):
    """One or more eval cases never produced an inference, so they were never
    scored. Subclasses ``AssertionError`` so pytest reports it as a plain test
    failure rather than an infrastructure error."""


def _describe(dropped: List[Any]) -> str:
    lines = [
        f"  - {result.eval_case_id}: {result.error_message}" for result in dropped
    ]
    return "\n".join(lines)


def resolve_parallelism() -> int:
    """How many eval cases to infer concurrently.

    ``SMART_ASSIGNMENT_EVAL_PARALLELISM`` when it is a positive integer, else
    ``DEFAULT_PARALLELISM``. A malformed or non-positive value is a loud warning
    and the default, never a crash mid-run and never a silent 0 (which would
    deadlock ADK's semaphore).
    """
    raw = os.environ.get(PARALLELISM_ENV)
    if not raw or not raw.strip():
        return DEFAULT_PARALLELISM
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using the default of %d.",
            PARALLELISM_ENV,
            raw,
            DEFAULT_PARALLELISM,
        )
        return DEFAULT_PARALLELISM
    if value < 1:
        logger.warning(
            "%s=%d must be >= 1; using the default of %d.",
            PARALLELISM_ENV,
            value,
            DEFAULT_PARALLELISM,
        )
        return DEFAULT_PARALLELISM
    return value


@contextlib.contextmanager
def pinned_parallelism(value: Optional[int] = None) -> Iterator[int]:
    """Pin how many eval cases ADK infers concurrently inside this block.

    ``AgentEvaluator`` hardcodes ``InferenceConfig()`` and offers no override, so
    this overrides ``parallelism`` on the request as it passes through
    ``LocalEvalService.perform_inference``. Only that one field is touched --
    ADK still owns the scheduling, the semaphore, and the result stream.

    ``value`` defaults to :func:`resolve_parallelism`. Yields the value actually
    applied. Idempotent and always restored; a no-op (yielding the value) when
    google-adk[eval] is absent, exactly like :func:`fail_on_dropped_cases`.
    """
    resolved = resolve_parallelism() if value is None else value

    try:
        from google.adk.evaluation.local_eval_service import LocalEvalService
    except ImportError:  # pragma: no cover - exercised only without the eval extra
        yield resolved
        return

    original = LocalEvalService.perform_inference

    async def _pinned_perform_inference(self, inference_request):
        config = getattr(inference_request, "inference_config", None)
        if config is not None and getattr(config, "parallelism", None) != resolved:
            logger.info(
                "Pinning eval inference parallelism %s -> %d (%s).",
                getattr(config, "parallelism", "?"),
                resolved,
                PARALLELISM_ENV,
            )
            config.parallelism = resolved
        async for result in original(self, inference_request=inference_request):
            yield result

    LocalEvalService.perform_inference = _pinned_perform_inference
    try:
        yield resolved
    finally:
        LocalEvalService.perform_inference = original


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
