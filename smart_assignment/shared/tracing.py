"""
Optional, opt-in OpenTelemetry tracing seam for the LLM-backed decision layers.

This module is *additive and defensive*: it exports one context manager,
``llm_span``, that wraps a unit of LLM work in an OpenTelemetry span when
tracing is enabled, and is a complete no-op otherwise. It follows the same
guarantees the rest of this repo holds itself to (see CLAUDE.md):

* **Opt-in, default off.** Nothing happens unless ``Config.use_tracing`` is on
  (env ``SMART_ASSIGNMENT_USE_TRACING``). Flag-off reproduces prior behavior
  exactly and never imports the OpenTelemetry SDK.
* **Never worse than the baseline.** Tracing observes; it never changes a value
  a decision layer acts on. Every failure path here (SDK missing, no exporter
  configured, span machinery erroring) degrades to a silent no-op, so a broken
  or unreachable trace backend can never break a decision.
* **Credential-free import.** The OpenTelemetry SDK and the exporter are imported
  lazily inside ``_configure`` (guarded by a lock, once per process), so importing
  this module -- or the whole package -- needs neither the ``observability`` extra
  nor any backend credentials.

Exporter selection is deliberately vendor-neutral. The primary path is the
standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` (+ ``OTEL_EXPORTER_OTLP_HEADERS``)
environment configuration, so the target backend is swappable without touching
code. As a convenience for the chosen dev/prod stack, if the ``LANGFUSE_*``
variables are set instead, the OTLP endpoint and Basic-auth header are derived
from them -- Langfuse ingests OpenTelemetry directly, so this is just standard
OTLP with a computed endpoint and header, not a Langfuse-specific dependency.

Phase 0 records only *generic* span attributes (backend, model, role label,
prompt/response sizes, latency, error status). Deliberately, it does NOT record
prompt or response *text*: those can carry customer PII (an evidence packet
includes an address), and richer per-layer payloads are an intentional,
per-call-site decision for a later phase -- not something to leak globally here.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)

# Custom span attributes are namespaced under this prefix so they never collide
# with OpenTelemetry semantic-convention keys or ADK's own instrumentation.
_ATTR_PREFIX = "smart_assignment."

# One-time, lock-guarded init state. ``_TRACER`` holds an OpenTelemetry Tracer
# once configuration succeeds, or ``None`` if tracing could not be set up (SDK
# missing / no exporter / error). ``_INIT_DONE`` records that we have already
# attempted configuration, so we try exactly once per process.
_LOCK = threading.Lock()
_TRACER: Any = None
_INIT_DONE = False


class _NoopSpan:
    """A stand-in span whose mutating methods do nothing, so callers can use the
    yielded object uniformly whether or not tracing is active."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        return None


_NOOP_SPAN = _NoopSpan()


def _service_name() -> str:
    """The ``service.name`` reported for our spans; overridable via the standard
    ``OTEL_SERVICE_NAME`` env var."""
    return os.environ.get("OTEL_SERVICE_NAME", "smart-assignment").strip() or "smart-assignment"


def _langfuse_otlp_settings() -> Optional[Tuple[str, Dict[str, str]]]:
    """Derive an OTLP HTTP endpoint + Basic-auth header from the ``LANGFUSE_*``
    env vars, or ``None`` if they are not all set.

    Langfuse ingests OpenTelemetry at ``<host>/api/public/otel``; the traces
    signal path is ``/v1/traces``. Auth is HTTP Basic with the public key as the
    username and the secret key as the password. Kept as pure stdlib string work
    so it is unit-testable without the OpenTelemetry SDK installed.
    """
    host = (os.environ.get("LANGFUSE_HOST") or "").strip()
    public = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not (host and public and secret):
        return None
    endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return endpoint, {"Authorization": f"Basic {token}"}


def _build_exporter() -> Any:
    """Construct an OTLP span exporter from the environment, or return ``None``
    (logging why) if the exporter package is unavailable or no endpoint is
    configured. Vendor-neutral: standard ``OTEL_*`` config wins; the ``LANGFUSE_*``
    convenience is the fallback."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except Exception:  # noqa: BLE001 - any import problem means "cannot export"
        logger.warning(
            "Tracing is on but the OTLP HTTP exporter is unavailable; install the "
            "'observability' extra. Tracing disabled."
        )
        return None

    # Standard OTLP configuration takes precedence and keeps the backend swappable.
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ):
        return OTLPSpanExporter()

    langfuse = _langfuse_otlp_settings()
    if langfuse is not None:
        endpoint, headers = langfuse
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)

    logger.warning(
        "Tracing is on but no exporter endpoint is configured. Set "
        "OTEL_EXPORTER_OTLP_ENDPOINT (+ OTEL_EXPORTER_OTLP_HEADERS) or the "
        "LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY trio. Tracing disabled."
    )
    return None


def _configure() -> Any:
    """Attempt one-time tracing setup; return a Tracer or ``None`` on any failure.

    Uses a *local* ``TracerProvider`` rather than the global
    ``trace.set_tracer_provider`` so this seam does not claim the process-global
    provider. That keeps the door open for Phase 0.5 (the ADK OpenTelemetry
    instrumentor, which drives the global provider) to coexist without either
    side clobbering the other.
    """
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:  # noqa: BLE001 - SDK not installed / import error
        logger.warning(
            "SMART_ASSIGNMENT_USE_TRACING is on but the OpenTelemetry SDK is not "
            "installed; install the 'observability' extra. Tracing disabled."
        )
        return None

    exporter = _build_exporter()
    if exporter is None:
        return None

    try:
        provider = TracerProvider(resource=Resource.create({"service.name": _service_name()}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return provider.get_tracer("smart_assignment.tracing")
    except Exception:  # noqa: BLE001 - never let setup raise into a caller
        logger.warning("Failed to initialize the tracer; tracing disabled.", exc_info=True)
        return None


def _get_tracer(config: "Config") -> Any:
    """Return the configured Tracer, or ``None`` if tracing is off or unavailable.

    Off-by-flag returns ``None`` before touching any OpenTelemetry import, so the
    disabled path stays free of side effects. When enabled, configuration is
    attempted exactly once per process under a lock and the result cached.
    """
    if not getattr(config, "use_tracing", False):
        return None

    global _INIT_DONE, _TRACER
    if _INIT_DONE:
        return _TRACER
    with _LOCK:
        if _INIT_DONE:
            return _TRACER
        _TRACER = _configure()
        _INIT_DONE = True
        return _TRACER


def _start_span(config: "Config", name: str, attributes: Dict[str, Any]) -> Any:
    """Return a context manager for the span (real when tracing is active, else a
    ``nullcontext`` yielding the no-op span). Building it never runs the caller's
    body, so ``llm_span`` can yield exactly once and let any caller exception
    propagate through the real span's ``__exit__`` (which records it)."""
    tracer = _get_tracer(config)
    if tracer is None:
        return contextlib.nullcontext(_NOOP_SPAN)
    try:
        otel_attrs = {
            _ATTR_PREFIX + key: value
            for key, value in attributes.items()
            if value is not None and value != ""
        }
        return tracer.start_as_current_span(name, attributes=otel_attrs)
    except Exception:  # noqa: BLE001 - a tracing hiccup must never break the call
        logger.debug("Could not start a tracing span; continuing without it.", exc_info=True)
        return contextlib.nullcontext(_NOOP_SPAN)


@contextlib.contextmanager
def llm_span(config: "Config", name: str, **attributes: Any) -> Iterator[Any]:
    """Wrap a unit of LLM work in an OpenTelemetry span when tracing is enabled.

    Yields the active span (or a no-op stand-in) so the caller may attach further
    attributes via ``span.set_attribute(...)``. Span duration is captured
    automatically; an exception raised in the body is recorded on the span and
    its status set to error (OpenTelemetry defaults), then re-raised unchanged.

    When ``Config.use_tracing`` is off -- or the SDK/exporter is unavailable --
    this is a transparent no-op: same control flow, same return value, no imports.
    """
    with _start_span(config, name, attributes) as span:
        yield span


def _reset_for_tests() -> None:
    """Clear the cached one-time init state so a test can exercise configuration
    from a clean slate. Not part of the public API."""
    global _INIT_DONE, _TRACER
    with _LOCK:
        _INIT_DONE = False
        _TRACER = None
