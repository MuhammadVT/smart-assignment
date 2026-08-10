"""
LLM factory: routes every model-creation and content-generation call through
one of two backends, selected by SMART_ASSIGNMENT_LLM_BACKEND:

  "sage"     → the Sysco Sage SDK (enterprise-governed, TLS-injected). Two
               sub-paths, selected by Config.use_sage_gateway:
                 - off (default): SageLlmRegistry/SageLiteLlm calls one
                   registered SAGE agent directly (SAGE_CLIENT_ID,
                   SAGE_CLIENT_SECRET, SAGE_ENVIRONMENT).
                 - on: the SDK's GatewayLlm routes the call through Sysco's
                   enterprise LLM Gateway instead (an OpenAI-compatible
                   litellm proxy with OAuth2 token injection --
                   LLM_GATEWAY_CLIENT_ID, LLM_GATEWAY_CLIENT_SECRET,
                   optional LLM_GATEWAY_ENV). `sage_model` then names a
                   gateway-exposed model id, not a sage-* agent name.
  "standard" → SMART_ASSIGNMENT_MODEL is either a bare Gemini model name
               (e.g. "gemini-2.5-flash", used as-is) or a litellm-style
               "<provider>/<model>" string (e.g. "openai/gpt-4o-mini",
               "anthropic/claude-3-7-sonnet-latest"), wrapped in ADK's
               built-in LiteLlm so litellm handles that provider -- see
               https://docs.litellm.ai/docs/providers for the full list.
               Each provider's own env vars apply (e.g. OPENAI_API_KEY);
               requires the `litellm` extra -- see pyproject.toml.

A single env-var flip switches the entire project between them.

Exports
-------
get_llm(config)
    Returns the value for an ADK LlmAgent ``model=`` parameter.

generate_text(config, prompt)
    One-shot content generation — used by the grounded layers so they also flow
    through the same backend.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from smart_assignment.shared import tracing

# Re-exported so every existing call site (and the module docstring's promise
# that ``generate_text`` is safe to call from anywhere) keeps working unchanged.
# ``shared/async_bridge.py`` owns the *why*: the backend's HTTP session belongs
# to exactly one event loop for the whole process, and these two functions are
# how synchronous code reaches it.
from smart_assignment.shared.async_bridge import (  # noqa: F401
    offload_to_worker_thread,
    run_coroutine_blocking as _run_coro_blocking,
)

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: Sage SDK bootstrap (lazy, cached, runs once per process)
# ---------------------------------------------------------------------------

_SAGE_REGISTRY: Any = None
_SAGE_GATEWAY_LLM_CLS: Any = None


# The sage backend needs the Sage SDK, installed via the `sage` optional extra
# (``pip install -e ".[sage]"`` / ``uv sync --extra sage``). It is imported lazily
# -- only inside these loaders, only when ``llm_backend == "sage"`` -- so importing
# this module never requires the SDK, and the standard/offline paths run without it.
_SAGE_INSTALL_HINT = (
    "The Sage SDK is not installed. Install the sage extra: "
    'uv sync --extra sage   (or: pip install -e ".[sage]")'
)


def _load_sage_registry() -> Any:
    """
    Import SageLlmRegistry (the direct-to-agent Sage path) and inject
    enterprise TLS. Cached after the first successful load so TLS is only
    injected once.
    """
    global _SAGE_REGISTRY
    if _SAGE_REGISTRY is not None:
        return _SAGE_REGISTRY

    try:
        from sage_adk import SageLlmRegistry  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_SAGE_INSTALL_HINT) from exc

    import truststore  # type: ignore[import-untyped]

    truststore.inject_into_ssl()
    _SAGE_REGISTRY = SageLlmRegistry
    return _SAGE_REGISTRY


def _load_sage_gateway_llm_cls() -> Any:
    """
    Import GatewayLlm -- the Sage SDK's ADK `LiteLlm` subclass that routes a
    model through Sysco's enterprise LLM Gateway (see Config.use_sage_gateway)
    -- and inject enterprise TLS. Same lazy/cached shape as _load_sage_registry,
    just a different class off the same SDK.
    """
    global _SAGE_GATEWAY_LLM_CLS
    if _SAGE_GATEWAY_LLM_CLS is not None:
        return _SAGE_GATEWAY_LLM_CLS

    try:
        from sage_adk import GatewayLlm  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_SAGE_INSTALL_HINT) from exc

    import truststore  # type: ignore[import-untyped]

    truststore.inject_into_ssl()
    _SAGE_GATEWAY_LLM_CLS = GatewayLlm
    return _SAGE_GATEWAY_LLM_CLS


def _check_sage_env_vars() -> None:
    """Raise RuntimeError early if any required direct-agent Sage credential
    is absent (the SageLlmRegistry/SageLiteLlm path)."""
    missing = [
        v
        for v in ("SAGE_CLIENT_ID", "SAGE_CLIENT_SECRET", "SAGE_ENVIRONMENT")
        if not os.environ.get(v)
    ]
    if missing:
        raise RuntimeError(
            "SMART_ASSIGNMENT_LLM_BACKEND=sage requires the following "
            f"environment variables to be set: {', '.join(missing)}"
        )


def _check_sage_gateway_env_vars() -> None:
    """Raise RuntimeError early if the LLM Gateway credentials are absent
    (the GatewayLlm path, Config.use_sage_gateway=True). LLM_GATEWAY_ENV is
    intentionally not required here -- the SDK's GatewayClient defaults it to
    "qa" when unset."""
    missing = [
        v
        for v in ("LLM_GATEWAY_CLIENT_ID", "LLM_GATEWAY_CLIENT_SECRET")
        if not os.environ.get(v)
    ]
    if missing:
        raise RuntimeError(
            "SMART_ASSIGNMENT_USE_SAGE_GATEWAY=true requires the following "
            f"environment variables to be set: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Diagnostic: reveal the real sage reply the SDK hides behind its error string
# ---------------------------------------------------------------------------

# The Sage SDK's ``SageLiteLlm._extract_response`` replaces the model's real answer
# with this exact sentinel whenever the agent returns a tool/function call whose
# name isn't in the tools we passed (the grounded path passes none), or an
# unexpected shape. Downstream we then only see an empty/"Something went wrong"
# reply and a JSONDecodeError -- with no clue what the agent actually did.
_SAGE_ERROR_SENTINEL = "Something went wrong, Please try again later"


def _install_sage_response_diagnostic(sage_litellm_cls: Any) -> None:
    """Wrap a ``SageLiteLlm``-like class's ``_extract_response`` so that whenever it
    returns the generic error sentinel, the REAL ``agent_response`` (a function call
    and its name, or the raw shape) is logged. Idempotent; never alters the returned
    value, so it can't change any decision or fallback -- it only adds a log line."""
    original = sage_litellm_cls._extract_response
    if getattr(original, "_sa_diagnostic", False):
        return

    def _logged_extract_response(response: Any, tools: Any):
        result = original(response, tools)
        try:
            _, text_response = result
            if text_response == _SAGE_ERROR_SENTINEL:
                data = getattr(response, "data", None) or {}
                agent_response = (data.get("responses") or {}).get("agent_response")
                logger.warning(
                    "Sage masked the reply with its generic error string; the real "
                    "agent_response (tools offered=%d) was: %r",
                    len(tools or []),
                    repr(agent_response)[:1000],
                )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never break the call
            logger.warning("Sage response diagnostic could not read agent_response: %s", exc)
        return result

    _logged_extract_response._sa_diagnostic = True  # type: ignore[attr-defined]
    sage_litellm_cls._extract_response = staticmethod(_logged_extract_response)


def _maybe_install_sage_response_diagnostic(config: "Config") -> None:
    """Install the diagnostic above when ``config.debug_sage_raw_response`` is on.
    A no-op otherwise, and silently skipped if the Sage SDK isn't importable, so it
    never affects the standard backend or offline imports."""
    if not getattr(config, "debug_sage_raw_response", False):
        return
    try:
        from sage_core import SageLiteLlm  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001 - diagnostic is best-effort only
        return
    _install_sage_response_diagnostic(SageLiteLlm)


# ---------------------------------------------------------------------------
# Compatibility: repair array-wrapped tool-call arguments
# ---------------------------------------------------------------------------

# A tool call's arguments are a NAMED mapping: ADK builds a genai ``FunctionCall``
# from them and pydantic requires a dict. The sage backend intermittently emits them
# wrapped in a JSON array alongside a stray sibling -- observed verbatim on an
# ``escalation_triage`` call (abridged):
#
#   [{"request": "{\"ok\": true, ..., \"rejected_alternatives\": [\"RTE-4200 ...\",
#                  \"RTE-4110 - Downtown / Midtown (WED): infeasible - truck capacity\"], ...}"},
#    ["RTE-4200 ...", "RTE-4110 - Downtown / Midtown (WED): infeasible - truck capacity"]]
#
# The first element is the complete, correct argument object. The second is a
# fragment of the escaped JSON *inside* it that leaked to the top level -- those
# strings are a verbatim duplicate of the ``rejected_alternatives`` array within
# ``request``, so nothing is lost by dropping them.
#
# ADK's own ``_parse_tool_call_arguments`` already repairs several malformed payloads
# (dict literals, unquoted keys), but this one is VALID JSON: it parses cleanly to a
# list and is handed straight to ``types.Part.from_function_call(args=<list>)``, where
# pydantic raises. That exception escapes the whole agent run -- in eval the case is
# silently dropped (a green suite that scored fewer cases than it looks), on a live
# turn the turn dies -- and no retry helps, since a ValidationError is not a
# retryable API error.
#
# The repair is deliberately narrow. A bare array carries no parameter names, so it
# cannot be a valid argument set for ANY tool -- it is provably debris, not data we
# might be discarding. So: exactly one object in the array -> use it and log what was
# dropped; any other shape -> change nothing and let ADK raise exactly as it does
# today. An argument value is never synthesized.

# Longest payload repr written to a log line, so a malformed-call log entry stays
# readable (matches the sage response diagnostic above).
_MAX_LOGGED_PAYLOAD_CHARS = 1000


def _coerce_tool_call_args(parsed: Any) -> "tuple[Optional[dict], list]":
    """Return ``(arguments, discarded)`` for an already-parsed tool-call payload.

    ``arguments`` is the mapping ADK should use, or ``None`` when the payload is not
    a shape that can be read with certainty -- the caller then leaves it untouched.
    ``discarded`` holds the non-object siblings dropped from an array-wrapped
    payload, so every repair can be logged and audited.

    Pure: no logging, no I/O, no imports -- the whole decision is testable offline.
    """
    if isinstance(parsed, dict):
        return parsed, []  # already valid; returned as-is, never copied
    if not isinstance(parsed, list):
        return None, []
    objects = [item for item in parsed if isinstance(item, dict)]
    if len(objects) != 1:
        # Zero objects (nothing to use) or several (they may disagree, and picking
        # one would be a guess) -- neither is ours to repair.
        return None, []
    return objects[0], [item for item in parsed if not isinstance(item, dict)]


def _install_litellm_tool_args_repair(lite_llm_module: Any) -> None:
    """Wrap ADK's ``lite_llm._parse_tool_call_arguments`` with the repair above.

    Idempotent, and it never raises: any unexpected failure inside the wrapper falls
    through to exactly the value ADK would have used without it. Takes the module as
    an argument (rather than importing it) so a test can drive it with a stub.

    The installed wrapper carries the function it replaced as ``_sa_original``. This
    patch is process-wide by design (ADK resolves that function as a module global),
    so that attribute is the supported way to restore ADK's untouched behavior --
    e.g. a test that needs to observe the unrepaired failure."""
    original = lite_llm_module._parse_tool_call_arguments
    if getattr(original, "_sa_tool_args_repair", False):
        return

    def _repairing_parse(arguments: Any) -> Any:
        parsed = original(arguments)
        if isinstance(parsed, dict):
            return parsed  # the overwhelmingly common path -- identical object out
        try:
            repaired, discarded = _coerce_tool_call_args(parsed)
            if repaired is None:
                logger.error(
                    "Tool-call arguments parsed to %s, not an object, in a shape that "
                    "cannot be repaired safely; passing it through unchanged (ADK will "
                    "reject it). Raw payload: %s",
                    type(parsed).__name__,
                    repr(arguments)[:_MAX_LOGGED_PAYLOAD_CHARS],
                )
                return parsed
            logger.warning(
                "Repaired array-wrapped tool-call arguments: kept the single argument "
                "object, discarded %d stray sibling(s): %s",
                len(discarded),
                repr(discarded)[:_MAX_LOGGED_PAYLOAD_CHARS],
            )
            return repaired
        except Exception as exc:  # noqa: BLE001 - a compat shim must never break a call
            logger.warning(
                "Tool-call argument repair failed (%s); using the unrepaired value.", exc
            )
            return parsed

    _repairing_parse._sa_tool_args_repair = True  # type: ignore[attr-defined]
    _repairing_parse._sa_original = original  # type: ignore[attr-defined]
    lite_llm_module._parse_tool_call_arguments = _repairing_parse


def _maybe_install_litellm_tool_args_repair(config: "Config") -> None:
    """Install the repair above when ``config.repair_tool_call_args`` is on (the
    default). Silently skipped when ADK's litellm model module isn't importable, so
    it never affects a backend that doesn't use it or an offline import."""
    if not getattr(config, "repair_tool_call_args", True):
        return
    try:
        from google.adk.models import lite_llm  # requires litellm
    except Exception:  # noqa: BLE001 - the shim is best-effort only
        return
    _install_litellm_tool_args_repair(lite_llm)


def _maybe_install_sage_hooks(config: "Config") -> None:
    """Install the process-wide hooks the sage backend needs, each behind its own
    flag: the raw-response diagnostic (opt-in) and the tool-call-args repair (on by
    default). Grouped so the sage entry points below can't drift on which hooks they
    install."""
    _maybe_install_sage_response_diagnostic(config)
    _maybe_install_litellm_tool_args_repair(config)


# ---------------------------------------------------------------------------
# Retrying a transient sage request failure
# ---------------------------------------------------------------------------


def _apply_request_retries(llm: Any, config: "Config") -> Any:
    """Give a LiteLlm-based sage model litellm's own retry, and return it.

    A sage request that times out surfaces as ``litellm.APIConnectionError``, which
    subclasses ``openai.APIError`` -- so litellm's async wrapper retries it as soon
    as ``num_retries`` is set on the call. ADK's ``LiteLlm`` merges its
    ``_additional_args`` into the litellm call, so setting the key there is enough;
    no wrapping or patching is needed.

    Why this is necessary at all: ADK's eval harness *intends* these to be retried
    -- it registers a plugin that sets ``HttpRetryOptions(attempts=7)`` -- but that
    is a google-genai construct and ADK's ``LiteLlm`` never reads it. On the sage
    path the retry was configured and silently ignored, so one transient timeout
    lost the whole turn.

    Defensive: an unexpected model object (no ``_additional_args`` dict) is logged
    and returned untouched rather than raising, and an explicit ``num_retries``
    already set by the SDK is respected.
    """
    attempts = int(getattr(config, "sage_request_attempts", 1) or 1)
    if attempts <= 1:
        return llm  # retrying disabled -- prior behavior, exactly

    additional = getattr(llm, "_additional_args", None)
    if not isinstance(additional, dict):
        logger.warning(
            "Sage model %s exposes no litellm argument dict; request retries are "
            "INACTIVE (%d attempts were requested).",
            type(llm).__name__,
            attempts,
        )
        return llm

    # litellm counts RETRIES, not attempts: it runs the call once itself and then
    # retries up to num_retries more times (tenacity stop_after_attempt).
    additional.setdefault("num_retries", attempts - 1)
    return llm


def _build_sage_llm(config: "Config") -> Any:
    """The configured sage model object: process hooks installed, direct-agent vs
    gateway sub-path selected, request retries applied. One place, so the three
    sage entry points below cannot drift on any of it."""
    _maybe_install_sage_hooks(config)
    llm = (
        get_sage_gateway_llm(config.sage_model)
        if config.use_sage_gateway
        else get_sage_llm(config.sage_model)
    )
    return _apply_request_retries(llm, config)


# ---------------------------------------------------------------------------
# Internal: async content generation through ADK BaseLlm
# ---------------------------------------------------------------------------


async def _sage_call_async(
    llm: Any, prompt: str, tools: Optional[list] = None
) -> "tuple[Optional[dict], str]":
    """Drive one content-generation turn through an ADK BaseLlm object.

    Returns ``(function_call_args, text)``:

      - ``function_call_args`` is the arguments dict of the model's tool call when
        ``tools`` were offered and the model answered by CALLING one (the reliable
        way to get structured output from the conversational SAGE agent, which
        narrates when merely asked for JSON but readily emits function calls).
        ``None`` when no tool call was made.
      - ``text`` is the concatenated free-text parts (the model's prose reply, or
        empty on a pure tool call).

    ADK's ``LlmResponse`` has no flat ``.text`` attribute (that was an incorrect
    assumption that raised ``AttributeError: 'LlmResponse' object has no attribute
    'text'`` on the first real sage reply); the generated text lives in
    ``response.content.parts[i].text`` and a tool call in
    ``response.content.parts[i].function_call``. Raise on an error response so the
    caller falls back deterministically with a clear reason rather than silently
    returning "".
    """
    from google.adk.models.llm_request import LlmRequest  # ADK 2.x
    from google.genai import types

    gen_config = types.GenerateContentConfig(tools=tools or None)
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=gen_config,
    )
    call_args: Optional[dict] = None
    chunks: list[str] = []
    async for response in llm.generate_content_async(request, stream=False):
        error_code = getattr(response, "error_code", None)
        if error_code:
            message = getattr(response, "error_message", None) or ""
            raise RuntimeError(f"Sage backend returned an error: {error_code} {message}".strip())
        content = getattr(response, "content", None)
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "args", None):
                # The model answered via the structured tool call -- its args ARE
                # the answer (already JSON-repaired by the SDK on the way through).
                call_args = dict(fc.args)
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return call_args, "".join(chunks)


async def _generate_via_sage_async(llm: Any, prompt: str) -> str:
    """Text-only convenience over ``_sage_call_async`` (no tools) for the plain
    ``generate_text`` path -- unchanged behavior: concatenated text, raises on an
    error response."""
    _, text = await _sage_call_async(llm, prompt, tools=None)
    return text


def _to_genai_schema(node: dict) -> Any:
    """Build a ``google.genai`` ``Schema`` from a small JSON-schema dict (the
    subset a tool-parameter declaration needs: type, description, enum, properties,
    required, items, nullable). Kept explicit so the JSON-schema ``"object"`` /
    ``"string"`` type strings map to the genai ``Type`` enum deterministically."""
    from google.genai import types

    type_map = {
        "object": types.Type.OBJECT,
        "array": types.Type.ARRAY,
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
    }
    kwargs: dict[str, Any] = {}
    if node.get("type") in type_map:
        kwargs["type"] = type_map[node["type"]]
    if node.get("description"):
        kwargs["description"] = node["description"]
    if node.get("enum"):
        kwargs["enum"] = list(node["enum"])
    if node.get("properties"):
        kwargs["properties"] = {k: _to_genai_schema(v) for k, v in node["properties"].items()}
    if node.get("required"):
        kwargs["required"] = list(node["required"])
    if node.get("items"):
        kwargs["items"] = _to_genai_schema(node["items"])
    if node.get("nullable"):
        kwargs["nullable"] = True
    return types.Schema(**kwargs)


def _to_genai_tools(tool: dict) -> list:
    """Wrap a provider-agnostic tool dict ``{name, description, parameters}`` into
    the ``google.genai`` ``Tool``/``FunctionDeclaration`` shape ADK expects."""
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=_to_genai_schema(tool["parameters"]),
                )
            ]
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _is_litellm_model(model: str) -> bool:
    """True for a litellm-style "<provider>/<model>" string, e.g.
    "openai/gpt-4o-mini" -- as opposed to a bare Gemini model name."""
    return "/" in model


def get_sage_llm(sage_model: str) -> Any:
    """Return a SageLlmRegistry LLM object (enterprise-governed, TLS-injected)
    for the given Sage-prefixed model name -- the direct-to-agent path. See
    ``get_sage_gateway_llm`` for the LLM-Gateway sibling.

    Standalone seam for callers that have a bare Sage model string rather than
    a full ``Config`` -- e.g. ``eval/sage_judge_llm.py``'s ADK-registry
    adapter, which lets ADK-INTERNAL model resolution (distinct from this
    repo's own ``get_llm()``/``generate_text()`` calls below) also address a
    Sage-approved model. ``get_llm()``'s sage branch is just this function
    plus the diagnostic hook, which needs the full ``Config``."""
    _check_sage_env_vars()
    registry = _load_sage_registry()
    return registry.get_llm(sage_model)


def get_sage_gateway_llm(model: str) -> Any:
    """Return a ``GatewayLlm`` (an ADK ``LiteLlm``) for ``model``, routed
    through Sysco's enterprise LLM Gateway instead of a registered SAGE agent
    -- see ``Config.use_sage_gateway``.

    ``model`` is a bare gateway-exposed model id (e.g. ``"gpt-4o"``), NOT a
    ``sage-*`` agent name -- ``GatewayLlm`` wraps it as ``"openai/{model}"``
    itself so litellm addresses the gateway's OpenAI-compatible endpoint.
    Credentials come from ``LLM_GATEWAY_CLIENT_ID``/``LLM_GATEWAY_CLIENT_SECRET``
    (read internally by the SDK's ``GatewayClient``), not the
    ``SAGE_CLIENT_ID``/``SAGE_CLIENT_SECRET``/``SAGE_ENVIRONMENT`` trio the
    direct-agent path uses."""
    _check_sage_gateway_env_vars()
    gateway_llm_cls = _load_sage_gateway_llm_cls()
    return gateway_llm_cls(model=model)


def get_llm(config: "Config") -> Any:
    """
    Return the value for an ADK LlmAgent ``model=`` parameter.

    sage     → SageLlmRegistry LLM object (enterprise-governed, TLS-injected),
               or -- when config.use_sage_gateway is True -- a GatewayLlm
               routed through Sysco's enterprise LLM Gateway instead.
    standard → config.model as-is if it's a bare Gemini name, or wrapped in
               ADK's LiteLlm if it's a "<provider>/<model>" string (any
               provider litellm supports, e.g. "openai/gpt-4o-mini")
    """
    if config.llm_backend == "sage":
        return _build_sage_llm(config)
    if _is_litellm_model(config.model):
        from google.adk.models.lite_llm import LiteLlm  # requires the `litellm` extra

        return LiteLlm(model=config.model)
    return config.model


def generate_text(config: "Config", prompt: str, role: Optional[str] = None) -> str:
    """
    One-shot content generation that honours the backend toggle.

    sage     → SageLlmRegistry LLM object via ADK BaseLlm (enterprise-governed)
    standard → litellm.completion(...) if config.model is a
               "<provider>/<model>" string, else google.genai.Client
               directly for a bare Gemini model name

    Raises on failure; callers should guard with ``except Exception``.

    ``role`` is an optional label (e.g. one of the ``Config.ROLE_*`` names) used
    only to annotate the tracing span; it does not affect model selection, which
    the caller has already applied via ``config.for_role(...)``.

    Note: the sage path is async under the hood. ``_run_coro_blocking`` (see
    ``shared/async_bridge.py``) drives it to completion on the one event loop
    this process's backend session is bound to, so this stays a safe synchronous
    call from the CLI pipeline, the eval suite, and the web app's request
    handlers alike -- where a bare ``asyncio.run`` would either raise or close
    the loop out from under the next call.

    When ``config.use_tracing`` is on, the whole call is wrapped in an
    OpenTelemetry span (see ``shared/tracing.py``); when off, ``llm_span`` is a
    transparent no-op, so this path is unchanged.
    """
    active_model = config.sage_model if config.llm_backend == "sage" else config.model
    with tracing.llm_span(
        config,
        "llm.generate_text",
        backend=config.llm_backend,
        model=active_model,
        role=role or "",
        prompt_chars=len(prompt),
    ) as span:
        result = _generate_text_impl(config, prompt)
        span.set_attribute("smart_assignment.response_chars", len(result))
        return result


def generate_tool_call(
    config: "Config", prompt: str, tool: dict, role: Optional[str] = None
) -> "tuple[Optional[dict], str]":
    """One-shot STRUCTURED generation: offer the model a single function whose
    arguments ARE the answer, and return ``(call_args, text)``.

    ``tool`` is a provider-agnostic declaration ``{name, description, parameters}``
    where ``parameters`` is a small JSON-schema dict.

    Why a tool instead of "reply with JSON": the direct SAGE agent is a
    conversational agent that narrates when asked for JSON (returning prose a
    downstream ``json.loads`` rejects), but it reliably emits *function calls* --
    so handing it one tool is the dependable channel for structured output. On a
    successful call ``call_args`` is the (SDK-repaired) arguments dict and ``text``
    is usually empty; if the model narrates anyway, ``call_args`` is ``None`` and
    ``text`` carries the prose for the caller to salvage/parse and log.

    Only the sage backend uses the tool channel today (both the direct agent and
    the LLM-Gateway sibling drive an ADK ``BaseLlm``). Other backends have no tool
    path here yet and return ``(None, <generated text>)`` -- they gain real JSON
    mode when the project moves to the gateway. Raises on backend failure; callers
    guard with ``except Exception`` and fall back deterministically.
    """
    active_model = config.sage_model if config.llm_backend == "sage" else config.model
    with tracing.llm_span(
        config,
        "llm.generate_tool_call",
        backend=config.llm_backend,
        model=active_model,
        role=role or "",
        prompt_chars=len(prompt),
    ) as span:
        call_args, text = _generate_tool_call_impl(config, prompt, tool)
        span.set_attribute("smart_assignment.tool_called", call_args is not None)
        span.set_attribute("smart_assignment.response_chars", len(text))
        return call_args, text


def _generate_tool_call_impl(
    config: "Config", prompt: str, tool: dict
) -> "tuple[Optional[dict], str]":
    """Backend-routing body of ``generate_tool_call``. The sage backend offers the
    tool through the ADK ``BaseLlm`` (direct agent or gateway); every other backend
    has no tool channel yet, so it degrades to plain text generation."""
    if config.llm_backend == "sage":
        llm = _build_sage_llm(config)
        return _run_coro_blocking(_sage_call_async(llm, prompt, _to_genai_tools(tool)))

    # No tool channel on the other backends yet -> text only (JSON mode arrives
    # with the LLM Gateway); the caller salvages/parses the text as before.
    return None, _generate_text_impl(config, prompt)


def _generate_text_impl(config: "Config", prompt: str) -> str:
    """The backend-routing body of ``generate_text``, kept separate so the public
    function owns the tracing span and this stays pure LLM-dispatch logic."""
    if config.llm_backend == "sage":
        llm = _build_sage_llm(config)
        return _run_coro_blocking(_generate_via_sage_async(llm, prompt))

    if _is_litellm_model(config.model):
        import litellm  # requires the `litellm` extra

        resp = litellm.completion(
            model=config.model, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()

    # bare Gemini model name — the original single-model implementation
    from google import genai  # type: ignore[import-untyped]

    client = genai.Client()
    resp = client.models.generate_content(model=config.model, contents=prompt)
    return (resp.text or "").strip()
