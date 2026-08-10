"""
Adapts this repo's own LLM-calling seam (``shared/llm.py``'s ``generate_text``)
to DeepEval's ``DeepEvalBaseLLM`` interface, so the G-Eval quality metrics in
``eval/test_quality.py`` work under WHATEVER backend this repo is configured
for -- ``sage``, standard bare-Gemini, or standard litellm-provider -- with no
per-backend branching in ``test_quality.py`` itself.

Why one adapter instead of a Sage-only one (contrast with
``eval/sage_judge_llm.py``, which exists ONLY because ADK's own
``LlmAsAJudgeCriterion`` resolves its judge model through ADK-core's generic
``LLMRegistry`` -- a mechanism this repo's Sage integration never touches, so a
registry-level bridge was the only way in): DeepEval imposes no such
constraint. ``GEval(model=...)`` happily accepts any ``DeepEvalBaseLLM``
instance, so the natural fit is to reuse ``generate_text(config, prompt)`` --
the SAME function ``routeslot/``, ``triage/`` etc. already call
for every other grounded decision in this repo, which already branches on
``Config.llm_backend`` internally. No second, divergent judge-model resolution
path to maintain.

Backend-specific correctness notes, both already solved by reusing existing
seams rather than reinventing them:

* **Sage's loop-bound aiohttp session.** The Sage SDK keeps ONE
  ``ClientSession`` per process, bound to the first event loop that touches it
  (see ``shared/async_bridge.py``). That module owns the problem: a synchronous
  ``generate_text`` runs on a dedicated, never-closing loop, so a judge can be
  called any number of times across any number of tests. ``a_generate`` below
  only has to stay out of its way -- see the comment there for why it must NOT
  pin pytest-asyncio's per-test loop.
* **Dead judge-model defaults.** Nothing here has its own default model
  string to go stale (unlike ADK's ``JudgeModelOptions.judge_model`` defaulting
  to the now-retired ``gemini-2.5-flash``, or DeepEval's own ``GeminiModel``
  defaulting to ``gemini-2.5-pro`` in newer DeepEval versions) -- the model
  comes entirely from ``Config``/``SMART_ASSIGNMENT_MODEL*`` env vars, the same
  single source of truth as the rest of the app.

Per-role model selection: this repo's convention (see ``shared/config.py``'s
``Config.for_role``) is that every LLM call site is scoped to a named role
before calling ``generate_text``, so an operator can independently override
just that role's model via ``SMART_ASSIGNMENT_MODEL_<ROLE>`` without touching
the app's main model. ``ROLE_QUALITY_JUDGE`` is that role for this adapter --
construct with an ALREADY-scoped config
(``DEFAULT_CONFIG.for_role(ROLE_QUALITY_JUDGE)``), not a bare ``DEFAULT_CONFIG``.

Credential-free import: no Sage/Gemini/litellm import happens at import time
of this module -- only when ``generate``/``a_generate`` actually runs, via
``generate_text``'s own lazy backend dispatch. Constructing
``SmartAssignmentDeepEvalLLM`` itself just stores the config; no network or
SDK import either.

One thing importing this module DOES always do: import ``deepeval`` itself.
[VERIFIED against installed deepeval 2.6.6]: `deepeval/__init__.py` makes an
outbound HTTPS GET to ``pypi.org`` at import time (a "newer version available"
check) UNLESS ``DEEPEVAL_UPDATE_WARNING_OPT_OUT=YES`` is set -- a SEPARATE
switch from ``DEEPEVAL_TELEMETRY_OPT_OUT`` (which only covers usage-analytics
events). Both are set via ``os.environ.setdefault`` at the top of THIS module
-- the actual deepeval-import boundary -- rather than relying on every caller
(a test file, ``test_quality.py``) to set them first; this keeps every
importer of this module network-silent by construction, including the
hermetic ``tests/eval/test_deepeval_llm.py`` (which still gracefully
``pytest.importorskip``s when the optional ``eval-quality`` extra isn't
installed at all, so it never becomes a hard hermetic-suite dependency).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Optional

# Must be set before ANY `deepeval` import, in this module or any caller's --
# deepeval reads both at import time (see module docstring above). Owned HERE,
# at the actual import boundary, rather than relying on every caller (e.g. a
# test file) to set them first: importing this module is what pulls deepeval
# in, so this module is what must make that safe.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402

from smart_assignment.shared.config import ROLE_QUALITY_JUDGE  # noqa: E402
from smart_assignment.shared.llm import generate_text, generate_tool_call  # noqa: E402

if TYPE_CHECKING:
    from pydantic import BaseModel

    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)


def tool_for_schema(schema: "type[BaseModel]") -> dict:
    """Translate a DeepEval verdict model into this repo's provider-agnostic tool
    declaration ``{name, description, parameters}``.

    DeepEval's verdict models are deliberately small and flat -- ``Steps`` is
    ``{steps: list[str]}``, ``ReasonScore`` is ``{reason: str, score: float}`` --
    so pydantic's own JSON schema is already the ``parameters`` object
    ``generate_tool_call`` wants, minus the ``title`` keys it ignores.

    Nothing here describes *what* to answer: G-Eval's prompt already specifies the
    rubric and the score range, and restating either would risk contradicting it.
    This only supplies the shape.

    Raises ``TypeError`` for a schema carrying ``$ref``/``$defs`` (a nested model),
    which this flat translation cannot express -- DeepEval catches ``TypeError``
    around its schema call and retries without one, so an unsupported schema
    degrades to the prose path instead of failing the metric.
    """
    json_schema = schema.model_json_schema()
    if "$defs" in json_schema or "$ref" in json.dumps(json_schema):
        raise TypeError(f"{schema.__name__} nests another model; no flat tool shape for it")

    return {
        "name": f"submit_{_snake_case(schema.__name__)}",
        "description": (
            "Submit your answer in this exact structure. Call this exactly once, "
            "and put the whole answer in the arguments -- do not also narrate it."
        ),
        "parameters": {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", []),
        },
    }


def _snake_case(name: str) -> str:
    """``ReasonScore`` -> ``reason_score``. Tool names are conventionally snake."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _salvage_json(text: str) -> "Optional[dict]":
    """The outermost JSON object in ``text``, or ``None``.

    The narration fallback: a conversational agent that ignores the tool usually
    still writes the JSON G-Eval's prompt asked for, wrapped in a sentence or a
    ```json fence. Outermost braces rather than a greedy regex so a fenced object
    containing nested objects survives intact; ``json_repair`` then covers the
    trailing commas and unquoted keys such prose tends to carry.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = _repair_json(candidate)
    return parsed if isinstance(parsed, dict) else None


def _repair_json(text: str) -> object:
    """Best-effort JSON recovery via ``json_repair``; ``None`` when the library
    isn't importable, so the caller degrades cleanly. Same helper, same reason, as
    ``routeslot/llm.py``."""
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:  # pragma: no cover - json_repair ships with the SDK
        return None
    return repair_json(text, return_objects=True)


class SmartAssignmentDeepEvalLLM(DeepEvalBaseLLM):
    """A DeepEval judge model backed by this repo's own ``generate_text``, so
    G-Eval metrics work under whatever ``SMART_ASSIGNMENT_LLM_BACKEND`` is
    configured -- including a Sage-only environment where no direct,
    non-Sage-approved model call is reachable at all."""

    def __init__(self, config: "Config") -> None:
        # `config` must already be scoped via `config.for_role(ROLE_QUALITY_JUDGE)`
        # -- see module docstring; this class does not scope it itself, matching
        # generate_text()'s own calling convention (role= is a tracing label
        # only, not a model-resolution step).
        self._config = config

    def load_model(self) -> "SmartAssignmentDeepEvalLLM":
        return self

    def generate(self, prompt: str, schema: "Optional[type[BaseModel]]" = None):
        """Free text, or -- when DeepEval asks for one -- an instance of ``schema``.

        The ``schema`` parameter is not optional decoration: G-Eval calls
        ``generate(prompt, schema=Steps)`` and falls back to a plain
        ``generate(prompt)`` on ``TypeError``. An adapter without the parameter
        therefore took that fallback on EVERY call, and G-Eval then ran
        ``json.loads`` over whatever came back. The direct SAGE agent is
        conversational and narrates when asked for JSON, so that reliably produced
        ``JSONDecodeError: Expecting value: line 1 column 1`` -- which is what made
        ``brief_quality`` flaky and ``response_clarity`` fail outright.

        The dependable structured channel for that agent is a function call, which
        is why this routes through ``generate_tool_call`` -- the same seam
        ``routeslot/`` uses, for the same reason.
        """
        if schema is None:
            return generate_text(self._config, prompt, role=ROLE_QUALITY_JUDGE)
        return self._generate_structured(prompt, schema)

    def _generate_structured(self, prompt: str, schema: "type[BaseModel]") -> "BaseModel":
        """One verdict, as an instance of ``schema``.

        Tool arguments first, salvaged JSON from narration second -- mirroring
        ``routeslot/llm.py``'s ``generate_route_slot_choice``. Raises when neither
        yields something the schema accepts: a judge that cannot produce a verdict
        must fail visibly, not return an invented score.

        Only the sage backend has a tool channel today (see ``generate_tool_call``),
        so under the standard backends every call lands on the salvage path. That
        is fine rather than accidental: G-Eval's prompt asks for JSON, and the
        models behind those backends comply -- the narrating direct SAGE agent is
        precisely the one that needed the tool channel.
        """
        tool = tool_for_schema(schema)
        call_args, text = generate_tool_call(
            self._config, prompt, tool, role=ROLE_QUALITY_JUDGE
        )
        if call_args is None:
            # The model narrated instead of calling the tool. Salvage the JSON it
            # very likely still wrote -- G-Eval's prompt asks for JSON in prose.
            logger.info("Judge narrated instead of calling %s; salvaging JSON", tool["name"])
            call_args = _salvage_json(text)
        if call_args is None:
            raise ValueError(
                f"Judge produced neither a {tool['name']} call nor parseable JSON "
                f"(len={len(text)}): {text[:300]!r}"
            )
        return schema.model_validate(call_args)

    async def a_generate(self, prompt: str, schema: "Optional[type[BaseModel]]" = None):
        # Plain ``to_thread``, deliberately NOT ``offload_to_worker_thread``.
        #
        # Both run ``generate`` off the calling loop; the difference is that
        # ``offload_to_worker_thread`` also declares "nested LLM calls belong on
        # MY loop" -- right for the ADK and web-app paths, where the agent's own
        # streaming call has already bound the backend session to that loop, and
        # wrong here. pytest-asyncio hands each async test a FRESH loop and closes
        # it afterwards, so pinning it would bind the process-global session to a
        # loop that dies with the test: the first judged metric would pass and
        # every later one would fail with "Event loop is closed".
        #
        # Declaring nothing lets ``shared/async_bridge.py`` put the call on its
        # dedicated, never-closing loop, which outlives every test in the file.
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self._config.sage_model if self._config.llm_backend == "sage" else self._config.model
