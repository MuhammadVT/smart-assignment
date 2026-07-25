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
the SAME function ``judgment/``, ``triage/``, ``slotpick/`` etc. already call
for every other grounded decision in this repo, which already branches on
``Config.llm_backend`` internally. No second, divergent judge-model resolution
path to maintain.

Backend-specific correctness notes, both already solved by reusing existing
seams rather than reinventing them:

* **Sage's loop-bound aiohttp session.** ``shared/llm.py``'s own docstring
  documents that the Sage SDK's aiohttp ``ClientSession`` is bound to the
  FIRST event loop that touches it; a naive ``asyncio.to_thread`` per call
  would spin up a fresh throwaway loop each time (via ``_run_coro_blocking``'s
  own fallback) and could break on the second live call. ``a_generate`` below
  uses ``offload_to_worker_thread`` -- the SAME mechanism the web app's own
  tools use to call ``generate_text`` from async code -- which records the
  CALLING coroutine's loop as the stable "host loop" so every nested sage call
  made from within one async test function lands back on the same loop.
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

import os
from typing import TYPE_CHECKING

# Must be set before ANY `deepeval` import, in this module or any caller's --
# deepeval reads both at import time (see module docstring above). Owned HERE,
# at the actual import boundary, rather than relying on every caller (e.g. a
# test file) to set them first: importing this module is what pulls deepeval
# in, so this module is what must make that safe.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402

from smart_assignment.shared.config import ROLE_QUALITY_JUDGE  # noqa: E402
from smart_assignment.shared.llm import generate_text, offload_to_worker_thread  # noqa: E402

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config


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

    def generate(self, prompt: str) -> str:
        return generate_text(self._config, prompt, role=ROLE_QUALITY_JUDGE)

    async def a_generate(self, prompt: str) -> str:
        return await offload_to_worker_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self._config.sage_model if self._config.llm_backend == "sage" else self._config.model
