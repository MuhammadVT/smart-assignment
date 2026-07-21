"""
Locking the eval dataset: declaration, no-fallback, and provenance.

An eval result depends on more than the agent code -- it depends on *which*
dataset the agent ran against: route capacity, tiers, delivery windows, and how
a street address geocodes. Historically eval inherited the ambient
``SMART_ASSIGNMENT_DATA_SOURCE`` (default ``cache``, read from an *uncommitted*
``data/dev/`` parquet snapshot) and silently fell back to the mock demo routes
when that snapshot was absent. So the same golden case could score against
different data on two machines -- or against ``mock`` in CI and real data on a
laptop -- with nothing recording which. That makes an eval score
irreproducible and a regression unattributable.

This module makes the eval dataset a first-class, *declared, versioned* input:

* **Declared, not defaulted.** Eval selects its dataset via
  ``SMART_ASSIGNMENT_EVAL_DATASET`` (default ``mock``, the committed offline
  world), *independent* of whatever ambient ``SMART_ASSIGNMENT_DATA_SOURCE`` a
  developer happens to have set for the app. An unknown name is a loud error,
  never a silent guess.
* **No silent substitution.** Locking turns on strict mode
  (``SMART_ASSIGNMENT_DATA_SOURCE_STRICT``), so a declared dataset that can't
  load *fails loudly* (see ``route_capacity_client``) instead of quietly
  becoming the mock routes.
* **Provenance.** Each captured response records the dataset identity (name +
  a content hash) and the resolved backend/model it was produced with, so a
  later score change is attributable to a data/model change, not guessed.

The design is deliberately dataset-agnostic: ``mock``, a scrubbed-synthetic
snapshot, or a sanitized-real snapshot are all just eval datasets that flow
through the same declare / lock / record path. Switching between them is a
*value* (``SMART_ASSIGNMENT_EVAL_DATASET=<name>``), not a code change -- add an
entry to ``_KNOWN_DATASETS``, never a new branch at a call site.

Imports stay credential-free and light: the heavy imports (route client,
customer fixtures, config) are lazy *inside* the functions, so importing this
module -- which the eval conftest and the hermetic lock test both do -- never
needs a backend, credentials, or pandas.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

# The declared-dataset selector (this module's own knob) and its default.
EVAL_DATASET_ENV = "SMART_ASSIGNMENT_EVAL_DATASET"
DEFAULT_EVAL_DATASET = "mock"

# The underlying data/geocoder/strict env vars this module drives so eval is
# pinned regardless of a developer's ambient settings. Kept as literals (not
# imported from the integrations layer) so this module stays import-light.
_DATA_SOURCE_ENV = "SMART_ASSIGNMENT_DATA_SOURCE"
_GEOCODER_ENV = "SMART_ASSIGNMENT_GEOCODER"
_STRICT_ENV = "SMART_ASSIGNMENT_DATA_SOURCE_STRICT"


@dataclass(frozen=True)
class EvalDataset:
    """A declared eval dataset: the name it's declared under, plus the concrete
    data source and geocoder it pins.

    ``kind`` distinguishes a *code-defined* dataset (``"code"`` -- e.g. ``mock``,
    whose content lives in the repo's Python) from a *file-backed* one
    (``"snapshot"``); ``dataset_content_ref`` reflects that when it computes the
    provenance hash."""

    name: str
    data_source: str
    geocoder: str
    kind: str


# The datasets eval knows how to lock onto. Extend THIS map (not any call site)
# to add a scrubbed-synthetic or sanitized-real snapshot: point it at that
# source + geocoder and give it a name; the declare/lock/record machinery is
# unchanged. Only ``mock`` exists today (fully committed, offline, PII-free).
_KNOWN_DATASETS: Dict[str, EvalDataset] = {
    "mock": EvalDataset(name="mock", data_source="mock", geocoder="mock", kind="code"),
}


def resolve_eval_dataset() -> EvalDataset:
    """The declared eval dataset (``SMART_ASSIGNMENT_EVAL_DATASET``, default
    ``mock``). Raises ``ValueError`` on an unknown name -- listing the valid set
    -- rather than silently picking one."""
    name = (os.environ.get(EVAL_DATASET_ENV) or DEFAULT_EVAL_DATASET).strip()
    dataset = _KNOWN_DATASETS.get(name)
    if dataset is None:
        raise ValueError(
            f"{EVAL_DATASET_ENV}={name!r} is not a known eval dataset. "
            f"Valid: {sorted(_KNOWN_DATASETS)}."
        )
    return dataset


def apply_eval_dataset() -> EvalDataset:
    """Resolve the declared eval dataset and LOCK this process onto it: pin the
    data source and geocoder, and turn on strict mode so a declared-but-unloadable
    dataset fails loudly rather than silently falling back to the mock routes.
    Returns the dataset so callers can record its provenance.

    Loud and idempotent -- it logs exactly what it pinned, so an eval run can
    never quietly inherit a developer's ambient ``SMART_ASSIGNMENT_DATA_SOURCE``."""
    dataset = resolve_eval_dataset()
    os.environ[_DATA_SOURCE_ENV] = dataset.data_source
    os.environ[_GEOCODER_ENV] = dataset.geocoder
    os.environ[_STRICT_ENV] = "1"
    # Drop any route cache populated before the pin (e.g. an ambient source read
    # at import elsewhere) so the next fetch honors the pinned source. Best-effort
    # and lazy so importing this module never pulls the heavy route client.
    try:
        from smart_assignment.integrations.route_capacity_client import clear_route_cache

        clear_route_cache()
    except Exception:  # pragma: no cover - cache clear is best-effort
        pass
    logger.info(
        "eval dataset locked: name=%s data_source=%s geocoder=%s (strict: no silent fallback)",
        dataset.name,
        dataset.data_source,
        dataset.geocoder,
    )
    return dataset


def dataset_content_ref(dataset: EvalDataset) -> str:
    """A stable content hash identifying the dataset's *actual data*, so two
    captures made against the same dataset content share a ref and any change to
    the data changes it.

    For the code-defined ``mock`` dataset this hashes the committed fixtures the
    agent runs against (the sample customers + the mock routes); a file-backed
    snapshot dataset would instead hash its bytes. Called only at capture time
    (never at import), so its heavy imports are lazy."""
    if dataset.kind == "code" and dataset.name == "mock":
        from smart_assignment.integrations.route_capacity_client import _mock_routes
        from smart_assignment.mock_customers import SAMPLE_CUSTOMERS

        # repr of frozen/field-based dataclasses is deterministic (no addresses),
        # so this hash is stable across processes and only moves when the fixture
        # data itself changes.
        payload = repr(list(SAMPLE_CUSTOMERS)) + repr(_mock_routes())
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    raise NotImplementedError(  # pragma: no cover - no snapshot datasets exist yet
        f"No content ref implemented for dataset {dataset.name!r} (kind {dataset.kind!r})."
    )


def run_provenance(dataset: EvalDataset) -> Dict[str, object]:
    """The full ``captured_with`` provenance block recorded next to each captured
    response: the dataset identity (name, kind, content ref) plus the resolved
    backend and model the responses were produced with -- so a later score change
    is *attributable* (data? model? code?) instead of a mystery.

    Lazy config import keeps this credential-free to import."""
    from smart_assignment.shared.config import ROLE_ROOT_AGENT, Config

    config = Config.from_env()
    return {
        "dataset": {
            "name": dataset.name,
            "kind": dataset.kind,
            "ref": dataset_content_ref(dataset),
        },
        "backend": config.llm_backend,
        "model": config.resolved_model(ROLE_ROOT_AGENT),
    }
