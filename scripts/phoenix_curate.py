"""
Curate a candidate eval dataset straight from Phoenix -- no manual span joining.

Feedback is emitted as two linked spans (see docs/architecture): a
``human_feedback`` span carrying the label + the decision's ``target_trace_id``,
and a ``webapp.recommendation`` span carrying the decision's ``input.value`` /
``output.value`` (when ``use_trace_dataset_payloads`` + scrub-off are on). Curating
"the thumbs-down cases" therefore means joining the two by trace id. This script
does that join and writes the SAME candidate-cases JSON schema that
``scripts/curate_feedback.py`` produces -- so it flows straight into
``eval/case_source.py`` / ``python3 -m eval.build_evalset --cases ...`` like any
other curated file. Optionally it also uploads a Phoenix Dataset for in-Phoenix
experiments.

The join/transform (:func:`spans_to_candidates`) is pure and unit-tested; only
``main`` touches Phoenix, and it imports the client lazily so this file is safe to
import without ``arize-phoenix`` installed.

Run (with a Phoenix instance reachable):
    python3 scripts/phoenix_curate.py --out eval/data/feedback_candidates.json
    python3 scripts/phoenix_curate.py --label thumbs_down --upload-dataset thumbs_down_cases
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# OpenInference / our own attribute keys, as flattened by Phoenix's span dataframe.
_A_LABEL = "attributes.smart_assignment.feedback.label"
_A_TARGET_TRACE = "attributes.smart_assignment.feedback.target_trace_id"
_A_NOTE = "attributes.smart_assignment.feedback.note"
_A_SCORE = "attributes.smart_assignment.feedback.score"
_A_SESSION = "attributes.smart_assignment.feedback.session_id"
_A_ANNOTATOR = "attributes.smart_assignment.feedback.annotator_id"
_A_INPUT = "attributes.input.value"
_A_OUTPUT = "attributes.output.value"
_A_OUTCOME = "attributes.smart_assignment.decision.outcome"
_C_TRACE = "context.trace_id"
_C_SPAN = "context.span_id"


def _as_obj(value: Any) -> Dict[str, Any]:
    """Parse a JSON string (or pass a dict through) into a dict; ``{}`` on failure."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _query_from_intake(intake: Dict[str, Any]) -> Optional[str]:
    """A natural-language intake message from the decision span's input payload."""
    parts: List[str] = []
    for key in ("name", "address"):
        val = str(intake.get(key) or "").strip()
        if val:
            parts.append(val)
    cases = intake.get("order_quantity_cases")
    if cases:
        parts.append(f"{cases} cases")
    day, window = intake.get("preferred_day"), intake.get("preferred_window")
    if day and window:
        parts.append(f"prefers {day} {window}")
    return ", ".join(parts) if parts else None


def spans_to_candidates(
    feedback_spans: List[Dict[str, Any]],
    decision_spans: List[Dict[str, Any]],
    *,
    label: str = "thumbs_down",
) -> List[Dict[str, Any]]:
    """Join feedback spans to their decision spans by trace id and build curated
    candidate records (the same schema as ``scripts/curate_feedback.py``).

    ``feedback_spans`` rows carry ``label`` / ``target_trace_id`` (+ optional
    note/score/session/annotator); ``decision_spans`` rows carry ``trace_id`` /
    ``span_id`` / ``input`` / ``output`` / ``outcome``. Only feedback rows matching
    ``label`` whose decision span has a parseable input are emitted; the rest are
    skipped (a decision with no replay payload can't be curated)."""
    dec_by_trace: Dict[str, Dict[str, Any]] = {}
    for dec in decision_spans:
        trace_id = dec.get("trace_id")
        if trace_id and _as_obj(dec.get("input")):
            dec_by_trace[trace_id] = dec

    candidates: List[Dict[str, Any]] = []
    for fb in feedback_spans:
        if (fb.get("label") or "") != label:
            continue
        dec = dec_by_trace.get(fb.get("target_trace_id"))
        if dec is None:
            continue
        intake = _as_obj(dec.get("input"))
        output = _as_obj(dec.get("output"))
        outcome = dec.get("outcome") or output.get("decision")
        trace_id = dec.get("trace_id")
        short = str(trace_id)[:8] if trace_id else "unknown"
        context = {
            "name": intake.get("name"),
            "address": intake.get("address"),
            "order_quantity_cases": intake.get("order_quantity_cases"),
            "preferred_day": intake.get("preferred_day"),
            "preferred_window": intake.get("preferred_window"),
            "outcome": outcome,
            "recommended_route_id": output.get("recommended_route_id"),
            "recommended_window": output.get("recommended_window"),
        }
        context = {k: v for k, v in context.items() if v is not None}
        candidates.append(
            {
                "eval_id": f"phoenix_{short}_negative",
                "query": _query_from_intake(intake),
                "observed_outcome": outcome,
                "human_verdict": "negative",
                "human_label": label,
                "human_score": fb.get("score"),
                "note": fb.get("note"),
                # Left for a human to set on promotion (a 👎 says wrong, not how).
                "suggested_expected_outcome": None,
                "context": context,
                "provenance": {
                    "trace_id": trace_id,
                    "span_id": dec.get("span_id"),
                    "session_id": fb.get("session_id"),
                    "annotator_id": fb.get("annotator_id"),
                },
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Phoenix I/O (lazy; only main touches it)
# ---------------------------------------------------------------------------


def _flatten_dict_columns(frame: Any) -> Any:
    """Unwrap any dict-valued column into dotted ``"<column>.<subkey>"`` columns.

    The Phoenix 19.x client's default span dataframe flattens *recognized*
    OpenInference attributes (``input.value``, ``output.value``, ...) into
    dotted columns, but buckets our own custom ``smart_assignment.*``
    namespace into a single dict-valued column (e.g. ``attributes.smart_assignment``
    holding ``{"feedback": {"label": ..., ...}}``) instead. Recursively
    normalizing those columns restores the dotted-column shape the rest of
    this module (and its ``_A_*`` attribute-path constants) expects.
    """
    import pandas as pd

    for col in list(frame.columns):
        if frame[col].map(lambda v: isinstance(v, dict)).any():
            nested = pd.json_normalize(frame[col], sep=".").add_prefix(f"{col}.")
            nested.index = frame.index
            frame = pd.concat([frame.drop(columns=[col]), nested], axis=1)
    return frame


def _rows(dataframe: Any, columns: Dict[str, str]) -> List[Dict[str, Any]]:
    """Extract the named ``{source_column: out_key}`` columns from a Phoenix span
    dataframe into plain dict rows, tolerating absent columns (-> None)."""
    records: List[Dict[str, Any]] = []
    # A filtered SpanQuery indexes the dataframe by "context.span_id", which the
    # client also returns as a plain column -- reset_index() would then try to
    # insert a column that already exists and raise. Drop the (redundant) index
    # instead of restoring it; every id we need is already a regular column.
    if dataframe.index.name is not None and dataframe.index.name in dataframe.columns:
        frame = dataframe.reset_index(drop=True)
    else:
        frame = dataframe.reset_index()
    frame = _flatten_dict_columns(frame)
    for _, row in frame.iterrows():
        record: Dict[str, Any] = {}
        for source, out_key in columns.items():
            value = row[source] if source in row and _present(row[source]) else None
            record[out_key] = value
        records.append(record)
    return records


def _present(value: Any) -> bool:
    """True unless the value is a pandas NaN/NA (which compares unequal to itself)."""
    try:
        return value == value  # noqa: PLR0124 - NaN != NaN is the point
    except Exception:  # noqa: BLE001
        return value is not None


def _fetch_candidates(project: str, endpoint: Optional[str], label: str) -> List[Dict[str, Any]]:
    """Query Phoenix for the two span sets and join them into candidate records."""
    # lazy: only needed for a live pull. Phoenix 19.x removed the legacy
    # top-level `phoenix.Client`; the client now lives at `phoenix.client.Client`
    # with namespaced resources (`client.spans`, `client.datasets`, ...).
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    client = Client(base_url=endpoint) if endpoint else Client()

    feedback_df = client.spans.get_spans_dataframe(
        query=SpanQuery().where("name == 'human_feedback'"), project_identifier=project
    )
    decision_df = client.spans.get_spans_dataframe(
        query=SpanQuery().where("name == 'webapp.recommendation'"), project_identifier=project
    )

    feedback_spans = _rows(
        feedback_df,
        {
            _A_LABEL: "label",
            _A_TARGET_TRACE: "target_trace_id",
            _A_NOTE: "note",
            _A_SCORE: "score",
            _A_SESSION: "session_id",
            _A_ANNOTATOR: "annotator_id",
        },
    )
    decision_spans = _rows(
        decision_df,
        {
            _C_TRACE: "trace_id",
            _C_SPAN: "span_id",
            _A_INPUT: "input",
            _A_OUTPUT: "output",
            _A_OUTCOME: "outcome",
        },
    )
    return spans_to_candidates(feedback_spans, decision_spans, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.environ.get("OTEL_SERVICE_NAME", "smart-assignment"),
        help="Phoenix project name (defaults to OTEL_SERVICE_NAME or 'smart-assignment').",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        help="Phoenix endpoint (e.g. http://localhost:6006). Defaults to the OTLP endpoint env.",
    )
    parser.add_argument("--label", default="thumbs_down", help="Feedback label to curate.")
    parser.add_argument(
        "--out",
        default="eval/data/feedback_candidates.json",
        help="Where to write the candidate-cases JSON (feeds eval/case_source.py).",
    )
    parser.add_argument(
        "--upload-dataset",
        default=None,
        help="Also upload a Phoenix Dataset of this name (for in-Phoenix experiments).",
    )
    args = parser.parse_args()

    try:
        candidates = _fetch_candidates(args.project, args.endpoint, args.label)
    except ModuleNotFoundError:
        raise SystemExit(
            "arize-phoenix is not installed. Install it (pip install arize-phoenix) "
            "or use the vendor-free path: python3 scripts/curate_feedback.py"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(candidates)} candidate case(s) from Phoenix to {args.out}")
    print(f"Build a runnable evalset:  python3 -m eval.build_evalset --cases {args.out}")

    if args.upload_dataset and candidates:
        _upload_dataset(args.upload_dataset, candidates, args.endpoint)


def _upload_dataset(name: str, candidates: List[Dict[str, Any]], endpoint: Optional[str]) -> None:
    """Best-effort Phoenix Dataset upload for in-Phoenix experiments (optional)."""
    try:
        import pandas as pd
        from phoenix.client import Client

        client = Client(base_url=endpoint) if endpoint else Client()
        frame = pd.DataFrame(
            [
                {
                    "input": c.get("query") or "",
                    "expected_outcome": c.get("observed_outcome"),
                    "human_label": c.get("human_label"),
                }
                for c in candidates
            ]
        )
        client.datasets.create_dataset(
            name=name,
            dataframe=frame,
            input_keys=["input"],
            output_keys=["expected_outcome"],
            metadata_keys=["human_label"],
        )
        print(f"Uploaded Phoenix dataset '{name}' with {len(frame)} example(s).")
    except Exception as exc:  # noqa: BLE001 - dataset upload is a convenience, never fatal
        logger.warning("Could not upload Phoenix dataset '%s': %s", name, exc)


if __name__ == "__main__":
    main()
