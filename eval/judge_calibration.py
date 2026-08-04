"""
Phase 0 -- judge calibration: measure how well the automated LLM judges agree
with the human labels being collected, so the auto judges can be *trusted* (or
not) before anything is gated on them.

This is the "calibrate LLM evals" box of the flywheel, and the connective tissue
that makes the auto-judge scores meaningful: an LLM judge (``brief_quality`` /
``response_clarity`` in ``eval/test_quality.py``) is itself an LLM and can be
lenient, biased, or drift when the judge model changes -- so its scores are only
worth acting on once benchmarked against human ground truth.

**No replay, no data source.** Calibration needs only the ``(human_label,
judge_verdict)`` pairs on the same decision -- both of which already exist. It is
purely ADVISORY (``Config.use_judge_calibration``, default off): it changes no
decision and gates nothing; it only reports agreement.

**Holistic vs. dimensional (the crux).** Human feedback today is a *holistic*
thumb on the whole decision, while the judges score *dimensions* (clarity, brief
quality). This module never fabricates a per-judge label from a holistic thumb.
It tiers the signal by how much dimensional information it carries:

* **Tier 3 -- explicit dimension** (``label.dimensions``): a direct per-judge
  label, used 1:1. (Authored via the annotation sources; see later phases.)
* **Tier 2 -- note-tagged** (a ``note_tagger`` maps the free-text note to
  dimensions): a 👎 whose note says "confusing" calibrates ``response_clarity``.
* **Tier 1.5 -- outcome-routed holistic**: an *escalate* thumb routes to
  ``brief_quality``, a *recommend* thumb to ``response_clarity`` (the same
  outcome→judge split ``test_quality.py`` already uses).
* **Tier 1 -- composite**: predict a thumb from ALL judge verdicts (all pass ->
  👍) and calibrate that against the holistic thumb -- granularity-matched.

Each aligned pair is tagged with its ``source`` tier, so a report can show the
sharp (dimensional) agreement separately from the coarse (holistic) one.

The dimension names are exactly the human-annotation vocabulary in
``deployment/phoenix/README.md`` (also the judge names), so the human label, the
Phoenix annotation, and the automated judge all speak one language.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

# Canonical dimensions -- the human-annotation vocabulary AND the judge names
# (see deployment/phoenix/README.md and eval/test_quality.py).
DIM_DECISION_CORRECT = "decision_correct"
DIM_SLOT_REASONABLE = "slot_reasonable"
DIM_BRIEF_QUALITY = "brief_quality"
DIM_RESPONSE_CLARITY = "response_clarity"
DIMENSIONS = (DIM_DECISION_CORRECT, DIM_SLOT_REASONABLE, DIM_BRIEF_QUALITY, DIM_RESPONSE_CLARITY)

# The composite "predicted thumb" pseudo-dimension (Tier 1).
COMPOSITE = "__composite__"

# Outcome -> the judge a holistic thumb most plausibly speaks to (Tier 1.5),
# mirroring test_quality.py: brief_quality is scored on escalations, response_clarity
# on recommends.
_OUTCOME_JUDGE = {
    "escalate": DIM_BRIEF_QUALITY,
    "recommend": DIM_RESPONSE_CLARITY,
}

# A note tagger maps a free-text note to zero or more dimensions (Tier 2). Kept as
# a plain callable so Phase 2 can inject a keyword/LLM tagger without this module
# depending on it.
NoteTagger = Callable[[str], Set[str]]


def route_by_outcome(outcome: Optional[str]) -> Optional[str]:
    """The dimension a holistic thumb routes to, given the decision outcome, or
    ``None`` when the outcome is unknown (then only composite calibration applies)."""
    return _OUTCOME_JUDGE.get((outcome or "").strip().lower())


@dataclass(frozen=True)
class HumanLabel:
    """One human judgment on one decision. ``thumb`` is the holistic 👍/👎
    (``"up"``/``"down"``); ``dimensions`` carries any explicit Tier-3 per-dimension
    verdicts (dimension -> is-good). Only ``annotator_kind == "HUMAN"`` records are
    ground truth for calibration."""

    decision_id: str
    outcome: Optional[str] = None
    thumb: Optional[str] = None
    note: Optional[str] = None
    dimensions: Dict[str, bool] = field(default_factory=dict)
    annotator_kind: str = "HUMAN"


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's verdict on one decision: did dimension ``dimension`` pass?"""

    decision_id: str
    dimension: str
    passed: bool
    score: Optional[float] = None


@dataclass(frozen=True)
class Pair:
    """An aligned (human, judge) observation for one dimension on one decision.
    ``human_positive`` is the ground truth; ``judge_positive`` the prediction."""

    dimension: str
    decision_id: str
    human_positive: bool
    judge_positive: bool
    source: str  # "dimensional" | "note" | "holistic"
    note: Optional[str] = None


def human_dimension_signals(
    label: HumanLabel, note_tagger: Optional[NoteTagger] = None
) -> Iterator[Tuple[str, bool, str]]:
    """Yield ``(dimension, human_positive, source)`` for a human label, using the
    highest-fidelity tier available. Explicit dimensions win; else a note tag
    (Tier 2) if a tagger resolves one; else outcome-routing (Tier 1.5). A holistic
    thumb with an unknown outcome yields nothing here (it still feeds the composite).

    A holistic 👎 is *never* turned into a per-judge negative beyond the single
    routed/ tagged dimension -- so we never blame a judge for a failure that may
    belong to a different dimension."""
    if label.dimensions:
        for dim, positive in label.dimensions.items():
            yield dim, bool(positive), "dimensional"
        return
    if label.thumb is None:
        return
    positive = label.thumb == "up"
    if note_tagger is not None and label.note:
        tagged = note_tagger(label.note)
        if tagged:
            for dim in sorted(tagged):
                yield dim, positive, "note"
            return
    routed = route_by_outcome(label.outcome)
    if routed:
        yield routed, positive, "holistic"


def build_pairs(
    labels: Iterable[HumanLabel],
    verdicts: Iterable[JudgeVerdict],
    note_tagger: Optional[NoteTagger] = None,
) -> List[Pair]:
    """Join human signals to judge verdicts by ``(decision_id, dimension)``. Only
    HUMAN labels are used; a signal with no matching judge verdict is dropped."""
    index: Dict[Tuple[str, str], JudgeVerdict] = {
        (v.decision_id, v.dimension): v for v in verdicts
    }
    pairs: List[Pair] = []
    for label in labels:
        if label.annotator_kind != "HUMAN":
            continue
        for dim, human_pos, source in human_dimension_signals(label, note_tagger):
            verdict = index.get((label.decision_id, dim))
            if verdict is None:
                continue
            pairs.append(
                Pair(dim, label.decision_id, human_pos, verdict.passed, source, label.note)
            )
    return pairs


def composite_pairs(
    labels: Iterable[HumanLabel], verdicts: Iterable[JudgeVerdict]
) -> List[Pair]:
    """Tier-1 composite: predict a thumb from ALL of a decision's judge verdicts
    (all pass -> predicted 👍) and pair it with the holistic human thumb."""
    by_decision: Dict[str, List[JudgeVerdict]] = defaultdict(list)
    for verdict in verdicts:
        by_decision[verdict.decision_id].append(verdict)
    pairs: List[Pair] = []
    for label in labels:
        if label.annotator_kind != "HUMAN" or label.thumb is None:
            continue
        decision_verdicts = by_decision.get(label.decision_id)
        if not decision_verdicts:
            continue
        predicted_positive = all(v.passed for v in decision_verdicts)
        pairs.append(
            Pair(
                COMPOSITE,
                label.decision_id,
                label.thumb == "up",
                predicted_positive,
                "holistic",
                label.note,
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Pure metrics (human_positive = ground truth, judge_positive = prediction)
# ---------------------------------------------------------------------------


def confusion(pairs: Iterable[Pair]) -> Dict[str, int]:
    """Confusion counts. ``fp`` is the *dangerous* cell: the judge passed an output
    the human rejected -- a bad decision that would ship if you gated on the judge."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for pair in pairs:
        if pair.human_positive and pair.judge_positive:
            counts["tp"] += 1
        elif not pair.human_positive and pair.judge_positive:
            counts["fp"] += 1
        elif pair.human_positive and not pair.judge_positive:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def cohen_kappa(pairs: List[Pair]) -> Optional[float]:
    """Chance-corrected agreement. ``None`` for an empty set. Returns 1.0 when the
    observed and expected agreement coincide with perfect agreement (kappa is
    undefined when one rater is constant; we report 1.0 iff they fully agree,
    else 0.0, so a rubber-stamp judge on skewed labels scores ~0, not ~1)."""
    n = len(pairs)
    if n == 0:
        return None
    c = confusion(pairs)
    po = (c["tp"] + c["tn"]) / n
    p_judge_yes = (c["tp"] + c["fp"]) / n
    p_human_yes = (c["tp"] + c["fn"]) / n
    pe = p_judge_yes * p_human_yes + (1 - p_judge_yes) * (1 - p_human_yes)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def dangerous_cell_rate(pairs: Iterable[Pair]) -> Optional[float]:
    """Of the human-rejected outputs, the fraction the judge wrongly passed
    (``fp / (fp + tn)``). ``None`` when there are no human-negatives to speak of."""
    c = confusion(pairs)
    negatives = c["fp"] + c["tn"]
    return c["fp"] / negatives if negatives else None


def trust_band(kappa: Optional[float], n: int, min_n: int = 20) -> str:
    """A blunt recommendation from kappa + sample size: ``insufficient`` (too few
    labels to say), ``distrust`` (kappa < 0.2), ``advisory`` (0.2-0.6), or
    ``gate`` (>= 0.6, substantial agreement -- safe to gate CI on)."""
    if n < min_n or kappa is None:
        return "insufficient"
    if kappa >= 0.6:
        return "gate"
    if kappa >= 0.2:
        return "advisory"
    return "distrust"


def _disagreements(pairs: List[Pair], limit: int = 10) -> List[Dict[str, object]]:
    """The most informative disagreements, dangerous cell first (judge passed,
    human rejected), then judge-too-harsh -- each with its note + decision id, so
    they double as judge few-shot fixes and candidate eval cases."""
    fp = [p for p in pairs if p.judge_positive and not p.human_positive]
    fn = [p for p in pairs if not p.judge_positive and p.human_positive]
    ordered = fp + fn
    return [
        {
            "decision_id": p.decision_id,
            "kind": "judge_passed_human_rejected" if p in fp else "judge_failed_human_liked",
            "source": p.source,
            "note": p.note,
        }
        for p in ordered[:limit]
    ]


def _report_for(pairs: List[Pair], *, min_n: int = 20) -> Dict[str, object]:
    kappa = cohen_kappa(pairs)
    return {
        "n": len(pairs),
        "confusion": confusion(pairs),
        "cohen_kappa": kappa,
        "dangerous_cell_rate": dangerous_cell_rate(pairs),
        "trust": trust_band(kappa, len(pairs), min_n=min_n),
        "sources": sorted({p.source for p in pairs}),
        "top_disagreements": _disagreements(pairs),
    }


def calibrate(
    labels: Iterable[HumanLabel],
    verdicts: Iterable[JudgeVerdict],
    *,
    note_tagger: Optional[NoteTagger] = None,
    min_n: int = 20,
) -> Dict[str, object]:
    """Full calibration report: per-dimension agreement (Tier 1.5/2/3) plus the
    Tier-1 composite. Pure and deterministic -- give it the labels and the judge
    verdicts and it computes the agreement, with no I/O of its own."""
    labels = list(labels)
    verdicts = list(verdicts)
    dim_pairs = build_pairs(labels, verdicts, note_tagger)

    by_dim: Dict[str, List[Pair]] = defaultdict(list)
    for pair in dim_pairs:
        by_dim[pair.dimension].append(pair)

    dimensions = {dim: _report_for(pairs, min_n=min_n) for dim, pairs in sorted(by_dim.items())}
    composite = _report_for(composite_pairs(labels, verdicts), min_n=min_n)
    return {
        "dimensions": dimensions,
        "composite": composite,
        "totals": {
            "human_labels": len(labels),
            "judge_verdicts": len(verdicts),
            "aligned_pairs": len(dim_pairs),
        },
    }


# ---------------------------------------------------------------------------
# Vendor-free label source (the durable JSONL feedback log). Phoenix / Langfuse
# label sources are added in a later phase behind the same HumanLabel shape.
# ---------------------------------------------------------------------------

_THUMB = {"thumbs_up": "up", "thumbs_down": "down", "up": "up", "down": "down"}


def human_labels_from_feedback(path: str) -> List[HumanLabel]:
    """Read the vendor-free JSONL feedback log into ``HumanLabel``s (HUMAN only).
    Imports the feedback store lazily so this module stays import-light."""
    from smart_assignment.feedback.store import read_records

    labels: List[HumanLabel] = []
    for record in read_records(path):
        if record.annotator_kind != "HUMAN":
            continue
        context = record.context or {}
        labels.append(
            HumanLabel(
                decision_id=record.target.decision_id,
                outcome=context.get("outcome"),
                thumb=_THUMB.get((record.label or "").strip().lower()),
                note=record.note,
                annotator_kind=record.annotator_kind,
            )
        )
    return labels


def verdicts_from_jsonl(path: str) -> List[JudgeVerdict]:
    """Read the durable judge log (``eval/judge_log.py``) into ``JudgeVerdict``s.

    That log is append-only, so the same case re-judged on a later run appends a
    NEW line rather than replacing the old one. Calibration wants one verdict per
    ``(decision_id, dimension)`` -- the CURRENT judge's opinion -- so the latest
    line wins, the same "latest record per decision" rule ``feedback/curate.py``
    applies to the human log. Without it a case judged five times would weigh
    five times as much as one judged once.

    Imports the log reader lazily so this module stays import-light."""
    from eval.judge_log import iter_verdicts

    latest: Dict[Tuple[str, str], JudgeVerdict] = {}
    for record in iter_verdicts(path):
        if not record.decision_id or not record.dimension:
            continue
        latest[(record.decision_id, record.dimension)] = JudgeVerdict(
            decision_id=record.decision_id,
            dimension=record.dimension,
            passed=record.passed,
            score=record.score,
        )
    return list(latest.values())


def verdicts_from_mapping(mapping: Dict[str, Dict[str, object]]) -> List[JudgeVerdict]:
    """Parse a precomputed ``{decision_id: {dimension: {passed, score}}}`` mapping
    (e.g. produced by running the judges) into ``JudgeVerdict``s."""
    verdicts: List[JudgeVerdict] = []
    for decision_id, dims in (mapping or {}).items():
        for dimension, verdict in (dims or {}).items():
            if isinstance(verdict, dict):
                passed = bool(verdict.get("passed"))
                score = verdict.get("score")
            else:  # tolerate a bare bool/score
                passed = bool(verdict)
                score = None
            numeric = score if isinstance(score, (int, float)) else None
            verdicts.append(JudgeVerdict(decision_id, dimension, passed, numeric))
    return verdicts
