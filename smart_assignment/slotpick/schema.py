"""
The slot-choice output contract. The model returns::

    {
      "chosen_index": 1,
      "rationale": "why this candidate slot is best",
      "citations": [
        {"index": 1, "field": "preference_overlap_minutes", "value": 150}
      ]
    }

`parse_slot_choice` is strict about SHAPE only; whether the chosen index is
valid and the cited values are correct is the verifier's job (`verifier.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SlotChoiceParseError(ValueError):
    """Raised when a raw slot-choice dict can't be parsed into the schema."""


@dataclass(frozen=True)
class SlotCitation:
    index: int
    field: str
    value: float


@dataclass
class SlotChoice:
    chosen_index: int
    rationale: str
    citations: list[SlotCitation] = field(default_factory=list)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SlotChoiceParseError(message)


def _as_int(raw: object, where: str) -> int:
    if isinstance(raw, bool):
        raise SlotChoiceParseError(f"{where}: expected an integer, got a boolean")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return int(raw.strip())
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    raise SlotChoiceParseError(f"{where}: {raw!r} is not an integer")


def _as_float(raw: object, where: str) -> float:
    if isinstance(raw, bool):
        raise SlotChoiceParseError(f"{where}: expected a number, got a boolean")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().rstrip("%").strip()
        try:
            return float(token)
        except ValueError as exc:
            raise SlotChoiceParseError(f"{where}: {raw!r} is not numeric") from exc
    raise SlotChoiceParseError(f"{where}: expected a number, got {type(raw).__name__}")


def parse_slot_choice(raw: object) -> SlotChoice:
    _require(isinstance(raw, dict), "slot choice must be a JSON object")
    _require("chosen_index" in raw, "missing 'chosen_index'")
    chosen_index = _as_int(raw["chosen_index"], "chosen_index")

    rationale = str(raw.get("rationale", "")).strip()
    _require(bool(rationale), "rationale must be a non-empty string")

    citations: list[SlotCitation] = []
    for i, c in enumerate(raw.get("citations") or []):
        _require(isinstance(c, dict), f"citations[{i}] must be an object")
        for k in ("index", "field", "value"):
            _require(c.get(k) is not None, f"citations[{i}] missing {k!r}")
        citations.append(
            SlotCitation(
                index=_as_int(c["index"], f"citations[{i}].index"),
                field=str(c["field"]).strip(),
                value=_as_float(c["value"], f"citations[{i}].value"),
            )
        )

    return SlotChoice(chosen_index=chosen_index, rationale=rationale, citations=citations)
