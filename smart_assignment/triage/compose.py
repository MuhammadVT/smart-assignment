"""
Non-agent escalation-brief composition -- the batch sibling of ``triage/agent.py``.

The conversational path authors the brief with an ADK ``LlmAgent`` driven through
an ``AgentTool`` and bound to session state. Batch has no agent and no session, so
this composes the *same* brief with a single grounded ``generate_text`` call --
the ``routeslot`` pattern -- then reuses the exact deterministic finalization
(``normalize_brief`` layout + ``verify_brief`` grounding scan), and falls back to a
deterministic brief on any failure.

Guarantees, matched to the rest of the repo:
  - **Never worse than a deterministic floor.** ``_deterministic_brief`` is always
    available (and is the fallback), assembled from the escalation context's raw,
    already-grounded facts -- so an escalation record is *always* actionable, even
    with no LLM, no credentials, or a verification failure.
  - **No fabricated actionable values.** The brief is explanation only; it changes
    no route, score, slot, or decision. Every figure in the *grounded* brief is
    checked by ``verify_brief`` and any ungrounded token is caveated (advisory,
    exactly as the chat after-model backstop does) -- never silently kept.
  - **Opt-in / credential-free import.** Honors ``Config.use_escalation_triage``;
    ``generate_text`` is imported lazily inside the grounded path, so importing
    this module needs no backend.
"""

from __future__ import annotations

import json
import logging

from smart_assignment.shared.config import ROLE_TRIAGE, Config
from smart_assignment.triage.formatting import normalize_brief
from smart_assignment.triage.prompts import BATCH_TRIAGE_PROMPT_TEMPLATE
from smart_assignment.triage.verifier import collect_grounding, verify_brief

logger = logging.getLogger(__name__)

_CONTEXT_SENTINEL = "__CONTEXT_JSON__"


def compose_brief(context: dict, config: Config) -> str:
    """A specialist brief for an escalation ``context`` (from
    ``build_escalation_context``), composed WITHOUT the ADK agent.

    Grounded when ``config.use_escalation_triage`` is on, else the deterministic
    floor. Any LLM/verify/transport failure falls back to the floor, logged. Returns
    an empty string when there is nothing to triage (a defensive guard; callers only
    invoke this on an escalation).
    """
    if not context or not context.get("ok"):
        return ""
    if not config.use_escalation_triage:
        return _deterministic_brief(context)
    try:
        return _grounded_brief(context, config)
    except Exception:  # noqa: BLE001 - never worse than the deterministic floor
        logger.warning("Batch triage brief fell back to deterministic.", exc_info=True)
        return _deterministic_brief(context)


# --- grounded path ----------------------------------------------------------


def _grounded_brief(context: dict, config: Config) -> str:
    """One grounded ``generate_text`` call, then deterministic normalize + verify,
    with a single corrective retry before an advisory caveat (mirroring the chat
    self-check loop and ``routeslot``'s one retry). Raises on an empty model reply
    so the caller falls back to the deterministic floor."""
    from smart_assignment.shared.llm import generate_text

    grounding = collect_grounding(context)
    prompt = BATCH_TRIAGE_PROMPT_TEMPLATE.replace(_CONTEXT_SENTINEL, _context_json(context))

    draft = generate_text(config.for_role(ROLE_TRIAGE), prompt, role=ROLE_TRIAGE)
    if not draft or not draft.strip():
        raise ValueError("model returned an empty triage brief")

    final = normalize_brief(draft)
    result = verify_brief(final, grounding)
    if result.ok:
        return final

    # One corrective retry with the failed tokens as feedback -- do not invent
    # replacements, quote the context verbatim.
    retry_prompt = (
        prompt
        + "\n\nYOUR PREVIOUS DRAFT (rejected):\n"
        + draft
        + "\n\n"
        + result.caveat()
        + "\nRewrite the brief so every figure, route, day, and time appears "
        "VERBATIM in the ESCALATION CONTEXT above. Drop any claim whose figure "
        "was flagged rather than inventing a replacement. Output only the brief."
    )
    retry = generate_text(config.for_role(ROLE_TRIAGE), retry_prompt, role=ROLE_TRIAGE)
    if retry and retry.strip():
        final = normalize_brief(retry)
        result = verify_brief(final, grounding)

    if not result.ok:
        # Advisory caveat, exactly as the chat after-model backstop does -- the
        # brief is human-reviewed in the Customer View, so we annotate rather than
        # drop it.
        logger.warning(
            "Batch triage brief still ungrounded after retry (numbers %s / routes %s); "
            "appending a caveat.",
            result.ungrounded_numbers,
            result.ungrounded_routes,
        )
        final = f"{final}\n\n{result.caveat()}"
    return final


def _context_json(context: dict) -> str:
    """The escalation context as pretty JSON for the prompt, minus the ``ok`` flag."""
    payload = {key: value for key, value in context.items() if key != "ok"}
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --- deterministic floor ----------------------------------------------------


def _deterministic_brief(context: dict) -> str:
    """The always-available floor: the canonical brief layout assembled from the
    context's raw facts only -- no LLM, no invented advice. It presents the facts a
    specialist needs (why it escalated, and each candidate route's state), not a
    fabricated action, so it is grounded by construction.

    Reflowed through ``normalize_brief`` so it shares the one canonical layout with
    the grounded brief."""
    customer = context.get("customer") or {}
    name = customer.get("name") or "This prospect"
    order = customer.get("order_quantity_cases")
    order_phrase = f"{order} cases" if order is not None else "an order of unstated size"

    situation = f"{name} — {order_phrase} — escalated for specialist review."
    root_cause = context.get("review_reason") or (
        "No feasible route-slot cleared the auto-assign criteria."
    )

    options = _deterministic_options(context)
    options_block = (
        "\n".join(options)
        if options
        else "1) No candidate route is available for this prospect."
    )

    # Deliberately avoids the section-header keywords (SITUATION / ROOT CAUSE /
    # OPTIONS / RECOMMENDATION / DECISION) in the body prose: normalize_brief matches
    # those case-insensitively and would reflow e.g. a bare "options" onto its own
    # line, corrupting the layout.
    recommendation = (
        "Review the routes above; none cleared the auto-assign bar, so nothing was "
        "assigned automatically."
    )
    decision_needed = (
        "Which route-slot (if any) should take this prospect, or should it stay "
        "unassigned?"
    )

    brief = (
        f"SITUATION\n{situation}\n\n"
        f"ROOT CAUSE\n{root_cause}\n\n"
        f"OPTIONS (most workable first)\n{options_block}\n\n"
        f"RECOMMENDATION\n{recommendation}\n\n"
        f"DECISION NEEDED\n{decision_needed}"
    )
    return normalize_brief(brief)


def _deterministic_options(context: dict) -> list[str]:
    """Numbered option lines: feasible routes first (best reference score first),
    then infeasible ones with their failure reasons -- most workable first."""
    feasible = list(context.get("feasible_candidates") or [])
    infeasible = list(context.get("infeasible_candidates") or [])
    feasible.sort(
        key=lambda c: (c.get("facts") or {}).get("reference_weighted_score") or 0.0,
        reverse=True,
    )

    lines: list[str] = []
    index = 1
    for cand in feasible:
        lines.append(_feasible_option_line(index, cand))
        index += 1
    for cand in infeasible:
        lines.append(_infeasible_option_line(index, cand))
        index += 1
    return lines


def _feasible_option_line(index: int, cand: dict) -> str:
    facts = cand.get("facts") or {}
    state_bits: list[str] = []
    util = facts.get("utilization_after")
    if util is not None:
        state_bits.append(f"utilization {util * 100:.0f}%")
    remaining = facts.get("remaining_capacity_after")
    if remaining is not None:
        state_bits.append(f"{remaining} cases headroom")
    state = ", ".join(state_bits) if state_bits else "feasible"
    return f"{index}) {_route_label(cand)} · {cand.get('day') or ''} — {state}"


def _infeasible_option_line(index: int, cand: dict) -> str:
    reasons = "; ".join(
        (fc.get("detail") or fc.get("name") or "")
        for fc in (cand.get("failed_constraints") or [])
    ).strip("; ")
    reasons = reasons or "not feasible"
    return f"{index}) {_route_label(cand)} · {cand.get('day') or ''} — {reasons}"


def _route_label(cand: dict) -> str:
    """``"<route_id> - <route_name>"`` -- the id and human name always together, the
    naming convention used everywhere a route is shown to a human."""
    rid = cand.get("route_id") or ""
    name = cand.get("name") or ""
    if rid and name:
        return f"{rid} - {name}"
    return rid or name or "route"
