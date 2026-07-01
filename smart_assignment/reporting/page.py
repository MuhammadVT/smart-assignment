"""
Generate the GitHub Pages overview site (``docs/index.html``) directly from
live workflow output, so the examples on the page can never drift from the
code.

Everything shown for each customer — the decision, confidence, proposed slot,
weighted factor breakdown, reasoning, and per-route constraint trace — comes
from running the real pipeline over ``mock_customers.SAMPLE_CUSTOMERS``. The
static chrome (hero, workflow steps, rules) and the configurable numbers
(factor weights, confidence threshold, Top-N) are rendered from the same
``Config`` the workflow uses.

Reasoning uses the DeterministicReasoner so the page is reproducible offline
and regenerating with no code change produces no diff.

CLI: ``python3 scripts/generate_page.py`` (see that thin wrapper).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.config import (
    DEFAULT_CONFIG,
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.models import (
    CandidateEvaluation,
    Decision,
    RecommendationResult,
)
from smart_assignment.shared.timeutils import fmt_window
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation
from smart_assignment.workflows.slot_recommendation.reasoning import DeterministicReasoner

# Repo root -> docs/index.html
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "index.html"

FACTOR_LABEL = {
    FACTOR_GEO_CLUSTERING: "Geographic clustering",
    FACTOR_CAPACITY_BUFFER: "Capacity buffer",
    FACTOR_WINDOW_MATCH: "Window match",
}
CONSTRAINT_LABEL = {
    "geographic_serviceability": "serviceability",
    "route_capacity": "capacity",
    "delivery_window_compatibility": "window",
}
DECISION_PILL = {
    Decision.RECOMMENDED: ("rec", "✔ Recommended"),
    Decision.ESCALATED_LOW_CONFIDENCE: ("low", "⚠ Low confidence — human review"),
    Decision.ESCALATED_NO_FEASIBLE_SLOT: ("no", "✖ No feasible slot — specialist"),
}
CONF_CLASS = {
    Decision.RECOMMENDED: "g",
    Decision.ESCALATED_LOW_CONFIDENCE: "a",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "r",
}
CONF_TEXT_COLOR = {
    Decision.RECOMMENDED: "var(--green)",
    Decision.ESCALATED_LOW_CONFIDENCE: "var(--amber)",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "var(--red)",
}

_STYLE = """
  :root {
    --navy: #0b2e59; --blue: #1257a6; --blue-soft: #e8f0fb; --ink: #1a2233;
    --muted: #5b6675; --line: #e3e8ef; --bg: #f6f8fb; --card: #ffffff;
    --green: #1a7f37; --green-soft: #e7f4ea; --amber: #9a6700; --amber-soft: #fdf3d8;
    --red: #b42318; --red-soft: #fbe9e7; --radius: 14px;
    --shadow: 0 1px 2px rgba(16,32,64,.06), 0 6px 20px rgba(16,32,64,.06);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; color: var(--ink); background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55; }
  a { color: var(--blue); }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
    letter-spacing: .02em; padding: 5px 11px; border-radius: 999px; background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3); }
  header.hero { background: linear-gradient(135deg, var(--navy), var(--blue)); color: #fff; padding: 54px 0 60px; }
  .hero h1 { margin: 14px 0 8px; font-size: 40px; line-height: 1.1; letter-spacing: -.02em; }
  .hero p.lead { margin: 0; font-size: 18px; max-width: 640px; color: #dbe6f5; }
  .hero .meta { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 10px; }
  section { padding: 48px 0; }
  section h2 { font-size: 26px; letter-spacing: -.01em; margin: 0 0 6px; }
  section .sub { color: var(--muted); margin: 0 0 26px; max-width: 720px; }
  .eyebrow { color: var(--blue); font-weight: 700; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
  .flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
  .step { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 18px 16px; box-shadow: var(--shadow); }
  .step .num { width: 30px; height: 30px; border-radius: 8px; background: var(--blue-soft); color: var(--blue);
    display: grid; place-items: center; font-weight: 700; font-size: 14px; margin-bottom: 10px; }
  .step h3 { margin: 0 0 4px; font-size: 15px; }
  .step p { margin: 0; font-size: 13px; color: var(--muted); }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 20px; box-shadow: var(--shadow); }
  .card .icon { font-size: 22px; }
  .card h3 { margin: 8px 0 4px; font-size: 16px; }
  .card p { margin: 0; font-size: 14px; color: var(--muted); }
  .tag { display: inline-block; font-size: 11px; font-weight: 700; color: var(--blue);
    background: var(--blue-soft); padding: 3px 8px; border-radius: 6px; margin-top: 10px; }
  .bar { height: 8px; border-radius: 6px; background: #eef1f6; overflow: hidden; margin-top: 8px; }
  .bar > span { display: block; height: 100%; background: var(--blue); border-radius: 6px; }
  .legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .pill { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px;
    padding: 6px 12px; border-radius: 999px; }
  .pill.rec { background: var(--green-soft); color: var(--green); }
  .pill.low { background: var(--amber-soft); color: var(--amber); }
  .pill.no  { background: var(--red-soft); color: var(--red); }
  .examples { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
  .result { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); overflow: hidden; }
  .result .rhead { padding: 18px 20px; border-bottom: 1px solid var(--line); }
  .result .cnum { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); }
  .result h3 { margin: 2px 0 2px; font-size: 18px; }
  .result .addr { font-size: 13px; color: var(--muted); }
  .result .facts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .fact { font-size: 12px; background: #f1f4f9; color: #33415c; border-radius: 7px; padding: 4px 9px; }
  .result .rbody { padding: 18px 20px; }
  .decision-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .slot { margin: 12px 0 4px; font-size: 15px; }
  .slot b { color: var(--navy); }
  .conf { margin-top: 4px; }
  .conf .bar { height: 10px; }
  .conf .bar > span.g { background: var(--green); }
  .conf .bar > span.a { background: var(--amber); }
  .conf .bar > span.r { background: var(--red); }
  .conf small { color: var(--muted); font-size: 12px; }
  .factors { margin-top: 14px; display: grid; gap: 9px; }
  .factor { display: grid; grid-template-columns: 150px 1fr 42px; align-items: center; gap: 10px; font-size: 12.5px; }
  .factor .fname { color: #33415c; }
  .factor .fval { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .reason { margin-top: 14px; background: var(--blue-soft); border-left: 3px solid var(--blue);
    border-radius: 8px; padding: 12px 14px; font-size: 13.5px; color: #22364f; }
  .reason .lbl { font-weight: 700; color: var(--blue); font-size: 11px; letter-spacing: .06em;
    text-transform: uppercase; display: block; margin-bottom: 4px; }
  details.routes { margin-top: 14px; border-top: 1px dashed var(--line); padding-top: 10px; }
  details.routes summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--blue); }
  .routelist { margin-top: 10px; display: grid; gap: 8px; }
  .route { border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; font-size: 12.5px; }
  .route .rtitle { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
  .route .status { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 5px; }
  .route .status.ok { background: var(--green-soft); color: var(--green); }
  .route .status.bad { background: var(--red-soft); color: var(--red); }
  .checks { margin: 6px 0 0; padding-left: 0; list-style: none; color: var(--muted); }
  .checks li { padding: 1px 0; }
  .checks li .ok { color: var(--green); font-weight: 700; }
  .checks li .no { color: var(--red); font-weight: 700; }
  .note { background: #fff8ec; border: 1px solid #f2e2bd; border-radius: var(--radius);
    padding: 18px 20px; color: #5c4b1f; font-size: 14px; }
  footer { padding: 34px 0 60px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); background: #fff; }
  footer a { text-decoration: none; }
  @media (max-width: 860px) {
    .flow { grid-template-columns: repeat(2, 1fr); }
    .grid-3, .legend { grid-template-columns: 1fr; }
    .examples { grid-template-columns: 1fr; }
    .hero h1 { font-size: 32px; }
    .factor { grid-template-columns: 120px 1fr 40px; }
  }
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _win(window_str: Optional[str]) -> str:
    """Render a window string with an en-dash, or 'any' if none."""
    if not window_str or window_str == "n/a":
        return "any"
    return window_str.replace("-", "–")


def _pref_fact(result: RecommendationResult) -> str:
    w = result.customer.preferred_window
    return f"prefers {_win(fmt_window(w))}" if w else "no window preference"


def _factor_rows(rec) -> str:
    rows = []
    for f in rec.factor_breakdown:
        label = FACTOR_LABEL.get(f.name, f.name)
        pct = round(f.value * 100)
        rows.append(
            f'<div class="factor"><span class="fname">{_esc(label)}</span>'
            f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<span class="fval">{f.value:.2f}</span></div>'
        )
    return "".join(rows)


def _route_rows(candidates: list[CandidateEvaluation]) -> str:
    rows = []
    for cand in candidates:
        r = cand.route
        if cand.feasible:
            status_cls, status = "ok", f"FEASIBLE · {cand.total_score:.2f}"
        else:
            status_cls, status = "bad", "INFEASIBLE"
        checks = []
        for oc in cand.constraint_outcomes:
            sym = '<span class="ok">✓</span>' if oc.passed else '<span class="no">✗</span>'
            label = CONSTRAINT_LABEL.get(oc.name, oc.name)
            checks.append(f"<li>{sym} {_esc(label)}: {_esc(oc.detail)}</li>")
        title = f"{_esc(r.route_id)} · {_esc(r.name)} · {r.day.value} · {cand.distance_miles:.1f} mi"
        rows.append(
            f'<div class="route"><div class="rtitle"><span>{title}</span>'
            f'<span class="status {status_cls}">{status}</span></div>'
            f'<ul class="checks">{"".join(checks)}</ul></div>'
        )
    return "".join(rows)


def _example_card(result: RecommendationResult) -> str:
    c = result.customer
    rec = result.recommendation
    pill_cls, pill_text = DECISION_PILL[rec.decision]
    conf_cls = CONF_CLASS[rec.decision]
    conf_color = CONF_TEXT_COLOR[rec.decision]
    n_routes = len(result.candidates_considered)

    # Confidence bar: no-feasible has no meaningful winner, so show a full red block.
    if rec.decision == Decision.ESCALATED_NO_FEASIBLE_SLOT:
        conf_bar = (
            f'<div class="bar"><span class="{conf_cls}" style="width:100%"></span></div>'
            f"<small>no candidate passed the hard rules</small>"
        )
    else:
        note = "<small>below the confidence threshold</small>" if conf_cls == "a" else ""
        conf_bar = (
            f'<div class="bar"><span class="{conf_cls}" style="width:{round(rec.confidence * 100)}%">'
            f"</span></div>{note}"
        )

    slot_html = ""
    if rec.recommended_route_id:
        prefix = "→ proposed: " if rec.requires_human_review else "→ "
        score = result.ranked_feasible[0].total_score if result.ranked_feasible else 0.0
        slot_html = (
            f'<p class="slot">{prefix}<b>{_esc(rec.recommended_route_id)} · '
            f"{_esc(rec.recommended_route_name)}</b> · {_esc(rec.recommended_day)} · "
            f'<b>{_win(rec.recommended_window)}</b> '
            f'<small style="color:var(--muted)">(score {score:.2f})</small></p>'
        )

    factors_html = (
        f'<div class="factors">{_factor_rows(rec)}</div>' if rec.factor_breakdown else ""
    )
    reason_label = "Why it escalated" if rec.requires_human_review else "Why this slot"

    return f"""
      <article class="result">
        <div class="rhead">
          <div class="cnum">{_esc(c.customer_number)}</div>
          <h3>{_esc(c.name)}</h3>
          <div class="addr">{_esc(c.address)}</div>
          <div class="facts">
            <span class="fact">{c.order_quantity_cases} cases</span>
            <span class="fact">{_esc(_pref_fact(result))}</span>
            <span class="fact">{n_routes} routes evaluated</span>
          </div>
        </div>
        <div class="rbody">
          <div class="decision-row">
            <span class="pill {pill_cls}">{pill_text}</span>
            <span style="font-size:13px;color:var(--muted)">confidence
              <b style="color:{conf_color}">{rec.confidence:.0%}</b></span>
          </div>
          <div class="conf">{conf_bar}</div>
          {slot_html}
          {factors_html}
          <div class="reason"><span class="lbl">{reason_label}</span>{_esc(rec.reasoning)}</div>
          <details class="routes"><summary>Routes evaluated ({n_routes})</summary>
            <div class="routelist">{_route_rows(result.candidates_considered)}</div>
          </details>
        </div>
      </article>"""


def _scoring_cards(config: Config) -> str:
    meta = {
        FACTOR_GEO_CLUSTERING: (
            "\U0001f9ed",
            "Geographic clustering",
            "How tightly the customer clusters with the route's existing stops — "
            "tighter clusters mean more predictable arrivals.",
        ),
        FACTOR_CAPACITY_BUFFER: (
            "\U0001f6e1️",
            "Capacity buffer",
            "How much headroom is left after the add — more buffer means a route "
            "that's resilient to day-of disruption.",
        ),
        FACTOR_WINDOW_MATCH: (
            "\U0001f3af",
            "Window match",
            "How well the route's window covers the customer's stated preference.",
        ),
    }
    cards = []
    for key in (FACTOR_GEO_CLUSTERING, FACTOR_CAPACITY_BUFFER, FACTOR_WINDOW_MATCH):
        icon, title, desc = meta[key]
        weight = config.factor_weights[key]
        cards.append(
            f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3>'
            f"<p>{desc}</p>"
            f'<div class="bar"><span style="width:{round(weight * 100)}%"></span></div>'
            f'<span class="tag">weight {weight:.2f}</span></div>'
        )
    return "".join(cards)


def build_page(results: list[RecommendationResult], config: Config) -> str:
    """Render the full overview HTML from live workflow results."""
    threshold = f"{config.confidence_threshold:.0%}"
    cards = "".join(_example_card(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Smart Assignment — Delivery Slot Recommendation</title>
<meta name="description" content="How the Smart Assignment agent recommends delivery slots for new Sysco customers, with live examples on mock data." />
<!-- GENERATED by scripts/generate_page.py from live workflow output. Do not edit by hand. -->
<style>{_STYLE}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <span class="chip">\U0001f69a Sysco · Foodservice Distribution</span>
    <h1>Smart Assignment</h1>
    <p class="lead">An agentic workflow that recommends the best delivery day &amp; time slot for a
      <strong>new customer</strong> — enforcing hard operational rules in code, ranking the
      survivors on weighted business factors, and escalating to a human when the call is close.</p>
    <div class="meta">
      <span class="chip">✅ Deterministic, auditable decisions</span>
      <span class="chip">\U0001f916 LLM-written reasoning</span>
      <span class="chip">\U0001f9ea Running on mock data</span>
    </div>
  </div>
</header>

<section>
  <div class="wrap">
    <span class="eyebrow">The problem</span>
    <h2>Onboarding a new account, every time, the same way</h2>
    <p class="sub">When a new foodservice customer signs on, someone has to decide which delivery route
      and time window they should join. That decision balances truck capacity, geography, and the
      customer's preference — and it needs to be consistent, explainable, and fast. Smart Assignment
      makes that call automatically, and flags the tricky ones for a specialist instead of guessing.</p>
  </div>
</section>

<section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
  <div class="wrap">
    <span class="eyebrow">How it works</span>
    <h2>Five steps, from intake to recommendation</h2>
    <p class="sub">The customer's details go in one end; a fully-explained slot recommendation
      (or a human escalation) comes out the other.</p>
    <div class="flow">
      <div class="step"><div class="num">1</div><h3>Intake</h3><p>Capture the customer's address, order quantity (cases), and preferred delivery window (optional).</p></div>
      <div class="step"><div class="num">2</div><h3>Geo-Lookup</h3><p>Geocode the address and pick the Top-{config.top_n_candidate_routes} nearest candidate routes by proximity.</p></div>
      <div class="step"><div class="num">3</div><h3>Constraint Check</h3><p>Drop any route that fails a hard rule — serviceability, capacity, or window.</p></div>
      <div class="step"><div class="num">4</div><h3>Score &amp; Rank</h3><p>Rank the survivors with weighted scoring across clustering, capacity buffer, and window fit.</p></div>
      <div class="step"><div class="num">5</div><h3>Recommend</h3><p>Return the top slot with a reasoning trace — or escalate to a human specialist.</p></div>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <span class="eyebrow">The rules it enforces</span>
    <h2>Hard constraints — non-negotiable, checked in code</h2>
    <p class="sub">These are objective facts, not judgment calls. A route that fails any of them is
      removed before ranking — the agent can never "reason" a customer onto a full truck or outside
      the serviceable area.</p>
    <div class="grid-3">
      <div class="card"><div class="icon">\U0001f4cd</div><h3>Geographic serviceability</h3><p>The customer must fall within the route's serviceable radius.</p></div>
      <div class="card"><div class="icon">\U0001f4e6</div><h3>Route capacity</h3><p>The truck stays at or below {config.max_utilization_after_assignment:.0%} capacity after adding this order.</p></div>
      <div class="card"><div class="icon">\U0001f551</div><h3>Delivery-window fit</h3><p>The route offers a window overlapping the customer's preference, if stated.</p></div>
    </div>

    <div style="height:34px"></div>
    <span class="eyebrow">How it ranks the rest</span>
    <h2>Weighted scoring factors</h2>
    <p class="sub">Among the routes that pass every hard rule, these weighted factors decide the winner.
      Weights reflect priority and are fully configurable.</p>
    <div class="grid-3">{_scoring_cards(config)}</div>
  </div>
</section>

<section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
  <div class="wrap">
    <span class="eyebrow">The outcome</span>
    <h2>Three possible decisions</h2>
    <p class="sub">Every run ends in one of three states. Anything below a {threshold} confidence threshold, or
      with no valid slot at all, goes to a human — with full context attached.</p>
    <div class="legend">
      <div class="card"><span class="pill rec">✔ Recommended</span><p style="margin-top:12px">A clear winner above the confidence threshold — ready to auto-assign.</p></div>
      <div class="card"><span class="pill low">⚠ Low confidence</span><p style="margin-top:12px">A slot is proposed, but the options are close — a specialist confirms before committing.</p></div>
      <div class="card"><span class="pill no">✖ No feasible slot</span><p style="margin-top:12px">Every candidate failed a hard rule — routed to a specialist for a manual decision.</p></div>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <span class="eyebrow">See it in action</span>
    <h2>Live examples on mock Sysco data</h2>
    <p class="sub">These cards are generated straight from the workflow's output. Each customer lands on
      a different outcome; expand <em>“Routes evaluated”</em> to audit the full decision.</p>
    <div class="examples">{cards}
    </div>
  </div>
</section>

<section style="padding-top:0;">
  <div class="wrap">
    <div class="note">
      <strong>About this data.</strong> These examples run on <strong>mock</strong> Houston-area routes and
      geocoding so the workflow can be demonstrated end-to-end. Capacities, service radii, scoring
      weights, and thresholds are illustrative starting points — not validated Sysco policy. The route
      source and geocoder are designed to be swapped for real systems without changing the decision logic.
    </div>
  </div>
</section>

<footer>
  <div class="wrap">
    Smart Assignment · delivery slot recommendation for new customers ·
    built on Google's Agent Development Kit (ADK).<br />
    Source &amp; docs on <a href="https://github.com/MuhammadVT/smart-assignment">GitHub</a>.
    Decisions are deterministic and auditable; reasoning narration is LLM-written with a deterministic fallback.
  </div>
</footer>

</body>
</html>
"""


def generate(output_path: Optional[Path] = None, config: Optional[Config] = None) -> Path:
    """Run the workflow over the sample customers and write ``docs/index.html``."""
    config = config or DEFAULT_CONFIG
    reasoner = DeterministicReasoner()  # reproducible, no API key/network
    results = [
        run_slot_recommendation(customer, config=config, reasoner=reasoner)
        for customer in SAMPLE_CUSTOMERS
    ]
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_page(results, config)
    out.write_text(html_text, encoding="utf-8")
    return out
