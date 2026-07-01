"""
Generate the GitHub Pages overview site (``docs/index.html``) directly from
live workflow output, so the examples on the page can never drift from the
code.

The page has three audiences-facing goals:
  1. make clear this is an **agent-automated** workflow (every step is badged
     and described as an agent action);
  2. show the **agentic architecture** as a diagram;
  3. let a product owner **run the agent** interactively by entering a mock
     customer number and watching the steps execute, then see the output.

All example/interactive data is precomputed here from the real pipeline
(``mock_customers.SAMPLE_CUSTOMERS``) and embedded as JSON, so the static site
needs no backend. Reasoning uses the DeterministicReasoner so the page is
reproducible offline and regenerating with no code change produces no diff.

CLI: ``python3 scripts/generate_page.py``.
"""

from __future__ import annotations

import html
import json
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
DECISION_SHORT = {
    Decision.RECOMMENDED: "Recommended — auto-assign",
    Decision.ESCALATED_LOW_CONFIDENCE: "Escalate — low confidence",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "Escalate — no feasible slot",
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
    --red: #b42318; --red-soft: #fbe9e7; --violet: #5b3fb0; --violet-soft: #efeafb;
    --radius: 14px; --shadow: 0 1px 2px rgba(16,32,64,.06), 0 6px 20px rgba(16,32,64,.06);
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
  .hero p.lead { margin: 0; font-size: 18px; max-width: 660px; color: #dbe6f5; }
  .hero .meta { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 10px; }
  section { padding: 48px 0; }
  section h2 { font-size: 26px; letter-spacing: -.01em; margin: 0 0 6px; }
  section .sub { color: var(--muted); margin: 0 0 26px; max-width: 720px; }
  .eyebrow { color: var(--blue); font-weight: 700; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
  .agent-banner { display: flex; gap: 12px; align-items: center; background: var(--violet-soft);
    border: 1px solid #ddd2f4; border-radius: var(--radius); padding: 14px 18px; margin-bottom: 24px;
    color: #3a2b6b; font-size: 14px; }
  .agent-banner .big { font-size: 22px; }
  .flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
  .step { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 18px 16px; box-shadow: var(--shadow); position: relative; }
  .step .abadge { position: absolute; top: 12px; right: 12px; font-size: 10px; font-weight: 700;
    color: var(--violet); background: var(--violet-soft); border-radius: 999px; padding: 3px 8px; }
  .step .num { width: 30px; height: 30px; border-radius: 8px; background: var(--blue-soft); color: var(--blue);
    display: grid; place-items: center; font-weight: 700; font-size: 14px; margin-bottom: 10px; }
  .step h3 { margin: 0 0 4px; font-size: 15px; }
  .step p { margin: 0; font-size: 13px; color: var(--muted); }
  .step .action { margin-top: 8px; font-size: 12px; color: var(--violet); font-weight: 600; }
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
  .arch { background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
    padding: 20px; box-shadow: var(--shadow); }
  .arch svg { width: 100%; height: auto; display: block; }
  .arch-legend { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }
  .arch-legend .card { padding: 16px; }
  .arch-legend h4 { margin: 6px 0 3px; font-size: 14px; }
  .arch-legend p { margin: 0; font-size: 12.5px; color: var(--muted); }
  .legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .pill { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px;
    padding: 6px 12px; border-radius: 999px; }
  .pill.rec { background: var(--green-soft); color: var(--green); }
  .pill.low { background: var(--amber-soft); color: var(--amber); }
  .pill.no  { background: var(--red-soft); color: var(--red); }
  /* interactive simulator */
  .sim { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 22px; }
  .sim-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .sim input { font-size: 15px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 9px;
    min-width: 200px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .sim button { background: var(--blue); color: #fff; border: 0; border-radius: 9px; padding: 12px 18px;
    font-weight: 700; font-size: 14px; cursor: pointer; }
  .sim button:disabled { opacity: .55; cursor: default; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  .chip-btn { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    border: 1px solid var(--line); background: #f5f8fc; color: var(--blue); border-radius: 999px;
    padding: 6px 11px; cursor: pointer; }
  .chip-btn:hover { background: var(--blue-soft); }
  .sim-error { color: var(--red); font-size: 13px; margin-top: 10px; min-height: 16px; }
  .sim-cust { margin: 14px 0 4px; font-size: 14.5px; }
  .sim-cust .cnum { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); font-size: 12px; }
  .sim-step { display: grid; grid-template-columns: 26px 1fr; gap: 12px; padding: 11px 0; border-top: 1px solid var(--line); }
  .sim-dot { width: 18px; height: 18px; border-radius: 50%; margin-top: 3px; border: 2px solid var(--line); background: #fff; }
  .sim-step.running .sim-dot { border-color: var(--blue); animation: pulse 1s infinite; }
  .sim-step.done .sim-dot { border-color: var(--green); background: var(--green); }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(18,87,166,.35); } 70% { box-shadow: 0 0 0 8px rgba(18,87,166,0); } 100% { box-shadow: 0 0 0 0 rgba(18,87,166,0); } }
  .sim-title { font-weight: 700; font-size: 14px; }
  .sim-state { font-weight: 700; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-left: 6px; }
  .sim-step.done .sim-state { color: var(--green); }
  .sim-action { font-size: 13px; color: var(--violet); margin-top: 2px; }
  .sim-lines { margin-top: 7px; font-size: 12.5px; color: #33415c; display: grid; gap: 3px; }
  .sim-line .ok { color: var(--green); font-weight: 700; }
  .sim-line .no { color: var(--red); font-weight: 700; }
  .sim-output { margin-top: 18px; }
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
    .grid-3, .legend, .arch-legend { grid-template-columns: 1fr; }
    .examples { grid-template-columns: 1fr; }
    .hero h1 { font-size: 32px; }
    .factor { grid-template-columns: 120px 1fr 40px; }
  }
"""

# Static architecture diagram (SVG). No curly braces -> safe inside the f-string.
_ARCH_SVG = """
<svg viewBox="0 0 980 668" role="img" aria-label="Agentic workflow architecture diagram">
  <defs>
    <marker id="arw" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1257a6"/></marker>
    <marker id="arwg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1a7f37"/></marker>
    <marker id="arwa" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#9a6700"/></marker>
    <marker id="arwd" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#7c8aa0"/></marker>
  </defs>

  <rect x="350" y="12" width="280" height="40" rx="20" fill="#0b2e59"/>
  <text x="490" y="38" text-anchor="middle" fill="#fff" font-size="14" font-weight="600">New customer · 067-NNNNNN</text>
  <line x1="490" y1="52" x2="490" y2="86" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <rect x="24" y="88" width="620" height="440" rx="18" fill="#eef5fd" stroke="#1257a6" stroke-width="2"/>
  <text x="44" y="118" fill="#1257a6" font-size="15" font-weight="800">🤖 AI AGENT — autonomous ADK Workflow orchestrator</text>

  <rect x="44" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="112" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">1 · Intake</text>
  <text x="112" y="199" text-anchor="middle" font-size="10.5" fill="#5b6675">validate profile</text>

  <rect x="194" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="262" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">2 · Geo-Lookup</text>
  <text x="262" y="199" text-anchor="middle" font-size="10.5" fill="#5b6675">geocode + Top-N</text>

  <rect x="344" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="412" y="176" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">3 · Constraint</text>
  <text x="412" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">Check</text>
  <text x="412" y="210" text-anchor="middle" font-size="10.5" fill="#5b6675">apply hard rules</text>

  <rect x="494" y="146" width="136" height="80" rx="12" fill="#fff" stroke="#c7d6ea"/>
  <text x="562" y="176" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">4 · Score &amp;</text>
  <text x="562" y="192" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2e59">Rank</text>
  <text x="562" y="210" text-anchor="middle" font-size="10.5" fill="#5b6675">weighted scoring</text>

  <line x1="180" y1="186" x2="192" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>
  <line x1="330" y1="186" x2="342" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>
  <line x1="480" y1="186" x2="492" y2="186" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <text x="334" y="288" text-anchor="middle" font-size="11" font-weight="700" fill="#0b2e59">5 · Decide</text>
  <polygon points="334,300 412,342 334,384 256,342" fill="#fff" stroke="#9a6700" stroke-width="2"/>
  <text x="334" y="338" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a6700">feasible &amp;</text>
  <text x="334" y="353" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9a6700">confident?</text>
  <path d="M562,226 L562,342 L414,342" fill="none" stroke="#1257a6" stroke-width="2" marker-end="url(#arw)"/>

  <rect x="150" y="560" width="250" height="62" rx="12" fill="#e7f4ea" stroke="#1a7f37"/>
  <text x="275" y="588" text-anchor="middle" font-size="13" font-weight="700" fill="#1a7f37">✅ Recommend &amp; auto-assign</text>
  <text x="275" y="606" text-anchor="middle" font-size="10.5" fill="#3b7a4e">no human needed</text>

  <rect x="452" y="560" width="270" height="62" rx="12" fill="#fdf3d8" stroke="#9a6700"/>
  <text x="587" y="588" text-anchor="middle" font-size="13" font-weight="700" fill="#9a6700">🙋 Escalate to specialist</text>
  <text x="587" y="606" text-anchor="middle" font-size="10.5" fill="#8a6a1f">human reviews &amp; decides</text>

  <path d="M320,382 L275,560" fill="none" stroke="#1a7f37" stroke-width="2" marker-end="url(#arwg)"/>
  <text x="286" y="474" fill="#1a7f37" font-size="11" font-weight="700">yes</text>
  <path d="M352,378 L585,558" fill="none" stroke="#9a6700" stroke-width="2" marker-end="url(#arwa)"/>
  <text x="474" y="474" fill="#9a6700" font-size="11" font-weight="700">no</text>

  <rect x="690" y="150" width="266" height="80" rx="12" fill="#f4f7fb" stroke="#c7d6ea"/>
  <text x="823" y="182" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0b2e59">🗄️ Mocked source systems</text>
  <text x="823" y="201" text-anchor="middle" font-size="10.5" fill="#5b6675">route capacity · geocoding</text>
  <line x1="690" y1="190" x2="648" y2="190" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <rect x="690" y="300" width="266" height="80" rx="12" fill="#efeafb" stroke="#cfc2f0"/>
  <text x="823" y="332" text-anchor="middle" font-size="12.5" font-weight="700" fill="#5b3fb0">🧠 LLM reasoner (Gemini)</text>
  <text x="823" y="351" text-anchor="middle" font-size="10.5" fill="#6b5aa0">writes the plain-English rationale</text>
  <line x1="690" y1="340" x2="648" y2="340" stroke="#7c8aa0" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arwd)"/>

  <text x="24" y="656" font-size="11.5" fill="#5b6675">→ agent flow &#160;&#160;·&#160;&#160; ⇢ agent calls a service &#160;&#160;·&#160;&#160; ◇ agent's own decision point</text>
</svg>
"""

# Browser logic for the interactive simulator. Plain string (NOT an f-string),
# so its many braces are safe; it reads the embedded JSON payload.
_JS = """
(function () {
  var DATA = JSON.parse(document.getElementById('workflow-data').textContent);
  var input = document.getElementById('cust-input');
  var runBtn = document.getElementById('run-btn');
  var chipsEl = document.getElementById('chips');
  var stepsEl = document.getElementById('sim-steps');
  var outEl = document.getElementById('sim-output');
  var errEl = document.getElementById('sim-error');
  var nums = Object.keys(DATA);

  nums.forEach(function (num) {
    var b = document.createElement('button');
    b.className = 'chip-btn';
    b.innerHTML = num + ' · ' + DATA[num].name;
    b.addEventListener('click', function () { input.value = num; run(); });
    chipsEl.appendChild(b);
  });

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  async function run() {
    var num = (input.value || '').trim();
    errEl.textContent = ''; outEl.innerHTML = ''; stepsEl.innerHTML = '';
    var d = DATA[num];
    if (!d) { errEl.textContent = 'Enter a valid mock customer number: ' + nums.join(', '); return; }
    runBtn.disabled = true;
    var head = document.createElement('div');
    head.className = 'sim-cust';
    head.innerHTML = '🤖 Agent running for <b>' + d.name + '</b> <span class="cnum">' + num + '</span>';
    stepsEl.appendChild(head);
    for (var i = 0; i < d.steps.length; i++) {
      var s = d.steps[i];
      var el = document.createElement('div');
      el.className = 'sim-step running';
      el.innerHTML = '<div class="sim-dot"></div><div class="sim-body">' +
        '<div class="sim-title">Step ' + (i + 1) + ' · ' + s.title +
        ' <span class="sim-state">running…</span></div>' +
        '<div class="sim-action">' + s.action + '</div></div>';
      stepsEl.appendChild(el);
      await sleep(720);
      el.classList.remove('running'); el.classList.add('done');
      el.querySelector('.sim-state').textContent = 'done';
      var lines = s.lines.map(function (l) { return '<div class="sim-line">' + l + '</div>'; }).join('');
      el.querySelector('.sim-body').insertAdjacentHTML('beforeend', '<div class="sim-lines">' + lines + '</div>');
      await sleep(260);
    }
    await sleep(180);
    outEl.innerHTML = d.resultHtml;
    runBtn.disabled = false;
  }

  runBtn.addEventListener('click', run);
  input.addEventListener('keydown', function (e) { if (e.key === 'Enter') { run(); } });
})();
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _win(window_str: Optional[str]) -> str:
    if not window_str or window_str == "n/a":
        return "any"
    return window_str.replace("-", "–")


def _pref_fact(result: RecommendationResult) -> str:
    w = result.customer.preferred_window
    return f"prefers {_win(fmt_window(w))}" if w else "no window preference"


def _factor_value(cand: CandidateEvaluation, name: str) -> float:
    for f in cand.factor_scores:
        if f.name == name:
            return f.value
    return 0.0


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
    reason_label = "Why the agent escalated" if rec.requires_human_review else "Why the agent chose this"

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
            <span style="font-size:13px;color:var(--muted)">agent confidence
              <b style="color:{conf_color}">{rec.confidence:.0%}</b></span>
          </div>
          <div class="conf">{conf_bar}</div>
          {slot_html}
          {factors_html}
          <div class="reason"><span class="lbl">{reason_label}</span>{_esc(rec.reasoning)}</div>
          <details class="routes"><summary>Routes the agent evaluated ({n_routes})</summary>
            <div class="routelist">{_route_rows(result.candidates_considered)}</div>
          </details>
        </div>
      </article>"""


def _sim_steps(result: RecommendationResult) -> list[dict]:
    c = result.customer
    rec = result.recommendation
    loc = c.location
    cands = result.candidates_considered
    n = len(cands)
    feasible = [e for e in cands if e.feasible]
    pref = _win(fmt_window(c.preferred_window)) if c.preferred_window else "any"

    intake = [
        f'Customer number <b>{_esc(c.customer_number)}</b> <span class="ok">✓ valid</span>',
        f"Order quantity: <b>{c.order_quantity_cases}</b> cases",
        f"Preferred window: <b>{_esc(pref)}</b>",
    ]
    geo = [
        f"Geocoded address → <b>({loc.latitude:.4f}, {loc.longitude:.4f})</b>",
        f"Selected the {n} nearest routes:",
    ]
    for e in cands:
        geo.append(f"• {_esc(e.route.route_id)} · {_esc(e.route.name)} — {e.distance_miles:.1f} mi")

    con = [f"{len(feasible)} of {n} routes passed every hard rule:"]
    for e in cands:
        if e.feasible:
            con.append(f'• {_esc(e.route.route_id)}: <span class="ok">FEASIBLE</span>')
        else:
            failed = ", ".join(CONSTRAINT_LABEL.get(o.name, o.name) for o in e.failed_constraints)
            con.append(
                f'• {_esc(e.route.route_id)}: <span class="no">INFEASIBLE</span> — failed {_esc(failed)}'
            )

    if result.ranked_feasible:
        score = ["Weighted scores (higher is better):"]
        for e in result.ranked_feasible:
            score.append(
                f"• {_esc(e.route.route_id)}: score <b>{e.total_score:.2f}</b> — "
                f"clustering {_factor_value(e, FACTOR_GEO_CLUSTERING):.2f}, "
                f"buffer {_factor_value(e, FACTOR_CAPACITY_BUFFER):.2f}, "
                f"window {_factor_value(e, FACTOR_WINDOW_MATCH):.2f}"
            )
    else:
        score = ["No feasible routes survived — nothing to score."]

    decide = [
        f"Decision: <b>{DECISION_SHORT[rec.decision]}</b>",
        f"Agent self-assessed confidence: <b>{rec.confidence:.0%}</b>",
    ]
    if rec.recommended_route_id:
        decide.append(
            f"Proposed slot: <b>{_esc(rec.recommended_route_id)} · "
            f"{_esc(rec.recommended_day)} · {_win(rec.recommended_window)}</b>"
        )
    else:
        decide.append("No valid slot — routing specialist required.")

    return [
        {
            "title": "Intake",
            "action": "The agent validates the customer number and captures the intake profile.",
            "lines": intake,
        },
        {
            "title": "Geo-Lookup",
            "action": "The agent geocodes the address and selects the nearest candidate routes.",
            "lines": geo,
        },
        {
            "title": "Constraint Check",
            "action": "The agent applies every hard rule and removes infeasible routes.",
            "lines": con,
        },
        {
            "title": "Score & Rank",
            "action": "The agent scores each feasible route on the weighted factors and ranks them.",
            "lines": score,
        },
        {
            "title": "Recommend / Decide",
            "action": "The agent picks the best slot, scores its own confidence, and decides to auto-assign or escalate.",
            "lines": decide,
        },
    ]


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
    top_n = config.top_n_candidate_routes
    cards = "".join(_example_card(r) for r in results)
    payload = {
        r.customer.customer_number: {
            "name": _esc(r.customer.name),
            "steps": _sim_steps(r),
            "resultHtml": _example_card(r),
        }
        for r in results
    }
    data_block = (
        '<script type="application/json" id="workflow-data">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script>"
    )
    js_block = "<script>" + _JS + "</script>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Smart Assignment — AI Agent for Delivery Slot Recommendation</title>
<meta name="description" content="An AI agent that autonomously assigns delivery slots for new Sysco customers. Explore the agentic workflow, architecture, and run it live on mock data." />
<!-- GENERATED by scripts/generate_page.py from live workflow output. Do not edit by hand. -->
<style>{_STYLE}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <span class="chip">🤖 Agentic workflow · Sysco foodservice</span>
    <h1>Smart Assignment</h1>
    <p class="lead">An <strong>AI agent</strong> that autonomously recommends the best delivery day &amp;
      time slot for a <strong>new customer</strong> — running every step end-to-end, enforcing hard
      operational rules in code, ranking options on weighted business factors, and escalating to a
      human only when it isn't confident.</p>
    <div class="meta">
      <span class="chip">🤖 Fully agent-automated</span>
      <span class="chip">✅ Deterministic, auditable decisions</span>
      <span class="chip">🧠 LLM-written reasoning</span>
      <span class="chip">🧪 Running on mock data</span>
    </div>
  </div>
</header>

<section>
  <div class="wrap">
    <span class="eyebrow">The problem</span>
    <h2>Onboarding a new account, every time, the same way</h2>
    <p class="sub">When a new foodservice customer signs on, someone has to decide which delivery route
      and time window they should join. That decision balances truck capacity, geography, and the
      customer's preference — and it needs to be consistent, explainable, and fast. The Smart
      Assignment agent makes that call automatically, and flags the tricky ones for a specialist
      instead of guessing.</p>
  </div>
</section>

<section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
  <div class="wrap">
    <span class="eyebrow">How the agent works</span>
    <h2>Five steps, executed autonomously by the agent</h2>
    <div class="agent-banner"><span class="big">🤖</span>
      <div><b>Every step below is performed by the AI agent — no human in the loop.</b>
      The agent orchestrates the whole flow end-to-end; a person is involved only if the agent
      decides to escalate at the final step.</div></div>
    <div class="flow">
      <div class="step"><span class="abadge">🤖 Agent</span><div class="num">1</div><h3>Intake</h3><p>Capture the customer's address, order quantity (cases), and preferred window.</p><p class="action">Agent validates the number &amp; builds the profile.</p></div>
      <div class="step"><span class="abadge">🤖 Agent</span><div class="num">2</div><h3>Geo-Lookup</h3><p>Geocode the address and find the nearest candidate routes.</p><p class="action">Agent geocodes &amp; picks the Top-{top_n}.</p></div>
      <div class="step"><span class="abadge">🤖 Agent</span><div class="num">3</div><h3>Constraint Check</h3><p>Drop any route that fails a hard rule.</p><p class="action">Agent enforces serviceability, capacity, window.</p></div>
      <div class="step"><span class="abadge">🤖 Agent</span><div class="num">4</div><h3>Score &amp; Rank</h3><p>Rank survivors on weighted business factors.</p><p class="action">Agent scores &amp; orders the options.</p></div>
      <div class="step"><span class="abadge">🤖 Agent</span><div class="num">5</div><h3>Recommend</h3><p>Return the top slot with a reasoning trace — or escalate.</p><p class="action">Agent self-scores confidence &amp; decides.</p></div>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <span class="eyebrow">Architecture</span>
    <h2>The agentic workflow, end to end</h2>
    <p class="sub">The agent is a Google ADK <em>Workflow</em> that orchestrates the five steps. It calls
      deterministic tools for the objective checks and an LLM only to narrate its decision — the
      decision itself is code, so it's reproducible and auditable.</p>
    <div class="arch">
      {_ARCH_SVG}
      <div class="arch-legend">
        <div class="card"><div class="icon">🤖</div><h4>Agent orchestrator</h4><p>An ADK Workflow drives all five steps and the branching, autonomously.</p></div>
        <div class="card"><div class="icon">🧭</div><h4>Deterministic tools</h4><p>Geo, hard constraints, and weighted scoring — plain code the agent calls.</p></div>
        <div class="card"><div class="icon">🧠</div><h4>LLM reasoner</h4><p>Gemini turns the agent's decision into a plain-English rationale (optional).</p></div>
        <div class="card"><div class="icon">🙋</div><h4>Human-in-the-loop</h4><p>Only engaged when the agent escalates — low confidence or no feasible slot.</p></div>
      </div>
    </div>
  </div>
</section>

<section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);">
  <div class="wrap">
    <span class="eyebrow">The rules the agent enforces</span>
    <h2>Hard constraints — non-negotiable, checked in code</h2>
    <p class="sub">These are objective facts, not judgment calls. The agent removes any route that fails
      one before ranking — it can never "reason" a customer onto a full truck or outside the
      serviceable area.</p>
    <div class="grid-3">
      <div class="card"><div class="icon">\U0001f4cd</div><h3>Geographic serviceability</h3><p>The customer must fall within the route's serviceable radius.</p></div>
      <div class="card"><div class="icon">\U0001f4e6</div><h3>Route capacity</h3><p>The truck stays at or below {config.max_utilization_after_assignment:.0%} capacity after adding this order.</p></div>
      <div class="card"><div class="icon">\U0001f551</div><h3>Delivery-window fit</h3><p>The route offers a window overlapping the customer's preference, if stated.</p></div>
    </div>

    <div style="height:34px"></div>
    <span class="eyebrow">How the agent ranks the rest</span>
    <h2>Weighted scoring factors</h2>
    <p class="sub">Among the routes that pass every hard rule, the agent uses these weighted factors to
      decide the winner. Weights reflect priority and are fully configurable.</p>
    <div class="grid-3">{_scoring_cards(config)}</div>
  </div>
</section>

<section>
  <div class="wrap">
    <span class="eyebrow">The outcome</span>
    <h2>Three possible agent decisions</h2>
    <p class="sub">Every run ends in one of three states. Anything below a {threshold} confidence
      threshold, or with no valid slot at all, the agent hands to a human — with full context attached.</p>
    <div class="legend">
      <div class="card"><span class="pill rec">✔ Recommended</span><p style="margin-top:12px">A clear winner above the confidence threshold — the agent auto-assigns.</p></div>
      <div class="card"><span class="pill low">⚠ Low confidence</span><p style="margin-top:12px">A slot is proposed, but the options are close — the agent asks a specialist to confirm.</p></div>
      <div class="card"><span class="pill no">✖ No feasible slot</span><p style="margin-top:12px">Every candidate failed a hard rule — the agent routes it to a specialist.</p></div>
    </div>
  </div>
</section>

<section style="background:#fff; border-top:1px solid var(--line); border-bottom:1px solid var(--line);" id="try">
  <div class="wrap">
    <span class="eyebrow">Try it yourself</span>
    <h2>Run the agent on a customer</h2>
    <p class="sub">Enter a mock customer number (or pick one below) and watch the agent execute each step,
      then render its recommendation. Everything runs in your browser — no data leaves the page.</p>
    <div class="sim">
      <div class="sim-controls">
        <input id="cust-input" placeholder="e.g. 067-100002" aria-label="Customer number" autocomplete="off" />
        <button id="run-btn">▶ Run agent workflow</button>
      </div>
      <div class="chips" id="chips"></div>
      <div class="sim-error" id="sim-error"></div>
      <div id="sim-steps"></div>
      <div class="sim-output" id="sim-output"></div>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <span class="eyebrow">All outcomes at a glance</span>
    <h2>What the agent decided for each mock customer</h2>
    <p class="sub">These cards are generated straight from the agent's output. Each customer lands on a
      different outcome; expand <em>“Routes the agent evaluated”</em> to audit the full decision.</p>
    <div class="examples">{cards}
    </div>
  </div>
</section>

<section style="padding-top:0;">
  <div class="wrap">
    <div class="note">
      <strong>About this data.</strong> The agent runs on <strong>mock</strong> Houston-area routes and
      geocoding so the workflow can be demonstrated end-to-end. Capacities, service radii, scoring
      weights, and thresholds are illustrative starting points — not validated Sysco policy. The route
      source and geocoder are designed to be swapped for real systems without changing the agent's logic.
    </div>
  </div>
</section>

<footer>
  <div class="wrap">
    Smart Assignment · an AI agent for delivery slot recommendation ·
    built on Google's Agent Development Kit (ADK).<br />
    Source &amp; docs on <a href="https://github.com/MuhammadVT/smart-assignment">GitHub</a>.
    Decisions are deterministic and auditable; reasoning narration is LLM-written with a deterministic fallback.
  </div>
</footer>

{data_block}
{js_block}
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
    out.write_text(build_page(results, config), encoding="utf-8")
    return out
