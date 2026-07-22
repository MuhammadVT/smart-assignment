/*
 * Shared human-feedback widget, used by both the Live-agent result card (app.js)
 * and the read-only Customer view (frontend.js).
 *
 * Design goals driven by review feedback:
 *  - PROMINENT: a distinct, accent-tinted panel with a clear heading, not a thin
 *    line under the result that's easy to miss.
 *  - SUBMIT ON A BUTTON: clicking 👍/👎 only *selects* a rating; the note and the
 *    rating are sent together when the user clicks "Send feedback". So a note
 *    typed after choosing a rating is never lost.
 *  - NO FABRICATED SCORE: a thumb is the categorical signal (label). We do not
 *    invent a numeric score the user never gave.
 *
 * Self-contained: it injects its own CSS once (scoped under `.saf`) using the
 * page's own CSS variables with hex fallbacks, so it looks right on either page
 * without duplicating styles. Exposes `window.SAFeedback.mount(container, fb, opts)`.
 */
(function () {
  if (window.SAFeedback) { return; }

  var STYLE = [
    '.saf{margin-top:18px;border:1px solid var(--line,#e1e6ee);',
    'border-radius:14px;padding:16px 18px;',
    'background:color-mix(in srgb, var(--violet,#6b4fd8) 7%, var(--card,#fff));',
    'box-shadow:0 1px 2px rgba(16,32,64,.06);}',
    '.saf-h{font-weight:700;font-size:15.5px;color:var(--ink,#131a24);',
    'display:flex;align-items:center;gap:8px;}',
    '.saf-h .saf-tag{font-size:11px;font-weight:700;letter-spacing:.04em;',
    'text-transform:uppercase;color:var(--violet,#6b4fd8);',
    'background:color-mix(in srgb,var(--violet,#6b4fd8) 15%,transparent);',
    'padding:2px 8px;border-radius:999px;}',
    '.saf-sub{font-size:13px;color:var(--muted,#59636f);margin-top:3px;}',
    '.saf-rate{display:flex;gap:10px;margin-top:13px;flex-wrap:wrap;}',
    '.saf-btn{display:inline-flex;align-items:center;gap:8px;cursor:pointer;',
    'font-size:14px;font-weight:600;color:var(--ink,#131a24);',
    'border:1.5px solid var(--line,#e1e6ee);background:var(--card,#fff);',
    'border-radius:10px;padding:8px 15px;transition:border-color .12s,background .12s;}',
    '.saf-btn:hover:not(:disabled){border-color:var(--violet,#6b4fd8);}',
    '.saf-btn .saf-ic{font-size:17px;line-height:1;}',
    '.saf-btn.sel{border-color:var(--violet,#6b4fd8);',
    'background:color-mix(in srgb,var(--violet,#6b4fd8) 14%,var(--card,#fff));}',
    '.saf-btn:disabled{opacity:.55;cursor:default;}',
    '.saf-note{display:block;width:100%;margin-top:12px;font:inherit;font-size:13px;',
    'color:var(--ink,#131a24);background:var(--card,#fff);',
    'border:1px solid var(--line,#e1e6ee);border-radius:10px;padding:9px 11px;',
    'resize:vertical;min-height:38px;box-sizing:border-box;}',
    '.saf-note:focus{outline:2px solid color-mix(in srgb,var(--violet,#6b4fd8) 55%,transparent);',
    'outline-offset:1px;border-color:var(--violet,#6b4fd8);}',
    '.saf-actions{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap;}',
    '.saf-send{cursor:pointer;font-size:13.5px;font-weight:650;color:#fff;',
    'background:var(--violet,#6b4fd8);border:0;border-radius:10px;padding:8px 18px;}',
    '.saf-send:disabled{opacity:.45;cursor:default;}',
    '.saf-status{font-size:12.5px;color:var(--muted,#59636f);}',
    '.saf-done{background:color-mix(in srgb,#16805a 9%,var(--card,#fff));',
    'border-color:color-mix(in srgb,#16805a 30%,var(--line,#e1e6ee));}',
    '.saf-thanks{display:flex;align-items:center;gap:9px;font-weight:620;font-size:14px;',
    'color:var(--ink,#131a24);}',
    '.saf-check{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;',
    'background:#16805a;color:#fff;font-size:13px;}'
  ].join('');

  function ensureStyle() {
    if (document.getElementById('saf-style')) { return; }
    var s = document.createElement('style');
    s.id = 'saf-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  function el(tag, cls) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    return e;
  }

  function finish(panel) {
    panel.classList.add('saf-done');
    panel.innerHTML = '';
    var t = el('div', 'saf-thanks');
    var c = el('span', 'saf-check'); c.textContent = '✓';
    var msg = document.createElement('span');
    msg.textContent = 'Thanks — your feedback was recorded.';
    t.appendChild(c); t.appendChild(msg);
    panel.appendChild(t);
  }

  // Mount the widget into `container`. `fb` is the payload's feedback block
  // ({enabled, decision_id, decision_kind, trace_id, span_id}); a no-op unless
  // it's present and enabled. `opts` tunes the copy + who the annotator is.
  function mount(container, fb, opts) {
    if (!container || !fb || !fb.enabled || !fb.decision_id) { return; }
    opts = opts || {};
    ensureStyle();

    var panel = el('div', 'saf');
    var h = el('div', 'saf-h');
    h.textContent = opts.question || 'Was this the right call?';
    if (opts.tag) {
      var tag = el('span', 'saf-tag'); tag.textContent = opts.tag; h.appendChild(tag);
    }
    panel.appendChild(h);
    if (opts.sub) { var sub = el('div', 'saf-sub'); sub.textContent = opts.sub; panel.appendChild(sub); }

    var rate = el('div', 'saf-rate');
    var up = el('button', 'saf-btn'); up.type = 'button'; up.setAttribute('aria-pressed', 'false');
    up.innerHTML = '<span class="saf-ic">👍</span>' + (opts.upLabel || 'Looks right');
    var down = el('button', 'saf-btn'); down.type = 'button'; down.setAttribute('aria-pressed', 'false');
    down.innerHTML = '<span class="saf-ic">👎</span>' + (opts.downLabel || 'Not right');
    rate.appendChild(up); rate.appendChild(down); panel.appendChild(rate);

    var note = el('textarea', 'saf-note');
    note.placeholder = opts.notePlaceholder || 'Add a note (optional) — what was right or wrong?';
    note.rows = 2; note.maxLength = 500;
    panel.appendChild(note);

    var actions = el('div', 'saf-actions');
    var send = el('button', 'saf-send'); send.type = 'button';
    send.textContent = 'Send feedback'; send.disabled = true;
    var status = el('span', 'saf-status');
    actions.appendChild(send); actions.appendChild(status); panel.appendChild(actions);

    var selected = null;
    function choose(label, btn) {
      selected = label;
      up.classList.toggle('sel', btn === up);
      down.classList.toggle('sel', btn === down);
      up.setAttribute('aria-pressed', String(btn === up));
      down.setAttribute('aria-pressed', String(btn === down));
      send.disabled = false;
      status.textContent = '';
    }
    up.addEventListener('click', function () { choose('thumbs_up', up); });
    down.addEventListener('click', function () { choose('thumbs_down', down); });

    send.addEventListener('click', function () {
      if (!selected) { return; }
      send.disabled = true; up.disabled = true; down.disabled = true;
      status.textContent = 'Saving…';
      fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision_id: fb.decision_id,
          label: selected,
          note: (note.value || '').trim() || null,
          session_id: opts.sessionId || '',
          annotator_id: opts.annotatorId || null,
          decision_kind: fb.decision_kind || 'final_response',
          trace_id: fb.trace_id || null,
          span_id: fb.span_id || null
          // No `score`: a thumb is the categorical signal; we don't invent a number.
        })
      })
        .then(function (r) { return r.json(); })
        .then(function (res) {
          if (res && res.ok) { finish(panel); }
          else { status.textContent = 'Could not record feedback. Please try again.'; }
        })
        .catch(function () {
          status.textContent = 'Could not record feedback. Please try again.';
          up.disabled = false; down.disabled = false; send.disabled = false;
        });
    });

    container.appendChild(panel);
    return panel;
  }

  window.SAFeedback = { mount: mount };
})();
