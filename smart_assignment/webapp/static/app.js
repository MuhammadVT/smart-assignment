/*
 * Chat + live workflow visualization.
 *
 * Two modes, chosen by the server (GET /api/mode):
 *  - "llm"           → Phase 2: stream the real ADK agent over POST /api/chat
 *                      (Server-Sent Events), true multi-turn natural language.
 *  - "deterministic" → Phase 1: POST /api/recommend, one message = one run.
 *
 * Either way the step-by-step animation is the same — adapted from the published
 * Simulator (reporting/page.py `_SIM_JS`) — and renders the identical
 * {steps, resultHtml} payload the real pipeline produces.
 */
(function () {
  var input = document.getElementById('msg-input');
  var form = document.getElementById('composer');
  var sendBtn = document.getElementById('send-btn');
  var chipsEl = document.getElementById('chips');
  var transcript = document.getElementById('transcript');
  var stepsEl = document.getElementById('sim-steps');
  var outEl = document.getElementById('sim-output');
  var viz = document.querySelector('.viz');

  var MODE = 'deterministic';
  var SESSION_ID = (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : 'sess-' + Date.now() + '-' + Math.floor(Math.random() * 1e9);

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function bubble(role, text, extraClass) {
    var el = document.createElement('div');
    el.className = 'bubble ' + role + (extraClass ? ' ' + extraClass : '');
    el.textContent = text;
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
    return el;
  }

  // Load the sample prospects as chips; clicking one drops it into the box.
  fetch('/api/samples')
    .then(function (r) { return r.json(); })
    .then(function (samples) {
      samples.forEach(function (s) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip-btn';
        b.textContent = s.name;
        b.title = s.message;
        b.addEventListener('click', function () {
          input.value = s.message;
          input.classList.remove('flash');
          void input.offsetWidth;  // restart the flash animation
          input.classList.add('flash');
          input.focus();
        });
        chipsEl.appendChild(b);
      });
    })
    .catch(function () { /* samples are optional; ignore fetch errors */ });

  async function animate(d) {
    stepsEl.innerHTML = '';
    outEl.innerHTML = '';
    viz.classList.remove('has-result');
    viz.classList.add('running');

    var head = document.createElement('div');
    head.className = 'sim-cust';
    head.innerHTML = '🤖 Agent running for <b>' + d.name + '</b> ' +
      '<span class="cnum">' + d.address + '</span>';
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
    viz.classList.remove('running');
    viz.classList.add('has-result');
  }

  // --- Phase 1: deterministic one-shot ---
  async function sendDeterministic(message) {
    var thinking = bubble('agent', 'Working on it…', 'thinking');
    try {
      var res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
      });
      var data = await res.json();
      thinking.remove();
      if (!data.ok) {
        bubble('agent', data.reply || 'I could not run that. Please try again.');
        return;
      }
      if (data.reply) { bubble('agent', data.reply); }
      await animate(data.payload);
    } catch (err) {
      thinking.remove();
      bubble('agent', 'Something went wrong talking to the agent. Please try again.', 'error');
    }
  }

  // --- Phase 2: stream the real ADK agent over SSE ---
  async function sendLlm(message) {
    var thinking = bubble('agent', 'Thinking…', 'thinking');
    var pendingViz = null;
    try {
      var res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: SESSION_ID, message: message })
      });
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buf = '';
      var first = true;

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) { break; }
        buf += decoder.decode(chunk.value, { stream: true });
        var parts = buf.split('\n\n');
        buf = parts.pop();  // keep the trailing partial frame
        for (var i = 0; i < parts.length; i++) {
          var line = parts[i].trim();
          if (line.indexOf('data:') !== 0) { continue; }
          var frame;
          try { frame = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
          if (first) { thinking.remove(); first = false; }

          if (frame.type === 'tool') {
            bubble('agent', '🔧 ' + frame.label + '…', 'thinking');
          } else if (frame.type === 'message') {
            bubble('agent', frame.text);
          } else if (frame.type === 'await_input') {
            bubble('agent', '🙋 ' + frame.message, 'await');
          } else if (frame.type === 'visualization') {
            pendingViz = frame.payload;  // animate after the stream closes
          } else if (frame.type === 'error') {
            bubble('agent', frame.message || 'Something went wrong.', 'error');
          }
          // 'done' needs no UI action.
        }
      }
      if (first) { thinking.remove(); }
      if (pendingViz) { await animate(pendingViz); }
    } catch (err) {
      thinking.remove();
      bubble('agent', 'Something went wrong streaming the agent. Please try again.', 'error');
    }
  }

  async function send(message) {
    bubble('user', message);
    sendBtn.disabled = true;
    input.disabled = true;
    try {
      if (MODE === 'llm') { await sendLlm(message); }
      else { await sendDeterministic(message); }
    } finally {
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var message = (input.value || '').trim();
    if (!message) { return; }
    input.value = '';
    send(message);
  });

  // Resolve the mode, then greet accordingly.
  fetch('/api/mode')
    .then(function (r) { return r.json(); })
    .then(function (m) { MODE = m.mode || 'deterministic'; return m; })
    .catch(function () { return {}; })
    .then(function (m) {
      if (MODE === 'llm') {
        bubble('agent', 'Hi! I assign delivery slots for new Sysco prospects. Tell me about ' +
          'the prospect in your own words — address, order size, any preferred day/time — ' +
          'and I’ll walk through it. You can also pick a sample below.');
      } else {
        bubble('agent', 'Hi! I assign delivery slots for new Sysco prospects. Give me an ' +
          'address and an order size in cases (a preferred day + time is optional), or pick ' +
          'a sample below.');
        if (m && m.reason) { bubble('agent', 'ℹ️ ' + m.reason, 'thinking'); }
      }
    });
})();
