/*
 * Chat + live workflow visualization.
 *
 * The step-by-step animation is adapted from the published Simulator
 * (reporting/page.py `_SIM_JS`): the only change is where the data comes from.
 * Instead of looking up a pre-computed entry in an embedded blob, we POST the
 * chat message to /api/recommend and animate the {steps, resultHtml} the real
 * pipeline returns. The card-by-card timing is identical to the static page.
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

  async function send(message) {
    bubble('user', message);
    sendBtn.disabled = true;
    input.disabled = true;
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

  bubble('agent', 'Hi! I assign delivery slots for new Sysco prospects. Give me an ' +
    'address and an order size in cases (a preferred day + time is optional), or pick ' +
    'a sample below.');
})();
