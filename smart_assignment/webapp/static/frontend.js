/*
 * Read-only customer view (/frontend).
 *
 * Shows the "Choose a delivery slot" options the chat (/) last produced for THIS
 * browser session — mirroring production, where a prospect flows Salesforce ->
 * Smart Assignment -> the sales consultant's view. There is no input here and no
 * way to run the pipeline: the page just fetches the latest result and renders
 * the same `frontendHtml` the published GitHub Pages Frontend tab uses.
 *
 * The slot-select / map-switch interaction is adapted from the static page's
 * _FRONTEND_JS (reporting/page.py) so a rep can click through the options.
 */
(function () {
  var host = document.getElementById('fe-view');
  var empty = document.getElementById('fe-empty');
  var source = document.getElementById('fe-source');
  var prospectEl = document.getElementById('fe-prospect');
  if (!host) { return; }

  // The same per-browser id the chat page persists, so this view resolves to the
  // slots that session just produced.
  function sessionId() {
    try { return localStorage.getItem('sa_session_id') || ''; } catch (e) { return ''; }
  }

  function decodeEntities(s) {
    var t = document.createElement('textarea');
    t.innerHTML = s || '';
    return t.value;
  }

  function showEmpty() {
    host.innerHTML = '';
    if (source) { source.hidden = true; }
    if (empty) { empty.hidden = false; }
  }

  function render(data) {
    host.innerHTML = data.frontendHtml || '';
    if (prospectEl) { prospectEl.textContent = decodeEntities(data.name); }
    if (source) { source.hidden = false; }
    if (empty) { empty.hidden = true; }
  }

  // Delegated so it survives the innerHTML swap: click (or Enter/Space on) a
  // selectable slot card to select it, update the confirm bar's label, and
  // switch the cluster map to that slot's route.
  function selectCard(card) {
    if (!card || !host.contains(card)) { return; }
    var cards = host.querySelectorAll('.fe-opt.selectable');
    for (var i = 0; i < cards.length; i++) {
      cards[i].classList.remove('selected');
      cards[i].setAttribute('aria-pressed', 'false');
    }
    card.classList.add('selected');
    card.setAttribute('aria-pressed', 'true');
    var sel = host.querySelector('#fe-sel');
    if (sel) { sel.textContent = card.getAttribute('data-when') || sel.textContent; }
    var rid = card.getAttribute('data-route');
    var bodies = host.querySelectorAll('.fe-mapbody');
    if (rid && bodies.length) {
      var match = false;
      for (var j = 0; j < bodies.length; j++) {
        if (bodies[j].getAttribute('data-route') === rid) { match = true; }
      }
      if (match) {
        for (var m = 0; m < bodies.length; m++) {
          bodies[m].hidden = (bodies[m].getAttribute('data-route') !== rid);
        }
      }
    }
  }
  host.addEventListener('click', function (ev) {
    selectCard(ev.target.closest ? ev.target.closest('.fe-opt.selectable') : null);
  });
  host.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter' && ev.key !== ' ' && ev.key !== 'Spacebar') { return; }
    var card = ev.target.closest ? ev.target.closest('.fe-opt.selectable') : null;
    if (card) { ev.preventDefault(); selectCard(card); }
  });

  var sid = sessionId();
  if (!sid) { showEmpty(); return; }
  fetch('/api/frontend?session=' + encodeURIComponent(sid))
    .then(function (r) { return r.json(); })
    .then(function (data) { if (data && data.ok) { render(data); } else { showEmpty(); } })
    .catch(showEmpty);
})();
