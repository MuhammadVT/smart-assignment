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
  var mapPanel = document.getElementById('map-panel');
  var routesPanel = document.getElementById('routes-panel');
  var filterBtn = document.getElementById('route-filter-btn');
  var filterMenu = document.getElementById('route-filter-menu');

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

  // --- Proximity map (Leaflet + OpenStreetMap tiles, free, no API key) ---
  var mapInstance = null;
  var mapLayer = null;
  var MILES_TO_METERS = 1609.34;
  var currentMapData = null;
  var selectedRoutes = {};  // route_id -> bool (which routes to draw)
  var routeColors = {};     // route_id -> hex colour (distinguishes routes)
  var routeFeasible = {};   // route_id -> bool
  // Distinct, saturated hues so two routes are never the same colour. Feasibility
  // is shown by FILL (solid = feasible, hollow ring = infeasible), not by hue --
  // so several feasible routes stay distinguishable. None is near-black, so the
  // prospect's dark star can't be mistaken for a route.
  var ROUTE_PALETTE = [
    '#1257a6', '#1a7f37', '#c2410c', '#7b2fb0',
    '#0e7c7b', '#b8860b', '#9d174d'
  ];
  var PROSPECT_COLOR = '#111827';  // near-black -- distinct from every route hue

  // The prospect is a STAR (its own shape + its own colour) so it never reads as
  // a route point; service centers are DIAMONDS; stops stay small circles.
  function prospectIcon() {
    var svg =
      '<svg width="28" height="28" viewBox="0 0 24 24" aria-label="prospect">' +
      '<path d="M12 1.6l3 6.1 6.7 1-4.85 4.73 1.15 6.67L12 17.9l-6 3.15 ' +
      '1.15-6.67L2.3 8.7l6.7-1z" fill="' + PROSPECT_COLOR + '" ' +
      'stroke="#ffffff" stroke-width="1.3" stroke-linejoin="round"/></svg>';
    return L.divIcon({ className: 'map-glyph', html: svg, iconSize: [28, 28], iconAnchor: [14, 14] });
  }

  function centerIcon(color, feasible) {
    var fill = feasible ? color : '#ffffff';
    var svg =
      '<svg width="20" height="20" viewBox="0 0 20 20" aria-label="route center">' +
      '<polygon points="10,1.5 18.5,10 10,18.5 1.5,10" fill="' + fill + '" ' +
      'stroke="' + color + '" stroke-width="2.5" stroke-linejoin="round"/></svg>';
    return L.divIcon({ className: 'map-glyph', html: svg, iconSize: [20, 20], iconAnchor: [10, 10] });
  }

  function ensureMap() {
    if (mapInstance) { return mapInstance; }
    mapInstance = L.map('map', { scrollWheelZoom: false });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(mapInstance);
    mapLayer = L.layerGroup().addTo(mapInstance);
    return mapInstance;
  }

  // Order a route's stops into a path by nearest-neighbour, starting from the
  // stop closest to the service center. committed_stops has no real visit
  // sequence, so this is only a legibility aid: it connects each stop to its
  // nearest neighbour rather than fanning every stop back to the center.
  function orderStops(stops, center) {
    if (stops.length <= 1) { return stops.slice(); }
    function d2(a, b) {
      var dx = a.lat - b.lat, dy = a.lng - b.lng;
      return dx * dx + dy * dy;
    }
    var remaining = stops.slice();
    var path = [];
    var curr = center;
    while (remaining.length) {
      var bestIdx = 0, bestD = Infinity;
      for (var i = 0; i < remaining.length; i++) {
        var dd = d2(curr, remaining[i]);
        if (dd < bestD) { bestD = dd; bestIdx = i; }
      }
      curr = remaining[bestIdx];
      path.push(curr);
      remaining.splice(bestIdx, 1);
    }
    return path;
  }

  // Marker fill by feasibility: solid dot when feasible, hollow ring (white
  // centre) when not -- so a route reads as "in play" vs "ruled out" while its
  // hue still identifies which route it is.
  function markerStyle(color, feasible, radius) {
    return feasible
      ? { radius: radius, color: color, fillColor: color, fillOpacity: 0.9, weight: 2 }
      : { radius: radius, color: color, fillColor: '#ffffff', fillOpacity: 1, weight: 2.5 };
  }

  function drawMapLayers() {
    if (!currentMapData || !mapInstance) { return; }
    mapLayer.clearLayers();
    var bounds = [];

    var custLatLng = [currentMapData.customer.lat, currentMapData.customer.lng];
    bounds.push(custLatLng);
    L.marker(custLatLng, { icon: prospectIcon(), zIndexOffset: 1000 })
      .bindPopup('<b>' + currentMapData.customer.name + '</b><br>Prospect location')
      .addTo(mapLayer);

    currentMapData.routes.forEach(function (r) {
      if (!selectedRoutes[r.route_id]) { return; }
      var color = routeColors[r.route_id];
      var centerLatLng = [r.service_center.lat, r.service_center.lng];
      bounds.push(centerLatLng);

      if (r.service_radius_miles) {
        L.circle(centerLatLng, {
          radius: r.service_radius_miles * MILES_TO_METERS,
          color: color, weight: 1, fillOpacity: 0.03, dashArray: '4 4'
        }).addTo(mapLayer);
      }

      var scoreLine = (r.total_score !== null && r.total_score !== undefined)
        ? ('<br>Score: ' + Math.round(r.total_score * 100) + '%') : '';
      L.marker(centerLatLng, { icon: centerIcon(color, r.feasible), zIndexOffset: 500 })
        .bindPopup(
          '<b>' + r.route_id + ' · ' + r.name + '</b><br>' + r.day + ' · ' + r.distance_miles + ' mi' +
          '<br>' + (r.feasible ? 'FEASIBLE' : 'INFEASIBLE') + scoreLine
        )
        .addTo(mapLayer);

      // Dashed path connecting adjacent stops (nearest-neighbour order), so it's
      // clear which stops belong to this route without implying a delivery order.
      var ordered = orderStops(r.stops, r.service_center);
      if (ordered.length > 1) {
        L.polyline(ordered.map(function (s) { return [s.lat, s.lng]; }), {
          color: color, weight: 1.5, opacity: 0.55, dashArray: '5 6'
        }).addTo(mapLayer);
      }
      ordered.forEach(function (s) {
        var latlng = [s.lat, s.lng];
        bounds.push(latlng);
        L.circleMarker(latlng, markerStyle(color, r.feasible, 3.5)).addTo(mapLayer);
      });
    });

    if (bounds.length) { mapInstance.fitBounds(bounds, { padding: [30, 30], maxZoom: 14 }); }
    // The panel may have just become visible (display:none -> block), so
    // Leaflet's last-known container size can be stale until it re-measures.
    setTimeout(function () { mapInstance.invalidateSize(); }, 60);
  }

  // Apply a route's colour to a swatch element (solid if feasible, ring if not).
  function paintSwatch(el, color, feasible) {
    el.className = 'route-swatch';
    if (feasible) {
      el.style.background = color;
      el.style.border = '2px solid ' + color;
    } else {
      el.style.background = 'transparent';
      el.style.border = '2px solid ' + color;
    }
  }

  function buildFilterMenu() {
    filterMenu.innerHTML = '';
    currentMapData.routes.forEach(function (r) {
      var label = document.createElement('label');
      label.className = 'route-filter-item';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!selectedRoutes[r.route_id];
      cb.addEventListener('change', function () {
        selectedRoutes[r.route_id] = cb.checked;
        drawMapLayers();
      });
      var swatch = document.createElement('span');
      paintSwatch(swatch, routeColors[r.route_id], r.feasible);
      var text = document.createElement('span');
      text.className = 'route-filter-name';
      text.textContent = r.route_id + ' · ' + r.name;
      text.title = r.route_id + ' · ' + r.name + (r.feasible ? ' (feasible)' : ' (infeasible)');
      label.appendChild(cb);
      label.appendChild(swatch);
      label.appendChild(text);
      filterMenu.appendChild(label);
    });
  }

  // Prefix each evaluated-route card with its map colour so the card, the map
  // marker, and the filter row all read as the same route.
  function paintRouteCards() {
    var cards = routesPanel.querySelectorAll('.routecard[data-route-id]');
    for (var i = 0; i < cards.length; i++) {
      var id = cards[i].getAttribute('data-route-id');
      var color = routeColors[id];
      if (!color) { continue; }
      var titleSpan = cards[i].querySelector('.rtitle span');
      if (!titleSpan || titleSpan.querySelector('.route-swatch')) { continue; }
      var swatch = document.createElement('span');
      paintSwatch(swatch, color, routeFeasible[id]);
      swatch.style.marginRight = '7px';
      titleSpan.insertBefore(swatch, titleSpan.firstChild);
    }
  }

  function renderMap(mapData) {
    if (!mapData || !window.L) { mapPanel.style.display = 'none'; return; }
    mapPanel.style.display = '';
    currentMapData = mapData;
    selectedRoutes = {};
    routeColors = {};
    routeFeasible = {};
    mapData.routes.forEach(function (r, i) {
      selectedRoutes[r.route_id] = true;  // all on by default
      routeColors[r.route_id] = ROUTE_PALETTE[i % ROUTE_PALETTE.length];
      routeFeasible[r.route_id] = r.feasible;
    });
    ensureMap();
    buildFilterMenu();
    drawMapLayers();
  }

  // Route-filter dropdown open/close.
  if (filterBtn) {
    filterBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = filterMenu.hidden;
      filterMenu.hidden = !open;
      filterBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (e) {
      if (!filterMenu.hidden && !filterMenu.contains(e.target) && e.target !== filterBtn) {
        filterMenu.hidden = true;
        filterBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

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
    routesPanel.innerHTML = d.routesHtml || '';
    renderMap(d.map);        // assigns each route its colour
    paintRouteCards();       // colour-match the cards to the map markers
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
