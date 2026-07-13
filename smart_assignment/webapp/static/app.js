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
  var windowsChart = document.getElementById('windows-chart');
  var windowsRationale = document.getElementById('windows-rationale');

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
  // so several feasible routes stay distinguishable. Deliberately NO green and NO
  // red/orange in this palette: those read as the "feasible ✓ / infeasible ✗"
  // status, so a route tinted green or red would contradict its own badge (a
  // feasible route must never look red, an infeasible one must never look green).
  var ROUTE_PALETTE = [
    '#1257a6', '#7b2fb0', '#0e7c7b', '#b8860b',
    '#4338ca', '#0891b2', '#475569'
  ];
  var PROSPECT_COLOR = '#111827';  // near-black -- distinct from every route hue
  // Infeasible routes are always drawn in red (an unfilled/ring marker), so a
  // ruled-out route reads the same everywhere: map, filter, evaluated cards, and
  // the delivery-window panels. Feasible routes keep their distinct palette hue.
  var INFEASIBLE_COLOR = '#b42318';
  // The recommended route is always drawn green ("go"); other feasible routes use
  // the palette above (no green, no red) so their hue can't be mistaken for the
  // recommended route or an infeasible one.
  var RECOMMENDED_COLOR = '#1a7f37';
  function routeColor(r) { return r.feasible ? routeColors[r.route_id] : INFEASIBLE_COLOR; }

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
      var color = routeColor(r);  // red (ring) for infeasible, palette hue otherwise
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
        var marker = L.circleMarker(latlng, markerStyle(color, r.feasible, 3.5));
        // Hover label: the stop's customer number (+ tier), so a point on the map
        // can be matched to its bar in the delivery-window timeline below.
        if (s.id) {
          marker.bindTooltip(
            s.id + (s.tier ? ' · ' + tierLabel(s.tier) : ''),
            { direction: 'top', offset: [0, -3], className: 'stop-tip', sticky: true }
          );
        }
        marker.addTo(mapLayer);
      });
    });

    if (bounds.length) { mapInstance.fitBounds(bounds, { padding: [30, 30], maxZoom: 14 }); }
    // The panel may have just become visible (display:none -> block), so
    // Leaflet's last-known container size can be stale until it re-measures.
    setTimeout(function () { mapInstance.invalidateSize(); }, 60);
  }

  // Apply a route's colour to a swatch element: a solid dot in the route's hue
  // when feasible, a red unfilled ring when infeasible (so "ruled out" reads red
  // and the same everywhere).
  function paintSwatch(el, color, feasible) {
    el.className = 'route-swatch';
    if (feasible) {
      el.style.background = color;
      el.style.border = '2px solid ' + color;
    } else {
      el.style.background = 'transparent';
      el.style.border = '2px solid ' + INFEASIBLE_COLOR;
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
        renderWindows();  // same selection drives the delivery-window timeline
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
      routeFeasible[r.route_id] = r.feasible;
      // The recommended route is always green; every other feasible route gets a
      // distinct palette hue (the palette has no green or red, so a feasible
      // route never reads as "recommended" or "infeasible"). Infeasible routes
      // fall through to red at draw time via routeColor().
      var isRec = r.slots && r.slots.some(function (s) { return s.recommended; });
      routeColors[r.route_id] = isRec
        ? RECOMMENDED_COLOR
        : ROUTE_PALETTE[i % ROUTE_PALETTE.length];
    });
    ensureMap();
    buildFilterMenu();
    drawMapLayers();
    renderWindows();
  }

  // --- Delivery-window timeline (below the map, same route selection) ---
  // Routes are shown in the agent's scored order (recommended first, then the
  // rest, then infeasible) -- matching the "Routes the agent evaluated" section.
  // For each route: a Gantt of its committed stops' delivery windows on a shared
  // time axis, each bar coloured by the customer's tier; then an "availability"
  // ribbon that coalesces overlap into a few busy/open segments (open gaps
  // labelled with how long they are) so the free slots read at a glance.
  var TW_BUCKET_MIN = 15;  // availability sampling resolution

  // Customer-tier palette (bar colour). Ordered high-to-low for the legend;
  // unknown/other tiers fall back to slate.
  var TIER_ORDER = ['Perks', '5', '4', 'Other'];
  // Distinct hues per tier -- amber / violet / teal / slate -- chosen so no two
  // adjacent tiers read as the same colour (Tier 4 and Tier 5 used to be a blue
  // and a purple that were easy to confuse).
  var TIER_COLORS = { 'Perks': '#d97706', '5': '#7c3aed', '4': '#0d9488', 'Other': '#64748b' };
  var TIER_FALLBACK = '#64748b';
  function tierColor(t) { return (t && TIER_COLORS[t]) || TIER_FALLBACK; }
  function tierLabel(t) { return t ? ('Tier ' + t).replace('Tier Perks', 'Perks') : 'Unknown'; }

  function toMin(hhmm) {
    var p = hhmm.split(':');
    return parseInt(p[0], 10) * 60 + parseInt(p[1], 10);
  }
  function fmtMin(m) {
    var h = Math.floor(m / 60), mm = m % 60;
    return (h < 10 ? '0' : '') + h + ':' + (mm < 10 ? '0' : '') + mm;
  }
  function fmtDur(m) {
    var h = Math.floor(m / 60), mm = m % 60;
    return (h ? h + 'h' : '') + (h && mm ? ' ' : '') + (mm ? mm + 'm' : (h ? '' : '0m'));
  }
  function renderWindows() {
    if (!windowsChart) { return; }
    var panel = document.getElementById('windows-panel');
    if (!currentMapData) { if (panel) { panel.style.display = 'none'; } return; }

    // Selected routes that actually have at least one stop with a window, in the
    // agent's ranked order (payload `rank`; falls back to input order).
    var routes = currentMapData.routes.filter(function (r) {
      return selectedRoutes[r.route_id] && r.stops.some(function (s) { return s.window; });
    }).slice().sort(function (a, b) {
      return (a.rank == null ? 1e9 : a.rank) - (b.rank == null ? 1e9 : b.rank);
    });
    if (!routes.length) {
      if (panel) { panel.style.display = 'none'; }
      windowsChart.innerHTML = '';
      if (windowsRationale) { windowsRationale.innerHTML = ''; }
      return;
    }
    if (panel) { panel.style.display = ''; }
    // "Why this slot" rationale for the recommended route-slot, under the chart.
    if (windowsRationale) { windowsRationale.innerHTML = currentMapData.rationaleHtml || ''; }

    // Shared time domain across all selected routes, snapped to the hour.
    var lo = Infinity, hi = -Infinity;
    routes.forEach(function (r) {
      r.stops.forEach(function (s) {
        if (!s.window) { return; }
        lo = Math.min(lo, toMin(s.window.open));
        hi = Math.max(hi, toMin(s.window.close));
      });
      // The recommended window may extend past the committed stops (e.g. a 3h
      // slot off a 2h historical window), so keep it inside the drawn domain.
      if (r.chosen_window) {
        lo = Math.min(lo, toMin(r.chosen_window.open));
        hi = Math.max(hi, toMin(r.chosen_window.close));
      }
    });
    lo = Math.floor(lo / 60) * 60;
    hi = Math.ceil(hi / 60) * 60;
    var span = Math.max(hi - lo, 60);
    function pct(m) { return ((m - lo) / span) * 100; }

    var html = '';

    // Tier legend -- only the tiers actually present among the shown stops.
    var present = {};
    routes.forEach(function (r) {
      r.stops.forEach(function (s) { if (s.window) { present[s.tier || 'Unknown'] = true; } });
    });
    var tiers = Object.keys(present).sort(function (a, b) {
      var ia = TIER_ORDER.indexOf(a), ib = TIER_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    var tierItems = tiers.map(function (t) {
      var c = t === 'Unknown' ? TIER_FALLBACK : tierColor(t);
      return '<span class="tw-tier-item"><span class="tw-tier-dot" style="background:' + c
        + '"></span>' + tierLabel(t === 'Unknown' ? null : t) + '</span>';
    }).join('');
    html += '<div class="tw-tier-legend"><span class="tw-tier-lead">Window colour · tier:</span>'
      + tierItems + '</div>';

    // Shared hour axis.
    var ticks = '';
    for (var t = lo; t <= hi; t += 60) {
      ticks += '<span class="tw-tick" style="left:' + pct(t) + '%">' + fmtMin(t) + '</span>';
    }
    html += '<div class="tw-axis"><span class="tw-label"></span>'
      + '<span class="tw-track">' + ticks + '</span></div>';

    routes.forEach(function (r) {
      var color = routeColor(r);  // red for infeasible, palette hue otherwise
      var stops = r.stops.filter(function (s) { return s.window; }).slice().sort(function (a, b) {
        return toMin(a.window.open) - toMin(b.window.open);
      });

      // Route dot: a filled hue dot when feasible, a red unfilled ring when not.
      var dotStyle = r.feasible
        ? 'background:' + color
        : 'background:transparent;border:2px solid ' + INFEASIBLE_COLOR;
      html += '<div class="tw-route">'
        + '<span class="tw-dot" style="' + dotStyle + '"></span>'
        + '<b>' + r.route_id + '</b> · ' + r.name + ' <span class="tw-day">(' + r.day + ')</span>'
        + ' <span class="tw-count">' + stops.length + ' stops</span></div>';

      html += '<div class="tw-route-block">';

      // Two dashed verticals marking the RECOMMENDED slot's start + end, spanning
      // the whole block (stops + availability + candidate slots) on the shared axis.
      var recSlot = (r.slots || []).filter(function (s) { return s.recommended; })[0];
      if (recSlot) {
        var gO = pct(toMin(recSlot.open)), gC = pct(toMin(recSlot.close));
        var gTitle = 'recommended slot ' + recSlot.open + '–' + recSlot.close;
        html += '<div class="tw-guides">'
          + '<span class="tw-guide" style="left:' + gO + '%;border-color:' + color
          + '" title="' + gTitle + '"></span>'
          + '<span class="tw-guide" style="left:' + gC + '%;border-color:' + color
          + '" title="' + gTitle + '"></span></div>';
      }

      stops.forEach(function (s) {
        var o = toMin(s.window.open), c = toMin(s.window.close);
        var bar = tierColor(s.tier);
        var title = s.id + ' · ' + s.window.open + '–' + s.window.close
          + ' · ' + tierLabel(s.tier);
        html += '<div class="tw-row"><span class="tw-label" title="' + s.id + '">' + s.id + '</span>'
          + '<span class="tw-track">'
          + '<span class="tw-bar" title="' + title + '" style="left:' + pct(o) + '%;width:'
          + (pct(c) - pct(o)) + '%;background:' + bar + '"></span></span></div>';
      });

      // Availability ribbon: sample overlap per bucket, then COALESCE adjacent
      // equal-demand buckets into a handful of segments -- one busy block per
      // demand level and one green block per open gap -- instead of a jagged
      // per-bucket histogram. Open gaps carry a duration label.
      var maxD = 0, samples = [];
      for (var b = lo; b < hi; b += TW_BUCKET_MIN) {
        var mid = b + TW_BUCKET_MIN / 2;
        var d = 0;
        stops.forEach(function (s) {
          if (toMin(s.window.open) <= mid && mid < toMin(s.window.close)) { d++; }
        });
        samples.push(d);
        maxD = Math.max(maxD, d);
      }
      var segs = [];
      for (var i = 0; i < samples.length; i++) {
        var start = lo + i * TW_BUCKET_MIN;
        if (segs.length && segs[segs.length - 1].d === samples[i]) {
          segs[segs.length - 1].end = start + TW_BUCKET_MIN;
        } else {
          segs.push({ start: start, end: start + TW_BUCKET_MIN, d: samples[i] });
        }
      }
      var ribbon = segs.map(function (sg) {
        var left = pct(sg.start), w = pct(sg.end) - pct(sg.start), dur = sg.end - sg.start;
        if (sg.d === 0) {
          var lbl = w > 7 ? '<span class="tw-open-txt">' + fmtDur(dur) + ' open</span>' : '';
          return '<span class="tw-seg tw-open" style="left:' + left + '%;width:' + w
            + '%" title="Open ' + fmtMin(sg.start) + '–' + fmtMin(sg.end) + ' · ' + fmtDur(dur)
            + '">' + lbl + '</span>';
        }
        // Busy: single-height block, shaded by how many windows overlap.
        var op = (0.32 + 0.68 * (sg.d / Math.max(maxD, 1))).toFixed(2);
        return '<span class="tw-seg tw-busy" style="left:' + left + '%;width:' + w
          + '%;opacity:' + op + '" title="' + sg.d + ' overlapping · ' + fmtMin(sg.start) + '–'
          + fmtMin(sg.end) + '"></span>';
      }).join('');
      html += '<div class="tw-row tw-lane-row"><span class="tw-label">availability</span>'
        + '<span class="tw-track tw-lane">' + ribbon + '</span></div>';

      // Candidate (route, slot) options: one row per scored slot, drawn on the
      // shared axis as an outlined window with its own route-slot score. The
      // overall recommended slot is filled + starred. This is what makes the
      // panel route-slot level (each slot is shown and scored on its own).
      if (r.slots && r.slots.length) {
        html += '<div class="tw-row tw-slots-head"><span class="tw-label">candidate slots</span>'
          + '<span class="tw-track"></span></div>';
        r.slots.forEach(function (sl) {
          var o = toMin(sl.open), c = toMin(sl.close);
          var scoreTxt = (sl.score != null) ? sl.score.toFixed(2) : '';
          var lbl = (sl.recommended ? '★ ' : '') + scoreTxt;
          var title = 'slot ' + sl.open + '–' + sl.close
            + (sl.score != null ? ' · score ' + scoreTxt : '')
            + (sl.recommended ? ' · recommended' : '');
          var barStyle = 'left:' + pct(o) + '%;width:' + (pct(c) - pct(o))
            + '%;border-color:' + color + (sl.recommended ? ';background:' + color : '');
          html += '<div class="tw-row tw-slot-row' + (sl.recommended ? ' tw-slot-rec' : '') + '">'
            + '<span class="tw-label" title="' + title + '">' + lbl + '</span>'
            + '<span class="tw-track"><span class="tw-slot-bar" title="' + title
            + '" style="' + barStyle + '">' + sl.open + '–' + sl.close + '</span></span></div>';
        });
      }
      html += '</div>';  // .tw-route-block
    });

    windowsChart.innerHTML = html;
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
    var noticeHtml = '';
    if (d.notices && d.notices.length) {
      noticeHtml = d.notices.map(function (n) {
        return '<div class="sim-notice ' + (n.kind || 'info') + '">' + n.text + '</div>';
      }).join('');
    }
    outEl.innerHTML = noticeHtml + d.resultHtml;
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
      // Always stream through /api/chat: it drives the ADK agent in llm mode and
      // the session-aware deterministic brain otherwise, so the conversation
      // stays multi-turn (remembers context, accepts revisions) in every mode.
      await sendLlm(message);
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
        bubble('agent', 'Hi! I assign delivery slots for new Sysco prospects. Tell me the ' +
          'address and order size in cases (a preferred day + time is optional) — you can ' +
          'add or change details as we go, like “try 20 cases”. Or pick a sample below.');
        if (m && m.reason) { bubble('agent', 'ℹ️ ' + m.reason, 'thinking'); }
      }
    });
})();
