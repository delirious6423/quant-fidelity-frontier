(function () {
  'use strict';
  var data = document.getElementById('bm-data'); if (!data) return;
  var D = JSON.parse(data.textContent), $ = function (id) { return document.getElementById(id); };
  var source = $('bm-source'), metric = $('bm-metric'), pick = $('bm-pick'), host = $('bm-plot'), zero = $('bm-zero'), limit = $('bm-limit');
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) { var n = document.createElementNS(NS, tag); Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); }); if (text != null) n.textContent = text; return n; }
  function option(value, label) { var o = document.createElement('option'); o.value = value; o.textContent = label; return o; }
  function currentSource() { return D.sources[+source.value]; }
  function currentMetric() { return currentSource().metrics[+metric.value]; }
  function unitText() { return currentMetric().unit === 'fraction' ? 'fraction (0–1)' : currentMetric().unit === 'percent' ? 'percent' : 'not stated'; }
  if (!D.sources.length) { $('bm-legend').hidden = true; $('bm-evidence').open = false; return; }
  D.sources.forEach(function (s, i) { source.appendChild(option(i, s.label)); });
  function points() { return currentMetric().points; }
  function changeSource() {
    metric.replaceChildren();
    currentSource().metrics.forEach(function (m, i) { metric.appendChild(option(i, m.name + ' · ' + m.points.length + ' result' + (m.points.length === 1 ? '' : 's'))); });
    $('bm-context').textContent = currentSource().context;
    changeMetric();
  }
  function detail() {
    var p = points()[+pick.value], box = $('bm-detail'); box.replaceChildren(); if (!p) return;
    var title = document.createElement('strong'); title.textContent = p.raw + ' published score · ' + p.size + ' GB'; box.appendChild(title);
    var desc = document.createElement('p'); desc.textContent = p.variant + ' — ' + p.repo + '. Original size: ' + p.size_raw + '. Score unit: ' + unitText() + '.'; box.appendChild(desc);
    var a = document.createElement('a'); a.href = p.src; a.textContent = 'Read publisher source · line ' + p.line; box.appendChild(a); var sizeLink = document.createElement('a'); sizeLink.href = p.size_src; sizeLink.textContent = 'File-size source · line ' + p.size_line; box.appendChild(document.createTextNode(' · ')); box.appendChild(sizeLink);
    var ul = document.createElement('ul'); p.flags.forEach(function (f) { var li = document.createElement('li'); li.textContent = f; ul.appendChild(li); }); box.appendChild(ul);
  }
  function draw() {
    $('bm-axis').textContent = zero.checked ? 'Linear score axis · zero baseline' : 'Zoomed linear score axis · small differences are magnified';
    var ps = points(), w = Math.max(240, host.clientWidth - 24), h = 320, L = 60, R = w - 18, T = 34, B = h - 54;
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img', 'aria-label': currentMetric().name + ' published score versus file size in GB' });
    svg.appendChild(el('title', {}, 'Source-linked reported results; no comparable frontier inferred.'));
    host.replaceChildren(svg);
    if (!ps.length) { $('bm-status').textContent = 'No eligible benchmark results with known file size. See the ledger for other claims.'; return; }
    var ys = ps.map(function (p) { return p.score; }), maxX = Math.max.apply(null, ps.map(function (p) { return p.size; })) * 1.12;
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys), pad = Math.max((ymax - ymin) * .2, currentMetric().unit === 'fraction' ? .001 : 1);
    var low = zero.checked ? Math.min(0, ymin - pad) : ymin - pad, high = ymax + pad;
    var X = function (v) { return L + v / maxX * (R - L); }, Y = function (v) { return B - (v - low) / (high - low) * (B - T); };
    var cap = limit.value !== '' && limit.validity.valid ? +limit.value : Infinity;
    if (cap < maxX) svg.appendChild(el('rect', { x: X(cap), y: T, width: R - X(cap), height: B - T, fill: 'var(--line2)' }));
    for (var i = 0; i <= 3; i++) {
      var yv = low + (high - low) * i / 3, yy = Y(yv), xv = maxX * i / 3;
      svg.appendChild(el('path', { d: 'M' + L + ' ' + yy + 'H' + R, stroke: 'var(--line)', fill: 'none' }));
      svg.appendChild(el('text', { x: L - 8, y: yy + 4, 'text-anchor': 'end' }, +yv.toPrecision(4)));
      svg.appendChild(el('text', { x: X(xv), y: B + 20, 'text-anchor': i === 3 ? 'end' : i === 0 ? 'start' : 'middle' }, Math.round(xv)));
    }
    svg.appendChild(el('text', { x: (L + R) / 2, y: h - 8, 'text-anchor': 'middle' }, 'File size (GB)'));
    svg.appendChild(el('text', { x: L, y: 17 }, 'Published score' + (currentMetric().unit === 'percent' ? ' (%)' : currentMetric().unit === 'fraction' ? ' (0–1)' : '') + ' ↑'));
    ps.forEach(function (p, j) {
      var cx = X(p.size), cy = Y(p.score), radius = j === +pick.value ? 8 : 6;
      var shape = p.reference ? el('path', { d: 'M' + cx + ' ' + (cy-radius) + 'l' + radius + ' ' + radius + 'l-' + radius + ' ' + radius + 'l-' + radius + ' -' + radius + 'Z' }) : el('circle', { cx: cx, cy: cy, r: radius });
      shape.setAttribute('class', 'bm-dot'); shape.setAttribute('opacity', p.size > cap ? '.3' : '1');
      var hit = el('a', { href: '#bm-selection', 'aria-label': p.variant + ', ' + p.raw + ', ' + p.size + ' GB' });
      hit.appendChild(el('circle', { cx: cx, cy: cy, r: 22, fill: 'transparent' })); hit.appendChild(shape);
      hit.appendChild(el('title', {}, p.variant + ': ' + p.raw + ' at ' + p.size + ' GB'));
      hit.addEventListener('click', function (e) { e.preventDefault(); pick.value = j; detail(); draw(); host.querySelectorAll('a')[j].focus(); }); svg.appendChild(hit);
      if (ps.length <= 4 || j === +pick.value) svg.appendChild(el('text', { x: cx, y: cy - 17, 'text-anchor': 'middle', 'class': 'bm-label' }, p.raw));
    });
    var n = ps.filter(function (p) { return p.size <= cap; }).length;
    $('bm-status').textContent = ps.length + ' reported result' + (ps.length === 1 ? ' — no pairwise comparison available.' : 's.') + (cap < Infinity ? ' ' + n + ' within file-size limit; shaded region exceeds it.' : '') + ' Unit: ' + unitText() + '.';
  }
  function changeMetric() { pick.replaceChildren(); points().forEach(function (p, i) { pick.appendChild(option(i, p.variant + ' · ' + p.size + ' GB')); }); $('bm-selection').hidden = !points().length; detail(); draw(); }
  $('bm-axis').hidden = false; $('bm-controls').hidden = false; host.hidden = false; $('bm-legend').hidden = false; $('bm-evidence').open = false;
  source.addEventListener('change', changeSource); metric.addEventListener('change', changeMetric); pick.addEventListener('change', function () { detail(); draw(); }); zero.addEventListener('change', draw); limit.addEventListener('input', draw);
  new ResizeObserver(draw).observe(host); changeSource();
})();
