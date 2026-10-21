(function () {
  'use strict';
  var catalog = JSON.parse(document.getElementById('home-data').textContent);
  var $ = function (id) { return document.getElementById(id); };
  var model = $('home-model'), set = $('home-set'), format = $('home-format'), basis = $('home-basis'), limit = $('home-limit');
  function currentModel() { return catalog.find(function (m) { return m.slug === model.value; }); }
  function currentSet() { return currentModel().sets.find(function (s) { return s.id === set.value; }); }
  function options(select, entries) {
    select.replaceChildren();
    entries.forEach(function (entry) { var o = document.createElement('option'); o.value = entry[0]; o.textContent = entry[1]; select.appendChild(o); });
  }
  function note() {
    var s = currentSet();
    if (!s) { $('selection-note').textContent = 'No comparable size-based set available. Browse the model below.'; $('start').disabled = true; return; }
    var matches = s.points.filter(function (p) { return p.basis === basis.value && (!format.value || p.fmt === format.value) && (!limit.value || p.x <= Number(limit.value)); });
    $('start').disabled = matches.length === 0;
    $('selection-note').textContent = matches.length + ' matching claims (zero values may not appear on log charts) · Reference: ' + s.reference + '. ' + (matches.length ? 'Opens one measurement set. ' + ((s.comparison || {}).label || 'Protocol unverified.') : 'Try another format, size limit or set.');
  }
  function formats() {
    var s = currentSet(), previous = format.value;
    var fs = s ? Array.from(new Set(s.points.filter(function (p) { return p.basis === basis.value; }).map(function (p) { return p.fmt; }))).sort() : [];
    options(format, [['', 'All formats']].concat(fs.map(function (f) { return [f, f]; })));
    if (fs.indexOf(previous) >= 0) format.value = previous;
    note();
  }
  function changeSet() {
    var s = currentSet();
    options(basis, s ? Array.from(new Set(s.points.map(function (p) { return p.basis; }))).sort().map(function (b) { return [b, b]; }) : []);
    formats();
  }
  function changeModel() {
    options(set, currentModel().sets.map(function (s) { return [s.id, s.title]; }));
    changeSet();
  }
  model.addEventListener('change', changeModel); set.addEventListener('change', changeSet); basis.addEventListener('change', formats); format.addEventListener('change', note); limit.addEventListener('change', note);
  $('start-form').addEventListener('submit', function (event) {
    event.preventDefault(); var s = currentSet(); if (!s || $('start').disabled) return;
    var params = new URLSearchParams({set:s.id, m:s.metric, basis:basis.value});
    if (format.value) params.set('fmt', format.value);
    if (limit.value) params.set('max', limit.value);
    location.href = currentModel().slug + '/?' + params.toString() + '#chart';
  });
  document.querySelectorAll('a[href]').forEach(function (a) { if (/^https?:/.test(a.getAttribute('href')) && new URL(a.getAttribute('href'), location.href).origin !== location.origin) { a.setAttribute('target', '_blank'); a.setAttribute('rel', 'noopener'); } });
  if (catalog.length) { changeModel(); $('quick').hidden = false; }
})();
