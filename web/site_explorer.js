(function () {
  var dataEl = document.getElementById('exdata'); if (!dataEl) return;
  var D = JSON.parse(dataEl.textContent), SETS = D.sets, MET = D.metrics;
  var $ = function (id) { return document.getElementById(id); };
  var NS = 'http://www.w3.org/2000/svg';
  var selSet = $('exset'), metBox = $('exmetric'), labelsCb = $('exlabels'), frontCb = $('exfront');
  var qIn = $('exq'), fmtBox = $('exfmts'), tip = $('extip'), host = $('exsvg');
  var state = { set: (SETS.some(function (x) { return x.id === 'all'; }) ? 'all' : SETS[0].id), metric: null, picked: {}, q: '', fmts: {}, sort: null, focus: null, pinned: null, xmax: null, ylim: null, onlyFront: false, basis: '', lin: false };

  // stable colour per publisher across all sets
  var pubs = []; SETS.forEach(function (s) { s.points.forEach(function (p) { if (pubs.indexOf(p.pub) < 0) pubs.push(p.pub); }); });
  pubs.sort(function (a, b) { return a.toLowerCase() < b.toLowerCase() ? -1 : 1; });
  function colour(pub) { return 'var(--s' + (pubs.indexOf(pub) % 8) + ')'; }
  var DASH = ['', '6 3', '2 3', '8 3 2 3'];
  function dash(pub) { return DASH[Math.floor(pubs.indexOf(pub) / 8) % DASH.length]; }

  function setById(id) { for (var i = 0; i < SETS.length; i++) if (SETS[i].id === id) return SETS[i]; }
  function el(tag, attrs, text) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; }
  function h(tag, attrs, html) { var e = document.createElement(tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function fmt(v) { if (v == null) return ''; var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : String(+v.toPrecision(3)); }
  function famOf(f) { f = String(f || '').trim(); if (!f) return 'other'; var m = f.match(/^[A-Za-z0-9_+.\-]+/); var k = m ? m[0].toUpperCase() : f; return k === 'OTHER' ? 'other' : k; }
  function owner(r) { r = String(r || 'unverified'); return r.indexOf('/') > 0 ? r.split('/')[0] : r; }
  function hay(p) { return (p.variant + ' ' + p.repo + ' ' + (p.fmt || '')).toLowerCase(); }
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

  // Ranking is an evidence-based, metric-specific permission. Legacy flags are never proof.
  function overview(S) { return S.id === 'all' || S.is_overview === true; }
  function comparison(S, k) {
    return S.comparisons && S.comparisons[k || state.metric] || {
      classification: 'different_or_unverified_protocol',
      label: 'Different or Unverified Protocol — Not Directly Comparable',
      explanation: 'Available methodology does not establish compatibility. Published values remain visible; direct numerical ranking is unavailable.',
      verified_matching_conditions: [], differences: [], unknowns: [{ condition: 'Protocol compatibility', explanation: 'Unverified for this metric.' }], evidence: [], ranking_permitted: false
    };
  }
  function comparable(S, k) { return !overview(S) && comparison(S, k).ranking_permitted === true; }
  function anyComparable(S) { return S.metrics.some(function (k) { return comparable(S, k); }); }
  function measurementOwner(p) { return Object.prototype.hasOwnProperty.call(p, 'measurement_owner') ? (p.measurement_owner || 'original evaluator unverified') : (p.measured_by || 'unverified'); }
  function quantizationOwner(p) { return p.quantization_owner || p.pub || 'unverified'; }
  function conditionText(c) {
    if (typeof c === 'string') return c.replace(/_/g, ' ');
    var label = c.condition || c.field || c.name || 'Methodology';
    var reason = c.explanation || c.reason || c.detail || c.message;
    var values = c.left != null && c.right != null ? ' (' + (typeof c.left === 'object' ? JSON.stringify(c.left) : String(c.left)) + ' / ' + (typeof c.right === 'object' ? JSON.stringify(c.right) : String(c.right)) + ')' : '';
    return String(label).replace(/_/g, ' ') + (reason ? ': ' + reason : '') + values;
  }
  function evidenceHtml(evidence) {
    var seen = {};
    return (evidence || []).filter(function (e) { var u = typeof e === 'string' ? e : e.url; if (!/^https?:\/\//i.test(u || '') || seen[u]) return false; seen[u] = true; return true; }).map(function (e, i) {
      var u = typeof e === 'string' ? e : e.url;
      return '<li><a href="' + esc(u) + '">' + esc(e.label || e.title || 'Publisher evidence ' + (i + 1)) + '</a>' +
        (e.snapshot_url && /^https?:\/\//i.test(e.snapshot_url) ? ' · <a href="' + esc(e.snapshot_url) + '">Captured source</a>' : '') +
        (e.sha256 ? ' <small>SHA-256: ' + esc(e.sha256) + '</small>' : '') +
        (e.source_revision_pinned === false ? ' <small>Publisher link uses a mutable revision; the captured source is content pinned.</small>' : '') +
        (e.quote ? ' <q>' + esc(e.quote) + '</q>' : '') + '</li>';
    }).join('');
  }
  function methodology(S) {
    var C = comparison(S), verified = C.verified_matching_conditions || [], differences = C.differences || [], unknowns = C.unknowns || [];
    var html = '<details class="comparison-method"><summary>Methodology &amp; evidence · ' + verified.length + ' verified · ' + (differences.length + unknowns.length) + ' different or unverified</summary>';
    html += '<p>' + esc(C.explanation || 'Compatibility has not been established.') + '</p>';
    html += '<p>' + (comparable(S) ? 'Direct numerical comparison is supported for this metric and these measurements. This measures fidelity to the stated reference; it does not establish overall model quality.' : (C.classification === 'different_base_model' ? 'Values may be viewed side by side, but their order does not rank model quality or quantization fidelity across base models.' : 'Use these published values as individual claims. Differences may reflect the evaluation protocol; direct ranking and efficiency frontiers are unavailable.')) + '</p>';
    [['Verified matching conditions', verified], ['Confirmed differences', differences], ['Unverified conditions', unknowns]].forEach(function (group) {
      if (group[1].length) html += '<p><b>' + group[0] + '</b></p><ul>' + group[1].map(function (c) { return '<li>' + esc(conditionText(c)) + '</li>'; }).join('') + '</ul>';
    });
    if ((C.limitations || []).length) html += '<p><b>Interpretation limits</b></p><ul>' + C.limitations.map(function (c) { return '<li>' + esc(conditionText(c)) + '</li>'; }).join('') + '</ul>';
    html += '<p><b>Reference</b> ' + esc(S.reference || 'unverified') + ' · <b>Evaluation text</b> ' + esc(S.data || 'unverified') + '</p>';
    if (S.note) html += '<p class="dim">' + esc(S.note) + '</p>';
    var evidence = evidenceHtml(C.evidence);
    html += evidence ? '<p><b>Supporting evidence</b></p><ul class="comparison-evidence">' + evidence + '</ul>' : '<p>Supporting protocol evidence is unavailable.</p>';
    var compatible = SETS.filter(function (other) { return other.id !== S.id && comparable(other, state.metric); });
    if (!comparable(S) && compatible.length) html += '<p><b>Verified comparison sets for this metric</b></p><div class="comparison-sets">' + compatible.map(function (other) { return '<button type="button" class="tiny comparison-jump" data-set="' + esc(other.id) + '">' + esc(other.title) + '</button>'; }).join('') + '</div><p class="dim">Compatibility applies within each listed set; it does not imply compatibility between sets.</p>';
    return html + '</details>';
  }

  // Detail and shortlist state is local to this measurement set, never a cross-set ranking.
  var detailPoint = null, shortlist = [], detailOpen = false, dismissedQid = null, suppressFocusTip = false;
  var duplicateTitles = {};
  SETS.forEach(function (s) { duplicateTitles[s.title] = (duplicateTitles[s.title] || 0) + 1; });
  function setSuffix(s) {
    var domains = [];
    s.metrics.forEach(function (k) {
      if (k.indexOf(':') < 0) return;
      var d = k.split(':').slice(1).join(':').split('/').pop();
      if (domains.indexOf(d) < 0) domains.push(d);
    });
    var text = domains.length ? domains.join(' / ') : s.data;
    return text.length > 78 ? text.slice(0, 75) + '…' : text;
  }
  function setLabel(s) {
    var suffix = '';
    if (duplicateTitles[s.title] > 1) {
      var context = setSuffix(s);
      var peers = SETS.filter(function (other) { return other.title === s.title && setSuffix(other) === context; });
      suffix = ' · ' + context + (peers.length > 1 ? ' · measurement ' + (peers.indexOf(s) + 1) : '');
    }
    return s.title + suffix + ' · ' + s.points.length + ' quants';
  }
  // ---- set chooser: grouped ----
  var groupsDef = [['Overview: every published claim', function (s) { return overview(s); }],
                   ['Curated measurement sets', function (s) { return !overview(s) && !s.auto; }],
                   ['From a single card (assembled automatically)', function (s) { return !overview(s) && s.auto; }]];
  groupsDef.forEach(function (g) {
    var og = document.createElement('optgroup'); og.label = g[0];
    SETS.filter(g[1]).slice().sort(function (a, b) { var x = a.title.split(' · ')[0].toLowerCase(), y = b.title.split(' · ')[0].toLowerCase(); return x < y ? -1 : x > y ? 1 : (a.title.toLowerCase() < b.title.toLowerCase() ? -1 : 1); }).forEach(function (s) { var o = document.createElement('option'); o.value = s.id; o.textContent = setLabel(s); o.dataset.search = (setLabel(s) + ' ' + s.data + ' ' + s.metrics.map(function (k) { return MET[k].label; }).join(' ')).toLowerCase(); og.appendChild(o); });
    if (og.children.length) selSet.appendChild(og);
  });
  var setSearch = $('exsetsearch');
  function filterSets() {
    var terms = setSearch ? setSearch.value.trim().toLowerCase().split(/\s+/).filter(Boolean) : [];
    Array.prototype.forEach.call(selSet.options, function (o) {
      // The selected option stays available: typing never changes the active measurement.
      o.hidden = o.value !== state.set && !terms.every(function (term) { return o.dataset.search.indexOf(term) >= 0; });
    });
    Array.prototype.forEach.call(selSet.querySelectorAll('optgroup'), function (g) { g.hidden = Array.prototype.every.call(g.children, function (o) { return o.hidden; }); });
  }
  if (setSearch) setSearch.addEventListener('input', filterSets);
  var saved = load('qff.set'); if (saved && setById(saved)) state.set = saved;
  if (/^#set-[a-z0-9-]+$/.test(location.hash) && setById(location.hash.slice(5))) state.set = location.hash.slice(5);
  selSet.value = state.set;
  function goSet(id, focusQid) {
    try { if (window.umami && id !== state.set) window.umami.track('set', { id: id }); } catch (e) {}   // which comparisons people open (cookie-free stats)
    state.set = id; selSet.value = id; state.metric = null; state.q = ''; qIn.value = ''; state.fmts = {}; state.sort = null; state.xmax = null; state.ylim = null; state.basis = ''; state.focus = focusQid || null; state.pinned = focusQid || null;
    detailPoint = null; shortlist = []; detailOpen = false; filterSets();
    store('qff.set', id); render();
  }
  selSet.addEventListener('change', function () { goSet(selSet.value); });
  labelsCb.addEventListener('change', function () { store('qcl.labels', labelsCb.checked ? '1' : '0'); drawChart(); });
  frontCb.addEventListener('change', function () { renderBody(); });
  labelsCb.checked = load('qcl.labels') === '1';

  // ---- find a quant across all sets ----
  var findIn = $('exfind'), findRes = $('exfindres');
  function find() {
    var s = findIn.value.trim().toLowerCase(); if (s.length < 2) { findRes.hidden = true; findRes.innerHTML = ''; return; }
    var terms = s.split(/\s+/), byQ = {}, order = [];
    SETS.forEach(function (S) { S.points.forEach(function (p) {
      var t = hay(p); if (!terms.every(function (w) { return t.indexOf(w) >= 0; })) return;
      if (!byQ[p.qid]) { byQ[p.qid] = { p: p, sets: [] }; order.push(p.qid); }
      if (byQ[p.qid].sets.indexOf(S) < 0) byQ[p.qid].sets.push(S);
    }); });
    order.sort(function (a, b) { return byQ[b].sets.filter(function (S) { return anyComparable(S); }).length - byQ[a].sets.filter(function (S) { return anyComparable(S); }).length; });
    var html = order.slice(0, 14).map(function (q) {
      var o = byQ[q]; return '<li><span class="fr-name">' + esc(o.p.variant) + ' <small>' + esc(o.p.repo) + '</small></span><span class="fr-sets">' +
        o.sets.map(function (S) { return '<button type="button" class="tiny' + (comparable(S) ? '' : ' nc') + '" data-set="' + S.id + '" data-q="' + esc(q) + '">' + esc(S.title.split(' · ')[0] + (S.auto || !comparable(S) ? '' : ' · ' + S.title.split(' · ')[1])) + '</button>'; }).join('') + '</span></li>';
    }).join('');
    findRes.innerHTML = order.length ? '<p class="dim">' + order.length + ' quant' + (order.length > 1 ? 's' : '') + ' match' + (order.length > 14 ? ', showing 14' : '') + '. Pick a set to open it there.</p><ul>' + html + '</ul>' : '<p class="dim">No measured quant matches “' + esc(s) + '”. It may be in the ledger without a comparable measurement.</p>';
    findRes.hidden = false;
  }
  findIn.addEventListener('input', find);
  findIn.addEventListener('keydown', function (e) { if (e.key === 'Escape') { findIn.value = ''; find(); } });
  findRes.addEventListener('click', function (e) { var b = e.target.closest('button[data-set]'); if (!b) return; findIn.value = ''; find(); goSet(b.dataset.set, b.dataset.q); $('explorer').scrollIntoView({ block: 'start' }); });

  // ---- picker filters ----
  qIn.addEventListener('input', function () { state.q = qIn.value.trim().toLowerCase(); renderBody(); });
  function matches(p) {
    if (state.q && !state.q.split(/\s+/).every(function (w) { return hay(p).indexOf(w) >= 0; })) return false;
    var on = Object.keys(state.fmts).filter(function (k) { return state.fmts[k]; });
    return !on.length || on.indexOf(famOf(p.fmt)) >= 0;
  }
  function isPicked(p) { return state.picked[p.qid] !== false; }
  function inRange(p) {
    var k = state.metric, M = MET[k];
    if (state.basis && (p.basis || 'unknown') !== state.basis) return false;
    if (state.xmax != null && p.x > state.xmax + 1e-9) return false;
    if (state.ylim != null) { if (!p.m[k] || p.m[k].v == null) return false; if (M.better === 'lower' ? p.m[k].v > state.ylim * (1 + 1e-9) : p.m[k].v < state.ylim * (1 - 1e-9)) return false; }
    return true;
  }
  function base(S) { return S.points.filter(function (p) { return matches(p) && isPicked(p) && inRange(p); }); }
  function shown(S) {
    var b = base(S); if (!(state.onlyFront && comparable(S))) return b;
    var k = state.metric, F = frontier(b.filter(function (p) { return p.m[k] && p.m[k].v != null; }), k);
    return b.filter(function (p) { return F[p.qid]; });
  }
  // a quant is dominated if another shown quant in the same set is no larger and strictly better (or smaller and no worse)
  function dominator(p, pool, k) {
    var lower = MET[k].better === 'lower', v = p.m[k] && p.m[k].v; if (v == null) return null; var best = null;
    pool.forEach(function (q) { if (q === p || q.is_ref || !q.m[k] || q.m[k].v == null) return; var w = q.m[k].v;
      var better = lower ? w < v : w > v, noWorse = lower ? w <= v : w >= v;
      if ((q.x <= p.x && better) || (q.x < p.x && noWorse)) { if (!best || (lower ? w < best.m[k].v : w > best.m[k].v) || (w === best.m[k].v && q.x < best.x)) best = q; } });
    return best;
  }
  $('exall').addEventListener('click', function () { setById(state.set).points.filter(matches).forEach(function (p) { state.picked[p.qid] = true; }); renderBody(); });
  $('exnone').addEventListener('click', function () { setById(state.set).points.filter(matches).forEach(function (p) { state.picked[p.qid] = false; }); renderBody(); });
  var exBody = document.querySelector('.ex-body'), exBtn = $('exexpand');
  function setWide(on) { exBody.classList.toggle('wide', on); exBtn.setAttribute('aria-pressed', String(on)); exBtn.textContent = on ? 'Collapse list' : 'Expand list'; store('qcl.wide', on ? '1' : '0'); drawChart(); }
  exBtn.addEventListener('click', function () { setWide(!exBody.classList.contains('wide')); });
  if (load('qcl.wide') === '1') exBody.classList.add('wide');
  $('exreset').addEventListener('click', function () { state.q = ''; qIn.value = ''; state.fmts = {}; setById(state.set).points.forEach(function (p) { state.picked[p.qid] = true; }); renderBody(); });

  // ---- size / quality limits ----
  var xr = $('exxr'), yr = $('exyr'), xo = $('exxo'), yo = $('exyo'), onlyCb = $('exonly'), rclr = $('exrclear');
  function bounds(S) {
    var k = state.metric, ps = S.points.filter(function (p) { return p.m[k] && p.m[k].v != null && (!MET[k].log || p.m[k].v > 0); });
    if (!ps.length) return null;
    var xs = ps.map(function (p) { return p.x; }), ys = ps.map(function (p) { return p.m[k].v; });
    return { x0: Math.min.apply(null, xs), x1: Math.max.apply(null, xs), y0: Math.min.apply(null, ys), y1: Math.max.apply(null, ys), log: MET[k].log && Math.min.apply(null, ys) > 0 };
  }
  function yAt(B, t) { return B.log ? Math.pow(10, Math.log10(B.y0) + t * (Math.log10(B.y1) - Math.log10(B.y0))) : B.y0 + t * (B.y1 - B.y0); }
  function tAt(B, v) { return B.y1 === B.y0 ? 1 : B.log ? (Math.log10(v) - Math.log10(B.y0)) / (Math.log10(B.y1) - Math.log10(B.y0)) : (v - B.y0) / (B.y1 - B.y0); }
  function syncRange() {
    var S = setById(state.set), k = state.metric, M = MET[k], B = bounds(S), lower = M.better === 'lower';
    $('exrange').hidden = !B;
    if (!B) return;
    $('exylab').textContent = (lower ? 'Max ' : 'Min ') + M.label;
    xr.value = state.xmax == null ? 1000 : Math.round(1000 * (B.x1 === B.x0 ? 1 : (state.xmax - B.x0) / (B.x1 - B.x0)));
    var yt = state.ylim == null ? (lower ? 1 : 0) : tAt(B, state.ylim);
    yr.value = Math.round(1000 * yt);
    xo.textContent = state.xmax == null ? 'any' : '≤ ' + fmt(state.xmax) + ' ' + S.x.unit;
    yo.textContent = state.ylim == null ? 'any' : (lower ? '≤ ' : '≥ ') + fmt(state.ylim) + (M.unit === '%' ? '%' : '');
    onlyCb.checked = state.onlyFront; onlyCb.parentNode.hidden = !comparable(S);
    rclr.hidden = state.xmax == null && state.ylim == null && !state.onlyFront && !state.basis;
  }
  xr.addEventListener('input', function () { var S = setById(state.set), B = bounds(S); var t = +xr.value / 1000; state.xmax = t >= 0.999 ? null : B.x0 + t * (B.x1 - B.x0); syncRange(); syncExtras(); renderBody(); });
  yr.addEventListener('input', function () { var S = setById(state.set), B = bounds(S), lower = MET[state.metric].better === 'lower'; var t = +yr.value / 1000; state.ylim = (lower ? t >= 0.999 : t <= 0.001) ? null : yAt(B, t); syncRange(); renderBody(); });
  onlyCb.addEventListener('change', function () { state.onlyFront = onlyCb.checked; syncRange(); renderBody(); });
  rclr.addEventListener('click', function () { state.xmax = null; state.ylim = null; state.onlyFront = false; state.basis = ''; syncRange(); syncExtras(); renderBody(); });

  // ---- size budget presets, size type, axis scale, shareable link ----
  var TIERS = [8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256];
  function syncExtras() {
    var S = setById(state.set), B = bounds(S), box = $('expresets'); box.innerHTML = '';
    if (B && S.x.unit === 'GB') TIERS.filter(function (t) { return t > B.x0 && t < B.x1; }).forEach(function (t) {
      var b = h('button', { type: 'button', class: 'fchip' + (state.xmax === t ? ' on' : ''), title: 'Only quants of at most ' + t + ' GB on this set\u2019s size measure' }, '≤ ' + t + ' GB');
      b.addEventListener('click', function () { state.xmax = state.xmax === t ? null : t; syncRange(); syncExtras(); renderBody(); }); box.appendChild(b);
    });
    var bases = {}; S.points.forEach(function (p) { var b = p.basis || 'unknown'; bases[b] = (bases[b] || 0) + 1; });
    var bk = Object.keys(bases), sel = $('exbasis'); $('exbasiswrap').hidden = bk.length < 2;
    sel.innerHTML = '<option value="">All size types (' + S.points.length + ')</option>' + bk.sort(function (a, b) { return bases[b] - bases[a]; }).map(function (b) { return '<option value="' + esc(b) + '"' + (state.basis === b ? ' selected' : '') + '>' + esc(b) + ' (' + bases[b] + ')</option>'; }).join('');
    var lg = MET[state.metric] && MET[state.metric].log; $('exlog').parentNode.hidden = !lg;
    $('exlog').setAttribute('aria-pressed', String(!state.lin)); $('exlin').setAttribute('aria-pressed', String(state.lin));
  }
  $('exbasis').addEventListener('change', function () { state.basis = this.value; syncRange(); renderBody(); });
  $('exlog').addEventListener('click', function () { state.lin = false; syncExtras(); drawChart(); });
  $('exlin').addEventListener('click', function () { state.lin = true; syncExtras(); drawChart(); });
  function viewParams() { var q = new URLSearchParams(); q.set('set', state.set); if (state.metric) q.set('m', state.metric); if (state.q) q.set('q', state.q); if (state.xmax != null) q.set('max', fmt(state.xmax)); if (state.onlyFront) q.set('eff', '1'); if (state.basis) q.set('basis', state.basis); Object.keys(state.fmts).filter(function (f) { return state.fmts[f]; }).sort().forEach(function (f) { q.append('fmt', f); }); return q; }
  $('exlink').addEventListener('click', function () {
    var btn = this, url = location.href.split(/[?#]/)[0] + '?' + viewParams().toString() + '#chart';
    function done(ok) { btn.textContent = ok ? 'Link copied' : url; setTimeout(function () { btn.textContent = 'Copy link to view'; }, 2500); }
    try { navigator.clipboard.writeText(url).then(function () { done(true); }, function () { done(false); }); } catch (e) { done(false); }
  });
  try {   // restore a shared view (works where the page has a query string, e.g. GitHub Pages)
    var U = new URLSearchParams(location.search);
    if (U.get('set') && setById(U.get('set'))) { state.set = U.get('set'); selSet.value = state.set; }
    if (U.get('m')) state.metric = U.get('m');
    if (U.get('q')) { state.q = U.get('q').toLowerCase(); qIn.value = U.get('q'); }
    if (U.get('max') && Number.isFinite(+U.get('max')) && +U.get('max') > 0) state.xmax = +U.get('max');
    var allowedFormats = setById(state.set).points.map(function (p) { return famOf(p.fmt); });
    U.getAll('fmt').forEach(function (f) { if (allowedFormats.indexOf(f) >= 0) state.fmts[f] = true; });
    if (U.get('eff') === '1') state.onlyFront = true;
    if (U.get('basis')) state.basis = U.get('basis');
  } catch (e) {}
  document.addEventListener('keydown', function (e) { if (e.key === '/' && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); findIn.focus(); } });

  function render() {
    var S = setById(state.set);
    if (!state.metric || S.metrics.indexOf(state.metric) < 0) {   // a quant found or jumped to opens on a metric it reports, so it can be pinned
      var fq = state.focus && S.points.filter(function (p) { return p.qid === state.focus; })[0];
      state.metric = (fq && fq.m && S.metrics.filter(function (k) { return fq.m[k] != null; })[0]) || S.metrics[0];
    }
    metBox.innerHTML = '';
    // many per-corpus series: show the most reported few, the rest behind "More metrics" (the chosen one always shows)
    var cnt = {}; S.metrics.forEach(function (k) { cnt[k] = S.points.filter(function (p) { return p.m[k] && p.m[k].v != null; }).length; });
    var ordered = S.metrics.slice(), FEW = 6, many = ordered.length > FEW + 2;
    var top = many ? ordered.slice().sort(function (a, b) { return cnt[b] - cnt[a] || S.metrics.indexOf(a) - S.metrics.indexOf(b); }).slice(0, FEW) : ordered;
    ordered.forEach(function (k) {
      var hide = many && !state.allMetrics && top.indexOf(k) < 0 && k !== state.metric;
      var b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(k === state.metric), class: 'mchip' + (k === state.metric ? ' on' : '') }, esc(MET[k].label) + (many ? ' <span class="dim">' + cnt[k] + '</span>' : ''));
      b.hidden = hide;
      b.addEventListener('click', function () { state.metric = k; state.sort = null; state.ylim = null; render(); }); metBox.appendChild(b);
    });
    if (many) {
      var mb = h('button', { type: 'button', class: 'mchip more' }, state.allMetrics ? 'Fewer metrics' : 'More metrics (' + (ordered.length - FEW) + ')');
      mb.addEventListener('click', function () { state.allMetrics = !state.allMetrics; render(); }); metBox.appendChild(mb);
    }
    var C = comparison(S);
    $('exmeta').innerHTML = methodology(S);
    Array.prototype.forEach.call($('exmeta').querySelectorAll('.comparison-jump'), function (b) { b.addEventListener('click', function () { var k = state.metric; goSet(b.dataset.set); if (setById(state.set).metrics.indexOf(k) >= 0) { state.metric = k; render(); } }); });
    if ($('exscope')) {
      $('exscope').textContent = (overview(S) ? 'Mixed protocols · ' : '') + (C.label || 'Different or Unverified Protocol — Not Directly Comparable') + ' · ' + (comparable(S) ? 'Direct comparison supported for ' + MET[state.metric].label + '.' : 'Published claims remain visible; direct ranking is unavailable.');
      $('exscope').classList.toggle('mixed', !comparable(S));
      $('exscope').dataset.classification = C.classification;
    }
    // format chips for this set
    var fm = {}; S.points.forEach(function (p) { var f = famOf(p.fmt); fm[f] = (fm[f] || 0) + 1; });
    var fks = Object.keys(fm).sort(function (a, b) { return fm[b] - fm[a] || (a < b ? -1 : 1); });
    fmtBox.innerHTML = ''; fmtBox.hidden = fks.length < 2;
    var TOP = 8, extra = fks.length - TOP;
    fks.forEach(function (f, i) {
      var b = h('button', { type: 'button', class: 'fchip' + (state.fmts[f] ? ' on' : ''), 'aria-pressed': String(!!state.fmts[f]), title: 'Show only ' + f + ' (click again to clear)' }, esc(f) + ' <span>' + fm[f] + '</span>');
      if (i >= TOP && !state.moreFmts && !state.fmts[f]) b.hidden = true;
      b.addEventListener('click', function () { state.fmts[f] = !state.fmts[f]; render(); }); fmtBox.appendChild(b);
    });
    if (extra > 0) { var mb = h('button', { type: 'button', class: 'fchip more' }, state.moreFmts ? 'fewer' : '+' + extra + ' more'); mb.addEventListener('click', function () { state.moreFmts = !state.moreFmts; render(); }); fmtBox.appendChild(mb); }
    frontCb.parentNode.hidden = !comparable(S);
    syncRange(); syncExtras();
    renderBody();
  }
  function renderBody() { var S = setById(state.set); picker(S); drawChart(); table(S); renderDetail(); }

  function picker(S) {
    var box = $('expick'); box.innerHTML = '';
    var vis = S.points.filter(matches), n = shown(S).length, lim = state.xmax != null || state.ylim != null || (state.onlyFront && comparable(S));
    $('excount').textContent = n + ' of ' + S.points.length + ' shown' + (lim ? ' (limits on)' : '');
    $('exreset').hidden = n === S.points.length && !state.q;
    var shownSet = {}; shown(S).forEach(function (p) { shownSet[p.qid] = true; });
    var groups = {}; vis.forEach(function (p) { (groups[p.repo] = groups[p.repo] || []).push(p); });
    var keys = Object.keys(groups).sort(function (a, b) { return a.toLowerCase() < b.toLowerCase() ? -1 : 1; });
    if (!keys.length) { box.innerHTML = '<p class="dim">No quant in this set matches the filter.</p>'; return; }
    keys.forEach(function (repo) {
      var ps = groups[repo].slice().sort(function (a, b) { return a.x - b.x; });
      var g = h('fieldset', { class: 'pgroup' });
      var on = ps.filter(isPicked).length;
      var gid = 'pg-' + S.id + '-' + S.points.indexOf(ps[0]);
      var lg = h('legend', {});
      var gcb = h('input', { type: 'checkbox', id: gid, 'aria-label': 'All quants from ' + repo }); gcb.checked = on === ps.length; gcb.indeterminate = on > 0 && on < ps.length;
      gcb.addEventListener('change', function () { ps.forEach(function (p) { state.picked[p.qid] = gcb.checked; }); renderBody(); });
      var gl = h('label', { for: gid }, '<span class="sw" style="background:' + colour(ps[0].pub) + '"></span><span class="pg-name">' + esc(owner(repo)) + '<small>' + esc(repo.indexOf('/') > 0 ? repo.split('/')[1] : '') + '</small></span>');
      lg.appendChild(gcb); lg.appendChild(gl); g.appendChild(lg);
      ps.forEach(function (p) {
        var id = 'pk-' + S.id + '-' + S.points.indexOf(p);
        var lab = h('label', { for: id, class: 'pitem' + (isPicked(p) && shownSet[p.qid] ? '' : isPicked(p) ? ' out' : ''), 'data-qid': p.qid, title: isPicked(p) && !shownSet[p.qid] ? 'Ticked, but outside the current limits' : '' });
        var cb = h('input', { type: 'checkbox', id: id }); cb.checked = isPicked(p);
        cb.addEventListener('change', function () { state.picked[p.qid] = cb.checked; renderBody(); });
        lab.appendChild(cb); lab.appendChild(h('span', {}, esc(p.variant) + (p.kind === 'cross' ? ' <span class="chip x">by ' + esc(owner(p.measured_by)) + '</span>' : '')));
        lab.addEventListener('pointerenter', function () { hl(p.qid, true); }); lab.addEventListener('pointerleave', function () { hl(p.qid, false); });
        g.appendChild(lab);
      });
      box.appendChild(g);
    });
  }

  function frontier(pts, k) {
    var lower = MET[k].better === 'lower', best = null, bx = null, out = {};
    pts.filter(function (p) { return !p.is_ref; }).sort(function (a, b) { return a.x - b.x || (lower ? a.m[k].v - b.m[k].v : b.m[k].v - a.m[k].v); }).forEach(function (p) {
      var v = p.m[k].v; if (best === null || (lower ? v < best : v > best)) { out[p.qid] = true; best = v; bx = p.x; } else if (v === best && p.x === bx) out[p.qid] = true;
    });
    return out;
  }

  var lastPts = [];
  function drawChart() {
    var S = setById(state.set), k = state.metric, M = MET[k];
    var pts = shown(S).filter(function (p) { return p.m[k] && p.m[k].v != null && (!M.log || p.m[k].v > 0); });
    lastPts = pts; host.innerHTML = ''; hideTip(true);
    var cw = Math.max(300, (host.clientWidth || 760) - 24), narrow = cw < 560;
    var W = cw, H = narrow ? Math.round(cw * 0.95) : Math.min(500, Math.round(cw * 0.6)), ml = narrow ? 48 : 62, mr = narrow ? 10 : 18, mt = 34, mb = narrow ? 44 : 50;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, class: 'chart', role: 'img', 'aria-label': M.label + ' against ' + S.x.label + ' for ' + pts.length + ' quants in ' + S.title });
    host.appendChild(svg);
    var lg = $('exlegend'); lg.innerHTML = '';
    if (pts.length === 0) { legend(S, svg); svg.appendChild(el('text', { x: W / 2, y: H / 2, class: 'axis', 'text-anchor': 'middle' }, S.points.some(function (p) { return p.m[k]; }) ? 'No quant selected. Tick some in the list.' : 'No quant in this set has this metric.')); return; }
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.m[k].v; });
    var ref = S.refs[k]; if (ref) ys.push(ref[0]);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs); var xp = (x1 - x0) * 0.08 || Math.max(1, x0 * 0.05); x0 -= xp; x1 += xp;
    var y0, y1, Y;
    var useLog = M.log && !state.lin;
    if (useLog) {
      y0 = Math.pow(10, Math.log10(Math.min.apply(null, ys)) - 0.12); y1 = Math.pow(10, Math.log10(Math.max.apply(null, ys)) + 0.12);
      Y = function (v) { return mt + (Math.log10(y1) - Math.log10(v)) / (Math.log10(y1) - Math.log10(y0)) * (H - mt - mb); };
    } else {
      var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.02 || 1; y0 = lo - pad; y1 = hi + pad;
      Y = function (v) { return mt + (y1 - v) / (y1 - y0) * (H - mt - mb); };
    }
    var X = function (v) { return ml + (v - x0) / (x1 - x0) * (W - ml - mr); };
    function nice(a, b, n) { var step = Math.pow(10, Math.floor(Math.log10((b - a) / n))); var err = (b - a) / n / step; step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1; var t = []; for (var v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) t.push(+v.toFixed(10)); return t; }
    nice(x0, x1, narrow ? 4 : 7).forEach(function (v) { svg.appendChild(el('line', { x1: X(v), x2: X(v), y1: mt, y2: H - mb, class: 'grid' })); svg.appendChild(el('text', { x: X(v), y: H - mb + 17, class: 'tick', 'text-anchor': 'middle' }, fmt(v))); });
    if (useLog) {
      for (var d = Math.pow(10, Math.floor(Math.log10(y0))); d <= y1 * 1.0001; d *= 10) [1, 2, 5].forEach(function (mlt) { var v = d * mlt; if (v >= y0 && v <= y1) { svg.appendChild(el('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: mlt === 1 ? 'grid' : 'grid minor' })); svg.appendChild(el('text', { x: ml - 7, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, String(+v.toPrecision(2)))); } });
    } else nice(y0, y1, 6).forEach(function (v) { svg.appendChild(el('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: 'grid' })); svg.appendChild(el('text', { x: ml - 7, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, fmt(v))); });
    svg.appendChild(el('text', { x: (ml + W - mr) / 2, y: H - 10, class: 'axis', 'text-anchor': 'middle' }, narrow ? S.x.label.replace(/\s*\(.*$/, '') : S.x.label));
    svg.appendChild(el('text', { x: ml - (narrow ? 40 : 54), y: 16, class: 'axis' }, M.label + (M.unit ? ' (' + M.unit + ')' : '') + (useLog ? ', log' : '') + ' · ' + (M.better === 'lower' ? '↓' : '↑') + ' ' + M.better + ' is better'));
    if (ref) { svg.appendChild(el('line', { x1: ml, x2: W - mr, y1: Y(ref[0]), y2: Y(ref[0]), class: 'refline' })); svg.appendChild(el('text', { x: ml + 6, y: Y(ref[0]) - 6, class: 'reflabel' }, ref[1])); }
    // series lines: one per file publisher and build family, never across measurers; none in the not-comparable view
    function fam(p) { var v = p.variant.toLowerCase(); return /affine/.test(v) ? ' · affine' : /^vq/.test(v) ? ' · VQ' : /q4_0 ple/.test(v) ? ' · Q4_0 n-gram' : ''; }
    var groups = {}; pts.forEach(function (p) { (groups[p.repo + fam(p)] = groups[p.repo + fam(p)] || []).push(p); });
    if (comparable(S)) Object.keys(groups).forEach(function (key) {
      var g = groups[key].slice().sort(function (a, b) { return a.x - b.x; }); if (g.length < 2) return;
      svg.appendChild(el('polyline', { points: g.map(function (p) { return X(p.x).toFixed(1) + ',' + Y(p.m[k].v).toFixed(1); }).join(' '), class: 'sline', 'data-repo': g[0].repo, style: 'stroke:' + colour(g[0].pub), 'stroke-dasharray': dash(g[0].pub) }));
    });
    var F = comparable(S) && (frontCb.checked || state.onlyFront) ? frontier(pts, k) : {};
    var fp = pts.filter(function (p) { return F[p.qid]; }).sort(function (a, b) { return a.x - b.x; });
    if (fp.length > 1) {
      var step = []; fp.forEach(function (p, i) { if (i) step.push(X(p.x).toFixed(1) + ',' + Y(fp[i - 1].m[k].v).toFixed(1)); step.push(X(p.x).toFixed(1) + ',' + Y(p.m[k].v).toFixed(1)); });
      svg.appendChild(el('polyline', { points: step.join(' '), class: 'frontier' }));
    }
    var boxes = [], r = narrow ? 6 : 5.4;
    pts.slice().sort(function (a, b) { return a.x - b.x; }).forEach(function (p) {
      var cx = X(p.x), cy = Y(p.m[k].v);
      if (F[p.qid]) svg.appendChild(el('circle', { cx: cx.toFixed(1), cy: cy.toFixed(1), r: r + 3.5, class: 'fring' }));
      var c = el('circle', { cx: cx.toFixed(1), cy: cy.toFixed(1), r: r, class: 'mk' + (p.kind === 'cross' ? ' cross' : ''), 'data-qid': p.qid, style: (p.is_ref ? 'fill:var(--panel, #fff);stroke:' : 'fill:') + colour(p.pub), tabindex: '0', role: 'button',
        'aria-label': p.variant + ' from ' + p.repo + ': ' + M.label + ' ' + p.m[k].raw + ', ' + fmt(p.x) + ' ' + S.x.unit });
      c.addEventListener('pointerenter', function (e) { if (e.pointerType === 'mouse') { if (dismissedQid !== p.qid) dismissedQid = null; showTip(p, c); hl(p.qid, true); } });
      c.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse') { hl(p.qid, false); if (state.pinned !== p.qid) hideTip(); } });
      c.addEventListener('focus', function () { if (suppressFocusTip) return; showTip(p, c, state.pinned === p.qid); hl(p.qid, true); });
      c.addEventListener('blur', function () { hl(p.qid, false); if (state.pinned !== p.qid) hideTip(); });
      function activatePoint(keyboard) {
        var hadFocus = document.activeElement === c;
        dismissedQid = null; detailPoint = p; detailOpen = true; state.pinned = p.qid;
        drawChart(); renderDetail();
        var mobileClose = keyboard && window.matchMedia('(max-width: 700px)').matches && $('exdetail') && $('exdetail').querySelector('.quant-close');
        if (mobileClose) mobileClose.focus({ preventScroll: true });
        else if (hadFocus) {
          var replacement = host.querySelector('circle.mk[data-qid="' + cssq(p.qid) + '"]');
          if (replacement) { suppressFocusTip = true; replacement.focus({ preventScroll: true }); suppressFocusTip = false; }
        }
      }
      c.addEventListener('click', function (e) { activatePoint(e.detail === 0); });
      c.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activatePoint(true); }
        if (e.key === 'Escape') { dismissedQid = p.qid; state.pinned = null; hideTip(true); }
      });
      svg.appendChild(c); boxes.push({ x0: cx - r - 1, x1: cx + r + 1, y0: cy - r - 1, y1: cy + r + 1 });
    });
    var maxLab = narrow ? 14 : 36;
    var labelPts = labelsCb.checked && pts.length <= maxLab ? pts : pts.filter(function (p) { return detailPoint === p; });
    if (labelPts.length) {
      var placed = boxes.slice();
      labelPts.slice().sort(function (a, b) { return a.x - b.x; }).forEach(function (p) {
        var cx = X(p.x), cy = Y(p.m[k].v), lim = narrow ? 14 : 22;
        var name = p.variant.length > lim ? p.variant.slice(0, lim - 1) + '…' : p.variant, val = p.m[k].raw.replace(/\s*±.*$/, '').replace(/\*\*/g, '');
        var w = Math.max(name.length, val.length) * 6.4 + 6, hgt = 26;
        var cands = [[0, -12 - hgt], [0, 12], [w / 2 + 10, -hgt / 2], [-w / 2 - 10, -hgt / 2], [0, -26 - hgt], [0, 26]];
        for (var i = 0; i < cands.length; i++) {
          var lx = Math.min(Math.max(cx + cands[i][0] - w / 2, ml + 2), W - mr - w);
          var bx = { x0: lx, x1: lx + w, y0: cy + cands[i][1], y1: cy + cands[i][1] + hgt };
          if (bx.y0 < mt - 8 || bx.y1 > H - mb) continue;
          if (placed.some(function (q) { return !(bx.x1 < q.x0 || bx.x0 > q.x1 || bx.y1 < q.y0 || bx.y0 > q.y1); })) continue;
          placed.push(bx);
          var t = el('text', { x: (bx.x0 + bx.x1) / 2, y: bx.y0 + 11, class: 'plab', 'data-qid': p.qid, 'text-anchor': 'middle', style: 'fill:' + colour(p.pub) });
          t.appendChild(el('tspan', { x: (bx.x0 + bx.x1) / 2 }, name));
          t.appendChild(el('tspan', { x: (bx.x0 + bx.x1) / 2, dy: 12, class: 'pval' }, val));
          svg.appendChild(t); break;
        }
      });
    }
    legend(S, svg);
    // legend: one toggle per file repo in the set (drawn even when every quant is hidden, so they can be shown again)
    function legend(S, svg) {
      var reps = {}; S.points.filter(matches).forEach(function (p) { (reps[p.repo] = reps[p.repo] || []).push(p); });
      Object.keys(reps).sort(function (a, b) { return a.toLowerCase() < b.toLowerCase() ? -1 : 1; }).forEach(function (repo) {
        var ps = reps[repo], on = ps.some(isPicked);
        var b = h('button', { type: 'button', class: 'lg' + (on ? '' : ' off'), 'aria-pressed': String(on), title: (on ? 'Hide ' : 'Show ') + repo }, '<span class="sw" style="background:' + colour(ps[0].pub) + '"></span>' + esc(repo) + ' <span class="dim">' + ps.length + '</span>');
        b.addEventListener('click', function () { ps.forEach(function (p) { state.picked[p.qid] = !on; }); renderBody(); });
        b.addEventListener('pointerenter', function () { svg.classList.add('dimmed'); Array.prototype.forEach.call(svg.querySelectorAll('[data-qid]'), function (n) { if (ps.some(function (p) { return p.qid === n.getAttribute('data-qid'); })) n.classList.add('hl'); }); });
        b.addEventListener('pointerleave', function () { svg.classList.remove('dimmed'); Array.prototype.forEach.call(svg.querySelectorAll('.hl'), function (n) { n.classList.remove('hl'); }); });
        lg.appendChild(b);
      });
      var nrep = lg.children.length;
      if (nrep > 16 && !state.legendOpen) { Array.prototype.forEach.call(lg.children, function (c) { c.hidden = true; }); }
      if (nrep > 16) { var tb = h('button', { type: 'button', class: 'lg more' }, state.legendOpen ? 'Hide legend' : 'Show legend (' + nrep + ' repos)'); tb.addEventListener('click', function () { state.legendOpen = !state.legendOpen; drawChart(); }); lg.appendChild(tb); }
    }
    var note = [];
    if (labelsCb.checked && pts.length > maxLab) note.push('Labels hidden with more than ' + maxLab + ' points: filter, or tap a point.');
    if (fp.length) note.push('Frontier: no other shown quant is both smaller and better on this metric, in this set only' + (pts.some(function (p) { return p.is_ref; }) ? '; the reference build is not ranked.' : '.'));
    $('exchartnote').textContent = note.join(' ');
    if (state.focus) { var fc = svg.querySelector('circle.mk[data-qid="' + cssq(state.focus) + '"]'); var fpnt = pts.filter(function (p) { return p.qid === state.focus; })[0]; if (fc && fpnt) { showTip(fpnt, fc, true); fc.classList.add('pulse'); } state.focus = null; }
    else if (state.pinned) { var pc = svg.querySelector('circle.mk[data-qid="' + cssq(state.pinned) + '"]'), pp = pts.filter(function (p) { return p.qid === state.pinned; })[0]; if (pc && pp) showTip(pp, pc, true); else state.pinned = null; }
  }
  function closeDetail() {
    detailOpen = false; dismissedQid = detailPoint && detailPoint.qid; state.pinned = null; hideTip(true); renderDetail();
    var marker = detailPoint && host.querySelector('circle.mk[data-qid="' + cssq(detailPoint.qid) + '"]');
    var target = marker || $('exdetailopen');
    if (target) { suppressFocusTip = true; target.focus({ preventScroll: true }); suppressFocusTip = false; }

  }
  function compatibleLinks(p, S) {
    var ids = p.compatible_sets || [];
    var sets = ids.map(setById).filter(function (other) { return other && other.id !== S.id && comparable(other); });
    return sets.length ? '<p><b>Compare this measurement in a verified set</b></p><div class="comparison-sets">' + sets.map(function (other) { return '<button type="button" class="tiny comparison-jump" data-set="' + esc(other.id) + '">' + esc(other.title) + '</button>'; }).join('') + '</div>' : '';
  }
  function renderDetail() {
    var box = $('exdetail'); if (!box) return;
    var S = setById(state.set), p = detailPoint, k = state.metric;
    var openBtn = $('exdetailopen');
    if (!openBtn) {
      openBtn = h('button', { id: 'exdetailopen', type: 'button', class: 'tiny', 'aria-controls': 'exdetail' }, 'Selected quant');
      box.parentNode.insertBefore(openBtn, box);
      openBtn.addEventListener('click', function () { detailOpen = !detailOpen; renderDetail(); });
    }
    openBtn.hidden = !p;
    openBtn.setAttribute('aria-expanded', String(!!p && detailOpen));
    openBtn.textContent = 'Selected quant' + (shortlist.length ? ' · shortlist ' + shortlist.length : '');
    box.hidden = !p || !detailOpen; box.dataset.open = String(!!p && detailOpen);
    if (!p) { box.innerHTML = ''; return; }
    var metrics = S.metrics.filter(function (m) { return p.m[m]; });
    box.innerHTML = '<button type="button" class="quant-close tiny" aria-label="Close selected quant">Close ×</button>' +
      '<p class="dim">Selected quant</p><h3>' + esc(p.variant) + '</h3><p class="tt-repo">' + esc(p.repo) + (p.fmt ? ' · ' + esc(p.fmt) : '') + '</p>' +
      '<div class="tt-grid"><div><span>' + esc(S.x.label) + '</span><b>' + fmt(p.x) + ' ' + esc(S.x.unit) + '</b></div>' +
      metrics.map(function (m) { return '<div><span>' + esc(MET[m].label) + (p.m[m].series ? ' · ' + esc(p.m[m].series.split(':').slice(1).join(':')) : '') + '</span><b>' + esc(p.m[m].raw) + '</b></div>'; }).join('') + '</div>' +
      (shown(S).indexOf(p) < 0 ? '<p class="dim">This selected quant is outside the current chart filters.</p>' : '') +
      '<p class="dim">Reference: ' + esc(S.reference) + '</p>' +
      (p.basis ? '<p class="dim">Size type: ' + esc(p.basis) + '</p>' : '') +
      '<p class="dim">Quant publisher: ' + esc(quantizationOwner(p)) + '<br>Measurement publisher: ' + esc(measurementOwner(p)) + '</p>' +
      (p.claim_provenance && p.claim_provenance.kind ? '<p class="dim">Claim provenance: ' + esc(p.claim_provenance.kind) + (p.claim_provenance.original_attribution ? ' · original attribution ' + (p.claim_provenance.original_attribution.verified === true ? 'verified by source evidence' : 'unverified') : '') + '</p>' : '') +
      compatibleLinks(p, S) +
      (p.is_ref ? '<p>This reference build is not ranked for efficiency.</p>' : '') +
      (p.inferred ? '<p>The source names this file but not its repository; attribution is inferred from the filename.</p>' : '') +
      '<p><a href="' + esc(p.src) + '">View source evidence: ' + esc(p.src_label) + '</a></p>' +
      (comparable(S) ? '<button type="button" class="tiny" id="exshortadd">' + (shortlist.indexOf(p) >= 0 ? 'Remove from shortlist' : 'Add to shortlist') + '</button><p class="dim">Shortlist stays within this measurement set and clears when you switch sets. Values use the current metric; no ranking is implied.</p>' : '<p class="dim">A shortlist requires verified protocol compatibility for this metric.</p>') +
      (shortlist.length ? '<ul class="quant-shortlist">' + shortlist.map(function (q, i) { return '<li><b>' + esc(q.variant) + '</b> · ' + fmt(q.x) + ' ' + esc(S.x.unit) + ' · ' + esc(MET[k].label) + ': ' + (q.m[k] ? esc(q.m[k].raw) : 'not reported') + ' <button type="button" class="tiny" data-short-remove="' + i + '" aria-label="Remove ' + esc(q.variant) + ' from shortlist">Remove</button></li>'; }).join('') + '</ul>' : '');
    box.querySelector('.quant-close').addEventListener('click', closeDetail);
    Array.prototype.forEach.call(box.querySelectorAll('.comparison-jump'), function (b) { b.addEventListener('click', function () {
      var target = setById(b.dataset.set), metric = state.metric;
      var point = target.points.filter(function (q) { return p.measurement_id ? q.measurement_id === p.measurement_id : q.qid === p.qid; })[0];
      goSet(target.id, point ? point.qid : null);
      if (target.metrics.indexOf(metric) >= 0) { state.metric = metric; render(); }
    }); });
    var add = $('exshortadd'); if (add) add.addEventListener('click', function () { var i = shortlist.indexOf(p); if (i >= 0) shortlist.splice(i, 1); else shortlist.push(p); renderDetail(); $('exshortadd').focus(); });
    Array.prototype.forEach.call(box.querySelectorAll('[data-short-remove]'), function (b) { b.addEventListener('click', function () { var removed = +b.dataset.shortRemove; shortlist.splice(removed, 1); renderDetail(); var buttons = box.querySelectorAll('[data-short-remove]'); var next = buttons[Math.min(removed, buttons.length - 1)] || $('exshortadd'); if (next) next.focus(); }); });
  }
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && detailOpen) closeDetail(); });
  function cssq(s) { return String(s).replace(/["\\]/g, '\\$&'); }

  function showTip(p, c, pinned) {
    if (dismissedQid === p.qid && !pinned) return;
    pinned = !!pinned || state.pinned === p.qid;
    var S = setById(state.set), k = state.metric, M = MET[k];
    var others = S.metrics.filter(function (m) { return m !== k && p.m[m]; }).map(function (m) { return '<div><span>' + esc(MET[m].label) + (p.m[m].series ? ' · ' + esc(p.m[m].series.split(':').slice(1).join(':')) : '') + '</span><b>' + esc(p.m[m].raw) + '</b></div>'; }).join('');
    tip.innerHTML = '<p class="tt-name">' + esc(p.variant) + '</p><p class="tt-repo">' + esc(p.repo) + (p.fmt ? ' · ' + esc(p.fmt) : '') + '</p>' +
      '<div class="tt-grid"><div><span>' + esc(M.label) + (p.m[k].series ? ' · ' + esc(p.m[k].series.split(':').slice(1).join(':')) : '') + '</span><b>' + esc(p.m[k].raw) + '</b></div>' + others +
      '<div><span>' + esc(S.x.unit) + '</span><b>' + fmt(p.x) + (p.size_raw && String(p.size_raw) !== fmt(p.x) ? ' <small>(card: ' + esc(p.size_raw) + ')</small>' : '') + '</b></div></div>' +
      '<p class="tt-by">Quant publisher: ' + esc(quantizationOwner(p)) + '<br>Measurement publisher: ' + esc(measurementOwner(p)) + '</p>' +
      (p.inferred ? '<p class="tt-by">The card names this file but not its repository; attributed from the file name.</p>' : '') +
      (p.is_ref ? '<p class="tt-by">This is the measurement\u2019s reference, so it is not ranked for efficiency.</p>' : '') +
      '<p class="tt-src"><a href="' + esc(p.src) + '">Source: ' + esc(p.src_label) + '</a>' + (pinned ? ' <button type="button" class="tiny" id="extipx" aria-label="Close">×</button>' : '') + '</p>';
    tip.hidden = false; tip.classList.toggle('pinned', !!pinned);
    var box = host.getBoundingClientRect(), cb = c.getBoundingClientRect(), par = tip.offsetParent ? tip.offsetParent.getBoundingClientRect() : box;
    var tw = tip.offsetWidth, th = tip.offsetHeight, x = cb.left - par.left + cb.width / 2 - tw / 2, y = cb.top - par.top - th - 10;
    if (y < box.top - par.top) y = cb.bottom - par.top + 10;
    x = Math.max(4, Math.min(x, par.width - tw - 4));
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
    var xb = $('extipx'); if (xb) xb.addEventListener('click', function () { state.pinned = null; hideTip(true); });
    Array.prototype.forEach.call(document.querySelectorAll('#extable tr.sel'), function (r) { r.classList.remove('sel'); });
    if (pinned) { detailPoint = p; detailOpen = true; renderDetail(); var row = document.querySelector('#extable tr[data-qid="' + cssq(p.qid) + '"]'); if (row) row.classList.add('sel'); }
  }
  function hideTip(force) { if (force || !state.pinned) { tip.hidden = true; tip.classList.remove('pinned'); } }
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && state.pinned) { state.pinned = null; hideTip(true); } });
  document.addEventListener('click', function (e) { if (state.pinned && document.contains(e.target) && !e.target.closest('#extip') && !e.target.closest('#exdetail') && !e.target.closest('#exdetailopen') && !e.target.closest('#exfindres') && !e.target.closest('.jump') && !e.target.closest('circle.mk') && !e.target.closest('#extable')) { state.pinned = null; hideTip(true); } });

  function hl(qid, on) {
    var svg = host.querySelector('svg'); if (!svg) return;
    svg.classList.toggle('dimmed', on);
    Array.prototype.forEach.call(svg.querySelectorAll('[data-qid="' + cssq(qid) + '"]'), function (n) { n.classList.toggle('hl', on); });
    var row = document.querySelector('#extable tr[data-qid="' + cssq(qid) + '"]'); if (row) row.classList.toggle('hov', on);
  }

  function table(S) {
    var t = $('extable'), ks = S.metrics, k = state.metric, pts = shown(S).slice();
    var F = comparable(S) && (frontCb.checked || state.onlyFront) ? frontier(pts.filter(function (p) { return p.m[k] && p.m[k].v != null; }), k) : {};
    var pool = base(S);
    var sort = state.sort;
    if (!sort || (sort.key.charAt(0) !== '_' && !comparable(S, sort.key))) sort = comparable(S) ? { key: k, dir: MET[k].better === 'lower' ? 1 : -1 } : { key: '_x', dir: 1 };
    function val(p) { return sort.key === '_x' ? p.x : sort.key === '_name' ? p.variant.toLowerCase() : (p.m[sort.key] ? p.m[sort.key].v : null); }
    pts.sort(function (a, b) { var x = val(a), y = val(b); if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * sort.dir; });
    function th(key, label, cls) { if (key.charAt(0) !== '_' && !comparable(S, key)) return '<th class="' + (cls || '') + '" title="Metric sorting requires verified protocol compatibility" aria-disabled="true">' + esc(label) + '</th>'; var on = sort.key === key; return '<th class="' + (cls || '') + '" data-key="' + key + '" tabindex="0" aria-sort="' + (on ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none') + '">' + esc(label) + '</th>'; }
    var head = '<thead><tr>' + th('_name', 'Quant') + th('_x', S.x.unit, 'n') + ks.map(function (m) { return th(m, MET[m].label, 'n'); }).join('') + (comparable(S) ? '<th title="Another quant in this set that is no larger and better on ' + esc(MET[k].label) + '">Beaten by</th>' : '') + '<th>Also measured in</th><th>Source</th></tr></thead>';
    var body = pts.map(function (p) {
      var others = p.inferred ? [] : SETS.filter(function (s) { return s.id !== S.id && comparable(s) && s.points.some(function (q) { return q.qid === p.qid && !q.inferred; }); });
      return '<tr data-qid="' + esc(p.qid) + '"' + (state.pinned === p.qid ? ' class="sel"' : '') + '><td><span class="sw" style="background:' + colour(p.pub) + '"></span><b>' + esc(p.variant) + '</b>' + (F[p.qid] ? ' <span class="chip f">frontier</span>' : '') + (p.kind === 'cross' ? ' <span class="chip x">measured by ' + esc(owner(p.measured_by)) + '</span>' : '') + (p.inferred ? ' <span class="chip q" title="The card names the file but not its repository">repo not named on card</span>' : '') + (p.is_ref ? ' <span class="chip">reference</span>' : '') +
        '<br><small class="dim">' + esc(p.repo) + (p.fmt ? ' · ' + esc(p.fmt) : '') + '</small></td><td class="n">' + fmt(p.x) + '</td>' +
        ks.map(function (m) { return '<td class="n' + (m === k ? ' cur' : '') + '">' + (p.m[m] ? esc(p.m[m].raw) : '<span class="dim">—</span>') + '</td>'; }).join('') +
        (comparable(S) ? '<td class="beat">' + (p.is_ref ? '<span class="chip">reference: not ranked</span>' : '') + (p.is_ref ? '' : (function () { var d = dominator(p, pool, k); return d ? esc(d.variant) + ' <small class="dim">' + fmt(d.x) + ' ' + esc(S.x.unit) + ', ' + esc(String(d.m[k].raw).replace(/\s*±.*$/, '')) + '</small>' : (p.m[k] ? '<span class="chip f">none: efficient</span>' : '<span class="dim">—</span>'); })()) + '</td>' : '') +
        '<td>' + (others.length ? others.map(function (s) { return '<button type="button" class="tiny jump" data-set="' + s.id + '" data-q="' + esc(p.qid) + '">' + esc(s.title.split(' · ')[0]) + (s.auto ? '' : ' · ' + esc(s.title.split(' · ')[1] || '')) + '</button>'; }).join(' ') : '<span class="dim">—</span>') + '</td>' +
        '<td class="src"><a href="' + esc(p.src) + '">' + esc(p.src_label) + '</a></td></tr>';
    }).join('');
    t.innerHTML = head + '<tbody>' + (body || '<tr><td colspan="' + (ks.length + 5) + '" class="dim">No quant matches the current list and limits.</td></tr>') + '</tbody>';
    t.classList.add('mobile-cards');
    var columnNames = Array.prototype.map.call(t.querySelectorAll('thead th'), function (c) { return c.textContent; });
    Array.prototype.forEach.call(t.querySelectorAll('tbody tr[data-qid]'), function (r) {
      Array.prototype.forEach.call(r.cells, function (c, i) { c.dataset.label = columnNames[i]; });
    });
    $('extcap').textContent = pts.length + ' quant' + (pts.length === 1 ? '' : 's') + ' · sorted by ' + (sort.key === '_x' ? S.x.unit : sort.key === '_name' ? 'name' : MET[sort.key].label) + (comparable(S) ? ' · click a header to sort' : ' · sort by quant or size; metric ranking requires verified compatibility');
    Array.prototype.forEach.call(t.querySelectorAll('th[data-key]'), function (c) {
      function go() { var key = c.dataset.key; state.sort = { key: key, dir: sort.key === key ? -sort.dir : (key === '_name' || key === '_x' || (MET[key] && MET[key].better === 'lower') ? 1 : -1) }; table(S); }
      c.addEventListener('click', go); c.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
    Array.prototype.forEach.call(t.querySelectorAll('.jump'), function (b) { b.addEventListener('click', function () { goSet(b.dataset.set, b.dataset.q); $('explorer').scrollIntoView({ block: 'start' }); }); });
    Array.prototype.forEach.call(t.tBodies[0].rows, function (r) {
      var q = r.getAttribute('data-qid'); if (!q) return;
      r.addEventListener('pointerenter', function () { hl(q, true); }); r.addEventListener('pointerleave', function () { hl(q, false); });
    });
    $('excopy').onclick = function () {
      var rows = [['quant', 'repo', 'format', 'measured_by', 'quant_publisher', 'comparison_metric', 'comparison_classification', 'ranking_permitted', S.x.unit].concat(ks.map(function (m) { return MET[m].label; })).concat(['source'])];
      pts.forEach(function (p) { rows.push([p.variant, p.repo, p.fmt || '', measurementOwner(p), quantizationOwner(p), state.metric, comparison(S).classification, comparable(S), fmt(p.x)].concat(ks.map(function (m) { return p.m[m] ? p.m[m].raw : ''; })).concat([p.src])); });
      var csv = rows.map(function (r) { return r.map(function (c) { c = String(c); return /[",\n]/.test(c) ? '"' + c.replace(/"/g, '""') + '"' : c; }).join(','); }).join('\n');
      var btn = $('excopy'), done = function (ok) { btn.textContent = ok ? 'Copied ' + pts.length + ' rows' : 'Copy failed: select the table instead'; setTimeout(function () { btn.textContent = 'Copy as CSV'; }, 2200); };
      try { navigator.clipboard.writeText(csv).then(function () { done(true); }, function () { done(false); }); } catch (e) { done(false); }
    };
  }

  var rt; if (window.ResizeObserver) new ResizeObserver(function () { clearTimeout(rt); rt = setTimeout(drawChart, 120); }).observe(host);
  render(); if (exBody.classList.contains('wide')) setWide(true);
})();
