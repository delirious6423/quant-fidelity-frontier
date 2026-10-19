(function () {   // links that leave QFF open in a new tab; QFF's own pages open in the same tab. Delegated, so links added later count too.
  var BASE = (function () { var p = location.pathname; var i = p.indexOf('/quant-fidelity-frontier/'); return location.origin + (i >= 0 ? p.slice(0, i) + '/quant-fidelity-frontier/' : '/'); })();
  function isExternal(a) {
    if (a.protocol === 'file:') return /^https?:/i.test(a.getAttribute('href'));   // local preview: only absolute web links leave
    return a.href.indexOf(BASE) !== 0;   // outside the QFF site (Hugging Face, GitHub, Ko-fi, ...)
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href]'); if (!a) return;
    var h = a.getAttribute('href'); if (!h || h.charAt(0) === '#' || /^(javascript:|mailto:|tel:)/i.test(h)) return;
    if (!a.target && isExternal(a)) { a.target = '_blank'; a.rel = ((a.rel || '') + ' noopener').trim(); }
  }, true);
  var m = document.querySelector('.msw');   // model menu: close on outside click and Escape
  if (m) {
    document.addEventListener('click', function (e) { if (m.open && !m.contains(e.target)) m.open = false; });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && m.open) { m.open = false; m.querySelector('summary').focus(); } });
  }
})();
(function () {
  var t = document.getElementById('t'); if (!t) return;
  var rows = Array.prototype.slice.call(t.tBodies[0].rows);
  var q = document.getElementById('q'), ffmt = document.getElementById('ffmt'), fref = document.getElementById('fref'),
      fq = document.getElementById('fq'), fx = document.getElementById('fx'), fm = document.getElementById('fm'), count = document.getElementById('count');
  rows.forEach(function (r) { r._txt = r.textContent.toLowerCase(); });
  function apply() {
    var terms = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean), n = 0;
    rows.forEach(function (r) {
      var ok = terms.every(function (w) { return r._txt.indexOf(w) >= 0; }) && (!ffmt.value || r.dataset.fmt === ffmt.value) &&
        (!fref.value || r.dataset.ref === fref.value) && (!fq.checked || r.dataset.q === '1') &&
        (fx.checked || (r.dataset.kind !== 'cross' && r.dataset.kind !== 'quoted')) && (fm.checked || r.dataset.mod !== '1');
      r.hidden = !ok; if (ok) n++;
    });
    count.textContent = n + ' of ' + rows.length + ' rows';
  }
  [q, ffmt, fref, fq, fx, fm].forEach(function (el) { el.addEventListener(el.type === 'search' ? 'input' : 'change', apply); });
  document.getElementById('freset').addEventListener('click', function () { q.value = ''; ffmt.value = ''; fref.value = ''; fq.checked = false; fx.checked = true; fm.checked = true; apply(); });
  var ths = t.tHead.rows[0].cells;
  Array.prototype.forEach.call(ths, function (th, i) {
    if (!th.dataset.sort) return;
    th.tabIndex = 0; th.setAttribute('aria-sort', 'none');
    function sort() {
      var dir = th.getAttribute('aria-sort') === 'ascending' ? -1 : 1;
      Array.prototype.forEach.call(ths, function (h) { if (h.dataset.sort) h.setAttribute('aria-sort', 'none'); });
      th.setAttribute('aria-sort', dir === 1 ? 'ascending' : 'descending');
      rows.sort(function (a, b) {
        if (th.dataset.sort === 's') { var u = a.cells[i].textContent.toLowerCase(), w = b.cells[i].textContent.toLowerCase(); return (u < w ? -1 : u > w ? 1 : 0) * dir; }
        var x = parseFloat(a.cells[i].dataset.v), y = parseFloat(b.cells[i].dataset.v);
        var ax = isNaN(x), by = isNaN(y);
        if (ax && by) return 0; if (ax) return 1; if (by) return -1;
        return (x - y) * dir;
      });
      var tb = t.tBodies[0]; rows.forEach(function (r) { tb.appendChild(r); });
    }
    th.addEventListener('click', sort);
    th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); } });
  });
  apply();
})();

(function () {
  var q = document.getElementById('cq'), f = document.getElementById('cflag'), c = document.getElementById('ccount'), box = document.getElementById('cards'); if (!q || !box) return;
  var cs = Array.prototype.slice.call(box.children); cs.forEach(function (x) { x._t = x.textContent.toLowerCase(); });
  function apply() { var terms = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean), n = 0;
    cs.forEach(function (x) { var ok = terms.every(function (w) { return x._t.indexOf(w) >= 0; }) && (!f.checked || x.dataset.f === '1'); x.hidden = !ok; if (ok) n++; });
    c.textContent = n + ' of ' + cs.length + ' notes'; }
  q.addEventListener('input', apply); f.addEventListener('change', apply); apply();
})();

(function () {
  var q = document.getElementById('ncq'), st = document.getElementById('ncs'), c = document.getElementById('nccount'), t = document.getElementById('nctable'); if (!q || !t) return;
  var rs = Array.prototype.slice.call(t.tBodies[0].rows); rs.forEach(function (r) { r._t = r.textContent.toLowerCase(); });
  function apply() { var terms = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean), n = 0;
    rs.forEach(function (r) { var ok = terms.every(function (w) { return r._t.indexOf(w) >= 0; }) && (!st.value || r.cells[1].textContent === st.value); r.hidden = !ok; if (ok) n++; });
    c.textContent = n + ' of ' + rs.length + ' repos'; }
  q.addEventListener('input', apply); st.addEventListener('change', apply); apply();
})();

(function () {  // highlight the current section in the nav
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[href^="#"]')); if (!links.length || !window.IntersectionObserver) return;
  var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) links.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id); }); }); }, { rootMargin: '-45% 0px -50% 0px' });
  links.forEach(function (a) { var s = document.querySelector(a.getAttribute('href')); if (s) io.observe(s); });
})();
