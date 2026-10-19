(function () {
  'use strict';
  var tabs = document.querySelector('.chart-tabs');
  if (!tabs) return;
  var links = Array.from(tabs.querySelectorAll('a'));
  var panels = links.map(function (a) { return document.querySelector(a.hash); });
  tabs.setAttribute('role', 'tablist');
  links.forEach(function (a, i) {
    a.setAttribute('role', 'tab');
    a.setAttribute('aria-controls', panels[i].id);
    panels[i].setAttribute('role', 'tabpanel');
    panels[i].setAttribute('aria-labelledby', a.id);
    a.addEventListener('click', function () { show(i); });
    a.addEventListener('keydown', function (e) {
      var next = e.key === 'Home' ? 0 : e.key === 'End' ? links.length - 1 :
        e.key === 'ArrowRight' ? (i + 1) % links.length :
        e.key === 'ArrowLeft' ? (i + links.length - 1) % links.length : -1;
      if (next < 0) return;
      e.preventDefault(); links[next].focus(); links[next].click();
    });
  });
  function show(index) {
    links.forEach(function (a, i) {
      a.setAttribute('aria-selected', String(i === index));
      a.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
  }
  function fromHash() {
    if (location.hash === '#benchmarks') show(1);
    else if (!location.hash || location.hash === '#chart' || location.hash === '#explorer') show(0);
  }
  show(location.hash === '#benchmarks' ? 1 : 0);
  window.addEventListener('hashchange', fromHash);
})();
