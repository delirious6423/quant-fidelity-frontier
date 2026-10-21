(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var data = JSON.parse($('shared-data').textContent), selector = $('shared-benchmark');
  var log = $('shared-log'), zero = $('shared-zero'), refs = $('shared-refs'), host = $('shared-plot'), pick = $('shared-pick');
  var limits = {size:null,score:null}, bounds = {}, hidden = {}, selected = null, ns = 'http://www.w3.org/2000/svg';
  function node(tag, attrs, text) { var n = document.createElementNS(ns, tag); Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); }); if (text != null) n.textContent = text; return n; }
  function option(value, text) { var n = document.createElement('option'); n.value = value; n.textContent = text; return n; }
  function view() { return data.views[+selector.value]; }
  function syncLimits() {
    var xr = $('shared-size'), yr = $('shared-score');
    xr.value = limits.size == null ? 1000 : Math.round(1000*(limits.size-bounds.x0)/(bounds.x1-bounds.x0 || 1));
    yr.value = limits.score == null ? 0 : Math.round(1000*(limits.score-bounds.y0)/(bounds.y1-bounds.y0 || 1));
    var sizeText = limits.size == null ? 'any' : '≤ '+fmt(limits.size)+' GB';
    var scoreText = limits.score == null ? 'any' : '≥ '+fmt(limits.score)+(view().unit==='percent'?'%':'');
    $('shared-size-output').textContent = sizeText; $('shared-score-output').textContent = scoreText;
    xr.setAttribute('aria-valuetext',sizeText);yr.setAttribute('aria-valuetext',scoreText);
    zero.disabled = limits.score != null;
    $('shared-reset').hidden = limits.size == null && limits.score == null;
    $('shared-options').querySelector('summary').textContent = 'Filters & chart options'+(limits.size != null || limits.score != null ? ' · limits active' : '');
  }
  function resetLimits() {
    var xs = view().points.map(function(p){return p.size;}), ys = view().points.map(function(p){return p.score;});
    bounds = {x0:Math.min.apply(null,xs),x1:Math.max.apply(null,xs),y0:Math.min.apply(null,ys),y1:Math.max.apply(null,ys)};
    limits = {size:null,score:null};syncLimits();
  }
  function points() {
    return view().points.filter(function (p) {
      return !hidden[p.slug] && (limits.size == null || p.size <= limits.size+1e-9) &&
        (limits.score == null || p.score >= limits.score-1e-9) && (refs.checked || !p.reference);
    });
  }
  function fmt(n) { return +n.toPrecision(4); }
  function url() {
    var u = new URL(location.href), q = new URLSearchParams(); q.set('benchmark', view().name);
    if (data.views.some(function (v) { return v !== view() && v.name === view().name; })) q.set('unit', view().unit);
    var enabled = view().models.filter(function (slug) { return !hidden[slug]; });
    if (enabled.length !== view().models.length) q.set('models', enabled.length ? enabled.join(',') : 'none');
    if (limits.size != null) q.set('max',limits.size);
    if (limits.score != null) q.set('score_min',limits.score);
    if (log.checked) q.set('scale', 'log');
    if (!zero.checked) q.set('zero', '0');
    if (!refs.checked) q.set('references', '0');
    u.search = q.toString(); u.hash = 'comparison'; return u.href;
  }
  function save() { try { history.replaceState(null, '', url()); } catch (e) {} }
  function models() {
    var box = $('shared-models'); box.replaceChildren();
    view().models.forEach(function (slug) {
      var m = data.models.filter(function (m) { return m.slug === slug; })[0];
      var label = document.createElement('label'); label.className = 'model-toggle'; label.style.setProperty('--model-color', data.colors[slug]);
      var input = document.createElement('input'); input.type = 'checkbox'; input.checked = !hidden[slug]; input.value = slug;
      input.setAttribute('aria-label', 'Show ' + m.name);
      input.addEventListener('change', function () { hidden[slug] = !input.checked; render(); save(); });
      var text = document.createElement('span'); text.className = 'model-label'; text.textContent = m.name + ' · ' + view().points.filter(function (p) { return p.slug === slug; }).length;
      label.appendChild(input); label.appendChild(text); box.appendChild(label);
    });
  }
  function contexts() {
    var box = $('shared-contexts'); box.replaceChildren(); var seen = {};
    view().points.forEach(function (p) {
      var key = p.slug + ':' + p.source_label; if (seen[key]) return; seen[key] = true;
      var article = document.createElement('article'), title = document.createElement('h3'), context = document.createElement('p');
      title.textContent = p.model_name + ' — ' + p.source_label; context.textContent = p.context;
      article.appendChild(title); article.appendChild(context);
      var list = document.createElement('ul'); p.notes.forEach(function (s) { var li = document.createElement('li'); li.textContent = s; list.appendChild(li); }); article.appendChild(list); box.appendChild(article);
    });
    var coverage = document.createElement('p');
    var missing = data.models.filter(function (m) { return view().models.indexOf(m.slug) < 0; }).map(function (m) { return m.name; });
    coverage.textContent = 'Score unit: ' + (view().unit === 'as published' ? 'not stated by the sources' : view().unit) + '. ' + (missing.length ? 'No reviewed result with a matched file size for this benchmark: ' + missing.join(', ') + '.' : 'All indexed models have matching results.');
    box.appendChild(coverage);
  }
  function detail() {
    var p = points().filter(function (p) { return p.id === selected; })[0], box = $('shared-detail'); box.replaceChildren();
    if (!p) return;
    var title = document.createElement('h2'); title.textContent = p.model_name; box.appendChild(title);
    var score = document.createElement('p'); score.className = 'point-score'; score.textContent = p.raw + ' score · ' + p.size + ' GB'; box.appendChild(score);
    [p.variant, 'Benchmark, as published: ' + p.metric, 'Evaluation: ' + p.source_label].forEach(function (s) { var n = document.createElement('p'); n.textContent = s; box.appendChild(n); });
    var source = document.createElement('a'); source.href = p.src; source.textContent = 'Score source · line ' + p.line; source.target = '_blank'; source.rel = 'noopener'; box.appendChild(source);
    var size = document.createElement('a'); size.href = p.size_src; size.textContent = 'Size source · line ' + p.size_line; size.target = '_blank'; size.rel = 'noopener'; box.appendChild(document.createTextNode(' · ')); box.appendChild(size);
    var model = document.createElement('p'), link = document.createElement('a'); link.href = '../' + p.slug + '/#benchmarks'; link.textContent = 'Explore this model →'; model.appendChild(link); box.appendChild(model);
    if (p.flags.length) { var ul = document.createElement('ul'); p.flags.forEach(function (f) { var li = document.createElement('li'); li.textContent = f; ul.appendChild(li); }); box.appendChild(ul); }
  }
  function draw() {
    var active = document.activeElement, focusId = host.contains(active) ? active.getAttribute('data-point') : null;
    var padding = getComputedStyle(host), ps = points(), w = Math.max(240, host.clientWidth-parseFloat(padding.paddingLeft)-parseFloat(padding.paddingRight)), narrow = w < 600, h = narrow ? Math.round(w*.95) : 410;
    var L = narrow ? 48 : 62, R = w-18, T = 35, B = h-52;
    var svg = node('svg', {viewBox:'0 0 '+w+' '+h, role:'img', 'aria-label':view().name+' score versus file size in GB across models'});
    svg.appendChild(node('title', {}, 'Publisher-reported scores. Model colors identify separate models; evaluation settings may differ.'));
    host.replaceChildren(svg);
    var ys = (ps.length ? ps : view().points).map(function (p) { return p.score; });
    var xs = (ps.length ? ps : view().points).map(function (p) { return p.size; });
    var lo = Math.min.apply(null,ys), hi = Math.max.apply(null,ys), pad = Math.max((hi-lo)*.12, view().unit==='fraction'?.001:1);
    lo = zero.checked ? Math.min(0,lo-pad) : lo-pad; hi += pad;
    if (limits.score != null) lo = limits.score;
    var xmin = log.checked ? Math.min.apply(null,xs)/1.08 : 0, xmax = Math.max.apply(null,xs)*1.08;
    if (limits.size != null) xmax = limits.size;
    if (log.checked && xmin <= 0) xmin = Math.min.apply(null,view().points.map(function(p){return p.size;}))/1.08;
    // Keep exact-match and empty intervals drawable, including zero on a log axis.
    if (hi <= lo) hi = lo+pad;
    if (xmax <= xmin) {
      if (log.checked) { xmin = xmax > 0 ? xmax/1.08 : xmin; xmax = xmax > 0 ? xmax : xmin*1.08; }
      else { xmin = Math.max(0,xmax-.01); xmax = Math.max(xmax,xmin+.01); }
    }
    var X = log.checked ? function (v) { return L+(Math.log(v)-Math.log(xmin))/(Math.log(xmax)-Math.log(xmin))*(R-L); } : function (v) { return L+(v-xmin)/(xmax-xmin)*(R-L); };
    svg.setAttribute('data-x-min',xmin);svg.setAttribute('data-x-max',xmax);svg.setAttribute('data-y-min',lo);svg.setAttribute('data-y-max',hi);
    var Y = function (v) { return B-(v-lo)/(hi-lo)*(B-T); };
    for(var i=0;i<=4;i++) {
      var y = lo+(hi-lo)*i/4, yy=Y(y), x=log.checked?Math.exp(Math.log(xmin)+(Math.log(xmax)-Math.log(xmin))*i/4):xmin+(xmax-xmin)*i/4;
      svg.appendChild(node('path',{d:'M'+L+' '+yy+'H'+R,'class':'gridline'}));
      svg.appendChild(node('text',{x:L-7,y:yy+4,'text-anchor':'end'},fmt(y)));
      svg.appendChild(node('text',{x:X(x),y:B+21,'text-anchor':i===0?'start':i===4?'end':'middle'},fmt(x)));
    }
    svg.appendChild(node('text',{x:(L+R)/2,y:h-9,'text-anchor':'middle'},'File size (GB)'+(log.checked?' · log scale':'')));
    svg.appendChild(node('text',{x:L,y:18},'Score'+(view().unit==='fraction'?' (0–1)':view().unit==='percent'?' (%)':'')+' ↑'));
    if (!ps.length) svg.appendChild(node('text',{x:(L+R)/2,y:(T+B)/2,'text-anchor':'middle','class':'empty-label'},'No results match these filters.'));
    ps.forEach(function (p) {
      var cx=X(p.size),cy=Y(p.score),r=p.id===selected?8:6;
      var a=node('a',{href:'#shared-inspect','data-point':p.id,'aria-label':p.model_name+', '+p.variant+', '+p.metric+', '+p.raw+', '+p.size+' GB, '+p.source_label});
      a.appendChild(node('circle',{cx:cx,cy:cy,r:22,fill:'transparent'}));
      var shape=p.reference?node('path',{d:'M'+cx+' '+(cy-r)+'l'+r+' '+r+'l-'+r+' '+r+'l-'+r+' -'+r+'Z'}):node('circle',{cx:cx,cy:cy,r:r});
      shape.setAttribute('class','shared-dot');shape.style.fill=data.colors[p.slug];shape.setAttribute('data-model',p.slug);a.appendChild(shape);
      a.appendChild(node('title',{},p.model_name+' · '+p.variant+' · '+p.metric+': '+p.raw+' at '+p.size+' GB · '+p.source_label));
      a.addEventListener('click',function(e){e.preventDefault();selected=p.id;pick.value=p.id;detail();draw();});svg.appendChild(a);
      if(p.id===selected)svg.appendChild(node('text',{x:Math.max(L+18,Math.min(R-18,cx)),y:Math.max(T+15,cy-17),'text-anchor':'middle','class':'point-label'},p.raw));
    });
    if(focusId){var replacement=Array.prototype.filter.call(svg.querySelectorAll('a'),function(a){return a.getAttribute('data-point')===focusId;})[0];if(replacement)replacement.focus({preventScroll:true});}
  }
  function render() {
    var ps=points();if(!ps.some(function(p){return p.id===selected;}))selected=ps.length?ps[0].id:null;
    pick.replaceChildren();ps.forEach(function(p){pick.appendChild(option(p.id,p.model_name+' · '+p.variant+(view().metrics.length>1?' · '+p.metric:'')+' · '+p.size+' GB · '+p.source_label));});if(selected)pick.value=selected;
    $('shared-inspect').hidden=!ps.length;
    var modelCount=new Set(ps.map(function(p){return p.slug;})).size;
    $('shared-status').textContent=modelCount+(modelCount===1?' model':' models')+' · '+ps.length+' of '+view().points.length+' results'+(limits.size != null || limits.score != null?' within limits':'');
    detail();draw();
  }
  if(!data.views.length)return;
  data.views.forEach(function(v,i){selector.appendChild(option(i,v.name+(data.views.some(function(o){return o!==v&&o.name===v.name;})?' · '+v.unit:'')));});
  zero.checked=true;
  var q=new URLSearchParams(location.search), initial=data.views.findIndex(function(v){return v.name===q.get('benchmark')&&(!q.has('unit')||v.unit===q.get('unit'));});
  // a link made before naming variants shared a chart names one published benchmark name
  if(initial<0)initial=data.views.findIndex(function(v){return v.metrics.indexOf(q.get('benchmark'))>=0&&(!q.has('unit')||v.unit===q.get('unit'));});
  selector.value=initial>=0?initial:0;
  if(q.has('models')){var enabled=q.get('models').split(',');view().models.forEach(function(slug){hidden[slug]=enabled.indexOf(slug)<0;});}
  resetLimits();
  [['size','max'],['score','score_min']].forEach(function(keys){
    var value=q.get(keys[1]);
    if(value!=null&&value.trim()!==''&&Number.isFinite(+value)&&(keys[0]!=='size'||+value>=0))limits[keys[0]]=+value;
  });
  syncLimits();
  log.checked=q.get('scale')==='log';zero.checked=q.get('zero')!=='0';refs.checked=q.get('references')!=='0';
  $('shared-ranges').hidden=false;$('shared-controls').hidden=false;$('shared-options').hidden=false;$('shared-evidence').open=false;
  selector.addEventListener('change',function(){hidden={};selected=null;resetLimits();models();contexts();render();save();});
  $('shared-size').addEventListener('input',function(){
    var t=+this.value/1000;limits.size=t>=1?null:bounds.x0+t*(bounds.x1-bounds.x0);syncLimits();render();save();
  });
  $('shared-score').addEventListener('input',function(){
    var t=+this.value/1000;limits.score=t<=0?null:bounds.y0+t*(bounds.y1-bounds.y0);syncLimits();render();save();
  });
  $('shared-reset').addEventListener('click',function(){resetLimits();render();save();});
  [log,zero,refs].forEach(function(n){n.addEventListener('change',function(){render();save();});});
  pick.addEventListener('change',function(){selected=pick.value;detail();draw();});
  $('shared-copy').addEventListener('click',function(){var b=this;function done(){b.textContent='Link copied';setTimeout(function(){b.textContent='Copy view link';},2000);}try{navigator.clipboard.writeText(url()).then(done,function(){b.textContent='Copy the page URL';});}catch(e){b.textContent='Copy the page URL';}});
  document.addEventListener('keydown',function(e){if(e.key==='Escape'){var picker=document.querySelector('.page-picker');if(picker.open){picker.open=false;picker.querySelector('summary').focus();}}});
  models();contexts();render();new ResizeObserver(draw).observe(host);
})();
