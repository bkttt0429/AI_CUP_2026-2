
  // ---- 長條圖與 data-n 數字 ----
  Object.keys(CHARTS).forEach(id => { const el = document.getElementById(id); if (!el) return; try { el.innerHTML = CHARTS[id](); } catch (err) { el.textContent = '圖表載入失敗：' + err.message; } });
  const DN = NUM.text || {};
  document.querySelectorAll('[data-n]').forEach(e => { if (DN[e.dataset.n] !== undefined) e.textContent = DN[e.dataset.n]; });

  // 放進頁面＋進度條。平常只露出底部細線（.an-mini），滑到底部才展開控制列；拖曳時暫停、放開後照原本狀態繼續；
  // 每次重新看到動畫（切頁、捲回來）就從頭播；減少動態效果時停在完成畫面（ANIM.x.final，預設倒數 1.5 秒），仍可拖曳。
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ctlHTML = d => '<div class="an-zone"><div class="an-ctl"><button type="button" class="an-pp" aria-label="暫停">❚❚</button>'
    + `<input type="range" class="an-seek" min="0" max="${d - 0.05}" step="0.05" value="0" aria-label="拖曳動畫進度"><span class="an-time">0.0 秒</span></div></div><div class="an-mini"></div>`;
  const anIO = 'IntersectionObserver' in window
    ? new IntersectionObserver(es => es.forEach(e => { if (e.target._an) e.target._an(e.isIntersecting); }), {threshold: 0.3}) : null;
  document.querySelectorAll('.anim[data-anim]').forEach(el => {
    try {
      const f = ANIM[el.dataset.anim];
      AD = (f && f.dur) || 9; HOLD = AD - 0.7; RAW = !!(f && f.raw);   // 這段的長度、要不要壓縮前搖（產生動畫字串時會用到）
      const an = f && f(), dur = AD, fin = (f && f.final) || AD - 1.5;
      if (!an) { el.remove(); return; }
      el.innerHTML = (an.steps && an.steps.length ? '<div class="an-steps">' + an.steps.map(t => `<span>${t}</span>`).join('') + '</div>' : '')
        + (an.tech ? `<div class="an-tech"><b>技術</b><span>${an.tech}</span></div>` : '')
        + `<div class="an-stage">${an.svg}${ctlHTML(dur)}</div>` + (an.note ? `<div class="an-note">${an.note}</div>` : '');
      const sv = el.querySelector('.an-stage > svg'), ctl = el.querySelector('.an-ctl'), mini = el.querySelector('.an-mini');
      if (!sv || !sv.setCurrentTime) { el.querySelector('.an-zone').remove(); mini.remove(); return; }
      const seek = ctl.querySelector('.an-seek'), pp = ctl.querySelector('.an-pp'), tm = ctl.querySelector('.an-time');
      let playing = true, resume = false, raf = 0, away = true;
      const show = t => { seek.value = t.toFixed(2); tm.textContent = t.toFixed(1) + ' 秒'; mini.style.width = (t / dur * 100).toFixed(2) + '%'; };
      const play = p => { playing = p; if (p) sv.unpauseAnimations(); else sv.pauseAnimations();
        pp.textContent = p ? '❚❚' : '▶'; pp.setAttribute('aria-label', p ? '暫停' : '播放'); };
      const loop = () => { if (!el.classList.contains('scrub')) show(sv.getCurrentTime() % dur); raf = requestAnimationFrame(loop); };
      const endScrub = () => { if (!el.classList.contains('scrub')) return; el.classList.remove('scrub'); if (resume) play(true); resume = false; };
      pp.addEventListener('click', () => play(!playing));
      seek.addEventListener('pointerdown', () => { resume = playing; play(false); el.classList.add('scrub'); });
      seek.addEventListener('input', () => { sv.setCurrentTime(+seek.value); show(+seek.value); });
      ['change', 'pointerup', 'pointercancel'].forEach(ev => seek.addEventListener(ev, endScrub));
      sv.querySelectorAll('[data-seek]').forEach(g => {   // 動畫裡的分段按鈕（第 8 頁）：跳到那一段
        const go = () => { const t = +g.dataset.seek; sv.setCurrentTime(t); show(t); };
        g.addEventListener('click', go);
        g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      });
      el._an = vis => {   // 看得到才更新進度條；從看不到變成看得到時從頭播
        if (vis && !raf) { if (away && playing) sv.setCurrentTime(0); away = false; loop(); }
        else if (!vis) { cancelAnimationFrame(raf); raf = 0; away = true; }
      };
      if (reduceMotion) { sv.setCurrentTime(fin); play(false); }
      show(reduceMotion ? fin : 0);
      if (anIO) anIO.observe(el); else loop();
    } catch (err) { el.textContent = '動畫載入失敗：' + err.message; }
  });
  AD = 9; HOLD = 8.3; RAW = false;

  // ---- 章節選單與封面路線 ----
  const menu = document.getElementById('ch-menu'), route = document.getElementById('route');
  let mh = '<a href="#cover"><span>封面</span><span class="meta">第 1 頁</span></a><hr>';
  pages.forEach((p, i) => {
    // 分隔頁（data-divider，例如「1106 報告」）在選單裡當成分段標題
    if (p.dataset.divider) mh += `<hr><a href="#${p.id}"><span>${p.dataset.divider}</span><span class="meta">第 ${i + 1} 頁</span></a>`;
    if (p.dataset.chtitle) {
      const n = pages.filter(q => q.dataset.ch === p.dataset.ch).length;
      mh += `<a href="#${p.id}"><span>${p.dataset.ch} · ${p.dataset.chtitle}</span><span class="meta">第 ${i + 1}${n > 1 ? '–' + (i + n) : ''} 頁</span></a>`;
      if (route) route.insertAdjacentHTML('beforeend', `<li><b>第 ${p.dataset.ch} 章</b><a href="#${p.id}">${p.dataset.chtitle}</a><span class="c">${n} 頁</span></li>`);
    }
  });
  menu.innerHTML = mh;
  document.getElementById('menu-btn').addEventListener('click', () => menu.showModal());
  menu.addEventListener('click', e => {
    if (e.target === menu) { menu.close(); return; }
    const a = e.target.closest('a'); if (!a) return;
    e.preventDefault(); menu.close(); go(pages.findIndex(p => '#' + p.id === a.getAttribute('href')));
  });
  if (route) route.addEventListener('click', e => {
    const a = e.target.closest('a'); if (!a) return;
    e.preventDefault(); go(pages.findIndex(p => '#' + p.id === a.getAttribute('href')));
  });

  // ---- 模式、換頁 ----
  const pg = document.getElementById('pg'), bPrev = document.querySelector('.pg-prev'), bNext = document.querySelector('.pg-next');
  function updatePager() { pg.textContent = `${cur + 1}／${pages.length}`; bPrev.disabled = cur === 0; bNext.disabled = cur === pages.length - 1; }
  function go(i) {
    if (i < 0 || i >= pages.length) return;
    cur = i;
    if (mode === 'present') { pages.forEach((p, j) => p.classList.toggle('cur', j === cur)); window.scrollTo(0, 0); fitFigs(pages[cur]); }
    else pages[cur].scrollIntoView({behavior: 'smooth', block: 'start'});
    history.replaceState(null, '', location.pathname + location.search + '#' + pages[cur].id);
    updatePager();
  }
  function setMode(m) {
    mode = m; document.body.classList.remove('m-read', 'm-present', 'm-notes'); document.body.classList.add('m-' + m);   // 只換模式，保留其他 class（例如預覽的 preview）
    document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === m ? 'true' : 'false'));
    pages.forEach(p => p.classList.remove('cur'));
    pages.forEach(fitFigs);
    go(cur);
  }
  document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  bPrev.addEventListener('click', () => go(cur - 1));
  bNext.addEventListener('click', () => go(cur + 1));
  window.addEventListener('scroll', () => {
    if (mode === 'present') return;
    let k = 0; pages.forEach((p, j) => { if (p.getBoundingClientRect().top < window.innerHeight * 0.4) k = j; });
    if (k !== cur) { cur = k; updatePager(); }
  }, {passive: true});
  window.addEventListener('resize', () => { if (mode === 'present') fitFigs(pages[cur]); });

  // ---- 推導彈窗 ----
  const dlg = document.getElementById('dlg'), lessons = [...document.querySelectorAll('.lesson')];
  // data-sec ＝ 推導節 id「sec-N」的 N。11-11 起依 id 找、找不到才當成順序：第 7 章新增的 sec-22–24 排在 sec-14 前面，
  // 其他章（含玩家辨識的 sec-15–21）的 id 與順序本來就相同，行為不變。上一節／下一節照頁面上的順序走。
  const secIdx = n => { const k = lessons.findIndex(L => L.id === 'sec-' + n); return k >= 0 ? k : n; };
  let sec = 0;
  function openSec(i) {
    if (i < 0 || i >= lessons.length) return;
    sec = i; const L = lessons[i];
    document.getElementById('dlg-t').textContent = L.dataset.title;
    dlg.querySelector('.dlg-body').innerHTML = L.innerHTML;
    dlg.querySelector('.prev').disabled = i === 0; dlg.querySelector('.nextb').disabled = i === lessons.length - 1;
    if (!dlg.open) dlg.showModal();
    dlg.querySelector('.dlg-body').scrollTop = 0;
  }
  document.querySelectorAll('.deriv-open').forEach(b => b.addEventListener('click', () => openSec(secIdx(+b.dataset.sec))));
  dlg.querySelector('.x').addEventListener('click', () => dlg.close());
  dlg.querySelector('.prev').addEventListener('click', () => openSec(sec - 1));
  dlg.querySelector('.nextb').addEventListener('click', () => openSec(sec + 1));
  dlg.querySelector('.whole').addEventListener('click', () => { dlg.close(); setMode('notes'); setTimeout(() => lessons[sec].scrollIntoView({block: 'start'}), 50); });
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  // ---- 鍵盤 ----
  document.addEventListener('keydown', e => {
    if (e.target.closest('input,textarea') || dlg.open || menu.open) return;
    if (mode === 'present' && ['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(cur + 1); }
    else if (mode === 'present' && ['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(cur - 1); }
    else if (e.key === 'd' || e.key === 'D') { const s = pages[cur].dataset.sec; if (s !== undefined) openSec(secIdx(+s)); }
    else if (e.key === 'Escape' && mode === 'present') setMode('read');
  });

  // ---- 起始狀態：?mode=present、#頁 ----
  const q = new URLSearchParams(location.search).get('mode');
  const h = pages.findIndex(p => '#' + p.id === location.hash);
  if (h >= 0) cur = h;
  if (['present', 'notes', 'read'].includes(q)) setMode(q); else { updatePager(); if (h > 0) go(h); }
})();

</script>
