  // ===== 每張卡的動畫（.anim[data-anim]）：預設一輪 9 秒，全部用 SVG <animate>，所以能用同一條進度條拖曳（setCurrentTime）。
  //       圖上的框都來自 overlays.js（我們提交檔的框）；只是示意的在動畫下方註明。 =====
  // AD：這段一輪幾秒；HOLD 之後淡出、重來；RAW：時間照寫的走、不壓縮前搖。放進頁面前依 ANIM.x.dur／raw 設定（第 8 頁 20 秒）。
  let AD = 9, HOLD = 8.3, RAW = false;
  // 前搖縮短：各段動畫寫的是「設計時間」，kt() 換成實際時間——HOLD 以前先提前 0.3 秒、再壓成 0.75 倍，HOLD 以後不變
  const warp = t => (RAW || t >= HOLD ? t : Math.max(0, (t - 0.3) * 0.75));
  const kt = a => a.map(t => Math.min(1, Math.max(0, warp(t) / AD)).toFixed(4)).join(';');
  // 出現：t0 淡入、停到 t2 再淡出
  const aShow = (t0, t2 = HOLD) => `<animate attributeName="opacity" dur="${AD}s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="${kt([0, t0, t0 + 0.3, t2, t2 + 0.3, AD])}"/>`;
  // 畫線（元素要有 pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"）
  const aDraw = (t0, d = 0.5, t2 = HOLD) => `<animate attributeName="stroke-dashoffset" dur="${AD}s" repeatCount="indefinite" values="1;1;0;0" keyTimes="${kt([0, t0, t0 + d, AD])}"/>` + aShow(t0, t2);
  // 屬性從 v0 變到 v1，停到 t2 再回來
  const aTo = (attr, v0, v1, t0, d = 0.6, t2 = HOLD) => `<animate attributeName="${attr}" dur="${AD}s" repeatCount="indefinite" values="${v0};${v0};${v1};${v1};${v0}" keyTimes="${kt([0, t0, t0 + d, t2, AD])}"/>`;
  // 自訂時間表 [[時間, 值], …]：第一個時間是 0、最後一個是 AD
  const aSeq = (attr, pairs, discrete) => `<animate attributeName="${attr}" dur="${AD}s" repeatCount="indefinite"${discrete ? ' calcMode="discrete"' : ''} values="${pairs.map(p => p[1]).join(';')}" keyTimes="${kt(pairs.map(p => p[0]))}"/>`;
  const tx = (x, y, cls, t, extra = '') => `<text class="${cls}" x="${x}" y="${y}" ${extra}>${t}</text>`;
  const txA = (x, y, cls, t, t0, t2, extra = '') => `<text class="${cls}" x="${x}" y="${y}" opacity="0" ${extra}>${t}${aShow(t0, t2)}</text>`;
  const polyS = (p, attrs) => `<polygon points="${pstr(p)}" vector-effect="non-scaling-stroke" ${attrs}/>`;
  const polyA = (p, attrs, kids) => `<polygon points="${pstr(p)}" vector-effect="non-scaling-stroke" ${attrs}>${kids}</polygon>`;
  const nest = (R, crop, x, y, w, h, inner) => `<svg x="${x}" y="${y}" width="${w}" height="${h}" viewBox="${crop.map(v => (+v).toFixed(1)).join(' ')}" preserveAspectRatio="xMidYMid meet">`
    + `<rect x="${crop[0]}" y="${crop[1]}" width="${crop[2]}" height="${crop[3]}" fill="#0b0d10"/><image href="${R.o.src}" width="${R.o.w}" height="${R.o.h}"/>${inner}</svg>`;
  const inPoly = (pt, p) => { let c = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) { const [xi, yi] = p[i], [xj, yj] = p[j];
      if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c; }
    return c; };
