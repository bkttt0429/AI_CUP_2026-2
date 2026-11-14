  //（競賽一：玩家辨識，11-06 加；數字在 NUM.pid）
  // ---- 第 8 章 ----
  CHARTS['chart-pid-score'] = () => '<h4>正解排第幾名得幾分（官方規則）</h4>'
    + barRows([1, 2, 3, 4, 5].map(r => [`第 ${r} 名`, Math.exp(-(r - 1)), Math.exp(-(r - 1)).toFixed(3), r === 1 ? 'var(--accent)' : '#8a93a3']).concat([['不在前 5 名', 0, '0', '#8a93a3']]), 0, 1);

  // ---- 第 9 章 ----
  // 第 17 頁（橫跨整張卡）：五種成分 → 融合 → 一對一配對 → 前 5 名（靜態圖）
  CHARTS['chart-pid-arch'] = () => {
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, fill, stroke, inner, dash) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}"${dash ? ' stroke-dasharray="5 4"' : ''}/>${inner}`;
    const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2 - 7}" y2="${y2}" stroke="#5d6370" stroke-width="2"/><polygon points="${x2},${y2} ${x2 - 8},${y2 - 4.5} ${x2 - 8},${y2 + 4.5}" fill="#5d6370"/>`;
    const C = NUM.pid.parts, top = 18, step = 38, H = top + C.length * step + 22, mid = top + (C.length * step - 8) / 2;
    let s = box(12, mid - 82, 158, 64, '#fff', '#c9ced6', `<text x="91" y="${mid - 56}" text-anchor="middle" style="${Hd}">題目</text><text x="91" y="${mid - 36}" text-anchor="middle" style="${Sm}">同一人 5–20 局（黑）</text>`)
      + box(12, mid + 18, 158, 64, '#fff', '#c9ced6', `<text x="91" y="${mid + 44}" text-anchor="middle" style="${Hd}">候選人 × 400</text><text x="91" y="${mid + 64}" text-anchor="middle" style="${Sm}">每人 100 局（黑）</text>`);
    C.forEach(([name, sub, w], i) => {
      const y = top + i * step, last = w == null;
      s += box(214, y, 300, 30, last ? '#f2f3f6' : '#e3f1ef', last ? '#c9ced6' : '#0b6e69',
        `<text x="226" y="${y + 20}" style="${Hd}">${name}</text><text x="502" y="${y + 20}" text-anchor="end" style="${Sm}">${sub}</text>`, last)
        + (last ? '' : `<text x="522" y="${y + 20}" style="${Sm}">× ${w.toFixed(2)}</text>`);
    });
    s += arrow(172, mid - 50, 210, mid - 50) + arrow(172, mid + 50, 210, mid + 50)
      + `<text x="364" y="${H - 8}" text-anchor="middle" style="${Sm}">每個成分：一張 400 題 × 400 人的 log 分數表；「× 數字」是最終檔的融合權重</text>`
      + arrow(570, mid, 600, mid)
      + box(604, mid - 36, 120, 72, '#fff', '#0b6e69', `<text x="664" y="${mid - 8}" text-anchor="middle" style="${Hd}">逐列標準化</text><text x="664" y="${mid + 12}" text-anchor="middle" style="${Hd}">加權相加</text><text x="664" y="${mid + 29}" text-anchor="middle" style="${Sm}">5 個權重</text>`)
      + arrow(726, mid, 752, mid)
      + box(756, mid - 36, 112, 72, '#fff', '#0b6e69', `<text x="812" y="${mid - 8}" text-anchor="middle" style="${Hd}">一對一配對</text><text x="812" y="${mid + 12}" text-anchor="middle" style="${Sm}">匈牙利法</text><text x="812" y="${mid + 29}" text-anchor="middle" style="${Sm}">決定 Top-1</text>`)
      + arrow(870, mid, 892, mid)
      + box(896, mid - 36, 56, 72, '#0b6e69', '#0b6e69', `<text x="924" y="${mid - 4}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">每題</text><text x="924" y="${mid + 16}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">前 5 名</text>`);
    return `<svg viewBox="0 0 960 ${H}" role="img" aria-label="玩家辨識的架構：五種成分、融合、一對一配對、前 5 名">${s}</svg>`;
  };

  CHARTS['chart-pid-comp'] = () => '<h4>每個成分單獨用時的 Public Top-1（答案已知後離線計算；綠色 ＝ 在新最終候選裡）</h4>'
    + barRows(NUM.pid.comp.map(([n, v, used]) => [n, v, v.toFixed(3), used ? 'var(--accent)' : '#a7afbb']), 0, 1);

  CHARTS['chart-pid-assign'] = () => '<h4>Public：同一份融合分數，每題各選 vs 一對一配對（長條從 0.94 起）</h4>'
    + NUM.pid.pub.map(([n, a, b]) => `<h4>${n}</h4>` + barRows([['每題各選', a, a.toFixed(4), '#8a93a3'], ['一對一配對', b, b.toFixed(4), 'var(--accent)']], 0.94, 1.0)).join('');

  // ---- 第 10 章 ----
  CHARTS['chart-pid-hard'] = () => '<h4>把模擬弄難後的 fusion3（不配對；長條從 0.80 起）</h4>'
    + barRows(NUM.pid.hard.map(([n, v]) => [n, v, v.toFixed(4), n === '隨機 400 人' ? 'var(--accent)' : '#8a93a3']), 0.8, 1.0);

  // ---- 第 11 章 ----
  // 第 21 頁右欄：Public 最後 3 題的 3-環。箭頭 ＝ 候選人從「現在佔著他的題」移到「他真正的題」
  CHARTS['chart-pid-cycle'] = () => {
    const N = NUM.pid.cycle, P = [[320, 62], [526, 226], [114, 226]], W = 212, H = 78;
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:12px;fill:#5d6370', Ok = 'font-size:12px;font-weight:700;fill:#0b6e69';
    const edge = (a, b) => {   // 從 a 框邊到 b 框邊
      const [x1, y1] = P[a], [x2, y2] = P[b], dx = x2 - x1, dy = y2 - y1;
      const cut = Math.min(W / 2 / Math.abs(dx || 1e-9), H / 2 / Math.abs(dy || 1e-9)) + 0.06;
      return [x1 + dx * cut, y1 + dy * cut, x2 - dx * cut, y2 - dy * cut];
    };
    let s = '<defs><marker id="pid-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#a4451c"/></marker></defs>';
    N.forEach(([q, n, top, truth, rk], i) => {
      const [cx, cy] = P[i];
      s += `<rect x="${cx - W / 2}" y="${cy - H / 2}" width="${W}" height="${H}" rx="8" fill="#fff" stroke="#c9ced6"/>`
        + `<text x="${cx}" y="${cy - 16}" text-anchor="middle" style="${Hd}">${q}（n＝${n}）</text>`
        + `<text x="${cx}" y="${cy + 5}" text-anchor="middle" style="${Sm}">原 Top-1：${top}</text>`
        + `<text x="${cx}" y="${cy + 26}" text-anchor="middle" style="${Ok}">正解：${truth}（${rk}）</text>`;
    });
    // 候選人 c 在題 i 是 Top-1、卻是題 j 的正解 → 箭頭 i → j
    N.forEach(([, , top], i) => {
      const j = N.findIndex(r => r[3] === top); if (j < 0) return;
      const [x1, y1, x2, y2] = edge(i, j), mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      s += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#a4451c" stroke-width="2.2" marker-end="url(#pid-ah)"/>`
        + `<text x="${mx.toFixed(1)}" y="${(my + (i === 0 ? -8 : 16)).toFixed(1)}" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#a4451c;paint-order:stroke;stroke:#fff;stroke-width:4px">${top}</text>`;
    });
    s += `<text x="320" y="300" text-anchor="middle" style="${Sm}">箭頭：候選人從目前佔著他的題，移回他真正的題；三個 Top-1 輪轉一次就全對</text>`;
    return `<svg viewBox="0 0 640 312" role="img" aria-label="Public 最後三題錯答形成的 3-環">${s}</svg>`;
  };

  // ---- 第 12 章 ----
  CHARTS['chart-pid-err'] = () => '<h4>Public：不配對時的錯誤率，依 8 個成分中幾個同意 Top-1（實測）</h4>'
    + barRows(NUM.pid.agree.map(([k, nq, w]) => [`${k} 個同意（${nq} 題）`, w / nq, `${w}／${nq}`, w ? '#d1343a' : '#8a93a3']), 0, 1);
