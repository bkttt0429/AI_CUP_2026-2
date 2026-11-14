  // ---- 第 5 章 ----
  // 第 11 頁右欄：production gate 上的集成增益曲線（NUM.gate，實測）
  ANIM.gate = () => {
    const G = NUM.gate, K = G.k, X = i => 90 + i * 118, lo = 0.705, hi = 0.737, Y = v => 214 - (v - lo) / (hi - lo) * 180;
    let s = '';
    [0.71, 0.72, 0.73].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="72" y1="${y}" x2="${X(K.length - 1) + 20}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(66, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    K.forEach((k, i) => { s += tx(X(i), 236, 'an-k', `${k} 個`, 'text-anchor="middle"'); });
    s += tx(X(2), 254, 'an-s', '集成的成員數（類別刻度，不等距）', 'text-anchor="middle"');
    const line = (vals, col, t0) => vals.slice(1).map((v, j) => `<line x1="${X(j)}" y1="${Y(vals[j]).toFixed(1)}" x2="${X(j + 1)}" y2="${Y(v).toFixed(1)}" stroke="${col}" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0 + j * 0.45, 0.45)}</line>`).join('')
      + vals.map((v, j) => `<circle cx="${X(j)}" cy="${Y(v).toFixed(1)}" r="4.5" fill="${col}" opacity="0">${aShow(t0 + j * 0.45)}</circle>`
        + txA(X(j), (Y(v) + (col === '#d1343a' ? -10 : 18)).toFixed(1), 'an-n', v.toFixed(3), t0 + j * 0.45 + 0.1, undefined, `text-anchor="middle" style="font-size:11px;fill:${col}"`)).join('');
    s += line(G.ct3, '#5d6370', 0.5) + line(G.ct6, '#d1343a', 3.0)
      + `<rect x="80" y="10" width="14" height="4" rx="2" fill="#5d6370"/>` + tx(100, 16, 'an-s', 'ct3：單調上升')
      + `<rect x="210" y="10" width="14" height="4" rx="2" fill="#d1343a"/>` + tx(230, 16, 'an-s', 'ct6：4 個以後下降')
      + txA(600, 160, 'an-h', '3 個弱 T 成員：' + G.weakT.map(v => v.toFixed(3)).join('／'), 5.4, undefined, 'text-anchor="end"')
      + txA(600, 180, 'an-b', '修正 → ct7：每個成員都有驗證分', 6.6, undefined, 'text-anchor="end"');
    return {steps: ['① ct3 的集成增益：一路上升', '② ct6：單模型較強，集成到 4 個以後下降', '③ 改用 CV 驗證過的成員'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="production gate 的集成增益曲線動畫">${s}</svg>`,
      note: '794 位留出玩家（788 人有題）、4,000 題（prod_heldout.json、gate_diag.json）；gate 的信賴區間約 ±0.016，只看方向。'};
  };

  // 第 12 頁右欄：事先登記比較的配對差與 95% CI（NUM.forest）
  ANIM.forest = () => {
    const F = NUM.forest, RH = 19, top = 40, lo = -0.032, hi = 0.028, XL = 250, XR = 624, X = v => XL + (v - lo) / (hi - lo) * (XR - XL);
    const col = {pass: '#0b6e69', adopt: '#0b6e69', sub: '#b98900', fail: '#d1343a', ref: '#8a93a3'};
    const H = top + F.length * RH + 34, dt = 0.28, tz = 0.3 + F.length * dt + 0.2;   // tz：所有列出現之後才畫通過區（11-11 列數從 18 加到 21）
    let s = `<line x1="${X(0)}" y1="${top - 8}" x2="${X(0)}" y2="${H - 30}" stroke="#1d1f24" stroke-width="1.2"/>`
      + `<line x1="${X(0.005)}" y1="${top - 8}" x2="${X(0.005)}" y2="${H - 30}" stroke="#0b6e69" stroke-width="1.4" stroke-dasharray="5 4"/>`
      + tx(X(0), top - 14, 'an-s', '0', 'text-anchor="middle"') + tx(X(0.005) + 4, top - 14, 'an-s', '門檻 +0.005')
      + [-0.03, -0.02, -0.01, 0.01, 0.02].map(v => tx(X(v), H - 16, 'an-s', (v > 0 ? '+' : '') + v.toFixed(2), 'text-anchor="middle"')).join('');
    F.forEach(([name, d, a, b, st], i) => {
      const y = top + i * RH + 8, t0 = 0.3 + i * dt, c = col[st];
      s += txA(XL - 10, y + 4, 'an-s', name, t0, undefined, 'text-anchor="end" style="font-size:11.5px;fill:#1d1f24"')
        + `<line x1="${X(a).toFixed(1)}" y1="${y}" x2="${X(b).toFixed(1)}" y2="${y}" stroke="${c}" stroke-width="2.4" opacity="0">${aShow(t0)}</line>`
        + `<circle cx="${X(d).toFixed(1)}" cy="${y}" r="4.5" fill="${st === 'adopt' ? '#fff' : c}"${st === 'adopt' ? ` stroke="${c}" stroke-width="2"` : ''} opacity="0">${aShow(t0)}</circle>`;
    });
    s += `<rect x="${X(0.005)}" y="${top - 8}" width="${(XR - X(0.005)).toFixed(1)}" height="${H - 22 - top}" fill="#0b6e69" fill-opacity=".07" opacity="0">${aShow(tz)}</rect>`
      + txA(10, 18, 'an-s', '綠：採用　空心綠：次門檻、依規則採用　黃：次門檻　紅：未通過　灰：雜訊對照', 0.3, undefined, 'style="fill:#1d1f24"')
      + txA(XR, top - 26, 'an-b', '通過區', tz, undefined, 'text-anchor="end"');
    return {steps: ['① 每一列是一個比較（多數事先登記）', '② 點是配對差、線是 95% CI', '③ 過門檻的：對手相關的幾步、讓子 5 維、對比損失、對手錨點'],
      svg: `<svg viewBox="0 0 640 ${H}" role="img" aria-label="預先登記比較的森林圖動畫">${s}</svg>`,
      note: '數字是設計文件 §3.1 與 §4M 的實測（玩家層 bootstrap 的配對差；多數為 folds 1–4，M21b 只用 fold 1、對手 L1b 一列為 fold 0；層級不同：R 層 MLP、單臂或集成，見 5.2 表）。ct4＋ct7 疊加是使用者臨時提問，非事先登記。M33、M73、M82 三列是 11-06 之後通過的（那時的門檻已改成 +0.0025 加上換一組 seed 重做，三列也都超過 +0.005；M73 一列是錨點對 6 成員集成的增益）。'};
  };
  ANIM.forest.dur = 10;
