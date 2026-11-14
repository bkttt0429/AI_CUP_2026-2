  // ---- 第 3 章 ----
  // 第 5 頁右欄：離線分數與 Public 一路上升（NUM.climb）；第 5 欄是口徑，口徑改變處只畫灰色虛線、不連實線
  //   11-11：點數從 9 加到 11（對比損失、對手錨點），橫向間距與縱軸上限跟著改
  ANIM.climb = () => {
    const C = NUM.climb, n = C.length, X = i => 74 + i * 54, lo = 0.40, hi = 0.82, Y = v => 222 - (v - lo) / (hi - lo) * 186;
    const t = i => 0.5 + i * 0.7;
    let s = '';
    [0.4, 0.5, 0.6, 0.7, 0.8].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="60" y1="${y}" x2="${X(n - 1) + 20}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(54, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(1), 'text-anchor="end"'); });
    s += `<circle cx="66" cy="14" r="5" fill="#0b6e69"/>` + tx(76, 18, 'an-s', '離線分數') + `<circle cx="146" cy="14" r="5" fill="#fff" stroke="#a4451c" stroke-width="2"/>` + tx(156, 18, 'an-s', 'Public');
    C.slice(1).forEach((c, j) => {
      const x1 = X(j), y1 = Y(C[j][1]).toFixed(1), x2 = X(j + 1), y2 = Y(c[1]).toFixed(1);
      if (C[j][4] === c[4]) s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#0b6e69" stroke-width="2.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t(j), t(j + 1) - t(j))}</line>`;
      else {
        // 「換口徑」三個字放在虛線下方的空白處：f0 → mlp、mlp → ens、ens → anc 各一個位置
        const [lx, ly, la] = ({mlp: [x1 + 40, 166, 'start'], ens: [(x1 + x2) / 2, 116, 'middle'], anc: [(x1 + x2) / 2, 106, 'middle']})[c[4]] || [(x1 + x2) / 2, 116, 'middle'];
        s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#8a93a3" stroke-width="1.6" stroke-dasharray="3 4" opacity="0">${aShow(t(j + 1))}</line>`
          + txA(lx, ly, 'an-s', '換口徑', t(j + 1), undefined, `text-anchor="${la}" style="fill:#8a93a3"`);
      }
    });
    C.forEach(([name, cv, pub], i) => {
      const t0 = t(i), x = X(i);
      const fmt = v => v.toFixed(Math.round(v * 1e4) % 10 === 0 ? 3 : 4);
      s += `<circle cx="${x}" cy="${Y(cv).toFixed(1)}" r="5" fill="#0b6e69" opacity="0">${aShow(t0)}</circle>`
        + txA(x, (Y(cv) + (pub != null ? 18 : -10)).toFixed(1), 'an-n', fmt(cv), t0 + 0.1, undefined, 'text-anchor="middle" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:#0b6e69"')
        + txA(x, 240, 'an-k', name, t0, undefined, 'text-anchor="middle" style="font-size:11px"');
      if (pub != null) s += `<circle cx="${x}" cy="${Y(pub).toFixed(1)}" r="5.5" fill="#fff" stroke="#a4451c" stroke-width="2" opacity="0">${aShow(t0 + 0.25)}</circle>`
        + txA(i ? x : x - 9, (i ? Y(pub) - 9 : Y(pub) + 4).toFixed(1), 'an-n', fmt(pub), t0 + 0.35, undefined, `text-anchor="${i ? 'middle' : 'end'}" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:#a4451c"`);
    });
    s += txA(220, 18, 'an-h', '大跳升只有 Human SL（+0.17）；之後每步 +0.005–0.035', t(n - 1) + 0.4);
    return {steps: ['① 每手似然（Human SL）', '② 等級頭（L1b）與對手', '③ 讓子、局面向量＋集成', '④ 更多局、對手 Human SL', '⑤ 對比損失、對手錨點'],
      svg: `<svg viewBox="0 0 640 252" role="img" aria-label="離線分數與 Public 分數一路上升的動畫">${s}</svg>`,
      note: '最左點是官方 baseline 在 fold 0 的分數；第 2–6 點是單一 MLP 的 CV（folds 1–4、類別平衡）；第 7–10 點是 6 模型集成的 CV；最後一點是 0.7458 加上對手錨點在同一個底上的配對增益 +0.0193（驗證端的錨點密度；測試端較低）。口徑改變處只用灰色虛線相接。Public 是使用者回報的上傳分數（C＋T 檔是 12 模型；最後一點是 ct44，另含兩個由 Public 選出的推移，見第 7 章）。'};
  };
  ANIM.climb.dur = 10;

  // 第 6 頁右欄：Public 上傳分數（NUM.uploads：模型檔與被保留的檔，共 13 個）；紅點是 ct6 與 ct40，NUM.final 是最終檔
  ANIM.uploads = () => {
    const U = NUM.uploads, n = U.length, X = i => 72 + i * 44, lo = 0.70, hi = 0.81, Y = v => 214 - (v - lo) / (hi - lo) * 160;
    const t = i => 0.5 + i * 0.5, fin = NUM.final, bad = {ct6: 1, ct40: 1};
    let s = '';
    [0.70, 0.72, 0.74, 0.76, 0.78, 0.80].forEach(g => { const y = Y(g).toFixed(1);
      s += `<line x1="50" y1="${y}" x2="${X(n - 1) + 26}" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(44, (Y(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    U.slice(1).forEach((u, j) => {
      s += `<line x1="${X(j)}" y1="${Y(U[j][2]).toFixed(1)}" x2="${X(j + 1)}" y2="${Y(u[2]).toFixed(1)}" stroke="#8a93a3" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t(j), t(j + 1) - t(j))}</line>`;
    });
    U.forEach(([name, day, v], i) => {
      const col = bad[name] ? '#d1343a' : name === fin ? '#0b6e69' : '#5d6370', r = name === fin ? 7 : 5, t0 = t(i);
      // 分數標籤的位置依前後兩點決定，避開連線：谷底放下方、峰頂放上方、一路上升的放左上
      const up = i === 0 || U[i - 1][2] < v, down = i < n - 1 && U[i + 1][2] > v;
      const [lx, ly, la] = !up || i === 0 ? (down ? [X(i), Y(v) + 20, 'middle'] : [X(i) + 7, Y(v) - 7, 'start'])
        : down ? [X(i) - 7, Y(v) - 7, 'end'] : [X(i), Y(v) - (name === fin ? 17 : 11), 'middle'];
      s += `<circle cx="${X(i)}" cy="${Y(v).toFixed(1)}" r="${r}" fill="${col}" opacity="0">${aShow(t0)}</circle>`
        + txA(lx, ly.toFixed(1), 'an-n', v.toFixed(4), t0 + 0.1, undefined, `text-anchor="${la}" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`)
        + txA(X(i), 234, 'an-k', name, t0, undefined, 'text-anchor="middle" style="font-size:12px"')
        + txA(X(i), 249, 'an-s', day, t0, undefined, 'text-anchor="middle" style="font-size:10.5px"');
    });
    const i6 = U.findIndex(u => u[0] === 'ct6'), i40 = U.findIndex(u => u[0] === 'ct40'), iF = U.findIndex(u => u[0] === fin);
    if (i6 >= 0) s += txA(X(i6) + 4, (Y(U[i6][2]) + 36).toFixed(1), 'an-h', 'CV 預期 +0.010 → −0.0095', t(i6) + 0.4);
    if (i40 >= 0) s += txA(X(i40), (Y(U[i40][2]) + 38).toFixed(1), 'an-h', '先錨點後推移 → 掉分', t(i40) + 0.4, undefined, 'text-anchor="middle"');
    if (iF >= 0) s += `<circle cx="${X(iF)}" cy="${Y(U[iF][2]).toFixed(1)}" r="13" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(t(n - 1) + 0.5)}</circle>`
      + txA(X(iF), (Y(U[iF][2]) - 32).toFixed(1), 'an-b', '最終檔', t(n - 1) + 0.5, undefined, 'text-anchor="middle"');
    return {steps: ['① 11-04 至 11-06：以 CV 決策，Public 看方向', '② 11-07 至 11-09：解碼推移（ct13、ct28）', '③ 11-10 至 11-11：對手錨點（ct42 → ct44）'],
      svg: `<svg viewBox="0 0 640 258" role="img" aria-label="Public 上傳分數的動畫">${s}</svg>`,
      note: '分數是使用者回報的 Public。只畫模型檔與被保留的檔；11-07 到 11-10 另有 ct10、ct12 與 14 個被丟棄的探測檔（見 3.2 與 7.1 的表）。紅點：ct6（12 個全資料模型）與 ct40（錨點加在推移之前）。ct4 與 ct7 同分（0.7687835），兩檔在 Public 有 47 題決策不同。'};
  };
