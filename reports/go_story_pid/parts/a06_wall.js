  // ---- 第 6 章 ----
  // 第 13 頁右欄：「普通池的 2D」與「強池的 1D」實力相同，對手也相同 → 棋譜分不出來
  ANIM.wall = () => {
    const Yl = v => 214 - v * 40;   // 實力刻度：0 → 214，每級 40
    const ladder = (x, shift, title, t0, names) => { let o = txA(x, 22, 'an-t', title, t0, undefined, 'text-anchor="middle"');
      names.forEach((n, i) => { const y = Yl(i + shift + 0.5);
        o += `<rect x="${x - 34}" y="${(y - 15).toFixed(1)}" width="68" height="30" rx="4" fill="#eef0f3" stroke="#c9ced6" opacity="0">${aShow(t0 + i * 0.08)}</rect>` + txA(x, (y + 5).toFixed(1), 'an-k', n, t0 + i * 0.08, undefined, 'text-anchor="middle"'); });
      return o; };
    let s = `<line x1="320" y1="40" x2="320" y2="214" stroke="#5d6370" stroke-width="2"/>` + tx(320, 232, 'an-s', '實際棋力', 'text-anchor="middle"')
      + ladder(120, 0, '普通池的掛牌', 0.4, ['1D', '2D', '3D', '4D']) + ladder(520, 1, '強池的掛牌', 0.9, ['1D', '2D', '3D']);   // 強池的 1D 與普通池的 2D 同高
    const yP = Yl(1.5);
    s += `<circle cx="142" cy="${yP}" r="9" fill="#0072b2" opacity="0">${aShow(1.8)}</circle>` + txA(80, yP + 5, 'an-n', '乙：掛 2D', 1.8, undefined, 'text-anchor="end" style="fill:#0072b2"')
      + `<circle cx="498" cy="${yP}" r="9" fill="#d1343a" opacity="0">${aShow(2.2)}</circle>` + txA(560, yP + 5, 'an-n', '甲：掛 1D', 2.2, undefined, 'style="fill:#d1343a"')
      + `<line x1="151" y1="${yP}" x2="310" y2="${yP}" stroke="#0072b2" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.8, 0.5)}</line>`
      + `<line x1="489" y1="${yP}" x2="330" y2="${yP}" stroke="#d1343a" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.8, 0.5)}</line>`
      + `<circle cx="320" cy="${yP}" r="7" fill="#1d1f24" opacity="0">${aShow(3.3)}</circle>` + txA(330, yP - 10, 'an-k', '同一個實力', 3.3);
    [-0.6, -0.25, 0.1, 0.45, 0.7].forEach((d, k) => {
      const y = (yP - d * 40).toFixed(1), t0 = 4.0 + k * 0.12;
      s += `<circle cx="${178 + k * 9}" cy="${y}" r="4" fill="#7fb8b3" opacity="0">${aShow(t0)}</circle><circle cx="${462 - k * 9}" cy="${y}" r="4" fill="#7fb8b3" opacity="0">${aShow(t0)}</circle>`;
    });
    s += txA(320, 252, 'an-k', '對手（綠點）也一樣強 → 兩人的棋譜分布相同', 4.2, 5.6, 'text-anchor="middle"')
      + txA(320, 252, 'an-h', '從棋譜分不出「1D」還是「2D」', 5.9, undefined, 'text-anchor="middle"')
      + txA(640 - 10, 40, 'an-n', '實測：本人與對手誤差相關 0.62', 6.4, undefined, 'text-anchor="end"');
    return {steps: ['① 條件命題：若兩個池的掛牌差一級', '② 甲、乙實力相同，對手也相同', '③ 棋譜相同、掛牌不同'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="共享偏差不可識別的動畫">${s}</svg>`,
      note: '示意：這是條件命題的圖解（若本人與對手的掛牌一起平移）；不需要兩個池，標籤是期間代表值、對手跟著當局掛牌配，也給出同樣的共享偏差（設計文件第 879、885 行）。0.62 是實測（anchor_kalman）。'};
  };

  // 第 14 頁右欄：比對分數（margin）的分布：A 池內、B 池外、測試題（形狀示意，標出的分位數實測）
  ANIM.ident = () => {
    const I = NUM.ident, X = m => 56 + m * 52, base = 210, MAXM = 10.5;
    const lpdf = (m, mu, sg) => m > 0 ? Math.exp(-Math.pow(Math.log(m) - mu, 2) / (2 * sg * sg)) / m : 0;
    const grid = Array.from({length: 85}, (_, i) => 0.02 + i * MAXM / 84);
    const norm = f => { const v = grid.map(f), mx = Math.max(...v); return v.map(x => x / mx); };
    const B = norm(m => lpdf(m, Math.log(I.Bq.q50), 1.04)), Aw = norm(m => lpdf(m, Math.log(I.medWrong), 1.04)), Ac = norm(m => lpdf(m, Math.log(I.medCorrect), 0.55));   // B：對數常態，q50 0.465、q90 ≈ 1.76、q99 ≈ 5.2，接近實測
    const A = Aw.map((v, i) => 0.55 * v + 0.45 * Ac[i]), T = B.map((v, i) => v * 0.985 + 0.015 * Ac[i]);
    const path = (v, h) => grid.map((m, i) => `${i ? 'L' : 'M'}${X(m).toFixed(1)},${(base - v[i] * h).toFixed(1)}`).join(' ');
    let s = `<line x1="50" y1="${base}" x2="${X(MAXM)}" y2="${base}" stroke="#c9ced6"/>`;
    [0, 2, 4, 6, 8, 10].forEach(m => { s += tx(X(m), base + 16, 'an-s', String(m), 'text-anchor="middle"'); });
    s += tx(X(5.2), base + 34, 'an-s', '比對分數（第一名減第二名，margin）', 'text-anchor="middle"')
      + `<path d="${path(B, 150)} L${X(MAXM)},${base} L${X(0.02)},${base} Z" fill="#8a93a3" fill-opacity=".25" stroke="#8a93a3" opacity="0">${aShow(0.4)}</path>`
      + txA(X(1.4), 66, 'an-k', 'B：玩家不在池裡', 0.4)
      + `<path d="${path(A, 120)}" fill="none" stroke="#0b6e69" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.6, 1.0)}</path>`
      + txA(X(I.Bq.q99) - 6, base - 70, 'an-b', `A：在池裡，認對的中位 ${I.medCorrect}`, 2.4, undefined, 'text-anchor="end"')
      + `<line x1="${X(I.Bq.q99).toFixed(1)}" y1="34" x2="${X(I.Bq.q99).toFixed(1)}" y2="${base}" stroke="#1d1f24" stroke-width="1.4" stroke-dasharray="5 4" opacity="0">${aShow(3.4)}</line>`
      + txA(X(I.Bq.q99) + 6, 44, 'an-k', `B 的 q99 ＝ ${I.Bq.q99.toFixed(2)}`, 3.4)
      + `<path d="${path(T, 150)}" fill="none" stroke="#e69f00" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(4.4, 1.0)}</path>`
      + txA(X(1.4), 86, 'an-h', '測試題：和 B 重疊', 5.0)
      + txA(X(I.Bq.q99) + 6, 64, 'an-n', '測試題超過 q99：', 5.8)
      + txA(X(I.Bq.q99) + 6, 82, 'an-n', I.testAboveQ99.map(([n, v]) => `${(v * 100).toFixed(1)}%`).join('／'), 5.8)
      + txA(X(I.Bq.q99) + 6, 100, 'an-s', '（n 5／10／15／20；虛無 1%）', 5.8);
    return {steps: ['① B：不在池裡的玩家', '② A：在池裡的玩家', '③ 測試題的分布與 B 相同'],
      tech: '開局語言模型、31 維風格、局部棋形、方位、讓子五個成分；每題對 8,007 位訓練玩家評分後融合',
      svg: `<svg viewBox="0 0 640 256" role="img" aria-label="身分比對分數分布的動畫">${s}</svg>`,
      note: '示意：曲線形狀是示意（B 用與實測分位數相近的對數常態）；中位數、q99 與超過 q99 的比例是實測（ident_pool_local.json）。'};
  };
