  // ---- 第 2 章 ----
  CHARTS['chart-byn'] = () => '<h4>production 留出題的分數，依局數（ct3 集成，實測；長條從 0.50 起）</h4>'
    + barRows(NUM.gateByN.map(([n, v]) => [`n ＝ ${n} 局`, v, v.toFixed(3), n === 20 ? 'var(--accent)' : '#8a93a3']), 0.5, 0.85);

  // 第 2 頁右欄：局數越多，十級的機率越集中；但中心是「看起來像幾段」，不一定是標籤
  ANIM.accum = () => {
    const L = NUM.levels, base = 206, H = 150, X = i => 64 + i * 54, W = 36, lab = 5;
    const P = [[0.04, 0.06, 0.08, 0.11, 0.13, 0.14, 0.14, 0.12, 0.10, 0.08],
      [0.00, 0.01, 0.02, 0.05, 0.12, 0.25, 0.30, 0.17, 0.06, 0.02],
      [0.00, 0.00, 0.00, 0.01, 0.04, 0.22, 0.52, 0.18, 0.03, 0.00]];
    const h = p => (p / 0.55 * H).toFixed(1), T = [0.4, 2.9, 5.3];
    let s = `<line x1="40" y1="${base}" x2="${X(9) + W + 12}" y2="${base}" stroke="#c9ced6"/>`
      + `<rect x="${X(lab) - 6}" y="${base - H - 14}" width="${W + 12}" height="${H + 14}" rx="4" fill="none" stroke="#a4451c" stroke-width="1.6" stroke-dasharray="5 4"/>`
      + tx(X(lab) + W / 2, base - H - 20, 'an-h', '標籤 2D', 'text-anchor="middle"');
    L.forEach((n, i) => {
      const hs = P.map(r => h(r[i])), ys = hs.map(v => (base - v).toFixed(1));
      const seq = vals => [[0, 0], [T[0], 0], [T[0] + 0.5, vals[0]], [T[1], vals[0]], [T[1] + 0.5, vals[1]], [T[2], vals[1]], [T[2] + 0.5, vals[2]], [HOLD, vals[2]], [HOLD + 0.3, 0], [AD, 0]];
      const ysq = seq(ys).map(([t, v], k) => [t, (k < 2 || k > 7) ? base : v]);
      s += `<rect x="${X(i)}" y="${base}" width="${W}" height="0" rx="3" fill="${i === 6 ? '#0b6e69' : '#7fb8b3'}">${aSeq('y', ysq)}${aSeq('height', seq(hs))}</rect>`
        + tx(X(i) + W / 2, base + 18, 'an-k', n, 'text-anchor="middle"');
    });
    s += txA(40, 24, 'an-t', '看了 1 局：10 級都有可能', T[0], T[1]) + txA(40, 24, 'an-t', '看了 5 局：集中到兩三級', T[1] + 0.3, T[2])
      + txA(40, 24, 'an-t', '看了 20 局：更集中，但中心在 3D', T[2] + 0.3)
      + txA(40, 242, 'an-h', '證據越多越集中；中心是「看起來像幾段」，不一定是標籤', 6.6);
    return {steps: ['① 一局：10 級都有可能', '② 5 局：集中到兩三級', '③ 20 局：更集中，但不一定在標籤上'],
      svg: `<svg viewBox="0 0 640 248" role="img" aria-label="局數越多機率越集中，但中心不一定是標籤的動畫">${s}</svg>`,
      note: '示意：機率是示意數字；左欄的 n ＝ 5／10／20 分數是實測。'};
  };

  // 第 3 頁右欄：每位玩家兩組不重疊的 40 局各估一次，兩次落在標籤的同一側
  ANIM.bias = () => {
    const L = NUM.levels, X = v => 76 + v * 55, Y = r => 56 + r * 32;
    const P = [[3, 0.8, -0.18, 0.14], [5, -0.7, 0.16, -0.12], [7, 0.35, -0.2, 0.22], [1, -0.55, 0.12, -0.2], [8, -1.0, -0.15, 0.17], [4, 0.6, 0.2, -0.1]];
    let s = `<line x1="64" y1="238" x2="${X(9) + 12}" y2="238" stroke="#c9ced6"/>`;
    L.forEach((n, i) => { s += `<line x1="${X(i)}" y1="234" x2="${X(i)}" y2="242" stroke="#c9ced6"/>` + tx(X(i), 256, 'an-k', n, 'text-anchor="middle"'); });
    P.forEach(([lab, b, e1, e2], r) => {
      const y = Y(r), t0 = 0.4 + r * 0.12, x0 = X(lab), xb = X(lab + b);
      s += tx(16, y + 4, 'an-s', `玩家 ${r + 1}`)
        + `<line x1="70" y1="${y}" x2="${X(9) + 8}" y2="${y}" stroke="#eef0f3"/>`
        + `<line x1="${x0}" y1="${y - 9}" x2="${x0}" y2="${y + 9}" stroke="#1d1f24" stroke-width="3" opacity="0">${aShow(t0)}</line>`
        + `<line x1="${x0}" y1="${y}" x2="${xb}" y2="${y}" stroke="#a4451c" stroke-width="2" stroke-dasharray="4 3" opacity="0">${aShow(4.3)}</line>`
        + `<circle cx="${(X(lab + b + e1)).toFixed(1)}" cy="${y - 4}" r="6" fill="#0072b2" opacity="0">${aShow(2.0)}${aTo('cx', X(lab + b + e1).toFixed(1), X(lab + b + e1 * 0.25).toFixed(1), 5.6, 0.7)}</circle>`
        + `<circle cx="${(X(lab + b + e2)).toFixed(1)}" cy="${y + 4}" r="6" fill="#e69f00" opacity="0">${aShow(2.7)}${aTo('cx', X(lab + b + e2).toFixed(1), X(lab + b + e2 * 0.25).toFixed(1), 5.6, 0.7)}</circle>`;
    });
    s += tx(16, 22, 'an-k', '黑線：標籤　藍：前 40 局的估計　橙：另 40 局的估計')
      + txA(16, 40, 'an-h', '兩次估計落在同一側：偏差跟著人走', 4.3, 5.5)
      + txA(16, 40, 'an-h', '局數加倍：兩點靠攏，但不回到標籤', 5.8)
      + txA(640 - 12, 22, 'an-n', '同一人兩組局的偏差相關 0.964', 6.4, undefined, 'text-anchor="end"');
    return {steps: ['① 每位玩家的標籤', '② 兩組不同的局（各 40 局）各估一次', '③ 兩次落在同一側，局數加倍也不回到標籤'],
      svg: `<svg viewBox="0 0 640 266" role="img" aria-label="玩家層偏差的動畫">${s}</svg>`,
      note: '示意：點的位置是示意；0.964 是實測（anchor_kalman：2,524 位有 80 局的玩家，兩組不重疊的 40 局各估一次的本人證據相關）。'};
  };

  // 第 4 頁右欄：RP 在三年間隨機漫步、多次跨過段位邊界；8 局是期間內的無序樣本，標籤只有一個值
  ANIM.rp = () => {
    const X0 = 64, X1 = 612, N = 64, band = [[44, 104, '3D'], [104, 164, '2D'], [164, 224, '1D']];
    let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    // 固定的轉折點（y）加 ±5 的小抖動：2D → 3D → 2D → 1D → 2D → 3D → 2D → 1D → 2D；2D 帶約占一半時間，與「標籤 2D」一致
    const K = [140, 134, 130, 70, 90, 140, 150, 205, 170, 120, 96, 70, 140, 156, 200, 190, 150];
    const pts = [];
    for (let i = 0; i < N; i++) { const u = i / (N - 1) * (K.length - 1), j = Math.min(K.length - 2, Math.floor(u)), f = u - j; pts.push([X0 + i * (X1 - X0) / (N - 1), K[j] + (K[j + 1] - K[j]) * f + (rnd() - 0.5) * 10]); }
    const pick = [4, 11, 19, 26, 34, 41, 49, 57], inBand = yy => band.find(([a, b]) => yy >= a && yy < b)[2];
    const cnt = {}; pick.forEach(i => { const b = inBand(pts[i][1]); cnt[b] = (cnt[b] || 0) + 1; });
    let s = '';
    band.forEach(([a, b, n], i) => {
      s += `<rect x="${X0}" y="${a}" width="${X1 - X0}" height="${b - a}" fill="${i % 2 ? '#f3f8f7' : '#ffffff'}"/>` + tx(X0 - 10, (a + b) / 2 + 4, 'an-k', n, 'text-anchor="end"');
      if (i) s += `<line x1="${X0}" y1="${a}" x2="${X1}" y2="${a}" stroke="#c9ced6" stroke-dasharray="4 4"/>`;
    });
    s += tx(X0, 244, 'an-s', '2021') + tx(X1, 244, 'an-s', '2023', 'text-anchor="end"') + tx((X0 + X1) / 2, 244, 'an-s', '時間（資料裡沒有）', 'text-anchor="middle"')
      + `<polyline points="${pstr(pts.map(p => [p[0].toFixed(1), p[1].toFixed(1)]))}" fill="none" stroke="#1d1f24" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(0.4, 2.6)}</polyline>`
      + txA(X0, 24, 'an-t', 'RP 每局約 ±100 點，三年間多次跨過段位邊界', 0.4, 3.6);
    pick.forEach((i, k) => {
      const [px, py] = pts[i], t0 = 3.8 + k * 0.2, c = {'1D': '#0072b2', '2D': '#0b6e69', '3D': '#d1343a'}[inBand(py)];
      s += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="6.5" fill="${c}" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(t0)}</circle>`;
    });
    s += txA(X0, 24, 'an-t', '資料只給這 8 局，而且不按時間排', 3.9, 5.5)
      + `<line x1="${X0}" y1="134" x2="${X1}" y2="134" stroke="#a4451c" stroke-width="4" opacity="0">${aShow(5.7)}</line>`
      + txA(626, 128, 'an-h', '標籤：2D（一個代表值）', 5.7, undefined, 'text-anchor="end"')
      + txA(X0, 24, 'an-t', `8 局中 1D 時期 ${cnt['1D'] || 0} 局、2D ${cnt['2D'] || 0} 局、3D ${cnt['3D'] || 0} 局`, 6.4);
    return {steps: ['① RP 在期間內隨機漫步', '② 資料是期間內的幾局，沒有時間', '③ 標籤只有一個值'],
      svg: `<svg viewBox="0 0 640 252" role="img" aria-label="RP 漫步與期間代表值標籤的動畫">${s}</svg>`,
      note: '示意：RP 軌跡與 8 局是示意；1,000 點區間來自幽玄の間手冊；每局約 ±100 RP 是依手冊（1 級約 116、8 段約 35）推估。'};
  };
