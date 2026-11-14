  // ---- 第 1 章 ----
  CHARTS['chart-players'] = () => '<h4>訓練集各級玩家數（每級 10 萬局）</h4>' + barRows(NUM.players.map(([n, c]) => [n, c, c.toLocaleString(), 'var(--accent)']), 0, 1600);

  // 小棋盤：s 是邊長，k 是線數，stones = [[x, y, 'b'|'w'], …]（線的索引）
  const miniBoard = (x, y, s, k, stones, extra = '') => {
    const g = s / (k + 1); let o = `<g ${extra}><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="#dcb36a" stroke="#9c7a3c"/>`;
    for (let i = 1; i <= k; i++) o += `<line x1="${x + g}" y1="${y + i * g}" x2="${x + k * g}" y2="${y + i * g}" stroke="#8a6a2f" stroke-width=".7"/><line x1="${x + i * g}" y1="${y + g}" x2="${x + i * g}" y2="${y + k * g}" stroke="#8a6a2f" stroke-width=".7"/>`;
    stones.forEach(([i, j, c]) => { o += `<circle cx="${(x + i * g).toFixed(1)}" cy="${(y + j * g).toFixed(1)}" r="${(g * 0.45).toFixed(1)}" fill="${c === 'b' ? '#1d1f24' : '#fff'}" stroke="#1d1f24" stroke-width=".8"/>`; });
    return o + '</g>';
  };

  // 第 1 頁右欄：一題 = 同一位玩家的幾局 → 模型 → 10 級之一 → 依真實等級計分
  ANIM.task = () => {
    const L = NUM.levels, RX = 452, RW = 80, RH = 19, top = 34, pick = 5;
    const ry = i => top + (9 - i) * (RH + 4);
    const games = [[[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 3, 'b']], [[2, 6, 'b'], [6, 2, 'w'], [6, 6, 'b'], [3, 3, 'w']],
      [[6, 2, 'b'], [2, 2, 'w'], [2, 6, 'b'], [5, 5, 'w'], [6, 4, 'b']], [[2, 2, 'b'], [6, 6, 'w'], [2, 6, 'b'], [6, 3, 'w']], [[6, 6, 'b'], [2, 2, 'w'], [6, 2, 'b'], [3, 6, 'w'], [4, 4, 'b']]];
    let s = tx(20, 22, 'an-k', '一題：同一位玩家');
    games.forEach((st, i) => {
      const x = 22 + (i % 3) * 64, y = 40 + Math.floor(i / 3) * 76, t0 = 0.4 + i * 0.32;
      s += `<g opacity="0">${aShow(t0)}${miniBoard(x, y, 56, 7, st)}` + tx(x + 28, y + 70, 'an-s', `第 ${i + 1} 局`, 'text-anchor="middle"') + '</g>';
    });
    s += txA(150, 140, 'an-k', '…', 2.0) + txA(22, 214, 'an-n', '共 5–20 局，目標執色固定', 2.2)
      + `<line x1="214" y1="128" x2="292" y2="128" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(2.6, 0.5)}</line>`
      + `<g opacity="0">${aShow(3.0)}<rect x="296" y="104" width="96" height="48" rx="8" fill="#e3f1ef" stroke="#0b6e69"/>` + tx(344, 133, 'an-b', '模型', 'text-anchor="middle"') + '</g>'
      + `<line x1="392" y1="128" x2="${RX - 8}" y2="${ry(pick) + RH / 2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(3.5, 0.4)}</line>`;
    L.forEach((n, i) => {
      const y = ry(i), t0 = 3.3 + i * 0.04;
      s += `<rect x="${RX}" y="${y}" width="${RW}" height="${RH}" rx="4" fill="#eef0f3" stroke="#c9ced6" opacity="0">${aShow(t0)}</rect>`
        + txA(RX + RW / 2, y + 14, 'an-k', n, t0, undefined, 'text-anchor="middle"');
    });
    s += `<rect x="${RX}" y="${ry(pick)}" width="${RW}" height="${RH}" rx="4" fill="#0b6e69" opacity="0">${aShow(4.0)}</rect>`
      + txA(RX + RW / 2, ry(pick) + 14, 'an-w', '預測 2D', 4.0, undefined, 'text-anchor="middle"')
      + txA(RX + RW + 8, ry(pick) + 14, 'an-v an-ok', '1', 4.9) + txA(RX + RW + 22, ry(pick) + 14, 'an-s', '若真的是 2D', 4.9)
      + txA(RX + RW + 8, ry(pick + 1) + 14, 'an-n', '0.368', 5.5) + txA(RX + RW + 8, ry(pick - 1) + 14, 'an-n', '0.368', 5.5)
      + txA(RX + RW + 8, ry(pick + 2) + 14, 'an-s', '其餘 0', 6.1) + txA(RX + RW + 8, ry(pick - 2) + 14, 'an-s', '其餘 0', 6.1)
      + txA(296, 196, 'an-k', '差一級', 5.5) + txA(296, 214, 'an-k', '也有 e⁻¹ 分', 5.5);
    return {steps: ['① 一題：同一位玩家的 5–20 局', '② 模型輸出 10 級之一', '③ 依真實等級計分'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="一題的輸入、輸出與計分方式的動畫">${s}</svg>`, note: '示意：棋盤是示意圖；計分規則是官方規則。'};
  };
