  // ---- 第 7 章（11-11 改寫：前三頁是 11-06 之後的三步，第四頁是最終檔與揭露）----
  // 7·1 右欄：16 個決策推移的 Public 解碼（NUM.probes）。長條 ＝ X（差一級的淨對錯題數），由分數差唯一解出
  ANIM.probes = () => {
    const P = NUM.probes, RH = 14, top = 62, X0 = 322, U = 19, H = top + P.length * RH + 24, X = v => X0 + v * U;
    const col = {keep: '#0b6e69', keep2: '#0b6e69', rule: '#b98900', drop: '#d1343a'}, tail = {keep: '　保留', keep2: '　保留', rule: '　刪除', drop: ''};
    const tEnd = 2.0 + P.length * 0.3 + 0.4;
    let s = tx(12, 18, 'an-t', '一次上傳只多一個推移：分數差就是那幾題的對錯')
      + txA(12, 38, 'an-n', 'ct13 − ct9 ＝ +0.0126424 ＝ 8 × 0.63212 ÷ 400', 0.8, undefined, 'style="fill:#0b6e69"')
      + txA(628, 38, 'an-k', '→ 8 題全對', 1.5, undefined, 'text-anchor="end"')
      + tx(160, 54, 'an-s', 'X ＝ 改對 − 改錯', 'text-anchor="end"')
      + [-5, 0, 5].map(v => tx(X(v), 54, 'an-s', (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v), 'text-anchor="middle"')).join('')
      + `<line x1="${X0}" y1="${top - 2}" x2="${X0}" y2="${top + P.length * RH}" stroke="#1d1f24" stroke-width="1.2"/>`;
    P.forEach(([dir, file, nq, x, y, dec, st], i) => {
      const yy = top + i * RH + 7, t0 = 2.0 + i * 0.3, c = col[st], kept = st === 'keep' || st === 'keep2';
      if (kept) s += `<rect x="8" y="${yy - 7}" width="624" height="${RH}" fill="#0b6e69" fill-opacity=".09" opacity="0">${aShow(tEnd)}</rect>`;
      s += txA(160, yy + 4, 'an-s', `${dir}（${nq} 題）`, t0, undefined, `text-anchor="end" style="fill:#1d1f24${kept ? ';font-weight:700' : ''}"`)
        + (x ? `<rect x="${Math.min(X0, X(x))}" y="${yy - 4.5}" width="${Math.abs(x) * U}" height="9" rx="2" fill="${c}" opacity="0">${aShow(t0)}</rect>`
          : `<circle cx="${X0}" cy="${yy}" r="3.5" fill="${c}" opacity="0">${aShow(t0)}</circle>`)
        + txA(x >= 0 ? X(x) + 6 : X(x) - 6, yy + 4, 'an-s', (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x), t0, undefined, `text-anchor="${x >= 0 ? 'start' : 'end'}" style="fill:${c};font-weight:700"`)
        + txA(628, yy + 4, 'an-s', dec + tail[st], t0 + 0.1, undefined, `text-anchor="end"${st === 'drop' ? '' : ` style="fill:${c};font-weight:700"`}`);
    });
    s += txA(12, H - 7, 'an-b', '只留兩個：1D→A（8 題全對）與 4D→5D（使用者決定）', tEnd);
    return {steps: ['① 每個探測檔只比前一檔多一個推移', '② 分數差唯一分解成對錯題數', '③ 16 個方向留下 2 個'],
      svg: `<svg viewBox="0 0 640 ${H}" role="img" aria-label="十六個決策推移的 Public 解碼結果動畫">${s}</svg>`,
      note: 'X 由 400 × ΔPublic ＝ 0.63212 X ＋ 0.36788 Y 唯一解出（設計文件 M52–M58；帳本 m58_decode.py）。括號內是 Public 翻動的題數，右欄是解碼出的對錯。第一列對 ct9，其餘對 ct13。4D→5D 沒達到事先寫的門檻（9 題要 8 對），由使用者決定保留；B→C 依事先登記的規則刪除。'};
  };
  ANIM.probes.dur = 11;

  // 7·2 右欄：對手錨點的機制（上排，示意）與校正表（下排，NUM.anchor，實測）
  ANIM.anchor = () => {
    const A = NUM.anchor, Hd = 'font-size:12.5px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#fff" stroke="#c9ced6"/>${inner}</g>`;
    const arrow = (x1, y, x2, t0) => `<line x1="${x1}" y1="${y}" x2="${x2 - 6}" y2="${y}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.3)}</line>`
      + `<polygon points="${x2},${y} ${x2 - 7},${y - 4} ${x2 - 7},${y + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.3)}</polygon>`;
    // 上排：一局 → 在訓練玩家中比對 → 讀到一個標籤與一個比對分數
    let s = box(10, 12, 170, 88, 0.4, miniBoard(18, 22, 68, 7, [[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 5, 'w'], [5, 3, 'b']])
        + `<text x="96" y="46" style="${Hd}">測試題的一局</text><text x="96" y="68" style="${Sm}">只看對手的落子</text>`)
      + arrow(182, 56, 204, 1.2);
    let dots = '';
    for (let r = 0; r < 3; r++) for (let c = 0; c < 14; c++) dots += `<circle cx="${226 + c * 12}" cy="${54 + r * 13}" r="3.2" fill="#c9ced6"/>`;
    s += box(206, 12, 194, 88, 1.5, `<text x="303" y="36" text-anchor="middle" style="${Hd}">在 8,007 位訓練玩家中比對</text>${dots}`)
      + `<circle cx="${226 + 9 * 12}" cy="67" r="5.5" fill="#0b6e69" stroke="#fff" stroke-width="1.5" opacity="0">${aShow(2.4)}</circle>`
      + arrow(402, 56, 424, 2.9)
      + box(426, 12, 204, 88, 3.2, `<text x="438" y="38" style="${Hd}">最像的那位：標籤 3D</text><text x="438" y="60" style="${Sm}">比對分數 ＝ 第一名 − 第二名</text><text x="438" y="80" style="${Sm}">和冒充者的分布比 → 落在哪一帶</text>`);
    // 下排左：九個帶的 a（比對為真的機率）
    const base = 226, BX = i => 20 + i * 35;
    s += txA(12, 124, 'an-k', 'a：這個比對是真的機率（依冒充者分位帶）', 4.2)
      + `<line x1="12" y1="${base}" x2="338" y2="${base}" stroke="#c9ced6"/>`
      + tx(BX(4) + 12, 255, 'an-s', '冒充者比對分數的分位（%）', 'text-anchor="middle"');
    A.a.forEach((v, i) => {
      const h = Math.max(1.5, v * 84), t0 = 4.4 + i * 0.12;
      s += `<rect x="${BX(i)}" y="${(base - h).toFixed(1)}" width="24" height="${h.toFixed(1)}" rx="2" fill="${v >= 0.5 ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`
        + txA(BX(i) + 12, (base - h - 5).toFixed(1), 'an-n', v ? v.toFixed(2) : '0', t0, undefined, 'text-anchor="middle" style="font-size:10.5px"')
        + tx(BX(i) + 12, 240, 'an-s', A.bands[i], 'text-anchor="middle" style="font-size:10px"');
    });
    // 下排右：真配對時標籤差 d 的倍率 K／P0
    const KX = i => 388 + i * 72;
    s += txA(372, 124, 'an-k', '真配對時，對手標籤落在各級的倍率 K／P0', 6.0)
      + `<line x1="384" y1="${base}" x2="628" y2="${base}" stroke="#c9ced6"/>`
      + `<line x1="384" y1="${base - 26}" x2="628" y2="${base - 26}" stroke="#5d6370" stroke-dasharray="4 3" opacity="0">${aShow(6.0)}</line>`
      + txA(634, base - 30, 'an-s', '背景 ＝ 1', 6.0, undefined, 'text-anchor="end" style="font-size:10.5px"')
      + tx(KX(1) + 24, 255, 'an-s', '對手的標籤比本人', 'text-anchor="middle"');
    ['低一級', '同級', '高一級'].forEach((n, i) => {
      const v = A.kp0[i], h = v * 26, t0 = 6.2 + i * 0.2;
      s += `<rect x="${KX(i)}" y="${(base - h).toFixed(1)}" width="48" height="${h.toFixed(1)}" rx="2" fill="${i === 1 ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`
        + txA(KX(i) + 24, (base - h - 5).toFixed(1), 'an-n', '×' + v.toFixed(1), t0, undefined, 'text-anchor="middle" style="font-size:11px"')
        + tx(KX(i) + 24, 240, 'an-s', n, 'text-anchor="middle"');
    });
    s += txA(12, 274, 'an-b', '題目後驗 × 各局證據 → 期望分決策；只在邊界區動，ct28 的 38 題鎖定', 7.4);
    return {steps: ['① 每一局：用對手的落子認人', '② 比對分數決定信多少（a）', '③ 標籤變成鄰近幾級的倍率，乘進後驗'],
      tech: '認對手：開局、風格、局部棋形、方位、讓子五個成分，加上學出來的條件化比對（下一頁）；校正：冒充者分位帶 × 混合模型，依驗證折交叉擬合',
      svg: `<svg viewBox="0 0 640 282" role="img" aria-label="對手錨點的機制與校正表動畫">${s}</svg>`,
      note: '上排是示意（「3D」只是例子）。下排是實測的校正表：互先局、只用開局模型時的值（設計文件 M71 第 2 步）；換成五成分後，同級的 K 從 0.50 升到 0.54。'};
  };
  ANIM.anchor.dur = 11;

  // 7·3 右欄：左 ＝ 錨點對底的 CV 增益（三種認對手的方法 × 兩組 seed）；右 ＝ Public 讀數（ct28 → ct40／ct42 → ct44）
  ANIM.ladder = () => {
    const A = NUM.anchor, BX = 112, SC = 9300;
    let s = tx(12, 20, 'an-t', '錨點加在集成上的 CV 增益') + tx(12, 37, 'an-s', '深：seed 0–2　淺：seed 6–8');
    A.cv.forEach(([name, src, v1, v2], i) => {
      const y = 52 + i * 46, t0 = 0.5 + i * 0.9;
      s += tx(BX - 8, y + 12, 'an-k', name, 'text-anchor="end" style="font-size:12px;fill:#1d1f24"') + tx(BX - 8, y + 27, 'an-s', src, 'text-anchor="end"')
        + [[v1, '#0b6e69', 0], [v2, '#7fb8b3', 17]].map(([v, c, dy]) => `<rect x="${BX}" y="${y + dy}" width="${(v * SC).toFixed(1)}" height="14" rx="3" fill="${c}" opacity="0">${aShow(t0 + dy / 60)}</rect>`
          + txA((BX + v * SC + 6).toFixed(1), y + dy + 11, 'an-n', '+' + v.toFixed(4), t0 + 0.2 + dy / 60, undefined, 'style="font-size:11px"')).join('');
    });
    s += txA(12, 206, 'an-b', '比對 對 五成分：+0.0058／+0.0071', 3.4) + txA(12, 224, 'an-k', '兩組 seed、八個折全正', 3.4);
    const P = A.pub, PX = i => 424 + i * 60, lo = 0.77, hi = 0.81, PY = v => 200 - (v - lo) / (hi - lo) * 150, fin = NUM.final;
    s += tx(372, 20, 'an-t', 'Public（同一批 400 題）');
    [0.77, 0.78, 0.79, 0.80, 0.81].forEach(g => { const y = PY(g).toFixed(1);
      s += `<line x1="394" y1="${y}" x2="630" y2="${y}" stroke="#e3e6ea" stroke-dasharray="3 4"/>` + tx(388, (PY(g) + 4).toFixed(1), 'an-s', g.toFixed(2), 'text-anchor="end"'); });
    const seg = (i, j, c, dash, t0) => `<line x1="${PX(i)}" y1="${PY(P[i][1]).toFixed(1)}" x2="${PX(j)}" y2="${PY(P[j][1]).toFixed(1)}" stroke="${c}" stroke-width="2.2"${dash ? ' stroke-dasharray="4 4"' : ''} opacity="0">${aShow(t0)}</line>`;
    const T = [4.0, 4.7, 5.5, 6.3];
    s += seg(0, 1, '#d1343a', true, T[1]) + seg(0, 2, '#8a93a3', false, T[2]) + seg(2, 3, '#0b6e69', false, T[3]);
    P.forEach(([name, v, sub], i) => {
      const col = name === 'ct40' ? '#d1343a' : name === fin ? '#0b6e69' : '#5d6370', r = name === fin ? 7 : 5, below = name === 'ct40';
      s += `<circle cx="${PX(i)}" cy="${PY(v).toFixed(1)}" r="${r}" fill="${col}" opacity="0">${aShow(T[i])}</circle>`
        + (name === 'ct42' ? txA(PX(i) - 8, (PY(v) - 6).toFixed(1), 'an-n', v.toFixed(4), T[i] + 0.1, undefined, `text-anchor="end" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`)   // 左上：避開往 ct44 的連線
          : txA(PX(i), (PY(v) + (below ? 18 : name === fin ? -20 : -10)).toFixed(1), 'an-n', v.toFixed(4), T[i] + 0.1, undefined, `text-anchor="middle" style="font-size:11px;paint-order:stroke;stroke:#fff;stroke-width:3px;fill:${col}"`))
        + tx(PX(i), 218, 'an-k', name, 'text-anchor="middle" style="font-size:12px"') + tx(PX(i), 233, 'an-s', sub, 'text-anchor="middle"');
    });
    const iF = P.findIndex(p => p[0] === fin);
    if (iF >= 0) s += `<circle cx="${PX(iF)}" cy="${PY(P[iF][1]).toFixed(1)}" r="13" fill="none" stroke="#0b6e69" stroke-width="2" opacity="0">${aShow(6.9)}</circle>`;
    s += txA(630, 253, 'an-h', 'ct44 對 ct28 +0.0240；CV 期望 +0.012–0.014', 7.3, undefined, 'text-anchor="end"');
    return {steps: ['① 認對手的方法變強，CV 增益跟著升', '② ct40：先錨點後推移，Public 掉分', '③ 鎖定推移後：ct42、ct44'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="錨點的 CV 增益與 Public 讀數動畫">${s}</svg>`,
      note: 'CV：錨點加在 6 成員集成上的配對增益（folds 1–4、類別平衡；驗證端的錨點密度，測試端約 0.74–0.85 倍）。M82 那次執行裡重算的五成分是 +0.0135／+0.0133。Public 是使用者回報的分數；兩檔差的雜訊約 ±0.008。'};
  };
  ANIM.ladder.dur = 10;

  // 7·4 右欄：Public 分數（橘色那一步是由 Public 回饋選出的兩個推移）
  CHARTS['chart-final'] = () => '<h4>Public 分數（使用者回報；長條從 0.40 起；橘色那一步用了 Public 回饋）</h4>'
    + barRows([['官方 baseline', 0.4631, '0.4631', '#9aa3b2'], ['主檔（MLP）', 0.7148, '0.7148', '#8a93a3'], ['C＋T', 0.7494, '0.7494', '#8a93a3'],
      ['ct7（11-06）', 0.7687835, '0.7687835', '#8a93a3'], ['ct9（對比損失）', 0.7622032, '0.7622032', '#8a93a3'], ['ct28（兩個推移）', 0.7827471, '0.7827471', 'var(--warn)'],
      ['ct42（五成分錨點）', 0.7880062, '0.7880062', '#7fb8b3'], ['ct44（最終檔）', 0.8067107, '0.8067107', 'var(--accent)']], 0.4, 0.82);

  // 7·4 左欄：最後一筆上傳決定 Private（NUM.uploads 的 13 個檔 ＋ 截止前的最後一筆）
  ANIM.last = () => {
    const U = NUM.uploads, n = U.length, X = i => 26 + i * 42, Y = 60, xl = X(n), fin = NUM.final;
    let s = `<line x1="12" y1="${Y}" x2="${xl + 24}" y2="${Y}" stroke="#d9dbe0" stroke-width="3"/>`
      + `<line x1="${xl + 28}" y1="${Y - 34}" x2="${xl + 28}" y2="${Y + 34}" stroke="#a4451c" stroke-width="2" stroke-dasharray="4 3"/>` + tx(xl + 34, Y - 22, 'an-k', '截止');
    U.forEach(([name], i) => { s += `<circle cx="${X(i)}" cy="${Y}" r="10" fill="${name === fin ? '#7fb8b3' : '#c9ced6'}" opacity="0">${aShow(0.3 + i * 0.12)}</circle>`
      + txA(X(i), Y + 28, 'an-s', name, 0.3 + i * 0.12, undefined, 'text-anchor="middle" style="font-size:10.5px"'); });
    // 先演錯的一支、最後停在正確狀態（減少動態效果時停在完成畫面，看到的是最終檔 ✓）
    s += `<circle cx="${xl}" cy="${Y}" r="15" fill="#d1343a" opacity="0">${aShow(2.6, 5.0)}</circle>` + txA(xl, Y + 4, 'an-l', '其他', 2.6, 5.0, 'text-anchor="middle"')
      + `<circle cx="${xl}" cy="${Y}" r="15" fill="#0b6e69" opacity="0">${aShow(5.6)}</circle>` + txA(xl, Y + 4, 'an-l', fin, 5.6, undefined, 'text-anchor="middle"')
      + txA(12, Y + 66, 'an-v an-no', `之後若再傳別的檔、沒有補傳 ${fin} → Private 用錯檔 ✗`, 3.0, 5.0)
      + txA(12, Y + 66, 'an-v an-ok', `Private 只看最後一筆：必須是 ${fin} ✓`, 6.0);
    return {steps: ['① 之前的上傳只影響 Public', '② 截止前最後一筆決定 Private'],
      svg: `<svg viewBox="0 0 640 ${Y + 82}" role="img" aria-label="最後一筆上傳的動畫">${s}</svg>`, note: '示意：只畫模型檔與被保留的檔；截止 11-24 23:59:59；每日 5 次上傳。'};
  };
