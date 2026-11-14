  // ---- 第 4 章 ----
  // 第 7 頁（橫跨整張卡）：從棋譜到等級的三個階段
  ANIM.pipeline = () => {
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, fill, stroke, t0, inner) => `<g opacity="0">${aShow(t0)}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}"/>${inner}</g>`;
    const arrow = (x1, y1, x2, y2, t0) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#5d6370" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(t0, 0.4)}</line>`
      + `<polygon points="${x2},${y2} ${x2 - 7},${y2 - 4} ${x2 - 7},${y2 + 4}" fill="#5d6370" opacity="0">${aShow(t0 + 0.4)}</polygon>`;
    let s = '';
    // 段落標題
    [['階段一　每手證據', 214, 0.3], ['階段二　整局與整題', 560, 3.5], ['階段三　集成與決策', 830, 6.2]].forEach(([t, x, t0]) => { s += txA(x, 22, 'an-t', t, t0, undefined, 'text-anchor="middle"'); });
    // 棋譜
    s += box(14, 96, 104, 112, '#fff', '#c9ced6', 0.4, miniBoard(30, 104, 72, 7, [[2, 2, 'b'], [6, 6, 'w'], [6, 2, 'b'], [2, 6, 'w'], [4, 4, 'b']]) + `<text x="66" y="198" text-anchor="middle" style="${Sm}">SGF 棋譜</text>`)
      + arrow(120, 152, 146, 152, 1.0)
      + box(150, 112, 104, 80, '#e3f1ef', '#0b6e69', 1.3, `<text x="202" y="144" text-anchor="middle" style="${Hd}">KataGo 重播</text><text x="202" y="164" text-anchor="middle" style="${Sm}">中國規則、不搜尋</text>`)
      + arrow(256, 152, 282, 152, 1.9);
    // token 162 維
    const seg = [['盤面 trunk', 64, '#9ecae1'], ['落點 trunk', 64, '#6baed6'], ['L1b', 11, '#0b6e69'], ['Human SL', 19, '#e69f00'], ['旗標', 4, '#8a93a3']];
    let y0 = 54; const SC = 0.95;
    s += txA(298, 46, 'an-k', '每手 token 162 維', 2.2, undefined, 'style="font-size:12px"');
    seg.forEach(([n, d, c], k) => {
      const h = Math.max(9, d * SC), t0 = 2.3 + k * 0.25;
      s += `<rect x="298" y="${y0.toFixed(1)}" width="26" height="${h.toFixed(1)}" fill="${c}" opacity="0">${aShow(t0)}</rect>`
        + txA(330, (y0 + h / 2 + 4).toFixed(1), 'an-s', `${n} ${d}`, t0);
      y0 += h + 2;
    });
    // 一局的 token 序列：本人手與對手手交替
    for (let k = 0; k < 14; k++) {
      const x = 420 + k * 9, own = k % 2 === 0, t0 = 3.6 + k * 0.05;
      s += `<rect x="${x}" y="110" width="7" height="84" rx="1.5" fill="${own ? '#0b6e69' : '#7fb8b3'}" opacity="0">${aShow(t0)}</rect>`;
    }
    s += txA(420, 92, 'an-s', '一局的每一手', 3.6) + txA(420, 104, 'an-s', '深：本人　淺：對手', 3.6) + arrow(548, 152, 574, 152, 4.4);
    // C / T
    s += box(578, 92, 110, 50, '#fff', '#0b6e69', 4.7, `<text x="633" y="114" text-anchor="middle" style="${Hd}">C　逐手 MLP</text><text x="633" y="132" text-anchor="middle" style="${Sm}">隱藏 110</text>`)
      + box(578, 152, 110, 50, '#fff', '#0b6e69', 5.0, `<text x="633" y="174" text-anchor="middle" style="${Hd}">T　擴張卷積</text><text x="633" y="192" text-anchor="middle" style="${Sm}">5 層、感受野 63 手</text>`)
      + box(578, 214, 110, 44, '#f2f3f6', '#c9ced6', 5.5, `<text x="633" y="232" text-anchor="middle" style="${Hd}">＋ R 64 維</text><text x="633" y="249" text-anchor="middle" style="${Sm}">題目特徵</text>`)
      + txA(633, 84, 'an-s', '平均／標準差池化：每局 → 每題', 5.2, undefined, 'text-anchor="middle"')
      + arrow(690, 152, 716, 152, 6.0);
    // 24 個成員
    for (let k = 0; k < 24; k++) {
      const x = 724 + (k % 6) * 17, y = 104 + Math.floor(k / 6) * 17, t0 = 6.3 + k * 0.03;
      s += `<rect x="${x}" y="${y}" width="13" height="13" rx="2" fill="${k % 2 ? '#e69f00' : '#0b6e69'}" opacity="0">${aShow(t0)}</rect>`;
    }
    s += txA(724, 96, 'an-s', '24 個成員', 6.3) + txA(724, 190, 'an-s', '4 折×{C,T}×3 seed', 6.6)
      + txA(724, 206, 'an-s', '十級機率平均', 6.8) + arrow(830, 152, 852, 152, 7.4)
      + box(856, 106, 96, 92, '#0b6e69', '#0b6e69', 7.8, `<text x="904" y="134" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">期望分最高</text><text x="904" y="172" text-anchor="middle" style="font-size:30px;font-weight:700;fill:#fff">3D</text>`)
      + txA(952, 222, 'an-s', '鄰級機率 × 0.368 也算分', 8.2, undefined, 'text-anchor="end"');
    return {steps: ['階段一（表示）：重播、每手 162 維 token', '階段二（聚合）：C／T 讀整局、池化到每題', '階段三：24 個成員平均、期望分決策'],
      tech: 'KataGo b18c384nbt-humanv0（凍結）＋ L1b 微調 trunk 與 10 個等級頭；PyTorch；集成成員都經 CV 驗證',
      svg: `<svg viewBox="0 0 960 264" role="img" aria-label="從棋譜到等級的三階段架構動畫">${s}</svg>`,
      note: 'token 各段的長度依維度比例縮放；「3D」只是示意輸出。'};
  };
  Object.assign(ANIM.pipeline, {dur: 12});

  // 第 8 頁右欄：同一手在 18 個 profile 下的 log p（示意），以及 L1b 10 個等級頭
  ANIM.profiles = () => {
    const X = i => 260 + i * 20.5, Y = v => 30 + (-v - 0.8) / 2.8 * 104;   // log p −0.8 … −3.6
    const A = Array.from({length: 18}, (_, i) => -1.15 - 1.9 * Math.pow(i / 17, 1.1));
    const B = Array.from({length: 18}, (_, i) => -3.3 + 2.2 * Math.pow(i / 17, 0.9));
    const pts = arr => arr.map((v, i) => [X(i).toFixed(1), Y(v).toFixed(1)]);
    let s = miniBoard(18, 34, 200, 9, [[3, 3, 'b'], [7, 7, 'w'], [7, 3, 'b'], [3, 7, 'w'], [5, 2, 'b'], [6, 7, 'w'], [4, 5, 'b'], [8, 6, 'w']])
      + `<circle cx="${18 + 6 * 20}" cy="${34 + 5 * 20}" r="9" fill="none" stroke="#e69f00" stroke-width="3" opacity="0">${aShow(0.4)}</circle>`
      + `<circle cx="${18 + 4 * 20}" cy="${34 + 7 * 20}" r="9" fill="none" stroke="#0072b2" stroke-width="3" opacity="0">${aShow(0.4)}</circle>`
      + tx(18, 254, 'an-s', '同一局面的兩個候選手');
    s += `<line x1="${X(0)}" y1="138" x2="${X(17)}" y2="138" stroke="#c9ced6"/>` + tx(X(0), 152, 'an-s', '12k', 'text-anchor="middle"') + tx(X(17), 152, 'an-s', '6d', 'text-anchor="middle"')
      + tx((X(0) + X(17)) / 2, 152, 'an-s', '18 個 profile', 'text-anchor="middle"') + tx(250, 24, 'an-k', '每手的 log p')
      + `<polyline points="${pstr(pts(A))}" fill="none" stroke="#e69f00" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(0.8, 1.2)}</polyline>`
      + `<polyline points="${pstr(pts(B))}" fill="none" stroke="#0072b2" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" opacity="0">${aDraw(1.6, 1.2)}</polyline>`
      + txA(X(17) + 4, (Y(A[17]) + 4).toFixed(1), 'an-s', '橙', 1.9) + txA(X(17) + 4, (Y(B[17]) + 4).toFixed(1), 'an-s', '藍', 2.7)
      + txA(250, 170, 'an-k', '橙：弱 profile 較常下　藍：強 profile 較常下', 2.8);
    const L = NUM.levels, Lv = [-2.9, -2.6, -2.3, -2.05, -1.85, -1.68, -1.56, -1.5, -1.52, -1.6];
    L.forEach((n, i) => {
      const x = 262 + i * 36, h = (3.2 + Lv[i]) * 26, t0 = 4.2 + i * 0.08;
      s += `<rect x="${x}" y="${(240 - h).toFixed(1)}" width="26" height="${h.toFixed(1)}" rx="2" fill="#0b6e69" opacity="0">${aShow(t0)}</rect>`
        + txA(x + 13, 254, 'an-s', n, t0, undefined, 'text-anchor="middle"');
    });
    s += txA(250, 188, 'an-k', 'L1b：10 個等級頭，同一手在各級的 log p', 4.0);
    return {steps: ['① 同一局面、兩個候選手', '② 18 個 profile 的 log p', '③ L1b 10 個等級頭'],
      tech: '每手只存實際落子那一格：Human SL 18 個值、L1b 10 個值；對手的手用同樣方式評分',
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="每手證據的動畫">${s}</svg>`,
      note: '示意：曲線與長條的形狀是示意；實際每手只取實際落子那一格。'};
  };

  // 第 9 頁右欄：token 的 Human SL 欄原本只在本人手上有值；M14 把對手手上也填上
  ANIM.opp = () => {
    const n = 16, X = k => 40 + k * 36, W = 26;
    let s = tx(30, 22, 'an-k', '一局的 16 手（左起）') + tx(30, 40, 'an-s', '上段：L1b　下段：Human SL');
    for (let k = 0; k < n; k++) {
      const own = k % 2 === 0, x = X(k);
      s += `<rect x="${x}" y="54" width="${W}" height="34" rx="2" fill="${own ? '#0b6e69' : '#7fb8b3'}"/>`
        + `<rect x="${x}" y="92" width="${W}" height="44" rx="2" fill="${own ? '#e69f00' : '#ffffff'}" stroke="${own ? '#e69f00' : '#c9ced6'}" stroke-dasharray="${own ? '0' : '3 3'}"/>`
        + tx(x + W / 2, 152, 'an-s', own ? '本人' : '對手', 'text-anchor="middle"');
      if (!own) s += `<rect x="${x}" y="92" width="${W}" height="44" rx="2" fill="#f0b45a" opacity="0">${aShow(2.4 + k * 0.08)}</rect>`;
    }
    s += txA(30, 178, 'an-h', '之前：對手手上的 Human SL 欄是 0', 0.6, 2.2)
      + txA(30, 178, 'an-b', 'M14：對手手上也填對手的 Human SL 似然（不加維度）', 2.5);
    const bar = (y, label, v, col, t0) => `<rect x="190" y="${y}" width="0" height="16" rx="3" fill="${col}">${aTo('width', 0, ((v - 0.70) / 0.05 * 380).toFixed(1), t0)}</rect>`
      + tx(182, y + 13, 'an-k', label, 'text-anchor="end"') + txA(190 + (v - 0.70) / 0.05 * 380 + 6, y + 13, 'an-n', v.toFixed(4), t0 + 0.6);
    s += bar(198, '6 模型集成：之前', 0.7288, '#8a93a3', 4.2) + bar(222, '加對手 HSL', 0.7383, '#0b6e69', 4.8)
      + `<line x1="190" y1="194" x2="190" y2="242" stroke="#1d1f24" stroke-width="1"/>`
      + txA(30, 258, 'an-k', '集成 +0.0096；局數越少增益越大：n 5 +0.0129、10 +0.0090、20 +0.0067', 5.8);
    return {steps: ['① 本人手有 L1b 與 Human SL', '② 對手手只有 L1b', '③ 對手手補上 Human SL'],
      svg: `<svg viewBox="0 0 640 268" role="img" aria-label="對手 Human SL 通道的動畫">${s}</svg>`,
      note: '長條是 M14 第 2 步的 CV 實測（folds 1–4，6 對 6），長度從 0.70 起算；方塊是示意。'};
  };

  // 第 10 頁右欄：pub_q_0177 的十級機率 → 期望分；機率最高 2D，期望分最高 3D
  ANIM.decide = () => {
    const L = NUM.levels, p = NUM.decide.p, e = Math.exp(-1);
    const es = p.map((v, c) => v + e * ((p[c - 1] || 0) + (p[c + 1] || 0)));
    const am = p.indexOf(Math.max(...p)), ed = es.indexOf(Math.max(...es));
    const base = 214, X = i => 56 + i * 56, W = 30, Hs = v => v / 0.40 * 170;
    let s = `<line x1="40" y1="${base}" x2="${X(9) + W + 14}" y2="${base}" stroke="#c9ced6"/>` + tx(40, 22, 'an-k', `ct7 對 ${NUM.decide.qid} 的十級機率`);
    L.forEach((n, i) => {
      const h = Hs(p[i]), h2 = Hs(es[i]), t0 = 0.4 + i * 0.06;
      s += `<rect x="${X(i)}" y="${base}" width="${W}" height="0" rx="3" fill="${i === am ? '#e69f00' : '#9aa3b2'}">${aTo('y', base, (base - h).toFixed(1), t0)}${aTo('height', 0, h.toFixed(1), t0)}</rect>`
        + `<rect x="${X(i) - 4}" y="${(base - h2).toFixed(1)}" width="${W + 8}" height="${h2.toFixed(1)}" rx="3" fill="none" stroke="${i === ed ? '#0b6e69' : '#5d6370'}" stroke-width="${i === ed ? 3 : 1.4}" stroke-dasharray="${i === ed ? '0' : '4 3'}" opacity="0">${aShow(3.4 + i * 0.05)}</rect>`
        + tx(X(i) + W / 2, base + 18, 'an-k', n, 'text-anchor="middle"');
    });
    s += txA(X(am) + W / 2, (base - Hs(p[am]) - 8).toFixed(1), 'an-n', p[am].toFixed(3), 1.4, 3.2, 'text-anchor="middle" style="fill:#b06f00"')
      + txA(632, 22, 'an-h', `機率最高：${L[am]}（${p[am].toFixed(3)}）`, 1.6, undefined, 'text-anchor="end"')
      + txA(632, 42, 'an-b', `期望分最高：${L[ed]}（${es[ed].toFixed(3)}）`, 4.4, undefined, 'text-anchor="end"')
      + txA(X(ed) + W / 2, (base - Hs(es[ed]) - 8).toFixed(1), 'an-n', es[ed].toFixed(3), 4.4, undefined, 'text-anchor="middle" style="fill:#0b6e69"')
      + txA(40, 252, 'an-k', '虛線框 ＝ 期望分 ＝ p(c) ＋ 0.368［p(c−1) ＋ p(c+1)］：鄰居的機率也算分', 3.4);
    return {steps: ['① 24 個成員平均後的十級機率', '② 每級的期望分', '③ 選期望分最高的等級'],
      svg: `<svg viewBox="0 0 640 262" role="img" aria-label="期望分決策的動畫">${s}</svg>`,
      note: `機率是 ct7 在 ${NUM.decide.qid} 的實際輸出（submission_rank_ct7_proba.csv）；ct7 的 800 題中有 23 題兩種選法不同。`};
  };
