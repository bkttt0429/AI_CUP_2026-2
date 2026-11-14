  //（競賽一：玩家辨識，v2 測試集（新增題池）；數字都在 P2，字串照 facts 抄，不計算）
  // 日期時間、檔名、Public 分數：facts_A A-P17／A-P18（pid_v2d_ft 0.8612697、pid_v2e_ft_emb5 0.8609492；METHOD_NOTES §2.33）、
  //   facts_B B-P02（pid_v2f_ft50_emb3 0.8554173；§2.35）、B-P03／B-P04（pid_v2gs_base_m90、pid_v2gs_m90；§2.42）、B-P05（pid_v2is_o1；§2.53）、
  //   facts_C C-P05（pid_v2w_n8s 04:50；§2.68）、C-P06（pid_v2sl_alls 13:42；§2.90、§2.104）、C-P07（pid_v2x_ns 19:00；§2.111）、C-P08（pid_v2c2_bs 21:03；§2.113、§2.114）；
  //   每一檔多了什麼：同一列的「差別」欄與 B-M01、B-M06、B-M07、C-R31 的概述
  const P2 = {
    climb: [
      ['11-13 00:03', 'pid_v2d_ft.csv', '0.8612697', '五個序列網路＋題池微調 100 步'],
      ['11-13 00:03', 'pid_v2e_ft_emb5.csv', '0.8609492', '嵌入改用五個種子平均'],
      ['11-13 01:30', 'pid_v2f_ft50_emb3', '0.8554173', '微調 50 步＋嵌入 emb3（退回 v2d）'],
      ['11-13 10:58', 'pid_v2gs_base_m90.csv', '0.8753896', '20 個網路＋2／3／5 局題軟性配對'],
      ['11-13 11:21', 'pid_v2gs_m90.csv', '0.8777615', '同上，題池微調改 50 步'],
      ['11-14 00:06', 'pid_v2is_o1.csv', '0.8829460', '第一批含外部帳號的官方網路'],
      ['11-14 04:50', 'pid_v2w_n8s.csv', '0.8903230', '棋盤對稱增強的網路'],
      ['11-14 13:42', 'pid_v2sl_alls.csv', '0.9016519', '對稱網路 12 個種子＋4 萬步＋加大'],
      ['11-14 19:00', 'pid_v2x_ns.csv', '0.9035072', '加長訓練＋有監督輸入（2 萬步快照）'],
      ['11-14 21:03', 'pid_v2c2_bs.csv', '0.9069275', '有監督輸入換成 3 萬步完成版'],
    ],
    // 新家族加在線上組成 v2sl_all 之上、權重 ×6 的擬題池增益：o7F、o9D、o8D、o10D 見 facts_C C-R18（§2.108）；
    //   o11U 見 C-X29（§2.113）；o12K、o12F 見 C-R26（§2.113）；全部是 ×6 的值
    fam: [
      ['o7F', '加長訓練（3–4 萬步）', '+0.0038'],
      ['o9D', '各種遮蔽', '+0.0022'],
      ['o8D', '各種設定（5 個）', '+0.0005'],
      ['o10D', '輸入子空間與資料抽樣', '+0.0014'],
      ['o11U', '有監督輸入、只用競賽棋局', '+0.0043'],
      ['o12K', '有監督輸入＋外部帳號，2 萬步快照', '+0.0045'],
      ['o12F', '有監督輸入＋外部帳號，3 萬步完成版', '+0.0087'],
    ],
    // 官方候選人上、微調前的單一網路 1 局題分數（Public 池／Private 池）：o3S、o6X、o7F、o12K 的 B／C／A 型見 facts_C C-R22、C-R23（§2.110）；
    //   o3S 的讀數前後不一（C-Q11），這裡用 C-R22 的 0.505／0.523
    single: [
      ['o3S', '一般對稱增強', '0.505', '0.523'],
      ['o6X', '對稱增強＋遮手', '0.518', '0.536'],
      ['o7F', '加長訓練', '0.515', '0.535'],
      ['o12 B 型', '落點那一半換成有監督特徵', '0.517', '0.540'],
      ['o12 C 型', '只留有監督特徵', '0.534', '0.548'],
      ['o12 A 型', '全盤那一半換成有監督特徵', '0.544', '0.557'],
    ],
  };

  CHARTS['chart-pid2-climb'] = () => '<h4>Public 分數（v2 測試集）：每個上傳檔</h4>'
    + barRows(P2.climb.map(([t, , s, note], i) => [`${t} ${note}`, parseFloat(s), s, i === P2.climb.length - 1 ? 'var(--accent)' : '#8a93a3']), 0.85, 0.91);

  CHARTS['chart-pid2-fam'] = () => '<h4>把一個新家族加進組成，擬題池分數的變化</h4>'
    + barRows(P2.fam.map(([n, note, g]) => [`${n} ${note}`, parseFloat(g), g, n === 'o12F' || n === 'o7F' ? 'var(--accent)' : '#8a93a3']), 0, 0.01);

  CHARTS['chart-pid2-single'] = () => '<h4>單一網路在官方候選人上的 1 局題分數（微調前）</h4>'
    + barRows(P2.single.map(([n, note, a, b]) => [`${n} ${note}`, parseFloat(a), `${a}／${b}`, n === 'o12 A 型' ? 'var(--accent)' : '#8a93a3']), 0.45, 0.58);

  // v2 流程（靜態圖）：題目與候選人 → 九個成分（逐手序列網路最大）→ 加權融合 → 2／3／5 局題軟性配對 → 前 5 名
  CHARTS['chart-pid2-arch'] = () => {
    const Hd = 'font-size:13px;font-weight:700;fill:#1d1f24', Sm = 'font-size:11.5px;fill:#5d6370';
    const box = (x, y, w, h, fill, stroke, inner, dash) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}"${dash ? ' stroke-dasharray="5 4"' : ''}/>${inner}`;
    const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2 - 7}" y2="${y2}" stroke="#5d6370" stroke-width="2"/><polygon points="${x2},${y2} ${x2 - 8},${y2 - 4.5} ${x2 - 8},${y2 + 4.5}" fill="#5d6370"/>`;
    const small = ['開局', '原座標開局', '方向（黑白視角）', '方向（合併）', '讓子', '局部棋形', '風格', '嵌入'];
    const top = 14, bigH = 64, step = 34, H = top + bigH + 10 + small.length * step + 26, mid = top + (bigH + 10 + small.length * step - 8) / 2;
    let s = box(12, mid - 82, 176, 64, '#fff', '#c9ced6', `<text x="100" y="${mid - 56}" text-anchor="middle" style="${Hd}">題目</text><text x="100" y="${mid - 36}" text-anchor="middle" style="${Sm}">同一人 1–5 局（黑白都有）</text>`)
      + box(12, mid + 18, 176, 64, '#fff', '#c9ced6', `<text x="100" y="${mid + 44}" text-anchor="middle" style="${Hd}">候選人 × 600</text><text x="100" y="${mid + 64}" text-anchor="middle" style="${Sm}">每人 10 局</text>`)
      + box(214, top, 300, bigH, '#0b6e69', '#0b6e69', `<text x="364" y="${top + 28}" text-anchor="middle" style="font-size:14px;font-weight:700;fill:#fff">逐手序列網路</text><text x="364" y="${top + 48}" text-anchor="middle" style="font-size:11.5px;fill:#e3f1ef">數十個平均，題池微調 50 步</text>`);
    small.forEach((name, i) => {
      const y = top + bigH + 10 + i * step;
      s += box(214, y, 300, 28, '#e3f1ef', '#0b6e69', `<text x="226" y="${y + 19}" style="${Hd}">${name}</text>`);
    });
    s += `<text x="364" y="${H - 8}" text-anchor="middle" style="${Sm}">九個成分各算一張「題目 × 候選人」的分數表</text>`
      + arrow(190, mid - 50, 210, mid - 50) + arrow(190, mid + 50, 210, mid + 50)
      + arrow(518, mid, 560, mid)
      + box(564, mid - 36, 112, 72, '#fff', '#0b6e69', `<text x="620" y="${mid + 5}" text-anchor="middle" style="${Hd}">加權融合</text>`)
      + arrow(678, mid, 712, mid)
      + box(716, mid - 36, 132, 72, '#fff', '#0b6e69', `<text x="782" y="${mid - 8}" text-anchor="middle" style="${Hd}">2／3／5 局題</text><text x="782" y="${mid + 14}" text-anchor="middle" style="${Hd}">軟性配對</text>`)
      + arrow(850, mid, 872, mid)
      + box(876, mid - 36, 72, 72, '#0b6e69', '#0b6e69', `<text x="912" y="${mid - 4}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">每題</text><text x="912" y="${mid + 16}" text-anchor="middle" style="font-size:12px;fill:#e3f1ef">前 5 名</text>`);
    return `<svg viewBox="0 0 960 ${H}" role="img" aria-label="v2 流程：題目與候選人、九個成分、加權融合、2／3／5 局題軟性配對、前 5 名">${s}</svg>`;
  };
