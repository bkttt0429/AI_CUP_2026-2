  // ---- 名詞註記（11-06 加；使用者：「第一次看到報告的人不知道這啥，希望簡單註記、內容不要太多」）----
  // 每一頁（含推導節）第一次出現的術語加虛線底；滑鼠指向、鍵盤聚焦或手指點一下，顯示一行說明。頁面文字不變。
  // 名詞與說明只改這裡（[術語, 說明]）；英數術語只在前後不是英數、底線、點、斜線時才標，避免標到檔名裡。
  {
    const TERMS = [
      ['KataGo', '目前最強的開源圍棋 AI。這裡只借用它的神經網路：每個局面算一次（不搜尋），取每一手的機率與內部向量。'],
      ['Human SL', 'KataGo 的人類棋風模型（b18c384nbt-humanv0）：給定局面與棋力設定（12k–6d），預測那個棋力的人會下哪裡。'],
      ['humanv0', 'Human SL 的網路權重（b18c384nbt-humanv0）。'],
      ['幽玄の間', '日本棋院的線上對弈平台；本比賽的棋譜都來自這裡（2021–2023）。'],
      ['讓子', '棋力差較多時，弱方執黑、先在星位擺 2–9 顆子；記錄裡寫成 HD[2]–HD[9]。'],
      ['讓先', '棋力只差一點時，弱方執黑先下、白方不貼目；記錄裡寫成 HD[1]。'],
      ['互先', '棋力相近的一般對局：黑方先下、要貼目；記錄裡寫成 HD[0]。'],
      ['HD', '棋譜（SGF）裡的讓子標記：HD[0] 互先、HD[1] 讓先、HD[2–9] 讓 2–9 子。'],
      ['執色', '這位玩家在那一局執黑還是執白。'],
      ['貼目', '黑方先下占便宜，終局時補給白方的目數（例如 6.5 或 7.5 目）。'],
      ['中國規則', '數子計分的規則；這裡只是重播棋局時餵給網路的設定，不是那盤棋實際用的規則。'],
      ['段級位', '級（k）的數字越小越強，1k 之上是 1 段（d），段數越大越強。'],
      ['RP', '幽玄の間的積分（レーティング點數）；每 1,000 點換一個段級位。'],
      ['profile', 'Human SL 的棋力設定，從 12k 到 6d 共 18 個。'],
      ['L1b', '在 KataGo 主幹網路（小幅微調）上，為 10 個等級各接一個落子預測頭，用本比賽的標籤訓練；每手輸出「這一級的人會下這手」的 10 個 log 機率。'],
      ['trunk', '神經網路的主幹：每個局面算出一組內部特徵；我們取整盤與落點兩組，各固定投影成 64 維，當作「這手棋的樣子」。'],
      ['token', '把一手棋整理成的一個特徵向量。'],
      ['C＋T', '兩種讀整局的模型：C 對每一手各用一個小 MLP，再對整局取平均與標準差；T 用擴張卷積，每次看連續約 63 手（前後各約 31 手）。'],
      ['MLP', '多層感知器：最基本的全連接神經網路。'],
      ['集成', '把多個模型的機率平均，降低單一模型的隨機誤差。'],
      ['期望分', '把差一級的部分分也算進去的預期得分；選期望分最高的等級，不一定是機率最高的那個。'],
      ['baseline', '主辦方提供的參考做法與它的分數。'],
      ['Public', '比賽期間排行榜即時顯示的 400 題分數。'],
      ['Private', '截止後才公布的另外 400 題分數，決定最終名次。'],
      ['CV', '交叉驗證：玩家分成 5 份（fold），輪流留一份當考題；分數都來自沒看過那些玩家的模型。'],
      ['fold', '交叉驗證裡的一份玩家；共 5 份，編號 0–4。'],
      ['OOF', 'out-of-fold：每位玩家的預測都來自沒看過他的模型。'],
      ['bootstrap', '把玩家重抽幾千次、每次重算差值，得到差值的 95% 信賴區間。'],
      ['CI', '95% 信賴區間；下界大於 0 才算確定有增益。'],
      // 11-11 加（第 7 章的名詞）
      ['對手錨點', '認出這一局的對手是訓練集裡的哪位玩家，把他的標籤當成本人等級的一次觀測。'],
      ['條件化比對', '學出來的比對：把一局棋和每位候選人自己的棋各變成一個向量，看哪位候選人最接近。'],
      ['邊界區', '模型最看好的兩個等級相鄰、合計機率至少 0.5、兩者相差不到 3 倍的題；對手錨點只在這些題上改決策。'],
      ['冒充者', '對照組：本人明明不在比對庫裡，卻被配到的那位最像的人；用它的比對分數分布，判斷一個比對有多可能是真的。'],
      ['決策推移', '在兩個相鄰等級之間，把其中一級的勝算乘上一個常數（這裡是 2），只改因此翻動的題。'],
      ['對比損失', '訓練時多加的一項：同一位玩家的兩組棋局要得到相近的表示，不同玩家的要分開。'],
      ['Top-1', '排第一的候選人；Top-5 指前五名。'],
      ['一對一配對', '每位候選人只能配給一題；在這個限制下找總分最大的配法。'],
      ['匈牙利法', '求最佳一對一指派的經典演算法。'],
      ['LDA', '線性判別分析：每位候選人一個平均特徵、所有人共用一個變異結構，看題目最像誰。'],
      ['back-off', '先用完整的前文預測下一手；資料不夠時退回較短的前文，再不夠就用全庫平均。'],
      ['n-gram', '用前面幾手預測下一手的計數模型。'],
      ['Dirichlet', '一種平滑：資料少的候選人，機率往全庫的平均靠攏。'],
      ['cosine', '兩個向量夾角的餘弦，越接近 1 表示方向越像。'],
      ['SGF', '圍棋棋譜的標準文字格式。'],
      ['fusion', '玩家辨識最早的融合：開局＋風格兩個成分。'],
      ['fusion3', '開局＋風格＋局部棋形三個成分的融合。'],
      ['fusion3o', 'fusion3 再加原始方向（開在哪個角）。'],
      ['fusion5_raw4', 'fusion3 加前 4 手方向與讓子旗標的版本（沒有進最終檔）。'],
      ['fusion3o_ref', '玩家辨識的最終檔：fusion3o 再加參考玩家相似度。'],
    ];
    const LATIN = /^[A-Za-z0-9]/, BEFORE = /[A-Za-z0-9_./\-]/, AFTER = /[A-Za-z0-9_/]/;
    const edgeOK = (txt, i, t) => !LATIN.test(t) || (!BEFORE.test(txt[i - 1] || ' ') && !AFTER.test(txt[i + t.length] || ' '));
    const SKIP = 'script,style,svg,code,button,textarea,input,.src,.chips,.num,.an-steps,.an-note,.an-tech,.lb-head,h1,h2,.kpi .v,.term';
    const mark = root => {   // root：一頁或一個推導節；每個術語只標第一次出現
      const used = new Set(), nodes = [];
      const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {acceptNode: n => (n.nodeValue.trim() && n.parentElement && !n.parentElement.closest(SKIP)) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT});
      for (let n; (n = tw.nextNode());) nodes.push(n);
      nodes.forEach(node => {
        let cur = node;
        while (cur) {
          const txt = cur.nodeValue;
          let best = null;
          TERMS.forEach(([t, d]) => {
            if (used.has(t)) return;
            let i = txt.indexOf(t);
            while (i >= 0 && !edgeOK(txt, i, t)) i = txt.indexOf(t, i + 1);
            if (i >= 0 && (!best || i < best[0] || (i === best[0] && t.length > best[1].length))) best = [i, t, d];
          });
          if (!best) break;
          const [i, t, d] = best;
          used.add(t);
          const mid = cur.splitText(i), rest = mid.splitText(t.length);
          const span = document.createElement('span');
          span.className = 'term'; span.tabIndex = 0; span.dataset.def = d; span.textContent = t;
          mid.parentNode.replaceChild(span, mid);
          cur = rest;
        }
      });
    };
    // check.py 的 node 假 DOM 沒有這些 API：在那裡整段略過
    if (typeof document.createTreeWalker === 'function' && typeof document.addEventListener === 'function') {
      try {
        document.querySelectorAll('main > .page, main > .lesson').forEach(mark);
        const tip = document.createElement('div');
        tip.className = 'term-tip'; tip.id = 'term-tip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true;
        let on = null;
        const show = el => {
          on = el;
          const host = el.closest('dialog') || document.body;   // 推導彈窗是 modal：說明框要放進彈窗裡才看得到
          if (tip.parentNode !== host) host.appendChild(tip);
          tip.textContent = el.dataset.def; tip.hidden = false; el.setAttribute('aria-describedby', 'term-tip');
          const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, vw = document.documentElement.clientWidth;
          let y = r.bottom + 6;
          if (y + h > window.innerHeight - 8) y = r.top - h - 6;
          tip.style.left = Math.max(8, Math.min(r.left, vw - w - 8)) + 'px';
          tip.style.top = Math.max(8, y) + 'px';
        };
        const hide = () => { if (on) on.removeAttribute('aria-describedby'); on = null; tip.hidden = true; };
        const termOf = e => e.target && e.target.closest ? e.target.closest('.term') : null;
        document.addEventListener('pointerover', e => { const t = termOf(e); if (t && e.pointerType !== 'touch') show(t); });
        document.addEventListener('pointerout', e => { const t = termOf(e); if (t && e.pointerType !== 'touch' && !t.contains(e.relatedTarget)) hide(); });
        document.addEventListener('focusin', e => { const t = termOf(e); if (t) show(t); });
        document.addEventListener('focusout', e => { if (termOf(e)) hide(); });
        document.addEventListener('click', e => { const t = termOf(e); if (t) show(t); else if (!tip.hidden) hide(); });   // 點術語＝顯示（點一下聚焦也會顯示，不做開關以免手機上一閃即逝）；點別處收起
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && !tip.hidden) hide(); });
        window.addEventListener('scroll', hide, {passive: true});
      } catch (err) { console.warn('名詞註記載入失敗：' + err.message); }
    }
  }
