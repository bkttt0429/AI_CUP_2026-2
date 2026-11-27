"""Move-level matching (METHOD_NOTES 2.56, atom 3): every own move of a player is a token, the stored input of that move
(board-pooled + played-point trunk vectors of the frozen network, 128 numbers: the situation and the move, no identity
training). A question game is scored against a candidate by how its tokens are matched among the tokens of the candidate's
games. Network-independent: computed once per pool from the per-ply parts."""
import numpy as np  # 陣列運算
import torch  # 在 GPU 上計算相似度

MAXTOK = 128  # 每局保留的己方手數（逐手分段檔存有 256 手）


def game_tokens(GL, MOV, LEN, SUBJ):  # 取出主體自己的每一手作為符記
    """(tokens [n_tok, 128] float32 un-normalised, game index [n_tok], ply [n_tok]) of the subject's own moves."""
    GL, MOV, LEN, SUBJ = (torch.as_tensor(np.asarray(x)) for x in (GL, MOV, LEN, SUBJ))  # 輸入可為 numpy 陣列，統一轉成張量
    T = GL.shape[1]  # 每局儲存的手數上限
    own = (MOV.long() == SUBJ.long()[:, None]) & (torch.arange(T)[None] < LEN.long()[:, None])  # 主體方下的、且在棋局長度內的手
    b, t = own.nonzero(as_tuple=True)  # 這些手的棋局索引與手數
    keep = own.cumsum(1)[b, t] <= MAXTOK  # 每局只留前 MAXTOK 手
    b, t = b[keep], t[keep]  # 套用上限
    return GL[b, t].float(), b, t  # 回傳符記、棋局索引與手數


@torch.no_grad()  # 只做推論，不計算梯度
def scores(c_tok, c_owner, c_ply, C, q_tok, q_game, q_ply, n_q, ks=(20, 5), split=40, chunk=1024, device="cuda"):  # 每局題目棋對每位候選人的比對分數
    """{name: [n_q games, C]} per question game, each row the mean over the game's tokens:
    vote<k>       every token votes for the owners of its k nearest candidate tokens (cosine, weights = cosine clipped at 0)
    vote<k0>_open / _mid   the first k, only the question tokens before / from ply `split`
    maxsim        the best cosine among the candidate's tokens
    Tokens are standardised with the candidate tokens' mean and deviation, then normalised."""
    mu, sd = c_tok.mean(0), c_tok.std(0) + 1e-6  # 候選符記的平均與標準差
    ct = torch.nn.functional.normalize((c_tok - mu) / sd, dim=-1).half().to(device)  # 標準化並正規化後的候選符記（半精度）
    qt_all = torch.nn.functional.normalize((q_tok - mu) / sd, dim=-1).half()  # 題目符記用同一組參數處理
    own = torch.as_tensor(np.asarray(c_owner)).long().to(device)  # 每個候選符記的擁有者
    cnt = torch.bincount(own, minlength=C)  # 每位候選人的符記數
    L = int(cnt.max())  # 單一候選人最多的符記數
    order = torch.argsort(own, stable=True)  # 依擁有者排序符記
    slot = torch.arange(len(order), device=device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)  # 每個符記在其擁有者內的序號
    grid_at = torch.full((C * L,), -1, dtype=torch.long, device=device)  # [C, L] 網格，空格為 -1
    grid_at[own[order] * L + slot] = order  # 填補後的 [C, L] 網格中，每一格放的是哪個候選符記
    k0 = ks[0]  # 分前後段統計時使用的 k
    out = {f"vote{k}": torch.zeros(n_q, C, device=device) for k in ks}  # 各 k 的投票累計
    out.update({f"vote{k0}_open": torch.zeros(n_q, C, device=device), f"vote{k0}_mid": torch.zeros(n_q, C, device=device), "maxsim": torch.zeros(n_q, C, device=device)})  # 前段、後段投票與最大相似度
    q_game_d, q_ply_d = torch.as_tensor(np.asarray(q_game)).long().to(device), torch.as_tensor(np.asarray(q_ply)).long().to(device)  # 題目符記所屬的棋局與手數
    for i in range(0, len(qt_all), chunk):  # 題目符記分塊處理以控制記憶體
        qt, qg, qp = qt_all[i:i + chunk].to(device), q_game_d[i:i + chunk], q_ply_d[i:i + chunk]  # 這一塊的符記、棋局與手數
        S = (qt @ ct.T).float()  # [chunk, 候選符記數]：題目符記對候選符記的餘弦
        top = S.topk(max(ks), dim=1)  # 每個題目符記最近的候選符記
        for k in ks:  # 每一種 k
            v = torch.zeros(len(qt), C, device=device).scatter_add_(1, own[top.indices[:, :k]], top.values[:, :k].clamp(min=0))  # 投票給前 k 個近鄰的擁有者，票重為截到 0 以上的餘弦
            out[f"vote{k}"].index_add_(0, qg, v)  # 累加到所屬的題目棋局
            if k == k0:  # 第一個 k 另外分前後段
                out[f"vote{k0}_open"].index_add_(0, qg, v * (qp < split)[:, None])  # 第 split 手之前
                out[f"vote{k0}_mid"].index_add_(0, qg, v * (qp >= split)[:, None])  # 第 split 手起
        G = torch.cat([S, torch.full((len(qt), 1), -1.0, device=device)], 1)[:, grid_at]  # 空格讀到附加的 -1 欄
        out["maxsim"].index_add_(0, qg, G.view(len(qt), C, L).max(-1).values)  # 每位候選人的最大餘弦
    ntok = torch.bincount(q_game_d, minlength=n_q).clamp(min=1)[:, None]  # 每局題目棋的符記數
    return {k: (x / ntok).cpu().numpy() for k, x in out.items()}  # 對各局的符記取平均


def component(arrays, cands, questions, players=None, **kw):  # 一個題池的逐手比對成分
    """{name: [Q, C]} for one pool. arrays: {key, GL, MOV, LEN, SUBJ} of the pool's games; cands {player: [(key, colour)]};
    questions [[(key, colour)]]. A question's row is the mean of its games' rows."""
    players = players or sorted(cands)  # 候選人預設依 id 排序
    at = {k: i for i, k in enumerate(np.asarray(arrays["key"]).astype(str))}  # 棋局鍵 -> 陣列列號
    c_rows = np.array([at[k] for p in players for k, _ in cands[p]])  # 候選棋局的列號
    c_own_game = np.array([i for i, p in enumerate(players) for _ in cands[p]])  # 每局候選棋的擁有者
    q_rows = np.array([at[k] for q in questions for k, _ in q])  # 題目棋局的列號
    q_of = np.array([j for j, q in enumerate(questions) for _ in q])  # 每局題目棋所屬的題目
    pick = lambda rows: (arrays["GL"][rows], arrays["MOV"][rows], arrays["LEN"][rows], arrays["SUBJ"][rows])  # noqa: E731  取出指定列的四個陣列
    c_tok, c_game, c_ply = game_tokens(*pick(c_rows))  # 候選人的符記
    q_tok, q_game, q_ply = game_tokens(*pick(q_rows))  # 題目的符記
    per_game = scores(c_tok, c_own_game[c_game.numpy()], c_ply, len(players), q_tok, q_game, q_ply, len(q_rows), **kw)  # 每局題目棋的分數
    n = np.bincount(q_of, minlength=len(questions))[:, None]  # 每題的局數
    out = {}  # 各分數的 [Q, C]
    for name, M in per_game.items():  # 每一種比對分數
        acc = np.zeros((len(questions), len(players)), np.float32)  # 每題的累計
        np.add.at(acc, q_of, M)  # 依所屬題目累加
        out[name] = acc / n  # 取各局的平均
    return out  # 回傳結果
