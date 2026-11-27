"""Line B8: a player embedding learnt on per-game trunk rows (METHOD_NOTES 2.23).

The trunk component (aicup_pid.trunkstyle) is a linear discriminant on the 256-dimensional per-game rows. Here a small
network maps every row to a unit vector so that the mean vector of a few games of one player is close to the mean vector
of his other games and far from other players' (the identification task itself, trained episode by episode):
  episode: P players; per player a query set of n games (n drawn from `query_sizes`) and a disjoint support set of up
  to `support` other games; set vector = L2-normalised mean of the game vectors; logits = scale * cos(query_i,
  support_j); loss = cross-entropy both ways (query -> support and support -> query).
Augmentation: with probability `p_aug` a game's row is replaced by its row under a random board symmetry when one is
given (`aug_rows`, from pid_trunk_extract.py --sym-seed), so the embedding learns to ignore orientation.
Scoring: score_embed gives the [Q, C] matrix scale * cos(question set vector, candidate set vector), the same shape and
row order as every other component. Players enter training only with >= min_games rows; fold 0 never enters (callers).
"""
import numpy as np  # 陣列運算
import torch  # 網路訓練
from torch import nn  # 網路層

DIM = 256  # 每局主幹列的預設維度


class EmbedNet(nn.Module):  # 把每局列映成單位向量的小網路
    def __init__(self, d_in=DIM, hidden=512, d_out=128, dropout=0.1):  # 建立網路層
        super().__init__()  # 初始化 nn.Module
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),  # 第一個隱藏層
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),  # 第二個隱藏層
                                 nn.Linear(hidden, d_out))  # 輸出層
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))  # 可學習的餘弦縮放（取對數，初值 10）

    def forward(self, x):  # 前向傳遞
        return nn.functional.normalize(self.net(x), dim=-1)  # 輸出 L2 正規化的向量


class PlayerEmbedding:  # 玩家嵌入：訓練、推論與存讀檔
    """fit({player: [G, 256] rows}, aug={player: [G, 256] rows aligned or None}); embed(rows) -> [G, d]; set_vector(rows)."""

    def __init__(self, hidden=512, d_out=128, dropout=0.1, steps=3000, players_per_episode=256, support=40, query_sizes=(1, 2, 3, 5, 10, 15, 20),  # 網路與情節的超參數
                 lr=1e-3, weight_decay=1e-4, p_aug=0.5, min_games=8, seed=0, device=None):  # 最佳化、擴增、每位玩家的最少局數、種子與裝置
        self.cfg = dict(hidden=hidden, d_out=d_out, dropout=dropout, steps=steps, players_per_episode=players_per_episode, support=support,  # 保存設定（存檔與重建用）
                        query_sizes=tuple(query_sizes), lr=lr, weight_decay=weight_decay, p_aug=p_aug, min_games=min_games, seed=seed)  # 其餘設定
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 就用 GPU
        self.net = None  # 網路在 fit 或 from_state 時建立
        self.mu = self.sd = None  # 輸入標準化用的平均與標準差
        self.history = []  # 訓練紀錄

    def _x(self, rows):  # 標準化輸入並轉成張量
        return torch.as_tensor((np.asarray(rows, np.float32) - self.mu) / self.sd, device=self.device)  # 以訓練列的平均與標準差標準化

    def fit(self, rows_by_player, aug_by_player=None, log=None):  # 逐情節訓練嵌入網路
        c = self.cfg  # 設定的簡寫
        rng = np.random.RandomState(c["seed"])  # 抽樣用的亂數
        torch.manual_seed(c["seed"])  # 固定 PyTorch 的種子
        players = sorted(p for p, r in rows_by_player.items() if len(r) >= c["min_games"])  # 棋局數足夠的玩家
        if len(players) < 2:  # 至少要兩位玩家才能對比
            raise ValueError("need at least two players with min_games rows")  # 無法訓練
        allrows = np.concatenate([rows_by_player[p] for p in players]).astype(np.float32)  # 所有訓練列
        self.mu, self.sd = allrows.mean(axis=0), allrows.std(axis=0) + 1e-6  # 標準化參數
        X = {p: self._x(rows_by_player[p]) for p in players}  # 每位玩家標準化後的列
        A = {p: self._x(aug_by_player[p]) for p in players if aug_by_player and aug_by_player.get(p) is not None} if aug_by_player else {}  # 對稱擴增的列（有提供才有）
        self.d_in = int(allrows.shape[1])  # 256 維主幹列，或含各階段平均的 640 維（METHOD_NOTES 2.33）
        self.net = EmbedNet(self.d_in, c["hidden"], c["d_out"], c["dropout"]).to(self.device)  # 建立網路
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])  # AdamW 最佳化器
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["steps"], pct_start=0.1)  # 單週期學習率排程
        P = min(c["players_per_episode"], len(players))  # 每個情節的玩家數
        self.net.train()  # 切換到訓練模式
        for step in range(c["steps"]):  # 每一步一個情節
            batch = rng.choice(len(players), P, replace=False)  # 抽出本情節的玩家
            q_vecs, s_vecs = [], []  # 未使用的暫存清單
            rows, owners, sides = [], [], []  # 本情節的列、所屬玩家、屬於查詢或支援
            for b, i in enumerate(batch):  # 逐位玩家組出查詢集與支援集
                p = players[i]  # 玩家 id
                G = len(X[p])  # 該玩家的局數
                n = int(min(rng.choice(c["query_sizes"]), max(1, G // 2)))  # 查詢集大小，最多一半的局數
                perm = rng.permutation(G)  # 打散棋局
                q_idx, s_idx = perm[:n], perm[n:n + c["support"]]  # 查詢與支援的棋局（不相交）
                for idx, side in ((q_idx, 0), (s_idx, 1)):  # 0 = 查詢、1 = 支援
                    xs = X[p][torch.as_tensor(idx, device=self.device)]  # 取出這些棋局的列
                    if p in A and c["p_aug"] > 0:  # 有對稱擴增的列時
                        m = torch.as_tensor(rng.rand(len(idx)) < c["p_aug"], device=self.device)  # 以機率 p_aug 決定要替換的列
                        xs = torch.where(m[:, None], A[p][torch.as_tensor(idx, device=self.device)], xs)  # 換成對稱後的列
                    rows.append(xs)  # 收集列
                    owners.append(torch.full((len(idx),), b, device=self.device))  # 記下所屬玩家（情節內編號）
                    sides.append(torch.full((len(idx),), side, device=self.device))  # 記下屬於查詢或支援
            x, own, side = torch.cat(rows), torch.cat(owners), torch.cat(sides)  # 合併成一個批次
            z = self.net(x)  # 每局的單位向量
            seg = own * 2 + side  # 每位玩家 2 個集合
            sums = torch.zeros(2 * P, z.shape[1], device=self.device).index_add_(0, seg, z)  # 各集合的向量和
            sets = nn.functional.normalize(sums, dim=-1)  # 集合向量（正規化後的平均）
            q, s = sets[0::2], sets[1::2]  # 查詢集與支援集的向量
            logits = self.net.log_scale.exp() * q @ s.T  # scale 乘以餘弦相似度
            target = torch.arange(P, device=self.device)  # 正解：同一位玩家
            loss = 0.5 * (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target))  # 雙向交叉熵
            opt.zero_grad()  # 清除梯度
            loss.backward()  # 反向傳遞
            opt.step()  # 更新權重
            sched.step()  # 更新學習率
            if step % 250 == 0 or step == c["steps"] - 1:  # 每 250 步與最後一步記錄
                acc = float((logits.argmax(dim=1) == target).float().mean())  # 情節內的 top-1 準確率
                rec = {"step": step, "loss": float(loss.detach()), "acc": acc, "scale": float(self.net.log_scale.detach().exp())}  # 紀錄內容
                self.history.append(rec)  # 加入訓練紀錄
                if log:  # 有給記錄函式時
                    log(f"  step {step}: loss {rec['loss']:.4f}, episode top-1 {acc:.3f} of {P}, scale {rec['scale']:.2f}", flush=True)  # 輸出進度
        self.net.eval()  # 訓練完切回評估模式
        self.players_trained = len(players)  # 記下訓練玩家數
        return self  # 回傳自身以便串接

    @torch.no_grad()  # 推論時不計算梯度
    def embed(self, rows, batch=8192):  # 把多列映成向量
        out = []  # 收集每批的輸出
        rows = np.asarray(rows, np.float32)  # 統一成 float32，與網路權重的精度一致
        for i in range(0, len(rows), batch):  # 分批處理以控制記憶體
            out.append(self.net(self._x(rows[i:i + batch])).cpu().numpy())  # 標準化後通過網路
        return np.concatenate(out) if out else np.zeros((0, self.cfg["d_out"]), np.float32)  # 合併（沒有列時回傳空陣列）

    def set_vector(self, rows):  # 一組棋局的集合向量
        v = self.embed(rows).sum(axis=0)  # 各局向量相加
        return v / (np.linalg.norm(v) + 1e-12)  # 正規化成單位向量

    @property  # 以屬性方式讀取
    def scale(self):  # 目前的餘弦縮放
        return float(self.net.log_scale.detach().exp())  # 由對數值還原

    def state_dict(self):  # 存檔用的狀態
        return {"cfg": self.cfg, "mu": self.mu, "sd": self.sd, "net": {k: v.cpu() for k, v in self.net.state_dict().items()}, "history": self.history,  # 設定、標準化參數、權重與紀錄
                "players_trained": self.players_trained, "d_in": getattr(self, "d_in", DIM)}  # 訓練玩家數與輸入維度

    @classmethod  # 類別方法：由存檔重建
    def from_state(cls, st, device=None):  # 從 state_dict 載入模型
        m = cls(**st["cfg"], device=device)  # 以存檔的設定建立物件
        m.mu, m.sd, m.history, m.players_trained = st["mu"], st["sd"], st["history"], st["players_trained"]  # 還原標準化參數與紀錄
        m.d_in = int(st.get("d_in", DIM))  # 舊檔沒有 d_in 時視為 256
        m.net = EmbedNet(m.d_in, st["cfg"]["hidden"], st["cfg"]["d_out"], st["cfg"]["dropout"]).to(m.device)  # 建立相同結構的網路
        m.net.load_state_dict(st["net"])  # 載入權重
        m.net.eval()  # 切換到評估模式
        return m  # 回傳模型


def score_embed(model, cand_keys_by_player, question_keys, store):  # 嵌入成分的分數矩陣
    """[Q, C] scale * cosine of the question and candidate set vectors (candidates in sorted id order)."""
    players = sorted(cand_keys_by_player)  # 候選人依 id 排序
    C = np.stack([model.set_vector(np.stack([store[k] for k in cand_keys_by_player[p]])) for p in players])  # 每位候選人的集合向量
    if not question_keys:  # 沒有題目
        return np.zeros((0, len(players)))  # 回傳空矩陣
    Q = np.stack([model.set_vector(np.stack([store[k] for k in keys])) for keys in question_keys])  # 每題的集合向量
    return model.scale * (Q @ C.T)  # scale 乘以餘弦


def adapted(model, rows_by_player, steps=100, lr=1e-4, players_per_episode=256, query_sizes=(1, 2, 3, 5), seed=0):  # 以題池的候選人微調嵌入
    """A copy of `model` fine-tuned on one pool's own candidates (METHOD_NOTES 2.33): the same two-way set loss as fit, every
    episode drawing a query set and the remaining games as support from each sampled candidate. No question enters."""
    import copy  # 深層複製網路用
    m = PlayerEmbedding(**model.cfg, device=model.device)  # 以相同設定建立新物件
    m.mu, m.sd, m.history, m.players_trained = model.mu, model.sd, list(model.history), model.players_trained  # 沿用標準化參數與紀錄
    m.net = copy.deepcopy(model.net)  # 複製網路，不動到原模型
    m.d_in = getattr(model, "d_in", DIM)  # 輸入維度
    rng = np.random.RandomState(seed)  # 抽樣用的亂數
    torch.manual_seed(seed)  # 固定 PyTorch 的種子
    players = sorted(p for p, r in rows_by_player.items() if len(r) >= 2)  # 至少 2 局才能分出查詢與支援
    X = {p: m._x(rows_by_player[p]) for p in players}  # 標準化後的列
    P = min(players_per_episode, len(players))  # 每個情節的玩家數
    opt = torch.optim.AdamW(m.net.parameters(), lr=lr, weight_decay=model.cfg["weight_decay"])  # 微調用的最佳化器
    m.net.train()  # 切換到訓練模式
    for _ in range(steps):  # 每一步一個情節
        rows, seg = [], []  # 本情節的列與集合編號
        for b, i in enumerate(rng.choice(len(players), P, replace=False)):  # 逐位抽到的候選人
            G = len(X[players[i]])  # 該候選人的局數
            n = int(min(rng.choice(query_sizes), G - 1))  # 查詢集大小，至少留 1 局當支援
            perm = rng.permutation(G)  # 打散棋局
            rows.append(X[players[i]][torch.as_tensor(perm, device=m.device)])  # 依打散後的順序取全部棋局
            seg.append(torch.as_tensor([2 * b] * n + [2 * b + 1] * (G - n), device=m.device))  # 前 n 局為查詢，其餘為支援
        z = m.net(torch.cat(rows))  # 每局的單位向量
        sets = nn.functional.normalize(torch.zeros(2 * P, z.shape[1], device=m.device).index_add_(0, torch.cat(seg), z), dim=-1)  # 各集合正規化後的向量
        logits = m.net.log_scale.exp() * sets[0::2] @ sets[1::2].T  # 查詢對支援：scale 乘以餘弦
        target = torch.arange(P, device=m.device)  # 正解：同一位候選人
        loss = 0.5 * (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target))  # 雙向交叉熵
        opt.zero_grad()  # 清除梯度
        loss.backward()  # 反向傳遞
        opt.step()  # 更新權重
    m.net.eval()  # 切回評估模式
    return m  # 回傳微調後的副本
