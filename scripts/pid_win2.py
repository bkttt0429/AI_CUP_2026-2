"""METHOD_NOTES 2.192: two variants of the window input of pid_win.py (probe W2), for the short-run filter.

pid_win.py shows the network, per ply, the size x size neighbourhood of the played point with six cell states (empty, own
stone, other stone, off the board, the previous ply's stone, the ply before that).  The merged retrain (notes 2.190 / 2.191)
found this input worth more than everything else tried since the by-games sampling, so two cheap extensions of the SAME
information source (the stones played so far; still no feature extraction, no file):

  age    every stone in the window also carries HOW LONG AGO it was played, in eight buckets (1, 2, 3-4, 5-8, 9-16, 17-32,
         33-64, more plies ago) - whether a move joins a fight that is going on or returns to an old position
  prev   a second window of the same size around the PREVIOUS ply's point (the opponent's last move), read by its own
         encoder - what the mover was answering, or left unanswered when the move is elsewhere on the board

cfg: {"size": odd, "ch": channels, "age": 0|1, "prev": 0|1}.  pid_win.wrap() sends a cfg with "age" or "prev" here, so the
loaders (fine-tune, embedding) need no change.  Every new part's contribution starts at zero: with age = prev = 0 the module
is pid_win's, and at the start of training a wrapped network equals the plain one.
"""


def windows2(pos, mov, size, age=False, prev=False):  # 組出視窗、年齡與第二個視窗
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours ->
    cells [B, T, size * size] (the six states of pid_win.windows), ok [B, T];
    ages [B, T, size * size] or None (0 no stone / off the board, 1..8 the age bucket of the stone);
    pcells, pages (the same around the previous ply's point, states still relative to the mover) and pok [B, T], or None."""
    import torch  # 延後匯入，模組本身不依賴 torch
    import torch.nn.functional as F  # 補邊用的 pad
    B, T = pos.shape  # 批次大小與手數
    dev = pos.device  # 在輸入所在的裝置上計算
    ok = pos < 361  # 有落點的手（不是 pass／補齊）
    t = torch.arange(1, T + 1, device=dev, dtype=torch.int16)[None].expand(B, T)  # 每手的手數（1 起算）
    stamp = torch.zeros(B, T, 363, dtype=torch.int16, device=dev)  # 棋子落下處記為（手數）* 2 + 顏色
    stamp.scatter_(2, pos[..., None], torch.where(ok, t * 2 + mov.to(torch.int16), torch.zeros_like(t))[..., None])  # 每手只在自己的落點蓋章
    last = torch.cummax(stamp[..., :361], dim=1).values  # 每一手之後各點上最新的棋子
    before = torch.zeros_like(last)  # 落子「之前」的盤面
    before[:, 1:] = last[:, :-1]  # 往後錯開一手
    del stamp, last  # 釋放大張量
    ply = (before >> 1).long()  # 0 = 空，否則是該子的手數（1 起算）
    col = (before & 1).long()  # 該子的顏色
    now = torch.arange(1, T + 1, device=dev)[None, :, None]  # 目前這一手的手數（1 起算）
    state = torch.where(ply == 0, 0, torch.where(col == mov[..., None], 1, 2))  # 空、己方或對方的子
    state = torch.where((ply > 0) & (ply == now - 1), 4, state)  # 上一手的子
    state = torch.where((ply > 0) & (ply == now - 2), 5, state)  # 再上一手（自己的上一手）的子
    del col, before  # 年齡還要用 ply，先不釋放
    p = size // 2  # 視窗半徑
    W = 19 + 2 * p  # 補邊後的盤面寬度
    d = torch.arange(size, device=dev)  # 視窗內的位移
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # 視窗各格相對其角落 (y - p, x - p) 的位移（補邊座標）
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)  # 四周補上「盤外」= 3
    del state  # 釋放記憶體
    aboard = None  # 年齡盤面（沒開 age 時為 None）
    if age:  # 要年齡分段
        bounds = torch.tensor([1, 2, 4, 8, 16, 32, 64], device=dev)  # 八段的分界
        bucket = torch.where(ply > 0, torch.bucketize(now - ply, bounds) + 1, torch.zeros_like(ply)).to(torch.uint8)  # 有子處為 1..8，空點為 0
        aboard = F.pad(bucket.reshape(B, T, 19, 19), (p, p, p, p), value=0).reshape(B, T, W * W)  # 盤外的年齡為 0
        del bucket  # 釋放記憶體
    del ply  # 釋放記憶體

    def cut(cen):  # 以 cen 為中心切出視窗
        idx = ((cen // 19) * W + cen % 19)[..., None] + off[None, None]  # 每格在補邊盤面中的索引
        return board.gather(2, idx), (aboard.gather(2, idx).long() if aboard is not None else None)  # 格子狀態與年齡

    cells, ages = cut(torch.where(ok, pos, torch.zeros_like(pos)))  # 自己落點周圍的視窗
    pcells = pages = pok = None  # 沒開 prev 時都是 None
    if prev:  # 第二個視窗
        ppos = torch.full_like(pos, 362)  # 上一手的落點；第一手沒有上一手
        ppos[:, 1:] = pos[:, :-1]  # 往後錯開一手
        pok = ok & (ppos < 361)  # 這手與上一手都有落點
        pcells, pages = cut(torch.where(pok, ppos, torch.zeros_like(ppos)))  # 上一手落點周圍的視窗
    return cells, ok, ages, pcells, pages, pok  # 六個輸出


def wrap(net, cfg):  # 為網路加上視窗編碼器（W2 變體）
    """Adds the window encoder(s) to a built m88_seqid.SeqID (in place) and returns it."""
    import torch  # 延後匯入
    from torch import nn  # 神經網路模組
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))  # 視窗大小與通道數
    age, prev = bool(cfg.get("age")), bool(cfg.get("prev"))  # 兩個變體的開關
    assert size % 2 == 1 and 3 <= size <= 19 and (age or prev)  # 至少要開一個變體（否則應由 pid_win 處理）
    D = net.gl.out_features  # 序列網路的寬度
    base = net.__class__  # 原本的網路類別

    class Enc(nn.Module):  # pid_win 的編碼器，加上可選的年齡嵌入（從零開始）
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.emb = nn.Embedding(6, 16)  # 六種格子狀態各一個 16 維嵌入
            if age:  # 有年齡
                self.age = nn.Embedding(9, 16)  # 年齡 0..8 的嵌入
                nn.init.zeros_(self.age.weight)  # 從零開始，不影響起點
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),  # 三層卷積，第二層解析度減半
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())  # 第三層卷積
            self.out = nn.Linear(2 * ch, D)  # 中心格與全視窗平均接起來，投影到網路寬度
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 輸出層從零開始，起點等於原網路

        def forward(self, cells, ages, flat, B, T):  # 編碼一種視窗
            x = self.emb(cells.reshape(B * T, -1)[flat])  # 只編碼有效的手
            if ages is not None:  # 有年齡
                x = x + self.age(ages.reshape(B * T, -1)[flat])  # 加上年齡嵌入
            h = self.conv(x.reshape(-1, size, size, 16).permute(0, 3, 1, 2))  # 轉成 [N, 16, size, size] 後卷積
            c = h.shape[-1] // 2  # 中心格的位置
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))  # 中心格＋全視窗平均 → 網路寬度
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)  # 無效的手為零
            res[flat] = f  # 填回有效的手
            return res.reshape(B, T, -1)  # 還原成每局逐手的輸出形狀 [B, T, D]

    class Win2(nn.Module):  # 一或兩個視窗編碼器的組合
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.own = Enc()  # 自己落點周圍的視窗
            if prev:  # 有第二個視窗
                self.prev = Enc()  # 上一手落點周圍的視窗，用它自己的編碼器
            self.size = size  # 記下視窗大小

        def forward(self, pos, mov):  # 每手的視窗向量
            B, T = pos.shape  # 批次大小與手數
            with torch.no_grad():  # 組視窗不需要梯度
                cells, ok, ages, pcells, pages, pok = windows2(pos, mov, size, age, prev)  # 組出所有視窗
            res = self.own(cells, ages, ok.reshape(-1), B, T)  # 自己視窗的向量
            if prev:  # 有第二個視窗
                res = res + self.prev(pcells, pages, pok.reshape(-1), B, T)  # 加上第二個視窗的向量
            return res  # 回傳輸出，形狀 [B, T, D]

    class SeqIDW2(base):  # 帶視窗變體的序列網路
        def forward(self, gl, pos, mov, length, subj):  # 與 SeqID.forward 相同，只多了視窗一項
            B, T = pos.shape  # 批次大小與手數
            x = (self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov)  # 主幹向量、落點、是否主體方、顏色的嵌入相加
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2)))  # 再加手序位置與兩種位移（對上一手、對自己上一手）的嵌入
            x = x + self.win(pos, mov).to(x.dtype)  # 加上視窗向量
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)  # 前面接上 [CLS]
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # 補齊遮罩：位置 0 是 [CLS]，超過局長的位置不參與注意力
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]  # Transformer 編碼後取 [CLS]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)  # 主體方的單位向量

    net.win = Win2().to(net.gl.weight.device)  # 編碼器放到網路所在的裝置
    net.__class__ = SeqIDW2  # 就地換掉類別，forward 多了視窗
    net.win_cfg = {"size": size, "ch": ch, "age": int(age), "prev": int(prev)}  # 在網路物件上記下實際採用的設定，可據此重建同樣的包裝
    return net  # 回傳包裝後的網路


def selftest():  # 自我檢查
    """windows2 against a plain replay (no captures) on random games, and its first two outputs against pid_win.windows."""
    import sys  # 調整搜尋路徑
    from pathlib import Path  # 取本檔所在的資料夾
    import numpy as np  # 單純重播用 numpy
    import torch  # 呼叫 windows2
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓 pid_win 可以匯入
    import pid_win  # 對照用的原視窗
    rng = np.random.default_rng(0)  # 固定種子
    B, T, size = 6, 90, 9  # 6 局、90 手、9 x 9 視窗
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 60:] = 362; pos[2, 7] = pos[2, 3]; pos[3, 10] = 362  # 一手 pass、補齊、同一點下兩次、一手隱藏
    mov = np.tile(np.arange(T) % 2, (B, 1))  # 黑白交替
    tp, tm = torch.as_tensor(pos), torch.as_tensor(mov)  # 轉成張量
    cells, ok, ages, pcells, pages, pok = [x.numpy() for x in windows2(tp, tm, size, True, True)]  # 兩個變體都開
    c0, ok0 = [x.numpy() for x in pid_win.windows(tp, tm, size)]  # 原視窗的結果
    same = int((cells != c0).sum() + (ok != ok0).sum())  # 與原視窗不同的數量
    c1, ok1, a1, p1, q1, r1 = windows2(tp, tm, size, False, False)  # 兩個變體都關
    assert a1 is None and p1 is None and q1 is None and r1 is None and int((c1.numpy() != c0).sum()) == 0  # 都關時必須等於原視窗
    p = size // 2  # 視窗半徑

    def bucket(d):  # 年齡分段的參考實作
        return 1 + int(np.searchsorted([1, 2, 4, 8, 16, 32, 64], d, side="left"))  # 與 torch.bucketize 相同的分段

    bad = {"cells": 0, "ages": 0, "pcells": 0, "pages": 0, "pok": 0}  # 各項不一致的數量
    n = 0  # 比對過的格數
    for b in range(B):  # 每一局
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)  # 各點的顏色＋1 與落子手數
        for t in range(T):  # 每一手
            has = pos[b, t] < 361  # 這手有落點
            hasp = has and t > 0 and pos[b, t - 1] < 361  # 這手與上一手都有落點
            bad["pok"] += int(pok[b, t] != hasp)  # 第二個視窗的遮罩
            for which, on, cy_cx in (("", has, pos[b, t]), ("p", hasp, pos[b, t - 1] if t > 0 else 0)):  # 自己的視窗與第二個視窗
                if not on:  # 這個視窗無效
                    continue  # 略過
                y, x = divmod(int(cy_cx), 19)  # 視窗中心
                got_c, got_a = (cells, ages) if which == "" else (pcells, pages)  # 取對應的輸出
                for i in range(size):  # 視窗的列
                    for j in range(size):  # 視窗的行
                        yy, xx = y - p + i, x - p + j  # 這格在盤上的座標
                        if not (0 <= yy < 19 and 0 <= xx < 19):  # 盤外
                            want, wa = 3, 0  # 狀態 3、年齡 0
                        elif board[yy, xx] == 0:  # 空點
                            want, wa = 0, 0  # 狀態 0、年齡 0
                        else:  # 有子
                            dd = (t + 1) - when[yy, xx]  # 這顆子是幾手前下的
                            want = 4 if dd == 1 else 5 if dd == 2 else (1 if board[yy, xx] - 1 == mov[b, t] else 2)  # 期望的狀態
                            wa = bucket(dd)  # 期望的年齡分段
                        bad[which + "cells"] += int(got_c[b, t, i * size + j] != want)  # 狀態不一致
                        bad[which + "ages"] += int(got_a[b, t, i * size + j] != wa)  # 年齡不一致
                        n += 1  # 計數
            if has:  # 有落點才落子
                y, x = divmod(int(pos[b, t]), 19)  # 落點座標
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1  # 落子（不提子）
    print(f"windows2 against a plain replay: differing {bad} of {n} cells; against pid_win.windows: {same} differences; "  # 回報與單純重播、原視窗的差異
          f"age buckets seen {sorted(set(ages.reshape(-1).tolist()))}")  # 以及出現過的年齡分段
    return 0 if same == 0 and not any(bad.values()) else 1  # 全部一致時回傳 0


if __name__ == "__main__":  # 直接執行時跑自我檢查
    import sys  # 設定結束碼
    sys.exit(selftest())  # 以檢查結果為結束碼
