"""METHOD_NOTES 2.202: further variants of the window input (probe W3), for the next short runs.

pid_win.py: the size x size neighbourhood of the played point, six cell states relative to the mover. pid_win2.py added a
second window around the PREVIOUS ply's point (the opponent's last move; "prev"), which met the short-run rule and became
the families o40-o43. Three more variants of the same information source (the stones played so far, no file):

  own2    a third window around the point of the ply BEFORE the previous one - the mover's OWN last move: what became of
          it (answered or ignored by the opponent, and whether the mover now continues there or leaves it)
  ch      channels of every window encoder (64 in all families so far)
  depth   further residual blocks (two 3x3 convolutions each, the second one starting at zero) after the three
          convolutions of every window encoder

cfg: {"size": odd, "ch": channels, "prev": 0|1, "own2": 0|1, "depth": n}. The stone ages of pid_win2 are not offered here
(closed, notes 2.193). The encoders' outputs start at zero, so a wrapped network equals the plain one at the start.

The shared modules are NOT changed for the short runs: the trainer scripts/pid_seq_train_w3.py calls wrap() here, and
the loaders are started through experiments/pid_w3_run.py, which makes pid_win.wrap() send a cfg with "own2" or "depth"
here (install()). A family that is expanded later gets the permanent dispatch in pid_win.py after a regression check.
"""


def windows3(pos, mov, size, prev=False, own2=False):  # 組出自己、上一手、自己上一手周圍的視窗
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours ->
    cells [B, T, size * size] (the six states of pid_win.windows), ok [B, T];
    pcells, pok: the same around the previous ply's point (states still relative to the mover), or None;
    ocells, ook: the same around the point of the ply before that (the mover's own last move), or None.
    The board part is pid_win2.windows2's (checked against it by selftest)."""
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
    del col, before, ply  # 釋放記憶體
    p = size // 2  # 視窗半徑
    W = 19 + 2 * p  # 補邊後的盤面寬度
    d = torch.arange(size, device=dev)  # 視窗內的位移
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # 視窗各格相對其角落 (y - p, x - p) 的位移（補邊座標）
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)  # 四周補上「盤外」= 3
    del state  # 釋放記憶體

    def cut(cen):  # 以 cen 為中心切出視窗
        return board.gather(2, ((cen // 19) * W + cen % 19)[..., None] + off[None, None])  # 取出各格的狀態

    def around(back):  # 以 `back` 手之前那一手的落點為中心的視窗
        q = torch.full_like(pos, 362)  # 前幾手沒有對應的手，先當補齊
        q[:, back:] = pos[:, :-back]  # 往後錯開 back 手
        qok = ok & (q < 361)  # 這手與那一手都有落點
        return cut(torch.where(qok, q, torch.zeros_like(q))), qok  # 視窗與遮罩

    cells = cut(torch.where(ok, pos, torch.zeros_like(pos)))  # 自己落點周圍的視窗
    pcells, pok = around(1) if prev else (None, None)  # 上一手（對手）落點周圍的視窗
    ocells, ook = around(2) if own2 else (None, None)  # 自己上一手落點周圍的視窗
    return cells, ok, pcells, pok, ocells, ook  # 六個輸出


def wrap(net, cfg):  # 為網路加上視窗編碼器（W3 變體）
    """Adds the window encoder(s) to a built m88_seqid.SeqID (in place) and returns it."""
    import torch  # 延後匯入
    from torch import nn  # 神經網路模組
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))  # 視窗大小與通道數
    prev, own2, depth = bool(cfg.get("prev")), bool(cfg.get("own2")), int(cfg.get("depth", 0))  # 變體的開關與殘差塊數
    assert size % 2 == 1 and 3 <= size <= 19 and 0 <= depth <= 8 and not cfg.get("age") and not cfg.get("exact")  # 視窗為 3–19 的奇數、殘差塊 0–8 個；這裡不支援棋子年齡與精確盤面
    D = net.gl.out_features  # 序列網路的寬度
    base = net.__class__  # 原本的網路類別

    class Enc(nn.Module):  # pid_win 的編碼器（相同的參數名稱），可加殘差塊
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.emb = nn.Embedding(6, 16)  # 六種格子狀態各一個 16 維嵌入
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),  # 三層卷積，第二層解析度減半
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())  # 第三層卷積
            self.more = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(depth)])  # depth 個殘差塊，每塊兩層 3x3 卷積
            for m in self.more:  # 每個殘差塊
                nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)  # 第二層卷積從零開始，殘差塊起點為恆等
            self.out = nn.Linear(2 * ch, D)  # 中心格與全視窗平均接起來，投影到網路寬度
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 輸出層從零開始，起點等於原網路

        def forward(self, cells, flat, B, T):  # 編碼一種視窗
            x = self.emb(cells.reshape(B * T, -1)[flat])  # 只編碼有效的手
            h = self.conv(x.reshape(-1, size, size, 16).permute(0, 3, 1, 2))  # 轉成 [N, 16, size, size] 後卷積
            for m in self.more:  # 每個殘差塊
                h = h + m(h)  # 殘差相加
            c = h.shape[-1] // 2  # 中心格的位置
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))  # 中心格＋全視窗平均 → 網路寬度
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)  # 無效的手為零
            res[flat] = f  # 填回有效的手
            return res.reshape(B, T, -1)  # 還原成每局逐手的輸出形狀 [B, T, D]

    class Win3(nn.Module):  # 一到三個視窗編碼器的組合
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.own = Enc()  # 自己落點周圍的視窗
            if prev:  # 有第二個視窗
                self.prev = Enc()  # 上一手落點周圍的視窗
            if own2:  # 有第三個視窗
                self.own2 = Enc()  # 自己上一手落點周圍的視窗
            self.size = size  # 記下視窗大小

        def forward(self, pos, mov):  # 每手的視窗向量
            B, T = pos.shape  # 批次大小與手數
            with torch.no_grad():  # 組視窗不需要梯度
                cells, ok, pcells, pok, ocells, ook = windows3(pos, mov, size, prev, own2)  # 組出所有視窗
            res = self.own(cells, ok.reshape(-1), B, T)  # 自己視窗的向量
            if prev:  # 有第二個視窗
                res = res + self.prev(pcells, pok.reshape(-1), B, T)  # 加上第二個視窗的向量
            if own2:  # 有第三個視窗
                res = res + self.own2(ocells, ook.reshape(-1), B, T)  # 加上第三個視窗的向量
            return res  # 回傳輸出，形狀 [B, T, D]

    class SeqIDW3(base):  # 帶 W3 視窗的序列網路
        def forward(self, gl, pos, mov, length, subj):  # 與 SeqID.forward 相同，只多了視窗一項
            B, T = pos.shape  # 批次大小與手數
            x = (self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov)  # 主幹向量、落點、是否主體方、顏色的嵌入相加
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2)))  # 再加手序位置與兩種位移（對上一手、對自己上一手）的嵌入
            x = x + self.win(pos, mov).to(x.dtype)  # 加上視窗向量
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)  # 前面接上 [CLS]
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # 補齊遮罩：位置 0 是 [CLS]，超過局長的位置不參與注意力
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]  # Transformer 編碼後取 [CLS]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)  # 主體方的單位向量

    net.win = Win3().to(net.gl.weight.device)  # 編碼器放到網路所在的裝置
    net.__class__ = SeqIDW3  # 就地換掉類別，forward 多了視窗
    net.win_cfg = {"size": size, "ch": ch, "prev": int(prev), "own2": int(own2), "depth": depth}  # 在網路物件上記下實際採用的設定（selftest 會印出），可據此重建同樣的包裝
    return net  # 回傳包裝後的網路


def install():  # 只在本行程內改變 pid_win.wrap 的分派
    """In THIS process only: pid_win.wrap() sends a cfg with "own2" or "depth" to wrap() above (the loaders call pid_win.wrap
    with the cfg stored in the checkpoint). Every other cfg goes where it went before."""
    import sys  # 調整搜尋路徑
    from pathlib import Path  # 取本檔所在的資料夾
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓 pid_win 可以匯入
    import pid_win  # 要改分派的模組
    if getattr(pid_win.wrap, "_w3", False):  # 已經裝過了
        return  # 不重複包裝
    plain = pid_win.wrap  # 保留原本的 wrap

    def dispatch(net, cfg):  # 新的分派函式
        return wrap(net, cfg) if (cfg.get("own2") or cfg.get("depth")) else plain(net, cfg)  # W3 的設定走這裡，其餘照舊

    dispatch._w3 = True  # 標記為已安裝
    pid_win.wrap = dispatch  # 換掉模組的 wrap


def selftest():  # 自我檢查
    """windows3 against pid_win2.windows2 (own and previous-ply windows) and against a plain replay (the third window);
    a wrapped toy module: zero output at the start, gradients reach every encoder, the state loads back."""
    import sys  # 調整搜尋路徑
    from pathlib import Path  # 取本檔所在的資料夾
    import numpy as np  # 單純重播用 numpy
    import torch  # 張量運算
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的模組可以匯入
    import pid_win2  # 對照用的 windows2
    rng = np.random.default_rng(0)  # 固定種子
    B, T, size = 6, 90, 9  # 6 局、90 手、9 x 9 視窗
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 60:] = 362; pos[2, 7] = pos[2, 3]; pos[3, 10] = 362  # 一手 pass、補齊、同一點下兩次、一手隱藏
    mov = np.tile(np.arange(T) % 2, (B, 1))  # 黑白交替
    tp, tm = torch.as_tensor(pos), torch.as_tensor(mov)  # 轉成張量
    cells, ok, pcells, pok, ocells, ook = [x.numpy() for x in windows3(tp, tm, size, True, True)]  # 三個視窗都開
    c2, ok2, _a, p2, _q, pok2 = pid_win2.windows2(tp, tm, size, False, True)  # 對照：只開第二個視窗
    same = int((cells != c2.numpy()).sum() + (ok != ok2.numpy()).sum() + (pcells != p2.numpy()).sum() + (pok != pok2.numpy()).sum())  # 與 windows2 不同的數量
    c1, ok1, a1, b1, d1, e1 = windows3(tp, tm, size)  # 變體都關
    assert a1 is None and b1 is None and d1 is None and e1 is None and int((c1.numpy() != cells).sum()) == 0  # 都關時只剩自己的視窗，且內容相同
    p = size // 2  # 視窗半徑
    bad = {"ocells": 0, "ook": 0}  # 第三個視窗的不一致數
    n = 0  # 比對過的格數
    for b in range(B):  # 每一局
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)  # 各點的顏色＋1 與落子手數
        for t in range(T):  # 每一手
            has = pos[b, t] < 361  # 這手有落點
            haso = bool(has and t > 1 and pos[b, t - 2] < 361)  # 這手與兩手前都有落點
            bad["ook"] += int(bool(ook[b, t]) != haso)  # 第三個視窗的遮罩
            if haso:  # 第三個視窗有效
                y, x = divmod(int(pos[b, t - 2]), 19)  # 視窗中心：自己的上一手
                for i in range(size):  # 視窗的列
                    for j in range(size):  # 視窗的行
                        yy, xx = y - p + i, x - p + j  # 這格在盤上的座標
                        if not (0 <= yy < 19 and 0 <= xx < 19):  # 盤外
                            want = 3  # 狀態 3
                        elif board[yy, xx] == 0:  # 空點
                            want = 0  # 狀態 0
                        else:  # 有子
                            dd = (t + 1) - when[yy, xx]  # 這顆子是幾手前下的
                            want = 4 if dd == 1 else 5 if dd == 2 else (1 if board[yy, xx] - 1 == mov[b, t] else 2)  # 期望的狀態
                        bad["ocells"] += int(ocells[b, t, i * size + j] != want)  # 狀態不一致
                        n += 1  # 計數
            if has:  # 有落點才落子
                y, x = divmod(int(pos[b, t]), 19)  # 落點座標
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1  # 落子（不提子）
    print(f"windows3 against pid_win2.windows2: {same} differences; the third window against a plain replay: differing {bad} of {n} cells")  # 回報視窗的比對結果

    class Toy(torch.nn.Module):  # 只含 wrap() 會用到的部分
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.gl = torch.nn.Linear(4, 32)  # wrap() 從這裡取網路寬度與裝置

    rc = 0 if same == 0 and not any(bad.values()) else 1  # 視窗比對的結果
    for cfg in ({"size": 9, "ch": 64, "prev": 1, "own2": 1}, {"size": 9, "ch": 128, "prev": 1}, {"size": 9, "ch": 64, "prev": 1, "depth": 2}):  # 三種設定：第三個視窗、加寬、加深
        torch.manual_seed(0)  # 固定初始化
        net = wrap(Toy(), dict(cfg))  # 包裝玩具模組
        out = net.win(tp, tm)  # 起點的輸出
        zero = float(out.abs().max())  # 應為 0
        for e in net.win.children():  # 把從零開始的層都移離零，之後每個參數都必須拿到梯度
            torch.nn.init.normal_(e.out.weight, std=0.02)  # 輸出層
            for m in e.more:  # 每個殘差塊
                torch.nn.init.normal_(m[2].weight, std=0.02)  # 第二層卷積
        net.win(tp, tm).square().sum().backward()  # 反向傳播
        nograd = [k for k, v in net.win.named_parameters() if v.grad is None or float(v.grad.abs().sum()) == 0]  # 沒拿到梯度的參數
        net2 = wrap(Toy(), dict(cfg))  # 另一個同設定的網路
        net2.win.load_state_dict(net.win.state_dict())  # 載入第一個網路的狀態
        back = float((net2.win(tp, tm) - net.win(tp, tm)).abs().max())  # 讀回後的輸出差
        print(f"cfg {net.win_cfg}: {sum(v.numel() for v in net.win.parameters())} parameters, output at the start {zero:g} (want 0), "  # 回報參數數與起點輸出
              f"parameters without gradient {nograd or 'none'}, reload difference {back:g}")  # 以及梯度與讀回的檢查
        rc |= int(zero != 0 or bool(nograd) or back != 0)  # 任一項不符就算失敗
    install()  # 載入程式走的路徑：以檢查點中存的 cfg 呼叫 pid_win.wrap()
    import pid_win  # 檢查分派結果
    names = [type(pid_win.wrap(Toy(), dict(c)).win).__name__ for c in ({"size": 9, "ch": 64, "prev": 1, "own2": 1}, {"size": 9, "ch": 64, "prev": 1, "depth": 2},  # 四種設定各包裝一次
                                                                    {"size": 9, "ch": 128, "age": 0, "prev": 1}, {"size": 9, "ch": 64})]  # 取得編碼器的類別名稱
    print(f"dispatch after install(): {names} (want ['Win3', 'Win3', 'Win2', 'Win'])")  # 回報分派結果
    rc |= int(names != ["Win3", "Win3", "Win2", "Win"])  # 分派必須正確
    return rc  # 0 表示全部通過


if __name__ == "__main__":  # 直接執行時跑自我檢查
    import sys  # 設定結束碼
    sys.exit(selftest())  # 以檢查結果為結束碼
