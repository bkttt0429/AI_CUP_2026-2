"""METHOD_NOTES 2.192: checks of pid_win2.py before a short run - the window builder against a replay (pid_win2.selftest), and
a wrapped m88_seqid network: equal to the plain network at the start (every new term starts at zero), a forward / backward pass
with mixed precision, and a save / load round trip through pid_win.wrap (the loaders' path). Run from the AI_CUP_2026-1 checkout."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd() / "logs/nano4")]
import m88_seqid as M  # noqa: E402
import pid_win  # noqa: E402
import pid_win2  # noqa: E402

rc = pid_win2.selftest()
dev = "cuda"
torch.manual_seed(0)
B, T = 64, 256
gl = torch.randint(-100, 100, (B, T, 128), device=dev).float()
pos = torch.randint(0, 361, (B, T), device=dev)
pos[:, 200:] = 362
pos[0, 3] = 361
mov = (torch.arange(T, device=dev)[None] % 2).expand(B, T).contiguous()
length = torch.full((B,), 200, device=dev)
subj = torch.zeros(B, dtype=torch.long, device=dev)
plain = M.build_model().to(dev).eval()
sd = {k: v.clone() for k, v in plain.state_dict().items()}
with torch.no_grad():
    ref = plain(gl, pos, mov, length, subj)
for cfg in ({"size": 9, "ch": 64, "age": 1, "prev": 0}, {"size": 9, "ch": 64, "age": 0, "prev": 1}, {"size": 9, "ch": 64, "age": 1, "prev": 1}):
    net = M.build_model().to(dev)
    net.load_state_dict(sd)
    pid_win.wrap(net, dict(cfg))  # through the dispatch of pid_win.wrap, as the trainer and the loaders do
    net.eval()
    with torch.no_grad():
        out = net(gl, pos, mov, length, subj)
    d0 = float((out - ref).abs().max())
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, pos, mov, length, subj)
            loss = -(z @ z.T).diagonal().mean() + (z @ z.T).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    g = sum(float(p.grad.abs().sum()) for n_, p in net.named_parameters() if n_.startswith("win.") and p.grad is not None)
    net.eval()
    with torch.no_grad():
        after = net(gl, pos, mov, length, subj)
    net2 = M.build_model().to(dev)
    pid_win.wrap(net2, dict(net.win_cfg))
    net2.load_state_dict(net.state_dict())
    net2.eval()
    with torch.no_grad():
        again = net2(gl, pos, mov, length, subj)
    d1 = float((after - again).abs().max())
    npar = sum(p.numel() for p in net.win.parameters())
    ok = d0 < 1e-6 and d1 < 1e-6 and g > 0 and float((after - ref).abs().max()) > 1e-4
    rc |= 0 if ok else 1
    print(f"cfg {net.win_cfg}: class {net.__class__.__name__}, window parameters {npar}; start equals plain {d0:.1e}; gradient into the window {g:.3g}; "
          f"moved after 3 steps {float((after - ref).abs().max()):.3g}; save/load round trip {d1:.1e} -> {'OK' if ok else 'FAILED'}")
net = M.build_model().to(dev)
pid_win.wrap(net, {"size": 9, "ch": 64})
print(f"plain window cfg still gives class {net.__class__.__name__} with {sum(p.numel() for p in net.win.parameters())} window parameters")
rc |= 0 if net.__class__.__name__ == "SeqIDW" else 1
print("ALL CHECKS PASSED" if rc == 0 else "CHECKS FAILED")
sys.exit(rc)
