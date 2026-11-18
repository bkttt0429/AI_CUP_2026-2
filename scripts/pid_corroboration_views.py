"""Extract eight candidate-only attention deletions and check the native readouts."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import ROOT, RUN, POOLS, atomic_json, atomic_npz, digest_file, guard, read_manifest
from aicup_pid.corroboration import quarter_labels

CACHE = ROOT / ".superpowers/corroboration_20261118"
PLAN = ROOT / "docs/superpowers/plans/2026-11-18-single-game-corroboration.md"
ASSETS = Path("/work/u2507115/pid_v2")
FAMILIES = ("o12Fs0e", "o33Fs0e")


@torch.inference_mode()
def masked_views(net, inputs, labels):
    captured = {}
    def remember(_module, args, kw):
        captured["tokens"] = args[0]
        captured["pad"] = kw["src_key_padding_mask"]
    handle = net.enc.register_forward_pre_hook(remember, with_kwargs=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native = net(*inputs)
    finally:
        handle.remove()
    groups = torch.as_tensor(labels, device=native.device)
    x, pad = captured["tokens"], captured["pad"]
    if pad.dtype != torch.bool or not net.enc.layers[0].self_attn.batch_first or pad[:, 0].any():
        raise ValueError("unexpected native transformer interface")
    outs = [native]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for kind in range(2):
            for quarter in range(4):
                mask = pad.clone()
                mask[:, 1:] |= groups[:, kind] == quarter
                cls = net.enc(x, src_key_padding_mask=mask)[:, 0]
                outs.append(F.normalize(net.head(cls).float(), dim=-1))
    return torch.stack(outs, dim=1).cpu().numpy().astype(np.float16)


def extract(family):
    budget = guard()
    torch.set_num_threads(3)
    manifest = read_manifest(RUN, verify=False)
    source = Path(manifest["source_root"])
    sys.path.insert(0, str(source / "scripts"))
    from pid_v2_seq_finetune import load_net, SUP_SETS
    import pid_sup

    games = [g for g in manifest["games"] if g["pool"] in POOLS]
    selected = {g["original_key"]: g for g in games}
    if len(selected) != 12000 or len(games) != 12000 or any(not k.startswith("c:") for k in selected):
        raise ValueError("ambiguous or unexpected candidate namespaces")
    ckpt = ASSETS / "seqnet" / f"ckpt_-1_{family}.pt"
    net = load_net(ckpt).eval()
    cfg = net.sup_cfg
    stores, _, stats = SUP_SETS[cfg.get("name")]
    sup = pid_sup.load_sup(stores, stats)
    mu, sd = net.mu.cpu().numpy(), net.sd.cpu().numpy()
    cache_info = {"budget": budget, "protocol": digest_file(PLAN), "checkpoint": digest_file(ckpt), "supervised_stats": digest_file(stats),
                  "archive_sha": {}, "manifest": digest_file(RUN / "manifest.json"), "family": family}
    if getattr(net, "win_exact", False):
        raise ValueError("these two registered checkpoints use unpacked colour input")
    archived = {}
    for pool in POOLS:
        src = ASSETS / "seq_ft_offsim" / f"offsim_{pool}_h3s1_net-1_{family}_t0.npz"
        with np.load(src) as z:
            a, v = z["key"].astype(str), z["emb"]
            for i, k in enumerate(a):
                if k in selected and selected[k]["pool"] == pool:
                    if k in archived:
                        raise ValueError("duplicate archived candidate")
                    archived[k] = v[i].astype(np.float32)
        cache_info["archive_sha"][pool] = digest_file(src)
    if set(archived) != set(selected):
        raise ValueError("archived intact vectors lack selected candidates")
    rows, vectors, lengths, worst_cos, max_diff = [], [], [], 1., 0.
    print("loaded", family, cfg, torch.cuda.get_device_name(0), flush=True)
    shards = sorted(p for p in (ASSETS / "ply_official").glob("ply_*.npz") if not p.name.endswith(".tmp.npz"))
    for part, shard in enumerate(shards):
        with np.load(shard) as z:
            allkeys = z["key"].astype(str)
            ix = np.array([i for i, k in enumerate(allkeys) if k in selected], dtype=int)
            if not len(ix):
                continue
            keys = allkeys[ix]
            gl, pos, mov, lens, subj = (z[field][ix] for field in ("GL", "POS", "MOV", "LEN", "SUBJ"))
        namespaced = [selected[k]["key"] for k in keys]
        if np.any(np.array([int(selected[k]["color"] == "W") for k in keys]) != subj):
            raise ValueError("subject differs from candidate manifest")
        ids, dense = pid_sup.dense(sup, keys.tolist(), gl.shape[1])
        if np.any(ids < 0):
            raise ValueError("incomplete supervised inputs")
        gl = pid_sup.put(gl.astype(np.float32), dense[ids].astype(np.float32), mu, sd, tuple(cfg["dims"]), bool(cfg["only"]))
        labels = np.stack([quarter_labels(k, m, n) for k, m, n in zip(namespaced, mov, lens)])
        out = []
        for start in range(0, len(keys), 80):
            sl = slice(start, start + 80)
            inputs = [torch.as_tensor(gl[sl], device="cuda")]
            inputs += [torch.as_tensor(a[sl].astype(np.int64), device="cuda") for a in (pos, mov, lens, subj)]
            out.append(masked_views(net, inputs, labels[sl]))
        vec = np.concatenate(out)
        ref = np.stack([archived[k] for k in keys])
        got = vec[:, 0].astype(np.float32)
        cosine = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
        worst_cos = min(worst_cos, float(cosine.min()))
        max_diff = max(max_diff, float(np.abs(ref-got).max()))
        if worst_cos < .999 or max_diff > .012 or not np.isfinite(vec).all():
            raise ValueError(f"native readout mismatch: min cosine {worst_cos}, max coordinate error {max_diff}")
        rows.extend(namespaced)
        lengths.extend(lens.tolist())
        vectors.append(vec)
        if part % 16 == 0:
            print("candidate rows", len(rows), "cos", worst_cos, "maxdiff", max_diff, flush=True)
    if sorted(rows) != sorted(g["key"] for g in games):
        raise ValueError("missing or duplicate per-ply candidate entries")
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / f"views_{family}.npz"
    atomic_npz(archive, keys=np.asarray(rows), views=np.concatenate(vectors), lengths=np.asarray(lengths))
    cache_info.update(rows=len(rows), min_cosine=worst_cos, max_abs_coordinate=max_diff, views_sha=digest_file(archive))
    atomic_json(CACHE / f"{family}.json", cache_info)
    print("verified views", family, len(rows), worst_cos, max_diff, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    extract(parser.parse_args().family)
