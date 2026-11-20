#!/usr/bin/env python
"""Does the pair network repair the near misses? (METHOD_NOTES 2.144) - the registered test.

Official pseudo-pools (per candidate 3 of the 10 reference games held out, true answers; four hold-out seeds, both added
pools; per candidate three 1-game questions, one 2-game and one 3-game question), the fused scores of one fusion document
in row-standard units. The pair network (scripts/pid_pair_train.py) reads every question game together with each of the 7
reference games of the first K candidates of the fused order; a candidate's pair score X is the mean over his reference
games and the question's games. Only the ORDER OF THE FIRST K changes:  new = fused + beta * X  (X as it is, or
standardised inside the K).
  --build   step 1 (CPU): the questions, the first K candidates and their fused scores, every candidate's look-alikes
            -> --tasks (a pickle; built once per document)
  default   step 2 (one process per GPU): the pair scores of every network, without and with a short fine-tune on the
            pool's own reference sets (question = one reference game, right candidate = other games of the same candidate,
            wrong candidates = his look-alikes among the pool's candidates; held-out games are never read)
            step 3: form, beta, fine-tune length and network (or the mean of several) are chosen on hold-out seeds 0, 1;
            the verdict is read on seeds 2, 3 (bootstrap over candidates).
From the AI_CUP_2026-1 checkout. Evaluation only: the answers of the real questions are never used.
"""
import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402

G = {}
SIZES = (1, 2, 3)
BETAS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))


def score_of(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def one_seed(task):
    from pid_probe_aug import fuse_v
    from pid_probe_qq import unit
    s, ft_dir, pool, K = task
    P = G["P"]
    C = P.C
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    qs = {1: [[g] for p in P.players for g in hold[p]], 2: [list(hold[p][:2]) for p in P.players], 3: [list(hold[p]) for p in P.players]}
    own = {1: np.repeat(np.arange(C), 3), 2: np.arange(C), 3: np.arange(C)}
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    vecs = []
    for key in P.net_keys:
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d_ = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        vecs.append(np.stack([d_[k] for k in ckeys]))
    out = {"pool": pool, "seed": s, "ref": [[k for k, _ in ref[p]] for p in P.players], "sizes": {}}
    for n, q in qs.items():
        F = standardise(fuse_v(P, ref, q, vecs, row))  # row-standard units (owner 3.57 against the best stranger's 3.34 on 1-game questions)
        top = np.argsort(-F, axis=1)[:, :K]
        rank = (F > F[np.arange(len(F)), own[n]][:, None]).sum(1)
        out["sizes"][n] = {"keys": [[k for k, _ in x] for x in q], "own": own[n], "top": top, "ftop": np.take_along_axis(F, top, 1), "rank": rank}
    # every candidate's look-alikes among the pool's candidates (reference games only): the wrong candidates of the fine-tune
    sets = [ref[p] for p in P.players]
    A = sparse.csr_matrix((np.ones(sum(len(x) for x in sets), dtype=np.float32), (np.repeat(np.arange(C), [len(x) for x in sets]), np.array([row[k] for x in sets for k, _ in x]))), shape=(C, len(row)))
    acc = 0.0
    for V, w in zip(vecs, P.wts):
        U = unit(np.asarray(A @ V, dtype=np.float64))
        acc = acc + w * (U @ U.T)
    np.fill_diagonal(acc, -np.inf)
    out["near"] = np.argsort(-acc, axis=1)[:, :16]
    r1 = out["sizes"][1]["rank"]
    print(f"  {pool} seed {s}: 1-game questions score {score_of(r1).mean():.4f}, Top-1 {(r1 == 0).mean():.3f}, owner inside the first {K}: {(r1 < K).mean():.3f}", flush=True)
    return (pool, s), out


def build(a, t0):
    import torch
    from aicup_pid.embed import PlayerEmbedding
    from aicup_pid.trunkstyle import load_store
    from pid_probe_qq import Pool
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    torch.set_num_threads(1)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    tasks = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        G.update(P=P)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            tasks.update(ex.map(one_seed, [(s, a.ft_dir, pool, a.topk) for s in a.seeds]))
        print(f"{pool}: tasks of seeds {a.seeds} ({time.time() - t0:.0f} s)", flush=True)
    tmp = Path(a.tasks).with_name(Path(a.tasks).name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        pickle.dump({"doc": a.doc, "topk": a.topk, "tasks": tasks}, fh)
    os.replace(tmp, a.tasks)
    print(f"wrote {a.tasks} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def load_arrays(ply_dir, need):
    parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
    want = np.array(sorted(need))
    for f in sorted(glob.glob(f"{ply_dir}/ply_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        key = z["key"].astype(str)
        m = np.isin(key, want)
        if m.any():
            for name in parts:
                parts[name].append(key[m] if name == "key" else z[name][m])
    A = {k: np.concatenate(x) for k, x in parts.items()}
    if len(A["key"]) != len(set(A["key"].tolist())) or set(A["key"].tolist()) != set(need):
        raise ValueError(f"{len(set(need) - set(A['key'].tolist()))} pool games without per-ply vectors ({len(A['key'])} rows for {len(need)} games)")
    return A, {k: i for i, k in enumerate(A["key"].tolist())}


def finetune(PT, net0, A, ref_rows, near, steps, a, tq, dev, sym, seed):
    import torch
    import torch.nn.functional as F
    net = copy.deepcopy(net0).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.ft_lr, weight_decay=0.05)
    rng = np.random.default_rng(seed)
    K, R, n = 8, 3, a.ft_anchors
    ok = np.flatnonzero([len(r) >= R + 1 for r in ref_rows])
    tgt = torch.zeros(n, dtype=torch.long, device=dev)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = a.ft_lr * min(1.0, (step + 1) / 10)
        q_rows, r_rows = np.zeros(n, np.int64), np.zeros((n, K, R), np.int64)
        for i, c in enumerate(rng.choice(ok, n, replace=False)):
            g = rng.permutation(ref_rows[c])[:R + 1]
            q_rows[i], r_rows[i, 0] = g[0], g[1:]
            cand = [w for w in near[c] if len(ref_rows[w]) >= R]
            for j, w in enumerate(rng.choice(cand, K - 1, replace=False)):
                r_rows[i, j + 1] = rng.permutation(ref_rows[w])[:R]
        loss = F.cross_entropy(PT.episode_logits(net, A, q_rows, r_rows, tq, dev, sym, rng.integers(8, size=n), 0.3), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net.eval(), float(loss.detach())


def score_pairs(PT, net, A, qrow, rrow, tq, dev, batch):
    import torch
    out = np.zeros(len(qrow), np.float32)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(qrow), batch):
            out[i:i + batch] = net(PT.pack(A, qrow[i:i + batch], tq, dev), PT.pack(A, rrow[i:i + batch], tq, dev)).float().cpu().numpy()
    return out * float(net.pair_scale.exp())


def score_worker(gpu, ckpts, tasks, A, row, a, t0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch  # noqa: F401
    import pid_pair_train as PT
    dev, sym = "cuda", PT.sym_table()
    for ck in ckpts:
        name = Path(ck).stem
        net0, cfg = PT.load_pair(ck)
        net0 = net0.to(dev).eval()
        tq = cfg["tq"]
        for (pool, s), t in tasks.items():
            todo = [ft for ft in a.ft if a.redo or not (Path(a.eval_dir) / f"{name}_{pool}_s{s}_t{ft}.npz").exists()]
            if not todo:
                continue
            ref_rows = [np.array([row[k] for k in ks], np.int64) for ks in t["ref"]]
            want = {}
            for n in SIZES:
                z = t["sizes"][n]
                for ks, top in zip(z["keys"][:a.limit], z["top"]):
                    for k in ks:
                        for c in top:
                            want[(k, int(c))] = None
            pairs = list(want)
            reps = np.array([len(ref_rows[c]) for _, c in pairs])
            qrow, rrow = np.repeat(np.array([row[k] for k, _ in pairs], np.int64), reps), np.concatenate([ref_rows[c] for _, c in pairs])
            grp = np.repeat(np.arange(len(pairs)), reps)
            for ft in todo:
                t1 = time.time()
                net, last = (net0, float("nan")) if ft == 0 else finetune(PT, net0, A, ref_rows, t["near"], ft, a, tq, dev, sym, 100 * s + (0 if pool == "pub2" else 1))
                sc = score_pairs(PT, net, A, qrow, rrow, tq, dev, a.batch)
                x = dict(zip(pairs, np.bincount(grp, weights=sc, minlength=len(pairs)) / np.maximum(np.bincount(grp, minlength=len(pairs)), 1)))
                res = {f"x{n}": np.array([[np.mean([x[(k, int(c))] for k in ks]) for c in top] for ks, top in zip(t["sizes"][n]["keys"][:a.limit], t["sizes"][n]["top"])], np.float32) for n in SIZES}
                tmp = Path(a.eval_dir) / f"{name}_{pool}_s{s}_t{ft}.tmp.npz"
                np.savez(tmp, **res)
                os.replace(tmp, Path(a.eval_dir) / f"{name}_{pool}_s{s}_t{ft}.npz")
                print(f"  {name} {pool} seed {s} fine-tune {ft}: {len(qrow)} pairs of games for {len(pairs)} (question game, candidate) pairs, {time.time() - t1:.0f} s"
                      + ("" if ft == 0 else f", last fine-tune loss {last:.3f}") + f" ({time.time() - t0:.0f} s)", flush=True)
                if ft:
                    del net
    return 0


def rerank(z, X, form, beta):
    """Per-question score when the first K are re-ordered by fused + beta * X (the owner's place is unchanged outside the K)."""
    q = len(X)
    top, ftop, rank, own = z["top"][:q], z["ftop"][:q], z["rank"][:q], z["own"][:q]
    Xv = X if form == "raw" else (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-9)
    new = Xv if np.isinf(beta) else ftop + beta * (0.2 * Xv if form == "raw" else Xv)
    pos = top == own[:, None]
    inside = pos.any(1)
    r_in = (new > np.where(pos, new, -np.inf).max(1)[:, None]).sum(1)
    return score_of(np.where(inside, r_in, rank))


def diffs(tasks, Xc, keys, form, beta):
    d, n_, cl = [], [], []
    for key in keys:
        for n in SIZES:
            z = tasks[key]["sizes"][n]
            new = rerank(z, Xc[key][n], form, beta)
            d.append(new - score_of(z["rank"][:len(new)]))
            n_.append(np.full(len(new), n))
            cl.append(np.char.add(key[0] + "|", z["own"][:len(new)].astype(str)))
    return np.concatenate(d), np.concatenate(n_), np.concatenate(cl)


def balanced(d, n):
    return float(np.mean([d[n == m].mean() for m in SIZES]))


def boot(d, n, cluster, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    per = num[:, pick].sum(2) / den[:, pick].sum(2)  # [size, B]
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"diff": balanced(d, n), "ci95": ci(per.mean(0)), "by_n": {str(m): float(d[n == m].mean()) for m in SIZES}, "ci95_by_n": {str(m): ci(per[i]) for i, m in enumerate(SIZES)}}


def analyse(a, tasks, names, t0):
    rng = np.random.default_rng(0)
    keys = sorted(tasks)
    sel, ver = [k for k in keys if k[1] in a.select_seeds], [k for k in keys if k[1] not in a.select_seeds]
    X = {}
    for name in names:
        for ft in a.ft:
            got = {}
            for pool, s in keys:
                f = Path(a.eval_dir) / f"{name}_{pool}_s{s}_t{ft}.npz"
                if not f.exists():
                    break
                z = np.load(f)
                got[(pool, s)] = {n: z[f"x{n}"].astype(np.float64) for n in SIZES}
            else:
                X[f"{name} t{ft}"] = got
    zs = lambda x: (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-9)  # noqa: E731
    for label, members in a.mean:
        for ft in a.ft:
            ms = [f"{m} t{ft}" for m in members]
            if len(ms) > 1 and all(m in X for m in ms):
                X[f"mean({label}) t{ft}"] = {k: {n: np.mean([zs(X[m][k][n]) for m in ms], 0) for n in SIZES} for k in keys}
    out = {"select_seeds": a.select_seeds, "configs": {}}
    z1 = [tasks[k]["sizes"][1] for k in keys]
    print(f"\n1-game questions, all seeds: fused Top-1 {np.mean([(z['rank'] == 0).mean() for z in z1]):.3f}, owner inside the first {a.topk_seen}: {np.mean([(z['rank'] < a.topk_seen).mean() for z in z1]):.3f}; "
          f"score now {np.mean([score_of(z['rank']).mean() for z in z1]):.4f}, with a perfect order of the first {a.topk_seen}: {np.mean([(z['rank'] < a.topk_seen).mean() for z in z1]):.4f}", flush=True)
    print(f"\nform and beta chosen on seeds {a.select_seeds}; the verdict columns are seeds {sorted({k[1] for k in ver})} (size-balanced over 1 / 2 / 3 games, bootstrap over candidates)")
    best = None
    for cfg, Xc in X.items():
        grid = {(form, b): balanced(*diffs(tasks, Xc, sel, form, b)[:2]) for form in ("z", "raw") for b in BETAS}
        (form, b), g_sel = max(grid.items(), key=lambda kv: kv[1])
        res = {"form": form, "beta": b, "select": g_sel, "grid": {f"{f_} {b_}": v for (f_, b_), v in grid.items()}}
        if ver:
            res["verdict"] = boot(*diffs(tasks, Xc, ver, form, b), a.boot, rng)
            res["verdict_by_pool"] = {p: balanced(*diffs(tasks, Xc, [k for k in ver if k[0] == p], form, b)[:2]) for p in sorted({k[0] for k in ver})}
        # how the pair score alone orders the first K of the 1-game questions (all seeds)
        hit_f, hit_p, ins = [], [], []
        for k in keys:
            z = tasks[k]["sizes"][1]
            x = Xc[k][1]
            q = len(x)
            pos = z["top"][:q] == z["own"][:q, None]
            ins.append(pos.any(1)); hit_f.append(z["rank"][:q] == 0); hit_p.append(pos[np.arange(q), x.argmax(1)])
        ins, hit_f, hit_p = np.concatenate(ins), np.concatenate(hit_f), np.concatenate(hit_p)
        res["one_game"] = {"pair_alone_top1": float(hit_p.mean()), "fused_top1": float(hit_f.mean()), "pair_right_fused_wrong": float((hit_p & ~hit_f).mean()), "fused_right_pair_wrong": float((hit_f & ~hit_p).mean())}
        out["configs"][cfg] = res
        v = res.get("verdict")
        print(f"{cfg:28s} {form:3s} beta {b:<5g} seeds {a.select_seeds} {g_sel:+.4f} | "
              + (f"verdict {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  1 game {v['by_n']['1']:+.4f} [{v['ci95_by_n']['1'][0]:+.4f}, {v['ci95_by_n']['1'][1]:+.4f}]  2 {v['by_n']['2']:+.4f}  3 {v['by_n']['3']:+.4f}  pools {({p: round(x_, 4) for p, x_ in res['verdict_by_pool'].items()})} | " if v else "")
              + f"1-game Top-1 inside the first K: pair alone {hit_p.mean():.3f}, fused {hit_f.mean():.3f}, pair right / fused wrong {(hit_p & ~hit_f).mean():.3f}, the reverse {(hit_f & ~hit_p).mean():.3f}", flush=True)
        if best is None or g_sel > best[1]:
            best = (cfg, g_sel)
    if best and ver:
        v = out["configs"][best[0]]["verdict"]
        pools_ok = all(x_ > 0 for x_ in out["configs"][best[0]]["verdict_by_pool"].values())
        a_ok, b_ok = v["diff"] >= 0.004 and v["ci95"][0] > 0, v["by_n"]["1"] > 0 and v["ci95_by_n"]["1"][0] > 0
        word = "PASS (build the re-ranking stage into the pipeline)" if a_ok and b_ok and pools_ok else "PARTIAL (one more round, no pipeline work yet)" if v["diff"] > 0 and v["ci95"][0] > 0 else "FAIL"
        out["chosen"], out["rule"] = best[0], {"a": bool(a_ok), "b": bool(b_ok), "c": bool(pools_ok), "verdict": word}
        print(f"\nchosen on seeds {a.select_seeds}: {best[0]} ({best[1]:+.4f}); rule 2.144: (a) balanced gain >= +0.004 with the interval above 0: {a_ok}; (b) 1-game gain with the interval above 0: {b_ok}; (c) both pools above 0: {pools_ok} -> {word}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", default="v2bgm_d2")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--nets", nargs="*", default=[], help="pair checkpoints")
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--eval-dir", default="/work/u2507115/pid_v2/pair_eval")
    ap.add_argument("--ft", nargs="+", type=int, default=[0, 50, 150], help="fine-tune steps on the pool's reference sets")
    ap.add_argument("--ft-lr", type=float, default=5e-5)
    ap.add_argument("--ft-anchors", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="only the first questions of every size (a start-up check)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="no scoring: step 3 on the pair scores already written for --nets")
    ap.add_argument("--select-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--mean", nargs="*", default=[], help="label=net,net,... : also the mean of these networks' standardised pair scores")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.build:
        return build(a, t0)
    with open(a.tasks, "rb") as fh:
        doc = pickle.load(fh)
    tasks, a.topk_seen = doc["tasks"], doc["topk"]
    a.mean = [(m.split("=", 1)[0], m.split("=", 1)[1].split(",")) for m in a.mean]
    Path(a.eval_dir).mkdir(parents=True, exist_ok=True)
    A = row = None
    if not a.analyse_only:
        need = {k for t in tasks.values() for ks in t["ref"] for k in ks} | {k for t in tasks.values() for n in SIZES for ks in t["sizes"][n]["keys"] for k in ks}
        A, row = load_arrays(a.ply_dir, need)
        print(f"tasks of document {doc['doc']}: {len(tasks)} pool x seed, first {a.topk_seen} candidates; {len(A['key'])} official games ({A['GL'].nbytes / 1e9:.1f} GB); {len(a.nets)} pair networks on GPUs {a.gpus} ({time.time() - t0:.0f} s)", flush=True)
    pids = {}
    for i, gpu in enumerate([] if a.analyse_only else a.gpus):
        mine = a.nets[i::len(a.gpus)]
        if not mine:
            continue
        sys.stdout.flush()
        pid = os.fork()
        if pid == 0:
            rc = 1
            try:
                rc = score_worker(gpu, mine, tasks, A, row, a, t0)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(int(rc or 0))
        pids[pid] = mine
    failed = 0
    while pids:
        pid, status = os.wait()
        ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        print(f"scorer {pid} ({[Path(x).stem for x in pids.pop(pid)]}) {'done' if ok else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
        failed |= not ok
    rc = analyse(a, tasks, [Path(x).stem for x in a.nets], t0) if a.out else 0
    return int(failed or rc)


if __name__ == "__main__":
    sys.exit(main())
