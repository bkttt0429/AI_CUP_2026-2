#!/usr/bin/env python
"""The pair network on the OLD-POOL simulation (METHOD_NOTES 2.147): a second, independent reading.

scripts/pid_pair_eval.py scores the pair network on the added pools' pseudo-pools (the candidates' own held-out games,
7-game reference sets, questions of 1 / 2 / 3 games). Here the same re-ranking is read on the simulated pools cut from the
ORIGINAL pools (aicup_pid.v2.original_sim, as scripts/pid_v2_oldsim.py): another population of official players (400 per
side, Black games), 10-game reference sets as in a real pool, questions of 1 / 1 / 1 / 2 / 2 / 3 / 5 games. Nothing is
chosen here: the settings (form, beta) come from the pseudo-pools and are given with --fixed; a small grid is printed
beside them for information only.
  --build     the tasks (CPU): fused scores of the document on both sides, the first K candidates, look-alikes
  default     pair scores of --nets (one process per GPU; fine-tune on the simulated pool's reference sets as --ft), then
              the gains of every network / mean at the --fixed settings, bootstrap over candidates
From the AI_CUP_2026-1 checkout. Evaluation only; the two sides are never mixed.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import pid_pair_eval as E  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402

OLD_SIZES = (1, 2, 3, 5)


def build(a, t0):
    import torch
    from aicup_pid.components import parse_all
    from aicup_pid.embed import PlayerEmbedding
    from aicup_pid.trunkstyle import load_store
    from pid_probe_qq import unit
    T = Path(a.tests_dir)
    doc = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq"
    torch.set_num_threads(4)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(a.store)
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    tasks = {}
    for side, tag in (("pub", "public"), ("priv", "private")):
        cache, vecs = {}, []
        for v, tg in nets:  # one entry per weight unit of the document, as scripts/pid_v2_oldsim.py
            f = Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{v}{tg}_t50.npz"
            if f not in cache:
                cache[f] = load(f)
            vecs.append(cache[f])
        cands, questions, owner, sgfs = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(vecs[0]), a.seed)
        players = sorted(cands)
        C = len(players)
        truth = np.array([players.index(p) for p in owner])
        n = np.array([len(q) for q in questions])
        moves = parse_all(sgfs, a.workers)
        S = v2.components(cands, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
        S["seq"] = v2.components(cands, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=vecs)["seq"]
        F = standardise(fus.scores([standardise(S[c]) for c in comps]))
        rank = (F > F[np.arange(len(F)), truth][:, None]).sum(1)
        out = {"pool": f"old{side}", "seed": a.seed, "ref": [[k for k, _ in cands[p]] for p in players], "sizes": {}}
        for m in OLD_SIZES:
            r = np.flatnonzero(n == m)
            top = np.argsort(-F[r], axis=1)[:, :a.topk]
            out["sizes"][m] = {"keys": [[k for k, _ in questions[i]] for i in r], "own": truth[r], "top": top, "ftop": np.take_along_axis(F[r], top, 1), "rank": rank[r]}
        ckeys = [k for p in players for k, _ in cands[p]]
        row = {k: i for i, k in enumerate(ckeys)}
        acc = 0.0
        for d_ in cache.values():
            V = np.stack([d_[k] for k in ckeys]).astype(np.float64)
            U = unit(np.add.reduceat(V, np.cumsum([0] + [len(cands[p]) for p in players])[:-1], axis=0))
            acc = acc + U @ U.T
        np.fill_diagonal(acc, -np.inf)
        out["near"] = np.argsort(-acc, axis=1)[:, :16]
        tasks[(f"old{side}", a.seed)] = out
        sc = E.score_of(rank)
        print(f"old{side}: {C} candidates, {len(questions)} questions; score " + "  ".join(f"n={m} {sc[n == m].mean():.4f}" for m in OLD_SIZES)
              + f"; 1-game Top-1 {(rank[n == 1] == 0).mean():.3f}, owner inside the first {a.topk}: {(rank[n == 1] < a.topk).mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        assert len(row) == len(set(ckeys))
    tmp = Path(a.tasks).with_name(Path(a.tasks).name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        pickle.dump({"doc": a.doc, "topk": a.topk, "tasks": tasks}, fh)
    os.replace(tmp, a.tasks)
    print(f"wrote {a.tasks} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def analyse(a, tasks, names, t0):
    rng = np.random.default_rng(0)
    keys = sorted(tasks)
    zs = lambda x: (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-9)  # noqa: E731
    X = {}
    for name in names:
        for ft in a.ft:
            fs = {k: Path(a.eval_dir) / f"{name}_{k[0]}_s{k[1]}_t{ft}.npz" for k in keys}
            if all(f.exists() for f in fs.values()):
                X[f"{name} t{ft}"] = {k: (lambda z: {m: z[f"x{m}"].astype(np.float64) for m in OLD_SIZES})(np.load(f)) for k, f in fs.items()}
    for label, members in a.mean:
        for ft in a.ft:
            ms = [f"{m} t{ft}" for m in members]
            if len(ms) > 1 and all(m in X for m in ms):
                X[f"mean({label}) t{ft}"] = {k: {m: np.mean([zs(X[c][k][m]) for c in ms], 0) for m in OLD_SIZES} for k in keys}
    z1 = [tasks[k]["sizes"][1] for k in keys]
    print(f"\nold-pool simulation, 1-game questions: fused Top-1 {np.mean([(z['rank'] == 0).mean() for z in z1]):.3f}, score {np.mean([E.score_of(z['rank']).mean() for z in z1]):.4f}, "
          f"owner inside the first {a.topk_seen} (= the score of a perfect order): {np.mean([(z['rank'] < a.topk_seen).mean() for z in z1]):.4f}", flush=True)
    fixed = [(a.fixed[i], float(a.fixed[i + 1])) for i in range(0, len(a.fixed), 2)]
    out = {"fixed": fixed, "configs": {}}
    for cfg, Xc in X.items():
        res = {"fixed": {}, "grid": {f"{f_} {b_}": E.balanced(*E.diffs(tasks, Xc, keys, f_, b_)[:2]) for f_ in ("z", "raw") for b_ in E.BETAS}}
        line = []
        for form, b in fixed:
            v = E.boot(*E.diffs(tasks, Xc, keys, form, b), a.boot, rng)
            v["by_pool"] = {k[0]: E.balanced(*E.diffs(tasks, Xc, [k], form, b)[:2]) for k in keys}
            res["fixed"][f"{form} {b:g}"] = v
            line.append(f"{form} {b:g}: {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] by n " + " ".join(f"{m}:{v['by_n'][str(m)]:+.4f}" for m in OLD_SIZES) + f" sides {({p: round(x_, 4) for p, x_ in v['by_pool'].items()})}")
        best = max(res["grid"].items(), key=lambda kv: kv[1])
        out["configs"][cfg] = res
        print(f"{cfg:28s} " + " | ".join(line) + f" | (information only) best of the grid: {best[0]} {best[1]:+.4f}", flush=True)
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
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nets", nargs="*", default=[])
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_original")
    ap.add_argument("--eval-dir", default="/work/u2507115/pid_v2/pair_eval")
    ap.add_argument("--ft", nargs="+", type=int, default=[0, 50, 150])
    ap.add_argument("--ft-lr", type=float, default=5e-5)
    ap.add_argument("--ft-anchors", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--fixed", nargs="+", default=["z", "0.5"], help="form beta [form beta ...]: the settings read off the pseudo-pools")
    ap.add_argument("--mean", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    E.SIZES = OLD_SIZES  # the helpers of pid_pair_eval read the question sizes from there
    if a.build:
        return build(a, t0)
    with open(a.tasks, "rb") as fh:
        doc = pickle.load(fh)
    tasks, a.topk_seen = doc["tasks"], doc["topk"]
    a.mean = [(m.split("=", 1)[0], m.split("=", 1)[1].split(",")) for m in a.mean]
    Path(a.eval_dir).mkdir(parents=True, exist_ok=True)
    failed = 0
    if not a.analyse_only:
        need = {k for t in tasks.values() for ks in t["ref"] for k in ks} | {k for t in tasks.values() for m in OLD_SIZES for ks in t["sizes"][m]["keys"] for k in ks}
        A, row = E.load_arrays(a.ply_dir, need)
        print(f"old-pool tasks of document {doc['doc']}: {len(tasks)} sides; {len(A['key'])} games ({A['GL'].nbytes / 1e9:.1f} GB); {len(a.nets)} pair networks on GPUs {a.gpus} ({time.time() - t0:.0f} s)", flush=True)
        pids = {}
        for i, gpu in enumerate(a.gpus):
            mine = a.nets[i::len(a.gpus)]
            if not mine:
                continue
            sys.stdout.flush()
            pid = os.fork()
            if pid == 0:
                rc = 1
                try:
                    rc = E.score_worker(gpu, mine, tasks, A, row, a, t0)
                except BaseException:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(int(rc or 0))
            pids[pid] = mine
        while pids:
            pid, status = os.wait()
            ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            print(f"scorer {pid} ({[Path(x).stem for x in pids.pop(pid)]}) {'done' if ok else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
            failed |= not ok
    rc = analyse(a, tasks, [Path(x).stem for x in a.nets], t0) if a.out else 0
    return int(failed or rc)


if __name__ == "__main__":
    sys.exit(main())
