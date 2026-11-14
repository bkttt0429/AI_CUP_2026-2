#!/usr/bin/env python
"""The one-to-one assignment of the multi-game questions on the OLD-POOL simulation (METHOD_NOTES 2.122): the confirmation of
scripts/pid_v2_assign_tune.py on pools that have 5-game questions (scripts/pid_v2_oldsim.py: the original pools' candidates,
official-population players no network was trained on; 10-game candidate sets, questions of 1 / 1 / 1 / 2 / 2 / 3 / 5 games;
Black only, 400 candidates per side).
Per draw: 67 candidates give their 5-game question, 67 OTHERS their 3-game question, 66 OTHERS their first 2-game question
(200 questions over 400 candidates, the density of the real added pools); with contamination p a share p of the questions is
replaced by another multi-game question of a candidate that already owns one. Methods and report as pid_v2_assign_tune.py;
the draws of each side are cut into four blocks so that the spread over 8 blocks can be read.
  python -I <this> --doc v2o13c_9 --out results/final/v2_assign_tune_old_v2o13c_9.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_assign_tune import methods, score  # noqa: E402

SPEC = ((5, 67), (3, 67), (2, 66))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--contam", nargs="+", type=float, default=[0.0, 0.05, 0.15])
    ap.add_argument("--ref", default="soft1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T = Path(a.tests_dir)
    doc = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq"
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(a.store)
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    weight = {}
    for key in nets:
        weight[key] = weight.get(key, 0) + 1
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
    res = {}
    for side, tag in (("pub", "public"), ("priv", "private")):
        first = load(Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{nets[0][0]}{nets[0][1]}_t{a.step}.npz")
        cands, questions, owner, sgfs = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(first), a.seed)
        players = sorted(cands)
        C = len(players)
        truth_all = np.array([players.index(p) for p in owner])
        size = np.array([len(q) for q in questions])
        moves = parse_all(sgfs, a.workers)
        S = v2.components(cands, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
        seq_acc, tot = 0.0, 0
        for (v, tg), w in weight.items():
            one = load(Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{v}{tg}_t{a.step}.npz")
            Cm = unit(np.stack([np.sum([one[k] for k, _ in cands[p]], axis=0) for p in players]).astype(np.float64))
            Qm = unit(np.stack([np.sum([one[k] for k, _ in q], axis=0) for q in questions]).astype(np.float64))
            seq_acc = seq_acc + w * standardise(Qm @ Cm.T)
            tot += w
        S["seq"] = seq_acc / tot
        F = fus.scores([standardise(S[c]) for c in comps])
        row = {}
        for j, (o, m) in enumerate(zip(truth_all, size)):
            row.setdefault((int(o), int(m)), j)  # the first question of that size of the candidate
        have = [i for i in range(C) if all((i, m) in row for m, _ in SPEC)]
        rng = np.random.default_rng(9300 + (0 if side == "pub" else 1))
        print(f"{side}: {C} candidates, {len(questions)} questions, {len(have)} candidates with questions of every size ({time.time() - t0:.0f} s)", flush=True)
        for block in range(4):
            acc = {}
            for _ in range(a.draws // 4):
                perm = rng.permutation(have)
                q, at = [], 0
                for m, cnt in SPEC:
                    q += [(int(c), m) for c in perm[at:at + cnt]]
                    at += cnt
                for p in a.contam:
                    qq = list(q)
                    k = int(round(p * len(qq)))
                    if k:
                        drop = set(rng.choice(len(qq), k, replace=False).tolist())
                        keep = [x for j, x in enumerate(qq) if j not in drop]
                        donors = rng.choice(len(keep), k, replace=False)
                        qq = keep + [(keep[j][0], rng.choice([m for m, _ in SPEC if m != keep[j][1]])) for j in donors]
                    rows = np.array([row[(c, int(m))] for c, m in qq])
                    truth = np.array([c for c, _ in qq])
                    Sq = F[rows]
                    base1 = Sq.argmax(1)
                    for name, idx in methods(Sq, a.lams, a.taus).items():
                        acc.setdefault((p, name), []).append((score(idx, truth), float((idx[:, 0] != base1).sum()), float((idx[:, 0] == truth).mean())))
            for key, vals in acc.items():
                res.setdefault(key, []).append(np.mean(np.array(vals), axis=0))
            print(f"{side} block {block}: " + "  ".join(f"{n} {np.mean([x[0] for x in acc[(0.0, n)]]):.4f}" for n in ("none", a.ref, "hard")) + f" ({time.time() - t0:.0f} s)", flush=True)
    out = {"doc": a.doc, "draws": a.draws, "spec": SPEC, "ref": a.ref, "table": {}}
    names = sorted({n for _p, n in res}, key=lambda n: (re.sub(r"[0-9.]+$", "", n), float(re.search(r"[0-9.]+$", n).group()) if re.search(r"[0-9.]+$", n) else 0.0))
    for p in a.contam:
        refv = np.array(res[(p, a.ref)])[:, 0]
        print(f"contamination {p:g}: score of the 200 multi-game questions; difference from {a.ref} = mean over the {len(refv)} blocks [min, max], blocks up")
        for n in names:
            v = np.array(res[(p, n)])
            dlt = v[:, 0] - refv
            out["table"][f"{p:g}|{n}"] = {"score": float(v[:, 0].mean()), "top1": float(v[:, 2].mean()), "changed_top1": float(v[:, 1].mean()), "diff": float(dlt.mean()),
                                         "diff_min": float(dlt.min()), "diff_max": float(dlt.max()), "blocks_up": int((dlt > 0).sum())}
            print(f"  {n:<12} {v[:, 0].mean():.4f} (Top-1 {v[:, 2].mean():.4f}, changes {v[:, 1].mean():5.1f} answers)  {dlt.mean():+.4f} [{dlt.min():+.4f}, {dlt.max():+.4f}]  up {int((dlt > 0).sum())}/{len(dlt)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
