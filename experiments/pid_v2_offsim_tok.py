#!/usr/bin/env python
"""Does move-level matching add to the full pipeline on OFFICIAL players? (METHOD_NOTES 2.56, atom 3)

The official pseudo-pools of scripts/pid_v2_offsim.py (each added pool's own candidates, 3 of their 10 games held out as
questions of 1 / 2 / 3 games; both colours), one fusion document at its 50-step vectors. The move-level matching scores
(aicup_pid.tokmatch: network-independent, from the per-ply parts) are row-standardised and ADDED to the document's fused score
with a weight; the weight used on a pool is the best one on the OTHER pool (the two pools are never mixed otherwise).
Reported per variant: the fused score with it, the difference from the document alone (bootstrap over candidates), by size.
"""
import argparse
import glob
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
from aicup.submission import check_game  # noqa: E402
from aicup_pid import tokmatch, v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_offsim import SIZES, boot_mix, mix, per_question  # noqa: E402

WEIGHTS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2i_eq.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    runs = []
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        parts = {x: [] for x in ("key", "GL", "MOV", "LEN", "SUBJ")}
        for f in sorted(glob.glob(f"{a.ply_dir}/ply_*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            key = z["key"].astype(str)
            m = np.array([k in ckeys for k in key])
            if m.any():
                for name in parts:
                    parts[name].append(key[m] if name == "key" else z[name][m])
        arrays = {k: np.concatenate(x) for k, x in parts.items()}
        for s in a.seeds:
            hold = v2.candidate_holdout(cands, a.holdout, s)
            held = {k for g in hold.values() for k, _ in g}
            ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
            questions, truth = [], []
            for i, p in enumerate(players):
                h = [(k, c) for k, c in hold[p] if k not in bad]
                for g in h:
                    questions.append([g]); truth.append(i)
                for m in (2, 3):
                    if len(h) >= m:
                        questions.append(h[:m]); truth.append(i)
            truth, n = np.array(truth), np.array([len(q) for q in questions])
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            seqs = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz") for v, tg in nets]
            Z["seq"] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=seqs)["seq"])
            F = fus.scores([Z[c] for c in comps])
            tok = {name: standardise(M) for name, M in tokmatch.component(arrays, ref, questions, players).items()}
            runs.append({"pool": pool, "seed": s, "F": F, "tok": tok, "truth": truth, "n": n})
            base = per_question(F, truth)
            print(f"{pool} split {s}: {len(questions)} questions; document {mix(base, n):.4f}; move-level scores alone: "
                  + "  ".join(f"{k} {mix(per_question(x, truth), n):.4f}" for k, x in tok.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    names = list(runs[0]["tok"])
    out = {"document": Path(a.fusion).name, "step": a.step, "variants": {}}
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    base_all = cat(runs, lambda r: per_question(r["F"], r["truth"]))
    n_all = cat(runs, lambda r: r["n"])
    cl_all = cat(runs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
    print(f"both pools: document alone {mix(base_all, n_all):.4f} (n=1 {base_all[n_all == 1].mean():.4f}, n=2 {base_all[n_all == 2].mean():.4f}, n=3 {base_all[n_all == 3].mean():.4f})", flush=True)
    for name in names:
        chosen = {}
        for pool in ("pub2", "priv2"):  # the weight for a pool comes from the other pool
            others = [r for r in runs if r["pool"] != pool]
            chosen[pool] = max(WEIGHTS, key=lambda w: mix(cat(others, lambda r: per_question(r["F"] + w * r["tok"][name], r["truth"])), cat(others, lambda r: r["n"])))
        d = cat(runs, lambda r: per_question(r["F"] + chosen[r["pool"]] * r["tok"][name], r["truth"])) - base_all
        b = boot_mix(d, n_all, cl_all, a.boot, rng)
        b["by_n"] = {str(m): float(d[n_all == m].mean()) for m in SIZES}
        b["weights"] = chosen
        b["per_pool"] = {pool: float(mix(d[np.char.startswith(cl_all, pool + "|")], n_all[np.char.startswith(cl_all, pool + "|")])) for pool in ("pub2", "priv2")}
        b["curve"] = {str(w): float(mix(cat(runs, lambda r: per_question(r["F"] + w * r["tok"][name], r["truth"])) - base_all, n_all)) for w in WEIGHTS}
        out["variants"][name] = b
        print(f"   + {name:12s} {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}  per pool {({k: round(v_, 4) for k, v_ in b['per_pool'].items()})}  "
              f"weights {chosen}; by weight {({k: round(v_, 4) for k, v_ in b['curve'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
