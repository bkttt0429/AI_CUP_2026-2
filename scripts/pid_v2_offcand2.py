#!/usr/bin/env python
"""Which mixture of sequence networks identifies OFFICIAL players best (METHOD_NOTES 2.47): the leave-games-out test of
scripts/pid_v2_offcand.py (each added pool's own candidates, owners given by the organisers; vectors before the pool
fine-tune; pools never mixed) for freely defined groups of networks and their compositions.
  --groups name=<file pattern>[,<file pattern>...]   one vector file per network; {pool} is replaced by pub2 / priv2; a pattern
                                                     may hold a brace list {a|b|c} that expands to several networks
  --comps  name=group[*weight]+group[*weight]...     sequence component = weighted mean of the networks' row-standardised
                                                     matrices (the weight applies to every network of the group)
  --ref    the composition the others are compared with (paired bootstrap over candidates)
Reported per pool and for both pools together: score (e^-(rank-1) over the top five) of every composition for n = 1 (every
candidate game against the other nine) and n = 3 (three games against the other seven), the mean single-network score of
every group, and the differences from --ref.
"""
import argparse
import itertools
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
from pid_v2_offcand import matrices, per_question  # noqa: E402


def expand(pattern):
    parts = re.split(r"\{([^{}]*\|[^{}]*)\}", pattern)  # {a|b|c} lists; {pool} has no bar and stays
    lists = [p.split("|") if i % 2 else [p] for i, p in enumerate(parts)]
    return ["".join(x) for x in itertools.product(*lists)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--comps", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    groups = {x.split("=", 1)[0]: [f for p in x.split("=", 1)[1].split(",") for f in expand(p)] for x in a.groups}
    comps = {x.split("=", 1)[0]: [(t.split("*")[0], float(t.split("*")[1]) if "*" in t else 1.0) for t in x.split("=", 1)[1].split("+")] for x in a.comps}
    for g, fs in groups.items():
        print(f"group {g}: {len(fs)} networks", flush=True)
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float64))))(np.load(f))  # noqa: E731
    out = {"groups": groups, "comps": {k: v for k, v in comps.items()}, "ref": a.ref, "pools": {}, "both": {}}
    keep = {}  # (n, comp) -> list over pools of (per-question scores, cluster ids)
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _, _, _ = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        keys = [k for p in players for k, _ in cands[p]]
        owner = np.array([i for i, p in enumerate(players) for _ in cands[p]])
        C = len(players)
        vec = {g: [load(f.replace("{pool}", pool)) for f in fs] for g, fs in groups.items()}
        res = {}
        for n in (1, 3):
            Z, truth = {}, None
            for g, vs in vec.items():
                Z[g] = []
                for v_ in vs:
                    M, truth = matrices(np.stack([v_[k] for k in keys]), owner, C, n)
                    Z[g].append(M)
            single = {g: float(np.mean([per_question(M, truth)[0].mean() for M in Ms])) for g, Ms in Z.items()}
            sc = {}
            for name, terms in comps.items():
                mats = [M for g, w in terms for M in Z[g]]
                ws = [w for g, w in terms for _ in Z[g]]
                sc[name] = per_question(np.average(mats, axis=0, weights=ws), truth)[0]
                keep.setdefault((n, name), []).append((sc[name], np.char.add(pool + "|", truth.astype(str))))
            res[str(n)] = {"single_by_group": single, "score": {k: float(x.mean()) for k, x in sc.items()}}
            print(f"{pool} n={n}: single network by group {({g: round(x, 4) for g, x in single.items()})}", flush=True)
            print("   " + "  ".join(f"{k} {x.mean():.4f}" for k, x in sc.items()), flush=True)
        out["pools"][pool] = res
        print(f"{pool} done ({time.time() - t0:.0f} s)", flush=True)
    for n in (1, 3):
        ref = np.concatenate([x for x, _ in keep[(n, a.ref)]])
        cl = np.concatenate([c for _, c in keep[(n, a.ref)]])
        ids, inv = np.unique(cl, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(ids))
        out["both"][str(n)] = {}
        for name in comps:
            x = np.concatenate([y for y, _ in keep[(n, name)]])
            d = x - ref
            per = np.bincount(inv, weights=d, minlength=len(ids)) / cnt
            bs = per[rng.integers(0, len(ids), (a.boot, len(ids)))].mean(1)
            out["both"][str(n)][name] = {"score": float(x.mean()), "diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            if name != a.ref:
                print(f"both pools n={n}: {name} {x.mean():.4f}  vs {a.ref} {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]", flush=True)
            else:
                print(f"both pools n={n}: {name} {x.mean():.4f}  (reference)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
