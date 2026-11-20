#!/usr/bin/env python
"""Are the fusion weights of a document still right for its (much stronger) sequence component? (METHOD_NOTES 2.214)

The document's weights are v2ge's (2026-11-13: seq 4.278, emb 0.781, ...). The cross-fits that scripts/pid_v2_offsim_par.py
stores with every comparison ("calibration": fitted on one half of a pseudo-pool's candidates) put emb at 0.04-0.15 and
seq at 3.4-3.6 for the composition v2fin4_f. A nine-weight fit per pool is noisy, so only two factors are tried here:
    emb weight x e, seq weight x l          e in --emb, l in --seq
Input: the split files of pid_v2_offsim_par.py --only POOL:SEED --dump DIR for the document (row-standardised component
matrices per split). The pair (e, l) is chosen on the --fit seeds (mean over n = 1, 2, 3 of both pools), then its paired
difference to the document's weights is reported on the --test seeds and on the --fresh seeds (bootstrap over candidates,
as offsim_par). Nothing is written but --out.   Compute node only (reads ~1 GB of pickles; CPU).
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--fit", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--test", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--fresh", nargs="*", type=int, default=[4, 5])
    ap.add_argument("--emb", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1, 0.0])
    ap.add_argument("--seq", nargs="+", type=float, default=[0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run inside a Slurm allocation (compute node)")
    import pid_v2_offsim_par as op
    from aicup_pid.fusion import Fusion
    doc = json.loads(Path(a.fusion).read_text())
    comps, w0 = doc["components"], np.asarray(doc["weights"], dtype=float)
    ie, iq = comps.index("emb"), comps.index("seq")
    runs = {}
    for pool in ("pub2", "priv2"):
        for s in sorted({*a.fit, *a.test, *a.fresh}):
            with open(Path(a.dump) / f"split_{pool}_{s}.pkl", "rb") as fh:
                runs[pool, s] = pickle.load(fh)

    def weights(e, l):
        w = w0.copy()
        w[ie] *= e
        w[iq] *= l
        return w

    def score(r, w):
        fus = Fusion()
        fus.w = w
        return op.per_question(fus.scores([r["Z"][("seq", a.step)] if c == "seq" else r["Z"][c] for c in comps]), r["truth"])

    def gather(seeds, w, pools=("pub2", "priv2")):
        rs = [(p, runs[p, s]) for p in pools for s in seeds]
        sc = np.concatenate([score(r, w) for _, r in rs])
        n = np.concatenate([r["n"] for _, r in rs])
        cl = np.concatenate([np.char.add(p + "|", r["truth"].astype(str)) for p, r in rs])
        return sc, n, cl

    rng = np.random.default_rng(0)
    base_fit, n_fit, _ = gather(a.fit, w0)
    grid = {}
    for e in a.emb:
        for l in a.seq:
            sc, _, _ = gather(a.fit, weights(e, l))
            grid[e, l] = op.mix(sc, n_fit)
    b0 = op.mix(base_fit, n_fit)
    best = max(grid, key=lambda k: (grid[k], -abs(k[0] - 1) - abs(k[1] - 1)))  # ties: the pair nearest to the document
    out = {"document": Path(a.fusion).name, "components": comps, "weights": w0.tolist(), "fit_seeds": a.fit, "test_seeds": a.test, "fresh_seeds": a.fresh,
           "fit_mix_document": b0, "fit_grid": {f"emb x{e:g}, seq x{l:g}": v for (e, l), v in grid.items()}, "chosen": {"emb": best[0], "seq": best[1], "fit_mix": grid[best], "fit_diff": grid[best] - b0}}
    print(f"document {out['document']}: weights " + ", ".join(f"{c} {x:.3f}" for c, x in zip(comps, w0)))
    print(f"fit seeds {a.fit}: document weights mix {b0:.4f}")
    for e in a.emb:
        print(f"   emb x{e:<5g}" + "  ".join(f"seq x{l:g} {grid[e, l]:.4f} ({grid[e, l] - b0:+.4f})" for l in a.seq))
    print(f"chosen on the fit seeds: emb x{best[0]:g}, seq x{best[1]:g} ({grid[best] - b0:+.4f})")
    wb = weights(*best)
    out["chosen"]["weights"] = wb.tolist()

    def report(name, seeds, w, key):
        res = {}
        for g, pools in (("pub2", ("pub2",)), ("priv2", ("priv2",)), ("both", ("pub2", "priv2"))):
            x, n, cl = gather(seeds, w, pools)
            y, _, _ = gather(seeds, w0, pools)
            b = op.boot_mix(x - y, n, cl, a.boot, rng)
            b["by_n"] = {str(m): float((x - y)[n == m].mean()) for m in sorted(set(n.tolist()))}
            b["mix"], b["mix_document"] = op.mix(x, n), op.mix(y, n)
            res[g] = b
            print(f"{name} seeds {seeds}, {g}: {b['mix_document']:.4f} -> {b['mix']:.4f}  {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}")
        out[key] = res

    report("TEST", a.test, wb, "test")
    if a.fresh:
        report("FRESH", a.fresh, wb, "fresh")
    # for the record only: the single change the cross-fits point at (emb x0.2, seq unchanged), on the same seeds
    print("-- for the record: emb x0.2 alone")
    w2 = weights(0.2, 1.0)
    report("  test", a.test, w2, "record_emb02_test")
    if a.fresh:
        report("  fresh", a.fresh, w2, "record_emb02_fresh")
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.rename(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
