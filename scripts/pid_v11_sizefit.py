#!/usr/bin/env python3
"""METHOD_NOTES 2.242: are fusion weights fitted PER QUESTION SIZE (1 / 2 / 3 games) worth more than one common weight vector?
A diagnostic on seen hold-out seeds, from the split files of scripts/pid_v2_offsim_par.py (--only POOL:SEED --dump).
Per pool the candidates are cut in two halves (the tool's own draw); weights are fitted on the primary questions of one half
and scored on ALL questions of the other half, both ways:
  common   one weight vector for all sizes (the tool's own calibration)
  by size  one weight vector per question size, fitted on that size's questions and applied to that size
Reported: by-size minus common (the value of size-specific weights), and both against the document's weights; size-balanced
mean, bootstrap over candidates as pid_v2_offsim_par.py.
  python -I scripts/pid_v11_sizefit.py --fusion results/pid_fusion_w11c.json --dump <dir> --seeds 0 1 2 3 --out results/final/v11_sizefit_w11c.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup_pid.fusion import Fusion  # noqa: E402
from pid_v2_offsim_par import SIZES, boot_mix, per_question  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    rng = np.random.default_rng(0)
    fdoc = Fusion()
    fdoc.w = np.asarray(doc["weights"], dtype=float)
    out = {"document": Path(a.fusion).name, "seeds": a.seeds, "pools": {}}
    alld = {"size_vs_common": [], "common_vs_doc": [], "size_vs_doc": []}
    alln, allc = [], []
    for pool in ("pub2", "priv2"):
        rs = []
        for s in a.seeds:
            with open(Path(a.dump) / f"split_{pool}_{s}.pkl", "rb") as fh:
                rs.append(pickle.load(fh))
        H = rng.permutation(rs[0]["players"]) % 2
        mats = lambda r: [r["Z"][("seq", a.step)] if c == "seq" else r["Z"][c] for c in comps]  # noqa: E731
        d = {k: [] for k in alld}
        nn, cl, wts = [], [], {"common": [], **{str(m): [] for m in SIZES}}
        for side in (0, 1):
            fit = [np.flatnonzero((H[r["truth"]] == side) & r["primary"]) for r in rs]
            stack = lambda rows: ([np.vstack(col) for col in zip(*[[M[i] for M in mats(r)] for r, i in zip(rs, rows)])], np.concatenate([r["truth"][i] for r, i in zip(rs, rows)]))  # noqa: E731
            fc = Fusion().fit(*stack(fit))
            wts["common"].append(fc.w.tolist())
            fs = {}
            for m in SIZES:
                rows = [i[r["n"][i] == m] for r, i in zip(rs, fit)]
                fs[m] = Fusion().fit(*stack(rows))
                wts[str(m)].append(fs[m].w.tolist())
            for r in rs:
                te = np.flatnonzero(H[r["truth"]] != side)
                M = [x[te] for x in mats(r)]
                tr, n = r["truth"][te], r["n"][te]
                s_doc, s_com = per_question(fdoc.scores(M), tr), per_question(fc.scores(M), tr)
                s_siz = np.zeros(len(te))
                for m in SIZES:
                    k = np.flatnonzero(n == m)
                    s_siz[k] = per_question(fs[m].scores([x[k] for x in M]), tr[k])
                d["size_vs_common"].append(s_siz - s_com); d["common_vs_doc"].append(s_com - s_doc); d["size_vs_doc"].append(s_siz - s_doc)
                nn.append(n); cl.append(np.char.add(pool + "|", tr.astype(str)))
        n, c = np.concatenate(nn), np.concatenate(cl)
        res = {"weights": {k: dict(zip(comps, np.mean(v, axis=0).round(4).tolist())) for k, v in wts.items()}}
        for k in d:
            x = np.concatenate(d[k])
            b = boot_mix(x, n, c, a.boot, rng)
            b["by_n"] = {str(m): float(x[n == m].mean()) for m in SIZES}
            res[k] = b
            alld[k].append(x)
            print(f"{pool} {k}: {b['diff']:+.5f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({kk: round(v, 4) for kk, v in b['by_n'].items()})}", flush=True)
        for k, v in res["weights"].items():
            print(f"  {pool} weights fitted on halves, mean ({k}): {v}", flush=True)
        alln.append(n); allc.append(c)
        out["pools"][pool] = res
    n, c = np.concatenate(alln), np.concatenate(allc)
    out["both"] = {}
    for k in alld:
        x = np.concatenate(alld[k])
        b = boot_mix(x, n, c, a.boot, rng)
        b["by_n"] = {str(m): float(x[n == m].mean()) for m in SIZES}
        out["both"][k] = b
        print(f"both {k}: {b['diff']:+.5f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({kk: round(v, 4) for kk, v in b['by_n'].items()})}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
