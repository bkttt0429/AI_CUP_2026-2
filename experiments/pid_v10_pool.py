#!/usr/bin/env python3
"""METHOD_NOTES 2.230: the pooled comparison of an alternative composition with a reference over MANY hold-out seeds,
from the per-split files of scripts/pid_v10_sc.py. The same numbers as the "talt vs t50" lines of
scripts/pid_v2_offsim_par.py (size-balanced mean, bootstrap over candidates, one cluster per pool and candidate), with the
same random draws in the same order (so the seeds 0-3 of a known comparison reproduce its stored result).

  python -I scripts/pid_v10_pool.py --dir <folder> --ref <name> --alt <name> --seeds 12 13 ... --out <json> [--boot 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from pid_v2_offsim_par import SIZES, boot_mix, mix  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    runs = []
    for pool in ("pub2", "priv2"):
        first = True
        for s in a.seeds:
            z = np.load(Path(a.dir) / f"sc_{a.alt}_vs_{a.ref}_{pool}_{s}.npz")
            assert str(z["pool"]) == pool and int(z["seed"]) == s and str(z["ref"]) == f"pid_fusion_{a.ref}.json" and str(z["alt"]) == f"pid_fusion_{a.alt}.json"
            if first:
                rng.permutation(int(z["players"]))  # the draw pid_v2_offsim_par.py makes per pool for its weight calibration
                first = False
            runs.append({k: z[k] for k in ("truth", "n", "sc_ref", "sc_alt")} | {"pool": pool, "seed": s})
    out = {"reference": a.ref, "alternative": a.alt, "seeds": a.seeds, "boot": a.boot, "scores": {}, "diff": {}}
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    for g, rs in (("pub2", [r for r in runs if r["pool"] == "pub2"]), ("priv2", [r for r in runs if r["pool"] == "priv2"]), ("both", runs)):
        n = cat(rs, lambda r: r["n"])
        cluster = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
        x, y = cat(rs, lambda r: r["sc_ref"]), cat(rs, lambda r: r["sc_alt"])
        out["scores"][g] = {"reference": {"mix": mix(x, n), **{str(m): float(x[n == m].mean()) for m in SIZES}}, "alternative": {"mix": mix(y, n), **{str(m): float(y[n == m].mean()) for m in SIZES}}}
        b = boot_mix(y - x, n, cluster, a.boot, rng)
        b["by_n"] = {str(m): float((y - x)[n == m].mean()) for m in SIZES}
        b["questions"], b["candidates"] = int(len(n)), int(len(np.unique(cluster)))
        out["diff"][g] = b
        print(f"{g}: reference {out['scores'][g]['reference']['mix']:.4f}  alternative {out['scores'][g]['alternative']['mix']:.4f}  "
              f"{a.alt} vs {a.ref}: {b['diff']:+.5f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}  ({b['questions']} questions, {b['candidates']} candidates)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
