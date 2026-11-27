#!/usr/bin/env python3
"""METHOD_NOTES 2.233 (recorded-only variant): scripts/pid_v10_pool.py with ONE ALTERNATIVE PER POOL - the per-question
files of <alt-pub2> are used for pub2 and those of <alt-priv2> for priv2 (fusion weights fitted on each pool's own
candidates). Same statistics and random draws as pid_v10_pool.py.

  python -I scripts/pid_v11_pool2.py --dir <folder> --ref <name> --alt-pub2 <name> --alt-priv2 <name> --seeds 12 13 ... --out <json>
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
    ap.add_argument("--alt-pub2", required=True)
    ap.add_argument("--alt-priv2", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    runs = []
    for pool, alt in (("pub2", a.alt_pub2), ("priv2", a.alt_priv2)):
        first = True
        for s in a.seeds:
            z = np.load(Path(a.dir) / f"sc_{alt}_vs_{a.ref}_{pool}_{s}.npz")
            assert str(z["pool"]) == pool and int(z["seed"]) == s and str(z["ref"]) == f"pid_fusion_{a.ref}.json" and str(z["alt"]) == f"pid_fusion_{alt}.json"
            if first:
                rng.permutation(int(z["players"]))
                first = False
            runs.append({k: z[k] for k in ("truth", "n", "sc_ref", "sc_alt")} | {"pool": pool, "seed": s})
    out = {"reference": a.ref, "alternative": {"pub2": a.alt_pub2, "priv2": a.alt_priv2}, "seeds": a.seeds, "boot": a.boot, "diff": {}}
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    for g, rs in (("pub2", [r for r in runs if r["pool"] == "pub2"]), ("priv2", [r for r in runs if r["pool"] == "priv2"]), ("both", runs)):
        n = cat(rs, lambda r: r["n"])
        cluster = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
        x, y = cat(rs, lambda r: r["sc_ref"]), cat(rs, lambda r: r["sc_alt"])
        b = boot_mix(y - x, n, cluster, a.boot, rng)
        b["by_n"] = {str(m): float((y - x)[n == m].mean()) for m in SIZES}
        b["reference"], b["alternative"] = mix(x, n), mix(y, n)
        out["diff"][g] = b
        print(f"{g}: reference {b['reference']:.4f}  per-pool weights {b['alternative']:.4f}  {b['diff']:+.5f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
