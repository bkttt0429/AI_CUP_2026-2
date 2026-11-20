#!/usr/bin/env python3
"""METHOD_NOTES 2.230: the per-question scores of ONE split of scripts/pid_v2_offsim_par.py (--only POOL:SEED --dump),
in a few kilobytes instead of the split's ~160 MB of score matrices, so that the splits of a comparison can be computed on
several nodes (vectors in node-local /dev/shm) and pooled on a shared file system (scripts/pid_v10_pool.py).

Exactly the scoring of pid_v2_offsim_par.py's statistics: the reference document's weights on the split's standardised
components with its sequence component after 50 steps, the alternative document's weights with the alternative's sequence
component; per-question score = exp(-(rank - 1)) inside the top five.

  python -I scripts/pid_v10_sc.py --fusion results/pid_fusion_<ref>.json --alt results/pid_fusion_<alt>.json --split <dir>/split_<pool>_<seed>.pkl --out <file>.npz
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
from pid_v2_offsim_par import per_question  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc, alt = json.loads(Path(a.fusion).read_text()), json.loads(Path(a.alt).read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq" and list(alt["components"]) == comps
    with open(a.split, "rb") as fh:
        r = pickle.load(fh)
    mats = lambda t: [r["Z"][("seq", t)] if c == "seq" else r["Z"][c] for c in comps]  # noqa: E731
    fus, fus_alt = Fusion(), Fusion()
    fus.w, fus_alt.w = np.asarray(doc["weights"], dtype=float), np.asarray(alt["weights"], dtype=float)
    sc_ref = per_question(fus.scores(mats(50)), r["truth"])
    sc_alt = per_question(fus_alt.scores(mats("alt")), r["truth"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, pool=np.array(r["pool"]), seed=np.array(r["seed"]), players=np.array(r["players"]), truth=r["truth"], n=r["n"], primary=r["primary"],
             sc_ref=sc_ref, sc_alt=sc_alt, ref=np.array(Path(a.fusion).name), alt=np.array(Path(a.alt).name))
    tmp.rename(out)
    print(f"{r['pool']} split {r['seed']}: {len(sc_ref)} questions, reference {sc_ref.mean():.4f}, alternative {sc_alt.mean():.4f} -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
