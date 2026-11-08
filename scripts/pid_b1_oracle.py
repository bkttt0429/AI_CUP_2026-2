#!/usr/bin/env python
"""B1 step 0 (docs/2026-11-08-public-truth-b1-check.md): upper bound of a strength component, with TRUE ranks (oracle).

On the final-CV pools (one question per player, fold pairs 1,2 / 3,4 x seeds), take a fusion document's Top-1 picks with
and without the assignment and ask, for every wrong pick, how far the picked candidate's rank is from the truth's rank
(10 ordered classes D < C < B < A < 1D < ... < 6D). The rank model agrees with a player's 100-game consensus within one
class >= 98% of the time but exactly only 55-77% (rank session, m62), so only errors with |rank gap| >= 2 are ones a real
strength component could plausibly fix; gap 0 errors are out of its reach.

  python -I scripts/pid_b1_oracle.py --fusion results/pid_fusion3o_ref_trunk_emb.json --embed-models 1,2=results/embed/embx_p12.pt \\
      3,4=results/embed/embx_p34.pt --trunk-store data/trunk/train --workers 12 --out results/b1/oracle.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
import torch  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
import pid_final_cv  # noqa: E402
from pid_cv import parse_pools  # noqa: E402

ORDER = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--embed-models", nargs="*", default=[])
    ap.add_argument("--trunk-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "train")])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.fusion).read_text())
    docs = {"main": doc}
    a.n_players, a.cand_games, a.per_size, a.order, a.sym_seed = pools.N_CANDIDATES, pools.CAND_GAMES, 100, "shuffle", None
    a.trunk_rows = load_store(*a.trunk_store)
    a.emb_models = {k: PlayerEmbedding.from_state(torch.load(v, map_location="cpu", weights_only=False)) for k, v in (e.split("=", 1) for e in a.embed_models)}
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    rank_of = games.groupby("player_id")["rank"].agg(lambda s: s.mode().iloc[0]).to_dict()
    specs = parse_pools(a.pools)
    if any(0 in f for f in specs):
        raise SystemExit("fold 0 refused")
    out = {"fusion": a.fusion, "arms": {}}
    gaps = {arm: collections.Counter() for arm in ("main", "main+assign")}
    examples = []
    for f in specs:
        for seed in a.seeds:
            r = pid_final_cv.run_pool(games, folds_df, f, seed, docs, a, ref_folds=[x for g in specs if g != f for x in g])
            for arm in gaps:
                for q, (truth, pick) in enumerate(zip(r["pid"], r["picks"][arm])):
                    if truth != pick:
                        g = abs(ORDER.index(rank_of[pick]) - ORDER.index(rank_of[truth]))
                        gaps[arm][g] += 1
                        examples.append({"arm": arm, "folds": f, "seed": seed, "n": int(r["n"][q]), "truth": truth, "pick": pick,
                                         "truth_rank": rank_of[truth], "pick_rank": rank_of[pick], "gap": g})
    for arm, c in gaps.items():
        tot = sum(c.values())
        out["arms"][arm] = {"wrong": tot, "gap_counts": {str(k): v for k, v in sorted(c.items())}, "gap0": c[0], "gap1": c[1], "gap_ge2": sum(v for k, v in c.items() if k >= 2)}
        print(f"{arm}: {tot} wrong; rank gap of the wrong pick: " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items()))
              + f"  -> reachable for a real strength component (gap >= 2): {out['arms'][arm]['gap_ge2']}", flush=True)
    out["examples"] = examples
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
