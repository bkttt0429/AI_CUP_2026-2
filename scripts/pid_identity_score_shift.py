"""Post-discovery signed-gap decomposition on the frozen disjoint actual finalist pairs.

Registered before reading any component changes. Align single-game scores through the
mean and population SD of the other 598 candidates in the same row. For each anchor
where A ranked second to B, measure A's change and B's change on the two reserved A
games. Decompose fixed-pair binary recovery symmetrically over both possible orders of
replacing these two scores. This is arithmetic conditional on selecting an extreme
impostor. Neither hybrid score vector is the output for a real game.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, ROOT, POOLS, baseline_path, atomic_json, atomic_npz, digest_file, guard
from aicup_pid.identity_variance import bootstrap_matrix

OUT = ROOT / "results/final/identity_score_shift_20261118.json"


def main():
    guard()
    first = json.loads((ROOT / "results/final/identity_finalists_20261118.json").read_text())
    manifest = json.loads((CACHE / "inputs.json").read_text())
    all_seeds = {}
    for seed in (1, 3):
        values, groups = [], []
        for pi, pool in enumerate(POOLS):
            base, result = baseline_path(pool, seed), CACHE / f"identity_{pool}_{seed}.npz"
            if digest_file(base) != manifest["pools"][pool][str(seed)] or digest_file(result) != first["splits"][str(seed)]["archives"][pool]:
                raise ValueError("cached inputs changed")
            with np.load(base, allow_pickle=False) as z:
                scores = z["scores"]
            with np.load(result, allow_pickle=False) as z:
                pairs = z["selected"]
            for a, b in pairs:
                s = scores[3*a:3*a+3]
                others = s[:, ~np.isin(np.arange(s.shape[1]), [a, b])]
                z = (s[:, [a,b]] - others.mean(1, keepdims=True)) / others.std(1, keepdims=True)
                if z[0,0] >= z[0,1] or len(np.unique(s[0])) != s.shape[1]:
                    raise ValueError("expected strict discovery rank-two loss")
                count_as_win = lambda x, y: ((x > y) | ((x == y) & (a < b))).astype(float)
                future = count_as_win(z[1:,0], z[1:,1])
                only_a = count_as_win(z[1:,0], z[0,1])
                only_b = count_as_win(z[0,0], z[1:,1])
                a_share = .5 * (only_a + future - only_b)
                b_share = .5 * (only_b + future - only_a)
                own = z[1:,0] - z[0,0]
                other = z[0,1] - z[1:,1]
                if not np.allclose(own + other, (z[1:,0] - z[1:,1]) - (z[0,0]-z[0,1])) or not np.allclose(a_share+b_share, future):
                    raise ValueError("gap decomposition failed")
                obs = {"original_true": z[0,0], "original_rival": z[0,1], "later_true": z[1:,0].mean(), "later_rival": z[1:,1].mean(),
                       "true_score_rise": own.mean(), "rival_score_fall": other.mean(), "rival_fall_minus_true_rise": (other-own).mean(),
                       "only_true_replaced_wins": only_a.mean(), "only_rival_replaced_wins": only_b.mean(), "both_replaced_wins": future.mean(),
                       "true_recovery_share": a_share.mean(), "rival_recovery_share": b_share.mean()}
                names = list(obs)
                values.append(list(obs.values()))
                groups.append(pi)
        stats = dict(zip(names, bootstrap_matrix(values, groups, seed=111920)))
        if not np.isclose(stats["both_replaced_wins"]["mean"], first["splits"][str(seed)]["persistence"]["new_A_beats_fixed_B"]["mean"]):
            raise ValueError("failed to match frozen fixed-pair recovery")
        archive = CACHE / f"score_shift_{seed}.npz"
        atomic_npz(archive, rows=np.array(values), names=np.array(names), pools=np.array(groups))
        all_seeds[str(seed)] = {"stats": stats, "pairs": len(values), "archive": digest_file(archive)}
    atomic_json(OUT, {"code": digest_file(__file__), "selected_after_rank_outcomes": True, "splits": all_seeds})
    print("verified both gap identities and exact recovery on both splits")


if __name__ == "__main__":
    main()
