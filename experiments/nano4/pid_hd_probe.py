"""Is the HD[1] flag player-specific? Per-candidate HD1 rate across the 400 players of each pool (label-free: candidate
player ids are given), and how concentrated it is compared with a binomial null at the pool rate."""
import re, sys
import numpy as np, pandas as pd
hd = re.compile(r"HD\[(\d+)\]")
R2 = "/home/u2507115/BK_AICUP_2026-2"
for pool, cf, qf in (("pub_", "player_identification_candidates_public.csv", "player_identification_test_public.csv"),
                     ("priv_", "player_identification_candidates_private.csv", "player_identification_test_private.csv")):
    c = pd.read_csv(f"{R2}/data/tests/{cf}", dtype=str, keep_default_na=False)
    c["hd1"] = c.sgf_content.map(lambda s: int(hd.search(s).group(1) != "0"))
    rate = c.groupby("player_id").hd1.mean()
    p = c.hd1.mean()
    rng = np.random.default_rng(0)
    null = rng.binomial(100, p, size=(2000, 400)) / 100
    print(pool, f"pool HD1 rate {p:.3f}; per-player rate quantiles 0/10/25/50/75/90/100%:", np.round(rate.quantile([0, .1, .25, .5, .75, .9, 1]).values, 2).tolist())
    print(pool, f"players with rate 0: {int((rate == 0).sum())}, >=0.5: {int((rate >= 0.5).sum())}; sd across players {rate.std():.3f} vs binomial null sd {null.std(axis=1).mean():.3f}")
    # mutual information proxy: how many bits does one game's HD flag give about the player, vs log2(400)=8.64
    pr = rate.values; w = np.full(400, 1 / 400)
    def H(x): x = x[x > 0]; return float(-(x * np.log2(x)).sum())
    p1 = (w * pr).sum(); post1 = w * pr / p1; post0 = w * (1 - pr) / (1 - p1)
    print(pool, f"information from one game's HD flag about the player: {H(w) - (p1 * H(post1) + (1 - p1) * H(post0)):.3f} bits of {np.log2(400):.2f}")
    q = pd.read_csv(f"{R2}/data/tests/{qf}", dtype=str, keep_default_na=False)
    qrates = []
    for _, r in q.iterrows():
        n = int(r.num_games); qrates.append(np.mean([int(hd.search(r[f"sgf_{i}"]).group(1) != "0") for i in range(1, n + 1)]))
    print(pool, "question HD1 rate quantiles:", np.round(np.quantile(qrates, [0, .25, .5, .75, 1]), 2).tolist(), "questions with rate >= 0.5:", int(sum(x >= 0.5 for x in qrates)))
