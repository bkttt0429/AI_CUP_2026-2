"""Player-level bootstrap of a paired difference of per-question scores (the candidates are the resampling unit)."""
import numpy as np


def paired_player_bootstrap(s_a, s_b, player_ids, n=2000, seed=42):
    """{diff, ci95, n_players, n_questions} of mean(s_a - s_b): players resampled with replacement, all their questions kept."""
    s_a, s_b, pid = np.asarray(s_a, float), np.asarray(s_b, float), np.asarray(player_ids).astype(str)
    d = s_a - s_b
    players = np.unique(pid)
    groups = [np.flatnonzero(pid == p) for p in players]
    sums = np.array([d[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups], dtype=float)
    rng = np.random.RandomState(seed)
    reps = np.empty(n)
    for i in range(n):
        pick = rng.randint(0, len(players), len(players))
        reps[i] = sums[pick].sum() / counts[pick].sum()
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "n_players": int(len(players)), "n_questions": int(len(d))}
