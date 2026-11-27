"""Remaining errors of fusion3o + assignment on ONE-QUESTION-PER-PLAYER pools (the official structure; the first analysis
used make_pool, several questions per player, where the assignment is wrong by construction). Seeds 0-2 x two fold sets.
Per wrong question: n, truth rank under fusion3o, margin, rank band truth -> predicted, component ranks of the truth
(opening/style/local/orient), and whether the question's raw first moves match the truth's modal first move."""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.components import ALL_COMPONENTS, features_all, fit_from_parsed, fit_hd, fit_local, fit_orient, parse_all, score_components  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.orient import orient_tokens  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402

R2 = "/home/u2507115/BK_AICUP_2026-2"
doc = json.load(open(f"{R2}/results/pid_fusion3o.json")); cfg = doc["cfg"]
folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
games = pools.black_games(aicup_pid.TRAIN_DIR)
rank_of = dict(zip(folds_df.player_id, folds_df["rank"]))
tot = Counter(); rows = []
for folds in ([1, 2], [3, 4]):
    for seed in (0, 1, 2):
        cands, qs, info = pools.make_pool_one_per_player(games, folds_df, folds, n_players=400, cand_games=100, seed=seed, pool_id=f"p{folds[0]}{folds[1]}s{seed}")
        keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
        sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
        moves, feats = parse_all(sgfs, 12), features_all(sgfs, 12)
        by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
        opening, style = fit_from_parsed(by_player, moves, feats, cfg)
        local, hd, orient = fit_local(by_player, moves, cfg), fit_hd(by_player, sgfs, cfg), fit_orient(by_player, moves, cfg)
        q_keys = [list(k) for k in qs["game_keys"]]
        S = score_components(ALL_COMPONENTS, opening, style, q_keys, moves, feats, local, hd, sgfs, orient)
        truth = np.array([opening.index[p] for p in qs["player_id"]]); players = opening.players
        fus = Fusion(); fus.w = np.array(doc["weights"]); S_f = fus.scores([S[c] for c in doc["components"]])
        idx_p, idx_a = topk(S_f, 5), assign_topk(S_f, 5)
        n = qs["n_games"].to_numpy()
        order = np.argsort(-S_f, axis=1)
        rank_truth = np.array([int(np.where(order[q] == truth[q])[0][0]) + 1 for q in range(len(truth))])
        wp, wa = idx_p[:, 0] != truth, idx_a[:, 0] != truth
        sd = S_f.std(axis=1)
        margin = (S_f[np.arange(len(truth)), idx_a[:, 0]] - S_f[np.arange(len(truth)), truth]) / sd
        comp_rank = {c: np.array([int((S[c][q] > S[c][q, truth[q]]).sum()) + 1 for q in range(len(truth))]) for c in ALL_COMPONENTS}
        modal = {p: Counter(t[1] for k in ks for t in orient_tokens(moves[k], 12) if t[0] == "first").most_common(1)[0][0] for p, ks in by_player.items()}
        score_p = np.mean([np.exp(-list(idx_p[q]).index(truth[q])) if truth[q] in idx_p[q] else 0 for q in range(len(truth))])
        score_a = np.mean([np.exp(-list(idx_a[q]).index(truth[q])) if truth[q] in idx_a[q] else 0 for q in range(len(truth))])
        print(f"== folds {folds} seed {seed}: {len(truth)} questions; score plain {score_p:.4f} assign {score_a:.4f}; wrong Top-1 plain {wp.sum()}, after assignment {wa.sum()} (fixed {(wp & ~wa).sum()}, broken {(~wp & wa).sum()}); wrong by n {dict(sorted(Counter(n[wa].tolist()).items()))}; truth rank {dict(sorted(Counter(rank_truth[wa].tolist()).items()))}")
        tot.update({"questions": len(truth), "wrong_plain": int(wp.sum()), "wrong_assign": int(wa.sum())}); tot.update({f"n{k}": v for k, v in Counter(n[wa].tolist()).items()})
        for q in np.where(wa)[0]:
            tru, pred = players[truth[q]], players[idx_a[q, 0]]
            firsts = [t[1] for k in q_keys[q] for t in orient_tokens(moves[k], 12) if t[0] == "first"]
            match_t = sum(f == modal[tru] for f in firsts); match_p = sum(f == modal[pred] for f in firsts)
            rows.append({"n": int(n[q]), "rank": int(rank_truth[q]), "margin": float(margin[q]), "truth_band": rank_of.get(tru), "pred_band": rank_of.get(pred),
                         **{f"r_{c}": int(comp_rank[c][q]) for c in ALL_COMPONENTS}, "q_first_match_truth": f"{match_t}/{len(firsts)}", "q_first_match_pred": f"{match_p}/{len(firsts)}"})
            print(f"   n={n[q]:2d} rank {rank_truth[q]:3d} margin {margin[q]:5.2f} {rank_of.get(tru,'?'):>2s}->{rank_of.get(pred,'?'):<2s} comps o/s/l/hd/or " + "/".join(str(comp_rank[c][q]) for c in ALL_COMPONENTS)
                  + f"  first-move match truth {match_t}/{len(firsts)} pred {match_p}/{len(firsts)}")
print("TOTAL", dict(tot))
df = pd.DataFrame(rows)
print("truth rank distribution of the wrong answers:", dict(sorted(Counter(df["rank"]).items())))
print("component rank of the truth among the wrong answers, median:", {c: float(df[f"r_{c}"].median()) for c in ALL_COMPONENTS})
print("share of wrong answers where the truth is top-3 in opening:", round(float((df.r_opening <= 3).mean()), 3), "| in style:", round(float((df.r_style <= 3).mean()), 3), "| in orient:", round(float((df.r_orient <= 3).mean()), 3))
print("same rank band truth/pred:", round(float((df.truth_band == df.pred_band).mean()), 3))
df.to_csv(f"{R2}/results/checks/error_analysis_f3o_assign.csv", index=False)
