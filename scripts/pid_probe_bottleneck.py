#!/usr/bin/env python
"""What is the remaining error made of? (METHOD_NOTES 2.139) - measurements only, nothing is chosen here.

Official pseudo-pools (per candidate 3 of the 10 reference games held out, true answers; four hold-out seeds, both added
pools), the fused scores of one fusion document:
  A  where the owner of a 1-game question ends up: Top-1 / 2 / 5 / 10 / 20 / 50 / 100 and the mean score
  B  is it the PLAYER or the GAME? per candidate the number of his 1-game questions answered right, against what equal
     chances would give (variance ratio; share of candidates never / always right)
  C  who gets the wrong answers: hubs (candidates that collect many foreign questions) and mutual pairs (a's games go to b
     and b's to a)
  D  by the game: subject colour, handicap, length
  E  do the network groups agree on the wrong candidate (a real look-alike) or scatter (noise)?
  F  how many reference games would be enough: the same questions against 1 / 2 / 4 / 7 of the reference games
  G  how many candidates: the same scores ranked inside random sub-pools of 100 / 200 / 300 / 450 candidates
  H  owner against the best stranger in row-standard units (mean, spread, how much of the spread is the player's)
  I  2- and 3-game questions, and whether summing the fused scores of the single games does as well as pooling the games
  J  the real pool, no answers: expected score of the real questions by size from the margin -> score relation of the
     pseudo-pool (before the one-to-one assignment and the prior of the file)
  python -I <this> --doc v2o19d_j --out results/final/v2_probe_bottleneck_v2o19d_j.json
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_aug import fuse_v  # noqa: E402
from pid_probe_qq import Pool, unit  # noqa: E402

G = {}
GROUPS = {"by-games": ("o19F", "o21F", "o22F", "o23F", "o24F", "o25F"), "js uniform": ("o16F", "o17F", "o18F", "o14F"), "first set": ("o12F", "o13F", "o12K", "o13X")}
RANKS = (1, 2, 5, 10, 20, 50, 100)


def family(tag):
    return re.fullmatch(r"_?(.*?)(s\d+)?(k\d+)?e?", tag).group(1)


def group_of(fam):
    for g, fs in GROUPS.items():
        if fam in fs:
            return g
    return "plain uniform"


def score_of(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def ranks_of(F, own):
    return (F > F[np.arange(len(F)), own][:, None]).sum(1)


def seq_groups(P, ref, queries, vecs, row, fams):
    """Per network group the weighted mean of the row-standardised cosine matrices [Q, C] (the sequence component of that group alone)."""
    names = sorted(ref)

    def member(sets):
        r = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        c = np.array([row[k] for s in sets for k, _ in s])
        return sparse.csr_matrix((np.ones(len(c), dtype=np.float32), (r, c)), shape=(len(sets), len(row)))
    A, B = member([ref[p] for p in names]), member(queries)
    acc, wsum = {}, {}
    for V, w, f in zip(vecs, P.wts, fams):
        g = group_of(f)
        acc[g] = acc.get(g, 0.0) + w * standardise(unit(np.asarray(B @ V, dtype=np.float64)) @ unit(np.asarray(A @ V, dtype=np.float64)).T)
        wsum[g] = wsum.get(g, 0) + w
    return {g: acc[g] / wsum[g] for g in acc}, wsum


def plies(sgf):
    return sgf.count(";B[") + sgf.count(";W[")


def handicap(sgf):
    m = re.search(r"HD\[(\d+)\]", sgf)
    return int(m.group(1)) if m else 0


def one_seed(task):
    s, ft_dir, pool, ks, sizes = task
    P = G["P"]
    C = P.C
    rng = np.random.default_rng(7300 + 17 * s + (0 if pool == "pub2" else 1))
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    one = [[g] for p in P.players for g in hold[p]]  # three 1-game questions per candidate
    own = np.repeat(np.arange(C), 3)
    two, three = [list(hold[p][:2]) for p in P.players], [list(hold[p]) for p in P.players]
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    vecs, fams = [], []
    for key in P.net_keys:
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        vecs.append(np.stack([d[k] for k in ckeys]))
        fams.append(family(tg))
    F1 = fuse_v(P, ref, one, vecs, row)
    r1 = ranks_of(F1, own)
    out = {"seed": s, "rank": r1, "own": own, "key": np.array([q[0][0] for q in one])}
    # H: owner against the best stranger, row-standard units of the fused score
    Z = standardise(F1)
    zo = Z[np.arange(len(Z)), own]
    Zi = Z.copy()
    Zi[np.arange(len(Z)), own] = -np.inf
    out["z_own"], out["z_best_other"] = zo, Zi.max(1)
    top2 = np.sort(F1, axis=1)[:, -2:]
    out["margin"], out["top1"] = top2[:, 1] - top2[:, 0], F1.argmax(1)
    # D: the game
    out["colour"] = np.array([q[0][1] for q in one])
    out["plies"] = np.array([plies(P.sgfs[q[0][0]]) for q in one])
    out["hd"] = np.array([handicap(P.sgfs[q[0][0]]) for q in one])
    # E: the groups' own Top-1
    Sg, wsum = seq_groups(P, ref, one, vecs, row, fams)
    out["group_top1"] = {g: M.argmax(1) for g, M in Sg.items()}
    out["group_rank"] = {g: ranks_of(M, own) for g, M in Sg.items()}
    out["group_weight"] = wsum
    # F: fewer reference games (the first question of every candidate)
    first = np.arange(0, 3 * C, 3)
    q_first = [one[i] for i in first]
    out["by_refs"] = {}
    for k in ks:
        if k >= 7:
            rk = r1[first]
        else:
            sub = {p: [ref[p][i] for i in rng.permutation(len(ref[p]))[:k]] for p in P.players}
            try:
                rk = ranks_of(fuse_v(P, sub, q_first, vecs, row), np.arange(C))
            except Exception as e:  # a counting model that cannot be fitted on so few games
                print(f"  seed {s}: {k} reference game(s) could not be scored ({type(e).__name__}: {str(e)[:80]})", flush=True)
                out["by_refs"][k] = [float("nan"), float("nan")]
                continue
        out["by_refs"][k] = [float(score_of(rk).mean()), float((rk == 0).mean())]
    # G: fewer candidates (rank inside random sub-pools that hold the owner)
    out["by_pool"] = {}
    for m in sizes:
        sc, hit = [], []
        for _ in range(4):
            keep = rng.random((len(F1), C)) < (m - 1) / (C - 1)
            keep[np.arange(len(F1)), own] = True
            rk = ((F1 > F1[np.arange(len(F1)), own][:, None]) & keep).sum(1)
            sc.append(score_of(rk).mean()); hit.append((rk == 0).mean())
        out["by_pool"][m] = [float(np.mean(sc)), float(np.mean(hit))]
    # I: 2- and 3-game questions, pooled games against summed single-game scores
    F2, F3 = fuse_v(P, ref, two, vecs, row), fuse_v(P, ref, three, vecs, row)
    cand = np.arange(C)
    r2, r3 = ranks_of(F2, cand), ranks_of(F3, cand)
    S2, S3 = F1[first] + F1[first + 1], F1[first] + F1[first + 1] + F1[first + 2]
    out["multi"] = {"2 pooled": [float(score_of(r2).mean()), float((r2 == 0).mean())], "2 summed": [float(score_of(ranks_of(S2, cand)).mean()), float((ranks_of(S2, cand) == 0).mean())],
                    "3 pooled": [float(score_of(r3).mean()), float((r3 == 0).mean())], "3 summed": [float(score_of(ranks_of(S3, cand)).mean()), float((ranks_of(S3, cand) == 0).mean())]}
    for name, Fm, rm in (("2", F2, r2), ("3", F3, r3)):
        t2 = np.sort(Fm, axis=1)[:, -2:]
        out[f"margin{name}"], out[f"score{name}"] = t2[:, 1] - t2[:, 0], score_of(rm)
    return s, out


def binned(margin, score, bins=12):
    """Monotone margin -> expected score table: quantile bins of the margin, mean score per bin."""
    q = np.quantile(margin, np.linspace(0, 1, bins + 1))
    mid, val = [], []
    for a, b in zip(q[:-1], q[1:]):
        m = (margin >= a) & (margin <= b)
        mid.append(float(margin[m].mean())); val.append(float(score[m].mean()))
    return np.array(mid), np.array(val)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--refs", nargs="+", type=int, default=[1, 2, 4, 7])
    ap.add_argument("--pools", nargs="+", type=int, default=[100, 200, 300, 450, 600])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    torch.set_num_threads(1)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    out = {"doc": a.doc, "pools": {}}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        C = P.C
        G.update(P=P)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            got = dict(ex.map(one_seed, [(s, a.ft_dir, pool, a.refs, a.pools) for s in a.seeds]))
        cat = lambda k: np.concatenate([got[s][k] for s in a.seeds])  # noqa: E731
        rank, own, top1 = cat("rank"), cat("own"), cat("top1")
        sc, hit = score_of(rank), rank == 0
        res = {"questions": int(len(rank)), "score": float(sc.mean()), "top1": float(hit.mean())}
        print(f"\n===== {pool}: {len(rank)} one-game questions of {C} candidates over seeds {a.seeds}; score {sc.mean():.4f}, Top-1 {hit.mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        # A
        res["owner_within"] = {str(k): float((rank < k).mean()) for k in RANKS}
        lost = 1.0 - sc
        res["loss_share_by_rank"] = {"2-5": float(lost[(rank >= 1) & (rank < 5)].sum() / lost.sum()), "6-20": float(lost[(rank >= 5) & (rank < 20)].sum() / lost.sum()), "beyond 20": float(lost[rank >= 20].sum() / lost.sum())}
        print("A  owner within the first k: " + "  ".join(f"{k}: {v:.3f}" for k, v in res["owner_within"].items())
              + f" | share of the lost score: owner at 2-5 {res['loss_share_by_rank']['2-5']:.2f}, at 6-20 {res['loss_share_by_rank']['6-20']:.2f}, beyond 20 {res['loss_share_by_rank']['beyond 20']:.2f}", flush=True)
        # B
        n_q = np.bincount(own, minlength=C)
        n_hit = np.bincount(own, weights=hit.astype(float), minlength=C)
        p = hit.mean()
        var_obs, var_bin = float(np.var(n_hit / n_q)), float(p * (1 - p) / n_q.mean())
        res["per_candidate"] = {"questions_each": float(n_q.mean()), "variance_observed": var_obs, "variance_if_equal_chances": var_bin, "ratio": var_obs / var_bin,
                                "never_right": float((n_hit == 0).mean()), "always_right": float((n_hit == n_q).mean()),
                                "at_most_a_quarter_right": float((n_hit / n_q <= 0.25).mean()), "at_least_three_quarters_right": float((n_hit / n_q >= 0.75).mean()),
                                "share_of_errors_from_worst_fifth": float(np.sort(n_q - n_hit)[::-1][:C // 5].sum() / (n_q - n_hit).sum())}
        pc = res["per_candidate"]
        # the same game is held out in several seeds: one outcome per DISTINCT game (mean over its appearances), then games within players
        keys = cat("key")
        uk, inv = np.unique(keys, return_inverse=True)
        y = np.bincount(inv, weights=hit.astype(float)) / np.bincount(inv)
        yo = np.zeros(len(uk), dtype=int)
        yo[inv] = own
        m_i = np.bincount(yo, minlength=C)
        mean_i = np.bincount(yo, weights=y, minlength=C) / np.maximum(m_i, 1)
        msw = float(((y - mean_i[yo]) ** 2).sum() / max(len(y) - C, 1))
        msb = float((m_i * (mean_i - y.mean()) ** 2).sum() / (C - 1))
        n0 = float((len(y) - (m_i ** 2).sum() / len(y)) / (C - 1))
        between = max((msb - msw) / n0, 0.0)
        consistent = float(np.mean((y == 0) | (y == 1)))
        pc.update({"distinct_games": int(len(y)), "games_per_candidate": float(m_i.mean()), "same_game_same_outcome_across_seeds": consistent, "variance_between_players": between,
                   "variance_between_games_of_a_player": msw, "share_of_variance_that_is_the_player": between / (between + msw), "sd_of_players_own_rates": float(np.sqrt(between))})
        print(f"B  per candidate ({pc['questions_each']:.0f} questions each, {pc['games_per_candidate']:.1f} distinct games; a game held out in several seeds gets the same outcome {consistent:.2f} of the time): "
              f"of the variance of a game's outcome {pc['share_of_variance_that_is_the_player']:.2f} is the player's (sd of the players' own hit rates {np.sqrt(between):.3f}), the rest is the game; "
              f"never right {pc['never_right']:.3f}, <= 1/4 right {pc['at_most_a_quarter_right']:.3f}, >= 3/4 right {pc['at_least_three_quarters_right']:.3f}, "
              f"always right {pc['always_right']:.3f}; the worst fifth of the candidates hold {pc['share_of_errors_from_worst_fifth']:.2f} of the errors", flush=True)
        # C
        wrong = ~hit
        hub = np.bincount(top1[wrong], minlength=C)
        pairs = {}
        for o_, t_ in zip(own[wrong], top1[wrong]):
            pairs[(int(o_), int(t_))] = pairs.get((int(o_), int(t_)), 0) + 1
        mutual = sum(v for (o_, t_), v in pairs.items() if (t_, o_) in pairs)
        repeated = sum(v for v in pairs.values() if v >= 2)
        res["wrong_answers"] = {"errors": int(wrong.sum()), "top_5pct_hubs_share": float(np.sort(hub)[::-1][:C // 20].sum() / max(wrong.sum(), 1)), "max_hub": int(hub.max()),
                                "hub_expected_if_uniform": float(wrong.sum() / C), "share_in_mutual_pairs": float(mutual / max(wrong.sum(), 1)),
                                "share_repeated_same_wrong_candidate": float(repeated / max(wrong.sum(), 1)), "distinct_owner_thief_pairs": int(len(pairs))}
        w_ = res["wrong_answers"]
        print(f"C  {w_['errors']} wrong Top-1: the 5% most-chosen candidates take {w_['top_5pct_hubs_share']:.2f} of them (largest {w_['max_hub']}, uniform would give {w_['hub_expected_if_uniform']:.1f} each); "
              f"in mutual pairs (a->b and b->a) {w_['share_in_mutual_pairs']:.2f}; the same owner sent to the same wrong candidate twice or more {w_['share_repeated_same_wrong_candidate']:.2f}", flush=True)
        # D
        colour, npl, hd = cat("colour"), cat("plies"), cat("hd")
        res["by_game"] = {"colour": {str(c): [int((colour == c).sum()), float(sc[colour == c].mean())] for c in np.unique(colour)},
                          "handicap": {str(h): [int((hd == h).sum()), float(sc[hd == h].mean())] for h in np.unique(hd)}, "plies": {}}
        edges = [0, 100, 150, 200, 250, 10 ** 6]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (npl >= lo) & (npl < hi)
            if m.sum():
                res["by_game"]["plies"][f"{lo}-{hi if hi < 10 ** 6 else ''}"] = [int(m.sum()), float(sc[m].mean())]
        print("D  score by colour " + "  ".join(f"{k}: {v[1]:.3f} (n {v[0]})" for k, v in res["by_game"]["colour"].items()) + " | handicap "
              + "  ".join(f"{k}: {v[1]:.3f} (n {v[0]})" for k, v in res["by_game"]["handicap"].items()) + " | plies "
              + "  ".join(f"{k}: {v[1]:.3f} (n {v[0]})" for k, v in res["by_game"]["plies"].items()), flush=True)
        # E
        gnames = sorted(got[a.seeds[0]]["group_top1"])
        gt = {g: np.concatenate([got[s]["group_top1"][g] for s in a.seeds]) for g in gnames}
        gr = {g: np.concatenate([got[s]["group_rank"][g] for s in a.seeds]) for g in gnames}
        res["groups"] = {g: {"weight": int(got[a.seeds[0]]["group_weight"][g]), "score_alone": float(score_of(gr[g]).mean()), "top1_alone": float((gr[g] == 0).mean())} for g in gnames}
        same_wrong = np.sum([gt[g][wrong] == top1[wrong] for g in gnames], axis=0)
        owner_votes = np.sum([gt[g][wrong] == own[wrong] for g in gnames], axis=0)
        res["agreement_on_errors"] = {"groups": len(gnames), "all_groups_pick_the_same_wrong": float((same_wrong == len(gnames)).mean()), "mean_groups_on_the_wrong": float(same_wrong.mean()),
                                      "some_group_picks_the_owner": float((owner_votes > 0).mean()), "no_group_picks_the_owner_or_the_wrong": float(((owner_votes == 0) & (same_wrong == 0)).mean())}
        any_right = np.any([gr[g] == 0 for g in gnames], axis=0)
        res["agreement_on_errors"]["oracle_any_group_right"] = float(any_right.mean())
        ag = res["agreement_on_errors"]
        print("E  groups alone: " + "  ".join(f"{g} (weight {v['weight']}) {v['score_alone']:.3f}" for g, v in res["groups"].items())
              + f" | on the fused errors: all {ag['groups']} groups pick the same wrong candidate {ag['all_groups_pick_the_same_wrong']:.2f}, mean groups on it {ag['mean_groups_on_the_wrong']:.2f}, "
              f"some group picks the owner {ag['some_group_picks_the_owner']:.2f}; any group right over all questions {ag['oracle_any_group_right']:.3f} (fused {hit.mean():.3f})", flush=True)
        # F, G
        res["by_reference_games"] = {str(k): [float(np.nanmean([got[s]["by_refs"][k][i] for s in a.seeds])) for i in (0, 1)] for k in a.refs}
        res["by_candidates"] = {str(m): [float(np.mean([got[s]["by_pool"][m][i] for s in a.seeds])) for i in (0, 1)] for m in a.pools}
        print("F  score by reference games per candidate (first question of each): " + "  ".join(f"{k}: {v[0]:.3f}" for k, v in res["by_reference_games"].items())
              + " | G  score by number of candidates: " + "  ".join(f"{k}: {v[0]:.3f}" for k, v in res["by_candidates"].items()), flush=True)
        # H
        zo, zb = cat("z_own"), cat("z_best_other")
        mean_p = np.bincount(own, weights=zo, minlength=C) / n_q
        within = float(np.mean([np.var(zo[own == c_]) for c_ in range(C)]) * n_q.mean() / (n_q.mean() - 1))
        total = float(np.var(zo))
        res["owner_vs_best_other"] = {"owner_mean": float(zo.mean()), "owner_sd": float(zo.std()), "best_other_mean": float(zb.mean()), "best_other_sd": float(zb.std()),
                                      "owner_variance_within_player": within, "owner_variance_between_players": max(total - within, 0.0), "sd_of_player_means": float(mean_p.std())}
        h_ = res["owner_vs_best_other"]
        print(f"H  row-standard units: owner {h_['owner_mean']:.2f} (sd {h_['owner_sd']:.2f}), best of the others {h_['best_other_mean']:.2f} (sd {h_['best_other_sd']:.2f}); "
              f"owner's variance: game to game {h_['owner_variance_within_player']:.2f}, player to player {h_['owner_variance_between_players']:.2f}", flush=True)
        # I
        res["multi"] = {k: [float(np.mean([got[s]["multi"][k][i] for s in a.seeds])) for i in (0, 1)] for k in got[a.seeds[0]]["multi"]}
        print("I  " + "  ".join(f"{k}: {v[0]:.3f}" for k, v in res["multi"].items()), flush=True)
        # J: the real pool by size, through the pseudo-pool's margin -> score relation
        F = fuse_v(P, P.ref, P.qs, P.vec["t50"], P.row)
        n = np.array([len(q) for q in P.qs])
        t2 = np.sort(F, axis=1)[:, -2:]
        mg = t2[:, 1] - t2[:, 0]
        cal = {1: binned(cat("margin"), sc), 2: binned(cat("margin2"), cat("score2")), 3: binned(cat("margin3"), cat("score3"))}
        res["real"] = {}
        for k in (1, 2, 3, 5):
            mid, val = cal[min(k, 3)]
            est = np.interp(mg[n == k], mid, val)
            res["real"][str(k)] = {"questions": int((n == k).sum()), "median_margin": float(np.median(mg[n == k])), "pseudo_median_margin": float(np.median(cat("margin" if k == 1 else f"margin{min(k, 3)}"))),
                                   "expected_score": float(est.mean())}
        print("J  real questions, expected score from the pseudo-pool's margin -> score relation (no assignment, no prior): "
              + "  ".join(f"n={k}: {v['expected_score']:.3f} (median margin {v['median_margin']:.2f}, pseudo {v['pseudo_median_margin']:.2f})" for k, v in res["real"].items())
              + f" | mean of the four sizes {np.mean([v['expected_score'] for v in res['real'].values()]):.4f}", flush=True)
        out["pools"][pool] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"\nwrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
