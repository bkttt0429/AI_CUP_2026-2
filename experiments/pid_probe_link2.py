#!/usr/bin/env python
"""Pair-level look at the linkage probe (METHOD_NOTES 2.166): WHO are the near-identical pairs, and what do they cost?

The first run found, for about 5 % of the added pools' candidates, another candidate of the same pool and a training
player at a cosine that is otherwise only seen for the same person (>= 0.9; different people: 99 % point 0.80). Before
anything is built on it, this script answers, from the saved training prototypes and the pools' vectors (CPU only):
  A  training players among themselves: how many have such a twin, are the pairs mutual, same rank file or not
  B  what "a known person, two unseen games" looks like (the control), and the candidates' matches in the same units
     (their ten games cut into five pairs)
  C  candidate -> training matches above the threshold: how many, one-to-one or a few hub players, mutual or not
  D  candidate <-> candidate pairs above the threshold: mutual or not, and DO THE TWO SHARE GAME RECORDS (two players of
     the same games: the networks read both sides of a game, so two frequent opponents can look alike without being one
     person); how many records of a pool appear under two candidates at all
  E  what the pairs cost: on the official pseudo-pools (fused scores of the final's composition, 1-game questions), the
     share of the questions and of the lost score that belongs to candidates with a twin, how often the first candidate
     is the owner's twin, and what a simple coupling (the twin of the first candidate is moved to the second place) does
Measurement only; no answer of a real question is used; the two sides are not mixed.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402

QS = (5, 25, 50, 75, 95, 99)
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def q(x):
    return [round(float(v), 3) for v in np.percentile(x, QS)] if len(x) else []


def score_of(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--protos", required=True)
    ap.add_argument("--ft-dir", default=str(ROOT / "data" / "seq_ft"))
    ap.add_argument("--tasks", default="/work/u2507115/pid_v2/pair_eval/tasks_v2bgm_d2_k10.pkl")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.85, 0.88, 0.9])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    T = Path(a.tests_dir)
    Z = np.load(a.protos)
    U, H, ctrl = Z["U"].astype(np.float64), Z["H"].astype(np.float64), Z["ctrl"]
    players, n_games, rank = Z["players"].astype(str), Z["n_games"], Z["rank"].astype(str)
    alias = Z["alias"].astype(str).tolist()
    K, P = U.shape[0], U.shape[1]
    out = {"thr": a.thr}
    thr = a.thr[1] if len(a.thr) > 1 else a.thr[0]
    # A  training players among themselves
    S = np.mean([U[k] @ U[k].T for k in range(K)], 0)
    np.fill_diagonal(S, -2)
    nb, bc = S.argmax(1), S.max(1)
    mutual = nb[nb] == np.arange(P)
    ri = np.array([RANKS.index(r) if r in RANKS else -9 for r in rank])
    out["training"] = {"players": int(P), "best_cos": q(bc), "by_thr": {}}
    print(f"A  {P} training players among themselves: best cosine to another player {QS} %: {q(bc)}", flush=True)
    for t in a.thr:
        m = bc >= t
        mm = m & mutual
        d = np.abs(ri[mm] - ri[nb[mm]])
        out["training"]["by_thr"][str(t)] = {"players": int(m.sum()), "mutual": int(mm.sum()), "same_rank_file": int((d == 0).sum()), "adjacent": int((d == 1).sum())}
        print(f"   cosine >= {t}: {int(m.sum())} players ({m.mean():.3f}); in mutual pairs {int(mm.sum())}; of those, partner in the same rank file {int((d == 0).sum())}, adjacent file {int((d == 1).sum())}, further {int((d > 1).sum())}; "
              f"games of the pair members (median) {int(np.median(n_games[mm])) if mm.any() else 0} (all players {int(np.median(n_games))})", flush=True)
    # B  the control: a known person, two unseen games
    Sc = np.mean([H[k] @ U[k].T for k in range(K)], 0)
    own = Sc[np.arange(len(ctrl)), ctrl]
    own_rank = (Sc > own[:, None]).sum(1)
    mu, sd = Sc.mean(1), Sc.std(1)
    c5, c25, c50 = np.percentile(own, [5, 25, 50])
    out["control2"] = {"n": int(len(ctrl)), "own_first": float((own_rank == 0).mean()), "own_cos": q(own), "own_z": q((own - mu) / sd), "stranger_best": q(np.where(np.arange(P)[None] == ctrl[:, None], -2, Sc).max(1))}
    print(f"B  control: {len(ctrl)} training players, 2 held-out games each against the prototypes: own first {(own_rank == 0).mean():.3f}; own cosine {q(own)}; best STRANGER cosine {out['control2']['stranger_best']}", flush=True)
    res_pools = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        names = sorted(cands)
        C = len(names)
        sets = [[k for k, _ in cands[p]] for p in names]
        full, pairs2 = [], []
        for al in alias:
            z = np.load(Path(a.ft_dir) / f"official_{pool}_net-1_{al}e_t0.npz")
            vec = dict(zip(z["key"].astype(str).tolist(), z["emb"].astype(np.float64)))
            full.append(unit(np.stack([np.sum([vec[k] for k in s], 0) for s in sets])))
            pairs2.append(np.stack([[unit(vec[s[2 * j]] + vec[s[2 * j + 1]]) for j in range(len(s) // 2)][:5] for s in sets if len(s) >= 10]))
        ok10 = np.array([len(s) >= 10 for s in sets])
        # C  candidates against the training players
        St = np.mean([full[k] @ U[k].T for k in range(K)], 0)
        tb, tc = St.argmax(1), St.max(1)
        back = St.argmax(0)  # the candidate nearest to every training player
        S2 = np.mean([np.einsum("cjd,pd->cjp", pairs2[k], U[k]) for k in range(K)], 0)  # [C10, 5, P]: two-game queries
        two = np.take_along_axis(S2, tb[ok10][:, None, None], 2)[:, :, 0].mean(1)  # mean over the five pairs, at the 10-game best match
        r = {"best_cos": q(tc), "by_thr": {}}
        print(f"\nC  {pool}: {C} candidates against the training players: best cosine {q(tc)}", flush=True)
        for t in a.thr:
            m = tc >= t
            hubs = np.bincount(tb[m], minlength=P)
            mm = m & (back[tb] == np.arange(C))
            m10 = m[ok10]
            r["by_thr"][str(t)] = {"candidates": int(m.sum()), "distinct_training_players": int((hubs > 0).sum()), "largest_hub": int(hubs.max()) if m.any() else 0, "mutual": int(mm.sum()),
                                   "two_game_cos": q(two[m10]), "two_game_at_least_control_q5": int((two[m10] >= c5).sum()), "two_game_at_least_control_q25": int((two[m10] >= c25).sum())}
            rk = [rank[tb[i]] for i in np.flatnonzero(m)]
            print(f"   cosine >= {t}: {int(m.sum())} candidates ({m.mean():.3f}) -> {int((hubs > 0).sum())} distinct training players (most candidates on one player: {int(hubs.max()) if m.any() else 0}); mutual {int(mm.sum())}; "
                  f"their two-game cosine to that player {q(two[m10])} (control: 5 % {c5:.3f}, 25 % {c25:.3f}, median {c50:.3f}) -> at the control's 5 % / 25 % point or above: {int((two[m10] >= c5).sum())} / {int((two[m10] >= c25).sum())}; "
                  f"rank files of the matched players {({x: rk.count(x) for x in sorted(set(rk), key=lambda v: RANKS.index(v) if v in RANKS else 99)})}", flush=True)
        r["two_game_all"] = q(two)
        print(f"   all candidates: two-game cosine to their best training match {q(two)}", flush=True)
        # D  candidates among themselves; shared records
        Sp = np.mean([full[k] @ full[k].T for k in range(K)], 0)
        np.fill_diagonal(Sp, -2)
        pb, pc = Sp.argmax(1), Sp.max(1)
        pm = pb[pb] == np.arange(C)
        rec = {}
        for i, s in enumerate(sets):
            for k in s:
                rec.setdefault(sgfs[k], set()).add(i)
        shared = {r_: c_ for r_, c_ in rec.items() if len(c_) > 1}
        n_shared = np.zeros((C, C), int)
        for c_ in shared.values():
            for x in c_:
                for y in c_:
                    if x != y:
                        n_shared[x, y] += 1
        r["pool"] = {"best_cos": q(pc), "records_under_two_candidates": int(len(shared)), "candidates_sharing_a_record": int((n_shared.sum(1) > 0).sum()), "by_thr": {}}
        print(f"D  {pool}: candidates among themselves: best cosine {q(pc)}; game records listed under two candidates: {len(shared)} (of {len(rec)}), candidates that share a record with another: {int((n_shared.sum(1) > 0).sum())}", flush=True)
        for t in a.thr:
            m = pc >= t
            mm = m & pm
            sh = n_shared[np.arange(C), pb]
            r["pool"]["by_thr"][str(t)] = {"candidates": int(m.sum()), "mutual": int(mm.sum()), "with_shared_records": int((sh[m] > 0).sum()), "shared_records_median": float(np.median(sh[m])) if m.any() else 0.0,
                                           "also_linked_to_training": int((tc[m] >= t).sum())}
            print(f"   cosine >= {t}: {int(m.sum())} candidates ({m.mean():.3f}); mutual {int(mm.sum())}; the partner shares game records with them: {int((sh[m] > 0).sum())} (records shared, median {np.median(sh[m]) if m.any() else 0:.0f} of 10); "
                  f"also linked to a training player at this level: {int((tc[m] >= t).sum())}; pairs WITHOUT a shared record: {int((sh[m] == 0).sum())}", flush=True)
        # how alike are two candidates that share records but are not above the threshold, and the cosine of record-sharing pairs
        ii, jj = np.nonzero(np.triu(n_shared, 1))
        if len(ii):
            print(f"   pairs of candidates that share records: {len(ii)}; records shared {q(n_shared[ii, jj])}; their cosine {q(Sp[ii, jj])}", flush=True)
            r["pool"]["sharing_pairs"] = {"n": int(len(ii)), "shared": q(n_shared[ii, jj]), "cos": q(Sp[ii, jj])}
        res_pools[pool] = {"twin": np.where((pc >= thr), pb, -1), "tc": tc, "n_shared": n_shared, "pb": pb}
        out[pool] = r
    # E  what the pairs cost (official pseudo-pools, fused scores of the final's composition)
    if Path(a.tasks).exists():
        with open(a.tasks, "rb") as fh:
            tasks = pickle.load(fh)["tasks"]
        print(f"\nE  cost on the pseudo-pools (1-game questions; a twin = the nearest candidate at cosine >= {thr})", flush=True)
        tot = {}
        for (pool, s), t in sorted(tasks.items()):
            z = t["sizes"][1]
            twin = res_pools[pool]["twin"]
            own, top, rank_ = z["own"], z["top"], z["rank"]
            sc = score_of(rank_)
            has = twin[own] >= 0
            first_is_twin = top[:, 0] == twin[own]
            # coupling: the twin of the first candidate goes to the second place
            new_rank = rank_.copy()
            for i in range(len(own)):
                tw = twin[top[i, 0]]
                if tw < 0:
                    continue
                order = top[i].tolist()
                if tw in order:
                    order.remove(tw)
                order.insert(1, tw)
                if own[i] in order[:10]:
                    new_rank[i] = order.index(own[i])
                elif rank_[i] < 10:
                    new_rank[i] = 10
            d = score_of(new_rank) - sc
            for k_, v_ in (("n", len(own)), ("has", has.sum()), ("lost", (1 - sc).sum()), ("lost_has", (1 - sc)[has].sum()), ("score_has", sc[has].sum()), ("score_not", sc[~has].sum()), ("first_is_twin", (first_is_twin & has).sum()),
                           ("first_is_twin_owner_second", (first_is_twin & has & (rank_ == 1)).sum()), ("gain", d.sum()), ("gain_has", d[has].sum()), ("moved_up", (d > 0).sum()), ("moved_down", (d < 0).sum())):
                tot[(pool, k_)] = tot.get((pool, k_), 0) + float(v_)
        out["cost"] = {}
        for pool in ("pub2", "priv2"):
            g = lambda k_: tot[(pool, k_)]  # noqa: E731
            out["cost"][pool] = {k_: g(k_) for k_ in ("n", "has", "lost", "lost_has", "first_is_twin", "gain", "moved_up", "moved_down")}
            print(f"   {pool}: questions of candidates with a twin {g('has') / g('n'):.3f}; their score {g('score_has') / max(g('has'), 1):.3f} against {g('score_not') / (g('n') - g('has')):.3f} for the others; share of the lost score {g('lost_has') / g('lost'):.3f}; "
                  f"first candidate is the owner's twin in {g('first_is_twin') / max(g('has'), 1):.3f} of them (owner then second: {g('first_is_twin_owner_second') / max(g('first_is_twin'), 1):.3f}); "
                  f"coupling: {g('gain') / g('n'):+.4f} per 1-game question ({int(g('moved_up'))} questions better, {int(g('moved_down'))} worse)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
