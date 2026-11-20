"""METHOD_NOTES 2.169: is there an ORDER structure in the added pools' ids?  (No answers are used; nothing is selected.)

The original pools were checked on 11-04 (notes 2.12: question number vs predicted candidate number, Spearman -0.18 / +0.02,
"no usable order leak").  The added pools (600 candidates x 10 games, 400 questions of 1 / 2 / 3 / 5 games) were not.
Reads the final submission file and the organisers' files only:
  1. question number vs the candidate number of the file's first answer - Spearman, per question size (the 3- and 5-game
     answers are ~0.9 correct, so a real relation would show there); a permutation null for the largest |rho|;
  2. where the first answers of each question size sit among the candidate numbers (blocks of 50; chi-square against even);
  3. per-candidate simple statistics along the candidate number (share of black games, HD[1] share, mean game length):
     Spearman with the candidate number and the lag-1 autocorrelation (a sorted or grouped pool would show a trend);
  4. neighbours: do candidates with adjacent numbers resemble each other more than random pairs (game length, black share)?
  5. the same for the question numbers: question size by question number (runs), game length by question number.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

csv.field_size_limit(10 ** 9)


def num(s):
    return int(s.rsplit("_", 1)[1])


def n_moves(sgf):
    return len(re.findall(r";[BW]\[", sgf))


def lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[1:] * x[:-1]).sum() / (x * x).sum())


def perm_p(x, y, rng, n=4000):
    r0 = abs(stats.spearmanr(x, y)[0])
    y = np.asarray(y)
    return float(np.mean([abs(stats.spearmanr(x, rng.permutation(y))[0]) >= r0 for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    with open(a.submission, newline="") as fh:
        sub = {r["question_id"]: [r[f"top_{i}"] for i in range(1, 6)] for r in csv.DictReader(fh)}
    res = {}
    for side in ("public", "private"):
        pre = "pub2" if side == "public" else "priv2"
        T = Path(a.tests)
        cand = {}
        with open(T / f"player_identification_candidates2_{side}.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                cand.setdefault(num(r["player_id"]), []).append((num(r["game_id"]), r["color"], "HD[1]" in r["sgf_content"][:20], n_moves(r["sgf_content"])))
        qs = []
        with open(T / f"player_identification_test2_{side}.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                n = int(r["num_games"])
                games = [(r[f"color_{i}"], "HD[1]" in r[f"sgf_{i}"][:20], n_moves(r[f"sgf_{i}"])) for i in range(1, n + 1)]
                qs.append((num(r["question_id"]), n, games, num(sub[r["question_id"]][0])))
        C = len(cand)
        print(f"\n===== {pre}: {C} candidates, {len(qs)} questions =====")
        out = {}
        # 1. question number vs first answer
        qn = np.array([q[0] for q in qs]); sz = np.array([q[1] for q in qs]); top = np.array([q[3] for q in qs])
        print("1. question number vs the candidate number of the first answer (Spearman rho, p; permutation p of |rho|):")
        for n in (1, 2, 3, 5, 0):
            m = sz == n if n else np.ones(len(qs), bool)
            rho, p = stats.spearmanr(qn[m], top[m])
            pp = perm_p(qn[m], top[m], rng)
            print(f"   {'all' if not n else str(n) + '-game'}: n {int(m.sum())}, rho {rho:+.3f} (p {p:.3f}; permutation {pp:.3f}); same number {int((qn[m] == top[m]).sum())}; mean candidate number {top[m].mean():.0f}")
            out[f"q_vs_top_{n}"] = {"rho": float(rho), "p": float(p), "perm_p": pp}
        # 2. blocks
        print("2. first answers per block of 50 candidate numbers (chi-square against even):")
        for n in (1, 2, 3, 5):
            h = np.bincount((top[sz == n] - 1) // 50, minlength=C // 50)
            chi, p = stats.chisquare(h)
            print(f"   {n}-game: {h.tolist()}  chi2 {chi:.1f} p {p:.3f}")
            out[f"blocks_{n}"] = {"hist": h.tolist(), "p": float(p)}
        multi = np.zeros(C + 1, int)
        for n in (2, 3, 5):
            for t in top[sz == n]:
                multi[t] = n
        owned = multi[1:] > 0
        print(f"   candidates that are the first answer of a multi-game question: {int(owned.sum())}; lag-1 autocorrelation of that flag along the candidate number {lag1(owned):+.3f}; runs z {runs_z(owned):+.2f}")
        seq = multi[1:]
        print(f"   question size (0 = none) along the candidate number, first 80: {''.join(str(v) for v in seq[:80])}")
        for mod in (2, 3, 4, 5, 6, 10):
            tab = np.array([[int(((np.arange(1, C + 1) % mod == k) & (seq == n)).sum()) for k in range(mod)] for n in (0, 2, 3, 5)])
            chi, p, _, _ = stats.chi2_contingency(tab)
            print(f"   candidate number mod {mod} x question size: chi2 p {p:.3f}")
        # 3. per-candidate statistics along the candidate number
        ids = np.arange(1, C + 1)
        black = np.array([np.mean([g[1] == "B" for g in cand[i]]) for i in ids])
        hd = np.array([np.mean([g[2] for g in cand[i]]) for i in ids])
        ln = np.array([np.mean([g[3] for g in cand[i]]) for i in ids])
        print("3. per-candidate statistics along the candidate number (Spearman rho p; lag-1 autocorrelation; permutation p of the lag-1):")
        for name, v in (("share of black games", black), ("share of HD[1] games", hd), ("mean game length", ln)):
            rho, p = stats.spearmanr(ids, v)
            l1 = lag1(v)
            null = np.array([lag1(rng.permutation(v)) for _ in range(4000)])
            pl = float(np.mean(np.abs(null) >= abs(l1)))
            print(f"   {name}: mean {v.mean():.3f}, rho {rho:+.3f} (p {p:.3f}); lag-1 {l1:+.3f} (permutation p {pl:.3f})")
            out[f"cand_{name}"] = {"rho": float(rho), "p": float(p), "lag1": l1, "lag1_p": pl}
        # is the game number inside a candidate ordered by anything (length trend over the 10 games; colour)?
        pos_len = np.array([[g[3] for g in sorted(cand[i])] for i in ids], float)
        pos_blk = np.array([[g[1] == "B" for g in sorted(cand[i])] for i in ids], float)
        print(f"   inside a candidate, by game order 1..10: mean length {np.round(pos_len.mean(0)).astype(int).tolist()}; black share {np.round(pos_blk.mean(0), 2).tolist()}")
        # 5. questions along the question number
        o = np.argsort(qn)
        szs = sz[o]
        qlen = np.array([np.mean([g[2] for g in qs[i][2]]) for i in o])
        qblk = np.array([np.mean([g[0] == "B" for g in qs[i][2]]) for i in o])
        print("5. questions along the question number:")
        print(f"   sizes, first 80: {''.join(str(v) for v in szs[:80])}; lag-1 of the size {lag1(szs):+.3f}; size vs number rho {stats.spearmanr(np.arange(len(szs)), szs)[0]:+.3f}")
        for n in (1, 2, 3, 5):
            h = np.bincount(np.flatnonzero(szs == n) // 50, minlength=len(szs) // 50)
            print(f"   {n}-game questions per block of 50 question numbers: {h.tolist()} (chi2 p {stats.chisquare(h)[1]:.3f})")
        print(f"   mean game length vs number rho {stats.spearmanr(np.arange(len(qlen)), qlen)[0]:+.3f}, lag-1 {lag1(qlen):+.3f}; black share lag-1 {lag1(qblk):+.3f}; black share of all question games {np.mean([g[0] == 'B' for q in qs for g in q[2]]):.3f}, of candidate games {black.mean():.3f}")
        # colour inside a multi-game question, HD
        print(f"   HD[1] share: question games {np.mean([g[1] for q in qs for g in q[2]]):.3f}, candidate games {hd.mean():.3f}")
        res[pre] = out
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


def runs_z(flag):
    """Wald-Wolfowitz runs test z of a 0/1 sequence (negative = clustered)."""
    f = np.asarray(flag, bool)
    n1, n0 = int(f.sum()), int((~f).sum())
    runs = 1 + int((f[1:] != f[:-1]).sum())
    mu = 2 * n1 * n0 / (n1 + n0) + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    return (runs - mu) / var ** 0.5


if __name__ == "__main__":
    main()
