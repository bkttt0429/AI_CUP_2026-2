#!/usr/bin/env python
"""Are the one-game questions of the added pools distinct among themselves, and what is an assignment among them worth?
(METHOD_NOTES 2.70; the structure of 2.37 was: questions of 2 / 3 / 5 games have 300 different owners, the one-game questions
share owners with them at the rate of independent draws; the pairs among the one-game questions looked like DISTINCT owners,
which was never used.)

Part 1 (no answers): collision counts of submission files on the real added pools: pairs of questions with the same Top-1,
by question size (1-1, 1-big, big-big).
Part 2 (official pseudo-pools as scripts/pid_v2_offsim.py: 600 candidates, 7 reference games, 3 held-out games each; one
fusion document at its 50-step vectors): sets of 100 one-game questions drawn two ways, --draws times per pool and split
   distinct      100 different candidates, one held-out game of each
   independent   100 draws of a candidate with replacement (another held-out game on a repeat)
For each: the number of same-Top-1 pairs (to compare with part 1) and the score per question with no assignment, with the soft
assignment among the 100 (aicup_pid.assign.assign_soft, several penalties) and with the hard one; and with a
hub correction (the mean of a candidate's k highest scores over the set, times a factor, taken off its column).
Pools are never mixed; no answer of a real question is used.
"""
import argparse
import csv
import json
import re
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
from aicup.submission import check_game  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft, assign_topk, topk  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402

LAMS = (0.5, 1.0, 2.0)


def pairs(top1):
    """Number of unordered pairs with the same Top-1."""
    _, c = np.unique(top1, return_counts=True)
    return int((c * (c - 1) // 2).sum())


def cross_pairs(a, b):
    ua, ca = np.unique(a, return_counts=True)
    cb = dict(zip(*np.unique(b, return_counts=True)))
    return int(sum(n * cb.get(u, 0) for u, n in zip(ua, ca)))


def score_lists(tops, truth):
    """e^-(rank-1) of the truth inside each Top-5 list."""
    hit = tops == truth[:, None]
    return np.where(hit.any(1), np.exp(-hit.argmax(1).astype(float)), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--files", nargs="*", default=[str(ROOT / "upload" / x) for x in ("pid_v2gs_m90.csv", "pid_v2is_o1.csv", "pid_v2w_n8s.csv")])
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2w_n8.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--set-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T = Path(a.tests_dir)
    out = {"files": {}, "pseudo": {}}
    csv.field_size_limit(10 ** 9)
    size = {}
    for tag in ("public", "private"):
        with open(T / f"player_identification_test2_{tag}.csv", newline="") as fh:
            size.update({r["question_id"]: int(r["num_games"]) for r in csv.DictReader(fh)})
    for f in a.files:
        with open(f, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["question_id"] in size]
        rep = {}
        for pool in ("pub2", "priv2"):
            q = [r for r in rows if r["question_id"].startswith(pool + "_")]
            n = np.array([size[r["question_id"]] for r in q])
            t1 = np.array([r["top_1"] for r in q])
            rep[pool] = {"one-one": pairs(t1[n == 1]), "one-big": cross_pairs(t1[n == 1], t1[n > 1]), "big-big": pairs(t1[n > 1]),
                         "one-2": cross_pairs(t1[n == 1], t1[n == 2]), "one-3": cross_pairs(t1[n == 1], t1[n == 3]), "one-5": cross_pairs(t1[n == 1], t1[n == 5]),
                         "questions": {str(m): int((n == m).sum()) for m in sorted(set(n.tolist()))}, "distinct_top1_of_one": int(len(set(t1[n == 1].tolist())))}
        out["files"][Path(f).name] = rep
        print(f"{Path(f).name}: same-Top-1 pairs  " + "  ".join(f"{pool} one-one {v['one-one']}, one-big {v['one-big']} (2: {v['one-2']}, 3: {v['one-3']}, 5: {v['one-5']}), big-big {v['big-big']}" for pool, v in rep.items()), flush=True)

    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    rng = np.random.default_rng(0)
    methods = ["none", *[f"soft{lam:g}" for lam in LAMS], "hard", *[f"hub{b:g}k{k}" for b in (0.5, 1.0) for k in (1, 3)]]
    acc = {s_: {"pairs": [], "top1": [], **{m: [] for m in methods}} for s_ in ("distinct", "independent")}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        for s in a.seeds:
            hold = v2.candidate_holdout(cands, a.holdout, s)
            held = {k for g in hold.values() for k, _ in g}
            ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
            questions, truth = [], []
            for i, p in enumerate(players):
                for g in [(k, c) for k, c in hold[p] if k not in bad]:
                    questions.append([g]); truth.append(i)
            truth = np.array(truth)
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            seqs = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz") for v, tg in nets]
            Z["seq"] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=seqs)["seq"])
            F = fus.scores([Z[c] for c in comps])
            by = [np.flatnonzero(truth == i) for i in range(len(players))]
            have = np.array([i for i in range(len(players)) if len(by[i])])
            print(f"{pool} split {s}: {len(questions)} one-game questions of {len(have)} candidates; Top-1 {np.mean(F.argmax(1) == truth):.4f} ({time.time() - t0:.0f} s)", flush=True)
            for struct in ("distinct", "independent"):
                for _ in range(a.draws):
                    if struct == "distinct":
                        rows = np.array([rng.choice(by[i]) for i in rng.choice(have, a.set_size, replace=False)])
                    else:
                        rows, used = [], {}
                        while len(rows) < a.set_size:
                            i = int(rng.choice(have))
                            left = [r for r in by[i] if r not in used.get(i, ())]
                            if left:
                                r = int(rng.choice(left))
                                used.setdefault(i, set()).add(r)
                                rows.append(r)
                        rows = np.array(rows)
                    Sx, tx = F[rows], truth[rows]
                    base = topk(Sx, 5)
                    acc[struct]["pairs"].append(pairs(base[:, 0]))
                    acc[struct]["top1"].append(float((base[:, 0] == tx).mean()))
                    acc[struct]["none"].append(float(score_lists(base, tx).mean()))
                    for lam in LAMS:
                        acc[struct][f"soft{lam:g}"].append(float(score_lists(assign_soft(Sx, lam, 5), tx).mean()))
                    acc[struct]["hard"].append(float(score_lists(assign_topk(Sx, 5), tx).mean()))
                    for b in (0.5, 1.0):  # hub correction (CSLS-like): a candidate that scores high for many of the questions is marked down
                        for k in (1, 3):
                            r = np.sort(Sx, axis=0)[-k:].mean(0)
                            acc[struct][f"hub{b:g}k{k}"].append(float(score_lists(topk(Sx - b * r[None], 5), tx).mean()))
    for struct, v in acc.items():
        p = np.array(v["pairs"])
        none = np.array(v["none"])
        rep = {"draws": len(p), "pairs_mean": float(p.mean()), "pairs_sd": float(p.std()), "pairs_quantiles_5_50_95": [float(x) for x in np.percentile(p, [5, 50, 95])],
               "top1": float(np.mean(v["top1"])), "score_none": float(none.mean())}
        for m in methods[1:]:
            d = np.array(v[m]) - none
            rep[m] = {"gain": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))), "share_of_draws_up": float((d > 0).mean()), "share_down": float((d < 0).mean())}
        out["pseudo"][struct] = rep
        print(f"{struct}: same-Top-1 pairs among {a.set_size} one-game questions {rep['pairs_mean']:.2f} (sd {rep['pairs_sd']:.2f}; 5/50/95 % {rep['pairs_quantiles_5_50_95']}); Top-1 {rep['top1']:.4f}; score {rep['score_none']:.4f}; "
              + "  ".join(f"{m} {rep[m]['gain']:+.4f} (se {rep[m]['se']:.4f}, up {rep[m]['share_of_draws_up']:.2f} / down {rep[m]['share_down']:.2f})" for m in methods[1:]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
