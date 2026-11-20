#!/usr/bin/env python
"""Does the behaviour policy's verifier score help against the fusion's hard negatives? (METHOD_NOTES 2.209, feasibility probe)

Input: the shard files of scripts/pid_verif_scores.py for ONE pseudo-pool split and the fused score matrix of the reference
document on the same split. The fused matrix is offsim's own: scripts/pid_v2_offsim_par.py --only POOL:SEED --dump DIR writes
split_<pool>_<seed>.pkl with the row-standardised component matrices Z (not their weighted sum), truth and n of all questions;
here Fusion.scores(Z in the document's component order) with the document's weights gives [questions, candidates], exactly as
offsim_par's statistics do. When DIR has no such file (or --dump is not given: a temporary folder) this script calls
pid_v2_offsim_par.main with --fusion / --ft-dir / --steps <step> / --only itself. The shard files carry every query game's row
in that question list; truth and n of those rows are checked against the split.

1-game questions only; V = intact - permute (mean log-probability per ply, mean over the model seeds), S = the fused score:
  D0  the document's score on the 1-game questions (offsim's per_question: e^-rank inside the Top-5)
  D1  questions with the true candidate at fused rank 2-5 (and 2-3): mean of V[true] - V[wrong first choice], nats per ply,
      95 % interval of a bootstrap over the candidates (players); control: the same with `intact` alone
  D2  re-ranking the fused Top-K with z_K(S) + w z_K(V) (z_K: standardised over the question's fused Top-K; the candidates
      below keep their fused order), K in (3, 5, 10, 600), w in (0, 0.125, 0.25, 0.5, 1, 2); control: `intact` instead of V.
      The best (K, w) is chosen on these same questions: its gain is optimistic (the pilot's own cell K = 3, w = 0.125 is
      reported next to it).
Everything goes to --out (JSON). Compute node only.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

KS, WS = (3, 5, 10, 600), (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)


def fused(a):
    """([Q, C] fused scores of the reference document, truth [Q], n [Q], offsim's per_question) for all questions of the split."""
    import pid_v2_offsim_par as op
    from aicup_pid.fusion import Fusion
    dump = Path(a.dump or tempfile.mkdtemp(prefix="verif_split_"))
    f = dump / f"split_{a.pool}_{a.holdout_seed}.pkl"
    try:
        if f.exists():
            print(f"split read from {f} (it must be the split of {Path(a.fusion).name})", flush=True)
        else:
            op.main(["--fusion", a.fusion, "--ft-dir", a.ft_dir, "--steps", str(a.step), "--seeds", str(a.holdout_seed), "--workers", str(a.workers),
                     "--only", f"{a.pool}:{a.holdout_seed}", "--dump", str(dump), "--out", os.devnull])
        with open(f, "rb") as fh:
            r = pickle.load(fh)
    finally:
        if not a.dump:
            shutil.rmtree(dump, ignore_errors=True)
    doc = json.loads(Path(a.fusion).read_text())
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    F = fus.scores([r["Z"][("seq", a.step)] if c == "seq" else r["Z"][c] for c in doc["components"]])
    return F, np.asarray(r["truth"]), np.asarray(r["n"]), op.per_question


def boot(d, who, K, B, rng):
    """(mean of d, 95 % interval): the K candidates are drawn with replacement, a candidate's questions go together."""
    num, den = np.bincount(who, weights=d, minlength=K), np.bincount(who, minlength=K).astype(float)
    pick = rng.integers(0, K, (B, K))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = num[pick].sum(1) / den[pick].sum(1)
    return float(d.mean()), [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]


def rerank(S, X, truth, K, w):
    """Per-question score after the fused Top-K is re-ordered by z_K(S) + w z_K(X); w = 0 is the fused ranking."""
    rows = np.arange(len(truth))
    top = np.argsort(-S, axis=1, kind="stable")[:, :K]

    def z(A):
        A = np.take_along_axis(A, top, axis=1)
        sd = A.std(axis=1, keepdims=True)
        return (A - A.mean(axis=1, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)

    new = z(S) + w * z(X)
    hit = top == truth[:, None]
    inside = hit.any(axis=1)
    own = new[rows, hit.argmax(axis=1)]
    rank = np.where(inside, (new > own[:, None]).sum(axis=1), (S > S[rows, truth][:, None]).sum(axis=1))
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True, choices=("pub2", "priv2"))
    ap.add_argument("--holdout-seed", type=int, required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="the shard files of pid_verif_scores.py for this split")
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2fin_x.json"))
    ap.add_argument("--ft-dir", default=str(ROOT / "data" / "pid_v2_seq_ft_offsim"))
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--dump", default=None, help="folder of split_<pool>_<seed>.pkl (read when it is there, written when not)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run inside a Slurm allocation (compute node)")
    parts = []
    for f in sorted(a.scores):
        with np.load(f, allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
    p0 = parts[0]
    assert all(str(x["pool"]) == a.pool and int(x["seed"]) == a.holdout_seed and np.array_equal(x["players"], p0["players"]) for x in parts)
    cat = lambda k: np.concatenate([x[k] for x in parts])  # noqa: E731
    row, truth, rows = cat("row"), cat("truth"), np.arange(sum(len(x["row"]) for x in parts))
    assert len(row) > 0 and len(set(row.tolist())) == len(row), "no query game, or a query game in two shard files"
    intact, permute = cat("intact").astype(float), cat("permute").astype(float)  # [q, model seed, C]
    V, I = (intact - permute).mean(axis=1), intact.mean(axis=1)
    F, t_all, n_all, per_question = fused(a)
    assert len(t_all) == int(p0["n_questions"]) and F.shape[1] == len(p0["players"]) and (n_all[row] == 1).all() and np.array_equal(t_all[row], truth), \
        "the shard files and the split disagree on the questions"
    S, rng = F[row], np.random.default_rng(0)
    _, who = np.unique(truth, return_inverse=True)
    K = int(who.max()) + 1
    sc = per_question(F, t_all)
    full = [x for x in parts if "ref_positions" in x]
    out = {"pool": a.pool, "seed": a.holdout_seed, "document": Path(a.fusion).name, "step": a.step, "model_seeds": p0["model_seeds"].tolist(),
           "questions": int((n_all == 1).sum()), "scored": len(row), "players_scored": K, "limit": int(p0["limit"]),
           "refused_by_replay": sorted({str(k) for x in parts for k in x["rejected"]}), "plies": [int(cat("plies").min()), float(cat("plies").mean()), int(cat("plies").max())],
           "ref_games": [int(full[0]["ref_games"].min()), int(full[0]["ref_games"].max())], "ref_positions": [int(full[0]["ref_positions"].min()), int(full[0]["ref_positions"].max())]}
    # D0
    alone = {"V": float(per_question(V, truth).mean()), "intact": float(per_question(I, truth).mean()), "neutral_logp_per_ply": float(cat("neutral").mean())}
    out["D0"] = {"all_1game": float(sc[n_all == 1].mean()), "scored_1game": float(sc[row].mean()), "split_by_n": {str(m): float(sc[n_all == m].mean()) for m in (1, 2, 3)},
                 "verifier_alone": alone}
    print(f"D0 {a.pool} split {a.holdout_seed}, {out['document']}: 1-game questions {out['D0']['all_1game']:.4f} ({out['questions']} questions); the {len(row)} scored here "
          f"{out['D0']['scored_1game']:.4f}; n=2 {out['D0']['split_by_n']['2']:.4f}, n=3 {out['D0']['split_by_n']['3']:.4f}", flush=True)
    print(f"D0 the verifier alone as a ranking of the {F.shape[1]} candidates: V {alone['V']:.4f}, intact {alone['intact']:.4f}; plies per query game {out['plies'][1]:.1f}, "
          f"reference positions per candidate {out['ref_positions'][0]}-{out['ref_positions'][1]}, {len(out['refused_by_replay'])} games refused by the replay", flush=True)
    # D1
    r0, first = (S > S[rows, truth][:, None]).sum(axis=1), S.argmax(axis=1)  # fused rank of the true candidate (0 = first), the first choice
    out["D1"] = {}
    for name, hi in (("rank2_5", 5), ("rank2_3", 3)):
        m = (r0 >= 1) & (r0 < hi)
        res = {"questions": int(m.sum()), "players": int(len(set(who[m].tolist())))}
        for tag, X in (("V", V), ("intact", I)):
            if m.any():
                d = X[m, truth[m]] - X[m, first[m]]
                cnt = np.bincount(who[m], minlength=K)
                mean, ci = boot(d, who[m], K, a.boot, rng)
                res[tag] = {"mean": mean, "ci95": ci, "positive": float((d > 0).mean()),
                            "mean_of_player_means": float((np.bincount(who[m], weights=d, minlength=K)[cnt > 0] / cnt[cnt > 0]).mean())}
        out["D1"][name] = res
        if m.any():
            v, c = res["V"], res["intact"]
            print(f"D1 true candidate at fused rank 2-{hi}: V[true] - V[wrong first] = {v['mean']:+.4f} nats/ply [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], positive in "
                  f"{100 * v['positive']:.1f} % of {res['questions']} questions ({res['players']} players; mean of player means {v['mean_of_player_means']:+.4f}); "
                  f"control intact alone {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], positive {100 * c['positive']:.1f} %", flush=True)
        else:
            print(f"D1 true candidate at fused rank 2-{hi}: no such question", flush=True)
    # D2
    out["D2"] = {}
    for tag, X in (("V", V), ("intact", I)):
        base = sc[row]  # w = 0 is the fused ranking itself (offsim's per_question), whatever K
        assert all((rerank(S, X, truth, k, 0.0) == base).mean() > 0.999 for k in KS), "rerank(w = 0) is not the fused ranking"
        cell = {(k, w): rerank(S, X, truth, k, w) if w > 0 else base for k in KS for w in WS}
        best = max(((k, w) for k in KS for w in WS if w > 0), key=lambda kw: cell[kw].mean())
        res = {"base": float(base.mean()), "score": {str(k): {str(w): float(cell[k, w].mean()) for w in WS} for k in KS},
               "diff": {str(k): {str(w): float((cell[k, w] - base).mean()) for w in WS} for k in KS}}
        for label, kw in (("best", best), ("pilot_cell", (3, 0.125))):
            diff, ci = boot(cell[kw] - base, who, K, a.boot, rng)
            res[label] = {"K": kw[0], "w": kw[1], "score": float(cell[kw].mean()), "diff": diff, "ci95": ci}
        out["D2"][tag] = res
        name = "V = intact - permute" if tag == "V" else "control, intact alone"
        for k in KS:
            print(f"D2 {name}: Top-{k:<3d} " + "  ".join(f"w{w:g} {res['score'][str(k)][str(w)]:.4f} ({res['diff'][str(k)][str(w)]:+.4f})" for w in WS), flush=True)
        b, p = res["best"], res["pilot_cell"]
        print(f"D2 {name}: best K={b['K']} w={b['w']:g}: {b['score']:.4f}, {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] vs w=0 (chosen on these questions); "
              f"the pilot's K=3 w=0.125: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]", flush=True)
    from aicup.ladder import to_json  # strict JSON: NaN (an interval nobody could draw) becomes null
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(to_json(out, indent=1) + "\n")
    tmp.rename(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
