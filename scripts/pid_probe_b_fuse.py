#!/usr/bin/env python
"""Probe B, second condition (METHOD_NOTES 2.88): is a sequence network on supervised per-ply features better than the usual
network PLUS the per-game embedding of the unprojected rows fused at the score level?

Simulated pools of folds 1 and 2 (scripts/pid_probe_fullrows.py). For every network tag given (fold vectors
data/seq/fold<v>_n<tag>.npz from scripts/pid_v2_seq_embed.py) the sequence scores are the cosine of the summed game vectors,
as the sequence component computes them. E is aicup_pid.embed.PlayerEmbedding on the unprojected per-game rows, trained per
fold on the players of the OTHER folds (never fold 0). Scores are row-standardised; the weight of E for a fold's pools is the
best one on the other fold's pools. Reported: every tag alone and with E, and the paired differences against --ref + E.
Run from the AI_CUP_2026-1 checkout (one GPU).
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
from pid_probe_fullrows import SIZES, boot, mix, per_question, pools_and_games  # noqa: E402

WEIGHTS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)


def standardise(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True, help="network tags: fold vectors data/seq/fold<v>_n<tag>.npz")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--rows", nargs="+", default=["/work/u2507115/pid_v2/fullrows_train/rows", "/work/u2507115/pid_v2/probep/rows", "/work/u2507115/pid_v2/fullrows/rows"],
                    help="directories of per-game rows with `full` (a game is taken from the first directory that has it)")
    ap.add_argument("--emb-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    from aicup.ladder import to_json
    from aicup_pid.embed import PlayerEmbedding, score_embed
    pools, games = pools_and_games(a.folds, a.seeds)
    fold_of = dict(zip(games.game_key, games.fold))
    owner_of = dict(zip(games.game_key, games.player_id))
    full = {}
    for d in a.rows:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if "full" in z.files:
                for k, r in zip(z["keys"].astype(str), z["full"]):
                    if fold_of.get(k, 0) != 0 and k not in full:
                        full[k] = r.astype(np.float32)
    print(f"unprojected rows of {len(full)} games ({time.time() - t0:.0f} s)", flush=True)
    E = {}
    for v in a.folds:
        by = {}
        for k in sorted(full):
            if fold_of[k] not in (0, v):
                by.setdefault(owner_of[k], []).append(k)
        rows_by = {p: np.stack([full[k] for k in ks]) for p, ks in by.items() if len(ks) >= 8}
        E[v] = [PlayerEmbedding(seed=s, p_aug=0.0).fit(rows_by) for s in a.emb_seeds]
        print(f"fold {v}: embedding of the unprojected rows trained on {sum(len(x) for x in rows_by.values())} games of {len(rows_by)} players of the other folds ({time.time() - t0:.0f} s)", flush=True)
    vec = {(v, tag): (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(R2 / "data" / "seq" / f"fold{v}_n{tag}.npz")) for v in a.folds for tag in a.tags}
    runs = []
    for v, s, cands, questions, owner in pools:
        players = sorted(cands)
        truth = np.array([players.index(p) for p in owner])
        n = np.array([len(q) for q in questions])
        ck = {p: [k for k, _ in cands[p]] for p in players}
        qk = [[k for k, _ in q] for q in questions]
        Se = standardise(np.mean([standardise(score_embed(m, ck, qk, full)) for m in E[v]], axis=0))
        S = {}
        for tag in a.tags:
            one = vec[(v, tag)]
            Cm = unit(np.stack([np.sum([one[k] for k in ck[p]], axis=0) for p in players]).astype(np.float64))
            Qm = unit(np.stack([np.sum([one[k] for k in ks], axis=0) for ks in qk]).astype(np.float64))
            S[tag] = standardise(Qm @ Cm.T)
        runs.append({"v": v, "s": s, "truth": truth, "n": n, "E": Se, "S": S})
        print(f"fold {v} split {s}: alone  E {mix(per_question(Se, truth), n):.4f}  " + "  ".join(f"{t} {mix(per_question(S[t], truth), n):.4f}" for t in a.tags) + f" ({time.time() - t0:.0f} s)", flush=True)
    n_all = np.concatenate([r["n"] for r in runs])
    cl_all = np.concatenate([np.array([f"{r['v']}|{r['s']}|{t}" for t in r["truth"]]) for r in runs])
    rng = np.random.default_rng(0)
    out = {"tags": a.tags, "ref": a.ref, "alone": {}, "with_E": {}, "weights": {}, "diff": {}}
    sc = {}
    xe = np.concatenate([per_question(r["E"], r["truth"]) for r in runs])
    out["alone"]["E"] = {"mix": mix(xe, n_all), **{str(m): float(xe[n_all == m].mean()) for m in SIZES}}
    for tag in a.tags:
        w_for = {}
        for v in a.folds:  # the weight of E for fold v's pools is chosen on the other fold's pools
            others = [r for r in runs if r["v"] != v]
            w_for[v] = max(WEIGHTS, key=lambda w: mix(np.concatenate([per_question(r["S"][tag] + w * r["E"], r["truth"]) for r in others]), np.concatenate([r["n"] for r in others])))
        sc[(tag, "alone")] = np.concatenate([per_question(r["S"][tag], r["truth"]) for r in runs])
        sc[(tag, "E")] = np.concatenate([per_question(r["S"][tag] + w_for[r["v"]] * r["E"], r["truth"]) for r in runs])
        out["weights"][tag] = {str(v): w for v, w in w_for.items()}
        for kind in ("alone", "E"):
            x = sc[(tag, kind)]
            out["alone" if kind == "alone" else "with_E"][tag] = {"mix": mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}}
        print(f"{tag}: alone {mix(sc[(tag, 'alone')], n_all):.4f} (n=1 {sc[(tag, 'alone')][n_all == 1].mean():.4f})   with E {mix(sc[(tag, 'E')], n_all):.4f} (n=1 {sc[(tag, 'E')][n_all == 1].mean():.4f}; weights {w_for})", flush=True)
    print(f"E alone {out['alone']['E']['mix']:.4f} (n=1 {out['alone']['E']['1']:.4f})", flush=True)
    for tag in a.tags:
        for kind in ("alone", "E"):
            for base in ("alone", "E"):
                if tag == a.ref and kind == base:
                    continue
                b = boot(sc[(tag, kind)] - sc[(a.ref, base)], n_all, cl_all, a.boot, rng)
                name = f"{tag}{'+E' if kind == 'E' else ''} vs {a.ref}{'+E' if base == 'E' else ''}"
                out["diff"][name] = b
                print(f"   {name}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n=1 {b['by_n']['1']:+.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
