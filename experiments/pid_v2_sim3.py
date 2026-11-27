#!/usr/bin/env python
"""Third simulation of the v2 added pools (METHOD_NOTES 2.31): fusion refinements on top of base+seq (pid_v2_sim2.py), same
single-fold pools (fold 1-4 x seeds 0, 1), weights fitted on the pools of the other folds.
Reference: ref = base+seq with every component row-standardised (the 11-12 second submission).
Arms (fixed before the run):
  raw_seq      the sequence model's cosine enters unstandardised (weight = its own scale): a question whose similarities are flat
               then weighs less, which row standardisation hides; the other components stay standardised
  raw_both     the same for the sequence model and the embedding
  by_n         ref with one weight vector per question size
  a0.5 / a1 / a4   ref with the smoothing of the per-player count models (opening, raw opening, orientation, local; default 2)
Rule: an arm replaces ref when its cross-fitted gain has a player-bootstrap CI above 0 and at least 6 of the 8 pools gain; the
best such arm is written to --fusion-out. Combinations of two passing arms go to a later run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
COUNT_MODELS = ("opening", "opening_raw", "orient4", "orient4_merged", "local")
ALPHAS = (0.5, 1.0, 4.0)


def fit_mixed(std, raw, truth, l2=1e-3, max_iter=300):
    """Weights of logits = sum_j w_j Z_j + sum_k v_k R_k (Z standardised rows, R raw scores): mean cross-entropy + l2 |w|^2 (no penalty on v,
    whose scale is the raw score's own). std / raw: lists of [Q, C]; returns (w, v)."""
    Z = torch.tensor(np.stack(std), dtype=torch.float64) if std else None
    R = torch.tensor(np.stack(raw), dtype=torch.float64) if raw else None
    y = torch.tensor(np.asarray(truth, dtype=np.int64))
    w = torch.ones(len(std), dtype=torch.float64, requires_grad=True)
    v = torch.full((len(raw),), 10.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, v] if len(raw) and len(std) else [v] if len(raw) else [w], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        logits = (torch.einsum("j,jqc->qc", w, Z) if Z is not None else 0) + (torch.einsum("k,kqc->qc", v, R) if R is not None else 0)
        loss = torch.nn.functional.cross_entropy(logits, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy().copy(), v.detach().numpy().copy()


def fused(w, v, std, raw):
    return (np.einsum("j,jqc->qc", w, np.stack(std)) if len(std) else 0) + (np.einsum("k,kqc->qc", v, np.stack(raw)) if len(raw) else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb2_p12.pt"))
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb2_p34.pt"))
    ap.add_argument("--seq-dir", default=str(ROOT / "data" / "seq"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    base_doc = json.loads(Path(a.cfg_from).read_text())
    cfg = base_doc["cfg"]
    emb_of = {k: PlayerEmbedding.from_state(torch.load(p, map_location="cpu", weights_only=False)) for k, p in (("12", a.emb12), ("34", a.emb34))}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v_ in (1, 2, 3, 4):
        z = np.load(Path(a.seq_dir) / f"fold{v_}.npz")
        seq = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            emb = emb_of["12" if v_ in (1, 2) else "34"]
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb, which=COMPS, seq=seq)
            alt = {al: v2.components(cands, questions, moves, sgfs, {**cfg, "alpha": al, "orient_alpha": al, "local_alpha": al}, which=COUNT_MODELS) for al in ALPHAS}
            players = sorted(cands)
            runs.append({"Z": {k: standardise(x) for k, x in S.items()}, "raw": {"seq": S["seq"], "emb": S["emb"] / emb.scale},
                         "alt": {al: {k: standardise(x) for k, x in d.items()} for al, d in alt.items()},
                         "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "info": {**info, "fold": v_, "seed": s}})
            print(f"pool fold {v_} seed {s}: {len(questions)} questions ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])

    def parts(r, arm):
        """(standardised list, raw list) of one pool for an arm."""
        if arm == "raw_seq":
            return [r["Z"][c] for c in COMPS if c != "seq"], [r["raw"]["seq"]]
        if arm == "raw_both":
            return [r["Z"][c] for c in COMPS if c not in ("seq", "emb")], [r["raw"]["seq"], r["raw"]["emb"]]
        if arm.startswith("a"):
            al = float(arm[1:])
            return [r["alt"][al][c] if c in COUNT_MODELS else r["Z"][c] for c in COMPS], []
        return [r["Z"][c] for c in COMPS], []

    def cross(arm, by_n=False):
        sc, ws = [], []
        for r in runs:
            others = [o for o in runs if o["info"]["fold"] != r["info"]["fold"]]
            s_r = np.zeros(len(r["truth"]))
            for m in (SIZES if by_n else (None,)):
                sel = [(o["n"] == m) if by_n else np.ones(len(o["truth"]), bool) for o in others]
                std = [np.vstack([x[k_] for x, k_ in zip(col, sel)]) for col in zip(*[parts(o, arm)[0] for o in others])]
                raw = [np.vstack([x[k_] for x, k_ in zip(col, sel)]) for col in zip(*[parts(o, arm)[1] for o in others])]
                w, v_ = fit_mixed(std, raw, np.concatenate([o["truth"][k_] for o, k_ in zip(others, sel)]))
                rs = (r["n"] == m) if by_n else np.ones(len(r["truth"]), bool)
                ps, pr = parts(r, arm)
                s_r[rs] = score(topk(fused(w, v_, [x[rs] for x in ps], [x[rs] for x in pr]), 5), r["truth"][rs])
                ws.append([*w.tolist(), *v_.tolist()])
            sc.append(s_r)
        return sc, np.round(np.mean(ws, axis=0), 3).tolist()

    out = {"pools": [r["info"] for r in runs], "arms": {}}
    per_pool, per_q = {}, {}
    for arm, by_n in (("ref", False), ("raw_seq", False), ("raw_both", False), ("by_n", True), *[(f"a{al:g}", False) for al in ALPHAS]):
        sc, w = cross("ref" if arm == "by_n" else arm, by_n)
        per_pool[arm], per_q[arm] = [float(x.mean()) for x in sc], np.concatenate(sc)
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES}, "weights_mean": w}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  weights {w} ({time.time() - t0:.0f} s)", flush=True)
    out["comparisons"], passing = {}, []
    for arm in per_q:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=a.n_boot)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 6)
        out["comparisons"][f"{arm} vs ref"] = d
        if d["pass"]:
            passing.append(arm)
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/8 -> {'PASS' if d['pass'] else 'no'}", flush=True)
    best = max(passing, key=lambda k: out["arms"][k]["score"]) if passing else "ref"
    out["chosen"] = best
    print(f"chosen: {best}", flush=True)
    if a.fusion_out and best not in ("ref", "by_n"):
        std = [np.vstack(col) for col in zip(*[parts(r, best)[0] for r in runs])]
        raw = [np.vstack(col) for col in zip(*[parts(r, best)[1] for r in runs])]
        w, v_ = fit_mixed(std, raw, np.concatenate([r["truth"] for r in runs]))
        names_std = [c for c in COMPS if not (best == "raw_seq" and c == "seq") and not (best == "raw_both" and c in ("seq", "emb"))]
        doc = {**base_doc, "components": names_std, "weights": w.tolist(), "raw_components": (["seq"] if best == "raw_seq" else ["seq", "emb"] if best == "raw_both" else []),
               "raw_weights": v_.tolist(), "arm": best, "simulated_score": out["arms"][best]["score"]}
        if best.startswith("a"):
            doc["cfg"] = {**cfg, "alpha": float(best[1:]), "orient_alpha": float(best[1:]), "local_alpha": float(best[1:])}
        Path(a.fusion_out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: {dict(zip(names_std, np.round(w, 3).tolist()))} raw {dict(zip(doc['raw_components'], np.round(v_, 2).tolist()))}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
