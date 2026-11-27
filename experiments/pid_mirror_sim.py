#!/usr/bin/env python
"""Feasibility bound for the rank session's "mirror" channel, in competition 1's own simulation (METHOD_NOTES 2.28).

Their question (reports/玩家預測可用之處_給另一個對話_2026-11-11.md, 5.1 / question 2): candidates are players known only
through a SMALL profile (a question's 5-20 games), queries are single games, almost all of which belong to nobody in the pool;
can competition 1's fusion pull the true ones out of the null's tail? Their first version (learned matcher only) saw no excess.
Here the truth is planted, so recall at a fixed false-accept rate is measured directly:
  pool (a fold pair, never fold 0): 400 candidates with >= 40 Black games; profile = 5 / 10 / 20 games (prob .25 / .25 / .5);
  planted queries = 20 other games of every candidate, one game each; null queries = up to 25 games of every other player of
  the fold pair (nobody of the pool). All Black, same colour as the profiles.
Components are fitted on the profiles only. Arms (weights cross-fitted on the other pool's planted queries, multinomial
likelihood as aicup_pid.fusion): A emb (their mirror v1) | B opening, style, local, orient N=1, hd (their five) | C = B + emb
(their forward recipe) | D = C with orient N=4 and the raw-coordinate opening added | E = D + trunk LDA.
Statistics, each thresholded at the null queries' 1 - alpha quantile (alpha = 1e-2, 1e-3, 1e-4); a planted query counts
when its Top-1 is its player AND the statistic passes:
  margin     Top-1 minus Top-2 of the fused row (what the rank session uses)
  cohort     (Top-1 - mean of the next 50) / their sd
  label      (Top-1 - mean over the other candidates with the Top-1's rank label) / their sd   (same-strength cohort)
  znorm      the fused score standardised per candidate over the null queries, then its row maximum (candidate-side null)
Reading fixed before the run (guidance only, nothing is adopted in competition 1): at alpha = 1e-3 the mirror is worth
porting the habit components to if arm C or D recalls >= 10% of the planted single games, and hopeless if every arm < 3%.
Also reported: C vs A, D vs C, E vs D, by profile size, closed-set Top-1.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import features_all, fit_from_parsed, fit_hd, fit_local, fit_opening_raw, fit_orient, parse_all, score_hd  # noqa: E402
from aicup_pid.embed import PlayerEmbedding, score_embed  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, score_trunk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402

PROFILE, PROFILE_P, PLANT, NULL_PER, MIN_BLACK = (5, 10, 20), (0.25, 0.25, 0.5), 20, 25, 40
ALPHAS = (1e-2, 1e-3, 1e-4)
ARMS = {"A_emb": ("emb",), "B_five": ("opening", "style", "local", "orient1", "hd"), "C_five_emb": ("opening", "style", "local", "orient1", "hd", "emb"),
        "D_raw": ("opening", "style", "local", "orient4", "hd", "emb", "opening_raw"), "E_trunk": ("opening", "style", "local", "orient4", "hd", "emb", "opening_raw", "trunk")}
_G = {}


def _chunk(args):
    name, lo, hi = args
    model, games = _G[name]
    return np.stack([model.log_scores([g]) for g in games[lo:hi]])


def par_scores(model, games, workers, size=500):
    """[Q, C] log scores of single-game queries, in forked workers (the opening models loop in Python)."""
    _G["m"] = (model, games)
    jobs = [("m", i, min(i + size, len(games))) for i in range(0, len(games), size)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_chunk, jobs)
    else:
        parts = [_chunk(j) for j in jobs]
    return np.concatenate(parts)


def build(games, folds_df, folds, seed):
    elig = pools.eligible_players(games, folds_df, folds, MIN_BLACK)
    rng = np.random.RandomState(seed)
    chosen = elig.iloc[np.sort(rng.permutation(len(elig))[:pools.N_CANDIDATES])]
    cand = chosen.player_id.tolist()
    label = dict(zip(chosen.player_id, chosen["rank"]))
    everyone = folds_df[folds_df.fold.isin(list(folds))].player_id.astype(str)
    others = sorted(set(everyone) - set(cand))
    by = games[games.player_id.isin(set(cand) | set(others))].groupby("player_id")["game_key"].apply(sorted).to_dict()
    sizes = rng.choice(PROFILE, size=len(cand), p=PROFILE_P)
    profile, queries = {}, []  # queries: (key, owner or None)
    for p, n in zip(cand, sizes):
        keys = pools._player_keys(by[p], "shuffle", seed, p)
        profile[p] = keys[:int(n)]
        queries += [(k, p) for k in keys[int(n):int(n) + PLANT]]
    n_planted = len(queries)
    for p in others:
        if p in by:
            queries += [(k, None) for k in pools._player_keys(by[p], "shuffle", seed, p)[:NULL_PER]]
    return cand, label, profile, queries, {"eligible": int(len(elig)), "candidates": len(cand), "planted": n_planted, "null": len(queries) - n_planted,
                                           "null_players": len(others), "profile_sizes": {str(k): int((sizes == k).sum()) for k in PROFILE}}


def components(games, folds_df, folds, seed, cfg, emb, store, workers, log):
    t0 = time.time()
    cand, label, profile, queries, info = build(games, folds_df, folds, seed)
    keys = sorted({k for ks in profile.values() for k in ks} | {k for k, _ in queries})
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, workers), features_all(sgfs, workers)
    log(f"pool {folds}: {info}; {len(keys)} games parsed in {time.time() - t0:.0f} s", flush=True)
    opening, style = fit_from_parsed(profile, moves, feats, cfg)
    players = list(opening.players)
    q_moves = [moves[k] for k, _ in queries]
    q_keys = [[k] for k, _ in queries]
    S = {}
    for name, model in (("opening", opening), ("opening_raw", fit_opening_raw(profile, moves, cfg)), ("local", fit_local(profile, moves, cfg)),
                        ("orient1", fit_orient(profile, moves, {**cfg, "orient_raw": 1})), ("orient4", fit_orient(profile, moves, {**cfg, "orient_raw": 4}))):
        if list(model.players) != players:
            raise ValueError(f"{name}: candidate order differs")
        t1 = time.time()
        S[name] = par_scores(model, q_moves, workers)
        log(f"  {name}: {time.time() - t1:.0f} s", flush=True)
    S["style"] = np.stack([style.log_scores([feats[k]]) for k, _ in queries])
    S["hd"] = score_hd(fit_hd(profile, sgfs, cfg), q_keys, sgfs)
    S["trunk"] = score_trunk(fit_trunk(profile, store, cfg), q_keys, store)
    S["emb"] = score_embed(emb, profile, q_keys, store)
    truth = np.array([players.index(o) if o is not None else -1 for _, o in queries])
    psize = np.array([len(profile[o]) if o is not None else 0 for _, o in queries])
    lab = np.array([label[p] for p in players])
    log(f"pool {folds}: components ready, {time.time() - t0:.0f} s", flush=True)
    if sorted(profile) != players:
        raise ValueError("candidate order differs from the sorted ids (embedding / trunk order)")
    return {"S": {k: standardise(v).astype(np.float32) for k, v in S.items()}, "truth": truth, "psize": psize, "labels": lab, "info": {**info, "folds": list(folds), "seed": seed}}


def stats(F, labels, null):
    """{statistic: ([Q] value, [Q] Top-1 index)} for a fused matrix F; `null` marks the null queries (candidate-side norm)."""
    order = np.argsort(-F, axis=1)[:, :51]
    top = np.take_along_axis(F, order, axis=1)
    t1 = order[:, 0]
    out = {"margin": (top[:, 0] - top[:, 1], t1), "cohort": ((top[:, 0] - top[:, 1:].mean(axis=1)) / (top[:, 1:].std(axis=1) + 1e-9), t1)}
    lab_stat = np.zeros(len(F))
    for lv in np.unique(labels):
        rows = np.flatnonzero(labels[t1] == lv)
        cols = np.flatnonzero(labels == lv)
        if len(rows) == 0 or len(cols) < 3:
            continue
        sub = F[np.ix_(rows, cols)]
        s1 = F[rows, t1[rows]]
        tot, sq = sub.sum(axis=1) - s1, (sub ** 2).sum(axis=1) - s1 ** 2
        m = tot / (len(cols) - 1)
        lab_stat[rows] = (s1 - m) / (np.sqrt(np.maximum(sq / (len(cols) - 1) - m ** 2, 0)) + 1e-9)
    out["label"] = (lab_stat, t1)
    Z = (F - F[null].mean(axis=0)) / (F[null].std(axis=0) + 1e-9)
    out["znorm"] = (Z.max(axis=1), Z.argmax(axis=1))
    return out


def evaluate(F, r):
    truth, null = r["truth"], r["truth"] < 0
    res = {"top1": float((F[~null].argmax(axis=1) == truth[~null]).mean())}
    for name, (val, t1) in stats(F, r["labels"], null).items():
        hit = (~null) & (t1 == truth)
        res[name] = {}
        for a in ALPHAS:
            thr = np.quantile(val[null], 1 - a)
            ok = hit & (val > thr)
            res[name][f"{a:g}"] = {"recall": float(ok[~null].mean()), "null_passing": int((val[null] > thr).sum()),
                                   "by_profile": {str(n): float(ok[r["psize"] == n].mean()) for n in PROFILE}}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion3o_ref_trunk_embt.json"), help="only its cfg is used")
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb_p12.pt"))
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb_p34.pt"))
    ap.add_argument("--trunk-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "train"), str(ROOT / "data" / "trunk" / "train_rest")])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    cfg = json.loads(Path(a.fusion).read_text())["cfg"]
    store = load_store(*a.trunk_store)
    embs = {(1, 2): PlayerEmbedding.from_state(torch.load(a.emb12, map_location="cpu", weights_only=False)),
            (3, 4): PlayerEmbedding.from_state(torch.load(a.emb34, map_location="cpu", weights_only=False))}
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    runs = [components(games, folds_df, f, a.seed, cfg, embs[f], store, a.workers, print) for f in ((1, 2), (3, 4))]
    out = {"pools": [r["info"] for r in runs], "arms": {}, "single": {}, "alphas": list(ALPHAS)}
    for i, r in enumerate(runs):
        o = runs[1 - i]
        planted_o = o["truth"] >= 0
        key = "p" + "".join(map(str, r["info"]["folds"]))
        for arm, comps in ARMS.items():
            fus = Fusion()
            if len(comps) == 1:
                fus.w = np.ones(1)
            else:
                fus.fit([o["S"][c][planted_o] for c in comps], o["truth"][planted_o])
            F = np.einsum("j,jqc->qc", fus.w, np.stack([r["S"][c] for c in comps]))
            res = evaluate(F, r)
            res["weights"] = dict(zip(comps, np.round(fus.w, 3).tolist()))
            out["arms"].setdefault(arm, {})[key] = res
            print(f"pool {key} {arm}: Top-1 {res['top1']:.3f}; " + "; ".join(
                f"{s} " + "/".join(f"{res[s][f'{al:g}']['recall']:.3f}" for al in ALPHAS) for s in ("margin", "cohort", "label", "znorm")) + f"  weights {res['weights']}", flush=True)
        for c in r["S"]:
            res = evaluate(r["S"][c], r)
            out["single"].setdefault(c, {})[key] = {"top1": res["top1"], "margin_1e-3": res["margin"]["0.001"]["recall"], "znorm_1e-3": res["znorm"]["0.001"]["recall"]}
    print("single components (Top-1, recall at alpha 1e-3 by margin / znorm), mean of the two pools:")
    for c, v in out["single"].items():
        print(f"  {c}: " + " ".join(f"{np.mean([x[k] for x in v.values()]):.3f}" for k in ("top1", "margin_1e-3", "znorm_1e-3")), flush=True)
    print(f"arms, mean of the two pools: recall at alpha {ALPHAS}")
    summary = {}
    for arm, v in out["arms"].items():
        summary[arm] = {"top1": float(np.mean([x["top1"] for x in v.values()])),
                        **{s: {f"{al:g}": float(np.mean([x[s][f"{al:g}"]["recall"] for x in v.values()])) for al in ALPHAS} for s in ("margin", "cohort", "label", "znorm")},
                        "margin_1e-3_by_profile": {str(n): float(np.mean([x["margin"]["0.001"]["by_profile"][str(n)] for x in v.values()])) for n in PROFILE},
                        "best_1e-3_by_profile": {}}
        best = max(("margin", "cohort", "label", "znorm"), key=lambda s: summary[arm][s]["0.001"])
        summary[arm]["best_statistic_1e-3"] = best
        summary[arm]["best_1e-3_by_profile"] = {str(n): float(np.mean([x[best]["0.001"]["by_profile"][str(n)] for x in v.values()])) for n in PROFILE}
        print(f"  {arm}: Top-1 {summary[arm]['top1']:.3f}; " + "; ".join(f"{s} " + "/".join(f"{summary[arm][s][f'{al:g}']:.3f}" for al in ALPHAS) for s in ("margin", "cohort", "label", "znorm"))
              + f"; best at 1e-3 = {best}, by profile size {summary[arm]['best_1e-3_by_profile']}", flush=True)
    out["summary"] = summary
    best_cd = max(max(summary[x][s]["0.001"] for s in ("margin", "cohort", "label", "znorm")) for x in ("C_five_emb", "D_raw"))
    best_all = max(max(summary[x][s]["0.001"] for s in ("margin", "cohort", "label", "znorm")) for x in summary)
    out["reading"] = {"best_recall_C_or_D_at_1e-3": best_cd, "worth_porting": bool(best_cd >= 0.10), "hopeless": bool(best_all < 0.03)}
    print(f"reading: {out['reading']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
