#!/usr/bin/env python
"""Simulation of the 11-11 release ("the original questions plus added ones", METHOD_NOTES 2.27): what happens to the
assignment and to the score when extra questions are added over the SAME candidates.

Per pool (a fold pair, never fold 0): players with >= cand_games + 40 Black games; candidates = 100 games each; block A =
one question per player, sizes 5/10/15/20 (the 10-30 design). Added blocks, each paired with A in its own scenario:
  same    one more question per player, sizes 5/10/15/20, games disjoint from A
  small   one more question per player, sizes 1/2/3 (harder questions)
  random  as many questions as players, drawn with replacement (some players get none, some several), sizes 5/10/15/20
Arms on the main fusion (weights as given in the document, embedding of the other fold pair):
  plain        per-question Top-5
  whole_auto   the current rule on the whole pool (assignment when distinct Top-1 >= 87.5% of the questions)
  block_auto   the same rule and assignment block by block (A, added)                       <- the candidate procedure
  cap          capacity assignment on the whole pool, capacity ceil(Q / C)                  (report)
  csls         hub correction 2 s(q, c) - r_c, r_c = mean of candidate c's 10 best scores   (report; needs no structure)
Checks fixed before the run: (1) whole_auto does not apply in any scenario; (2) block_auto applies to A and to `same`,
not to `random`; (3) block_auto >= plain in every scenario and pool (never worse), with the A block equal to A alone.
Second run (11-10, after the first showed the 87.5% rule missing the one-to-one `small` block: only 84% distinct Top-1 at
78% accuracy, where the assignment is worth +0.02): rule v2 = one_to_one_v2 (threshold fixed from the birthday rate before
looking at the statistic); arms block_auto2 (v2 per block) and block_force (assignment in both blocks, what knowing the
structure would give). Checks: v2 applies to `same` and `small`, never to `random`; block_auto2 never worse than plain;
seeds 2, 3 are fresh pools.
"""
import argparse
import json
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
from aicup_pid.assign import assign_blocks, assign_topk  # noqa: E402
from aicup_pid.checks import CONFIDENT_FRACTION, ONE_TO_ONE_FRACTION, confident_distinct, top1_collisions  # noqa: E402
from aicup_pid.components import features_all, fit_from_parsed, fit_local, fit_orient, fit_ref, parse_all, score_components, score_ref  # noqa: E402
from aicup_pid.embed import PlayerEmbedding, score_embed  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.ref import reference_players  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402

SIZES, SMALL, EXTRA = (5, 10, 15, 20), (1, 2, 3), 40
SCENARIOS = ("same", "small", "random")


def build(games, folds_df, folds, seed, n_players, cand_games):
    elig = pools.eligible_players(games, folds_df, folds, cand_games + EXTRA)
    rng = np.random.RandomState(seed)
    chosen = elig.iloc[np.sort(rng.permutation(len(elig))[:n_players])]
    players = chosen.player_id.tolist()
    P = len(players)
    by = games[games.player_id.isin(players)].groupby("player_id")["game_key"].apply(sorted).to_dict()
    size_a, size_b, size_s = (rng.permutation(np.resize(x, P)) for x in (SIZES, SIZES, SMALL))
    cands, rest, qs = {}, {}, []
    for i, p in enumerate(players):
        keys = pools._player_keys(by[p], "shuffle", seed, p)
        cands[p], rest[p] = keys[:cand_games], keys[cand_games:]
        a, b = int(size_a[i]), int(size_b[i])
        qs.append(("A", p, rest[p][:a]))
        qs.append(("same", p, rest[p][a:a + b]))
        qs.append(("small", p, rest[p][a:a + int(size_s[i])]))
    used = {p: int(size_a[i]) for i, p in enumerate(players)}
    skipped = 0
    for d, m in zip(rng.randint(0, P, P), rng.permutation(np.resize(SIZES, P))):
        p, m = players[d], int(m)
        if used[p] + m > len(rest[p]):
            skipped += 1
            continue
        qs.append(("random", p, rest[p][used[p]:used[p] + m]))
        used[p] += m
    return players, cands, qs, {"eligible": int(len(elig)), "players": P, "random_skipped": skipped}


def score(idx, truth):
    out = np.zeros(len(truth))
    for q in range(len(truth)):
        hit = np.where(idx[q] == truth[q])[0]
        out[q] = np.exp(-hit[0]) if len(hit) else 0.0
    return out


def one_to_one(S):
    return bool(len(set(S.argmax(axis=1).tolist())) >= ONE_TO_ONE_FRACTION * len(S)) if len(S) else False


def conf_distinct(S, frac=CONFIDENT_FRACTION):
    return confident_distinct(S, frac)


def one_to_one_v2(S):
    """Rule v2 as predict_pid.py --assign auto applies it (aicup_pid.checks.top1_collisions)."""
    return bool(top1_collisions(S, np.zeros(len(S), dtype=int))["one_to_one_likely_v2"]) if len(S) else False


def csls(S, k=10):
    r_c = np.sort(S, axis=0)[-min(k, len(S)):].mean(axis=0)
    return 2 * S - r_c[None, :]


def run_pool(games, folds_df, folds, seed, doc, emb, store, a, log=print):
    t0 = time.time()
    cfg, comps = doc["cfg"], tuple(doc["components"])
    players, cands, qs, info = build(games, folds_df, folds, seed, a.n_players, a.cand_games)
    other = tuple(f for f in (1, 2, 3, 4) if f not in folds)
    refs = reference_players(games, folds_df, list(other), min_black=cfg.get("ref_min_black", 30), cap=cfg.get("ref_cap", 40), seed=cfg.get("ref_seed", 0),
                             exclude=set(players))
    pool_keys = {k for ks in cands.values() for k in ks} | {k for _b, _p, ks in qs for k in ks}
    keys = pool_keys | {k for ks in refs.values() for k in ks}
    sgfs = {k: v[0] for k, v in load_games(sorted(keys), aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all({k: sgfs[k] for k in pool_keys}, a.workers)
    q_keys = [list(ks) for _b, _p, ks in qs]
    opening, style = fit_from_parsed(cands, moves, feats, cfg)
    if list(opening.players) != sorted(players):
        raise ValueError("candidate order differs")
    S = score_components(("opening", "style", "local", "orient"), opening, style, q_keys, moves, feats, local=fit_local(cands, moves, cfg),
                         orient=fit_orient(cands, moves, cfg))
    S["ref"] = score_ref(fit_ref(refs, moves, cfg), cands, q_keys, moves)
    S["trunk"] = score_trunk(fit_trunk(cands, store, cfg), q_keys, store)
    S["emb"] = score_embed(emb, cands, q_keys, store)
    fus = Fusion()
    fus.w = np.array(doc["weights"])
    S_f = fus.scores([S[c] for c in comps])
    block = np.array([b for b, _p, _k in qs])
    pid = np.array([p for _b, p, _k in qs])
    n = np.array([len(k) for _b, _p, k in qs])
    truth = np.array([opening.index[p] for p in pid])
    C = len(players)
    rows_a = np.flatnonzero(block == "A")
    alone = {"plain": score(topk(S_f[rows_a], 5), truth[rows_a]), "assign": score(assign_topk(S_f[rows_a], 5), truth[rows_a])}
    out = {"info": {**info, "folds": list(folds), "seed": seed}, "A_alone": {k: {"score": float(v.mean()), "wrong": int((v < 1).sum())} for k, v in alone.items()},
           "scenarios": {}, "per_q": {}}
    for s in SCENARIOS:
        rows = np.flatnonzero((block == "A") | (block == s))
        Sf, bl, tr = S_f[rows], np.where(block[rows] == "A", "A", "added"), truth[rows]
        fired = {"whole": one_to_one(Sf), "A": one_to_one(Sf[bl == "A"]), "added": one_to_one(Sf[bl == "added"])}
        fired2 = {"A": one_to_one_v2(Sf[bl == "A"]), "added": one_to_one_v2(Sf[bl == "added"])}
        conf = {k: {"half": conf_distinct(Sf[m]), "quarter": conf_distinct(Sf[m], 0.25)} for k, m in (("A", bl == "A"), ("added", bl == "added"))}
        cap = int(np.ceil(len(rows) / C))
        arms = {"plain": topk(Sf, 5), "whole_auto": assign_topk(Sf, 5) if fired["whole"] else topk(Sf, 5),
                "block_auto": assign_blocks(Sf, bl, {"A": fired["A"], "added": fired["added"]}), "block_auto2": assign_blocks(Sf, bl, fired2),
                "block_force": assign_blocks(Sf, bl, {"A": True, "added": True}), "cap": assign_topk(Sf, 5, capacity=cap), "csls": topk(csls(Sf), 5)}
        res = {"questions": int(len(rows)), "added": int((bl == "added").sum()), "capacity": cap, "fired": fired, "fired_v2": fired2, "conf_distinct": conf,
               "distinct_top1": {k: int(len(set(Sf[m].argmax(axis=1).tolist()))) for k, m in (("whole", np.ones(len(rows), bool)), ("A", bl == "A"), ("added", bl == "added"))},
               "arms": {}}
        for arm, idx in arms.items():
            sc = score(idx, tr)
            res["arms"][arm] = {"score": float(sc.mean()), "wrong": int((idx[:, 0] != tr).sum()), "A": float(sc[bl == "A"].mean()), "added": float(sc[bl == "added"].mean()),
                                "wrong_A": int((idx[:, 0] != tr)[bl == "A"].sum()), "wrong_added": int((idx[:, 0] != tr)[bl == "added"].sum()),
                                "added_by_n": {str(m): float(sc[(bl == "added") & (n[rows] == m)].mean()) for m in sorted(set(n[rows][bl == "added"].tolist()))}}
            out["per_q"][f"{s}:{arm}"] = sc
        out["per_q"][f"{s}:pid"], out["per_q"][f"{s}:block"] = pid[rows], bl
        out["scenarios"][s] = res
        log(f"pool {folds} seed {seed} [{s}] {res['questions']} q ({res['added']} added), fired {fired}, v2 {fired2}, conf half/quarter A {conf['A']['half']:.3f}/{conf['A']['quarter']:.3f} "
            f"added {conf['added']['half']:.3f}/{conf['added']['quarter']:.3f}, distinct {res['distinct_top1']}: "
            + "  ".join(f"{k} {v['score']:.4f} ({v['wrong_A']}+{v['wrong_added']} wrong)" for k, v in res["arms"].items()), flush=True)
    log(f"pool {folds} seed {seed}: {C} candidates (eligible {info['eligible']}), A alone plain {out['A_alone']['plain']['score']:.4f} / assign {out['A_alone']['assign']['score']:.4f}, "
        f"{time.time() - t0:.0f} s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion3o_ref_trunk_emb.json"))
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb_p12.pt"), help="embedding for the pool of folds 1, 2 (trained on folds 3, 4)")
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb_p34.pt"))
    ap.add_argument("--trunk-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "train"), str(ROOT / "data" / "trunk" / "train_rest")])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    doc = json.loads(Path(a.fusion).read_text())
    store = load_store(*a.trunk_store)
    embs = {(1, 2): PlayerEmbedding.from_state(torch.load(a.emb12, map_location="cpu", weights_only=False)),
            (3, 4): PlayerEmbedding.from_state(torch.load(a.emb34, map_location="cpu", weights_only=False))}
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    runs = [run_pool(games, folds_df, f, s, doc, embs[f], store, a) for f in ((1, 2), (3, 4)) for s in a.seeds]
    comps, checks = {}, {}
    for s in SCENARIOS:
        pid = np.concatenate([r["per_q"][f"{s}:pid"] for r in runs])
        for x, z in (("block_auto", "plain"), ("block_auto2", "plain"), ("block_auto2", "block_auto"), ("block_force", "block_auto2"), ("block_auto", "cap"), ("csls", "plain"),
                     ("whole_auto", "plain")):
            qx, qz = (np.concatenate([r["per_q"][f"{s}:{k}"] for r in runs]) for k in (x, z))
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            d["per_pool"] = [float(np.mean(r["per_q"][f"{s}:{x}"] - r["per_q"][f"{s}:{z}"])) for r in runs]
            comps[f"{s}: {x} vs {z}"] = d
            print(f"[{s}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in d["per_pool"]), flush=True)
        checks[s] = {"whole_auto_never_applies": not any(r["scenarios"][s]["fired"]["whole"] for r in runs),
                     "A_applies": all(r["scenarios"][s]["fired"]["A"] for r in runs),
                     "added_applies": [bool(r["scenarios"][s]["fired"]["added"]) for r in runs],
                     "added_applies_v2": [bool(r["scenarios"][s]["fired_v2"]["added"]) for r in runs],
                     "block_auto2_never_worse": all(v >= 0 for v in comps[f"{s}: block_auto2 vs plain"]["per_pool"]),
                     "block_auto_never_worse": all(v >= 0 for v in comps[f"{s}: block_auto vs plain"]["per_pool"]),
                     "A_block_equals_A_alone": all(abs(r["scenarios"][s]["arms"]["block_auto"]["A"] - r["A_alone"]["assign"]["score"]) < 1e-12 for r in runs)}
        print(f"checks [{s}]: {checks[s]}", flush=True)
    for r in runs:
        r.pop("per_q")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json({"pools": runs, "comparisons": comps, "checks": checks, "fusion": a.fusion, "emb": [a.emb12, a.emb34]}, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
