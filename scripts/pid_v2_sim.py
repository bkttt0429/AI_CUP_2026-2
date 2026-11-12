#!/usr/bin/env python
"""Simulation of the 11-08 v2 added pools on training players (METHOD_NOTES 2.29): fusion weights, component choice and the
assignment rule for candidates with 10 games of either colour and questions of 1 / 2 / 3 / 5 games of either colour.

Per pool (a fold pair, never fold 0; HD >= 2 games left out, as the official added pools have none): 600 players, 10 random
games each as candidates. Two question blocks of 400 (100 per size): D = 400 distinct owners (at most one question per
candidate), R = owners drawn with replacement; question games are disjoint from the candidate games and from each other.
Components: aicup_pid.v2.components (per-view opening / raw opening / orientation / handicap, merged local / style, embedding
of the other fold pair trained by pid_v2_embed.py). Weights: multinomial likelihood (aicup_pid.fusion), fitted on the other
pools, scored on this one.
Fixed before the run:
  arms  all = opening, opening_raw, orient4, local, style, hd, emb | sym = all without emb | lsplit = all with local_split for
        local | omerge = all + orient4_merged | emb = emb alone
  choice = the arm with the highest cross-fitted official score over both blocks without assignment; its weights are refitted
        on every pool and written to --fusion-out
  assignment: reported per block for the chosen arm (rule v2 of aicup_pid.checks, and forced); the official run applies rule
        v2 as it is. Report only: weights fitted per question size.
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
from aicup.sgf import handicap  # noqa: E402
from aicup_pid import pools, v2  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.checks import top1_collisions  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402

N_CAND, CAND_GAMES, N_Q, SIZES, TAKE = 600, 10, 400, (1, 2, 3, 5), 32
ARMS = {"all": ("opening", "opening_raw", "orient4", "local", "style", "hd", "emb"), "sym": ("opening", "opening_raw", "orient4", "local", "style", "hd"),
        "lsplit": ("opening", "opening_raw", "orient4", "local_split", "style", "hd", "emb"), "omerge": ("opening", "opening_raw", "orient4", "orient4_merged", "local", "style", "hd", "emb"),
        "emb": ("emb",)}


def build(games, folds, seed):
    rng = np.random.RandomState(seed)
    g = games[games.fold.isin(folds)]
    n = g.groupby("player_id").size()
    elig = sorted(n[n >= TAKE].index)
    drawn = rng.choice(elig, N_CAND + 150, replace=False).tolist()  # some spare players: a few have too few games without a handicap >= 2
    chosen = drawn
    colour = dict(zip(g.game_key, g.color))
    by = g[g.player_id.isin(chosen)].groupby("player_id")["game_key"].apply(sorted).to_dict()
    take = {p: pools._player_keys(by[p], "shuffle", seed, p)[:TAKE] for p in chosen}
    sgfs = {k: v[0] for k, v in load_games(sorted(k for ks in take.values() for k in ks), aicup_pid.TRAIN_DIR).items()}
    ok = {p: [k for k in ks if (handicap(sgfs[k]) or 0) < 2] for p, ks in take.items()}
    chosen = sorted([p for p in drawn if len(ok[p]) >= CAND_GAMES + max(SIZES)][:N_CAND])
    if len(chosen) < N_CAND:
        raise ValueError(f"only {len(chosen)} players with enough games")
    cands = {p: [(k, colour[k]) for k in ok[p][:CAND_GAMES]] for p in chosen}
    used = {p: CAND_GAMES for p in chosen}
    questions, owner, block = [], [], []

    def add(p, m, b):
        if used[p] + m > len(ok[p]):
            return False
        questions.append([(k, colour[k]) for k in ok[p][used[p]:used[p] + m]])
        owner.append(p); block.append(b)
        used[p] += m
        return True

    for p, m in zip(rng.choice(chosen, N_Q, replace=False), rng.permutation(np.repeat(SIZES, N_Q // len(SIZES)))):
        add(p, int(m), "D")
    skipped = sum(not add(p, int(m), "R") for p, m in zip(rng.choice(chosen, N_Q, replace=True), rng.permutation(np.repeat(SIZES, N_Q // len(SIZES)))))
    keys = {k for ks in cands.values() for k, _ in ks} | {k for q in questions for k, _ in q}
    return cands, questions, np.array(owner), np.array(block), {k: sgfs[k] for k in keys}, {"eligible": len(elig), "skipped_R": int(skipped)}


def score(idx, truth):
    out = np.zeros(len(truth))
    for q in range(len(truth)):
        hit = np.where(idx[q] == truth[q])[0]
        out[q] = np.exp(-hit[0]) if len(hit) else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion3o_ref_trunk_embt.json"))
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb2_p12.pt"))
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb2_p34.pt"))
    ap.add_argument("--emb-final", default="results/embed/emb2_final.pt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    a = ap.parse_args(argv)
    import torch
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    embs = {(1, 2): PlayerEmbedding.from_state(torch.load(a.emb12, map_location="cpu", weights_only=False)),
            (3, 4): PlayerEmbedding.from_state(torch.load(a.emb34, map_location="cpu", weights_only=False))}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    games = games[games.game_key.isin(rows.keys())]
    print(f"{len(rows)} rows, {len(games)} games with rows ({time.time() - t0:.0f} s)", flush=True)
    runs = []
    for f in ((1, 2), (3, 4)):
        for s in a.seeds:
            t1 = time.time()
            cands, questions, owner, block, sgfs, info = build(games, f, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, embs[f])
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            n = np.array([len(q) for q in questions])
            both = np.array([len({c for _, c in q}) > 1 for q in questions])
            runs.append({"S": {k: standardise(v) for k, v in S.items()}, "truth": truth, "n": n, "block": block, "owner": owner, "both": both,
                         "info": {**info, "folds": list(f), "seed": s, "questions": len(questions),
                                  "cand_black_games": float(np.mean([sum(c == "B" for _, c in v_) for v_ in cands.values()]))}})
            alone = {k: float(score(topk(v, 5), truth).mean()) for k, v in runs[-1]["S"].items()}
            print(f"pool {f} seed {s}: {info}, {len(questions)} questions, {time.time() - t1:.0f} s; alone " + " ".join(f"{k} {v:.3f}" for k, v in alone.items()), flush=True)
    out = {"pools": [r["info"] for r in runs], "arms": {}, "alone": {}}
    for c in runs[0]["S"]:
        sc = np.concatenate([score(topk(r["S"][c], 5), r["truth"]) for r in runs])
        nn = np.concatenate([r["n"] for r in runs])
        out["alone"][c] = {"score": float(sc.mean()), "by_n": {str(m): float(sc[nn == m].mean()) for m in SIZES}}
    per_q, fused = {}, {}
    for arm, comps in ARMS.items():
        sc_all, ws = [], []
        for i, r in enumerate(runs):
            others = [o for j, o in enumerate(runs) if j != i and o["info"]["folds"] != r["info"]["folds"]]  # weights from the other fold pair's pools
            fus = Fusion()
            if len(comps) == 1:
                fus.w = np.ones(1)
            else:
                fus.fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            F = fus.scores([r["S"][c] for c in comps])
            fused[(arm, i)] = F
            sc_all.append(score(topk(F, 5), r["truth"]))
            ws.append(fus.w.tolist())
        per_q[arm] = np.concatenate(sc_all)
        nn, bl, bo = (np.concatenate([r[k] for r in runs]) for k in ("n", "block", "both"))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": [float(x.mean()) for x in sc_all], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "by_block": {b: float(per_q[arm][bl == b].mean()) for b in ("D", "R")}, "both_colours": float(per_q[arm][bo].mean()), "one_colour": float(per_q[arm][~bo].mean()),
                            "weights_mean": dict(zip(comps, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(v, 3) for k, v in out['arms'][arm]['by_n'].items()})}  pools "
              + " ".join(f"{x:.3f}" for x in out['arms'][arm]['per_pool']) + f"  weights {out['arms'][arm]['weights_mean']}", flush=True)
    best = max(ARMS, key=lambda k: out["arms"][k]["score"])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out["comparisons"] = {}
    for x, z in (("all", "sym"), ("all", "emb"), ("lsplit", "all"), ("omerge", "all")):
        d = paired_player_bootstrap(per_q[x], per_q[z], pid, n=a.n_boot)
        out["comparisons"][f"{x} vs {z}"] = d
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    # assignment on the chosen arm, block by block
    asg = {b: {"plain": [], "rule": [], "forced": [], "fired": [], "conf_half": [], "distinct": []} for b in ("D", "R")}
    for i, r in enumerate(runs):
        F = fused[(best, i)]
        for b in ("D", "R"):
            m = r["block"] == b
            col = top1_collisions(F[m], r["n"][m])
            plain, forced = score(topk(F[m], 5), r["truth"][m]), score(assign_topk(F[m], 5), r["truth"][m])
            asg[b]["plain"].append(float(plain.mean())); asg[b]["forced"].append(float(forced.mean()))
            asg[b]["rule"].append(float((forced if col["one_to_one_likely_v2"] else plain).mean()))
            asg[b]["fired"].append(bool(col["one_to_one_likely_v2"])); asg[b]["conf_half"].append(round(col["confident_distinct"], 3))
            asg[b]["distinct"].append(round(col["distinct_top1"] / col["n_questions"], 3))
    out["assignment"] = {b: {**{k: float(np.mean(v[k])) for k in ("plain", "rule", "forced")}, "fired": v["fired"], "conf_half": v["conf_half"], "distinct": v["distinct"]} for b, v in asg.items()}
    for b, v in out["assignment"].items():
        print(f"assignment [{best}] block {b}: plain {v['plain']:.4f}, rule v2 {v['rule']:.4f}, forced {v['forced']:.4f}; fired {v['fired']}; confident-half distinct {v['conf_half']}; distinct Top-1 {v['distinct']}", flush=True)
    # report only: weights per question size
    comps = ARMS[best]
    sc_n = []
    for i, r in enumerate(runs):
        others = [o for o in runs if o["info"]["folds"] != r["info"]["folds"]]
        s_i = np.zeros(len(r["truth"]))
        for m in SIZES:
            fus = Fusion().fit([np.vstack([o["S"][c][o["n"] == m] for o in others]) for c in comps], np.concatenate([o["truth"][o["n"] == m] for o in others]))
            rows_m = r["n"] == m
            s_i[rows_m] = score(topk(fus.scores([r["S"][c][rows_m] for c in comps]), 5), r["truth"][rows_m])
            out.setdefault("weights_by_n", {}).setdefault(str(m), []).append(fus.w.tolist())
        sc_n.append(s_i)
    d = paired_player_bootstrap(np.concatenate(sc_n), per_q[best], pid, n=a.n_boot)
    out["per_size_weights_vs_one"] = d
    out["weights_by_n"] = {m: dict(zip(comps, np.round(np.mean(w, axis=0), 3).tolist())) for m, w in out["weights_by_n"].items()}
    print(f"weights per size vs one set [{best}]: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]; {out['weights_by_n']}", flush=True)
    out["chosen"] = best
    fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps], np.concatenate([r["truth"] for r in runs]))
    doc = {"components": list(comps), "weights": fus.w.tolist(), "cfg": cfg, "emb_model": a.emb_final, "geometry": {"candidates": N_CAND, "candidate_games": CAND_GAMES, "sizes": list(SIZES)},
           "fitted_on": [r["info"] for r in runs], "questions": int(sum(len(r["truth"]) for r in runs)), "simulated_score": out["arms"][best]["score"],
           "note": "v2 added pools (pub2_ / priv2_): both colours, per-view models (aicup_pid.v2); no official game enters any training"}
    print(f"chosen arm {best}: weights {dict(zip(comps, np.round(fus.w, 3).tolist()))}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    if a.fusion_out:
        Path(a.fusion_out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
