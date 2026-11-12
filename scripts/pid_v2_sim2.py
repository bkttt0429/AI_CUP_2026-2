#!/usr/bin/env python
"""Second simulation of the v2 added pools (METHOD_NOTES 2.30): the rank session's sequence identity model as a component,
and a soft assignment for pools where most players have one question but some have several.

Pools are SINGLE-fold (fold v's players, their games under net v, which never saw them; scripts/pid_v2_seq_embed.py), 600
candidates x 10 games, block D (400 distinct owners) and block R (owners drawn with replacement), as pid_v2_sim.py. The
embedding emb2 comes from the other fold pair. Weights are fitted on the pools of the other folds.
Fixed before the run:
  arms    base = the arm chosen on 11-12 (opening, opening_raw, orient4, orient4_merged, local, style, hd, emb)
          base+seq | seq alone | sym+seq (no emb) | emb+seq (no habit components)
  choice  base+seq replaces base if its cross-fitted score is higher with a player-bootstrap CI above 0; the chosen arm's
          weights are refitted on every pool and written to --fusion-out
  soft assignment (aicup_pid.assign.assign_soft, penalty lam per extra question of a candidate, lam in 0.5 / 1 / 2 / 4 / 8):
          the official Public pool lost 0.0114 under the forced assignment, between block D (gain) and block R (loss), about a
          third of the way to R. Adopted lam = the one with the highest (2 D + R) / 3 gain over no assignment, and only if that
          gain is positive on D and not below -0.003 on R; otherwise no assignment.
Report only: an empirical-Bayes prior over candidates (EM on the mixture proportions).
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
from aicup_pid.assign import assign_soft, assign_topk  # noqa: E402
from aicup_pid.checks import top1_collisions  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402

N_CAND, CAND_GAMES, N_Q, SIZES, TAKE = 600, 10, 400, (1, 2, 3, 5), 32
BASE = ("opening", "opening_raw", "orient4", "orient4_merged", "local", "style", "hd", "emb")
ARMS = {"base": BASE, "base+seq": (*BASE, "seq"), "seq": ("seq",), "sym+seq": (*[c for c in BASE if c != "emb"], "seq"), "emb+seq": ("emb", "seq")}
LAMS = (0.5, 1.0, 2.0, 4.0, 8.0)


def build(games, seed):
    rng = np.random.RandomState(seed)
    n = games.groupby("player_id").size()
    elig = sorted(n[n >= TAKE].index)
    drawn = rng.choice(elig, min(len(elig), N_CAND + 150), replace=False).tolist()
    colour = dict(zip(games.game_key, games.color))
    by = games[games.player_id.isin(drawn)].groupby("player_id")["game_key"].apply(sorted).to_dict()
    take = {p: pools._player_keys(by[p], "shuffle", seed, p)[:TAKE] for p in drawn}
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


def eb_prior(F, iters=50, pseudo=0.5):
    """F + log prior, the prior over candidates estimated by EM from the questions themselves (mixture proportions)."""
    Q, C = F.shape
    logpi = np.zeros(C)
    for _ in range(iters):
        Z = F + logpi
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        pi = (P.sum(axis=0) + pseudo) / (Q + pseudo * C)
        logpi = np.log(pi * C)
    return F + logpi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2.json"))
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb2_p12.pt"))
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb2_p34.pt"))
    ap.add_argument("--emb-final", default="results/embed/emb2_final.pt")
    ap.add_argument("--seq-dir", default="/work/u2507115/pid_v2/seq")
    ap.add_argument("--seq-official", default="/work/u2507115/pid_v2/seq/official.npz")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    a = ap.parse_args(argv)
    import torch
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    emb_of = {}
    for name, path in (("12", a.emb12), ("34", a.emb34)):
        emb_of[name] = PlayerEmbedding.from_state(torch.load(path, map_location="cpu", weights_only=False))
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v in (1, 2, 3, 4):
        z = np.load(Path(a.seq_dir) / f"fold{v}.npz")
        seq = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        g = games[(games.fold == v) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq.keys())]
        print(f"fold {v}: {len(seq)} sequence vectors, {len(g)} usable games of {g.player_id.nunique()} players ({time.time() - t0:.0f} s)", flush=True)
        for s in a.seeds:
            t1 = time.time()
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb_of["12" if v in (1, 2) else "34"], which=(*BASE, "seq"), seq=seq)
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            runs.append({"S": {k: standardise(x) for k, x in S.items()}, "truth": truth, "n": np.array([len(q) for q in questions]), "block": block, "owner": owner,
                         "info": {**info, "fold": v, "seed": s, "questions": len(questions)}})
            alone = {k: float(score(topk(x, 5), truth).mean()) for k, x in runs[-1]["S"].items()}
            print(f"pool fold {v} seed {s}: {info}, {len(questions)} questions, {time.time() - t1:.0f} s; alone " + " ".join(f"{k} {x:.3f}" for k, x in alone.items()), flush=True)
    out = {"pools": [r["info"] for r in runs], "arms": {}, "alone": {}}
    nn, bl = (np.concatenate([r[k] for r in runs]) for k in ("n", "block"))
    for c in runs[0]["S"]:
        sc = np.concatenate([score(topk(r["S"][c], 5), r["truth"]) for r in runs])
        out["alone"][c] = {"score": float(sc.mean()), "by_n": {str(m): float(sc[nn == m].mean()) for m in SIZES}}
    per_q, fused = {}, {}
    for arm, comps in ARMS.items():
        sc_all, ws = [], []
        for i, r in enumerate(runs):
            others = [o for o in runs if o["info"]["fold"] != r["info"]["fold"]]
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
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": [float(x.mean()) for x in sc_all], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "by_block": {b: float(per_q[arm][bl == b].mean()) for b in ("D", "R")}, "weights_mean": dict(zip(comps, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  pools "
              + " ".join(f"{x:.3f}" for x in out['arms'][arm]['per_pool']) + f"  weights {out['arms'][arm]['weights_mean']}", flush=True)
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out["comparisons"] = {}
    for x, z_ in (("base+seq", "base"), ("base+seq", "sym+seq"), ("base+seq", "emb+seq"), ("seq", "base")):
        d = paired_player_bootstrap(per_q[x], per_q[z_], pid, n=a.n_boot)
        out["comparisons"][f"{x} vs {z_}"] = d
        print(f"{x} vs {z_}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    d = out["comparisons"]["base+seq vs base"]
    best = "base+seq" if d["diff"] > 0 and d["ci95"][0] > 0 else "base"
    # assignment variants on the chosen arm, block by block
    res = {b: {"plain": [], "forced": [], "eb": [], **{f"soft{lam:g}": [] for lam in LAMS}, "conf_half": [], "distinct": []} for b in ("D", "R")}
    for i, r in enumerate(runs):
        F = fused[(best, i)]
        for b in ("D", "R"):
            m = r["block"] == b
            Fb, tb = F[m], r["truth"][m]
            col = top1_collisions(Fb, r["n"][m])
            res[b]["plain"].append(float(score(topk(Fb, 5), tb).mean()))
            res[b]["forced"].append(float(score(assign_topk(Fb, 5), tb).mean()))
            res[b]["eb"].append(float(score(topk(eb_prior(Fb), 5), tb).mean()))
            for lam in LAMS:
                res[b][f"soft{lam:g}"].append(float(score(assign_soft(Fb, lam), tb).mean()))
            res[b]["conf_half"].append(round(col["confident_distinct"], 3)); res[b]["distinct"].append(round(col["distinct_top1"] / col["n_questions"], 3))
    out["assignment"] = {b: {k: (float(np.mean(x)) if k not in ("conf_half", "distinct") else x) for k, x in v_.items()} for b, v_ in res.items()}
    for b, v_ in out["assignment"].items():
        print(f"assignment [{best}] block {b}: plain {v_['plain']:.4f}, forced {v_['forced']:+.4f} vs plain -> " + ", ".join(f"{k} {v_[k] - v_['plain']:+.4f}" for k in ("forced", "eb", *[f"soft{lam:g}" for lam in LAMS]))
              + f"; confident-half {v_['conf_half']}; distinct {v_['distinct']}", flush=True)
    gain = {lam: {b: out["assignment"][b][f"soft{lam:g}"] - out["assignment"][b]["plain"] for b in ("D", "R")} for lam in LAMS}
    mix = {lam: (2 * g["D"] + g["R"]) / 3 for lam, g in gain.items()}
    lam_best = max(LAMS, key=lambda lam: mix[lam])
    adopt = bool(mix[lam_best] > 0 and gain[lam_best]["D"] > 0 and gain[lam_best]["R"] >= -0.003)
    out["soft"] = {"gain": {f"{k:g}": v_ for k, v_ in gain.items()}, "mix": {f"{k:g}": v_ for k, v_ in mix.items()}, "best": lam_best, "adopt": adopt}
    print(f"soft assignment: (2D + R) / 3 gains {({f'{k:g}': round(v_, 4) for k, v_ in mix.items()})}; best lam {lam_best:g}; adopt {adopt}", flush=True)
    out["chosen"] = best
    comps = ARMS[best]
    fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps], np.concatenate([r["truth"] for r in runs]))
    doc = {"components": list(comps), "weights": fus.w.tolist(), "cfg": cfg, "emb_model": a.emb_final, "seq_official": a.seq_official, "soft_lam": float(lam_best) if adopt else None,
           "geometry": {"candidates": N_CAND, "candidate_games": CAND_GAMES, "sizes": list(SIZES)}, "fitted_on": [r["info"] for r in runs],
           "questions": int(sum(len(r["truth"]) for r in runs)), "simulated_score": out["arms"][best]["score"],
           "note": "v2 added pools: both colours, per-view models, emb2 and the rank session's sequence identity model (net -1 for official games); no official game enters any training"}
    print(f"chosen arm {best}: weights {dict(zip(comps, np.round(fus.w, 3).tolist()))}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    if a.fusion_out:
        Path(a.fusion_out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
