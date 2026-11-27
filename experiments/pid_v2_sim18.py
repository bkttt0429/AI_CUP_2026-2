#!/usr/bin/env python
"""Where the added-pool pipeline loses (METHOD_NOTES 2.48): the per-question scores of the simulated pools (the multi-network
reference of pid_v2_sim11.py's arm ft50, weights cross-fitted) broken down by what is known about a question:
  owner's rank class; for 1-game questions the game's length, the owner's colour and the handicap flag;
  the owner's own consistency (mean cosine between the owner's ten candidate games and their prototype);
  for wrong first places: whether the candidate put first has the owner's rank class (the model mistakes equally strong
  players for each other) and how far apart the two classes are.
Label-free comparison with the official pools: the same descriptive quantities of the official 1-game questions and
candidates (length, colour, handicap, consistency), to see whether the official pools are a harder mix.
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
from aicup.metric import LEVELS  # noqa: E402
from aicup.sgf import handicap  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
LEN_BINS = (0, 100, 150, 200, 256, 10 ** 6)
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731


def consistency(vec, cands, players):
    """Per candidate: mean cosine of its games to its own prototype without the game (one network's vectors)."""
    out = []
    for p in players:
        E = unit(np.stack([vec[k] for k, _ in cands[p]]).astype(np.float64))
        tot = E.sum(0)
        out.append(float(np.mean([e @ unit(tot - e) for e in E])))
    return np.array(out)


def table(name, keys, val, out, fmt="{:.3f}"):
    rows = {}
    for k in sorted(set(keys.tolist()), key=lambda x: (str(type(x)), x)):
        m = keys == k
        rows[str(k)] = {"n": int(m.sum()), "score": float(val[m].mean())}
    out[name] = rows
    print(f"{name}: " + "  ".join(f"{k} {fmt.format(r['score'])} ({r['n']})" for k, r in rows.items()), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--official-seq", default=str(ROOT / "data" / "seq_ft" / "official_{pool}_net-1_m90s0e_t0.npz"))
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rank_of = dict(zip(games.player_id, games["rank"]))
    lvl = {r: i for i, r in enumerate(LEVELS)}
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    ft = lambda v_, s, name: Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if name == "m88" else f"sim_f{v_}_s{s}{nets[name][1]}_t50.npz")  # noqa: E731
    first = [n for n in nets if n != "m88"][0]
    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        raw0 = vecs(ROOT / "data" / "seq" / f"fold{v_}{nets[first][0]}.npz")  # one network before the fine-tune, for the consistency
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            Z = {c: standardise(x) for c, x in S.items()}
            Z["seq"] = standardise(np.mean([standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(ft(v_, s, name)))["seq"]) for name in nets], axis=0))
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            q1 = [q[0] for q in questions]
            runs.append({"Z": Z, "truth": truth, "n": np.array([len(q) for q in questions]), "fold": v_, "prank": np.array([lvl[rank_of[p]] for p in players]),
                         "len1": np.array([len(moves[k]) for k, _ in q1]), "col1": np.array([c for _, c in q1]), "hd1": np.array([handicap(sgfs[k]) or 0 for k, _ in q1]),
                         "cons": consistency(raw0, cands, players), "clen": np.array([np.mean([len(moves[k]) for k, _ in cands[p]]) for p in players])})
            print(f"pool fold {v_} seed {s} ({time.time() - t0:.0f} s)", flush=True)
    sc, top1, pred = [], [], []
    for r in runs:
        others = [o for o in runs if o["fold"] != r["fold"]]
        fus = Fusion().fit([np.vstack([o["Z"][c] for o in others]) for c in COMPS], np.concatenate([o["truth"] for o in others]))
        idx = topk(fus.scores([r["Z"][c] for c in COMPS]), 5)
        sc.append(score(idx, r["truth"])); pred.append(idx[:, 0]); top1.append(idx[:, 0] == r["truth"])
    cat = lambda f: np.concatenate([f(r, i) for i, r in enumerate(runs)])  # noqa: E731
    s_all, n_all, hit = np.concatenate(sc), cat(lambda r, i: r["n"]), np.concatenate(top1)
    orank = cat(lambda r, i: r["prank"][r["truth"]])
    prank = cat(lambda r, i: r["prank"][pred[i]])
    out = {"score": float(s_all.mean()), "levels": list(LEVELS)}
    print(f"all questions: score {s_all.mean():.4f}, first place right {hit.mean():.3f}; rank classes in order: {list(LEVELS)}", flush=True)
    for m in SIZES:
        table(f"n={m} by owner's rank class", np.array(LEVELS)[orank[n_all == m]], s_all[n_all == m], out)
    one = n_all == 1
    ln, col, hd = cat(lambda r, i: r["len1"]), cat(lambda r, i: r["col1"]), cat(lambda r, i: r["hd1"])
    table("n=1 by game length (plies)", np.array([f"{LEN_BINS[b]}-{LEN_BINS[b + 1]}" for b in np.digitize(ln[one], LEN_BINS[1:-1])]), s_all[one], out)
    table("n=1 by owner's colour", col[one], s_all[one], out)
    table("n=1 by handicap flag", hd[one], s_all[one], out)
    cons = cat(lambda r, i: r["cons"][r["truth"]])
    qs = np.quantile(cons, [0.25, 0.5, 0.75])
    for m in (1, 2):
        table(f"n={m} by the owner's consistency quartile (low to high)", np.digitize(cons[n_all == m], qs), s_all[n_all == m], out)
    wrong = ~hit
    gap = np.abs(orank - prank)
    base_same = float(np.mean([np.mean(r["prank"][:, None] == r["prank"][None]) for r in runs]))
    out["wrong_first_place"] = {"n": int(wrong.sum()), "same_class": float((gap[wrong] == 0).mean()), "within_one_class": float((gap[wrong] <= 1).mean()), "same_class_by_chance": base_same,
                                "by_n": {str(m): {"n": int((wrong & (n_all == m)).sum()), "same_class": float((gap[wrong & (n_all == m)] == 0).mean()), "within_one": float((gap[wrong & (n_all == m)] <= 1).mean())} for m in SIZES}}
    w = out["wrong_first_place"]
    print(f"wrong first places ({w['n']}): candidate put first has the owner's rank class {w['same_class']:.3f}, within one class {w['within_one_class']:.3f} (two random candidates share a class {base_same:.3f}); "
          + "  ".join(f"n={m}: {x['same_class']:.3f}/{x['within_one']:.3f} ({x['n']})" for m, x in w["by_n"].items()), flush=True)
    sim_desc = {"q1_len": np.percentile(ln[one], [10, 25, 50, 75, 90]).tolist(), "q1_black": float((col[one] == "B").mean()), "q1_hd1": float((hd[one] > 0).mean()),
                "cand_len_mean": float(np.mean(cat(lambda r, i: r["clen"]))), "cand_consistency": np.percentile(cat(lambda r, i: r["cons"]), [10, 50, 90]).tolist()}
    out["simulated_pools"] = sim_desc
    print(f"simulated pools: 1-game question length p10/25/50/75/90 {[round(x) for x in sim_desc['q1_len']]}, Black {sim_desc['q1_black']:.3f}, handicap flag {sim_desc['q1_hd1']:.3f}; "
          f"candidate games mean length {sim_desc['cand_len_mean']:.0f}; candidates' consistency p10/50/90 {[round(x, 3) for x in sim_desc['cand_consistency']]}", flush=True)
    T = Path(a.tests_dir)
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, questions, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        moves = parse_all(sgfs, a.workers)
        players = sorted(cands)
        vec = vecs(a.official_seq.replace("{pool}", pool))
        q1 = [q[0] for q in questions if len(q) == 1]
        d = {"q1_len": np.percentile([len(moves[k]) for k, _ in q1], [10, 25, 50, 75, 90]).tolist(), "q1_black": float(np.mean([c == "B" for _, c in q1])), "q1_hd1": float(np.mean([(handicap(sgfs[k]) or 0) > 0 for k, _ in q1])),
             "cand_len_mean": float(np.mean([len(moves[k]) for p in players for k, _ in cands[p]])), "cand_consistency": np.percentile(consistency(vec, cands, players), [10, 50, 90]).tolist(),
             "q_len_all": np.percentile([len(moves[k]) for q in questions for k, _ in q], [10, 25, 50, 75, 90]).tolist()}
        out[f"official_{pool}"] = d
        print(f"official {pool}: 1-game question length p10/25/50/75/90 {[round(x) for x in d['q1_len']]}, Black {d['q1_black']:.3f}, handicap flag {d['q1_hd1']:.3f}; candidate games mean length {d['cand_len_mean']:.0f}; "
              f"candidates' consistency p10/50/90 {[round(x, 3) for x in d['cand_consistency']]}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
