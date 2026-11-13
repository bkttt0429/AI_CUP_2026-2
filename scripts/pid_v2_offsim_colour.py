#!/usr/bin/env python
"""Colour-matched prototypes for the sequence component (METHOD_NOTES 2.71).

A candidate's prototype is the normalised sum of the vectors of ALL its reference games. The White deficit of the one-game
questions is an opening effect (2.56): as White a player answers, as Black he leads, so a White question game may be matched
better against the candidate's White games. Here every question game is scored against a prototype in which the candidate's
games of the same colour weigh 1 + a and the others 1 - a:
    s(q, c) = sum over the question's games g of  v_g . unit(sum over c's games h of w(colour_g, colour_h) v_h)
a = 0 is the usual prototype (up to a factor per question, which the row standardisation removes); a = 1 uses only the games
of the same colour (all games when the candidate has none of that colour).
Official pseudo-pools of scripts/pid_v2_offsim.py (3 of every candidate's 10 games held out; both colours), one fusion document
at its 50-step vectors; everything else of the document unchanged. The a used on a pool is the best one on the OTHER pool.
Reported: the sequence component alone and the fused score by a, differences from a = 0 (bootstrap over candidates), by
question size and by the colour of one-game questions.
"""
import argparse
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
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_offsim import SIZES, boot_mix, mix, per_question  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def seq_colour(vecs, cands, players, questions, alpha):
    """Mean over the networks of the row-standardised colour-weighted matrices [Q, C]."""
    mats = []
    for one in vecs:
        PB = np.stack([np.sum([one[k] for k, c in cands[p] if c == "B"] or [np.zeros(len(next(iter(one.values()))))], axis=0) for p in players]).astype(np.float64)
        PW = np.stack([np.sum([one[k] for k, c in cands[p] if c != "B"] or [np.zeros(PB.shape[1])], axis=0) for p in players]).astype(np.float64)
        proto = {"B": (1 + alpha) * PB + (1 - alpha) * PW, "W": (1 - alpha) * PB + (1 + alpha) * PW}
        for col in ("B", "W"):  # a candidate without games of that colour at a = 1: all its games
            empty = np.linalg.norm(proto[col], axis=1) < 1e-9
            proto[col][empty] = (PB + PW)[empty]
            proto[col] = unit(proto[col])
        QB = np.stack([np.sum([one[k] for k, c in q if c == "B"] or [np.zeros(PB.shape[1])], axis=0) for q in questions]).astype(np.float64)
        QW = np.stack([np.sum([one[k] for k, c in q if c != "B"] or [np.zeros(PB.shape[1])], axis=0) for q in questions]).astype(np.float64)
        mats.append(standardise(QB @ proto["B"].T + QW @ proto["W"].T))
    return np.mean(mats, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2w_n8.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    runs = []
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
                h = [(k, c) for k, c in hold[p] if k not in bad]
                for g in h:
                    questions.append([g]); truth.append(i)
                for m in (2, 3):
                    if len(h) >= m:
                        questions.append(h[:m]); truth.append(i)
            truth, n = np.array(truth), np.array([len(q) for q in questions])
            col1 = np.array([q[0][1] if len(q) == 1 else "-" for q in questions])
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            cache = {}
            seqs = []
            for v, tg in nets:
                f = Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz"
                if f not in cache:
                    cache[f] = load(f)
                seqs.append(cache[f])
            usual = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=seqs)["seq"])
            seq_a = {al: standardise(seq_colour(seqs, ref, players, questions, al)) for al in ALPHAS}
            fused = {al: per_question(fus.scores([seq_a[al] if c == "seq" else Z[c] for c in comps]), truth) for al in ALPHAS}
            alone = {al: per_question(seq_a[al], truth) for al in ALPHAS}
            chk = per_question(fus.scores([usual if c == "seq" else Z[c] for c in comps]), truth)
            share_b = float(np.mean([c == "B" for g in ref.values() for _, c in g]))
            runs.append({"pool": pool, "seed": s, "fused": fused, "alone": alone, "truth": truth, "n": n, "col1": col1})
            print(f"{pool} split {s}: {len(questions)} questions; reference games Black {share_b:.2f}; usual document {mix(chk, n):.4f}, a=0 here {mix(fused[0.0], n):.4f}; "
                  + "  ".join(f"a={al:g} fused {mix(fused[al], n):.4f} / alone {mix(alone[al], n):.4f}" for al in ALPHAS) + f" ({time.time() - t0:.0f} s)", flush=True)
    cat = lambda f: np.concatenate([f(r) for r in runs])  # noqa: E731
    n_all = cat(lambda r: r["n"])
    cl_all = cat(lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
    col_all = cat(lambda r: r["col1"])
    out = {"document": Path(a.fusion).name, "alphas": list(ALPHAS), "fused": {}, "alone": {}}
    for kind in ("alone", "fused"):
        base = cat(lambda r: r[kind][0.0])
        for al in ALPHAS:
            x = cat(lambda r: r[kind][al])
            rec = {"mix": mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}, "one_game_black": float(x[col_all == "B"].mean()), "one_game_white": float(x[col_all == "W"].mean())}
            msg = f"{kind} a={al:g}: mix {rec['mix']:.4f} (n=1 {rec['1']:.4f} [Black {rec['one_game_black']:.4f}, White {rec['one_game_white']:.4f}], n=2 {rec['2']:.4f}, n=3 {rec['3']:.4f})"
            if al:
                b = boot_mix(x - base, n_all, cl_all, a.boot, rng)
                rec["vs_a0"] = b
                msg += f"  vs a=0 {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
            out[kind][str(al)] = rec
            print(msg, flush=True)
    chosen = {}
    for pool in ("pub2", "priv2"):  # the a for a pool comes from the other pool
        others = [r for r in runs if r["pool"] != pool]
        chosen[pool] = max(ALPHAS, key=lambda al: mix(np.concatenate([r["fused"][al] for r in others]), np.concatenate([r["n"] for r in others])))
    d = cat(lambda r: r["fused"][chosen[r["pool"]]] - r["fused"][0.0])
    b = boot_mix(d, n_all, cl_all, a.boot, rng)
    b["by_n"] = {str(m): float(d[n_all == m].mean()) for m in SIZES}
    b["chosen"] = chosen
    out["cross_pool_choice"] = b
    print(f"a chosen on the other pool {chosen}: fused {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
