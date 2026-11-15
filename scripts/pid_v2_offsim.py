#!/usr/bin/env python
"""The whole added-pool pipeline scored on OFFICIAL players: labelled pseudo-pools cut from each official pool's own candidates
(METHOD_NOTES 2.41). The organisers give the owner of every candidate game (600 players x 10 games; candidate games may be
used for calibration, Q&A Q10). Per pool and split seed, aicup_pid.v2.candidate_holdout takes h = 3 games of every candidate
out: the other 7 are the candidate's reference games, the 3 make its questions
  n = 1: each held-out game (1,800 questions)   n = 2: the first two (600)   n = 3: all three (600)
The sequence networks are fine-tuned on the reference games only (scripts/pid_v2_seq_finetune.py --holdout 3), so question
games are as unseen as official question games. Everything else is the official prediction path (predict_pid_v2.predict_pool:
the fusion document's components, embedding model and weights). Pools are never mixed: weights are fitted and scored inside
one pool. Differences from the real pools: 7 reference games instead of 10, no 5-game questions.

Reported per pool and for both: the score by question size and their mean ("mix") for the fine-tune steps asked for, with
the document's weights; paired differences between steps (bootstrap over candidates); and the value of calibrating the
fusion weights on the pool's own candidates: weights fitted on the questions of one half of the candidates, scored on the
other half, both ways (fit set: one 1-game question per candidate plus its 2- and 3-game questions).
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

SIZES = (1, 2, 3)


def per_question(F, truth):
    rank = (F > F[np.arange(len(truth)), truth][:, None]).sum(1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def mix(sc, n, sel=None):
    sel = np.ones(len(n), bool) if sel is None else sel
    return float(np.mean([sc[sel & (n == m)].mean() for m in SIZES]))


def boot_mix(d, n, cluster, B, rng):
    """Bootstrap over clusters of the size-balanced mean of d."""
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])  # [3, K]
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    bs = (num[:, pick].sum(2) / den[:, pick].sum(2)).mean(0)
    return {"diff": float((num.sum(1) / den.sum(1)).mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2ge.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50, 100])
    ap.add_argument("--alt", default=None, help="a second fusion document (another network composition): scored at --alt-step next to the first, paired difference")
    ap.add_argument("--alt-step", type=int, default=50)
    ap.add_argument("--mix", nargs="*", type=int, default=[], help="an extra arm 'mix': every network's vectors after each of these step counts, all averaged (e.g. 50 100)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq" and isinstance(doc["emb_model"], str)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets_of = lambda d_: [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in d_["seq_official"]]  # noqa: E731  (net, tag) of every network of a document
    nets = nets_of(doc)
    alt = json.loads(Path(a.alt).read_text()) if a.alt else None
    assert alt is None or list(alt["components"]) == comps
    print(f"document {Path(a.fusion).name}: {len(nets)} sequence networks, weights {dict(zip(comps, np.round(doc['weights'], 3).tolist()))}", flush=True)
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    runs, half = [], {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        half[pool] = rng.permutation(len(players)) % 2  # the two halves of the candidates for the weight calibration
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        for s in a.seeds:
            hold = v2.candidate_holdout(cands, a.holdout, s)
            held = {k for g in hold.values() for k, _ in g}
            ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
            questions, truth, primary = [], [], []
            for i, p in enumerate(players):
                h = [(k, c) for k, c in hold[p] if k not in bad]
                for j, g in enumerate(h):
                    questions.append([g]); truth.append(i); primary.append(j == 0)
                for m in (2, 3):
                    if len(h) >= m:
                        questions.append(h[:m]); truth.append(i); primary.append(True)
            truth, primary, n = np.array(truth), np.array(primary), np.array([len(q) for q in questions])
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            by_t = {}
            for t in a.steps:
                by_t[t] = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{t}.npz") for v, tg in nets]
                Z[("seq", t)] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=by_t[t])["seq"])
            if alt is not None:
                seqs = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.alt_step}.npz") for v, tg in nets_of(alt)]
                cfg_alt = {**doc["cfg"], **({"seq_near": alt["cfg"]["seq_near"]} if "seq_near" in alt.get("cfg", {}) else {})}  # the alternative's own nearest-game setting (METHOD_NOTES 2.141)
                Z[("seq", "alt")] = standardise(v2.components(ref, questions, moves, sgfs, cfg_alt, which=("seq",), seq=seqs)["seq"])
            if a.mix:
                Z[("seq", "mix")] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=[x for t in a.mix for x in by_t[t]])["seq"])
            runs.append({"pool": pool, "seed": s, "Z": Z, "truth": truth, "n": n, "primary": primary})
            alone = {t: mix(per_question(Z[("seq", t)], truth), n) for t in a.steps}
            print(f"{pool} split {s}: {len(questions)} questions of {len(players)} candidates ({len(bad)} games left out); sequence component alone (mix) "
                  + "  ".join(f"t{t} {x:.4f}" for t, x in alone.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    mats = lambda r, t: [r["Z"][("seq", t)] if c == "seq" else r["Z"][c] for c in comps]  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    out = {"document": Path(a.fusion).name, "holdout": a.holdout, "seeds": a.seeds, "steps": a.steps, "scores": {}, "steps_diff": {}, "calibration": {}}
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    groups = {"pub2": [r for r in runs if r["pool"] == "pub2"], "priv2": [r for r in runs if r["pool"] == "priv2"], "both": runs}
    arms = [*a.steps, *(["mix"] if a.mix else []), *(["alt"] if alt is not None else [])]
    fus_alt = Fusion()
    fus_alt.w = np.asarray((alt or doc)["weights"], dtype=float)
    sc = {(id(r), t): per_question((fus_alt if t == "alt" else fus).scores(mats(r, t)), r["truth"]) for r in runs for t in arms}
    for g, rs in groups.items():
        n = cat(rs, lambda r: r["n"])
        cluster = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
        out["scores"][g] = {}
        for t in arms:
            x = cat(rs, lambda r: sc[(id(r), t)])
            out["scores"][g][str(t)] = {"mix": mix(x, n), **{str(m): float(x[n == m].mean()) for m in SIZES}}
        print(f"{g}: document weights, by fine-tune steps: " + "  ".join(f"t{t} mix {v_['mix']:.4f} (n=1 {v_['1']:.4f}, n=2 {v_['2']:.4f}, n=3 {v_['3']:.4f})" for t, v_ in out["scores"][g].items()), flush=True)
        for x_, y_ in ((50, 0), (100, 0), (100, 50), *[(t, 50) for t in a.steps if t not in (0, 50, 100)], *[("mix", t) for t in a.mix], ("alt", a.alt_step), *([("alt", 50)] if a.alt_step != 50 else [])):
            if x_ in arms and y_ in arms:
                d = cat(rs, lambda r: sc[(id(r), x_)] - sc[(id(r), y_)])
                b = boot_mix(d, n, cluster, a.boot, rng)
                b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}
                out["steps_diff"][f"{g}: t{x_} vs t{y_}"] = b
                print(f"   t{x_} vs t{y_}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)
    # calibration of the weights on the pool's own candidates: fit on one half of the candidates, score on the other
    for t in [x for x in a.steps if x > 0]:
        cal = {}
        for pool in ("pub2", "priv2"):
            rs, H = groups[pool], half[pool]
            d_all, n_all, cl_all, w_fit = [], [], [], []
            for side in (0, 1):
                fit_rows = [np.flatnonzero((H[r["truth"]] == side) & r["primary"]) for r in rs]
                f2 = Fusion().fit([np.vstack(col) for col in zip(*[[M[i] for M in mats(r, t)] for r, i in zip(rs, fit_rows)])], np.concatenate([r["truth"][i] for r, i in zip(rs, fit_rows)]))
                w_fit.append(f2.w)
                for r in rs:
                    te = np.flatnonzero(H[r["truth"]] != side)
                    d_all.append(per_question(f2.scores([M[te] for M in mats(r, t)]), r["truth"][te]) - sc[(id(r), t)][te])
                    n_all.append(r["n"][te]); cl_all.append(r["truth"][te])
            d, n, cl = np.concatenate(d_all), np.concatenate(n_all), np.concatenate(cl_all)
            b = boot_mix(d, n, cl, a.boot, rng)
            b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}
            rows_all = [np.flatnonzero(r["primary"]) for r in rs]
            f_all = Fusion().fit([np.vstack(col) for col in zip(*[[M[i] for M in mats(r, t)] for r, i in zip(rs, rows_all)])], np.concatenate([r["truth"][i] for r, i in zip(rs, rows_all)]))
            b["weights_half"] = [dict(zip(comps, w.tolist())) for w in w_fit]
            b["weights_all"] = dict(zip(comps, f_all.w.tolist()))
            cal[pool] = b
            print(f"calibrated vs document weights, t{t}, {pool}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}; "
                  f"weights fitted on all its candidates {({c: round(w, 2) for c, w in b['weights_all'].items()})}", flush=True)
        out["calibration"][str(t)] = cal
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
