#!/usr/bin/env python
"""A per-game embedding on the UNPROJECTED trunk statistics as a new fusion component, scored on OFFICIAL players
(METHOD_NOTES 2.75).

Probe A (2.69 c): the same small network (aicup_pid.embed.PlayerEmbedding) scores 0.28 on the usual 256-dim rows (mean / std
of the 64-dim random projections) and 0.48 on the 1,536-dim rows of the unprojected vectors. Here that network is trained on
training players' unprojected rows (--train-rows: directories of scripts/pid_trunk_extract.py --full parts; folds 1-4 only,
never fold 0) and scored on the official pseudo-pools of scripts/pid_v2_offsim.py (3 of every candidate's 10 games held out;
rows of the official games in --pool-rows). Reported: the new component alone, next to the document's emb and seq components;
and added to the document's fused score with a weight chosen on the OTHER pool (bootstrap over candidates).
"""
import argparse
import glob
import json
import re
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
from aicup.submission import check_game  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding, score_embed  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_offsim import SIZES, boot_mix, mix, per_question  # noqa: E402

WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def load_full(dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if "full" in z.files:
                out.update(zip(z["keys"].astype(str), z["full"].astype(np.float32)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2w_n8.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--train-rows", nargs="+", default=["/work/u2507115/pid_v2/fullrows/rows"])
    ap.add_argument("--pool-rows", nargs="+", default=["/work/u2507115/pid_v2/fullrows_off2"])
    ap.add_argument("--emb-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--emb-steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--max-players", type=int, default=0, help="train on at most this many players (0: all), to read the trend by the number of players")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save", default=None, help="write the trained embedding models here (.pt)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    owner_of = dict(zip(games.game_key[games.fold != 0], games.player_id[games.fold != 0]))
    tr = load_full(a.train_rows)
    by = {}
    for k in tr:
        p = owner_of.get(k)  # fold-0 games and games of other files have no owner here: left out
        if p is not None:
            by.setdefault(p, []).append(k)
    by = {p: ks for p, ks in sorted(by.items()) if len(ks) >= 8}
    if a.max_players and len(by) > a.max_players:
        keep = np.random.default_rng(1).permutation(len(by))[:a.max_players]
        names = list(by)
        by = {names[i]: by[names[i]] for i in sorted(keep.tolist())}
    rows_by = {p: np.stack([tr[k] for k in ks]) for p, ks in by.items()}
    print(f"training rows: {sum(len(v_) for v_ in by.values())} games of {len(by)} players (folds 1-4) ({time.time() - t0:.0f} s)", flush=True)
    models = [PlayerEmbedding(seed=s, p_aug=0.0, steps=a.emb_steps, hidden=a.hidden).fit(rows_by) for s in a.emb_seeds]
    print(f"{len(models)} embedding networks trained; episode top-1 at the end {np.mean([m.history[-1]['acc'] for m in models]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    if a.save:
        torch.save([m.state_dict() for m in models], a.save)
    del tr, rows_by
    pool_rows = load_full(a.pool_rows)

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
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store or k not in pool_rows}
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
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            cache, seqs = {}, []
            for v, tg in nets:
                f = Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz"
                if f not in cache:
                    cache[f] = load(f)
                seqs.append(cache[f])
            Z["seq"] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=seqs)["seq"])
            F = fus.scores([Z[c] for c in comps])
            ck = {p: [k for k, _ in ref[p]] for p in players}
            qk = [[k for k, _ in q] for q in questions]
            E = np.mean([standardise(score_embed(m, ck, qk, pool_rows)) for m in models], axis=0)
            E = standardise(E)
            runs.append({"pool": pool, "seed": s, "F": F, "E": E, "truth": truth, "n": n, "emb": Z["emb"], "seq": Z["seq"]})
            print(f"{pool} split {s}: {len(questions)} questions ({len(bad)} games left out); alone: new {mix(per_question(E, truth), n):.4f}, document's emb {mix(per_question(Z['emb'], truth), n):.4f}, "
                  f"seq {mix(per_question(Z['seq'], truth), n):.4f}; document {mix(per_question(F, truth), n):.4f} ({time.time() - t0:.0f} s)", flush=True)
    cat = lambda f: np.concatenate([f(r) for r in runs])  # noqa: E731
    n_all = cat(lambda r: r["n"])
    cl_all = cat(lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
    base = cat(lambda r: per_question(r["F"], r["truth"]))
    out = {"document": Path(a.fusion).name, "train_players": len(by), "alone": {}, "curve": {}}
    for name in ("E", "emb", "seq", "F"):
        x = cat(lambda r: per_question(r[name], r["truth"]))
        out["alone"][name] = {"mix": mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}}
    print("both pools, alone: " + "  ".join(f"{k} {v_['mix']:.4f} (n=1 {v_['1']:.4f}, n=2 {v_['2']:.4f}, n=3 {v_['3']:.4f})" for k, v_ in out["alone"].items()), flush=True)
    for w in WEIGHTS:
        d = cat(lambda r: per_question(r["F"] + w * r["E"], r["truth"])) - base
        out["curve"][str(w)] = float(mix(d, n_all))
    print("fused gain by weight of the new component: " + "  ".join(f"{w}: {v_:+.4f}" for w, v_ in out["curve"].items()), flush=True)
    chosen = {}
    for pool in ("pub2", "priv2"):  # the weight for a pool comes from the other pool
        others = [r for r in runs if r["pool"] != pool]
        chosen[pool] = max(WEIGHTS, key=lambda w: mix(np.concatenate([per_question(r["F"] + w * r["E"], r["truth"]) for r in others]), np.concatenate([r["n"] for r in others])))
    d = cat(lambda r: per_question(r["F"] + chosen[r["pool"]] * r["E"], r["truth"])) - base
    b = boot_mix(d, n_all, cl_all, a.boot, rng)
    b["by_n"] = {str(m): float(d[n_all == m].mean()) for m in SIZES}
    b["weights"] = chosen
    b["per_pool"] = {pool: float(mix(d[np.char.startswith(cl_all, pool + "|")], n_all[np.char.startswith(cl_all, pool + "|")])) for pool in ("pub2", "priv2")}
    out["added"] = b
    print(f"new component added (weight from the other pool {chosen}): fused {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}  per pool {({k: round(v_, 4) for k, v_ in b['per_pool'].items()})}", flush=True)
    # without the document's own emb component (the new one may replace it)
    w_emb = float(fus.w[comps.index("emb")])
    d2 = cat(lambda r: per_question(r["F"] - w_emb * r["emb"] + chosen[r["pool"]] * r["E"], r["truth"])) - base
    b2 = boot_mix(d2, n_all, cl_all, a.boot, rng)
    out["replacing_emb"] = b2
    print(f"new component INSTEAD of the document's emb: fused {b2['diff']:+.4f} [{b2['ci95'][0]:+.4f}, {b2['ci95'][1]:+.4f}]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
