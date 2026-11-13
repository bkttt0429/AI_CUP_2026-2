#!/usr/bin/env python
"""Do the added pools share players (or games) with the original pools? (METHOD_NOTES 2.54)

The original pools (pub / priv: 400 candidates x 100 Black games, questions of 5-20 games, all answered) and the added pools
(pub2 / priv2: 600 candidates x 10 games) are separate files with separate ids. If a candidate of an added pool is the same
person as a candidate of the original pool of the same side, the original pool's 100 games could strengthen that candidate.
Nothing here uses an answer: only candidate games and their organiser-given owners.

1. Identical records: move sequences (the SGF text without the HD/SZ header differences) that occur in more than one file or
   under more than one owner, counted by file pair.
2. Identity match with the embedding of the fusion document (the per-game trunk rows already stored):
   prototype of every original candidate from its 100 games; every added-pool candidate as a 10-game set (and its Black games
   only) against the prototypes of the SAME side and, as a reference, of the OTHER side.
   Controls cut from the original pools themselves, same set sizes:
     positive  10 games of an original candidate against the prototypes built from everybody's other 90 games (own = match)
     null      the same 10-game sets against the prototypes of the other side's original pool (no match can exist if the two
               original pools are disjoint; their own best-match cosines are reported too)
   Reported: quantiles of the best cosine and of the gap to the second best for the positive control, the null and the four
   added-vs-original pairings; how many added-pool candidates exceed the null's 99.9th percentile and the positive control's
   5th percentile; mutual best matches; the listed pairs.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import pools, v2  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402

MOVES = re.compile(r";([BW])\[([a-s]{0,2})\]")
Q = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def move_hash(sgf):
    return hashlib.blake2b(";".join(c + p for c, p in MOVES.findall(sgf)).encode(), digest_size=12).hexdigest()


def old_pool(T, tag):
    """{player: [key]} of an original pool, keys as scripts/pid_trunk_extract.py --official writes them, and {key: sgf}."""
    c = pd.read_csv(T / f"player_identification_candidates_{tag}.csv", dtype=str, keep_default_na=False)
    cands, sgfs = defaultdict(list), {}
    for i, r in enumerate(c.itertuples()):
        key = f"c:{i}:{r.game_id}"
        cands[r.player_id].append(key)
        sgfs[key] = r.sgf_content
    for q in pools.read_questions(T / f"player_identification_test_{tag}.csv"):
        for i, (sgf, _) in enumerate(q["games"]):
            sgfs[f"q:{q['question_id']}:{i}"] = sgf
    return dict(cands), sgfs


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def best(S):
    o = np.sort(S, axis=1)
    return o[:, -1], o[:, -1] - o[:, -2], S.argmax(1)


def quant(x):
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in Q}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--emb", default=str(ROOT / "results" / "embed" / "emb2_final.pt"))
    ap.add_argument("--old-store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--new-store", default=str(ROOT / "data" / "trunk" / "tests_1111_v2"))
    ap.add_argument("--draws", type=int, default=3, help="10-game sets drawn per original candidate for the controls")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    out = {"records": {}, "identity": {}}

    # 1. identical records
    where = defaultdict(set)  # move hash -> {(file, owner or question)}
    old, new = {}, {}
    for tag, pre in (("public", "pub"), ("private", "priv")):
        cands, sgfs = old_pool(T, tag)
        old[pre] = cands
        owner = {k: p for p, ks in cands.items() for k in ks}
        for k, s in sgfs.items():
            where[move_hash(s)].add((f"{pre}:{'cand' if k.startswith('c:') else 'question'}", owner.get(k, k.split(":")[1])))
        c2, q2, qid2, sgf2 = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        new[pre + "2"] = c2
        own2 = {k: p for p, ks in c2.items() for k, _ in ks}
        for k, s in sgf2.items():
            where[move_hash(s)].add((f"{pre}2:{'cand' if k.startswith('c:') else 'question'}", own2.get(k, k.split(":")[1])))
    pairs = Counter()
    for h, ws in where.items():
        if len(ws) > 1:
            files = sorted({f for f, _ in ws})
            pairs[" + ".join(files) if len(files) > 1 else f"{files[0]} (two owners)"] += 1
    out["records"] = {"distinct_move_sequences": len(where), "shared_by_file_pair": dict(pairs)}
    print(f"records: {len(where)} distinct move sequences; shared: {dict(pairs) or 'none'} ({time.time() - t0:.0f} s)", flush=True)

    # 2. identity match
    emb = PlayerEmbedding.from_state(torch.load(a.emb, map_location="cpu", weights_only=False))
    so, sn = load_store(a.old_store), load_store(a.new_store)
    vec_old = {k: v for k, v in zip(so.keys(), unit(emb.embed(np.stack(list(so.values())))))}
    vec_new = {k: v for k, v in zip(sn.keys(), unit(emb.embed(np.stack(list(sn.values())))))}
    print(f"embedded {len(vec_old)} + {len(vec_new)} games ({time.time() - t0:.0f} s)", flush=True)
    players = {pre: sorted(c) for pre, c in old.items()}
    def hundred(keys):  # a candidate's 100 game vectors; a game without a stored row (unusable record) is replaced by a repeat
        v = [vec_old[k] for k in keys if k in vec_old]
        return np.stack((v * (100 // len(v) + 1))[:100])

    G = {pre: np.stack([hundred(old[pre][p]) for p in players[pre]]) for pre in old}  # [400, 100, d]
    proto = {pre: unit(G[pre].sum(1)) for pre in old}
    ctrl = {"positive": {"best": [], "gap": [], "own_is_best": []}, "null": {"best": [], "gap": []}}
    for pre, other in (("pub", "priv"), ("priv", "pub")):
        n = len(players[pre])
        for _ in range(a.draws):
            idx = np.stack([rng.permutation(100)[:10] for _ in range(n)])
            sub = np.take_along_axis(G[pre], idx[:, :, None], axis=1)  # [n, 10, d]
            q = unit(sub.sum(1))
            rest = unit(G[pre].sum(1) - sub.sum(1))  # own prototype without the drawn games; the others keep all 100 (a slightly harder control)
            S = q @ proto[pre].T
            S[np.arange(n), np.arange(n)] = (q * rest).sum(1)
            b, g, am = best(S)
            ctrl["positive"]["best"] += b.tolist(); ctrl["positive"]["gap"] += g.tolist(); ctrl["positive"]["own_is_best"] += (am == np.arange(n)).tolist()
            b, g, _ = best(q @ proto[other].T)
            ctrl["null"]["best"] += b.tolist(); ctrl["null"]["gap"] += g.tolist()
    pos_b, null_b = np.array(ctrl["positive"]["best"]), np.array(ctrl["null"]["best"])
    pos_g, null_g = np.array(ctrl["positive"]["gap"]), np.array(ctrl["null"]["gap"])
    thr_null, thr_pos = float(np.quantile(null_b, 0.999)), float(np.quantile(pos_b, 0.05))
    thr_gap = float(np.quantile(null_g, 0.999))
    out["identity"]["controls"] = {"positive": {"own_is_best": float(np.mean(ctrl["positive"]["own_is_best"])), "best": quant(pos_b), "gap": quant(pos_g)},
                                   "null": {"best": quant(null_b), "gap": quant(null_g)}, "null_q999_best": thr_null, "null_q999_gap": thr_gap, "positive_q05_best": thr_pos}
    print(f"controls (10 games against 100-game prototypes): positive own-is-best {np.mean(ctrl['positive']['own_is_best']):.4f}; best cosine positive {quant(pos_b)}", flush=True)
    print(f"   null best cosine {quant(null_b)}; null 99.9th pct {thr_null:.3f} (gap {thr_gap:.3f}); positive 5th pct {thr_pos:.3f}", flush=True)
    for npre in ("pub2", "priv2"):
        npl = sorted(new[npre])
        for games, label in ((lambda p: [k for k, _ in new[npre][p]], "all 10 games"), (lambda p: [k for k, c in new[npre][p] if c == "B"] or [k for k, _ in new[npre][p]], "Black games")):
            q = unit(np.stack([np.sum([vec_new[k] for k in games(p) if k in vec_new], axis=0) for p in npl]))
            for opre in ("pub", "priv"):
                S = q @ proto[opre].T
                b, g, am = best(S)
                back = S.argmax(0)  # each original candidate's best added-pool candidate
                mutual = back[am] == np.arange(len(npl))
                hit = (b > thr_null) & (g > thr_gap)
                strong = hit & (b > thr_pos) & mutual
                side = "same side" if opre == npre[:-1] else "other side"
                rec = {"best": quant(b), "gap": quant(g), "above_null_q999": int((b > thr_null).sum()), "above_null_q999_and_gap": int(hit.sum()), "also_above_positive_q05_and_mutual": int(strong.sum()),
                       "pairs": [{"added": npl[i], "original": players[opre][am[i]], "cos": float(b[i]), "gap": float(g[i]), "mutual": bool(mutual[i])} for i in np.flatnonzero(hit)]}
                out["identity"][f"{npre} vs {opre} ({label})"] = rec
                print(f"{npre} candidates ({label}) vs {opre} ({side}): best cosine {quant(b)}; above the null's 99.9th pct {rec['above_null_q999']} of {len(npl)} "
                      f"(with the gap too: {rec['above_null_q999_and_gap']}; also above the positive 5th pct and mutual: {rec['also_above_positive_q05_and_mutual']})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
