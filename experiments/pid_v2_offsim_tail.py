#!/usr/bin/env python
"""Tail terms from single reference games on the official pseudo-pools (METHOD_NOTES 2.117): the whole pipeline of a fusion
document as in scripts/pid_v2_offsim.py (3 games of every official candidate held out as questions of 1 / 2 / 3 games, the
other 7 are the reference games, sequence vectors after the pool fine-tune), with and without terms of the form
    + w x sum over the question's games of max(z - tau, 0)          (aicup_pid.oppside)
where z is, per question game and candidate, the standardised best single reference game under
    S  the opponents' vectors (the same opponent again)
    U  the subject's own vectors (one reference game that looks like this game)
    P  both sides summed (the same pairing again)
Arms: S, U, P alone and S + U together. Vectors: the same plain-input networks as they are (no pool fine-tune), own side
from --own-dir, opponent side from --opp-dir. Per arm the configuration is chosen on ONE pool and scored on the OTHER, both
ways (pools are never mixed inside a score); differences are bootstrapped over candidates.
  python -I <this> --fusion results/pid_fusion_v2c2_b.json --nets o7Fs0e ... --out results/final/v2_offsim_tail_v2c2_b.json
"""
import argparse
import itertools
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
from aicup_pid.oppside import nearest_z, opp_scores  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_offsim import SIZES, boot_mix, mix, per_question  # noqa: E402

TAUS = (2.0, 2.5, 3.0, 3.5, 4.0)
WS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
TAUS2 = (2.5, 3.0, 3.5)  # the two-term arm: a smaller grid per term
WS2 = (0.5, 1.0, 2.0, 3.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--opp-dir", default="/work/u2507115/pid_v2/seq_opp")
    ap.add_argument("--own-dir", default=str(ROOT / "data" / "seq_ft"))
    ap.add_argument("--nets", nargs="+", required=True)
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
    assert comps[-1] == "seq" and isinstance(doc["emb_model"], str)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    print(f"document {Path(a.fusion).name}: {len(nets)} sequence networks at t{a.step}; tail terms from {len(a.nets)} networks", flush=True)
    arms = {t: [((t, tau, w),) for tau in TAUS for w in WS] for t in ("S", "U", "P")}
    one = lambda t: [(t, tau, w) for tau in TAUS2 for w in WS2]  # noqa: E731
    arms["S+U"] = [c for c in itertools.product(one("S"), one("U"))]
    runs = []
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        oppv = [load(Path(a.opp_dir) / f"official_{pool}_net-1_{name}_opp_t0.npz") for name in a.nets]
        ownv = [load(Path(a.own_dir) / f"official_{pool}_net-1_{name}_t0.npz") for name in a.nets]
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
            seqs = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz") for v, tg in nets]
            S["seq"] = v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=seqs)["seq"]
            del seqs
            F = fus.scores([standardise(S[c]) for c in comps])
            zq = {"S": nearest_z(ref, questions, [oppv]), "U": nearest_z(ref, questions, [ownv]), "P": nearest_z(ref, questions, [ownv, oppv])}
            H = {(t, tau): opp_scores(ref, questions, None, tau, zq=zq[t]) for t in zq for tau in TAUS}
            base = per_question(F, truth)
            sc = {arm: {cfg: per_question(F + sum(w * H[(t, tau)] for t, tau, w in cfg), truth).astype(np.float32) for cfg in cfgs} for arm, cfgs in arms.items()}
            runs.append({"pool": pool, "seed": s, "truth": truth, "n": n, "base": base, "sc": sc})
            print(f"{pool} split {s}: {len(questions)} questions; base mix {mix(base, n):.4f} (n=1 {base[n == 1].mean():.4f}, n=2 {base[n == 2].mean():.4f}, n=3 {base[n == 3].mean():.4f}) "
                  f"({time.time() - t0:.0f} s)", flush=True)
            del H, zq, S, F
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    groups = {"pub2": [r for r in runs if r["pool"] == "pub2"], "priv2": [r for r in runs if r["pool"] == "priv2"], "both": runs}
    base_mix = {g: mix(cat(rs, lambda r: r["base"]), cat(rs, lambda r: r["n"])) for g, rs in groups.items()}
    out = {"document": Path(a.fusion).name, "step": a.step, "nets": a.nets, "base": base_mix, "arms": {}}

    def diff(rs, arm, cfg):
        d = cat(rs, lambda r: r["sc"][arm][cfg].astype(np.float64) - r["base"])
        n = cat(rs, lambda r: r["n"])
        cl = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
        return d, n, cl

    def summary(d, n, cl):
        b = boot_mix(d, n, cl, a.boot, rng)
        b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}
        return b

    show = lambda cfg: " + ".join(f"{t}(tau {tau:g}, w {w:g})" for t, tau, w in cfg)  # noqa: E731
    for arm, cfgs in arms.items():
        gain = {g: {cfg: mix(cat(rs, lambda r: r["sc"][arm][cfg]), cat(rs, lambda r: r["n"])) - base_mix[g] for cfg in cfgs} for g, rs in groups.items()}
        res = {"cross": {}}
        if arm != "S+U":
            print(f"arm {arm}: gain of the size-balanced score by (tau, w), both pools / pub2 / priv2:")
            for tau in TAUS:
                print(f"  tau {tau}: " + "  ".join(f"w{w:g} {gain['both'][((arm, tau, w),)]:+.4f}/{gain['pub2'][((arm, tau, w),)]:+.4f}/{gain['priv2'][((arm, tau, w),)]:+.4f}" for w in WS), flush=True)
        parts = []
        for pool, other in (("pub2", "priv2"), ("priv2", "pub2")):
            cfg = max(cfgs, key=lambda c: gain[other][c])
            d, n, cl = diff(groups[pool], arm, cfg)
            b = summary(d, n, cl)
            b["config_from_other_pool"] = show(cfg)
            res["cross"][pool] = b
            parts.append((d, n, cl))
            print(f"arm {arm}: {pool}, {show(cfg)} chosen on {other}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
        b = summary(*(np.concatenate([p[i] for p in parts]) for i in range(3)))
        res["cross"]["both"] = b
        print(f"arm {arm}: BOTH pools, each with the other's choice: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
        best = max(gain["both"].values())
        near = [c for c in cfgs if gain["both"][c] >= best - 0.0005]
        pick = min(near, key=lambda c: (sum(w for _t, _tau, w in c), -sum(tau for _t, tau, _w in c)))  # the smallest weights on the plateau, then the highest thresholds
        b = summary(*diff(groups["both"], arm, pick))
        res["file_config"] = {"config": [list(x) for x in pick], "gain_both": gain["both"][pick], "best_gain_both": best, "diff": b}
        print(f"arm {arm}: configuration for a file (plateau rule): {show(pick)}; gain on both pools {gain['both'][pick]:+.4f} (best of the grid {best:+.4f}); in-sample interval [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]", flush=True)
        res["grid_both"] = {show(c): v for c, v in gain["both"].items()}
        out["arms"][arm] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
