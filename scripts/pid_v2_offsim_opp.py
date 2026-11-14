#!/usr/bin/env python
"""The opponent-side term on the official pseudo-pools (METHOD_NOTES 2.117): the whole pipeline of a fusion document as in
scripts/pid_v2_offsim.py (3 games of every official candidate held out as questions of 1 / 2 / 3 games, the other 7 are the
reference games, sequence vectors after the pool fine-tune), with and without
    + w x aicup_pid.oppside.opp_scores(tau)      (the opponents of the question games against the candidates' opponents).
The pair (tau, w) is chosen on ONE pool and scored on the OTHER, both ways (pools are never mixed inside a score); differences
are bootstrapped over candidates. Also printed: the whole grid on both pools, the configuration fixed beforehand from the
leave-one-game-out probe (--fixed), and the configuration for a submission file by the plateau rule (smallest weight within
0.0005 of the best gain on both pools).
  python -I <this> --fusion results/pid_fusion_v2c2_b.json --opp-nets o7Fs0e ... --out results/final/v2_offsim_opp_v2c2_b.json
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
from aicup_pid.oppside import opp_scores, opp_z  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_v2_offsim import SIZES, boot_mix, mix, per_question  # noqa: E402

TAUS = (2.5, 3.0, 3.5, 4.0)
WS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--opp-dir", default="/work/u2507115/pid_v2/seq_opp")
    ap.add_argument("--opp-nets", nargs="+", required=True)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--fixed", nargs=2, type=float, default=[3.0, 1.5], help="tau and weight fixed before this run")
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
    print(f"document {Path(a.fusion).name}: {len(nets)} sequence networks at t{a.step}; opponent vectors of {len(a.opp_nets)} networks", flush=True)
    grid = [(tau, w) for tau in TAUS for w in WS]
    fixed = (float(a.fixed[0]), float(a.fixed[1]))
    assert fixed in grid, "--fixed must be a grid point"
    runs = []
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        oppv = [load(Path(a.opp_dir) / f"official_{pool}_net-1_{name}_opp_t0.npz") for name in a.opp_nets]
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
            F = fus.scores([standardise(S[c]) for c in comps])
            zq = opp_z(ref, questions, oppv)
            O = {tau: opp_scores(ref, questions, None, tau, zq=zq) for tau in TAUS}
            base = per_question(F, truth)
            sc = {(tau, w): per_question(F + w * O[tau], truth) for tau, w in grid}
            top0 = F.argmax(1)
            Ff = F + fixed[1] * O[fixed[0]]
            runs.append({"pool": pool, "seed": s, "truth": truth, "n": n, "base": base, "sc": sc,
                         "changed": Ff.argmax(1) != top0, "own_z": zq[0][[zq[1][q[0][0]] for q in questions], truth]})
            print(f"{pool} split {s}: {len(questions)} questions; base mix {mix(base, n):.4f} (n=1 {base[n == 1].mean():.4f}); fixed tau {fixed[0]} w {fixed[1]}: "
                  f"{mix(sc[fixed], n) - mix(base, n):+.4f}, Top-1 changed {int((Ff.argmax(1) != top0).sum())}; owner's z of the first game above 3: {float((runs[-1]['own_z'] > 3).mean()):.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731
    groups = {"pub2": [r for r in runs if r["pool"] == "pub2"], "priv2": [r for r in runs if r["pool"] == "priv2"], "both": runs}
    gain = {g: {c: mix(cat(rs, lambda r: r["sc"][c]), cat(rs, lambda r: r["n"])) - mix(cat(rs, lambda r: r["base"]), cat(rs, lambda r: r["n"])) for c in grid} for g, rs in groups.items()}
    out = {"document": Path(a.fusion).name, "step": a.step, "opp_nets": a.opp_nets, "taus": TAUS, "weights": WS, "fixed": fixed,
           "base": {g: mix(cat(rs, lambda r: r["base"]), cat(rs, lambda r: r["n"])) for g, rs in groups.items()},
           "grid": {g: {f"{tau}|{w}": v for (tau, w), v in d_.items()} for g, d_ in gain.items()}, "cross": {}, "fixed_diff": {}}
    print("gain of the size-balanced score by (tau, w), both pools / pub2 / priv2:")
    for tau in TAUS:
        print(f"  tau {tau}: " + "  ".join(f"w{w:g} {gain['both'][(tau, w)]:+.4f}/{gain['pub2'][(tau, w)]:+.4f}/{gain['priv2'][(tau, w)]:+.4f}" for w in WS), flush=True)

    def diff(rs, cfg):
        d = cat(rs, lambda r: r["sc"][cfg] - r["base"])
        n = cat(rs, lambda r: r["n"])
        cl = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))
        b = boot_mix(d, n, cl, a.boot, rng)
        b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}
        return b, d, n, cl

    parts = []
    for pool, other in (("pub2", "priv2"), ("priv2", "pub2")):
        cfg = max(grid, key=lambda c: gain[other][c])
        b, d, n, cl = diff(groups[pool], cfg)
        b["config_from_other_pool"] = cfg
        out["cross"][pool] = b
        parts.append((d, n, cl))
        print(f"{pool}, (tau, w) = {cfg} chosen on {other}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
    d, n, cl = (np.concatenate([p[i] for p in parts]) for i in range(3))
    b = boot_mix(d, n, cl, a.boot, rng)
    b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}
    out["cross"]["both"] = b
    print(f"both pools, each with the other's choice: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
    for g, rs in groups.items():
        b, _d, _n, _cl = diff(rs, fixed)
        out["fixed_diff"][g] = b
        print(f"{g}, fixed (tau, w) = {fixed}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v, 4) for k, v in b['by_n'].items()})}", flush=True)
    best = max(gain["both"].values())
    near = [c for c in grid if gain["both"][c] >= best - 0.0005]
    pick = min(near, key=lambda c: (c[1], -c[0]))  # the smallest weight on the plateau, then the highest threshold
    out["file_config"] = {"tau": pick[0], "w": pick[1], "gain_both": gain["both"][pick], "best_gain_both": best}
    print(f"configuration for a file (plateau rule): tau {pick[0]}, w {pick[1]} (gain on both pools {gain['both'][pick]:+.4f}; best of the grid {best:+.4f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
