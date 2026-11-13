#!/usr/bin/env python
"""Where in a game is the identity information? (METHOD_NOTES 2.56, atom 4)

Simulated pools of the training folds (scripts/pid_v2_sim2.py: 600 candidates x 10 games, both colours, rank classes known),
one held-out network per fold, no pool fine-tune. Every question game is a 1-game query against the candidates' prototypes
(their ten full games). The query is shown to the network only up to ply K (its length is cut; nothing else changes):
  K = 30, 60, 90, 120, 150, 180, 256 (all the network ever sees)
and, as a second reading, with the candidates' games cut at the same ply (prefix against prefix).
Reported: the 1-game score (e^-(rank-1) inside the Top-5) by K, split by the subject's colour, by kyu / dan class of the owner
and by rank class; and the share of the full-game score already reached at each K.
Run from the AI_CUP_2026-1 checkout."""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
import m88_seqid as m88  # noqa: E402
import pid_v2_seq_finetune as FT  # noqa: E402

KS = (30, 60, 90, 120, 150, 180, 256)
DAN = ("1D", "2D", "3D", "4D", "5D", "6D")
ORDER = ("D", "C", "B", "A", *DAN)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--ckpt-suffix", default="_x2e")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import aicup_pid
    from aicup.ladder import to_json
    from aicup_pid import v2
    from aicup_pid.paths import FOLDS_CSV
    from pid_v2_sim2 import build
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    rec = []  # per query game: fold, colour, class, full length, and its score at every K (full prototypes / cut prototypes)
    for v in a.folds:
        z = np.load(R2 / "data" / "seq" / f"fold{v}.npz")
        g = games[(games.fold == v) & games.game_key.isin(rows.keys()) & games.game_key.isin(set(z["key"].astype(str)))]
        net = FT.load_net(Path(a.ckpt_dir) / f"ckpt_{v}{a.ckpt_suffix}.pt")
        for s in a.seeds:
            cands, questions, owner, _block, _sgfs, _info = build(g, s)
            players = sorted(cands)
            pidx = {p: i for i, p in enumerate(players)}
            need = {k for ks in cands.values() for k, _ in ks} | {k for q in questions for k, _ in q}
            files = [f for d in m88.DIRS for f in sorted((d / f"fold{v}").glob("games_s*_p*.csv"))]
            parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
            with mp.get_context("fork").Pool(12) as pool:
                for i in range(0, len(files), 48):
                    for key, _pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                        m = np.array([k in need for k in key])
                        if m.any():
                            for name, arr in (("key", key), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):
                                parts[name].append(arr[m])
            A = {k: np.concatenate(x) for k, x in parts.items()}
            at = {k: i for i, k in enumerate(A["key"])}
            c_rows = np.array([at[k] for p in players for k, _ in cands[p]])
            c_own = np.array([pidx[p] for p in players for _ in cands[p]])
            q_rows = np.array([at[k] for q in questions for k, _ in q])
            q_own = np.array([pidx[p] for q, p in zip(questions, owner) for _ in q])
            q_col = np.array([c for q in questions for _, c in q])
            q_cls = np.array([k.split(":")[0] for q in questions for k, _ in q])
            full_len = A["LEN"][q_rows].copy()

            def protos(K):
                B = dict(A)
                B["LEN"] = np.minimum(A["LEN"], K).astype(A["LEN"].dtype)
                E = FT.embed(net, B, c_rows)
                P = np.zeros((len(players), E.shape[1]), np.float32)
                np.add.at(P, c_own, E)
                return unit(P)

            def score(K, P):
                B = dict(A)
                B["LEN"] = np.minimum(A["LEN"], K).astype(A["LEN"].dtype)
                S = FT.embed(net, B, q_rows) @ P.T
                rank = (S > S[np.arange(len(q_own)), q_own][:, None]).sum(1)
                return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)

            P_full = protos(256)
            sc_full = {K: score(K, P_full) for K in KS}
            sc_cut = {K: score(K, protos(K)) for K in KS}
            for i in range(len(q_rows)):
                rec.append({"fold": v, "seed": s, "colour": q_col[i], "cls": q_cls[i], "len": int(full_len[i]), **{f"f{K}": sc_full[K][i] for K in KS}, **{f"c{K}": sc_cut[K][i] for K in KS}})
            print(f"fold {v} seed {s}: {len(q_rows)} query games; full game {sc_full[256].mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
    d = pd.DataFrame(rec)
    out = {"network": a.ckpt_suffix, "queries": len(d), "tables": {}}

    def table(name, sel):
        x = d[sel]
        row = {"n": int(len(x)), "full_prototypes": {str(K): float(x[f"f{K}"].mean()) for K in KS}, "cut_prototypes": {str(K): float(x[f"c{K}"].mean()) for K in KS}}
        out["tables"][name] = row
        full = row["full_prototypes"]["256"]
        print(f"{name:22s} n {len(x):5d} | query cut at K, full prototypes: " + "  ".join(f"{K} {row['full_prototypes'][str(K)]:.3f}" for K in KS)
              + " | share of the full score: " + "  ".join(f"{K} {row['full_prototypes'][str(K)] / full:.0%}" for K in KS[:-1])
              + " | both cut: " + "  ".join(f"{K} {row['cut_prototypes'][str(K)]:.3f}" for K in KS), flush=True)

    table("all", np.ones(len(d), bool))
    table("subject Black", d.colour == "B")
    table("subject White", d.colour == "W")
    table("kyu classes (D-A)", ~d.cls.isin(DAN))
    table("dan classes (1D-6D)", d.cls.isin(DAN))
    for c in ORDER:
        if (d.cls == c).any():
            table(f"class {c}", d.cls == c)
    for lo, hi in ((0, 150), (150, 200), (200, 256), (256, 9999)):
        table(f"length {lo}-{hi if hi < 9999 else ''}", (d.len >= lo) & (d.len < hi))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
