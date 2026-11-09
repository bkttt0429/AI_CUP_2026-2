#!/usr/bin/env python
"""B3 on the void 10-30 Public pool: the rank model's wider inner layers as an identification signal (METHOD_NOTES 2.25).

B1 (2.24) used only ct9's last layer, 10 rank probabilities. Before that layer every ct9 member has a 32-dim question
embedding zq (trained jointly with the M33 player-contrastive loss, whose projection head was training-only and is not
saved) and, below it, the mean / std of 64-dim per-game vectors (128 dims). This script re-runs the rank session's
test-side predict (AI_CUP_2026-1 scripts/trunk_predict.py predict, read-only, on their shards and vec of the Public
candidates' 5 x 20-game chunks and the 400 questions) with a hook that also keeps, for each of the 48 passes (24
checkpoints x 2 L1b heads), zq and the pooled game vectors. Sanity: the hook's softmax must equal the unhooked file.

Pre-set component: per pass, candidate vector = mean of its 5 chunks; every dimension z-scored with the 400 candidates'
mean / sd; cosine; the 48 passes' similarity matrices averaged. Report only: the pooled 128-dim game vectors, C vs T arms.
Official players are in no member's training set, so nothing leaks; truth = upload_fix Top-1 (Public 1.0000000).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401  (AI_CUP_2026-1 src and scripts on sys.path)
import trunk_predict as tp  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

CAP, LOADED = [], {}


def hooked_predict(model, R, questions, X, offsets, device, batch=64):
    model.eval()
    out, zqs, pooled = [], [], []
    with torch.no_grad():
        for i in range(0, len(R), batch):
            r = torch.as_tensor(R[i:i + batch], device=device)
            tokens, mask, game_of = ts.batch_tokens(X, offsets, questions[i:i + batch], device)
            v = model.encode_games(tokens, mask)
            pg = model.pool_games(v, game_of, r.shape[0])
            zq = torch.relu(model.question(pg))
            logits = model.mlp(torch.cat([r, zq], dim=1))
            out.append(torch.softmax(logits, dim=1).double().cpu().numpy())
            zqs.append(zq.float().cpu().numpy())
            pooled.append(pg.float().cpu().numpy())
    P = np.concatenate(out)
    CAP.append({"model": LOADED.get(id(model), "?"), "zq": np.concatenate(zqs), "pooled": np.concatenate(pooled), "P": P})
    return P


_orig_load = ts.load_checkpoint


def hooked_load(path, device="cpu"):
    m, stats, r, side = _orig_load(path, device)
    LOADED[id(m)] = Path(path).stem
    return m, stats, r, side


def official(S, tix, n):
    order = np.argsort(-S, axis=1)
    rank = np.array([int(np.where(order[q] == tix[q])[0][0]) + 1 for q in range(len(S))])
    sc = np.where(rank <= 5, np.exp(-(rank - 1)), 0.0)
    return {"score": float(sc.mean()), "top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()), "median_rank": float(np.median(rank)),
            "by_n": {str(k): float(sc[n == k].mean()) for k in sorted(set(n.tolist()))}}, rank


def zcos(C, Q):
    mu, sd = C.mean(axis=0), C.std(axis=0) + 1e-6
    Zc, Zq = (C - mu) / sd, (Q - mu) / sd
    Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12
    Zq /= np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-12
    return Zq @ Zc.T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--b1-dir", default=str(Path.home() / "BK_AICUP_2026-1" / "results" / "b1_public"))
    ap.add_argument("--work", default="/work/u2507115/pid_b3")
    ap.add_argument("--questions", default=str(ROOT / "data" / "tests" / "player_identification_test_public.csv"))
    ap.add_argument("--truth", default=str(ROOT / "results" / "probe" / "upload_fix.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d, work = Path(a.b1_dir), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    models = sorted(str(p) for p in (Path.home() / "BK_AICUP_2026-1" / "results" / "models" / "ct_cv_m33").glob("val[1234]_[CT]-pc-f0-x40-oh_s[012].pt"))
    if len(models) != 24:
        raise SystemExit(f"expected the 24 ct9 members, found {len(models)}")
    ts.predict, ts.load_checkpoint = hooked_predict, hooked_load
    rc = tp.main(["predict", "--inputs", str(d / "inputs" / "cands.csv"), str(d / "inputs" / "quest.csv"), "--shards", str(d / "submission_rank.csv"),
                  "--nshards", "8", "--vec", str(d / "vec"), "--models", *models, "--fallback-submission", str(d / "submission_rank.csv"),
                  "--allow-missing-hd", "--out", str(work / "submission_rank_ct9.csv"), "--expect-rows", "2400", "--expect-public", "2400", "--expect-private", "0"])
    print(f"predict exit {rc}; {len(CAP)} passes captured", flush=True)
    mine = pd.read_csv(work / "submission_rank_ct9_proba.csv")
    ref = pd.read_csv(d / "submission_rank_ct9_proba.csv").set_index("question_id").loc[mine.question_id]
    cols = [c for c in mine.columns if c.startswith("p_")]
    diff = float(np.abs(mine[cols].to_numpy() - ref[cols].to_numpy()).max())
    print(f"sanity: max |p hooked - p of the rank session's file| = {diff:.2e}", flush=True)
    if diff > 1e-4:
        raise SystemExit("hooked predict does not reproduce the rank session's probabilities")
    qids = mine.question_id.tolist()  # rows of P, in the order the passes saw the questions
    cmap = pd.read_csv(d / "inputs" / "cands_map.csv", dtype=str)
    chunk_owner = dict(zip(cmap.question_id, cmap.player_id))
    players = sorted(set(chunk_owner.values()))
    pidx = {p: i for i, p in enumerate(players)}
    qs = pd.read_csv(a.questions, dtype=str, keep_default_na=False)
    n = qs.num_games.astype(int).to_numpy()
    truth = pd.read_csv(a.truth, dtype=str).set_index("question_id").top_1
    tix = np.array([pidx[truth[q]] for q in qs.question_id])
    row = {q: i for i, q in enumerate(qids)}
    cand_rows = [[row[c] for c in cmap.question_id[cmap.player_id == p]] for p in players]
    q_rows = [row[q] for q in qs.question_id]
    res = {}
    for layer in ("zq", "pooled"):
        sims, by_arm = [], {"C": [], "T": []}
        for c in CAP:
            M = c[layer]
            C = np.stack([M[r].mean(axis=0) for r in cand_rows])
            S = zcos(C, M[q_rows])
            sims.append(S)
            by_arm["T" if "_T-" in c["model"] else "C"].append(S)
        S = np.mean(sims, axis=0)
        r, rank = official(S, tix, n)
        res[layer] = {"all_passes": r, **{f"arm_{k}": official(np.mean(v, axis=0), tix, n)[0] for k, v in by_arm.items() if v}}
        np.save(Path(a.out).with_name(Path(a.out).stem + f"_{layer}.npy"), S)
        print(f"B3 {layer} ({CAP[0][layer].shape[1]} dims, {len(sims)} passes): score {r['score']:.4f}, Top-1 {r['top1']:.3f}, Top-5 {r['top5']:.3f}, "
              f"median rank {r['median_rank']:.0f}/400, by n {({k: round(v, 3) for k, v in r['by_n'].items()})}; "
              + "  ".join(f"{k} {v['score']:.4f}" for k, v in res[layer].items() if k.startswith("arm_")), flush=True)
        for name in ("submission_f3oreft", "embed/oldcheck_off"):
            f = ROOT / "results" / name / "submission_player.csv"
            if f.exists():
                s = pd.read_csv(f, dtype=str).set_index("question_id")
                for qi, q in enumerate(qs.question_id):
                    if s.loc[q, "top_1"] != truth[q]:
                        pk = pidx[s.loc[q, "top_1"]]
                        print(f"  {name} wrong on {q} (n={n[qi]}): {layer} rank of the truth {rank[qi]}, of the wrong pick {int((S[qi] > S[qi, pk]).sum()) + 1}")
    out = {"layers": res, "passes": [c["model"] for c in CAP], "sanity_max_abs_p_diff": diff, "players": players, "questions": qs.question_id.tolist(),
           "b1_10dim_reference": 0.1352}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
