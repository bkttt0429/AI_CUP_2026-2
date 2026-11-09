#!/usr/bin/env python
"""B3 on official files (METHOD_NOTES 2.26, user's option A: used only when the one-to-one assignment does not apply): build | capture.

build    official folder -> per pool (pub, priv) the rank-test-format inputs under --work/<pool>/inputs/: cands.csv (every
         candidate's games in chunks of <= 20, aicup_pid.b3.rank_inputs), quest.csv (the test file, unchanged) and
         cands_map.csv (chunk -> player, games), plus meta.json (rows, expect flags).
capture  runs the rank session's trunk_predict.py predict (read-only; run from AI_CUP_2026-1 after its predict_submission.py
         shards + merge and trunk_predict.py extract on the same inputs) with all 24 ct9 members (official players are in no
         member's training set) and both L1b heads, hooked to keep every pass's 32-dim zq -> --work/<pool>/b3.npz.
         Rows are aligned by the questions the rank path could use (trunk_predict: sorted question indices with a usable
         game); the mean of the captured passes' probabilities must equal the written probability file, or nothing is saved.
predict_pid.py --b3-dir <work> then scores the component with aicup_pid.b3.score_b3, as the pool test did.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401  (AI_CUP_2026-1 src and scripts on sys.path)
from aicup_pid.b3 import RANK_COLUMNS, rank_inputs  # noqa: E402

POOLS = {"pub": "public", "priv": "private"}
R1 = Path.home() / "BK_AICUP_2026-1"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]


def build(a):
    T, W = Path(a.tests_dir), Path(a.work)
    for prefix, name in POOLS.items():
        cands = pd.read_csv(T / f"player_identification_candidates_{name}.csv", dtype=str, keep_default_na=False)
        q = pd.read_csv(T / f"player_identification_test_{name}.csv", dtype=str, keep_default_na=False)
        if list(q.columns) != RANK_COLUMNS:
            raise SystemExit(f"{name} test file: columns {list(q.columns)[:4]}... differ from the 10-30 layout; adapt the rank inputs first")
        if not q.question_id.str.startswith(f"{prefix}_").all():
            raise SystemExit(f"{name} test file: question ids without the {prefix}_ prefix")
        sizes = q.num_games.astype(int)
        if not sizes.between(1, 20).all():
            raise SystemExit(f"{name} test file: num_games outside 1..20")
        rows, mp = rank_inputs(cands, prefix)
        if set(rows.question_id) & set(q.question_id):
            raise SystemExit(f"{name}: chunk ids collide with question ids")
        out = W / prefix / "inputs"
        out.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / "cands.csv", index=False)
        q.to_csv(out / "quest.csv", index=False)
        mp.to_csv(out / "cands_map.csv", index=False)
        n = len(rows) + len(q)
        meta = {"pool": prefix, "rows": n, "chunks": len(rows), "questions": len(q), "players": int(cands.player_id.nunique()), "games": len(cands),
                "chunk_sizes": {str(k): int(v) for k, v in rows.num_games.astype(int).value_counts().sort_index().items()},
                "expect": ["--expect-rows", str(n), "--expect-public", str(n if prefix == "pub" else 0), "--expect-private", str(n if prefix == "priv" else 0)],
                "md5": {"cands.csv": md5(out / "cands.csv"), "quest.csv": md5(out / "quest.csv")}, "source": str(T)}
        (W / prefix / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{prefix}: {meta['players']} candidates, {meta['games']} games -> {meta['chunks']} chunks (sizes {meta['chunk_sizes']}); "
              f"{len(q)} questions; {n} rows; md5 {meta['md5']}", flush=True)


def capture(a):
    import torch
    import trunk_predict as tp
    import trunk_seq_check as ts
    W = Path(a.work)
    meta = json.loads((W / "meta.json").read_text())
    models = sorted(str(p) for p in (R1 / "results" / "models" / "ct_cv_m33").glob("val[1234]_[CT]-pc-f0-x40-oh_s[012].pt"))
    if len(models) != 24:
        raise SystemExit(f"expected the 24 ct9 members, found {len(models)}")
    cap, loaded = [], {}
    orig_load = ts.load_checkpoint

    def hooked_load(path, device="cpu"):
        m, stats, r, side = orig_load(path, device)
        loaded[id(m)] = Path(path).stem
        return m, stats, r, side

    def hooked_predict(model, R, questions, X, offsets, device, batch=64):
        model.eval()
        P, zqs = [], []
        with torch.no_grad():
            for i in range(0, len(R), batch):
                r = torch.as_tensor(R[i:i + batch], device=device)
                tokens, mask, game_of = ts.batch_tokens(X, offsets, questions[i:i + batch], device)
                pg = model.pool_games(model.encode_games(tokens, mask), game_of, r.shape[0])
                zq = torch.relu(model.question(pg))
                P.append(torch.softmax(model.mlp(torch.cat([r, zq], dim=1)), dim=1).double().cpu().numpy())
                zqs.append(zq.float().cpu().numpy())
        P = np.concatenate(P) if P else np.zeros((0, 10))
        cap.append((loaded.get(id(model), "?"), np.concatenate(zqs) if zqs else np.zeros((0, 32), np.float32), P))
        return P

    ts.predict, ts.load_checkpoint = hooked_predict, hooked_load
    inp = W / "inputs"
    rc = tp.main(["predict", "--inputs", str(inp / "cands.csv"), str(inp / "quest.csv"), "--shards", str(W / "submission_rank.csv"), "--nshards", "8",
                  "--vec", str(W / "vec"), "--models", *models, "--fallback-submission", str(W / "submission_rank.csv"), "--allow-missing-hd",
                  "--allow-fallback", "--out", str(W / "submission_rank_ct9.csv"), *meta["expect"]])
    if rc != 0:
        raise SystemExit(f"rank predict exit {rc}")
    rep = json.loads((W / "submission_rank_ct9_report.json").read_text())
    proba = pd.read_csv(W / "submission_rank_ct9_proba.csv", dtype={"question_id": str})
    cols = [c for c in proba.columns if c.startswith("p_")]
    ok = np.isfinite(proba[cols].to_numpy()).all(axis=1)
    qids = proba.question_id[ok].to_numpy(dtype=str)  # trunk_predict rows = questions with a usable game, in input order
    if any(len(c[1]) != len(qids) for c in cap):
        raise SystemExit(f"captured rows {sorted({len(c[1]) for c in cap})} != {len(qids)} decided questions")
    diff = float(np.abs(np.mean([c[2] for c in cap], axis=0) - proba[cols].to_numpy()[ok]).max())
    if diff > 1e-6:
        raise SystemExit(f"mean of the captured passes differs from the probability file by {diff:.2e}: rows misaligned, nothing saved")
    np.savez(W / "b3.npz", qids=qids, names=np.array([c[0] for c in cap], dtype=str), zq=np.stack([c[1] for c in cap]))
    info = {"passes": len(cap), "rows": int(len(qids)), "not_decided": proba.question_id[~ok].tolist(), "sanity_max_abs_p_diff": diff,
            "l1b_models_used": rep.get("l1b_models_used"), "models": sorted({c[0] for c in cap})}
    (W / "b3_capture.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"{meta['pool']}: {len(cap)} passes over {len(qids)} rows ({len(info['not_decided'])} not decided), sanity {diff:.1e} -> {W / 'b3.npz'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "capture"))
    ap.add_argument("--tests-dir", help="build: folder of the official files (10-30 names)")
    ap.add_argument("--work", required=True, help="build: the run folder (pools go to <work>/pub, <work>/priv); capture: one pool's folder")
    a = ap.parse_args(argv)
    return {"build": build, "capture": capture}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
