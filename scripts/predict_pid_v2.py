#!/usr/bin/env python
"""The 11-08 v2 release -> one submission of 1,600 rows (METHOD_NOTES 2.29).

The original pools (pub_ / priv_, byte-identical to the 10-30 files) keep the rows of an existing submission (--old-submission:
predict_pid.py's output on those files). The added pools (pub2_ / priv2_: 600 candidates x 10 games, questions of 1-5 games,
either colour) are scored here with aicup_pid.v2.components and the fusion document of scripts/pid_v2_sim.py; --assign auto
applies the one-to-one assignment only when aicup_pid.checks' rule v2 says the pool looks one-to-one. Games that do not replay
(aicup.submission.check_game) or have no trunk row are left out; a question without a usable game takes the first five
candidates and is listed. The result is checked against the sample submission before it is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.submission import check_game  # noqa: E402
from aicup_pid import pools, v2  # noqa: E402
from aicup_pid.assign import assign_soft, assign_topk  # noqa: E402
from aicup_pid.checks import top1_collisions  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402


def predict_pool(cand_path, q_path, doc, emb, store, assign, workers, log=print, seq=None, big_lam=None):
    cands, questions, qids, sgfs = v2.official_pool(cand_path, q_path)
    colour = {k: c for games in (*cands.values(), *questions) for k, c in games}
    need_rows = "emb" in doc["components"]
    bad = sorted(k for k in sgfs if not check_game(sgfs[k], colour[k])[0] or (need_rows and k not in store) or (seq is not None and k not in (seq[0] if isinstance(seq, list) else seq)))
    if bad:
        log(f"  {len(bad)} games left out (do not replay or have no trunk row): {bad[:5]}", flush=True)
    cands = {p: [(k, c) for k, c in g if k not in bad] for p, g in cands.items()}
    empty_c = sorted(p for p, g in cands.items() if not g)
    if empty_c:
        raise ValueError(f"{cand_path}: candidates without a usable game: {empty_c[:5]}")
    q_use = [[(k, c) for k, c in q if k not in bad] for q in questions]
    usable = [i for i, q in enumerate(q_use) if q]
    moves = parse_all({k: s for k, s in sgfs.items() if k not in bad}, workers)
    S = v2.components(cands, [q_use[i] for i in usable], moves, sgfs, doc["cfg"], store, emb, which=doc["components"], seq=seq)
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    F = fus.scores([S[c] for c in doc["components"]])
    players = sorted(cands)
    n = np.array([len(q_use[i]) for i in usable])
    col = top1_collisions(F, n)
    applied = assign in ("on", "big") or (assign == "auto" and bool(col["one_to_one_likely_v2"])) or (assign == "soft" and doc.get("soft_lam") is not None)
    base = topk(F, pools.TOP_K)
    if assign == "big":  # METHOD_NOTES 2.37: the questions of two or more games are one one-to-one set; the one-game questions stay per question
        idx, rows_big = base.copy(), np.flatnonzero(n > 1)
        idx[rows_big] = assign_topk(F[rows_big], pools.TOP_K) if big_lam is None else assign_soft(F[rows_big], big_lam, pools.TOP_K)
    else:
        idx = assign_soft(F, doc["soft_lam"], pools.TOP_K) if (applied and assign == "soft") else assign_topk(F, pools.TOP_K) if applied else base
    rows, j = [], 0
    for i, qid in enumerate(qids):
        if i in set(usable):
            rows.append((qid, [players[c] for c in idx[j]]))
            j += 1
        else:
            rows.append((qid, players[:pools.TOP_K]))
    top = np.sort(F, axis=1)
    rep = {"candidates": str(cand_path), "questions": str(q_path), "n_candidates": len(players), "n_questions": len(qids), "left_out_games": bad,
           "fallback_questions": [qids[i] for i in range(len(qids)) if i not in set(usable)], "games_by_colour": pd.Series(list(colour.values())).value_counts().to_dict(),
           "assign": {"mode": assign, "applied": applied, "big_lam": big_lam, "n_top1_changed": int((idx[:, 0] != base[:, 0]).sum()),
                      "top1_changed_by_n": {str(m): int((idx[:, 0] != base[:, 0])[n == m].sum()) for m in sorted(set(n.tolist()))}, "top1_collisions": col},
           "median_top1_margin": float(np.median(top[:, -1] - top[:, -2])), "margin_by_n": {str(m): float(np.median((top[:, -1] - top[:, -2])[n == m])) for m in sorted(set(n.tolist()))},
           "distinct_top1": int(len(set(base[:, 0].tolist())))}
    log(f"{Path(q_path).name}: {len(players)} candidates, {len(qids)} questions ({len(rep['fallback_questions'])} fallback); distinct Top-1 {rep['distinct_top1']}; "
        f"confident-half distinct {col['confident_distinct']:.3f}; assignment {'applied' if applied else 'not applied'} (changed {rep['assign']['n_top1_changed']}); "
        f"median margin by n {({k: round(v, 2) for k, v in rep['margin_by_n'].items()})}", flush=True)
    return rows, rep, set(players)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", required=True)
    ap.add_argument("--fusion", required=True, help="scripts/pid_v2_sim.py --fusion-out")
    ap.add_argument("--trunk-store", nargs="*", default=[], help="pid_trunk_extract.py --official2 output (needed when the fusion names emb)")
    ap.add_argument("--old-submission", required=True, help="800 rows for the original pools (predict_pid.py on the 10-30 files)")
    ap.add_argument("--assign", choices=["off", "auto", "on", "soft", "big"], default="off",
                    help="soft = aicup_pid.assign.assign_soft with the document's soft_lam (off when it has none); big = one-to-one over the questions of two or more games only (2.37)")
    ap.add_argument("--big-lam", type=float, default=None, help="--assign big: soft assignment with this penalty instead of the hard one")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    T = Path(a.tests_dir)
    doc = json.loads(Path(a.fusion).read_text())
    emb = store = None
    if "emb" in doc["components"]:
        import torch
        names = doc["emb_model"] if isinstance(doc["emb_model"], list) else [doc["emb_model"]]
        embs = [PlayerEmbedding.from_state(torch.load(str(ROOT / m), map_location="cpu", weights_only=False)) for m in names]
        emb = embs[0] if len(embs) == 1 else embs
        wide = getattr(embs[0], "d_in", 256) == 640  # emb3: rows with the phase means (pid_trunk_extract.py parts that carry `phase`)
        store = v2.load_store_phase(*a.trunk_store) if wide else load_store(*a.trunk_store)
        print(f"embedding {names} ({embs[0].players_trained} training players, {640 if wide else 256}-dim rows); {len(store)} trunk rows", flush=True)
    seq = None
    if "seq" in doc["components"]:
        paths = doc["seq_official"] if isinstance(doc["seq_official"], list) else [doc["seq_official"]]
        seqs = []
        for path in paths:  # one entry per network; an entry may list several files (one per pool: pool-adapted vectors)
            one = {}
            for f in (path if isinstance(path, list) else [path]):
                z = np.load(f)
                one.update(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
            seqs.append(one)
        seq = seqs[0] if len(seqs) == 1 else seqs
        print(f"sequence identity vectors: {len(paths)} network(s), {len(seqs[0])} games", flush=True)
    new, reports, cand_ids = {}, [], {}
    for tag, prefix in (("public", "pub2_"), ("private", "priv2_")):
        rows, rep, players = predict_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv", doc, emb, store, a.assign, a.workers, seq=seq, big_lam=a.big_lam)
        new.update({q: top for q, top in rows})
        reports.append(rep)
        cand_ids[prefix] = players
    old = pd.read_csv(a.old_submission, dtype=str)
    sample = pd.read_csv(T / "sample_submission_player_v2.csv", dtype=str)
    old_rows = {r.question_id: [getattr(r, f"top_{i}") for i in range(1, 6)] for r in old.itertuples()}
    for tag, prefix in (("public", "pub_"), ("private", "priv_")):
        cand_ids[prefix] = set(pd.read_csv(T / f"player_identification_candidates_{tag}.csv", dtype=str, usecols=["player_id"])["player_id"])
    out_rows, problems = [], []
    for qid in sample.question_id:
        top = new.get(qid) or old_rows.get(qid)
        if top is None:
            problems.append(f"no prediction for {qid}")
            continue
        prefix = qid[:qid.index("_") + 1]
        if len(set(top)) != 5 or any(p not in cand_ids[prefix] for p in top):
            problems.append(f"{qid}: Top-5 not five distinct candidates of its own pool")
        out_rows.append([qid, *top])
    sub = pd.DataFrame(out_rows, columns=list(sample.columns))
    if len(sub) != len(sample) or list(sub.question_id) != list(sample.question_id) or sub.isna().any().any():
        problems.append("rows / order differ from the sample submission")
    counts = sub.question_id.str.extract(r"^([a-z0-9]+)_")[0].value_counts().to_dict()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    report = {"submission": str(a.out), "rows": len(sub), "rows_by_prefix": counts, "problems": problems, "fusion": doc, "old_submission": str(a.old_submission), "pools": reports,
              "seconds": time.time() - t0}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    if problems:
        print(f"NOT written: {len(problems)} problems, e.g. {problems[:3]}", file=sys.stderr)
        return 1
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sub)} rows {counts}; format check against the sample passed ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
