#!/usr/bin/env python
"""B3 fusion test on leak-free simulated pools (METHOD_NOTES 2.26): build | capture | eval.

Leakage rules (the rank session's reports/b1_reply_2026-11-08.md section 2, plus one more):
  * one pool per SINGLE fold v (one question per player, all eligible players of fold v, 75 questions per size), so that
    every player of the pool is scored by the same rank models and none of them was seen by those models;
  * rank models: only the six ct9 members val{v}_* (trained without fold v); per-ply L1b heads trained on the other fold
    pair (folds 1, 2 -> heads3_B, folds 3, 4 -> heads3_A);
  * identification components as in pid_final_cv: references and the embedding from the other fold pair, the rest fitted
    on the pool's own candidates; fusion weights cross-fitted (pools of folds 1, 2 weighted from pools 3, 4 and back),
    the same procedure for both arms.
build    writes, per fold, the rank-test-format CSV (candidate chunks pub_c<idx>_k<j> of 20 games, questions pub_q_<idx>)
         and pool.json (players, question owners, sizes, game keys) under --work/f<v>/
capture  runs the rank session's trunk_predict.py predict (read-only) with a hook keeping each pass's 32-dim question
         embedding zq and 128-dim pooled game vectors -> --work/f<v>/b3.npz
eval     rebuilds every component on the same pools, adds b3 (candidate = mean of its 5 chunks, z-scored, cosine, passes
         averaged; zq is the pre-set layer), compares main+b3 vs main, with and without the assignment.
Pre-specified (primary, the no-one-to-one scenario): main+b3 vs main WITHOUT assignment, player bootstrap; both
cross-fit halves positive, CI lower > 0, and >= 0.003 overall or >= 0.005 at n=5. Also required: no loss with assignment.
Added 11-09 before any eval output (the 8 remaining errors leave the primary test almost no power):
  secondary  held-out log P(truth) of the cross-fitted fusion, main+b3 vs main: CI lower > 0 and both halves > 0;
  stress     candidates cut to their first 2 chunks (40 games; every component refitted on them), official score
             without assignment, main+b3 vs main: CI lower > 0 and both halves > 0; 1 chunk (20 games) report only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402

FOLDS = (1, 2, 3, 4)
HEADS = {1: "heads3_B", 2: "heads3_B", 3: "heads3_A", 4: "heads3_A"}
PER_SIZE, CHUNK = 75, 20


def make(games, folds_df, v):
    return pools.make_pool_one_per_player(games, folds_df, [v], n_players=10_000, cand_games=pools.CAND_GAMES, per_size=PER_SIZE, seed=0,
                                          pool_id=f"f{v}", order="shuffle")


def build(a):
    from aicup.scoring import load_games
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    cols = ["question_id", "num_games", *[f"sgf_{i}" for i in range(1, 21)], *[f"color_{i}" for i in range(1, 21)]]
    for v in FOLDS:
        cands, qs, info = make(games, folds_df, v)
        by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
        players = sorted(by_player)
        keys = sorted(set(cands.game_key) | {k for ks in qs.game_keys for k in ks})
        sgf = {k: s for k, (s, _c) in load_games(keys, aicup_pid.TRAIN_DIR).items()}
        rows, chunks = [], {}
        for ci, p in enumerate(players):
            ks = by_player[p]
            if len(ks) % CHUNK:
                raise ValueError(f"{p}: {len(ks)} candidate games")
            for j in range(len(ks) // CHUNK):
                qid = f"pub_c{ci:04d}_k{j}"
                part = ks[j * CHUNK:(j + 1) * CHUNK]
                rows.append({"question_id": qid, "num_games": str(CHUNK), **{f"sgf_{i + 1}": sgf[k] for i, k in enumerate(part)},
                             **{f"color_{i + 1}": "B" for i in range(CHUNK)}})
                chunks[qid] = p
        owners = {}
        for qi, r in enumerate(qs.itertuples()):
            qid = f"pub_q_{qi:04d}"
            ks = list(r.game_keys)
            rows.append({"question_id": qid, "num_games": str(len(ks)), **{f"sgf_{i + 1}": sgf[k] for i, k in enumerate(ks)},
                         **{f"color_{i + 1}": "B" for i in range(len(ks))}})
            owners[qid] = r.player_id
        df = pd.DataFrame(rows).reindex(columns=cols).fillna("")
        out = Path(a.work) / f"f{v}"
        (out / "inputs").mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "inputs" / "pool.csv", index=False)
        (out / "pool.json").write_text(json.dumps({"fold": v, "info": info, "players": players, "chunks": chunks, "owners": owners,
                                                   "n": {f"pub_q_{qi:04d}": int(n) for qi, n in enumerate(qs.n_games)}, "heads": HEADS[v],
                                                   "rows": len(df), "games": int(df.num_games.astype(int).sum())}, indent=1) + "\n")
        print(f"fold {v}: {len(players)} players, {len(qs)} questions (sizes {qs.n_games.value_counts().sort_index().to_dict()}), {len(df)} rows, "
              f"{int(df.num_games.astype(int).sum())} games, heads {HEADS[v]} -> {out}", flush=True)


def capture(a):
    import torch
    import trunk_predict as tp
    import trunk_seq_check as ts
    v, out = a.fold, Path(a.work) / f"f{a.fold}"
    meta = json.loads((out / "pool.json").read_text())
    cap, loaded = [], {}
    orig_load = ts.load_checkpoint

    def hooked_load(path, device="cpu"):
        m, stats, r, side = orig_load(path, device)
        if v in side.get("train_folds", []) or side.get("val_fold") != v:
            raise SystemExit(f"{path}: trained on folds {side.get('train_folds')}, val fold {side.get('val_fold')} -> would leak into fold {v}")
        loaded[id(m)] = Path(path).stem
        return m, stats, r, side

    def hooked_predict(model, R, questions, X, offsets, device, batch=64):
        model.eval()
        P, zqs, pooled = [], [], []
        with torch.no_grad():
            for i in range(0, len(R), batch):
                r = torch.as_tensor(R[i:i + batch], device=device)
                tokens, mask, game_of = ts.batch_tokens(X, offsets, questions[i:i + batch], device)
                pg = model.pool_games(model.encode_games(tokens, mask), game_of, r.shape[0])
                zq = torch.relu(model.question(pg))
                P.append(torch.softmax(model.mlp(torch.cat([r, zq], dim=1)), dim=1).double().cpu().numpy())
                zqs.append(zq.float().cpu().numpy())
                pooled.append(pg.float().cpu().numpy())
        cap.append((loaded.get(id(model), "?"), np.concatenate(zqs), np.concatenate(pooled)))
        return np.concatenate(P)

    ts.load_checkpoint, ts.predict = hooked_load, hooked_predict
    models = sorted(str(p) for p in (Path.home() / "BK_AICUP_2026-1" / "results" / "models" / "ct_cv_m33").glob(f"val{v}_[CT]-pc-f0-x40-oh_s[012].pt"))
    if len(models) != 6:
        raise SystemExit(f"fold {v}: expected six val{v} members, found {len(models)}")
    n = str(meta["rows"])
    rc = tp.main(["predict", "--inputs", str(out / "inputs" / "pool.csv"), "--shards", str(out / "submission_rank.csv"), "--nshards", "8",
                  "--vec", str(out / "vec"), "--models", *models, "--fallback-submission", str(out / "submission_rank.csv"), "--allow-missing-hd",
                  "--out", str(out / "submission_rank_ct9.csv"), "--expect-rows", n, "--expect-public", n, "--expect-private", "0", "--l1b-names", HEADS[v]])
    rep = json.loads((out / "submission_rank_ct9_report.json").read_text())
    used = rep.get("l1b_models_used")
    folds_of = {m["name"]: m.get("train_folds") for m in rep.get("l1b_models", [])}
    if used != [HEADS[v]] or v in (folds_of.get(HEADS[v]) or [v]) or rep.get("fallback_questions"):
        raise SystemExit(f"fold {v}: L1b heads used {used} (train folds {folds_of}), fallbacks {len(rep.get('fallback_questions') or [])} -> not leak-free")
    qids = pd.read_csv(out / "submission_rank_ct9_proba.csv").question_id.to_numpy(dtype=str)
    np.savez(out / "b3.npz", qids=qids, names=np.array([c[0] for c in cap]), zq=np.stack([c[1] for c in cap]), pooled=np.stack([c[2] for c in cap]))
    print(f"fold {v}: predict exit {rc}, {len(cap)} passes ({sorted(set(c[0] for c in cap))}) -> {out / 'b3.npz'}", flush=True)


def b3_matrix(work, v, players, q_ids, layer="zq", k=5):
    """[Q, C] aicup_pid.b3.score_b3 (candidate = mean of its first k chunks), candidates in `players` order."""
    from aicup_pid.b3 import score_b3
    out = Path(work) / f"f{v}"
    meta = json.loads((out / "pool.json").read_text())
    z = np.load(out / "b3.npz", allow_pickle=True)  # our own capture file; 11-09 captures stored qids as an object array
    row = {str(q): i for i, q in enumerate(z["qids"])}
    chunks = {}
    for qid, p in meta["chunks"].items():
        if int(qid.rsplit("_k", 1)[1]) < k:
            chunks.setdefault(p, []).append(qid)
    S, info = score_b3(z[layer], row, chunks, players, q_ids)
    if info["missing_questions"] or info["missing_candidates"]:
        raise ValueError(f"fold {v}: captures miss {len(info['missing_questions'])} questions / {len(info['missing_candidates'])} candidates")
    return S


def fit_ragged(comp_lists, truths, l2=1e-3, max_iter=200):
    """aicup_pid.fusion.Fusion fitted over pools with different candidate counts: the same row standardisation and loss
    (mean cross-entropy over all questions + l2 |w|^2), summed pool by pool instead of over one stacked matrix."""
    import torch
    from aicup_pid.fusion import Fusion, standardise
    Zs = [torch.tensor(np.stack([standardise(S) for S in comps]), dtype=torch.float64) for comps in comp_lists]
    ys = [torch.tensor(np.asarray(t, dtype=np.int64)) for t in truths]
    total = sum(len(y) for y in ys)
    w = torch.ones(Zs[0].shape[0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = sum(torch.nn.functional.cross_entropy(torch.einsum("j,jqc->qc", w, Z), y, reduction="sum") for Z, y in zip(Zs, ys)) / total + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    fus = Fusion(l2=l2, max_iter=max_iter)
    fus.w = w.detach().numpy().copy()
    return fus


SCENARIOS = {"full": 5, "k2": 2, "k1": 1}  # candidate chunks of 20 games used: 100 (the pre-specified test), 40, 20 games


def log_p_truth(Sf, truth):
    m = Sf.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Sf - m).sum(axis=1))
    return Sf[np.arange(len(Sf)), truth] - lse


def evaluate(a):
    import torch
    from aicup.ladder import to_json
    from aicup.scoring import load_games
    from aicup_pid.assign import assign_topk
    from aicup_pid.components import features_all, fit_from_parsed, fit_local, fit_orient, fit_ref, parse_all, score_components, score_ref
    from aicup_pid.embed import PlayerEmbedding, score_embed
    from aicup_pid.fusion import topk
    from aicup_pid.ref import reference_players
    from aicup_pid.stats import paired_player_bootstrap
    from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk
    doc = json.loads(Path(a.fusion).read_text())
    cfg, MAIN = doc["cfg"], tuple(doc["components"])
    store = load_store(*a.trunk_store)
    emb = {k: PlayerEmbedding.from_state(torch.load(p, map_location="cpu", weights_only=False)) for k, p in (("12", a.emb12), ("34", a.emb34))}
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    scen = {s: SCENARIOS[s] for s in a.scenarios}
    runs = {s: {} for s in scen}
    for v in FOLDS:
        meta = json.loads((Path(a.work) / f"f{v}" / "pool.json").read_text())
        cands, qs, _ = make(games, folds_df, v)
        by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
        players = sorted(by_player)
        if players != meta["players"]:
            raise ValueError(f"fold {v}: pool differs from the built one")
        q_ids = [f"pub_q_{i:04d}" for i in range(len(qs))]
        if [meta["owners"][q] for q in q_ids] != qs.player_id.tolist():
            raise ValueError(f"fold {v}: question owners differ from the built pool")
        other = (3, 4) if v in (1, 2) else (1, 2)
        refs = reference_players(games, folds_df, list(other), min_black=cfg.get("ref_min_black", 30), cap=cfg.get("ref_cap", 40), seed=cfg.get("ref_seed", 0),
                                 exclude=set(players))
        pool_keys = set(cands.game_key) | {k for ks in qs.game_keys for k in ks}
        keys = pool_keys | {k for ks in refs.values() for k in ks}
        sgfs = {k: s for k, (s, _c) in load_games(sorted(keys), aicup_pid.TRAIN_DIR).items()}
        moves, feats = parse_all(sgfs, a.workers), features_all({k: sgfs[k] for k in pool_keys}, a.workers)
        q_keys = [list(k) for k in qs.game_keys]
        ref_model = fit_ref(refs, moves, cfg)
        first = pd.read_csv(Path(a.work) / f"f{v}" / "inputs" / "pool.csv", usecols=["question_id", "sgf_1"], dtype=str).set_index("question_id").sgf_1
        bad = [q for q, p in meta["chunks"].items() if first[q].strip() != sgfs[by_player[p][CHUNK * int(q.rsplit("_k", 1)[1])]].strip()]
        if bad:  # the stress scenarios need chunk j = candidate games [20 j, 20 j + 20) exactly as captured
            raise ValueError(f"fold {v}: {len(bad)} chunks do not start with the expected game, e.g. {bad[:3]}")
        for s, k in scen.items():
            bp = {p: ks[:CHUNK * k] for p, ks in by_player.items()}  # chunk j = games [20 j, 20 j + 20) of the build
            opening, style = fit_from_parsed(bp, moves, feats, cfg)
            S = score_components(("opening", "style", "local", "orient"), opening, style, q_keys, moves, feats, local=fit_local(bp, moves, cfg),
                                 orient=fit_orient(bp, moves, cfg))
            S["ref"] = score_ref(ref_model, bp, q_keys, moves)
            S["trunk"] = score_trunk(fit_trunk(bp, store, cfg), q_keys, store)
            S["emb"] = score_embed(emb["12" if v in (1, 2) else "34"], bp, q_keys, store)  # embx_p12 never saw folds 1, 2; embx_p34 never saw 3, 4
            S["b3"] = b3_matrix(a.work, v, players, q_ids, "zq", k)
            if s == "full":
                S["b3_pooled"] = b3_matrix(a.work, v, players, q_ids, "pooled", k)
            truth = np.array([opening.index[p] for p in qs.player_id])
            runs[s][v] = {"S": S, "truth": truth, "pid": qs.player_id.to_numpy(dtype=str), "n": qs.n_games.to_numpy()}
            print(f"fold {v} [{s}: {CHUNK * k} candidate games]: {len(players)} candidates, {len(qs)} questions; components ready", flush=True)
    if a.final_fusion_out:  # the no-assignment fallback document (METHOD_NOTES 2.26, user's option A): main + b3 fitted on all four pools
        if "full" not in runs:
            raise ValueError("--final-fusion-out needs the full scenario")
        R = runs["full"]
        cs = (*MAIN, "b3")
        fus = fit_ragged([[R[v]["S"][c] for c in cs] for v in FOLDS], [R[v]["truth"] for v in FOLDS])
        out_doc = {k: v for k, v in doc.items() if k not in ("components", "weights", "fitted_on_pools", "questions")}
        out_doc.update({"components": list(cs), "weights": fus.w.tolist(), "fitted_on_pools": [list(FOLDS)], "questions": int(sum(len(R[v]["truth"]) for v in FOLDS)),
                        "fitted_by": "scripts/pid_b3_pools.py eval --final-fusion-out (single-fold leak-free pools f1-f4, 100 candidate games)",
                        "b3_official": {"layer": "zq", "chunk": CHUNK, "models": "AI_CUP_2026-1 results/models/ct_cv_m33/val{1..4}_{C,T}-pc-f0-x40-oh_s{0,1,2}.pt",
                                        "l1b_heads": ["heads3_A", "heads3_B"], "capture": "scripts/pid_b3_official.py capture"},
                        "base_fusion": a.fusion})
        Path(a.final_fusion_out).write_text(json.dumps(out_doc, indent=1) + "\n")
        print(f"final fusion {list(cs)} weights {[round(w, 3) for w in fus.w]} -> {a.final_fusion_out}", flush=True)
    comps, summary = {}, {}
    for s in scen:
        R = runs[s]
        arms = {"main": MAIN, "main+b3": (*MAIN, "b3")}
        if s == "full":
            arms["main+b3_pooled"] = (*MAIN, "b3_pooled")
        per_q, per_lp, summ = {}, {}, {}
        for v, r in R.items():
            others = [R[u] for u in R if (u in (1, 2)) != (v in (1, 2))]
            for name, cs in arms.items():
                fus = fit_ragged([[o["S"][c] for c in cs] for o in others], [o["truth"] for o in others])
                Sf = fus.scores([r["S"][c] for c in cs])
                per_lp.setdefault(name, {})[v] = log_p_truth(Sf, r["truth"])
                for arm, idx in ((name, topk(Sf, 5)), (name + "+assign", assign_topk(Sf, 5))):
                    hit = [np.where(idx[q] == r["truth"][q])[0] for q in range(len(idx))]
                    sc = np.array([np.exp(-h[0]) if len(h) else 0.0 for h in hit])
                    per_q.setdefault(arm, {})[v] = sc
                    summ.setdefault(arm, {})[v] = {"score": float(sc.mean()), "wrong": int((idx[:, 0] != r["truth"]).sum()), "weights": fus.w.tolist()}
            for c in ("trunk", "emb", "b3", "b3_pooled"):
                if c not in r["S"]:
                    continue
                idx = topk(r["S"][c], 5)
                sc = np.array([np.exp(-np.where(idx[q] == r["truth"][q])[0][0]) if (idx[q] == r["truth"][q]).any() else 0.0 for q in range(len(idx))])
                summ.setdefault(c + " alone", {})[v] = {"score": float(sc.mean())}
        pid = np.concatenate([R[v]["pid"] for v in FOLDS])
        n = np.concatenate([R[v]["n"] for v in FOLDS])
        # (metric, x, z, group). Rules fixed 2026-11-09 before any eval output was seen (METHOD_NOTES 2.26):
        #   pre_specified        full, official score, no assignment: CI lower > 0, both halves > 0, >= 0.003 overall or >= 0.005 at n=5
        #   required_no_loss     full, with assignment: diff >= 0
        #   secondary            full, held-out log P(truth) of the cross-fitted fusion: CI lower > 0 and both halves > 0 (= b3 adds information)
        #   stress               k2 (40 candidate games), official score, no assignment: CI lower > 0 and both halves > 0
        if s == "full":
            todo = [("score", "main+b3", "main", "pre_specified"), ("score", "main+b3+assign", "main+assign", "required_no_loss"),
                    ("logp", "main+b3", "main", "secondary"),
                    ("score", "main+b3_pooled", "main", "report_only"), ("score", "main+b3_pooled+assign", "main+assign", "report_only"),
                    ("logp", "main+b3_pooled", "main", "report_only")]
        else:
            todo = [("score", "main+b3", "main", "stress" if s == "k2" else "report_only"), ("score", "main+b3+assign", "main+assign", "report_only"),
                    ("logp", "main+b3", "main", "report_only")]
        for metric, x, z, group in todo:
            src = per_q if metric == "score" else per_lp
            qx, qz = np.concatenate([src[x][v] for v in FOLDS]), np.concatenate([src[z][v] for v in FOLDS])
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            halves = [float(np.mean(np.concatenate([src[x][v] - src[z][v] for v in hv]))) for hv in ((1, 2), (3, 4))]
            n5 = n == 5
            d.update({"halves": halves, "diff_n5": float(np.mean((qx - qz)[n5])), "by_n": {str(k): float(np.mean((qx - qz)[n == k])) for k in sorted(set(n.tolist()))}})
            base = bool(d["ci95"][0] > 0 and all(h > 0 for h in halves))
            d["pass"] = {"pre_specified": base and (d["diff"] >= 0.003 or d["diff_n5"] >= 0.005), "required_no_loss": bool(d["diff"] >= 0),
                         "secondary": base, "stress": base}.get(group)
            key = f"{metric}: {x} vs {z}" if s == "full" else f"{s} {metric}: {x} vs {z}"
            comps[key if metric == "logp" or s != "full" else f"{x} vs {z}"] = {"scenario": s, "metric": metric, "group": group, **d}
            print(f"[{group}] {s} {metric} {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  halves {halves[0]:+.4f} {halves[1]:+.4f}  "
                  f"n=5 {d['diff_n5']:+.4f}" + (f"  -> {'PASS' if d['pass'] else 'no'}" if d["pass"] is not None else ""), flush=True)
        for arm, per in summ.items():
            tot = {"score": float(np.mean([p["score"] for p in per.values()]))}
            if "wrong" in next(iter(per.values())):
                tot["wrong"] = int(sum(p["wrong"] for p in per.values()))
            print(f"{s} {arm}: " + "  ".join(f"f{v} {p['score']:.4f}" + (f" ({p['wrong']} wrong)" if "wrong" in p else "") for v, p in per.items())
                  + f"  | mean {tot['score']:.4f}" + (f", {tot['wrong']} wrong of {len(pid)}" if "wrong" in tot else ""), flush=True)
        summary[s] = summ
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json({"comparisons": comps, "summary": summary, "questions": int(len(pid)), "fusion": a.fusion,
                                    "scenarios": {s: CHUNK * k for s, k in scen.items()}}, indent=1) + "\n")
    print(f"wrote {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "capture", "eval"))
    ap.add_argument("--work", default="/work/u2507115/pid_b3/pools")
    ap.add_argument("--fold", type=int, choices=FOLDS)
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion3o_ref_trunk_emb.json"))
    ap.add_argument("--trunk-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "train"), str(ROOT / "data" / "trunk" / "train_rest")])
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "embx_p12.pt"), help="embedding trained on folds 3,4 (+ old official): for pools of folds 1, 2")
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "embx_p34.pt"), help="embedding trained on folds 1,2 (+ old official): for pools of folds 3, 4")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--final-fusion-out", default=None, help="also fit main+b3 on all four full pools and write that fusion document")
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    ap.add_argument("--out", default=str(ROOT / "results" / "b1" / "pools_b3.json"))
    a = ap.parse_args(argv)
    return {"build": build, "capture": capture, "eval": evaluate}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
