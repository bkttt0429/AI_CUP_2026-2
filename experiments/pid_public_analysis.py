#!/usr/bin/env python
"""Analyse the official Public pool with its now-known answers and match the same features on Private (METHOD_NOTES 2.20).

Fits every component on both official pools (opening, style, local, hd, orient N=1, orient N=4, opening_raw, ref with
the training folds 1-4 as the reference basis) and saves the score matrices. Public: per-component accuracy against the
answers, every fusion document's plain / assigned score (must reproduce the leaderboard), the profile of every wrong
answer, and error rates by margin and component agreement. Private: the same features for the submitted assignment, their
distribution next to Public's, the questions that look like Public's errors, and what a fusion refit on the Public
answers would change.

  python -I scripts/pid_public_analysis.py --workers 12 --out results/probe/public_analysis.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.components import (fit_hd, fit_local, fit_models, fit_opening_raw, fit_orient, fit_ref, parse_all, score_hd,  # noqa: E402
                                  score_local, score_questions, score_ref)
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.ref import reference_players  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402

T = ROOT / "data" / "tests"
COMPS = ("opening", "style", "local", "hd", "orient1", "orient4", "opening_raw", "ref", "trunk")
DOCS = {"fusion3": "pid_fusion3.json", "fusion3o": "pid_fusion3o.json", "fusion3or4": "pid_fusion3or4.json",
        "fusion5_raw4": "pid_fusion5_raw4.json", "fusion3o_ref": "pid_fusion3o_ref.json", "fusion3o_ref_trunk": "pid_fusion3o_ref_trunk.json"}
MAIN = "fusion3o_ref"  # the submitted assignment (submission_f3oref_assign / upload_fix)
DECAY = np.exp(-np.arange(5))


def doc_parts(doc):
    raw = doc["cfg"].get("orient_raw", 1)
    return [("orient4" if raw == 4 else "orient1") if c == "orient" else c for c in doc["components"]]


def question_scores(idx, truth):
    out = np.zeros(len(truth))
    for q, t in enumerate(truth):
        hit = np.where(idx[q] == t)[0]
        out[q] = DECAY[hit[0]] if len(hit) else 0.0
    return out


def trunc7(x):
    return math.floor(x * 1e7) / 1e7


def pool_matrices(cand_path, q_path, cfg, ref_model, workers, log=print, trunk_rows=None):
    t0 = time.time()
    cands = pools.read_candidates(cand_path)
    questions = pools.read_questions(q_path)
    sgfs = {f"c:{i}:{r.game_id}": r.sgf for i, r in enumerate(cands.itertuples())}
    by_player = {}
    for i, r in enumerate(cands.itertuples()):
        by_player.setdefault(r.player_id, []).append(f"c:{i}:{r.game_id}")
    q_keys = []
    for q in questions:
        keys = []
        for i, (sgf, colour) in enumerate(q["games"]):
            if check_game(sgf, colour or "B")[0]:
                sgfs[f"q:{q['question_id']}:{i}"] = sgf
                keys.append(f"q:{q['question_id']}:{i}")
        q_keys.append(keys)
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=workers)
    S = score_questions(opening, style, q_keys, moves, feats)
    S["local"] = score_local(fit_local(by_player, moves, cfg), q_keys, moves)
    S["hd"] = score_hd(fit_hd(by_player, sgfs, cfg), q_keys, sgfs)
    S["orient1"] = score_local(fit_orient(by_player, moves, {**cfg, "orient_raw": 1}), q_keys, moves)
    S["orient4"] = score_local(fit_orient(by_player, moves, {**cfg, "orient_raw": 4}), q_keys, moves)
    S["opening_raw"] = score_local(fit_opening_raw(by_player, moves, cfg), q_keys, moves)
    S["ref"] = score_ref(ref_model, by_player, q_keys, moves)
    S["trunk"] = score_trunk(fit_trunk(by_player, trunk_rows, cfg), q_keys, trunk_rows)
    qids = [q["question_id"] for q in questions]
    n = np.array([q["declared_games"] for q in questions], dtype=int)
    log(f"{Path(q_path).name}: {len(qids)} questions, all components in {time.time() - t0:.0f} s", flush=True)
    return {"S": S, "players": opening.players, "qids": qids, "n": n}


def features(P, S_f, idx_a):
    """Per-question features of the submitted assignment."""
    Z = standardise(S_f)
    Q = len(Z)
    top = idx_a[:, 0]
    argmax = Z.argmax(axis=1)
    srt = np.sort(Z, axis=1)
    margin_plain = srt[:, -1] - srt[:, -2]
    other = Z.copy()
    other[np.arange(Q), top] = -np.inf
    margin_assigned = Z[np.arange(Q), top] - other.max(axis=1)
    argmaxes = {c: P["S"][c].argmax(axis=1) for c in COMPS}
    agree = np.array([sum(int(argmaxes[c][q] == top[q]) for c in COMPS) for q in range(Q)])
    ranks = {c: np.array([int((P["S"][c][q] > P["S"][c][q, top[q]]).sum()) + 1 for q in range(Q)]) for c in COMPS}
    return {"margin_plain": margin_plain, "margin_assigned": margin_assigned, "changed": top != argmax, "agree": agree, "ranks_of_assigned": ranks,
            "argmaxes": argmaxes}


def quant(x):
    return [round(float(v), 2) for v in np.quantile(x, [0, 0.05, 0.1, 0.25, 0.5])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--truth", default=str(ROOT / "results" / "probe" / "upload_fix.csv"), help="file whose Public Top-1 are the answers (Public 1.0)")
    ap.add_argument("--main", default=MAIN, help="document whose assignment is the submitted one (feature tables and refit compare against it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    docs = {k: json.load(open(ROOT / "results" / v)) for k, v in DOCS.items()}
    cfg = docs[a.main]["cfg"]
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    refs = reference_players(pools.black_games(aicup_pid.TRAIN_DIR), folds_df, docs[a.main].get("ref_folds_official", [1, 2, 3, 4]),
                             min_black=cfg.get("ref_min_black", 30), cap=cfg.get("ref_cap", 40), seed=cfg.get("ref_seed", 0))
    ref_sgfs = {k: v[0] for k, v in load_games(sorted({k for ks in refs.values() for k in ks}), aicup_pid.TRAIN_DIR).items()}
    ref_model = fit_ref(refs, parse_all(ref_sgfs, a.workers), cfg)
    print(f"reference basis: {len(refs)} training players, {len(ref_sgfs)} games ({time.time() - t0:.0f} s)", flush=True)
    trunk_rows = load_store(ROOT / "data" / "trunk" / "official")
    P = {"pub": pool_matrices(T / "player_identification_candidates_public.csv", T / "player_identification_test_public.csv", cfg, ref_model, a.workers, trunk_rows=trunk_rows),
         "priv": pool_matrices(T / "player_identification_candidates_private.csv", T / "player_identification_test_private.csv", cfg, ref_model, a.workers, trunk_rows=trunk_rows)}
    np.savez_compressed(ROOT / "results" / "probe" / "official_scores_trunk.npz", **{f"{p}_{c}": P[p]["S"][c] for p in P for c in COMPS})
    fix = pd.read_csv(a.truth, dtype=str).set_index("question_id")
    pub = P["pub"]
    index = {p: i for i, p in enumerate(pub["players"])}
    truth = np.array([index[fix.loc[q, "top_1"]] for q in pub["qids"]])
    out = {"components": {}, "documents": {}}

    print("\n== Public: each component alone (answers known)")
    print(f"  {'component':12s} {'top1':>6s} {'top5':>6s} {'mean rank':>9s}  top1 by n=5/10/15/20")
    for c in COMPS:
        r = np.array([int((pub["S"][c][q] > pub["S"][c][q, truth[q]]).sum()) + 1 for q in range(len(truth))])
        by_n = [float((r[pub["n"] == k] == 1).mean()) for k in (5, 10, 15, 20)]
        out["components"][c] = {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()), "mean_rank": float(r.mean()), "top1_by_n": by_n}
        print(f"  {c:12s} {(r == 1).mean():6.3f} {(r <= 5).mean():6.3f} {r.mean():9.1f}  " + " / ".join(f"{x:.2f}" for x in by_n))

    print("\n== Public: fusion documents, plain and assigned (offline, must equal the leaderboard)")
    fused = {}
    for name, doc in docs.items():
        fus = Fusion()
        fus.w = np.array(doc["weights"])
        for p in P:
            fused[(name, p)] = fus.scores([P[p]["S"][c] for c in doc_parts(doc)])
        S_f = fused[(name, "pub")]
        sp, sa = question_scores(topk(S_f, 5), truth), question_scores(assign_topk(S_f, 5), truth)
        out["documents"][name] = {"plain": trunc7(sp.mean()), "assigned": trunc7(sa.mean()), "wrong_plain": int((topk(S_f, 1)[:, 0] != truth).sum()),
                                  "wrong_assigned": int((assign_topk(S_f, 5)[:, 0] != truth).sum())}
        d = out["documents"][name]
        print(f"  {name:18s} plain {d['plain']:.7f} ({d['wrong_plain']:2d} wrong)   assigned {d['assigned']:.7f} ({d['wrong_assigned']} wrong)")
    print("\n== Top-1 of each assigned document vs the submitted one (pub / priv questions that differ)")
    base_top = {p: assign_topk(fused[(a.main, p)], 5)[:, 0] for p in P}
    out["top1_diff_vs_main"] = {}
    for name in docs:
        diff = {p: [P[p]["qids"][q] for q in np.where(assign_topk(fused[(name, p)], 5)[:, 0] != base_top[p])[0]] for p in P}
        out["top1_diff_vs_main"][name] = diff
        print(f"  {name:18s} pub {len(diff['pub']):3d}  priv {len(diff['priv']):3d}" + (f"   priv: {diff['priv'][:12]}" if 0 < len(diff['priv']) <= 12 else ""))

    S_f = fused[(a.main, "pub")]
    idx_p, idx_a = topk(S_f, 5), assign_topk(S_f, 5)
    F = {p: features(P[p], fused[(a.main, p)], assign_topk(fused[(a.main, p)], 5)) for p in P}
    fp = F["pub"]
    Z = standardise(S_f)
    wrong_plain, wrong_assigned = idx_p[:, 0] != truth, idx_a[:, 0] != truth
    print(f"\n== Public: wrong answers of {a.main} (plain {int(wrong_plain.sum())}, after assignment {int(wrong_assigned.sum())}; "
          f"assignment fixed {int((wrong_plain & ~wrong_assigned).sum())}, broke {int((~wrong_plain & wrong_assigned).sum())})")
    print("  question    n  plain/assigned  truth rank  margin(assigned)  agree  component ranks of the truth (" + "/".join(COMPS) + ")")
    errs = []
    for q in np.where(wrong_plain | wrong_assigned)[0]:
        rank_t = int((Z[q] > Z[q, truth[q]]).sum()) + 1
        cr = [int((pub["S"][c][q] > pub["S"][c][q, truth[q]]).sum()) + 1 for c in COMPS]
        tag = ("wrong/wrong" if wrong_assigned[q] else "wrong/fixed") if wrong_plain[q] else "right/broken"
        errs.append({"question": pub["qids"][q], "n": int(pub["n"][q]), "status": tag, "truth_rank": rank_t, "margin_assigned": float(fp["margin_assigned"][q]),
                     "agree": int(fp["agree"][q]), "component_ranks_of_truth": dict(zip(COMPS, cr))})
        print(f"  {pub['qids'][q]} {pub['n'][q]:2d}  {tag:12s}  {rank_t:9d}  {fp['margin_assigned'][q]:+15.2f}  {fp['agree'][q]:5d}  " + "/".join(map(str, cr)))
    out["public_errors"] = errs

    print("\n== Public: error rate of the plain argmax by fused margin and by component agreement")
    plain_agree = np.array([sum(int(P['pub']['S'][c][q].argmax() == idx_p[q, 0]) for c in COMPS) for q in range(len(truth))])
    rows = []
    for lo, hi in ((-9, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        m = (fp["margin_plain"] >= lo) & (fp["margin_plain"] < hi)
        rows.append({"margin": [lo, hi], "questions": int(m.sum()), "wrong_plain": int(wrong_plain[m].sum()), "wrong_assigned": int(wrong_assigned[m].sum())})
        print(f"  plain margin [{lo:5.2f}, {hi:5.2f}): {int(m.sum()):3d} questions, plain wrong {int(wrong_plain[m].sum()):2d}, still wrong after assignment {int(wrong_assigned[m].sum())}")
    for k in range(len(COMPS) + 1):
        m = plain_agree == k
        if m.any():
            print(f"  {k} of {len(COMPS)} components agree with the plain Top-1: {int(m.sum()):3d} questions, plain wrong {int(wrong_plain[m].sum())}")
    out["error_by_margin"] = rows

    print("\n== Public vs Private: features of the submitted assignment")
    cmp = {}
    for key, f in (("margin_plain quantiles 0/5/10/25/50%", lambda x: quant(x["margin_plain"])),
                   ("margin_assigned quantiles 0/5/10/25/50%", lambda x: quant(x["margin_assigned"])),
                   ("assignment overrode the argmax", lambda x: int(x["changed"].sum())),
                   ("questions with margin_assigned < 0.5", lambda x: int((x["margin_assigned"] < 0.5).sum())),
                   ("component agreement 8/7/6/5/<=4", lambda x: [int((x["agree"] == 8).sum()), int((x["agree"] == 7).sum()), int((x["agree"] == 6).sum()),
                                                              int((x["agree"] == 5).sum()), int((x["agree"] <= 4).sum())])):
        cmp[key] = {p: f(F[p]) for p in P}
        print(f"  {key:42s} pub {cmp[key]['pub']}   priv {cmp[key]['priv']}")
    print("  per component: share of questions whose argmax equals the submitted Top-1 (pub | priv)")
    for c in COMPS:
        cmp[f"agree_{c}"] = {p: float((F[p]["argmaxes"][c] == assign_topk(fused[(a.main, p)], 5)[:, 0]).mean()) for p in P}
        print(f"    {c:12s} {cmp[f'agree_{c}']['pub']:.3f} | {cmp[f'agree_{c}']['priv']:.3f}   (pub accuracy {out['components'][c]['top1']:.3f})")
    out["pub_vs_priv"] = cmp

    # Public-like risk on Private: the Public errors sat at low margin / overridden / low agreement
    fq = F["priv"]
    risk = np.argsort(fq["margin_assigned"])[:20]
    print("\n== Private: the 20 lowest-margin questions of the submitted assignment (Public's errors live here)")
    print("  question     n  margin(assigned)  overridden  agree  component ranks of the assigned candidate")
    priv_risk = []
    for q in risk:
        cr = [int(fq["ranks_of_assigned"][c][q]) for c in COMPS]
        priv_risk.append({"question": P["priv"]["qids"][q], "n": int(P["priv"]["n"][q]), "margin_assigned": float(fq["margin_assigned"][q]),
                          "overridden": bool(fq["changed"][q]), "agree": int(fq["agree"][q]), "component_ranks_of_assigned": dict(zip(COMPS, cr))})
        print(f"  {P['priv']['qids'][q]} {P['priv']['n'][q]:2d}  {fq['margin_assigned'][q]:+15.2f}  {str(bool(fq['changed'][q])):>10s}  {fq['agree'][q]:5d}  " + "/".join(map(str, cr)))
    out["private_risk"] = priv_risk
    pub_rate = [(r["wrong_assigned"], r["questions"]) for r in rows]
    exp = 0.0
    for (lo, hi), (w, nq) in zip([tuple(r["margin"]) for r in rows], pub_rate):
        m = (fq["margin_plain"] >= lo) & (fq["margin_plain"] < hi)
        exp += (w / nq if nq else 0) * m.sum()
    print(f"\nexpected Private errors if Private's error rate per plain-margin bin equals Public's: {exp:.1f}")
    out["expected_private_errors_by_public_rates"] = float(exp)

    # refit the fusion weights on the Public answers (8 components), 5-fold within Public, then apply to Private
    comps = list(COMPS)
    folds = np.arange(len(truth)) % 5
    cv_scores = np.zeros(len(truth))
    for k in range(5):
        tr = folds != k
        fus = Fusion().fit([pub["S"][c][tr] for c in comps], truth[tr])
        S_k = fus.scores([pub["S"][c] for c in comps])
        cv_scores[~tr] = question_scores(assign_topk(S_k, 5), truth)[~tr]
    fus_pub = Fusion().fit([pub["S"][c] for c in comps], truth)
    S_pub = fus_pub.scores([pub["S"][c] for c in comps])
    in_sample = question_scores(assign_topk(S_pub, 5), truth).mean()
    S_priv = fus_pub.scores([P["priv"]["S"][c] for c in comps])
    new_top = assign_topk(S_priv, 5)[:, 0]
    old_top = assign_topk(fused[(a.main, "priv")], 5)[:, 0]
    changed_priv = [P["priv"]["qids"][q] for q in np.where(new_top != old_top)[0]]
    print(f"\n== Fusion refit on the Public answers (8 components): weights {np.round(fus_pub.w, 2).tolist()}")
    print(f"  Public assigned: in-sample {in_sample:.5f}; 5-fold within Public {cv_scores.mean():.5f} (submitted {a.main}+assign {out['documents'][a.main]['assigned']:.5f})")
    print(f"  applied to Private: {len(changed_priv)} Private Top-1 would change: {changed_priv}")
    out["public_refit"] = {"weights": dict(zip(comps, fus_pub.w.tolist())), "public_in_sample": float(in_sample), "public_5fold": float(cv_scores.mean()),
                           "private_top1_changes": changed_priv}
    out.update({"seconds": time.time() - t0, **pid_environment("cpu")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
