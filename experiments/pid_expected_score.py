#!/usr/bin/env python
"""Label-free score estimates for the official files: the fusion is a softmax over the candidates, so each question's
expected official score is sum_k p(rank k candidate) e^-(k-1) over the submitted Top-5 (and p(Top-1) is the expected
Top-1 accuracy). Computed per pool for several fusion documents, with and without the one-to-one assignment, at a few
temperatures T (logits / T); the Public scores already observed calibrate T (--observed name=score, pub_ pool, e.g.
fusion=0.94873 fusion:assign=0.95735). All components are fitted once per pool and reused across the documents.

  python -I scripts/pid_expected_score.py --candidates ... --questions ... --fusions results/pid_fusion.json results/pid_fusion3.json \
      --observed fusion=0.94873 fusion3=0.97049 fusion:assign=0.95735 --out results/checks/expected_scores.json --workers 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.components import ALL_COMPONENTS, fit_hd, fit_local, fit_models, fit_orient, score_components  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

DECAY = np.exp(-np.arange(5))


def expected(S_f, idx, T):
    """(expected score, expected Top-1 accuracy) of the Top-5 lists idx under softmax(S_f / T)."""
    z = S_f / T
    z = z - z.max(axis=1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(axis=1, keepdims=True)
    p_top = np.take_along_axis(P, idx, axis=1)
    return float((p_top * DECAY[: idx.shape[1]]).sum(axis=1).mean()), float(p_top[:, 0].mean())


def pool_components(cand_path, q_path, cfg, workers, log=print):
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
            ok, _ = check_game(sgf, colour or "B")
            if ok:
                sgfs[f"q:{q['question_id']}:{i}"] = sgf
                keys.append(f"q:{q['question_id']}:{i}")
        q_keys.append(keys)
    usable = [k for k in q_keys if k]
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=workers)
    local, hd, orient = fit_local(by_player, moves, cfg), fit_hd(by_player, sgfs, cfg), fit_orient(by_player, moves, cfg)
    S = score_components(ALL_COMPONENTS, opening, style, usable, moves, feats, local, hd, sgfs, orient)
    n_games = np.array([q["declared_games"] for q, keys in zip(questions, q_keys) if keys], dtype=int)
    log(f"{Path(cand_path).name}: {len(by_player)} candidates, {len(usable)} questions; components in {time.time() - t0:.0f} s", flush=True)
    return S, n_games


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--fusions", nargs="+", required=True, help="fusion documents; the component cfg of the first is used for all")
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.8, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--observed", nargs="*", default=[], help="name[:assign]=Public score of the first pool, for the temperature fit")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    docs = {Path(f).stem.replace("pid_", ""): json.load(open(f)) for f in a.fusions}
    cfg = next(iter(docs.values()))["cfg"]
    for d in docs.values():  # every document must agree on the component hyper-parameters it names
        if any(d["cfg"].get(k) != v for k, v in d["cfg"].items() if k in cfg and cfg[k] != d["cfg"][k]):
            raise ValueError("fusion documents disagree on a shared cfg key")
    cfg = {**cfg, **{k: v for d in docs.values() for k, v in d["cfg"].items()}}
    observed = {}
    for o in a.observed:
        k, v = o.split("=")
        observed[k] = float(v)
    out = {"pools": [], "cfg": cfg, "temperatures": a.temperatures, "observed_public": observed}
    for cand_path, q_path in zip(a.candidates, a.questions):
        S, n_games = pool_components(cand_path, q_path, cfg, a.workers)
        prefix = Path(q_path).stem.split("_")[-1]
        res = {"pool": prefix, "n_questions": int(len(n_games)), "estimates": {}}
        for name, doc in docs.items():
            fus = Fusion()
            fus.w = np.array(doc["weights"])
            S_f = fus.scores([S[c] for c in doc["components"]])
            for mode, idx in (("", topk(S_f, 5)), (":assign", assign_topk(S_f, 5))):
                key = name + mode
                res["estimates"][key] = {}
                for T in a.temperatures:
                    sc, t1 = expected(S_f, idx, T)
                    by_n = {str(n): expected(S_f[n_games == n], idx[n_games == n], T)[0] for n in sorted(set(n_games.tolist()))}
                    res["estimates"][key][str(T)] = {"score": sc, "top1": t1, "by_n": by_n}
        out["pools"].append(res)
        print(f"== {prefix}: expected score (expected Top-1) by temperature " + " ".join(f"T={T}" for T in a.temperatures))
        for key, by_T in res["estimates"].items():
            obs = f"  Public {observed[key]:.5f}" if (prefix == "public" and key in observed) else ""
            print(f"  {key:16s} " + "  ".join(f"{v['score']:.4f} ({v['top1']:.3f})" for v in by_T.values()) + obs)
    # temperature fit on the observed Public scores (first pool)
    if observed and out["pools"]:
        est = out["pools"][0]["estimates"]
        errs = {T: float(np.sqrt(np.mean([(est[k][str(T)]["score"] - v) ** 2 for k, v in observed.items() if k in est]))) for T in a.temperatures}
        T_best = min(errs, key=errs.get)
        out["temperature_fit"] = {"rmse_by_T": {str(T): e for T, e in errs.items()}, "best_T": T_best}
        print(f"temperature fit on {len(observed)} Public scores: rmse by T " + " ".join(f"T={T}: {e:.4f}" for T, e in errs.items()) + f" -> T* = {T_best}")
        for p in out["pools"]:
            print(f"== {p['pool']} at T* = {T_best}: " + "  ".join(f"{k} {v[str(T_best)]['score']:.4f}" for k, v in p["estimates"].items()))
    out.update(pid_environment("cpu"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
