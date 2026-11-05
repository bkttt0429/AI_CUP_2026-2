#!/usr/bin/env python
"""Public-leaderboard probing of the final one-to-one submission (aicup_pid.probe; METHOD_NOTES 2.17).

  state   fit the fusion document on the official files, apply the assignment, save per-question Top-5 lists, fused
          scores and margins (results/probe/state.json), and print the decompositions of the Public loss
  design  write upload k: the current best file (base + decoded fixes) with a group of unresolved Public questions
          cycled (probe.CYCLE); the group is the least-confident questions, or half of a pending group with wrong members
  decode  read the observed Public score of upload k, recover the truth positions of its tested questions, record the
          fixes; print the predicted score of the next best file

  python -I scripts/pid_probe.py state --candidates ... --questions ... --fusion results/pid_fusion3.json --base-score 0.9873479 --workers 12
  python -I scripts/pid_probe.py design --k 1 --out results/probe/upload_1.csv
  python -I scripts/pid_probe.py decode --k 1 --score 0.98xxxxx
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
from aicup_pid import pools, probe  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.components import fit_hd, fit_local, fit_models, fit_opening_raw, fit_orient, score_components  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

STATE = ROOT / "results" / "probe" / "state.json"


def pool_lists(cand_path, q_path, doc, workers, log=print):
    """[(question_id, declared n, top5 ids in assigned order, their fused scores, margin)] for one official pool."""
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
    if not all(q_keys):
        raise ValueError("a question without a usable game: probing needs every question scored")
    cfg, comps = doc["cfg"], doc["components"]
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=workers)
    local = fit_local(by_player, moves, cfg) if "local" in comps else None
    hd = fit_hd(by_player, sgfs, cfg) if "hd" in comps else None
    orient = fit_orient(by_player, moves, cfg) if "orient" in comps else None
    opening_raw = fit_opening_raw(by_player, moves, cfg) if "opening_raw" in comps else None
    S = score_components(comps, opening, style, q_keys, moves, feats, local, hd, sgfs, orient, opening_raw)
    fus = Fusion()
    fus.w = np.array(doc["weights"])
    S_f = fus.scores([S[c] for c in comps])
    idx = assign_topk(S_f, 5)
    sd = S_f.std(axis=1)
    out = []
    for q, question in enumerate(questions):
        top = [opening.players[i] for i in idx[q]]
        scores = [float(S_f[q, i]) for i in idx[q]]
        others = np.delete(S_f[q], idx[q, 0])
        out.append({"question_id": question["question_id"], "n": int(question["declared_games"]), "top5": top, "scores": scores,
                    "margin": float((S_f[q, idx[q, 0]] - others.max()) / sd[q])})
    log(f"{Path(q_path).name}: {len(out)} questions scored and assigned in {time.time() - t0:.0f} s", flush=True)
    return out


def load_state():
    return json.load(open(STATE))


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(to_json(state, indent=1) + "\n")


def current_lists(state):
    """{question_id: list} of the best file so far: assigned lists with the decoded fixes applied."""
    lists = {}
    for pool in state["pools"]:
        for q in pool["questions"]:
            lst = list(q["top5"])
            t = q.get("truth_pos")
            if isinstance(t, int) and t >= 2:
                lst = probe.fixed_list(lst, t)
            lists[q["question_id"]] = lst
    return lists


def predicted_score(state):
    gain = sum(probe.gain_of_fix(q["truth_pos"]) for pool in state["pools"] if pool["prefix"] == "pub_" for q in pool["questions"]
               if isinstance(q.get("truth_pos"), int) and q["truth_pos"] >= 2)
    return state["base_score"] + gain / probe.N_PUBLIC


def write_submission(lists, order, path):
    pools.write_submission([(qid, lists[qid]) for qid in order], path)


def cmd_state(a):
    doc = json.load(open(a.fusion))
    state = {"fusion": a.fusion, "components": doc["components"], "weights": doc["weights"], "base_score": a.base_score, "pools": [], "uploads": [], "groups": []}
    for cand_path, q_path in zip(a.candidates, a.questions):
        prefix = "pub_" if "public" in Path(q_path).name else "priv_"
        qs = pool_lists(cand_path, q_path, doc, a.workers)
        for q in qs:
            q["truth_pos"] = None
            q["tested_in"] = []
        state["pools"].append({"prefix": prefix, "candidates": str(cand_path), "questions_file": str(q_path), "questions": qs})
    state.update(pid_environment("cpu"))
    save_state(state)
    pub = [p for p in state["pools"] if p["prefix"] == "pub_"][0]["questions"]
    margins = sorted(q["margin"] for q in pub)
    print(f"public: {len(pub)} questions; assigned-Top-1 margin quantiles 0/5/10/25/50%: {[round(margins[int(f * (len(margins) - 1))], 2) for f in (0, .05, .1, .25, .5)]}; "
          f"{sum(m < 0 for m in margins)} questions where the assignment overrode the argmax")
    combos = probe.deficit_decompositions(a.base_score)
    state["bounds"] = list(probe.bounds_from_score(a.base_score))
    save_state(state)
    print(f"Public loss {probe.N_PUBLIC * (1 - a.base_score):.4f} points; decompositions (truth at rank 2, 3, 4, 5, outside): {combos}; per-group bounds {state['bounds']}")
    print(f"wrote {STATE}")
    return 0


def unresolved_pub(state):
    pub = [p for p in state["pools"] if p["prefix"] == "pub_"][0]["questions"]
    return pub, [q for q in pub if q.get("truth_pos") is None]


def apply_resolution(state):
    """Run probe.resolve over the decoded groups, record derived groups and the positions they pin down."""
    positions, derived = probe.resolve(state["groups"])
    for g in derived:
        if not any(sorted(x["members"]) == sorted(g["members"]) for x in state["groups"]):
            state["groups"].append(g)
    pub, _ = unresolved_pub(state)
    by_id = {q["question_id"]: q for q in pub}
    newly = []
    for qid, t in positions.items():
        q = by_id[qid]
        if q.get("truth_pos") is None:
            q["truth_pos"] = "out" if t == probe.OUT else int(t)
            newly.append((qid, q["truth_pos"]))
    return newly


def next_group(state, size):
    """The group to test next: a pending decoded group with wrong members gets its lower-margin half; otherwise the
    `size` least-confident unresolved questions."""
    pub, todo = unresolved_pub(state)
    margin = {q["question_id"]: q["margin"] for q in pub}
    pending = []
    for g in state["groups"]:
        if g.get("hist") is None:
            continue
        members = [m for m in g["members"] if m in {q["question_id"] for q in todo}]
        wrong = len(g["members"]) - g["hist"][0]
        if wrong > 0 and len(members) > 1 and set(members) == set(g["members"]):
            pending.append((wrong, members))
    if pending:
        wrong, members = max(pending, key=lambda x: (x[0], len(x[1])))
        members = sorted(members, key=lambda m: margin[m])
        return members[: max(1, len(members) // 2)], f"half of a pending group with {wrong} wrong among {len(members)}"
    chosen = sorted(todo, key=lambda q: q["margin"])[:size]
    return [q["question_id"] for q in chosen], f"the {len(chosen)} least-confident unresolved questions"


def cmd_design(a):
    state = load_state()
    state.setdefault("groups", [])
    bounds = tuple(state.get("bounds") or probe.bounds_from_score(state["base_score"]))
    cap = probe.max_group_size(a.tol, bounds)
    size = a.size or min(a.default_size, cap)
    if a.members:
        members, why = a.members, "given"
    else:
        members, why = next_group(state, size)
    if len(members) > cap:
        raise ValueError(f"group of {len(members)} exceeds the decodable size {cap}")
    lists = current_lists(state)
    ref = predicted_score(state)
    for m in members:
        lists[m] = probe.reorder(lists[m], probe.CYCLE)
    order = [q["question_id"] for pool in state["pools"] for q in pool["questions"]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(lists, order, out)
    pub, _ = unresolved_pub(state)
    margin = {q["question_id"]: q["margin"] for q in pub}
    plan = {"k": a.k, "reference_score": ref, "members": members, "why": why, "tol": a.tol, "bounds": list(bounds), "file": str(out), "margins": [margin[m] for m in members]}
    (STATE.parent / f"plan_{a.k}.json").write_text(to_json(plan, indent=1) + "\n")
    state["groups"].append({"k": a.k, "members": members, "hist": None})
    state.setdefault("uploads", []).append({"k": a.k, "file": str(out), "reference_score": ref, "n_tested": len(members)})
    save_state(state)
    print(f"upload {a.k}: {len(members)} questions cycled ({why}); margins {[round(margin[m], 2) for m in members]}")
    print(f"reference score {ref:.7f} = the current best file (fixes applied) before the reorders; decodable sizes up to {cap} under the bounds {bounds} (truths at rank 2, 3, 4, 5, outside)")
    print(f"wrote {out} and {STATE.parent / f'plan_{a.k}.json'}")
    return 0


def cmd_decode(a):
    state = load_state()
    plan = json.load(open(STATE.parent / f"plan_{a.k}.json"))
    members = plan["members"]
    delta = (a.score - plan["reference_score"]) * probe.N_PUBLIC
    matches = probe.decode_histogram(delta, len(members), tol=plan["tol"], bounds=tuple(plan["bounds"]))
    print(f"upload {a.k}: observed {a.score:.7f}, reference {plan['reference_score']:.7f}, delta {delta:+.5f} points -> {len(matches)} matching histogram(s)")
    if len(matches) != 1:
        print("no unique decode; nothing recorded", file=sys.stderr)
        return 1
    hist = matches[0]
    print(f"  histogram over the {len(members)} tested questions (truth at rank 1, 2, 3, 4, 5, outside): {hist}; wrong {len(members) - hist[0]}")
    for g in state["groups"]:
        if g.get("k") == a.k:
            g["hist"] = list(hist)
    for u in state["uploads"]:
        if u["k"] == a.k:
            u["observed_score"] = a.score
            u["hist"] = list(hist)
    newly = apply_resolution(state)
    save_state(state)
    for qid, t in newly:
        print(f"  {qid}: " + ("right already" if t == 1 else "truth outside the five (needs a candidate search)" if t == "out" else f"truth at rank {t} -> fixed in the next file"))
    pub, todo = unresolved_pub(state)
    print(f"{len(pub) - len(todo)}/400 Public questions resolved; predicted score of the current best file (fixes applied, no tests): {predicted_score(state):.7f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("state")
    s.add_argument("--candidates", nargs="+", required=True)
    s.add_argument("--questions", nargs="+", required=True)
    s.add_argument("--fusion", required=True)
    s.add_argument("--base-score", type=float, required=True, help="observed Public score of the assigned submission of this fusion document")
    s.add_argument("--workers", type=int, default=1)
    d = sub.add_parser("design")
    d.add_argument("--k", type=int, required=True)
    d.add_argument("--size", type=int, default=0, help="group size for a fresh group (default: the largest decodable)")
    d.add_argument("--members", nargs="*", default=None, help="explicit question ids to cycle (otherwise chosen automatically)")
    d.add_argument("--tol", type=float, default=probe.TOL)
    d.add_argument("--default-size", type=int, default=20, help="size of a fresh group when --size is not given (capped by the decodable size)")
    d.add_argument("--out", required=True)
    c = sub.add_parser("decode")
    c.add_argument("--k", type=int, required=True)
    c.add_argument("--score", type=float, required=True)
    a = ap.parse_args(argv)
    return {"state": cmd_state, "design": cmd_design, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
