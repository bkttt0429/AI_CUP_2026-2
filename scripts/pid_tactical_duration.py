"""Independent fold-4 replication using known players and never-mixed physical games."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from katago.game.board import Board

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE, ROOT, RUN, atomic_json, atomic_npz, digest_file, guard, read_manifest
from aicup_pid.paths import FOLDS_CSV, TRAIN_DIR
from aicup.sgf import setup_and_moves
from aicup_pid.tactical_replies import opportunity, play_checked
from aicup_pid.tactical_duration import advance_streak, duration_probabilities, duration_tables, log_scores

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-nearby-replication.md"
REPORT = ROOT / "results/final/tactical_duration_20261118.json"


def hashed(s):
    return hashlib.sha256(s.encode()).hexdigest()


def physical(sgf):
    stones, moves = setup_and_moves(sgf)
    return hashed(repr((sorted(stones), moves)))


def select(parent, vector_file):
    manifests = [RUN / "manifest.json", RUN / "refpolicy/manifest.json",
                 RUN / "local_choice_20261118/source_verified_data.json"]
    historic = parent["games"] + sum([json.loads(p.read_text())["games"] for p in manifests[1:]], [])
    excluded = {g["owner"] for g in historic}
    forbidden = {physical(g["sgf"]) for g in historic}
    with np.load(vector_file, allow_pickle=False) as z:
        keys, pids, vecs = z["key"].astype(str), z["pid"].astype(str), z["emb"].astype(np.float64)
    key_index = dict(zip(keys, range(len(keys))))
    if len(key_index) != len(keys) or not np.isfinite(vecs).all():
        raise ValueError("invalid fold-4 vector file")
    with open(FOLDS_CSV) as f:
        table = {r["player_id"]: r for r in csv.DictReader(f)}
    if any(int(table[p]["fold"]) != 4 for p in set(pids)):
        raise ValueError("wrong vector fold")
    groups = defaultdict(list)
    sources = [*manifests, FOLDS_CSV, vector_file]
    for path in sorted(TRAIN_DIR.glob("train_*.csv")):
        sources.append(path)
        with open(path) as file:
            for r in csv.DictReader(file):
                owner, key = r["player_id"], r["rank"] + ":" + r["game_id"]
                if owner in excluded or key not in key_index:
                    continue
                if pids[key_index[key]] != owner or table[owner]["rank"] != r["rank"]:
                    raise ValueError("fold-4 source labels do not match")
                groups[owner].append((hashed("duration-game-v1|" + key), key, r["color"], r["sgf_content"]))
    roster, games, mask = [], [], set()
    for p in sorted(groups, key=lambda s: hashed("duration-owner-v1|" + s)):
        if len(groups[p]) < 10:
            continue
        chosen, seen = [], set()
        for _, key, colour, sgf in sorted(groups[p]):
            sig = physical(sgf)
            if sig in forbidden or sig in seen or sig in mask:
                continue
            chosen.append((key, colour, sgf, sig))
            seen.add(sig)
            if len(chosen) == 10:
                break
        if len(chosen) < 10:
            continue
        roster.append(p)
        games.extend(chosen)
        mask.update(seen)
        if len(roster) == 600:
            break
    if len(roster) < 300 or len(mask) != 10 * len(roster) or excluded.intersection(roster) or forbidden.intersection(mask):
        raise ValueError("insufficient or overlapping independent roster")
    v = vecs[[key_index[g[0]] for g in games]].reshape(len(roster), 10, -1)
    return games, np.array(roster), np.array([table[p]["rank"] for p in roster]), v, {
        "inputs": {str(path): digest_file(path) for path in sources}, "excluded_known_ids": len(excluded),
        "forbidden_histories": len(forbidden), "overlap_histories": len(forbidden & mask),
        "overlap_known_ids": len(excluded.intersection(roster))}


def extract(game):
    _, colour, sgf, sig = game
    stones, moves = setup_and_moves(sgf)
    b = Board(19)
    for x, y in stones:
        b.set_stone(Board.BLACK, b.loc(x, y))
    if stones:
        b.set_pla(Board.WHITE)
    previous, prev_xy, run = Board.PASS_LOC, None, 0
    contexts, streaks, outcomes, phases = [], [], [], []
    for t, (own, x, y) in enumerate(moves[:256]):
        who = Board.BLACK if own == "B" else Board.WHITE
        xy = None if x is None else (x, y)
        loc = Board.PASS_LOC if xy is None else b.loc(x, y)
        after = advance_streak(prev_xy, xy, run)
        if t >= 12 and own == colour and previous != Board.PASS_LOC:
            contexts.append(opportunity(b, who, previous))
            streaks.append(run)
            outcomes.append(after > 0)
            phases.append(int(t >= 80) + int(t >= 160))
        play_checked(b, who, loc)
        previous, prev_xy, run = loc, xy, after
    return duration_tables(contexts, streaks, outcomes, phases, int(hashed("duration-perm-v1|" + sig)[:16], 16))


def compare(refs, totals, query, a, b):
    scores = []
    for ref, total in zip(refs, totals):
        pop = total - ref[a] - ref[b]
        left = log_scores(query, duration_probabilities(ref[a], pop))
        right = log_scores(query, duration_probabilities(ref[b], pop))
        scores.append(left - right)
    return np.array([*scores[0], scores[1][1], scores[2][2]])


def summarize(rows, mask, truth, boot):
    vals = rows[mask]
    ids = truth[mask]
    if not len(ids):
        raise ValueError("empty comparison")
    den = boot @ np.bincount(ids, minlength=boot.shape[1])
    def metric(x):
        sums = np.bincount(ids, weights=x, minlength=boot.shape[1])
        return {"mean": float(x.mean()), "ci95": np.quantile((boot @ sums) / den, [.025, .975]).tolist()}
    names = ("marginal", "context", "duration", "shuffled_context", "shuffled_duration")
    out = {name: {"signed_nats_per_response": metric(x), "pair_accuracy": metric((x > 0) + .5 * (x == 0))}
           for name, x in zip(names, vals.T)}
    for name, i, j in (("context_minus_marginal", 1, 0), ("context_minus_shuffle", 1, 3),
                       ("duration_minus_context", 2, 1), ("duration_minus_shuffle", 2, 4)):
        out[name] = metric(vals[:, i] - vals[:, j])
    return {"rows": len(ids), "owners": len(set(ids)), **out}


def main():
    budget = guard()
    m = read_manifest(RUN, verify=False)
    vector_file = Path(m["source_root"]) / "data/seq/fold4.npz"
    games, roster, rank, v, audit = select(m, vector_file)
    n = len(roster)
    with ProcessPoolExecutor(max_workers=10) as pool:
        tables = np.asarray(list(pool.map(extract, games, chunksize=32))).reshape(n, 10, 3, 5, 3, 2)
    if not np.array_equal(tables[:, :, 0].sum(3), tables[:, :, 2].sum(3)):
        raise ValueError("streak shuffle changed within-context response counts")
    refs = tables[:, :7].sum(1).transpose(1, 0, 2, 3, 4)
    totals = refs.sum(1)
    query = tables[:, 7:, 0].reshape(n * 3, 5, 3, 2)
    def unit(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("zero embedding")
        return x / norm
    similarity = unit(v[:, 7:].reshape(n * 3, -1)) @ unit(v[:, :7].mean(1)).T
    ordered = np.argsort(-similarity, axis=1, kind="stable")
    top2 = ordered[:, :2]
    truth = np.repeat(np.arange(n), 3)
    actual = np.empty((n * 3, 5))
    same = np.empty_like(actual)
    matched = np.empty(len(truth), dtype=int)
    for i, p in enumerate(truth):
        a, b = top2[i]
        actual[i] = compare(refs, totals, query[i], a, b)
        rivals = ordered[i][(rank[ordered[i]] == rank[p]) & (ordered[i] != p)]
        if not len(rivals):
            raise ValueError("incomplete rank-matched roster")
        matched[i] = rivals[0]
        same[i] = compare(refs, totals, query[i], p, matched[i])
    selected = (top2 == truth[:, None]).any(1)
    signed = actual * np.where(top2[:, 0] == truth, 1., -1.)[:, None]
    boot = np.random.default_rng(111816).multinomial(n, np.full(n, 1 / n), size=5000).astype(float)
    primary = summarize(signed, selected, truth, boot)
    secondary = summarize(same, np.ones(len(truth), bool), truth, boot)
    archive = CACHE / "duration_decisions.npz"
    atomic_npz(archive, roster=roster, rank=rank, keys=np.array([g[0] for g in games]).reshape(n, 10),
               signatures=np.array([g[3] for g in games]).reshape(n, 10), tables=tables, top2=top2,
               truth=truth, matched=matched, actual=actual, same=same)
    report = {"plan": digest_file(PLAN), "after": digest_file(ROOT / "results/final/tactical_replies_20261118.json"),
              "code": {p.name: digest_file(p) for p in (Path(__file__), ROOT / "src/aicup_pid/tactical_duration.py",
                                                        ROOT / "src/aicup_pid/tactical_replies.py")},
              "decisions": digest_file(archive), "audit": audit, "players": n, "games": len(games),
              "evaluation_games": n * 3, "eval_responses": int(query[..., 0].sum()), "base_top1": int((top2[:, 0] == truth).sum()),
              "eval_games_no_observations": int((query[..., 0].sum((1, 2)) == 0).sum()),
              "actual_finalists": primary, "same_rank_nearest_other": secondary, "budget_start": budget}
    report["replicated"] = all(primary[k]["ci95"][0] > 0 for k in ("context_minus_marginal", "context_minus_shuffle"))
    report["temporal_increment"] = all(primary[k]["ci95"][0] > 0 for k in ("duration_minus_context", "duration_minus_shuffle"))
    atomic_json(REPORT, report)
    print(json.dumps({"players": n, "pairs": int(selected.sum()), "replicated": report["replicated"],
                      "temporal_increment": report["temporal_increment"]}), flush=True)


if __name__ == "__main__":
    main()
