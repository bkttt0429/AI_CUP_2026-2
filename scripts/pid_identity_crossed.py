"""Separate query randomness from enrollment randomness on physically disjoint seven-game panels."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, PLAN, PRIOR, ROOT, RUN, atomic_json, atomic_npz, digest_file, guard, read_manifest
from pid_tactical_duration import FOLDS_CSV, TRAIN_DIR, hashed, physical
from aicup_pid.identity_variance import bootstrap_matrix, crossed_endpoints, matched_pairs, unit

OUT = ROOT / "results/final/identity_crossed_20261118.json"


def select(m, path):
    historic = list(m["games"])
    for f in (RUN / "refpolicy/manifest.json", RUN / "local_choice_20261118/source_verified_data.json"):
        historic.extend(json.loads(f.read_text())["games"])
    blocked_ids = {r["owner"] for r in historic}
    blocked_games = {physical(r["sgf"]) for r in historic}
    earlier = PRIOR / "duration_decisions.npz"
    with np.load(earlier, allow_pickle=False) as z:
        blocked_ids.update(z["roster"])
        blocked_games.update(z["signatures"].ravel())
    with np.load(path, allow_pickle=False) as z:
        ids, keys, vectors = z["pid"], z["key"], z["emb"]
    lookup = dict(zip(keys, range(len(keys))))
    if len(lookup) != len(keys) or not np.isfinite(vectors).all():
        raise ValueError("inconsistent embedding store")
    with open(FOLDS_CSV) as f:
        folds = {r["player_id"]: int(r["fold"]) for r in csv.DictReader(f)}
    if any(folds[p] != 2 for p in ids):
        raise ValueError("encoder and owner fold do not match")
    per_owner = defaultdict(list)
    csvs = sorted(TRAIN_DIR.glob("train_*.csv"))
    for path_csv in csvs:
        with open(path_csv) as f:
            for row in csv.DictReader(f):
                p = row["player_id"]
                if p in blocked_ids:
                    continue
                k = row["rank"] + ":" + row["game_id"]
                if k in lookup:
                    if ids[lookup[k]] != p:
                        raise ValueError("owner mismatch")
                    per_owner[p].append((hashed("crossed-game-v1|" + k), k, row["sgf_content"]))
    roster, selected, signatures, seen = [], [], [], set()
    for p in sorted(per_owner, key=lambda v: hashed("crossed-owner-v1|" + v)):
        if len(per_owner[p]) < 24:
            continue
        rows, sigs = [], set()
        for _, k, sgf in sorted(per_owner[p]):
            sig = physical(sgf)
            if sig in blocked_games or sig in seen or sig in sigs:
                continue
            rows.append((k, sig))
            sigs.add(sig)
            if len(rows) == 24:
                break
        if len(rows) == 24:
            roster.append(p)
            selected.append([k for k, _ in rows])
            signatures.extend(s for _, s in rows)
            seen.update(sigs)
        if len(roster) == 600:
            break
    if len(roster) < 250 or len(seen) != 24 * len(roster) or blocked_ids.intersection(roster) or seen.intersection(blocked_games):
        raise ValueError("insufficient or overlapping independent players/games")
    v = np.asarray([[vectors[lookup[k]] for k in ks] for ks in selected], dtype=float)
    return np.array(roster), np.array(selected), np.array(signatures).reshape(len(roster), 24), unit(v), {
        "forbidden_histories": len(blocked_games), "known_excluded_ids": len(blocked_ids),
        "overlap_histories": len(seen.intersection(blocked_games)), "known_id_overlap": len(blocked_ids.intersection(roster)),
        "embedding_path": str(path), "embedding_sha": digest_file(path), "fold_sha": digest_file(FOLDS_CSV),
        "prior_duration_sha": digest_file(earlier), "training_files": {f.name: digest_file(f) for f in csvs}}


def design_pairs(v):
    means = unit(v[:, :7].mean(1))
    sim = means @ means.T
    np.fill_diagonal(sim, -np.inf)
    near = np.argsort(-sim, axis=1, kind="stable")[:, :5]
    edges = sorted(((-sim[a, b], a, int(b)) for a in range(len(v)) for b in near[a] if a < b and a in near[b]))
    return np.array(matched_pairs([(a, b) for _, a, b in edges]))


def evaluate(v, pairs):
    means = np.stack([v[:, 8:15].mean(1), v[:, 15:22].mean(1)], axis=1)
    refs = unit(means)
    queries = v[:, 22:24]
    margins, metrics = [], []
    for a, b in pairs:
        diff = refs[a] - refs[b]
        q = queries[[a, b]]
        sign = np.array([1., -1.])
        # Both subjects must use the same two reference panels and see their own two queries.
        gap = (q @ diff.T).transpose(0, 2, 1) * sign[:, None, None]
        margins.append(gap)
        rec = crossed_endpoints(gap)
        ref_double = unit(means[a].mean(0)) - unit(means[b].mean(0))
        qr = (q @ ref_double) * sign[:, None]
        qq = (unit(q.mean(1)) @ diff.T) * sign[:, None]
        accuracy = lambda x: float(np.mean((x > 0) + .5 * (x == 0)))
        rec["gain_double_refs"] = accuracy(qr) - rec["accuracy"]
        rec["gain_double_queries"] = accuracy(qq) - rec["accuracy"]
        rec["gain_queries_minus_refs"] = accuracy(qq) - accuracy(qr)
        rec["query_minus_ref_flip"] = rec["query_flip"] - rec["reference_flip"]
        da, db = means[a] - means[b]
        rec["cross_split_separation"] = float(da @ db)
        rec["same_split_squared_distance"] = float((da @ da + db @ db) / 2.)
        rec["cross_split_nonpositive"] = float(da @ db <= 0.)
        metrics.append(rec)
    return np.array(margins), metrics


def main():
    guard()
    CACHE.mkdir(parents=True, exist_ok=True)
    m = read_manifest(RUN, verify=False)
    path = Path(m["source_root"]) / "data/seq/fold2.npz"
    owners, keys, sig, v, audit = select(m, path)
    pairs = design_pairs(v)
    if len(pairs) < 50 or len(set(pairs.ravel())) != pairs.size:
        raise ValueError("too few independent hard pairs")
    gaps, records = evaluate(v, pairs)
    names = list(records[0])
    values = np.array([[r[k] for k in names] for r in records])
    stats = dict(zip(names, bootstrap_matrix(values, np.zeros(len(values)))))
    archive = CACHE / "crossed.npz"
    atomic_npz(archive, owners=owners, keys=keys, signatures=sig, pairs=pairs, margins=gaps, metrics=values, names=np.array(names))
    atomic_json(OUT, {"plan": digest_file(PLAN), "code": digest_file(__file__), "audit": audit, "archive": digest_file(archive),
                       "owners": len(owners), "games": keys.size, "pairs": len(pairs), "stats": stats})
    print("crossed experiment finished", len(owners), len(pairs), flush=True)


if __name__ == "__main__":
    main()
