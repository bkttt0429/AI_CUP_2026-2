"""Paired descriptive audit of v2fin3_a versus v2fin_x on candidate-only one-game questions.

Fixed before running: both pools, holdouts 2/3. Group by colour and by reference-set
tertiles of the fraction of moves more than four squares from the preceding opponent
move (Chebyshev distance, first 256 plies, omit pass pairs). Contrast the high and
low groups with a paired owner bootstrap. Seek recurrent rivals in split 2 only and
measure them on split 3 questions that were never used to find the pair.
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_local_choice_data import DEST, LEDGER, RUN, ROOT, atomic_json, check, read_manifest, setup_and_moves, source_record, stable_hash
from pid_choice_eval import load_baseline_inputs, ranks
import numpy as np
import torch


def away_fraction(game):
    _, moves = setup_and_moves(game["sgf"])
    distances = [max(abs(x - px), abs(y - py)) for (pc, px, py), (c, x, y) in zip(moves[:255], moves[1:256])
                 if c == game["color"] and pc != c and px is not None and x is not None]
    return np.mean(np.asarray(distances) > 4) if distances else np.nan


def cluster_stats(rows):
    count = len({r["cluster"] for r in rows})
    ids = np.array([r["cluster"] for r in rows])
    rng = np.random.default_rng(1126)
    # Sample owners with replacement inside each pool, preserving the pool sizes.
    draw = np.concatenate([rng.integers(off, off + 600, (5000, 600)) for off in (0, 600)], axis=1)
    if count != 1200 or ids.max() != count - 1:
        raise ValueError("incomplete owner coverage")
    metrics = np.asarray([[r["rank0"], r["rank1"]] for r in rows])
    scores = np.where(metrics < 5, np.exp(-metrics.astype(float)), 0)
    delta = scores[:, 1] - scores[:, 0]
    samples, result = {}, {}
    groups = {"all": np.ones(len(rows), bool)}
    groups.update({c: np.array([r["colour"] == c for r in rows]) for c in ("B", "W")})
    groups.update({g: np.array([r["away_bin"] == g for r in rows]) for g in ("low", "mid", "high")})
    recurrent = np.asarray([r["selected_rival"] >= 0 for r in rows])
    if recurrent.any():
        groups["distinct_queries_of_recurrently_confused_owners"] = recurrent
    for group, mask in groups.items():
        num = np.bincount(ids[mask], delta[mask], minlength=count)
        den = np.bincount(ids[mask], minlength=count)
        bs_den = den[draw].sum(1)
        ok = bs_den > 0
        samples[group] = num[draw[ok]].sum(1) / bs_den[ok]
        a, b = metrics[mask].T
        result[group] = {"rows": int(mask.sum()), "unique_games": len({r["key"] for r, keep in zip(rows, mask) if keep}),
                         "owners": int((den > 0).sum()), "base_rank_score": float(scores[mask, 0].mean()),
                         "alt_rank_score": float(scores[mask, 1].mean()), "delta": float(delta[mask].mean()),
                         "ci95": np.quantile(samples[group], [.025, .975]).tolist(),
                         "rescued": int(((a > 0) & (b == 0)).sum()), "broken": int(((a == 0) & (b > 0)).sum()),
                         "base_rank_2_to_5": int(((a > 0) & (a < 5)).sum()),
                         "rescued_from_rank_2_to_5": int(((a > 0) & (a < 5) & (b == 0)).sum())}
    differences = {}
    for a, b in (("high", "low"), ("W", "B")):
        differences[f"{a} minus {b}"] = {"delta": result[a]["delta"] - result[b]["delta"],
                                        "ci95": np.quantile(samples[a] - samples[b], [.025, .975]).tolist()}
    if recurrent.any():
        rr = [r for r in rows if r["selected_rival"] >= 0]
        differences["distinct_queries_against_seed2_rival"] = {
            "rows": len(rr), "base_top1_is_previous_rival": sum(r["top0"] == r["selected_rival"] for r in rr),
            "alt_top1_is_previous_rival": sum(r["top1"] == r["selected_rival"] for r in rr),
            "base_pair_wins": sum(r["winner0"] for r in rr), "alt_pair_wins": sum(r["winner1"] for r in rr)}
    return result, differences


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("use production Slurm")
    torch.set_num_threads(3)
    check(RUN, LEDGER)
    parent = read_manifest(RUN, verify=False)
    src = Path(parent["source_root"])
    names = ("v2fin_x", "v2fin3_a")
    files = [src / f"results/pid_fusion_{n}.json" for n in names]
    docs = [json.loads(p.read_text()) for p in files]
    if any(docs[0][k] != docs[1][k] for k in ("components", "weights", "cfg", "trunk_store", "emb_model")) or docs[0]["cfg"].get("seq_near"):
        raise ValueError("documents do not match the declared fixed non-sequence components")
    expected_file = src / "results/final/v2_offsim_v2fin3_a_vs_v2fin_x_s2_3.json"
    expected = json.loads(expected_file.read_text())["scores"]
    inputs = [source_record(p) for p in [*files, expected_file, Path(__file__), ROOT / "scripts/pid_choice_eval.py"]]
    rows, thresholds, vector_ids, baseline_checks = [], {}, {}, {}
    for pi, pool in enumerate(("pub2", "priv2")):
        games = {g["key"]: g for g in parent["games"] if g["pool"] == pool}
        fractions = {k: away_fraction(g) for k, g in games.items()}
        known_rivals, seen_games = {}, set()
        for seed in (2, 3):
            arms = []
            for name, path in zip(names, files):
                view = {**parent, "assets": {**parent["assets"], "fusion": {"path": str(path)}}}
                data = load_baseline_inputs(view, pool, seed)
                vector_ids[f"{name}/{pool}/{seed}"] = stable_hash(json.dumps(data.pop("sources"), sort_keys=True))
                arms.append(data)
                print(f"scored {name} {pool} split {seed}", flush=True)
            a, b = arms
            if any(a[k] != b[k] for k in ("questions", "owners", "references")) or not np.array_equal(a["truth"], b["truth"]):
                raise ValueError("unpaired candidate splits")
            one = a["sizes"] == 1
            ids = a["truth"][one]
            fa, fb = [d["baseline"][one] for d in arms]
            ranks_a, ranks_b = ranks(fa, ids), ranks(fb, ids)
            top_a, top_b = fa.argmax(1), fb.argmax(1)
            keys = [q[0] for q, flag in zip(a["questions"], one) if flag]
            support = [k for ks in a["references"].values() for k in ks]
            lo, hi = np.nanquantile([fractions[k] for k in support], [1 / 3, 2 / 3])
            thresholds[f"{pool}/{seed}"] = [lo, hi]
            for j, (owner, key) in enumerate(zip(ids, keys)):
                r = known_rivals.get(int(owner), -1) if seed == 3 and key not in seen_games else -1
                f = fractions[key]
                group = "unknown" if not np.isfinite(f) else "low" if f <= lo else "high" if f > hi else "mid"
                rows.append({"cluster": int(owner + 600 * pi), "pool": pool, "seed": seed, "key": key,
                             "rank0": int(ranks_a[j]), "rank1": int(ranks_b[j]), "top0": int(top_a[j]), "top1": int(top_b[j]),
                             "colour": games[key]["color"], "away_bin": group, "selected_rival": r,
                             "winner0": int(r >= 0 and fa[j, owner] > fa[j, r]), "winner1": int(r >= 0 and fb[j, owner] > fb[j, r])})
            if seed == 2:
                seen_games = set(keys)
                mistakes = defaultdict(Counter)
                for owner, pred in zip(ids, top_a):
                    if owner != pred:
                        mistakes[int(owner)][int(pred)] += 1
                known_rivals = {owner: best[0] for owner, counts in mistakes.items()
                                if (best := counts.most_common(1)[0])[1] >= 2}
        for rank_field, name, doc_key in (("rank0", names[0], "50"), ("rank1", names[1], "alt")):
            rb = np.asarray([r[rank_field] for r in rows if r["pool"] == pool])
            got = np.where(rb < 5, np.exp(-rb.astype(float)), 0).mean()
            want = expected[pool][doc_key]["1"]
            if abs(got - want) > 1e-6:
                raise ValueError(f"{pool} cannot reproduce {name}: {got} expected {want}")
            baseline_checks[f"{pool}/{name}"] = {"computed": float(got), "published": want}
    groups, contrasts = cluster_stats(rows)
    result = {"job": os.environ["SLURM_JOB_ID"], "reference": names[0], "alternative": names[1],
              "seeds": [2, 3], "thresholds": thresholds, "baseline_checks": baseline_checks,
              "candidate_sources_sha256": vector_ids, "sources": inputs, "groups": groups, "contrasts": contrasts,
              "scope": "One-game candidate holdouts, not official questions. Adding four families changes ensemble composition. Descriptive groups."}
    atomic_json(DEST / "window_rescue_rows.json", rows)
    atomic_json(DEST / "window_rescues.json", result)
    print(json.dumps(groups["all"]), flush=True)
    check(RUN, LEDGER)


if __name__ == "__main__":
    main()
