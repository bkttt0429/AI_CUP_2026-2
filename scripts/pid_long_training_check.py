"""Independent candidate-vector and decision-count verification of longer training."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_long_training_inputs import ARMS, CACHE, DOCUMENTS, POOLS, ROOT, RUN, atomic_json, digest_file, eligible_sources, guard, read_manifest, seed_split
from pid_tactical_score import competition

OUT = ROOT / "results/final/long_training_check_20261119.json"


def center(matrix):
    return (matrix-matrix.mean(1, keepdims=True)) / matrix.std(1, keepdims=True)


def literal_checks(source, manifest, pool):
    """Read original files and physically average the checkpoint lists on sampled queries."""
    seed = 4
    split = seed_split(manifest, pool, seed)
    owners = sorted(split["reference"])
    refs = np.array([[k.split("|", 1)[1] for k in split["reference"][o]] for o in owners])
    queries = []
    for o in owners:
        h = [k.split("|",1)[1] for k in split["heldout"][o]]
        queries.extend([[h[0]], [h[1]], [h[2]], h[:2], h])
    with np.load(CACHE / f"{pool}_s{seed}.npz") as z:
        ix, fixed, expected = z["audit"], z["audit_fixed"], z["audit_scores"]
    docs = {arm: json.loads((source / "results" / f"pid_fusion_{DOCUMENTS[arm]}.json").read_text()) for arm in ("baseline", "replace", "add")}
    members = {arm: [Path(pair[0]).name for pair in doc["seq_official"]] for arm, doc in docs.items()}
    matrices = {}
    for name in set(members["baseline"] + members["add"]):
        fname = name.replace("official_pub2_", f"offsim_{pool}_h3s{seed}_")
        with np.load(source / "data/pid_v2_f0tmp/ft" / fname) as z:
            key, emb = z["key"].astype(str), z["emb"].astype(np.float32)
            lookup = {k: i for i, k in enumerate(key)}
        panels = np.array([[lookup[k] for k in group] for group in refs])
        c = emb[panels].sum(1).astype(float)
        q = np.stack([emb[[lookup[k] for k in queries[i]]].sum(0) for i in ix]).astype(float)
        co = (q/np.linalg.norm(q, axis=1, keepdims=True)) @ (c/np.linalg.norm(c, axis=1, keepdims=True)).T
        matrices[name] = center(co)
    # Work in the baseline's actual multiplicities to compare synthetic arms independently.
    old = members["baseline"]
    e = [k for k in old if "_o44F" in k or "_o47F" in k]
    l = [k.replace("_o44F", "_o58F").replace("_o47F", "_o57F") for k in e]
    # Two copies of the original list, with one early copy moved to its late checkpoint.
    mixed = list((Counter(old+old)-Counter(e)+Counter(l)).elements())
    members.update(weight=old+e, mix=mixed)
    worst = 0.
    for arm, want in zip(ARMS, expected):
        mm = members[arm]
        z = np.mean([matrices[k] for k in mm], axis=0)
        multiplier = docs["baseline"]["weights"][docs["baseline"]["components"].index("seq")]
        got = fixed + multiplier*center(z)
        worst = max(worst, float(np.abs(got-want).max()))
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)
    return {"queries": len(ix), "candidates": len(owners), "arms": len(ARMS), "maximum_score_error": worst}


def main():
    guard()
    manifest = read_manifest(RUN, verify=False)
    source = Path(manifest["source_root"])
    final = ROOT / "results/final"
    ranking = json.loads((final / "long_training_attribution_20261119.json").read_text())
    joint = json.loads((final / "two_game_recovery_20261119.json").read_text())
    details = {p: literal_checks(source, manifest, p) for p in POOLS}
    fresh, score_rows, all_pairs = set(), 0, 0
    for group, seeds in (("0_3", range(4)), ("4_5", (4,5))):
        count = {k: Counter() for k in ("replace", "add", "weight", "mix")}
        pair_totals = Counter()
        single_totals = []
        decomp, own, mask2 = [], [], []
        for s in seeds:
            for j, p in enumerate(POOLS):
                with np.load(CACHE / f"{p}_s{s}.npz") as z:
                    truth, sizes, scores, leaders, wins = z["truth"], z["sizes"], z["ranks"], z["leaders"], z["wins"]
                    changes = z["change"]
                    keys = z["keys"]
                score_rows += scores.size
                nr = scores[sizes == 1]
                who = truth[sizes == 1]
                old = nr[:, 0]
                np.testing.assert_array_equal(leaders == who[:,None], nr == 0)
                for k in count:
                    a = ARMS.index(k)
                    v = count[k]
                    for br, ar, beat in zip(old, nr[:,a], wins[:,a]):
                        v["rescued_any_rank"] += int(br > 0 and ar == 0)
                        v["broken_first"] += int(br == 0 and ar > 0)
                        if br == 1:
                            bucket = "from_second_to_first" if ar == 0 else ("passed_rival_third_wins" if beat else "still_behind_rival")
                            v[bucket] += 1
                fresh.update(keys.tolist())
                decomp.append(changes[:,ARMS.index("add")])
                own.extend(who+600*j)
                mask2.extend(old == 1)
                single_totals.append(competition(nr))
                for i in range(600):
                    sub = scores[5*i:5*i+5, 0]
                    if sub[0] > 0 and sub[1] > 0:
                        rival = leaders[3*i:3*i+2, 0]
                        name = "same_wrong" if rival[0] == rival[1] else "different_wrong"
                        pair_totals[name] += 1
                        pair_totals[name+"_correct"] += int(sub[3] == 0)
                    all_pairs += 1
        for k, values in count.items():
            for col, n in values.items():
                assert int(ranking[group]["transitions"][k][col]) == n
        for k in ("same_wrong", "different_wrong"):
            assert joint[group][k]["pairs"] == pair_totals[k] and joint[group][k]["joint_correct"] == pair_totals[k+"_correct"]
        u = np.vstack(single_totals)
        effect = .5*((u[:,4]-u[:,0])+(u[:,2]-u[:,3]))
        np.testing.assert_allclose(effect.mean(), ranking[group]["factorial"]["single"]["attributed_checkpoint_mix"]["mean"])
        mask = np.array(mask2)
        ids = np.asarray(own)[mask]
        diff = np.vstack(decomp)[mask]
        margin = diff[:,0]-diff[:,1]
        rng = np.random.default_rng(111930)
        weights = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS])
        expected = ranking[group]["transitions"]["add"]["on_old_second"]["change_margin"]
        resampled = (weights @ np.bincount(ids, weights=margin, minlength=1200))/(weights @ np.bincount(ids, minlength=1200))
        np.testing.assert_allclose(np.quantile(resampled, [.025,.975]), expected["ci95"])
    comparison = json.loads((final / "two_game_averaging_20261119.json").read_text())
    overlap = Counter()
    dups, dup_switches, high = 0, 0, 0.
    for seed in (4,5):
        for pool in POOLS:
            path = CACHE / f"average_{pool}_{seed}.npz"
            assert digest_file(path) == comparison["inputs"][f"{pool}/{seed}"]
            with np.load(path) as z:
                for row in z["rank"]:
                    if min(row[:2]) > 0:
                        outcome = {(True,True): "both", (False,True): "only_average", (True,False): "only_native", (False,False): "neither"}
                        overlap[outcome[row[2] == 0, row[5] == 0]] += 1
                    dups += int(row[0] != row[3])+int(row[1] != row[4])
                dup_switches += int((z["winners"][:,:2] != z["winners"][:,3:5]).sum())
                high = max(high, float(z["duplicate_error"].max()))
    assert all(overlap[k] == v for k,v in comparison["recovered"].items())
    assert sum(overlap.values()) == joint["4_5"]["both_wrong"]["pairs"]
    assert overlap["both"]+overlap["only_native"] == joint["4_5"]["both_wrong"]["joint_correct"]
    assert (dups, dup_switches, high) == (comparison["duplicate_rank_changes"], comparison["duplicate_top_changes"],
                                           comparison["duplicate_max_score_difference"])
    atomic_json(OUT, {"literal_score_checks": details, "family_gates": eligible_sources(source), "rank_values_in_verified_records": score_rows,
                     "game_pairs_in_verified_counts": all_pairs, "distinct_single_games_across_six_splits": len(fresh),
                     "report_hashes": {name: digest_file(final / name) for name in ("long_training_attribution_20261119.json","two_game_recovery_20261119.json","two_game_averaging_20261119.json")},
                     "code": {p.name: digest_file(p) for p in [*sorted((ROOT/"scripts").glob("pid_long_training_*.py")),
                                                                 ROOT/"scripts/pid_two_game_averaging.py", ROOT/"src/aicup_pid/long_training.py"]}})
    print("verified literal compositions, transitions and conditional intervals", details, flush=True)


if __name__ == "__main__":
    main()
