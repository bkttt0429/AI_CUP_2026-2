"""Count true-owner recoveries from two games when their individual winners disagree."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_long_training_inputs import ARMS, CACHE, POOLS, ROOT, atomic_json, digest_file, guard
from pid_long_training_report import OwnerBootstrap

PLAN = ROOT / "docs/superpowers/plans/2026-11-19-two-game-emergence.md"
OUT = ROOT / "results/final/two_game_recovery_20261119.json"


def matched_difference(ids, seed, single_ranks, same, joint, wrong, boot):
    bands = np.sort(np.where(single_ranks < 2, 0, np.where(single_ranks < 5, 1, 2)), axis=1)
    stratum = (ids//600)*1000 + seed*100 + bands[:, 0]*10 + bands[:, 1]
    counts = []
    for cell in np.unique(stratum[wrong]):
        s = wrong & same & (stratum == cell)
        d = wrong & ~same & (stratum == cell)
        if s.any() and d.any():
            counts.append([np.bincount(ids[mask], weights=values[mask], minlength=1200)
                           for mask in (s, d) for values in (np.ones(len(ids)), joint)])
    table = np.array(counts)
    def estimate(weights):
        ns, hs, nd, hd = [weights @ table[:, j].T for j in range(4)]
        valid = ((ns == 0) | (nd > 0)).all(1) & (ns.sum(1) > 0)
        same_acc = hs[valid].sum(1)/ns[valid].sum(1)
        different_acc = (ns[valid]*hd[valid]/np.maximum(nd[valid], 1)).sum(1)/ns[valid].sum(1)
        return np.column_stack([same_acc, different_acc, different_acc-same_acc])
    means = estimate(np.ones((1, 1200)))[0]
    sampled = estimate(boot.weights)
    ci = np.quantile(sampled,[.025,.975],axis=0).T
    return {"included_same": int(table[:, 0].sum()), "included_different": int(table[:, 2].sum()),
            "eligible_same": int((same & wrong).sum()), "eligible_different": int((~same & wrong).sum()),
            "cells": len(table), "bootstrap_draws": len(sampled),
            **{k: {"mean": float(mu), "ci95": v.tolist()} for k, mu, v in zip(("same", "different", "difference"), means, ci)}}


def summarize(records, boot):
    ids = np.concatenate([r["owners"] for r in records])
    seed = np.concatenate([r["seed"] for r in records])
    ra = np.concatenate([r["ranks"] for r in records])
    lead = np.concatenate([r["leaders"] for r in records])
    if ra.shape != (len(ids), 5, len(ARMS)) or lead.shape != (len(ids), 3, len(ARMS)):
        raise ValueError("invalid within-owner observation ordering")
    r = ra[:, :2, 0]
    hit = r == 0
    count = hit.sum(1)
    w = count == 0
    joint = ra[:, 3, 0] == 0
    same = lead[:, 0, 0] == lead[:, 1, 0]
    increment = joint-count/2
    j = ARMS.index("add")
    late_hit = ra[:, :2, j] == 0
    vals = {"single_random_top1": count/2, "joint_top1": joint, "joint_minus_random_single": increment,
            "both_wrong_contribution": increment*w, "one_right_contribution": increment*(count == 1),
            "both_right_contribution": increment*(count == 2),
            "add_joint_minus_random_single": (ra[:, 3, j] == 0)-late_hit.mean(1)}
    np.testing.assert_allclose(increment, sum(vals[k] for k in ("both_wrong_contribution", "one_right_contribution", "both_right_contribution")))
    result = {"pairs": len(ids), "owners": len(set(ids)), "population": dict(zip(vals, boot.means(ids, np.column_stack(list(vals.values())))))}
    for label, use in (("both_wrong", w), ("same_wrong", w & same), ("different_wrong", w & ~same),
                       ("one_right", count == 1), ("both_right", count == 2), ("truth_below_second_twice", (r >= 2).all(1))):
        result[label] = {"pairs": int(use.sum()), "joint_correct": int(joint[use].sum()),
                         "constant_in_sample": bool(np.all(joint[use] == joint[use][0])),
                         **boot.means(ids[use], joint[use])[0]}
    result["long_training_fixes_any_single_among_both_wrong"] = {"pairs": int(w.sum()), "correct": int(late_hit[w].any(1).sum()),
                                                                  **boot.means(ids[w], late_hit[w].any(1))[0]}
    result["matched_rival_difference"] = matched_difference(ids, seed, r, same, joint, w, boot)
    return result


def main():
    guard()
    sets, fingerprints = {}, {}
    for pidx, p in enumerate(POOLS):
        metadata = json.loads((CACHE / f"inputs_{p}.json").read_text())
        for seed in range(6):
            path = CACHE / f"{p}_s{seed}.npz"
            fingerprints[f"{p}/{seed}"] = digest_file(path)
            if fingerprints[f"{p}/{seed}"] != metadata["splits"][str(seed)]["records"]:
                raise ValueError("decision archive changed")
            with np.load(path) as z:
                np.testing.assert_array_equal(z["sizes"], np.tile([1, 1, 1, 2, 3], 600))
                np.testing.assert_array_equal(z["truth"], np.repeat(np.arange(600), 5))
                sets[p,seed] = {"ranks": z["ranks"].reshape(600,5,len(ARMS)), "leaders": z["leaders"].reshape(600,3,len(ARMS)),
                                "owners": np.arange(600)+600*pidx, "seed": np.full(600,seed)}
    boot = OwnerBootstrap()
    out = {k: summarize([sets[p,s] for s in seeds for p in POOLS], boot)
           for k, seeds in (("0_3", range(4)), ("4_5", (4,5)))}
    atomic_json(OUT, {**out, "inputs": fingerprints, "plan": digest_file(PLAN), "interval_note":
                     "Empirical owner-cluster resampling gives a point interval for a constant observed outcome. Such bounds do not quantify the unobserved event rate."})
    print("fresh conditional two-game recovery", out["4_5"]["both_wrong"], flush=True)


if __name__ == "__main__":
    main()
