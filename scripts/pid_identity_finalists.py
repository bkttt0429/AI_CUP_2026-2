"""Test fixed confusing pairs across games and a pair-adaptive discriminant on full ensemble finalists."""
from collections import Counter
from itertools import combinations_with_replacement
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, PLAN, PRIOR, ROOT, RUN, POOLS, baseline_path, atomic_json, atomic_npz, digest_file, guard, read_manifest
from pid_choice_eval import load_vectors
from pid_tactical_score import competition
from pid_tactical_temporal_transfer import anchored_fit
from aicup_pid.tactical_replies import owner_folds, eligible_players, fit_comparator, swapped_ranks
from aicup_pid.identity_variance import bootstrap_matrix, matched_pairs, pair_margins

ARMS = ("residual", "placebo", "isotropic", "fisher")
OUT = ROOT / "results/final/identity_finalists_20261118.json"


def persistence(b, pool):
    order = np.argsort(-b["scores"], axis=1, kind="stable")
    truth = b["truth"]
    starts = np.arange(0, len(truth), 3)
    failed = starts[b["rank"][starts] == 1]
    edges = [(int(truth[i]), int(order[i, 0])) for i in failed]
    def priority(pair):
        a, z = pair
        return hashlib.sha256(f"identity-pair-v1|{pool}|{b['owners'][a]}|{b['owners'][z]}".encode()).digest()
    selected = np.asarray(matched_pairs(sorted(edges, key=priority)))
    if not len(selected):
        raise ValueError("no disjoint discovery cases")
    rows = []
    for a, z in selected:
        ia, iz = 3 * a + np.array([1, 2]), 3 * z + np.array([1, 2])
        s = b["scores"]
        def wins(x, pos, neg):
            # Same stable tie rule as the full ranking.
            return (x[:, pos] > x[:, neg]) | ((x[:, pos] == x[:, neg]) & (pos < neg))
        aa, zz = wins(s[ia], a, z), wins(s[iz], z, a)
        top = order[ia, 0]
        pooled = s[ia].mean(0, keepdims=True)
        rows.append({"new_A_beats_fixed_B": float(aa.mean()), "new_B_beats_fixed_A": float(zz.mean()),
                     "balanced_pair_accuracy": float((aa.mean() + zz.mean()) / 2),
                     "full_roster_A_top1": float((top == a).mean()), "same_wrong_B_still_top1": float((top == z).mean()),
                     "different_wrong_player_top1": float(((top != z) & (top != a)).mean()),
                     "both_new_A_games_lose_to_B": float(not aa.any()), "both_new_A_games_beat_B": float(aa.all()),
                     "two_known_A_games_beat_B": float(wins(pooled, a, z)[0])})
    return len(edges), selected, rows


@torch.no_grad()
def build_kernel(source, doc, pool, seed, original_keys, fingerprints):
    # Weighted block concatenation is exactly equivalent to this sum of Gram matrices.
    members = Counter(Path(v[0] if isinstance(v, list) else v).name for v in doc["seq_official"])
    if len(members) != 128 or sum(members.values()) != 1728:
        raise ValueError("unexpected sequence committee")
    k = torch.zeros((len(original_keys), len(original_keys)), dtype=torch.float64, device="cuda")
    for name, weight in sorted(members.items()):
        match = re.fullmatch(r"official_pub2_(net-?\d+.*)_t\d+\.npz", name)
        if not match:
            raise ValueError(name)
        path = source / "data/pid_v2_f0tmp/ft" / f"offsim_{pool}_h3s{seed}_{match[1]}_t50.npz"
        if fingerprints[str(path.resolve())] != digest_file(path):
            raise ValueError("sequence embedding changed")
        vec = torch.as_tensor(load_vectors(path, original_keys), dtype=torch.float64, device="cuda")
        norms = torch.linalg.norm(vec, dim=1, keepdim=True)
        if not torch.all(norms > 1e-10):
            raise ValueError("zero game vector")
        vec = vec / norms
        k.addmm_(vec, vec.T, alpha=weight / sum(members.values()))
    if not torch.allclose(k.diagonal(), torch.ones_like(k.diagonal()), rtol=0., atol=1e-9):
        raise ValueError("pooled unit kernel mismatch")
    return k.cpu().numpy()


def donor_pair(a, b, folds):
    # A fixed symmetric map with both donor players in the corresponding source folds.
    picked = []
    for c in sorted((a, b)):
        group = np.flatnonzero(folds == folds[c])
        offset = int(np.flatnonzero(group == c)[0])
        for step in range(1, len(group)):
            candidate = int(group[(offset + step) % len(group)])
            if candidate not in (a, b) and candidate not in picked:
                picked.append(candidate)
                break
    if len(picked) != 2:
        raise ValueError("no disjoint donor references")
    return picked


def score(pool, b, kernel):
    s, truth = b["scores"], b["truth"]
    refs = np.arange(b["refs"].size).reshape(b["refs"].shape)
    qi = np.arange(len(truth)) + refs.size
    pair = np.argsort(-s, axis=1, kind="stable")[:, :2]
    folds = owner_folds(pool, b["owners"])
    first, second = pair.T
    gap = s[np.arange(len(s)), first] - s[np.arange(len(s)), second]
    features = np.empty((len(s), len(ARMS)))
    audits = {}
    for i, (a, other) in enumerate(pair):
        target = refs[[a, other]].ravel()
        noise = refs[donor_pair(a, other, folds)].ravel()
        iso, fisher = pair_margins(kernel, target, target, qi[i])
        _, sham = pair_margins(kernel, target, noise, qi[i])
        features[i] = [fisher - iso, sham - iso, iso, fisher]
        if i in (0, 123, 601, 1118, 1799):
            indices = np.r_[target, noise, qi[i]]
            audits[f"gram_{i}"] = kernel[np.ix_(indices, indices)]
    is_eval = np.arange(len(truth)) % 3 != 0
    groups = np.sort(folds[pair], axis=1)
    inside = (pair == truth[:, None]).any(1)
    sign = np.where(first == truth, 1., -1.)
    outputs = np.full_like(features, np.nan)
    fits = {}
    for fa, fb in combinations_with_replacement(range(4), 2):
        keep = eligible_players(folds, fa, fb)
        fit = ~is_eval & inside & keep[pair].all(1)
        test = is_eval & (groups == [fa, fb]).all(1)
        if not fit.any() or not np.all(keep[truth[fit]]) or np.any(keep[pair[test]]):
            raise ValueError("unexpected owner-fold membership")
        base = gap[fit] * sign[fit]
        slope = float(fit_comparator(base[:, None])[0])
        coef = [anchored_fit(base * slope, x[fit] * sign[fit]) for x in features.T]
        outputs[test] = gap[test, None] * slope + features[test] * coef
        fits[f"{fa},{fb}"] = {"fit_count": int(fit.sum()), "base": slope, "coefficients": coef}
    if not np.isfinite(outputs[is_eval]).all():
        raise ValueError("incomplete predictions")
    ranks = np.stack([swapped_ranks(b["rank"][is_eval], g < 0) for g in outputs[is_eval].T], axis=1)
    return {"pair": pair, "raw": features, "gap": outputs[is_eval], "base_rank": b["rank"][is_eval],
            "new_rank": ranks, "truth": truth, "keys": b["keys"], **audits}, fits


def rank_report(items):
    base = np.concatenate([v["base_rank"] for v in items])
    ranks = np.concatenate([v["new_rank"] for v in items])
    change = competition(ranks) - competition(base)[:, None]
    values = np.column_stack([change, change[:, 0] - change[:, 1], change[:, 3] - change[:, 2]])
    # Both evaluation games for each owner travel together in the bootstrap.
    effects = bootstrap_matrix(values.reshape(1200, 2, -1).mean(1), np.repeat([0, 1], 600))
    report = {}
    for j, name in enumerate(ARMS):
        nr = ranks[:, j]
        report[name] = {**effects[j], "score": float(competition(nr).mean()),
                        "rescued": int(np.sum((base > 0) & (nr == 0))), "broken": int(np.sum((base == 0) & (nr > 0)))}
    return {"owners": 1200, "questions": len(base), "baseline": float(competition(base).mean()), "arms": report,
            "residual_minus_placebo": effects[-2], "fisher_minus_isotropic": effects[-1],
            "screen_pass": bool(effects[0]["mean"] >= .002 and effects[0]["ci95"][0] > 0),
            "above_placebo": bool(effects[-2]["ci95"][0] > 0)}


def main():
    guard()
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    doc_path = source / "results/pid_fusion_v2fin4_f.json"
    doc = json.loads(doc_path.read_text())
    old = json.loads((PRIOR / "baseline.json").read_text())
    check = json.loads((CACHE / "inputs.json").read_text())
    if check["plan"] != digest_file(PLAN) or old["ensemble"] != digest_file(doc_path):
        raise ValueError("source registration mismatch")
    out = {"plan": digest_file(PLAN), "code": digest_file(__file__), "ensemble": digest_file(doc_path), "splits": {}}
    for seed in (1, 3):
        all_records, mechanism, strata, inputs, fits, nominated = [], [], [], {}, {}, {}
        for pi, p in enumerate(POOLS):
            path = baseline_path(p, seed)
            if check["pools"][p][str(seed)] != digest_file(path):
                raise ValueError("baseline cache changed")
            with np.load(path, allow_pickle=False) as z:
                b = dict(z)
            if not np.array_equal(b["truth"], np.repeat(np.arange(600), 3)):
                raise ValueError("rows do not follow the preassigned slots")
            keys = np.r_[b["refs"].ravel(), b["keys"]]
            if len(set(keys)) != 6000 or not all(k.startswith(p + "|c:") for k in keys):
                raise ValueError("reference/query roles overlap")
            n, selected, rows = persistence(b, p)
            named = list(rows[0])
            values = np.asarray([[r[k] for k in named] for r in rows])
            mechanism.extend(values)
            strata.extend([pi] * len(rows))
            nominated[p] = {"eligible_anchors": n, "disjoint_pairs": len(rows)}
            gram = build_kernel(source, doc, p, seed, [k.split("|", 1)[1] for k in keys], old["pools"][p]["sources"][str(seed)])
            rec, fits[p] = score(p, b, gram)
            del gram
            path = CACHE / f"identity_{p}_{seed}.npz"
            atomic_npz(path, **rec, selected=selected, pair_values=values, pair_names=np.asarray(named))
            inputs[p] = digest_file(path)
            all_records.append(rec)
            print("finished actual finalists", p, seed, flush=True)
        out["splits"][str(seed)] = {"nomination": nominated, "persistence": dict(zip(named, bootstrap_matrix(mechanism, strata))),
                                      "ranking": rank_report(all_records), "fits": fits, "archives": inputs}
    atomic_json(OUT, out)


if __name__ == "__main__":
    main()
