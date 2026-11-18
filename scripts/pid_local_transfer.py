"""Frozen policy evidence on candidate-only one-game questions for the full dual-window ensemble.

Primary fixed rule: row-standardize v2fin3_a and add 0.10 times the row-standardized
mean of the two L checkpoint log-likelihood-ratio matrices. G is the same rule with
the global-key controls. Score all 600 candidates, all three single-game questions,
both pools, holdout seeds 2 and 3. Bootstrap by owner within pool. A useful gain
requires mean rank score >= +0.002 with lower 95% bound > 0. No weight is fitted.
Separately count which of the baseline's first two candidates the specialist selects
when the true player is among them, overall and in the quartile with the smallest
baseline margin. This subset uses truth only after the two candidates were selected.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_local_choice_data import DEST, LEDGER, ROOT, RUN, atomic_json, check, physical_signature, read_data
from pid_choice_eval import load_baseline_inputs, ranks
from pid_choice_manifest import read_manifest, seed_split, source_record, stable_hash, verify_source
from pid_local_choice_features import extract
from pid_local_choice_fit import Arm, evaluate
from pid_reference_policy import setup_and_moves, split_indices
from aicup_pid.fusion import standardise

SOURCE = DEST / "simple_ko_v2"
OUT = DEST / "transfer_v2fin3a"
WEIGHT = .10
ARMS = ("L", "G")
SEEDS = (17, 29)


def inputs():
    adapter = read_data()
    parent = read_manifest(RUN, verify=False)
    root = Path(parent["source_root"])
    doc = root / "results/pid_fusion_v2fin3_a.json"
    expected = root / "results/final/v2_offsim_v2fin3_a_vs_v2fin_x_s2_3.json"
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    checked = json.loads((SOURCE / "checked.json").read_text())
    if not checked["verified"] or checked["protocol"] != source_protocol["id"]:
        raise ValueError("unverified source policy")
    for rec in source_protocol["code"]:
        verify_source(rec)
    paths = [Path(__file__), ROOT / "scripts/pid_choice_eval.py", ROOT / "src/aicup_pid/choice_eval.py",
             OUT.parent / "source_verified_data.json", RUN / "manifest.json", doc, expected,
             SOURCE / "protocol.json", SOURCE / "normalizer.npz", SOURCE / "scores.npz"]
    paths += [SOURCE / f"{arm}_{s}.pt" for arm in ARMS for s in SEEDS]
    records = [source_record(path) for path in paths]
    body = {"inputs": records, "source_policy": checked["protocol"], "source_data": adapter["id"],
            "base": "v2fin3_a", "weight": WEIGHT, "holdouts": [2, 3], "seeds": SEEDS,
            "criterion": "both pools together, rank gain >=0.002 and owner bootstrap lower bound >0"}
    ident = stable_hash(json.dumps(body, sort_keys=True))[:12]
    return adapter, parent, doc, json.loads(expected.read_text())["scores"], body, ident


def frozen_coordinates(data, temporary):
    """Undo temporary anchor-only scaling and apply the saved training normalization."""
    device = data["local_key"].device
    tmp = {k: torch.as_tensor(v, device=device) for k, v in temporary.items()}
    with np.load(SOURCE / "normalizer.npz", allow_pickle=False) as saved:
        old = {k: torch.as_tensor(saved[k], device=device) for k in saved}
    for prod in (True, False):
        fixed = old[f"action_rms_{int(prod)}"]
        data["residual", prod] *= data["scale", prod] / fixed
        data["scale", prod] = fixed
    global_temp, local_temp = data["local_key"].split([388, 96], -1)
    def restore(x, name):
        raw = x * tmp[name + "_sd"] + tmp[name + "_mean"]
        return (raw - old[name + "_mean"]) / old[name + "_sd"]
    gz, lz = restore(global_temp, "global"), restore(local_temp, "local")
    extra = (torch.tanh(gz @ old["fixed_global_projection"]) - old["control_mean"]) / old["control_sd"]
    data["local_key"] = torch.cat([gz, lz], -1)
    data["global_key"] = torch.cat([gz, extra], -1)


def rank_score(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.)


def run_pool(pool):
    adapter, parent, doc, _, registration, ident = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / f"{pool}_registered.json"
    if marker.exists() and json.loads(marker.read_text())["id"] != ident:
        raise ValueError("changed transfer registration")
    atomic_json(marker, {"id": ident, **registration})
    if (OUT / f"{pool}.json").exists():
        raise FileExistsError("this pool already has a result")
    # Temporary extraction statistics only use the original training anchors. The saved
    # training statistics replace them before scoring any candidates.
    anchors = [g for g in adapter["games"] if g["split"] == "train" and g["slot"] < 7][:32]
    prior_owners, prior_idx = split_indices(adapter["games"], "test")
    proof = [dict(adapter["games"][i], split="proof") for i in prior_idx[:2].ravel()]
    targets = [dict(g, split="application", slot=-1) for g in parent["games"] if g["pool"] == pool]
    if len(targets) != 6000 or len({g["owner"] for g in targets}) != 600:
        raise ValueError("unexpected candidate coverage")
    physical = {physical_signature(g) for g in adapter["games"]}
    target_sigs = {physical_signature(g) for g in targets}
    if len(target_sigs) != len(targets) or physical & target_sigs:
        raise ValueError("physical game overlap in transfer")
    shortest = min(sum(c == g["color"] for c, _, _ in setup_and_moves(g["sgf"])[1][:256]) for g in targets)
    if shortest < 32:
        raise ValueError(f"official reference needs explicit short-game support: minimum {shortest}")
    candidate_key = {g["key"]: i + len(anchors) + len(proof) for i, g in enumerate(targets)}
    games = [*anchors, *proof, *targets]
    data, temporary = extract({"games": games, "checkpoint": adapter["checkpoint"]})
    frozen_coordinates(data, temporary)
    proof_indices = np.arange(len(anchors), len(anchors) + len(proof)).reshape(2, 10)
    all_arms, errors = {}, {}
    with np.load(SOURCE / "scores.npz", allow_pickle=False) as golden:
        if golden["owners"].tolist()[:2] != prior_owners[:2]:
            raise ValueError("proof owners changed")
        for label in ARMS:
            for seed in SEEDS:
                name = f"{label}_{seed}"
                ck = torch.load(SOURCE / f"{name}.pt", weights_only=True)
                if ck["protocol"] != registration["source_policy"]:
                    raise ValueError("checkpoint mismatch")
                arm = Arm(label, data)
                arm.model.load_state_dict(ck["state"])
                score, _, _, _ = evaluate(arm, data, proof_indices)
                errors[name] = float(np.abs(score - golden[name][:2, :, :2]).max())
                if errors[name] > .002:
                    raise ValueError(f"transfer numerical equivalence failed: {name} {errors[name]}")
                all_arms[name] = arm
    print(f"{pool} restored four frozen models, proof errors {errors}", flush=True)
    saved = {}
    for seed in (2, 3):
        split = seed_split(parent, pool, seed)
        owners = sorted(split["reference"])
        indices = np.asarray([[candidate_key[k] for k in split["reference"][p] + split["heldout"][p]] for p in owners])
        view = {**parent, "assets": {**parent["assets"], "fusion": {"path": str(doc)}}}
        baseline = load_baseline_inputs(view, pool, seed)
        if baseline["owners"] != owners or baseline["references"] != split["reference"]:
            raise ValueError("baseline reference mismatch")
        is_one = baseline["sizes"] == 1
        queries = [q[0] for q, one in zip(baseline["questions"], is_one) if one]
        if queries != [k for p in owners for k in split["heldout"][p]]:
            raise ValueError("baseline query mismatch")
        base = standardise(baseline["baseline"][is_one])
        truth = baseline["truth"][is_one]
        order = np.argsort(-base, axis=1, kind="stable")[:, :2]
        row = np.arange(len(truth))
        margin = base[row, order[:, 0]] - base[row, order[:, 1]]
        saved[f"s{seed}_truth"] = truth
        saved[f"s{seed}_small_margin"] = margin <= np.quantile(margin, .25)
        saved[f"s{seed}_contains_owner"] = (order == truth[:, None]).any(1)
        saved[f"s{seed}_base"] = ranks(base, truth)
        for label in ARMS:
            models = [evaluate(all_arms[f"{label}_{model_seed}"], data, indices)[0] for model_seed in SEEDS]
            policy = standardise(np.mean(models, axis=0).reshape(len(truth), len(owners)))
            saved[f"s{seed}_{label}"] = ranks(base + WEIGHT * policy, truth)
            saved[f"s{seed}_{label}_pair_won"] = order[row, np.argmax(policy[row[:, None], order], axis=1)] == truth
            print(f"{pool} split {seed} scored all 600 candidates with {label}", flush=True)
        saved[f"s{seed}_qkeys"] = np.asarray(queries)
        saved[f"s{seed}_owners"] = np.asarray(owners)
        saved[f"s{seed}_base_sources"] = np.asarray(stable_hash(json.dumps(baseline["sources"], sort_keys=True)))
    score_file = OUT / f"{pool}_ranks.npz"
    np.savez_compressed(score_file, **saved)
    report = {"id": ident, "pool": pool, "sources": source_record(score_file), "min_own_moves": shortest,
              "application_games": len(targets), "proof_max_abs_llr_error": errors, "job": os.environ["SLURM_JOB_ID"]}
    atomic_json(OUT / f"{pool}.json", report)


def summarize():
    _, _, _, expected, _, ident = inputs()
    rng = np.random.default_rng(1811)
    draws = np.concatenate([rng.integers(off, off + 600, (5000, 600)) for off in (0, 600)], 1)
    all_rows = []
    baseline_check, source = {}, {}
    for i, pool in enumerate(("pub2", "priv2")):
        info = json.loads((OUT / f"{pool}.json").read_text())
        if info["id"] != ident:
            raise ValueError("pool protocol mismatch")
        verify_source(info["sources"])
        source[pool] = info
        groups = []
        with np.load(info["sources"]["path"], allow_pickle=False) as z:
            for seed in (2, 3):
                keys = ("truth", "base", "L", "G", "small_margin", "contains_owner", "L_pair_won", "G_pair_won", "qkeys")
                block = {k: z[f"s{seed}_{k}"] for k in keys}
                block["owner"] = i * 600 + block["truth"]
                groups.append(block)
        computed = rank_score(np.concatenate([r["base"] for r in groups])).mean()
        target = expected[pool]["alt"]["1"]
        if abs(computed - target) > 1e-6:
            raise ValueError("baseline reproduction failed")
        baseline_check[pool] = float(computed)
        all_rows.extend(groups)
    data = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    summaries = {}
    for left, right in (("L", "base"), ("G", "base"), ("L", "G")):
        a, b, cluster = data[left], data[right], data["owner"]
        d = rank_score(a) - rank_score(b)
        num = np.bincount(cluster, d, minlength=1200)
        den = np.bincount(cluster, minlength=1200)
        simulated = num[draws].sum(1) / den[draws].sum(1)
        summaries[f"{left} minus {right}"] = {"difference": float(d.mean()), "ci95": np.quantile(simulated, [.025, .975]).tolist(),
                                            "rescued": int(((a == 0) & (b > 0)).sum()), "broken": int(((a > 0) & (b == 0)).sum())}
    pairs = {}
    for label, sel in (("all_baseline_top2", data["contains_owner"]), ("small_baseline_margin_top2", data["contains_owner"] & data["small_margin"])):
        pairs[label] = {"rows": int(sel.sum()), "baseline_correct": int((data["base"][sel] == 0).sum()),
                        "L_correct": int(data["L_pair_won"][sel].sum()), "G_correct": int(data["G_pair_won"][sel].sum())}
    primary = summaries["L minus base"]
    report = {"protocol": ident, "base": "v2fin3_a", "alpha": WEIGHT, "rank_by_method": {name: float(rank_score(data[name]).mean()) for name in ("base", "L", "G")},
              "rows": len(data["qkeys"]), "unique_games": len(np.unique(data["qkeys"])), "sources": source,
              "qsize": 1, "baseline_checks": baseline_check, "comparisons": summaries, "top_two_diagnostic": pairs,
              "passed": bool(primary["difference"] >= .002 and primary["ci95"][0] > 0), "job": os.environ["SLURM_JOB_ID"]}
    atomic_json(OUT / "summary.json", report)
    atomic_json(ROOT / "results/final/local_choice_transfer_20261118.json", report)
    print(json.dumps({"passed": report["passed"], "primary": primary}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pub2", "priv2", "summary"))
    stage = p.parse_args().stage
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("use production Slurm")
    torch.set_num_threads(2)
    check(RUN, LEDGER)
    if stage == "summary":
        summarize()
    else:
        run_pool(stage)
    check(RUN, LEDGER)
