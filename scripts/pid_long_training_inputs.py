"""Reconstruct the valid longer-training compositions and their matched controls."""
import argparse
from collections import Counter
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import ROOT, RUN, POOLS, atomic_json, atomic_npz, digest_file, guard, read_manifest
from pid_identity_inputs import baseline_path
from pid_choice_eval import PlayerEmbedding, load_store, load_vectors, parse_all, questions_from_split, seed_split, source_record, standardise, v2
from aicup_pid.long_training import sequence_variants, pair_scores, sibling_rows

CACHE = ROOT / ".superpowers/long_training_20261119"
PLAN = ROOT / "docs/superpowers/plans/2026-11-19-long-training-attribution.md"
ARMS = ("baseline", "replace", "add", "weight", "mix")
DOCUMENTS = {"baseline": "zz_r4f", "replace": "v2fin6_x", "add": "v2fin6_a", "accepted": "v2fin4_f"}


def load_composition(source):
    docs, counts, hashes = {}, {}, {}
    for arm, name in DOCUMENTS.items():
        path = source / "results" / f"pid_fusion_{name}.json"
        doc = json.loads(path.read_text())
        members = Counter()
        for public, private in doc["seq_official"]:
            filename = Path(public).name
            match = re.fullmatch(r"official_pub2_(net-?\d+_\w+)_t50\.npz", filename)
            if not match or Path(private).name != filename.replace("pub2", "priv2"):
                raise ValueError("unrecognized member or cross-pool pair")
            members[match[1]] += 1
        divisor = reduce(math.gcd, members.values())
        counts[arm] = Counter({k: value//divisor for k, value in members.items()})
        docs[arm] = doc
        hashes[arm] = digest_file(path)
    static = ("weights", "components", "cfg", "trunk_store", "emb_model", "arm", "geometry", "soft_lam")
    base = docs["baseline"]
    if any(d[k] != base[k] for d in docs.values() for k in static) or base["cfg"].get("seq_near"):
        raise ValueError("fusion has changes outside the registered intervention")
    original = counts["baseline"]
    early = Counter({k: v for k, v in original.items() if re.search(r"_o(?:44|47)Fs", k)})
    late = Counter({k.replace("_o44F", "_o58F").replace("_o47F", "_o57F"): v for k, v in early.items()})
    if (len(early) != 16 or set(late) & set(original) or original != counts["accepted"]
            or original-early+late != counts["replace"] or original+late != counts["add"]):
        raise ValueError("actual documents do not match the registered two-family interventions")
    return base, original, early, late, hashes


def eligible_sources(source):
    selected = {"o57F", "o58F"}
    controls = {"o57F": "o47F", "o58F": "o44F", "o53KF": "o45F", "o53F": "o45F",
                "o56KF": "o46F", "o56F": "o46F"}
    info = {}
    for family, control in controls.items():
        path = source / f"results/final/v2_offsim_f0p_{family}_vs_f0p_{control}_s0_1_2_3.json"
        v = json.loads(path.read_text())["steps_diff"]["both: talt vs t50"]
        if (v["diff"] >= .003 and v["ci95"][0] > 0) != (family in selected):
            raise ValueError("published family gate disagrees with the registered L2 branch")
        info[family] = {"file_sha": digest_file(path), "effect": v["diff"], "ci95": v["ci95"]}
    return info


def membership(sets, row):
    sizes = [len(x) for x in sets]
    a = np.repeat(np.arange(len(sets)), sizes)
    b = [row[k] for group in sets for k in group]
    return sparse.csr_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(len(sets), len(row)))


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def collect_split(pool, seed, source, manifest, ctx):
    doc, original, early, late, games, sgfs, moves, store, emb = ctx
    split = seed_split(manifest, pool, seed)
    owners = sorted(split["reference"])
    queries, truth, sizes = questions_from_split(split, owners)
    ref_keys = [split["reference"][p] for p in owners]
    if any(len(rr) != 7 for rr in ref_keys) or len(set(k for g in ref_keys for k in g)) != 4200:
        raise ValueError("unexpected enrollment panels")
    single = sizes == 1
    qkeys = np.array([q[0] for q in queries if len(q) == 1])
    if len(set(qkeys) | {k for panel in ref_keys for k in panel}) != 6000 or len(qkeys) != 1800:
        raise ValueError("enrollment and held-out games are not disjoint")
    by_key = {g["key"]: (g["original_key"], g["color"]) for g in games}
    ref = {p: [by_key[k] for k in split["reference"][p]] for p in owners}
    qs = [[by_key[k] for k in q] for q in queries]
    components = [k for k in doc["components"] if k != "seq"]
    nonseq = v2.components(ref, qs, moves, sgfs, doc["cfg"], store, emb, which=components)
    weights = dict(zip(doc["components"], doc["weights"]))
    fixed = sum(weights[c] * standardise(nonseq[c]) for c in components)
    row = {g["key"]: i for i, g in enumerate(games)}
    rx, qx = membership(ref_keys, row), membership(queries, row)
    keys = [g["original_key"] for g in games]
    base_sum, early_sum, late_sum = [np.zeros_like(fixed) for _ in range(3)]
    asset_sha = {}
    for k in sorted(set(original) | set(late)):
        path = source / "data/pid_v2_f0tmp/ft" / f"offsim_{pool}_h3s{seed}_{k}_t50.npz"
        x = load_vectors(path, keys)
        asset_sha[k] = source_record(path)["sha256"]
        matrix = standardise(unit(np.asarray(qx @ x, dtype=np.float64)) @ unit(np.asarray(rx @ x, dtype=np.float64)).T)
        if k in original:
            base_sum += original[k] * matrix
        if k in early:
            early_sum += early[k] * matrix
        if k in late:
            late_sum += late[k] * matrix
    share = early.total()/original.total()
    raw = sequence_variants(base_sum/original.total(), early_sum/early.total(), late_sum/late.total(), share)
    scores = [fixed + weights["seq"] * standardise(raw[k]) for k in ARMS]
    if not all(np.isfinite(x).all() for x in scores):
        raise ValueError("nonfinite reconstructed scores")
    if seed in (1, 3):
        with np.load(baseline_path(pool, seed)) as z:
            np.testing.assert_array_equal(qkeys, z["keys"])
            np.testing.assert_array_equal(ref_keys, z["refs"])
            np.testing.assert_allclose(scores[0][single], z["scores"], rtol=1e-10, atol=1e-9)
    ranks = np.stack([(x > x[np.arange(len(x)), truth, None]).sum(1) for x in scores], axis=1)
    one_truth = truth[single]
    one_scores = [x[single] for x in scores]
    order = np.argsort(-one_scores[0], axis=1, kind="stable")[:, :2]
    b = np.where(order[:, 0] == one_truth, order[:, 1], order[:, 0])
    sib = sibling_rows(one_truth)
    leaders = np.stack([x.argmax(1) for x in one_scores], axis=1)
    if not np.array_equal(leaders == one_truth[:, None], ranks[single] == 0):
        raise ValueError("winner tie changes the rank convention")
    changes, common, sibling, wins = [], [], [], []
    for j, x in enumerate(one_scores):
        current, sd = pair_scores(x, one_truth, b)
        if j == 0:
            origin, old_sd = current, sd
        fixed_sd, _ = pair_scores(x, one_truth, b, old_sd)
        other = np.mean([pair_scores(x[k], one_truth, b)[0] for k in sib.T], axis=0)
        if j == 0:
            other_base = other
        changes.append(current-origin)
        common.append(fixed_sd-origin)
        sibling.append(other-other_base)
        if np.any(current[:, 0] == current[:, 1]):
            raise ValueError("exact finalist margin tie")
        wins.append(current[:, 0] > current[:, 1])
    audit = np.array([0, 1, 3, 4, 120, 1490, 2137, 2891])
    record = {"keys": qkeys, "truth": truth, "sizes": sizes, "ranks": ranks, "leaders": leaders, "rival": b,
              "wins": np.stack(wins, axis=1), "change": np.stack(changes, axis=1), "common": np.stack(common, axis=1),
              "sibling": np.stack(sibling, axis=1), "audit": audit,
              "audit_scores": np.array([x[audit] for x in scores]), "audit_fixed": fixed[audit],
              "audit_sequence": np.array([raw["baseline"][audit], (early_sum/early.total())[audit], (late_sum/late.total())[audit]])}
    out = CACHE / f"{pool}_s{seed}.npz"
    atomic_npz(out, **record)
    return {"members": asset_sha, "records": digest_file(out), "share": share,
            "references": hashlib.sha256("\n".join(np.asarray(ref_keys).ravel()).encode()).hexdigest()}


def main(pool):
    budget = guard()
    torch.set_num_threads(3)
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    gates = eligible_sources(source)
    doc, original, early, late, docs = load_composition(source)
    games = [g for g in m["games"] if g["pool"] == pool]
    if len(games) != 6000 or any(not g["original_key"].startswith("c:") for g in games):
        raise ValueError("unexpected source games")
    sgfs = {g["original_key"]: g["sgf"] for g in games}
    moves = parse_all(sgfs, workers=8)
    store = load_store(str(source / doc["trunk_store"]))
    emb = PlayerEmbedding.from_state(torch.load(source / doc["emb_model"], map_location="cpu", weights_only=False), device="cpu")
    ctx = (doc, original, early, late, games, sgfs, moves, store, emb)
    CACHE.mkdir(parents=True, exist_ok=True)
    info = {"budget": budget, "plan": digest_file(PLAN), "docs": docs, "gate": gates["o56F"]["file_sha"], "family_checks": gates,
            "manifest": digest_file(RUN / "manifest.json"), "splits": {}}
    for s in (0, 1, 2, 3, 4, 5):
        info["splits"][str(s)] = collect_split(pool, s, source, m, ctx)
        print("completed", pool, s, flush=True)
    atomic_json(CACHE / f"inputs_{pool}.json", info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", choices=POOLS)
    main(parser.parse_args().pool)
