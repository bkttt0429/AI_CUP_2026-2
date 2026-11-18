"""Replay original, duplicate and genuine two-game queries with the full accepted model."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_long_training_inputs import CACHE, POOLS, ROOT, RUN, PlayerEmbedding, atomic_json, atomic_npz, digest_file, guard, load_composition, load_store, load_vectors, membership, parse_all, read_manifest, seed_split, standardise, unit, v2
from pid_long_training_report import OwnerBootstrap
from pid_tactical_score import competition

PLAN = ROOT / "docs/superpowers/plans/2026-11-19-score-averaging-control.md"
OUT = ROOT / "results/final/two_game_averaging_20261119.json"


def run_pool(pool):
    guard()
    torch.set_num_threads(3)
    manifest = read_manifest(RUN, verify=False)
    source = Path(manifest["source_root"])
    doc, members, _, _, _ = load_composition(source)
    games = [g for g in manifest["games"] if g["pool"] == pool]
    by_key = {g["key"]: (g["original_key"], g["color"]) for g in games}
    keys = [g["original_key"] for g in games]
    row = {g["key"]: i for i, g in enumerate(games)}
    weights = dict(zip(doc["components"], doc["weights"]))
    sgf = {g["original_key"]: g["sgf"] for g in games}
    parsed = parse_all(sgf, workers=8)
    store = load_store(str(source / doc["trunk_store"]))
    emb = PlayerEmbedding.from_state(torch.load(source / doc["emb_model"], map_location="cpu", weights_only=False), device="cpu")
    meta = json.loads((CACHE / f"inputs_{pool}.json").read_text())
    for seed in (4, 5):
        split = seed_split(manifest, pool, seed)
        owners = sorted(split["reference"])
        refs = [split["reference"][p] for p in owners]
        groups = []
        for p in owners:
            a, b = split["heldout"][p][:2]
            groups.extend([[a], [b], [a,b], [a,a], [b,b]])
        panels = {p: [by_key[k] for k in rr] for p, rr in zip(owners, refs)}
        queries = [[by_key[k] for k in q] for q in groups]
        components = v2.components(panels, queries, parsed, sgf, doc["cfg"], store, emb, which=[x for x in weights if x != "seq"])
        other = sum(weights[k]*standardise(components[k]) for k in weights if k != "seq")
        rx, qx = membership(refs, row), membership(groups, row)
        accum = np.zeros_like(other)
        for k, n in members.items():
            path = source / "data/pid_v2_f0tmp/ft" / f"offsim_{pool}_h3s{seed}_{k}_t50.npz"
            if digest_file(path) != meta["splits"][str(seed)]["members"][k]:
                raise ValueError("accepted sequence member changed")
            x = load_vectors(path, keys)
            acc = unit(np.asarray(qx@x, float)) @ unit(np.asarray(rx@x, float)).T
            accum += n*standardise(acc)
        scores = (other + weights["seq"]*standardise(accum/members.total())).reshape(600,5,600)
        average = scores[:, :2].mean(axis=1, keepdims=True)
        scores = np.concatenate([scores, average], axis=1)
        truth = np.arange(600)
        true_scores = np.take_along_axis(scores, truth[:,None,None], axis=2)
        rank = (scores > true_scores).sum(2)
        winners = scores.argmax(2)
        if not np.array_equal(winners == truth[:,None], rank == 0):
            raise ValueError("winner tie changes rank convention")
        with np.load(CACHE / f"{pool}_s{seed}.npz") as prev:
            baseline = prev["ranks"][:,0].reshape(600,5)
            np.testing.assert_array_equal(rank[:, :3], baseline[:, [0,1,3]])
            np.testing.assert_array_equal(winners[:, :2], prev["leaders"][:,0].reshape(600,3)[:, :2])
        duplicate_error = np.abs(scores[:, 3:5]-scores[:, :2]).max(2)
        rival_scores = np.take_along_axis(average[:,0], winners[:,:2], axis=1)
        beat_both = (true_scores[:,5] > rival_scores).all(1)
        wrong_same = (rank[:,:2] > 0).all(1) & (winners[:,0] == winners[:,1])
        if (wrong_same & (rank[:,5] == 0)).any():
            raise ValueError("an arithmetic mean reversed two strictly losing margins")
        atomic_npz(CACHE / f"average_{pool}_{seed}.npz", rank=rank, winners=winners, beat_both=beat_both,
                   duplicate_error=duplicate_error, ids=truth+600*POOLS.index(pool))
        print("verified native single/pair ranks", pool, seed, flush=True)


def summarize():
    guard()
    fields = {k: [] for k in ("ids","rank","winners","beat_both","duplicate_error")}
    source = {}
    for s in (4,5):
        for p in POOLS:
            path = CACHE / f"average_{p}_{s}.npz"
            with np.load(path) as z:
                for k in fields:
                    fields[k].append(z[k])
            source[f"{p}/{s}"] = digest_file(path)
    d = {k: np.concatenate(v) for k,v in fields.items()}
    ids, rank, winners = d["ids"], d["rank"], d["winners"]
    native, avg = rank[:,2] == 0, rank[:,5] == 0
    boot = OwnerBootstrap()
    measures = {"native_top1": native, "average_top1": avg, "native_minus_avg_top1": native.astype(float)-avg,
                "native_minus_avg_score": competition(rank[:,2])-competition(rank[:,5])}
    wrong = (rank[:, :2] > 0).all(1)
    different = wrong & (winners[:,0] != winners[:,1])
    agreement = {k: int((wrong & v).sum()) for k, v in (("both", native & avg), ("only_native", native & ~avg),
                                                          ("only_average", ~native & avg), ("neither", ~native & ~avg))}
    duplicate = rank[:,3:5]-rank[:,:2]
    result = {"rows": len(ids), "all": dict(zip(measures, boot.means(ids, np.column_stack(list(measures.values()))))),
              "both_wrong": int(wrong.sum()), "recovered": agreement, "different_wrong": int(different.sum()),
              "both_old_nominees_beaten": int((different & d["beat_both"]).sum()),
              "both_beaten_third_wins": int((different & d["beat_both"] & ~avg).sum()),
              "identical_top_nominee_avg_wins": int((wrong & ~different & avg).sum()),
              "both_wrong_native_minus_avg_accuracy": boot.means(ids[wrong], native[wrong].astype(float)-avg[wrong])[0],
              "duplicate_game_tests": int(duplicate.size), "duplicate_rank_changes": int(np.count_nonzero(duplicate)),
              "duplicate_top_changes": int(np.count_nonzero(winners[:,3:5] != winners[:,:2])),
              "duplicate_max_score_difference": float(d["duplicate_error"].max()), "inputs": source, "protocol": digest_file(PLAN)}
    atomic_json(OUT, result)
    print("joint and average", agreement, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        summarize()
    elif sys.argv[1] in POOLS:
        run_pool(sys.argv[1])
    else:
        raise ValueError("use a registered pool or report")
