#!/usr/bin/env python
"""Descriptive equality/truncation audit; never reads official questions."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aicup_pid import paths  # Configure the shared parser dependency before importing it.
from aicup.sgf import board_size, handicap, setup_and_moves
from aicup_pid.opening import _sym
from pid_choice_manifest import atomic_json, read_manifest, source_record, stable_hash


def signature(sgf, subject, limit=None, symmetric=False):
    stones, moves = setup_and_moves(sgf)
    if board_size(sgf) != 19 or subject not in ("B", "W"):
        raise ValueError("unsupported observation")
    moves = moves if limit is None else moves[:limit]
    def transformed(k):
        return (handicap(sgf), subject, tuple(sorted(_sym(k, x, y) for x, y in stones)),
                tuple((c, *_sym(k, 19 if x is None else x, 19 if y is None else y)) for c, x, y in moves))
    return stable_hash(repr(min(transformed(k) for k in range(8 if symmetric else 1))))


def audit(root):
    m = read_manifest(root)
    groups = {name: defaultdict(list) for name in ("raw_full", "raw_256", "d4_full", "d4_256")}
    lengths = defaultdict(list)
    for g in m["games"]:
        lengths[g["pool"]].append(len(setup_and_moves(g["sgf"])[1]))
        for name, buckets in groups.items():
            key = signature(g["sgf"], g["color"], 256 if name.endswith("256") else None, name.startswith("d4"))
            buckets[key].append(g)
    out = {"run_id": m["run_id"], "job_id": os.environ["SLURM_JOB_ID"], "source": source_record(__file__),
           "scope": "same subject, handicap, setup and actual moves; headers excluded. D4 equivalence is not identical network input. Truncation describes sequence encoder only.",
           "lengths": {}, "collisions": {}}
    for pool, ns in lengths.items():
        a = np.array(ns)
        out["lengths"][pool] = {"games": len(ns), "total_plies": int(a.sum()), "games_over256": int((a > 256).sum()),
            "fraction_over256": float((a > 256).mean()), "omitted_plies": int(np.maximum(a - 256, 0).sum()),
            "omitted_fraction": float(np.maximum(a - 256, 0).sum() / a.sum()), "quantiles": np.quantile(a, [0, .25, .5, .75, .9, .95, 1]).tolist()}
    for name, buckets in groups.items():
        result = {"within_pool": {}, "cross_pool_groups": 0, "calibration_official_overlap_groups": 0}
        for pool in lengths:
            duplicate = conflicts = games = 0
            examples = []
            for rows in buckets.values():
                rs = [g for g in rows if g["pool"] == pool]
                duplicate += int(len(rs) > 1)
                if len({g["owner"] for g in rs}) > 1:
                    conflicts += 1; games += len(rs)
                    if len(examples) < 5:
                        examples.append([g["key"] for g in rs])
            result["within_pool"][pool] = {"duplicate_groups": duplicate, "different_owner_groups": conflicts, "different_owner_records": games, "examples": examples}
        for rows in buckets.values():
            pools = {g["pool"] for g in rows}
            result["cross_pool_groups"] += int(len(pools) > 1)
            result["calibration_official_overlap_groups"] += int("calibration" in pools and len(pools) > 1)
        out["collisions"][name] = result
    atomic_json(Path(root) / "events/observation_audit.json", out)
    atomic_json(ROOT / "results/final/observation_audit_20261116.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use production Slurm")
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    audit(ap.parse_args().root)
