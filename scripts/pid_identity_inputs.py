"""Recover the accepted split-1 baseline and bind both split caches to published sources."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE as PRIOR, ROOT, RUN, POOLS, guard, atomic_npz, atomic_json, digest_file, read_manifest
from pid_choice_eval import load_baseline_inputs, ranks, rank_scores

CACHE = ROOT / ".superpowers/identity_variance_20261118"
PLAN = ROOT / "docs/superpowers/plans/2026-11-18-identity-variance.md"


def baseline_path(pool, seed):
    return (PRIOR / f"base_{pool}.npz") if seed == 3 else CACHE / f"base_{pool}_s{seed}.npz"


def main():
    budget = guard()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    doc = source / "results/pid_fusion_v2fin4_f.json"
    prior = json.loads((PRIOR / "baseline.json").read_text())
    if prior["ensemble"] != digest_file(doc) or prior["manifest"] != digest_file(RUN / "manifest.json"):
        raise ValueError("baseline source changed")
    out = {"budget": budget, "plan": digest_file(PLAN), "pools": {}}
    with tempfile.TemporaryDirectory(dir="/dev/shm") as scratch:
        root = Path(scratch)
        (root / "data").mkdir()
        for link, path in [("results", "results"), ("data/trunk", "data/trunk"), ("data/pid_v2_seq_ft_offsim", "data/pid_v2_f0tmp/ft")]:
            (root / link).symlink_to(source / path)
        view = {**m, "source_root": str(root), "assets": {**m["assets"], "fusion": {"path": str(doc)}}}
        for p in POOLS:
            old = prior["pools"][p]
            if digest_file(baseline_path(p, 3)) != old["cache_sha"]:
                raise ValueError("split-3 cache changed")
            b = load_baseline_inputs(view, p, 1)
            use = b["sizes"] == 1
            sc, truth = b["baseline"][use], b["truth"][use]
            if abs(rank_scores(sc, truth).mean() - old["seed_means"][1]) > 1e-9:
                raise ValueError("split-1 single-game baseline mismatch")
            if {v["realpath"]: v["sha256"] for v in b["sources"]} != old["sources"]["1"]:
                raise ValueError("split-1 vector fingerprints changed")
            owners = np.asarray(b["owners"])
            atomic_npz(baseline_path(p, 1), scores=sc, truth=truth, owners=owners, rank=ranks(sc, truth),
                       keys=np.array([q[0] for q in b["questions"] if len(q) == 1]),
                       refs=np.array([b["references"][o] for o in owners]))
            out["pools"][p] = {str(s): digest_file(baseline_path(p, s)) for s in (1, 3)}
            print("verified baseline", p, flush=True)
    atomic_json(CACHE / "inputs.json", out)


if __name__ == "__main__":
    main()
