"""Reconstruct the current full ensemble on candidate-only games for the tactical probe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pid_choice_eval import load_baseline_inputs, ranks, rank_scores
from pid_choice_manifest import atomic_json, digest_file, read_manifest
from pid_choice_runtime import accounting

RUN = Path("/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8")
CACHE = ROOT / ".superpowers" / "tactical_replies_20261118"
PLAN = ROOT / "docs/superpowers/plans/2026-11-18-tactical-replies.md"
LEDGER = ROOT / ".superpowers/sdd/2026-11-16-choice-signal/jobs.txt"
POOLS = ("pub2", "priv2")
SEED = 3


def guard():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("run the study on production Slurm")
    ids = set(LEDGER.read_text().split()) | {job}
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", ",".join(sorted(ids)),
                                  "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES"], text=True)
    budget = accounting(raw, ids)
    status = os.statvfs(ROOT)
    if status.f_bavail * status.f_frsize < 3 * 1024 ** 3:
        raise ValueError("home has less than 3 GiB available")
    CACHE.mkdir(parents=True, exist_ok=True)
    return {k: budget[k] for k in ("gpu_seconds", "reserved_gpu_seconds", "active_gpus")}


def atomic_npz(path, **items):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        name = tmp.name
    try:
        np.savez_compressed(name, **items)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def build():
    budget = guard()
    torch.set_num_threads(3)
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    doc = source / "results/pid_fusion_v2fin4_f.json"
    expected = source / "results/final/v2_offsim_v2fin4_f_vs_v2fin_x_s0_1_2_3.json"
    published = json.loads(expected.read_text())
    assert published["seeds"] == [0, 1, 2, 3]
    rec = {"plan": digest_file(PLAN), "ensemble": digest_file(doc), "expected": digest_file(expected),
           "manifest": digest_file(RUN / "manifest.json"), "budget_start": budget, "pools": {}}
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="pid-tactical-") as tmp:
        base = Path(tmp)
        (base / "data").mkdir()
        (base / "results").symlink_to(source / "results")
        (base / "data/trunk").symlink_to(source / "data/trunk")
        (base / "data/pid_v2_seq_ft_offsim").symlink_to(source / "data/pid_v2_f0tmp/ft")
        view = {**m, "source_root": str(base), "assets": {**m["assets"], "fusion": {"path": str(doc)}}}
        for pool in POOLS:
            checks, fingerprints = [], {}
            for seed in published["seeds"]:
                b = load_baseline_inputs(view, pool, seed)
                one = b["sizes"] == 1
                scores, truth = b["baseline"][one], b["truth"][one]
                checks.append(rank_scores(scores, truth).mean())
                fingerprints[str(seed)] = {s["realpath"]: s["sha256"] for s in b["sources"]}
                if seed == SEED:
                    keys = np.array([q[0] for q in b["questions"] if len(q) == 1])
                    owners = np.array(b["owners"])
                    refs = np.array([b["references"][p] for p in owners])
                    if scores.shape != (1800, 600) or refs.shape != (600, 7):
                        raise ValueError("unexpected full-ensemble layout")
                    atomic_npz(CACHE / f"base_{pool}.npz", scores=scores, truth=truth, keys=keys,
                               owners=owners, refs=refs, rank=ranks(scores, truth))
                print(f"finished original ensemble {pool} split {seed}", flush=True)
            got = float(np.mean(checks))
            want = published["scores"][pool]["alt"]["1"]
            if abs(got - want) > 1e-7:
                raise ValueError(f"cannot reproduce ensemble in {pool}: {got} versus {want}")
            rec["pools"][pool] = {"single_game": got, "published": want, "seed_means": checks,
                                  "sources": fingerprints, "cache_sha": digest_file(CACHE / f"base_{pool}.npz")}
    atomic_json(CACHE / "baseline.json", rec)


if __name__ == "__main__":
    build()
