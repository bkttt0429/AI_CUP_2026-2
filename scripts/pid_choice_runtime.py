#!/usr/bin/env python
"""Account only this study's allocations, including failures and idle time."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def accounting(raw, job_ids):
    records = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or fields[0] not in job_ids:
            continue
        jid, state, elapsed, limit, tres = fields[:5]
        state = state.split()[0].rstrip("+")
        gpu_match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        # Every pending study allocation explicitly requests one GPU.
        gpu = int(gpu_match[1]) if gpu_match else 1 if state not in TERMINAL else 0
        seconds = int(elapsed) * gpu
        reserved = seconds if state in TERMINAL else int(limit) * 60 * gpu
        records[jid] = {"state": state, "gpu_seconds": seconds, "reserved_gpu_seconds": reserved, "gpus": gpu}
    if set(records) != set(job_ids):
        raise ValueError("missing Slurm allocation records")
    out = {"jobs": records, "gpu_seconds": sum(r["gpu_seconds"] for r in records.values()),
           "reserved_gpu_seconds": sum(r["reserved_gpu_seconds"] for r in records.values()),
           "active_gpus": sum(r["gpus"] for r in records.values() if r["state"] not in TERMINAL | {"PENDING"})}
    if out["active_gpus"] > 2:
        raise ValueError("more than two study GPUs active")
    if out["reserved_gpu_seconds"] > 6 * 3600:
        raise ValueError("six GPU-hour budget exceeded")
    return out


def check(run_dir, job_file):
    jobs = Path(job_file).read_text().split()
    if len(jobs) != len(set(jobs)) or any(not j.isdigit() for j in jobs):
        raise ValueError("invalid study job ledger")
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", ",".join(jobs), "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES"], text=True)
    out = accounting(raw, jobs)
    root = Path(run_dir)
    out["bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if out["bytes"] >= 1024 ** 3:
        raise ValueError("one GiB artifact budget exceeded")
    out["quota_view"] = {}
    for mount in ("/work/u2507115", "/home/u2507115"):
        user, parent = os.statvfs(mount), os.statvfs(str(Path(mount).parent))
        total, parent_total = user.f_blocks * user.f_frsize, parent.f_blocks * parent.f_frsize
        # Weka exposes the directory quota through statfs. Never silently
        # substitute the PB-scale global filesystem for the 100GiB user view.
        if total >= parent_total:
            raise ValueError("quota-scoped statfs view unavailable")
        out["quota_view"][mount] = {"total": total, "parent_total": parent_total, "available": user.f_bavail * user.f_frsize}
    out["free_bytes"] = {k: v["available"] for k, v in out["quota_view"].items()}
    for mount, gib in (("/work/u2507115", 6), ("/home/u2507115", 3)):
        if out["free_bytes"][mount] < gib * 1024 ** 3:
            raise ValueError(f"free disk threshold failed: {mount}")
    out["verdict"] = "passed"
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix="runtime-", suffix=".tmp", delete=False) as f:
        f.write(json.dumps(out, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, root / "runtime.json")
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--job-file", required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use Slurm")
    check(a.run_dir, a.job_file)
