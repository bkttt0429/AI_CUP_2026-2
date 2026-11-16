#!/usr/bin/env python
"""Export small, auditable research artifacts; no model selection occurs here."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pid_choice_manifest import atomic_json, digest_file, read_manifest
from pid_choice_exact import read_protocol
from pid_choice_runtime import check


def report(run_dir):
    root = Path(run_dir)
    m = read_manifest(root)
    protocol = read_protocol(root)
    def read(name):
        return json.loads((root / name).read_text())
    runtime = check(root, ROOT / ".superpowers/sdd/2026-11-16-choice-signal/jobs.txt")
    original = {}
    for pool in ("pub2", "priv2"):
        a = read(f"audit_reference_{pool}_s2.json")
        original[pool] = {"verdict": a["verdict"], "seed": 2,
                          "arms": {arm: {ridge: {k: value[k] for k in ("errors", "ranking", "passed")} for ridge, value in values.items()} for arm, values in a["arms"].items()}}
    primary = read("exact_eval_s2_3.json")
    robust = read("exact_eval_s4_5.json") if (root / "exact_eval_s4_5.json").exists() else None
    if primary["protocol_id"] != protocol["protocol_id"] or robust and robust["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("report mixes exact protocols")
    groups = [(2, 3)] + ([(4, 5)] if robust else [])
    selection, diagnostics, baseline, arithmetic, checksums = {}, {}, {}, {}, {}
    for seeds in groups:
        group = "_".join(map(str, seeds))
        for pool in ("pub2", "priv2"):
            aa = read(f"exact_arithmetic_{pool}_s{group}.json")
            arithmetic[pool + "_s" + group] = {"verdict": aa["verdict"], "checks": len(aa["checks"]), "max_abs_error": max(c["max_abs_error"] for c in aa["checks"].values())}
            for seed in seeds:
                key = f"{pool}_s{seed}"
                selection[key] = read(f"exact_selection_{key}.json")
                diagnostics[key] = read(f"exact_diagnostics_{key}.json")
                with np.load(root / f"exact_grid_{key}.npz", allow_pickle=False) as grid:
                    baseline[key] = {"equal_size_rank_score": float(grid["base"].mean()), "by_size": grid["base"].mean(axis=(0, 1)).tolist()}
                for name in (f"exact_grid_{key}.npz", f"exact_selection_{key}.json", f"exact_diagnostics_{key}.json", f"baseline_{key}.npz"):
                    checksums[name] = digest_file(root / name)
    audit = {"run_id": m["run_id"], "exact_protocol_id": protocol["protocol_id"], "input": read("audit_input.json"),
             "baseline_reproduction": read("baseline_check.json"), "calibration": read("audit_calibration.json"),
             "original_quadratic_study": {"verdict": "approximation_invalid", "effectiveness_evaluated": False, "reference_audits": original},
             "exact_arithmetic": arithmetic, "runtime_snapshot": runtime, "source_run_dir": str(root), "exact_protocol": protocol}
    result = {"run_id": m["run_id"], "primary": primary, "robustness": robust,
              "verdict": robust["verdict"] if robust else primary["verdict"], "baseline": baseline, "diagnostics": diagnostics,
              "checksums": checksums, "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "count_scope": "descriptive question counts include repeated owners/games and 3 singles per owner; inferential score equally weights sizes1/2/3 and bootstrap clusters by owner"}
    out = ROOT / "results/final"
    stem = f"choice_probe_{m['run_id']}"
    for suffix, data in (("audit", audit), ("selection", selection), ("eval", result)):
        atomic_json(out / f"{stem}_{suffix}.json", data)
    print(json.dumps({"verdict": result["verdict"], "primary": primary["arms"], "contrasts": primary["contrasts"], "baseline": baseline,
                      "A2_counts": {k: {n: d["A2"][n] for n in ("recovered_top1", "broken_top1")} for k, d in diagnostics.items()},
                      "gpu_hours_snapshot": runtime["gpu_seconds"] / 3600, "artifact_bytes_snapshot": runtime["bytes"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use Slurm")
    report(a.run_dir)
