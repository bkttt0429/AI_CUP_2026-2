"""Run the local conditional choice pilot in a single production Slurm allocation."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_local_choice_data import DEST, LEDGER, ROOT, RUN, atomic_json, check, read_data, source_record, stable_hash
from pid_local_choice_features import extract
from pid_reference_policy import AICUP1
import json
import numpy as np
import torch


def protocol(m):
    files = [ROOT / p for p in ("src/aicup_pid/local_choice.py", "src/aicup_pid/reference_policy.py",
                                "scripts/pid_local_choice_data.py", "scripts/pid_local_choice_features.py",
                                "scripts/pid_local_choice_fit.py", "scripts/pid_local_choice.py",
                                "docs/superpowers/plans/2026-11-18-local-choice.md",
                                "docs/superpowers/specs/2026-11-18-local-choice-signal-design.md",
                                "scripts/pid_reference_policy.py", "scripts/pid_choice_manifest.py",
                                "scripts/pid_choice_runtime.py", "scripts/pid_trunk_extract.py")]
    files += [AICUP1 / p for p in ("src/aicup/sgf.py", "src/aicup/pilot.py", "src/aicup/humansl.py", "src/aicup/l1b.py",
                                    "scripts/trunk_extract.py", "repo/KataGo/python/katago/game/board.py",
                                    "repo/KataGo/python/katago/game/gamestate.py", "repo/KataGo/python/katago/game/features.py",
                                    "repo/KataGo/python/katago/train/model_pytorch.py")]
    body = {"data_id": m["id"], "code": [source_record(path) for path in files], "p0": m["checkpoint"],
            "arms": ["U", "G", "L", "A"], "seeds": [17, 29], "steps": 4000,
            "rank_gate": "both seeds: L-G, L-U and L-permutation paired rank CI lower bound > 0",
            "scope": "100 fresh adapter test players, unknown humanv0 pretraining overlap"}
    body["id"] = stable_hash(json.dumps(body, sort_keys=True))[:12]
    path = DEST / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != body:
        raise ValueError("protocol changed after registration")
    if not path.exists():
        atomic_json(path, body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run"))
    stage = parser.parse_args().stage
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use production Slurm")
    torch.set_num_threads(1)
    check(RUN, LEDGER)
    m = read_data()
    start = time.monotonic()
    if stage == "smoke":
        data, norm = extract({**m, "games": m["games"][:32]})
        if not all(np.isfinite(v).all() for v in norm.values()):
            raise ValueError("invalid normalization")
        stats = {"train_games": 32, "data_id": m["id"], "key_width": data["local_key"].size(-1),
                 "action_width": data["scale", True].numel(), "seconds": time.monotonic() - start,
                 "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
        atomic_json(DEST / "smoke.json", stats)
        print(stats, flush=True)
        return
    import pid_local_choice_fit as fit
    registration = protocol(m)
    if (DEST / "report.json").exists():
        raise FileExistsError("a test report already exists")
    print(f"registered {registration['id']} for data {m['id']}", flush=True)
    data, norm = extract(m)
    np.savez(DEST / "normalizer.npz", **norm)
    extraction = {"seconds": time.monotonic() - start, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    report = fit.experiment(data, m["games"], DEST, registration)
    if protocol(m)["id"] != registration["id"]:
        raise ValueError("source changed during run")
    report.update(extraction=extraction, seconds=time.monotonic() - start, data_audit=m["audit"],
                  job_id=os.environ["SLURM_JOB_ID"])
    check(RUN, LEDGER)
    atomic_json(DEST / "report.json", report)
    print(f"saved report for {registration['id']} with verdict {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()
