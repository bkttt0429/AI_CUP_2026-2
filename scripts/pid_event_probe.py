#!/usr/bin/env python
"""Extract and falsify an opponent-event response identity hypothesis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aicup_pid.event_response import adjust_record, evaluate_repeatability, fit_nuisance, response_moments
from pid_choice_manifest import atomic_json, digest_file, read_manifest, source_record, stable_hash, validate_projection, verify_source
from pid_choice_extract import forward_choices, te, valid_shard
from pid_choice_data_audit import replay_choices_checked
from pid_choice_exact import _write
from pid_choice_runtime import check
from aicup.humansl import load_humansl

FIELDS = ("t", "mover", "residual", "value", "entropy", "surprisal")


def protocol(root, create=False):
    root = Path(root)
    m = read_manifest(root)
    path = root / "events/protocol.json"
    if not path.exists():
        if not create:
            raise ValueError("event protocol not frozen")
        p = {"parent_run": m["run_id"], "hypothesis": "opponent-surprise-aligned retrospective changes have repeatable identity",
             "lags": [1, 2, 4, 8], "common_support": "eligible at lag8 before matched-cell centering and one shared permutation",
             "primary": "E", "negative_control": "P", "calibration_only": True,
             "sources": [source_record(ROOT / s) for s in ("scripts/pid_event_probe.py", "src/aicup_pid/event_response.py", "docs/superpowers/plans/2026-11-16-event-response.md", "scripts/pid_choice_data_audit.py", "scripts/pid_choice_exact.py")],
             "nuisance_reference_only": True, "gate": {"coverage20": .95, "permutable": .6, "E_minus_P_CI_lower": 0, "E_identity_margin_CI_lower": 0}}
        p["id"] = stable_hash(json.dumps(p, sort_keys=True))[:12]
        atomic_json(path, p)
    p = json.loads(path.read_text())
    if p["parent_run"] != m["run_id"] or p["id"] != stable_hash(json.dumps({k: v for k, v in p.items() if k != "id"}, sort_keys=True))[:12]:
        raise ValueError("event protocol mismatch")
    for s in p["sources"]:
        verify_source(s)
    return m, p


def replay_both(g):
    halves = [replay_choices_checked(g["key"], g["sgf"], c) for c in ("B", "W")]
    out = {k: np.concatenate([h[k] for h in halves]) for k in halves[0] if k != "key"}
    order = np.argsort(out["t"])
    return {"key": g["key"], **{k: v[order] for k, v in out.items()}}


def extract(root, pilot=False):
    started = time.monotonic()
    root = Path(root)
    m, p = protocol(root)
    games = [g for g in m["games"] if g["pool"] == "calibration"]
    ckpt = m["assets"]["checkpoint"]["path"]
    config = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    project = validate_projection(m["assets"]["projection"]["path"])
    model, _ = load_humansl(ckpt, "cuda")
    model.eval(); torch.set_num_threads(1)
    dest = root / "events/raw"
    dest.mkdir(parents=True, exist_ok=True)
    measured = []
    with ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context("spawn"), initializer=te.init_worker, initargs=(config,)) as ex:
        for part, i in enumerate(range(0, len(games), 64)):
            if pilot and part > 0:
                break
            gs = games[i:i + 64]
            meta = {"protocol": p["id"], "part": part, "keys": [g["key"] for g in gs]}
            path = dest / f"calibration_p{part:04d}.npz"
            if valid_shard(path, meta):
                continue
            tick = time.monotonic()
            replays = list(ex.map(replay_both, gs))
            records = []
            for replay in replays:
                x = forward_choices(model, replay, project)
                lp = x["logp"].double(); lp = lp - torch.logsumexp(lp, 1)[:, None]
                prob = lp.exp()
                played = torch.as_tensor(replay["played"], device="cuda").long()
                psi = x["psi"].double()
                residual = psi[torch.arange(len(psi), device="cuda"), played] - (prob[:, :, None] * psi).sum(1)
                surprisal = -lp.gather(1, played[:, None])[:, 0]
                entropy = -(prob * torch.where(torch.isfinite(lp), lp, 0.)).sum(1)
                records.append({"t": replay["t"], "mover": replay["mover"], "residual": residual.cpu().numpy().astype(np.float32),
                                "value": x["value"].cpu().numpy(), "entropy": entropy.cpu().numpy().astype(np.float32), "surprisal": surprisal.cpu().numpy().astype(np.float32)})
            arrays = {f: np.concatenate([r[f] for r in records]) for f in FIELDS}
            arrays.update(keys=np.array([g["key"] for g in gs]), n=np.array([len(r["t"]) for r in records]), subject=np.array([int(g["color"] == "W") for g in gs]))
            if any(not np.isfinite(v).all() for k, v in arrays.items() if k != "keys"):
                raise ValueError("nonfinite event raw record")
            used = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
            if used + sum(a.nbytes for a in arrays.values()) > 1024 ** 3:
                raise RuntimeError("artifact budget exceeded")
            _write(path, arrays, meta)
            item = {"part": part, "games": len(gs), "plies": int(arrays["n"].sum()), "seconds": time.monotonic() - tick, "bytes": path.stat().st_size}
            measured.append(item); print(json.dumps(item), flush=True)
    report = {"protocol": p["id"], "parts": measured, "elapsed_seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "job_id": os.environ["SLURM_JOB_ID"]}
    atomic_json(root / f"events/{'pilot' if pilot else 'extract'}.json", report)
    return report


def load_raw(root, m, p):
    gs = [g for g in m["games"] if g["pool"] == "calibration"]
    expected = {g["key"] for g in gs}
    records = {}
    for part, i in enumerate(range(0, len(gs), 64)):
        path = Path(root) / f"events/raw/calibration_p{part:04d}.npz"
        meta = {"protocol": p["id"], "part": part, "keys": [g["key"] for g in gs[i:i + 64]]}
        if not valid_shard(path, meta):
            raise ValueError("event shard missing/corrupted/wrong provenance")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        start = 0
        for key, n, subject in zip(arrays["keys"].tolist(), arrays["n"], arrays["subject"]):
            if key in records:
                raise ValueError("duplicate event key")
            records[key] = {f: arrays[f][start:start + n] for f in FIELDS} | {"subject": int(subject)}
            start += n
        if start != len(arrays["t"]):
            raise ValueError("event shard length mismatch")
    if set(records) != expected:
        raise ValueError("incomplete event coverage")
    return records


def study(root):
    root = Path(root)
    m, p = protocol(root)
    raw = load_raw(root, m, p)
    games = [g for g in m["games"] if g["pool"] == "calibration"]
    refs = [g["key"] for g in games if g["role"] == "reference"]
    fit = fit_nuisance([raw[k] for k in refs])
    np.savez(root / "events/nuisance.npz", **fit)
    features = {k: response_moments(adjust_record(a, fit), k, float(fit["entropy_median"])) for k, a in raw.items()}
    result = evaluate_repeatability(games, features)
    result.update(protocol=p["id"], games=len(games), players=len({g["owner"] for g in games}), nuisance_reference_games=len(refs),
                  nuisance_sha256=digest_file(root / "events/nuisance.npz"), raw_sources=[source_record(f) for f in sorted((root / "events/raw").glob("*.npz"))])
    keys = sorted(features)
    np.savez_compressed(root / "events/moments.npz", keys=np.array(keys), **{k: np.stack([features[g][k] for g in keys]) for k in features[keys[0]]})
    atomic_json(root / "events/calibration_report.json", result)
    atomic_json(ROOT / "results/final/event_response_calibration_20261116.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True, choices=("protocol", "pilot", "extract", "study"))
    a = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use production Slurm")
    check(a.run_dir, ROOT / ".superpowers/sdd/2026-11-16-choice-signal/jobs.txt")
    out = protocol(a.run_dir, True)[1] if a.stage == "protocol" else study(a.run_dir) if a.stage == "study" else extract(a.run_dir, a.stage == "pilot")
    print(json.dumps(out), flush=True)
    check(a.run_dir, ROOT / ".superpowers/sdd/2026-11-16-choice-signal/jobs.txt")
