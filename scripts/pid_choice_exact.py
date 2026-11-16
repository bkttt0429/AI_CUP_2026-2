#!/usr/bin/env python
"""Distinct exact-score follow-up; never relabels the failed Taylor audit."""
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
from aicup_pid.choice_exact import exact_scores
from aicup_pid.choice_signal import context_features, exact_delta, permute_context
from pid_choice_manifest import atomic_json, digest_file, read_manifest, seed_split, source_record, stable_hash, validate_projection, verify_source
from pid_choice_extract import forward_choices, load_moments, te, valid_shard
from pid_choice_data_audit import replay_choices_checked
from pid_choice_audit import fit_reference, load_calibration
from aicup.humansl import load_humansl

CONFIGS = [(arm, ridge) for arm in ("A1", "A2", "A2perm") for ridge in (1., 10.)]


def prepare_protocol(run_dir):
    root = Path(run_dir)
    m = read_manifest(root)
    if (root / "exact_protocol.json").exists():
        return read_protocol(root)
    sources = [source_record(ROOT / p) for p in ("scripts/pid_choice_exact.py", "src/aicup_pid/choice_exact.py", "scripts/pid_choice_eval.py", "src/aicup_pid/choice_eval.py", "scripts/pid_choice_audit.py", "scripts/pid_choice_data_audit.py")]
    sources += [source_record(root / p) for p in ("audit_reference_pub2_s2.json", "audit_reference_priv2_s2.json", "audit_calibration.json", "normalizer.npz", "audit_input.json", "baseline_check.json")]
    p = {"run_id": m["run_id"], "method": "exact conditional score; quadratic reference parameter estimator", "configs": CONFIGS,
         "sources": sources, "alphas": m["settings"]["alphas"], "revision_basis": "reference-only numerical failure; no new-method heldout identification metrics read",
         "contrast_limit": "A1-A0 combines conditional geometry and exact-normalizer effects; not a pure covariance contrast"}
    p["protocol_id"] = stable_hash(json.dumps(p, sort_keys=True))[:12]
    atomic_json(root / "exact_protocol.json", p)
    return p


def read_protocol(run_dir):
    p = json.loads((Path(run_dir) / "exact_protocol.json").read_text())
    body = {k: v for k, v in p.items() if k != "protocol_id"}
    if stable_hash(json.dumps(body, sort_keys=True))[:12] != p["protocol_id"]:
        raise ValueError("exact protocol modified")
    for source in p["sources"]:
        verify_source(source)
    if p["configs"] != [list(c) for c in CONFIGS]:
        raise ValueError("exact config mismatch")
    return p


def aggregate_evidence(values, questions, index):
    if any(k not in values for q in questions for k in q):
        raise ValueError("missing exact question evidence")
    return np.stack([sum(values[k][index] for k in q) for q in questions])


def _replay(g):
    return replay_choices_checked(g["key"], g["sgf"], g["color"])


def _write(path, arrays, metadata):
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("incomplete/different exact shard exists")
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp, "xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(path.with_suffix(".json"), {"metadata": metadata, "sha256": digest_file(path)})


def extract(run_dir, pool, seeds=(2, 3), pilot=False, workers=10):
    started = time.monotonic()
    root = Path(run_dir)
    m, protocol = read_manifest(root), read_protocol(root)
    if protocol["run_id"] != m["run_id"]:
        raise ValueError("exact protocol/run mismatch")
    for gate in ("audit_input.json", "baseline_check.json"):
        g = json.loads((root / gate).read_text())
        if g["run_id"] != m["run_id"] or g["verdict"] != "passed":
            raise ValueError("input/baseline prerequisite failed")
    scale, _, ridges = load_calibration(root)
    if ridges != [1., 10.]:
        raise ValueError("frozen exact follow-up grid differs")
    moments, _ = load_moments(root, m, pool)
    splits = {s: seed_split(m, pool, s) for s in seeds}
    owners = sorted(splits[seeds[0]]["reference"])
    held = {s: {k for ks in split["heldout"].values() for k in ks} for s, split in splits.items()}
    z = {}
    for s, split in splits.items():
        if held[s].intersection(k for ks in split["reference"].values() for k in ks):
            raise ValueError("exact parameter reference leakage")
        for arm, ridge in CONFIGS:
            zz = fit_reference(moments, split["reference"], arm, scale, ridge)
            z[s, arm, ridge] = torch.as_tensor(np.stack([zz[p] for p in owners]), device="cuda", dtype=torch.float64)
    del moments
    games = [g for g in m["games"] if g["pool"] == pool and any(g["key"] in h for h in held.values())]
    projection = validate_projection(m["assets"]["projection"]["path"])
    checkpoint = m["assets"]["checkpoint"]["path"]
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    te.init_worker(config)
    model, _ = load_humansl(checkpoint, "cuda")
    model.eval(); torch.set_num_threads(1)
    output = root / "exact_scores"
    output.mkdir(exist_ok=True)
    group = "_".join(map(str, seeds))
    audit_path = root / f"exact_arithmetic_{pool}_s{group}.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"protocol_id": protocol["protocol_id"], "checks": {}}
    if audit["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("arithmetic audit protocol mismatch")
    parts = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=te.init_worker, initargs=(config,)) as executor:
        for part, start in enumerate(range(0, len(games), 64)):
            if pilot and part >= 1:
                break
            gs = games[start:start + 64]
            path = output / f"{pool}_s{group}_p{part:04d}.npz"
            metadata = {"protocol_id": protocol["protocol_id"], "pool": pool, "seeds": list(seeds), "part": part, "owners": owners, "configs": [list(c) for c in CONFIGS]}
            if valid_shard(path, metadata):
                continue
            tick = time.monotonic()
            replays = list(executor.map(_replay, gs))
            replay_seconds = time.monotonic() - tick
            values = {s: [] for s in seeds}
            keys = {s: [] for s in seeds}
            for g, replay in zip(gs, replays):
                x = forward_choices(model, replay, projection)
                psi = x["psi"].double() / torch.as_tensor(scale, device="cuda")
                lp = x["logp"].double()
                lp = lp - torch.logsumexp(lp, 1)[:, None]
                played = torch.as_tensor(replay["played"], device="cuda").long()
                perm = permute_context(x["value"].cpu().numpy(), replay["t"], pool=pool, key=g["original_key"])
                phis = {"A1": psi, "A2": context_features(psi, x["value"]), "A2perm": context_features(psi, perm)}
                for s in seeds:
                    if g["key"] not in held[s]:
                        continue
                    scores = []
                    for arm, ridge in CONFIGS:
                        zz, phi = z[s, arm, ridge], phis[arm]
                        score = exact_scores(lp, phi, played, zz)
                        name = f"s{s}_{arm}_{ridge}"
                        if name not in audit["checks"]:
                            ix = sorted(set(list(range(8)) + torch.topk(torch.linalg.vector_norm(zz, dim=1), 8).indices.tolist()))
                            ref = torch.stack([exact_delta(lp, phi, played, zz[i]).sum() for i in ix])
                            error = float((score[ix] - ref).abs().max())
                            if error > 1e-8:
                                raise ValueError(f"exact arithmetic check failed: {error}")
                            audit["checks"][name] = {"game": g["key"], "candidates": ix, "max_abs_error": error}
                        scores.append(score.cpu().numpy())
                    values[s].append(np.stack(scores)); keys[s].append(g["key"])
            arrays = {}
            for s in seeds:
                arrays[f"keys_s{s}"] = np.array(keys[s], dtype="U100")
                arrays[f"scores_s{s}"] = np.stack(values[s]) if values[s] else np.empty((0, len(CONFIGS), len(owners)))
            used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if used + sum(a.nbytes for a in arrays.values()) > m["settings"]["max_bytes"]:
                raise RuntimeError("exact evidence exceeds disk budget")
            _write(path, arrays, metadata)
            audit["verdict"] = "passed" if len(audit["checks"]) == len(seeds) * len(CONFIGS) else "incomplete"
            atomic_json(audit_path, audit)
            row = {"part": part, "games": len(gs), "seconds": time.monotonic() - tick, "replay_seconds": replay_seconds, "bytes": path.stat().st_size}
            parts.append(row); print(json.dumps(row), flush=True)
    report = {"protocol_id": protocol["protocol_id"], "pool": pool, "seeds": list(seeds), "pilot": pilot, "union_games": len(games), "parts": parts,
              "elapsed_seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "job": os.environ["SLURM_JOB_ID"]}
    atomic_json(root / f"exact_{'pilot' if pilot else 'complete'}_{pool}_s{group}.json", report)
    return report


def load_evidence(run_dir, manifest, pool, seed, owners):
    root = Path(run_dir)
    protocol = read_protocol(root)
    group = "2_3" if seed in (2, 3) else "4_5"
    audit = json.loads((root / f"exact_arithmetic_{pool}_s{group}.json").read_text())
    if audit["protocol_id"] != protocol["protocol_id"] or audit["verdict"] != "passed":
        raise ValueError("exact arithmetic gate not passed")
    if protocol["run_id"] != manifest["run_id"]:
        raise ValueError("exact evidence run mismatch")
    values = {}
    for path in sorted((root / "exact_scores").glob(f"{pool}_s{group}_p*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
        if meta["protocol_id"] != protocol["protocol_id"] or meta["pool"] != pool or meta["seeds"] != list(map(int, group.split("_"))) or meta["owners"] != owners or meta["configs"] != [list(c) for c in CONFIGS] or not valid_shard(path, meta):
            raise ValueError("exact evidence provenance mismatch")
        with np.load(path, allow_pickle=False) as a:
            keys, scores = a[f"keys_s{seed}"].tolist(), a[f"scores_s{seed}"]
            if scores.shape != (len(keys), len(CONFIGS), len(owners)) or not np.isfinite(scores).all():
                raise ValueError("invalid exact evidence shape/values")
            for k, v in zip(keys, scores):
                if k in values:
                    raise ValueError("duplicate exact key")
                values[k] = v
    expected = {k for ks in seed_split(manifest, pool, seed)["heldout"].values() for k in ks}
    if set(values) != expected:
        raise ValueError("incomplete or non-heldout exact evidence")
    return values


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=("protocol", "pilot", "extract"), required=True)
    p.add_argument("--pool", choices=("pub2", "priv2"))
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use Slurm")
    if a.seeds not in ([2, 3], [4, 5]):
        raise ValueError("follow-up seeds must be 2/3 or 4/5")
    print(json.dumps(prepare_protocol(a.run_dir) if a.stage == "protocol" else extract(a.run_dir, a.pool, tuple(a.seeds), pilot=a.stage == "pilot")), flush=True)
