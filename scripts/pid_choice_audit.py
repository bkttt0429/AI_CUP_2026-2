#!/usr/bin/env python
"""Numerical gates for a second-order choice model; no query labels used."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aicup_pid.choice_signal import context_features, fit_candidate, normalise_moments, permute_context, validate_moments
from pid_choice_manifest import atomic_json, digest_file, read_manifest, seed_split, stable_hash, validate_projection
from pid_choice_extract import forward_choices, load_moments, replay_choices, te
from aicup.humansl import load_humansl


def position_scores(logp, phi, played, z):
    """Per-position exact and Taylor LLR; one coefficient vector per position."""
    tilt = torch.einsum("tad,td->ta", phi, z.expand(len(phi), -1))
    actual = tilt.gather(1, played.long()[:, None])[:, 0]
    prob = logp.exp()
    mean = (prob * tilt).sum(1)
    var = (prob * (tilt - mean[:, None]).square()).sum(1)
    exact = actual - torch.logsumexp(logp + tilt, 1)
    approx = actual - mean - .5 * var
    return exact, approx


def error_gate(errors):
    x = np.asarray(errors)
    finite = bool(x.size and np.isfinite(x).all())
    median, p95 = np.quantile(x, [.5, .95]).tolist() if finite else (None, None)
    return {"positions": int(x.size), "median_abs_nat": median, "p95_abs_nat": p95,
            "passed": finite and median <= .02 and p95 <= .10}


def rank_gate(exact, approx, target):
    exact, approx = np.asarray(exact), np.asarray(approx)
    sd = exact.std(1)
    target = np.asarray(target)
    if not np.isfinite(exact).all() or not np.isfinite(approx).all():
        return {"passed": False, "eligible_pairs": 0, "reason": "nonfinite"}
    d = exact[np.arange(len(exact)), target, None] - exact
    da = approx[np.arange(len(approx)), target, None] - approx
    eligible = (sd[:, None] >= 1e-12) & (np.abs(d) / np.maximum(sd[:, None], 1e-12) >= .05)
    n = int(eligible.sum())
    agreement = float((np.sign(d[eligible]) == np.sign(da[eligible])).mean()) if n else None
    return {"eligible_pairs": n, "sign_agreement": agreement, "uninformative_blocks": int((sd < 1e-12).sum()),
            "passed": n >= 100 and agreement >= .95}


def fit_reference(moments, references, arm, scale, ridge):
    suffix = {"A1": "0", "A2": "2", "A2perm": "2perm"}[arm]
    d = scale if suffix == "0" else np.tile(scale, 2)
    z = {}
    for p, keys in references.items():
        U = sum(moments[k]["U" + suffix].astype(np.float64) for k in keys)
        F = sum(moments[k]["F" + suffix].astype(np.float64) for k in keys)
        n = sum(int(moments[k]["n"]) for k in keys)
        u, f = normalise_moments(U, F, d)
        z[p] = fit_candidate(u, f, n, ridge)
    return z


def calibration(run_dir):
    m = read_manifest(run_dir)
    moments, audit = load_moments(run_dir, m, "calibration")
    games = [g for g in m["games"] if g["pool"] == "calibration"]
    for row in moments.values():
        for suffix in ("0", "2", "2perm"):
            validate_moments(row["U" + suffix], row["F" + suffix])
    total_f = sum(r["F0"].astype(np.float64) for r in moments.values())
    total_n = sum(int(r["n"]) for r in moments.values())
    scale = np.sqrt(np.maximum(np.diag(total_f) / total_n, 1e-6))
    fbar = (total_f / total_n) / (scale[:, None] * scale[None, :])
    ref = {}
    for g in games:
        if g["role"] == "reference":
            ref.setdefault(g["owner"], []).append(g["key"])
    owners = {g["key"]: g["owner"] for g in games}
    held = {g["key"] for g in games if g["role"] == "heldout"}
    if len(audit["key"]) != 1024 or not set(audit["key"].tolist()) <= held:
        raise ValueError("calibration audit must contain exactly 1024 held-out positions")
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = torch.as_tensor(audit["lp"], device=device, dtype=torch.float64)
    psi = torch.as_tensor(audit["psi"] / scale, device=device, dtype=torch.float64)
    played = torch.as_tensor(audit["played"], device=device).long()
    report = {"run_id": m["run_id"], "stage": "calibration", "arms": {}, "accepted_ridges": []}
    for ridge in m["settings"]["ridges"]:
        all_pass = True
        for arm in ("A1", "A2", "A2perm"):
            zs = fit_reference(moments, ref, arm, scale, ridge)
            z = torch.as_tensor(np.stack([zs[owners[k]] for k in audit["key"]]), device=device)
            phi = psi if arm == "A1" else context_features(psi, audit["v"] if arm == "A2" else audit["vperm"])
            exact, approx = position_scores(lp, phi, played, z)
            result = error_gate((exact - approx).abs().cpu().numpy())
            report["arms"].setdefault(arm, {})[str(ridge)] = result
            all_pass &= result["passed"]
        if all_pass:
            report["accepted_ridges"].append(ridge)
    report["verdict"] = "passed" if report["accepted_ridges"] else "approximation_invalid"
    path = Path(run_dir) / "normalizer.npz"
    with open(path.with_suffix(".tmp"), "wb") as f:
        np.savez(f, scale=scale, fbar=fbar, n=total_n)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path.with_suffix(".tmp"), path)
    report["normalizer_sha256"] = digest_file(path)
    report["code_sha256"] = digest_file(__file__)
    atomic_json(Path(run_dir) / "audit_calibration.json", report)
    print(json.dumps(report), flush=True)
    return report


def load_calibration(run_dir):
    report = json.loads((Path(run_dir) / "audit_calibration.json").read_text())
    if report["verdict"] != "passed" or digest_file(Path(run_dir) / "normalizer.npz") != report["normalizer_sha256"]:
        raise ValueError("calibration gate not passed or normalizer modified")
    with np.load(Path(run_dir) / "normalizer.npz") as z:
        return z["scale"], z["fbar"], report["accepted_ridges"]


def reference(run_dir, pool, seed):
    from pid_choice_data_audit import replay_choices_checked
    start = time.monotonic()
    m = read_manifest(run_dir)
    scale, _, ridges = load_calibration(run_dir)
    moments, _ = load_moments(run_dir, m, pool)
    references = seed_split(m, pool, seed)["reference"]
    allowed = {k for ks in references.values() for k in ks}
    gs = {g["key"]: g for g in m["games"] if g["key"] in allowed}
    projection = validate_projection(m["assets"]["projection"]["path"])
    checkpoint = m["assets"]["checkpoint"]["path"]
    te.init_worker(torch.load(checkpoint, map_location="cpu", weights_only=True)["config"])
    model, _ = load_humansl(checkpoint, "cuda")
    model.eval()
    torch.set_num_threads(1)
    cache = {}
    report = {"pool": pool, "seed": seed, "run_id": m["run_id"], "arms": {}, "read_keys": []}

    def read_block(k):
        if k not in allowed:
            raise ValueError("audit attempted to read an outer-held game")
        if k not in cache:
            g = gs[k]
            r = replay_choices_checked(k, g["sgf"], g["color"])
            x = forward_choices(model, r, projection)
            perm = permute_context(x["value"].cpu().numpy(), r["t"], pool=pool, key=g["original_key"])
            ix = sorted(range(len(r["t"])), key=lambda i: stable_hash(f"choice-refaudit-ply-v1|{k}|{r['t'][i]}"))[:8]
            if len(ix) < 8:
                raise ValueError("reference audit block has fewer than 8 own moves")
            cache[k] = {"lp": x["logp"][ix].cpu().numpy(), "psi": x["psi"][ix].cpu().numpy(), "v": x["value"][ix].cpu().numpy(),
                        "vperm": perm[ix], "played": r["played"][ix]}
        return cache[k]

    all_pass, inconclusive = True, False
    for ridge in ridges:
        for arm in ("A1", "A2", "A2perm"):
            z = fit_reference(moments, references, arm, scale, ridge)
            biggest = sorted(z, key=lambda p: (-float(np.linalg.norm(z[p])), p))[:16]
            hashed = sorted((p for p in z if p not in biggest), key=lambda p: stable_hash(f"choice-refaudit-player-v1|{pool}|{p}"))[:16]
            chosen = biggest + hashed
            blocks, targets, block_keys = [], [], []
            for ci, owner in enumerate(chosen):
                own_key = min(references[owner], key=lambda k: stable_hash(k))
                other_owner = min((p for p in references if p != owner), key=lambda p: stable_hash(p))
                other_key = min(references[other_owner], key=lambda k: stable_hash(k))
                for k in (own_key, other_key):
                    blocks.append(read_block(k)); targets.append(ci); block_keys.append(k)
            cat = lambda k: np.concatenate([b[k] for b in blocks])
            lp = torch.as_tensor(cat("lp"), device="cuda", dtype=torch.float64)
            psi = torch.as_tensor(cat("psi") / scale, device="cuda", dtype=torch.float64)
            phi = psi if arm == "A1" else context_features(psi, cat("v") if arm == "A2" else cat("vperm"))
            played = torch.as_tensor(cat("played"), device="cuda").long()
            exacts, approxs = [], []
            for owner in chosen:
                exact, approx = position_scores(lp, phi, played, torch.as_tensor(z[owner][None], device="cuda"))
                exacts.append(exact.cpu().numpy()); approxs.append(approx.cpu().numpy())
            exact, approx = np.stack(exacts, 1), np.stack(approxs, 1)
            e = error_gate(np.abs(exact - approx))
            r = rank_gate(exact.reshape(-1, 8, 32).sum(1), approx.reshape(-1, 8, 32).sum(1), targets)
            passed = e["passed"] and r["passed"]
            all_pass &= passed
            inconclusive |= r["eligible_pairs"] < 100
            report["arms"].setdefault(arm, {})[str(ridge)] = {"errors": e, "ranking": r, "passed": passed, "players": chosen, "blocks": block_keys}
            print(f"reference {pool} seed {seed} {arm} ridge={ridge}: {e}, {r}", flush=True)
    report.update(verdict="passed" if all_pass else "approximation_inconclusive" if inconclusive else "approximation_invalid",
                  read_keys=sorted(cache), elapsed_seconds=time.monotonic() - start, code_sha256=digest_file(__file__))
    atomic_json(Path(run_dir) / f"audit_reference_{pool}_s{seed}.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=("calibration", "reference"), required=True)
    p.add_argument("--pool", choices=("pub2", "priv2"))
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use Slurm")
    result = calibration(a.run_dir) if a.stage == "calibration" else reference(a.run_dir, a.pool, a.seed)
    if result["verdict"] != "passed":
        raise SystemExit(2)
