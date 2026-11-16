#!/usr/bin/env python
"""Bounded reference-conditioned policy mechanism experiment."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from aicup_pid.paths import AICUP1, TRAIN_DIR, FOLDS_CSV
from aicup.sgf import setup_and_moves, handicap
from aicup.humansl import load_humansl
from aicup.l1b import _GraphExtraOutputs, neutral_rows
from aicup_pid.reference_policy import ReferencePolicy, episodic_loss, played_scores, owner_metrics, interval
from pid_choice_manifest import atomic_json, source_record, verify_source, stable_hash, read_manifest
from pid_choice_data_audit import replay_choices_checked
from pid_choice_extract import te
from pid_choice_runtime import check

LEDGER = ROOT / ".superpowers/sdd/2026-11-16-choice-signal/jobs.txt"
COUNTS = {"train": 200, "dev": 50, "test": 50}


def position_indices(n, count=32):
    if n < count:
        raise ValueError("too few subject moves")
    return np.linspace(0, n - 1, count, dtype=int)


def physical_signature(g):
    stones, moves = setup_and_moves(g["sgf"])
    return stable_hash(repr((sorted(stones), moves)))


def select_games(rows, folds, counts=None):
    counts = COUNTS if counts is None else counts
    by = defaultdict(list)
    for g in rows:
        by[g["owner"]].append(g)
    result = []
    for split, allowed in (("train", (1, 2)), ("dev", (3,)), ("test", (4,))):
        owners = sorted((p for p, gs in by.items() if folds.get(p) in allowed and len(gs) >= 10), key=lambda p: stable_hash("refpolicy-owner-v1|" + p))
        if len(owners) < counts[split]:
            raise ValueError("insufficient eligible players")
        for p in owners[:counts[split]]:
            games = sorted(by[p], key=lambda g: stable_hash("refpolicy-game-v1|" + g["key"]))[:10]
            result.extend(dict(g, split=split, slot=i) for i, g in enumerate(games))
    return result


def validate_games(games, require_counts=True):
    seen, owners = set(), {}
    for g in games:
        sig = physical_signature(g)
        if sig in seen:
            raise ValueError("duplicate physical game")
        seen.add(sig)
        if g["owner"] in owners and owners[g["owner"]] != g["split"]:
            raise ValueError("owner crosses split")
        owners[g["owner"]] = g["split"]
    if require_counts:
        for split, n in COUNTS.items():
            rows = [g for g in games if g["split"] == split]
            if len(rows) != 10 * n or len({g["owner"] for g in rows}) != n:
                raise ValueError("wrong split size")
            for p in {g["owner"] for g in rows}:
                if sorted(g["slot"] for g in rows if g["owner"] == p) != list(range(10)):
                    raise ValueError("invalid game slots")


def prepare(root):
    parent = read_manifest(root)
    dest = Path(root) / "refpolicy"; dest.mkdir(exist_ok=True)
    path = dest / "manifest.json"
    if path.exists():
        return read_protocol(root)
    table = list(csv.DictReader(open(FOLDS_CSV)))
    folds = {r["player_id"]: int(r["fold"]) for r in table}
    allowed = set()
    for split, fs in (("train", (1, 2)), ("dev", (3,)), ("test", (4,))):
        ps = sorted((r["player_id"] for r in table if int(r["fold"]) in fs and int(r["n_games"]) >= 10), key=lambda p: stable_hash("refpolicy-owner-v1|" + p))
        allowed.update(ps[:COUNTS[split] * 2])
    by = defaultdict(list); sources = [source_record(FOLDS_CSV)]
    for path in sorted(TRAIN_DIR.glob("train_*.csv")):
        sources.append(source_record(path))
        for r in csv.DictReader(open(path)):
            p = r["player_id"]
            if p not in allowed:
                continue
            _, moves = setup_and_moves(r["sgf_content"])
            if sum(c == r["color"] for c, _, _ in moves[:256]) < 32:
                continue
            g = {"owner": p, "key": r["rank"] + ":" + r["game_id"], "color": r["color"], "sgf": r["sgf_content"], "fold": folds[p]}
            by[p].append(g)
            if len(by[p]) > 20:
                by[p].sort(key=lambda x: stable_hash("refpolicy-game-v1|" + x["key"]))
                by[p].pop()
    # Deterministic global deduplication before selecting the final ten games.
    rows, seen = [], set()
    for p in sorted(by, key=lambda p: stable_hash("refpolicy-owner-v1|" + p)):
        for g in sorted(by[p], key=lambda x: stable_hash("refpolicy-game-v1|" + x["key"])):
            sig = physical_signature(g)
            if sig not in seen:
                rows.append(g); seen.add(sig)
    games = select_games(rows, folds); validate_games(games)
    m = {"games": games, "sources": sources, "checkpoint": parent["assets"]["checkpoint"], "parent_run": parent["run_id"],
         "dependency_code": parent["code"] + [source_record(AICUP1 / "src/aicup/sgf.py")],
         "code": [source_record(ROOT / p) for p in ("scripts/pid_reference_policy.py", "src/aicup_pid/reference_policy.py", "docs/superpowers/plans/2026-11-16-reference-policy.md", "scripts/pid_choice_data_audit.py", "scripts/pid_choice_extract.py")],
         "settings": {"counts": COUNTS, "positions": 32, "seeds": [17, 29], "steps": 4000, "arms": ["attention", "uniform"], "hidden": 64}}
    m["id"] = stable_hash(json.dumps(m, sort_keys=True))[:12]
    atomic_json(dest / "manifest.json", m)
    return m


def read_protocol(root):
    m = json.loads((Path(root) / "refpolicy/manifest.json").read_text())
    if stable_hash(json.dumps({k: v for k, v in m.items() if k != "id"}, sort_keys=True))[:12] != m["id"]:
        raise ValueError("modified reference policy manifest")
    for s in [*m["sources"], *m["code"], *m["dependency_code"], m["checkpoint"]]:
        verify_source(s)
    validate_games(m["games"])
    return m


def replay(g):
    r = replay_choices_checked(g["key"], g["sgf"], g["color"])
    ix = position_indices(len(r["t"]))
    return {k: v if k == "key" else v[ix] for k, v in r.items()}


def extract(root, m, pilot=False):
    start = time.monotonic()
    games = m["games"][:64] if pilot else m["games"]
    n = len(games)
    # ~26.8 GB for 3000 games; transient RAM only, never written to disk.
    arrays = {"action": np.empty((n, 32, 362, 385), np.float16), "context": np.empty((n, 32, 388), np.float32),
              "value": np.empty((n, 32, 385), np.float32), "base": np.empty((n, 32, 362), np.float32), "played": np.empty((n, 32), np.int64)}
    ckpt = m["checkpoint"]["path"]
    config = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    backbone, _ = load_humansl(ckpt, "cuda"); backbone.eval()
    rows = torch.as_tensor(neutral_rows(), device="cuda", dtype=torch.float32)
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"), initializer=te.init_worker, initargs=(config,)) as pool:
        for start_i in range(0, n, 32):
            for j, r in enumerate(pool.map(replay, games[start_i:start_i + 32])):
                i = start_i + j
                b = torch.as_tensor(r["bins"], device="cuda").float(); g = torch.as_tensor(r["globs"], device="cuda").float()
                side = torch.as_tensor(r["mover"], device="cuda").long()
                legal = torch.as_tensor(r["legal"], device="cuda")
                extra = _GraphExtraOutputs(["trunkfinal"])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = backbone(b, g, input_meta=rows[side] if backbone.get_has_metadata_encoder() else None, extra_outputs=extra)
                    post = backbone.postprocess_output(outputs)[0]
                trunk = extra.returned["trunkfinal"].float().flatten(2).transpose(1, 2)
                if trunk.shape != (32, 361, 384):
                    raise ValueError("unexpected full trunk shape")
                lp = F.log_softmax(post[0][:, 0].float().masked_fill(~legal, -torch.inf), -1)
                action = torch.zeros(32, 362, 385, device="cuda")
                action[:, :361, :384] = trunk; action[:, 361, 384] = 1
                prob = lp.exp(); played = torch.as_tensor(r["played"], device="cuda").long()
                residual = action[torch.arange(32, device="cuda"), played] - (prob[..., None] * action).sum(1)
                vp = F.softmax(post[1].float(), -1)
                entropy = -(prob * torch.where(torch.isfinite(lp), lp, 0)).sum(1)
                context = torch.cat([trunk.mean(1), (vp[:, 0] - vp[:, 1])[:, None], entropy[:, None], torch.as_tensor(r["t"], device="cuda")[:, None] / 256, side[:, None]], 1)
                arrays["action"][i] = action.cpu().numpy().astype(np.float16)
                arrays["base"][i] = lp.cpu().numpy(); arrays["context"][i] = context.cpu().numpy()
                arrays["value"][i] = residual.cpu().numpy(); arrays["played"][i] = r["played"]
            print(json.dumps({"extract_games": min(start_i + 32, n), "total": n, "elapsed": time.monotonic() - start}), flush=True)
    del backbone; gc.collect(); torch.cuda.empty_cache()
    for key, a in arrays.items():
        if key == "base":
            if np.isnan(a).any() or np.isposinf(a).any():
                raise ValueError("invalid baseline policy")
        elif not np.isfinite(a).all():
            raise ValueError("invalid extracted features")
    report = {"games": n, "seconds": time.monotonic() - start, "ram_bytes": sum(a.nbytes for a in arrays.values()), "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "protocol": m["id"]}
    atomic_json(Path(root) / f"refpolicy/{'pilot' if pilot else 'extract'}.json", report)
    return arrays, report


def normalize(arrays, games):
    train = [i for i, g in enumerate(games) if g["split"] == "train" and g["slot"] < 7]
    c = arrays["context"][train].reshape(-1, 388)
    mu, sd = c.mean(0), np.maximum(c.std(0), 1e-4)
    vrms = np.maximum(np.sqrt(np.mean(arrays["value"][train] ** 2, axis=(0, 1))), 1e-4)
    asum = np.zeros(385, np.float64); count = 0
    for i in train:
        a = arrays["action"][i].astype(np.float32)
        asum += np.square(a).sum((0, 1)); count += a.shape[0] * a.shape[1]
    arms = np.maximum(np.sqrt(asum / count), 1e-4).astype(np.float32)
    data = {k: torch.from_numpy(v).to("cuda") for k, v in arrays.items()}
    data["context"] = (data["context"] - torch.as_tensor(mu, device="cuda")) / torch.as_tensor(sd, device="cuda")
    data["value"] /= torch.as_tensor(vrms, device="cuda")
    # Keep actions FP16; divide by RMS per selected batch to avoid a second full tensor.
    data["action_scale"] = torch.as_tensor(arms, device="cuda")
    return data, {"context_mean": mu, "context_std": sd, "value_rms": vrms, "action_rms": arms}


def split_indices(games, split):
    owners = sorted({g["owner"] for g in games if g["split"] == split})
    by = {p: sorted((i for i, g in enumerate(games) if g["owner"] == p), key=lambda i: games[i]["slot"]) for p in owners}
    return owners, np.array([by[p] for p in owners])


@torch.no_grad()
def evaluate(model, data, index, ablation=None):
    model.eval(); p = len(index)
    ref = torch.as_tensor(index[:, :7], device="cuda")
    rc, rv = data["context"][ref], data["value"][ref]
    if ablation == "zero":
        rv = torch.zeros_like(rv)
    elif ablation == "permuted_keys":
        rng = np.random.default_rng(612)
        order = torch.as_tensor(np.stack([rng.permutation(32) for _ in range(p * 7)]).reshape(p, 7, 32), device="cuda")
        rc = rc.gather(2, order[..., None].expand(-1, -1, -1, 388))
    queries = index[:, 7:].ravel()
    scores, nlls, bases, nulls = [], [], [], []
    for start in range(0, len(queries), 2):
        ids = torch.as_tensor(queries[start:start + 2], device="cuda")
        ctx = data["context"][ids].reshape(-1, 388)
        actions = data["action"][ids].float().reshape(-1, 362, 385) / data["action_scale"]
        base = data["base"][ids].reshape(-1, 362); played = data["played"][ids].ravel()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, null = model(ctx, actions, base, rc, rv)
        s = played_scores(lp, played).reshape(len(ids), 32, p).mean(1)
        b = base.gather(1, played[:, None]).reshape(len(ids), 32).mean(1)
        scores.append((s - b[:, None]).cpu().numpy()); nlls.append((-s).cpu().numpy()); bases.append((-b).cpu().numpy()); nulls.append(null.mean().item())
    score = np.concatenate(scores).reshape(p, 3, p)
    ns = np.concatenate(nlls).reshape(p, 3, p)
    own = ns[np.arange(p), :, np.arange(p)]
    wrong = ns[np.arange(p), :, (np.arange(p) + 1) % p]
    per, metrics = owner_metrics(score)
    logits = torch.from_numpy(score.mean(1)) * 10
    dev_loss = F.cross_entropy(logits, torch.arange(p)).item() + .1 * own.mean()
    metrics.update(move_nll=float(own.mean()), base_nll=float(np.concatenate(bases).mean()), null_mass=float(np.mean(nulls)), objective=float(dev_loss))
    return {"scores": score, "per_owner_rank": per, "move_advantage": (wrong - own).mean(1), "metrics": metrics}


def train_one(data, games, seed, mode, dest, protocol_id):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = ReferencePolicy(388, 385, mode=mode).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    _, train = split_indices(games, "train"); _, dev = split_indices(games, "dev")
    # Mining uses training-only first7 reference-game profiles; no dev/test information.
    refids = torch.as_tensor(train[:, :7], device="cuda")
    profile = F.normalize(data["value"][refids].mean((1, 2)), dim=-1)
    nearest = torch.argsort(profile @ profile.T, descending=True).cpu().numpy()
    best, history, tick = float("inf"), [], time.monotonic()
    path = dest / f"{mode}_s{seed}.pt"
    for step in range(1, 4001):
        model.train()
        if step > 1000 and rng.random() < .5:
            chosen = []
            for a in rng.permutation(len(train)):
                if a in chosen:
                    continue
                chosen.append(int(a))
                chosen.append(next(int(b) for b in nearest[a] if b not in chosen))
                if len(chosen) == 16:
                    break
            chosen = np.array(chosen)
        else:
            chosen = rng.choice(len(train), 16, replace=False)
        slots = np.stack([rng.choice(10, 4, replace=False) for _ in chosen])
        refs = torch.as_tensor(train[chosen[:, None], slots[:, :3]], device="cuda")
        qi = torch.as_tensor(train[chosen, slots[:, 3]], device="cuda").repeat_interleave(4)
        pos = torch.as_tensor(np.concatenate([rng.choice(32, 4, replace=False) for _ in chosen]), device="cuda")
        qctx = data["context"][qi, pos]
        action = data["action"][qi, pos].float() / data["action_scale"]
        base, played = data["base"][qi, pos], data["played"][qi, pos]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, _ = model(qctx, action, base, data["context"][refs], data["value"][refs])
            loss = episodic_loss(lp, played, torch.arange(16, device="cuda").repeat_interleave(4))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if step % 250 == 0:
            print(json.dumps({"arm": mode, "seed": seed, "step": step, "loss": float(loss), "seconds": time.monotonic() - tick}), flush=True)
        if step % 1000 == 0:
            ev = evaluate(model, data, dev)["metrics"]
            history.append({"step": step, **ev})
            if ev["objective"] < best:
                best = ev["objective"]
                torch.save({"state": model.state_dict(), "step": step, "mode": mode, "seed": seed, "protocol": protocol_id}, path)
            atomic_json(dest / f"{mode}_s{seed}_dev.json", history)
    return path


def experiment(root, m, arrays):
    dest = Path(root) / "refpolicy"
    data, norm = normalize(arrays, m["games"])
    np.savez(dest / "normalizer.npz", **norm)
    checkpoints = []
    for seed in (17, 29):
        for mode in ("attention", "uniform"):
            checkpoints.append(train_one(data, m["games"], seed, mode, dest, m["id"]))
            check(root, LEDGER)
    # No test evaluation until every arm/seed has finished development-only selection.
    owners, index = split_indices(m["games"], "test")
    reports, results, score_arrays = {}, {}, {}
    for path in checkpoints:
        ck = torch.load(path, weights_only=True)
        model = ReferencePolicy(388, 385, mode=ck["mode"]).cuda(); model.load_state_dict(ck["state"])
        label = path.stem
        results[label] = evaluate(model, data, index)
        ev = results[label]
        reports[label] = {"selected_step": ck["step"], "metrics": ev["metrics"], "own_minus_wrong_move_ll": interval(ev["move_advantage"])}
        score_arrays[label] = ev["scores"]
        if ck["mode"] == "attention":
            for abl in ("permuted_keys", "zero"):
                a = evaluate(model, data, index, abl)
                reports[label][abl] = a["metrics"]
                reports[label]["minus_" + abl] = interval(ev["per_owner_rank"] - a["per_owner_rank"])
                score_arrays[label + "_" + abl] = a["scores"]
    comparisons = {str(seed): interval(results[f"attention_s{seed}"]["per_owner_rank"] - results[f"uniform_s{seed}"]["per_owner_rank"]) for seed in (17, 29)}
    personalized = all(reports[f"attention_s{s}"]["own_minus_wrong_move_ll"]["ci95"][0] > 0 for s in (17, 29))
    aligned = all(comparisons[str(s)]["ci95"][0] > 0 for s in (17, 29))
    verdict = "personalization_and_alignment_supported" if personalized and aligned else "personalization_only_alignment_not_established" if personalized else "mechanism_not_established"
    report = {"protocol": m["id"], "verdict": verdict, "runs": reports, "attention_minus_uniform": comparisons,
              "scope": "50 adapter-unseen test players,32positions/game; fixed pretrained backbone; not full fusion improvement or official evaluation", "test_owners": owners,
              "checkpoints": [source_record(p) for p in checkpoints], "job_id": os.environ["SLURM_JOB_ID"]}
    np.savez_compressed(dest / "test_scores.npz", **score_arrays)
    atomic_json(dest / "report.json", report)
    atomic_json(ROOT / "results/final/reference_policy_20261116.json", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--stage", choices=("prepare", "pilot", "run"), required=True)
    a = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("production Slurm required")
    torch.set_num_threads(1)
    check(a.run_dir, LEDGER)
    m = prepare(a.run_dir) if a.stage == "prepare" else read_protocol(a.run_dir)
    if a.stage == "prepare":
        print(json.dumps({"protocol": m["id"], "games": len(m["games"])}), flush=True)
    else:
        arrays, resources = extract(a.run_dir, m, a.stage == "pilot")
        print(json.dumps(resources), flush=True)
        if a.stage == "run":
            print(json.dumps(experiment(a.run_dir, m, arrays)), flush=True)
    read_protocol(a.run_dir); check(a.run_dir, LEDGER)
