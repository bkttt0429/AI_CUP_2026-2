#!/usr/bin/env python
"""Extract new conditional-choice moments without changing existing features.

Only candidate and calibration SGFs from the immutable manifest are read.
All computation belongs in a production Slurm allocation.
"""
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
from aicup_pid.choice_signal import choice_moments, context_features, permute_context
from pid_choice_manifest import atomic_json, digest_file, read_manifest, stable_hash, validate_projection
import pid_trunk_extract as PT
from aicup.humansl import load_humansl
from aicup.l1b import _GraphExtraOutputs, neutral_rows

te = PT.te


def replay_choices(key, sgf, color):
    if color not in ("B", "W"):
        raise ValueError("invalid target color")
    stones, moves = te.setup_and_moves(sgf)
    gs = te.start_state(stones, te.RULES)
    out = {k: [] for k in ("bins", "globs", "played", "mover", "t", "legal")}
    for t, (c, x, y) in enumerate(moves[:256]):
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE
        loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if not gs.board.would_be_legal(pla, loc):
            raise ValueError(f"{key} ply {t}: illegal actual move")
        if c == color:
            b, g = gs.get_input_features(te._FEATURES)
            legal = np.zeros(362, dtype=bool)
            for yy in range(19):
                for xx in range(19):
                    l = gs.board.loc(xx, yy)
                    legal[te._FEATURES.loc_to_tensor_pos(l, gs.board)] = gs.board.would_be_legal(pla, l)
            legal[361] = True
            p = 361 if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board)
            if not legal[p]:
                raise ValueError(f"{key}: played coordinate does not match mask")
            out["bins"].append(b[0].astype(np.uint8))
            out["globs"].append(g[0].astype(np.float32))
            out["played"].append(p)
            out["mover"].append(int(c == "W"))
            out["t"].append(t)
            out["legal"].append(legal)
        gs.play(pla, loc)
    if not out["played"]:
        raise ValueError(f"{key}: no target move in first 256 plies")
    return {"key": key, **{k: np.stack(v) for k, v in out.items()}}


def _replay_game(g):
    return replay_choices(g["key"], g["sgf"], g["color"])


def forward_choices(model, replay, projection, batch=128):
    """Return p0, all legal action features, played moves and BEFORE-move value."""
    device = next(model.parameters()).device
    rows = torch.as_tensor(neutral_rows(), device=device, dtype=torch.float32)
    project = torch.as_tensor(projection, device=device, dtype=torch.float32)
    out = {k: [] for k in ("logp", "psi", "value")}
    with torch.inference_mode():
        for i in range(0, len(replay["t"]), batch):
            sl = slice(i, i + batch)
            b = torch.as_tensor(replay["bins"][sl], device=device).float()
            g = torch.as_tensor(replay["globs"][sl], device=device).float()
            side = torch.as_tensor(replay["mover"][sl], device=device).long()
            legal = torch.as_tensor(replay["legal"][sl], device=device)
            extra = _GraphExtraOutputs(["trunkfinal"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(b, g, input_meta=rows[side] if model.get_has_metadata_encoder() else None, extra_outputs=extra)
                post = model.postprocess_output(outputs)[0]
            logits = post[0][:, 0, :].float()
            lp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
            trunk = extra.returned["trunkfinal"].float().flatten(2).transpose(1, 2)
            psi = torch.zeros((len(b), 362, project.shape[1] + 1), device=device)
            psi[:, :361, :-1] = trunk @ project
            psi[:, 361, -1] = 1.
            vp = torch.softmax(post[1].float(), dim=1)
            out["logp"].append(lp)
            out["psi"].append(psi)
            out["value"].append(vp[:, 0] - vp[:, 1])
    return {k: torch.cat(v) for k, v in out.items()}


def extract_game(model, replay, projection, *, pool, key, audit_times=()):
    x = forward_choices(model, replay, projection)
    played = torch.as_tensor(replay["played"], device=x["psi"].device).long()
    perm = permute_context(x["value"].cpu().numpy(), replay["t"], pool=pool, key=key)
    psis = {"0": x["psi"], "2": context_features(x["psi"], x["value"]), "2perm": context_features(x["psi"], perm)}
    out = {}
    with torch.inference_mode():
        for name, phi in psis.items():
            u, f, n = choice_moments(x["logp"], phi, played)
            out["U" + name] = u.cpu().numpy().astype(np.float32)
            out["F" + name] = f.cpu().numpy().astype(np.float32)
        out["n"] = n
    ix = np.flatnonzero(np.isin(replay["t"], list(audit_times)))
    out["audit"] = {"lp": x["logp"][ix].cpu().numpy(), "psi": x["psi"][ix].cpu().numpy(), "v": x["value"][ix].cpu().numpy(),
                    "vperm": perm[ix], "played": replay["played"][ix], "t": replay["t"][ix], "key": np.array([replay["key"]] * len(ix), dtype="U100")}
    return out


def valid_shard(path, metadata):
    path = Path(path)
    done = path.with_suffix(".json")
    try:
        record = json.loads(done.read_text())
        return record["metadata"] == metadata and path.is_file() and record["sha256"] == digest_file(path)
    except (FileNotFoundError, ValueError, KeyError):
        return False


def write_shard(run_dir, part_id, arrays, metadata):
    path = Path(run_dir) / "moments" / (part_id + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = path.with_suffix(".json")
    if done.exists() and json.loads(done.read_text())["metadata"] != metadata:
        raise ValueError("existing shard has different provenance")
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError("refusing to replace different shard contents")
    else:
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with open(tmp, "xb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    atomic_json(done, {"metadata": metadata, "sha256": digest_file(path), "bytes": path.stat().st_size})
    return path


def audit_positions(manifest):
    candidates = []
    for g in manifest["games"]:
        if g["pool"] != "calibration" or g["role"] != "heldout":
            continue
        _, moves = te.setup_and_moves(g["sgf"])
        for t, (c, _, _) in enumerate(moves[:256]):
            if c == g["color"]:
                candidates.append((stable_hash(f"choice-audit-v1|{g['key']}|{t}"), g["key"], t))
    if len(candidates) < 1024:
        raise ValueError("not enough calibration audit positions")
    selected = {}
    for _, key, t in sorted(candidates)[:1024]:
        selected.setdefault(key, []).append(t)
    return selected


def load_moments(run_dir, manifest, pool):
    need = [g["key"] for g in manifest["games"] if g["pool"] == pool]
    rows, audits = {}, []
    for path in sorted((Path(run_dir) / "moments").glob(pool + "_p*.npz")):
        metadata = {"run_id": manifest["run_id"], "pool": pool, "part": path.stem}
        if not valid_shard(path, metadata):
            raise ValueError(f"invalid/uncommitted shard {path}")
        with np.load(path, allow_pickle=False) as z:
            for i, key in enumerate(z["keys"].tolist()):
                if key in rows:
                    raise ValueError("duplicate moment key")
                rows[key] = {k: z[k][i] for k in ("U0", "F0", "U2", "F2", "U2perm", "F2perm", "n")}
            if "audit_key" in z and len(z["audit_key"]):
                audits.append({k[6:]: z[k] for k in z.files if k.startswith("audit_")})
    if set(rows) != set(need):
        raise ValueError(f"{pool}: incomplete coverage {len(rows)}/{len(need)}")
    return rows, {k: np.concatenate([a[k] for a in audits]) for k in audits[0]} if audits else {}


def extract(run_dir, stage, shard=0, nshards=1, workers=10):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run extraction via Slurm")
    start = time.monotonic()
    manifest = read_manifest(run_dir)
    projection = validate_projection(manifest["assets"]["projection"]["path"])
    selected = audit_positions(manifest)
    checkpoint = manifest["assets"]["checkpoint"]["path"]
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    pools = ["calibration"] if stage in ("pilot", "calibration") else ["pub2", "priv2"]
    if nshards < 1 or not 0 <= shard < nshards:
        raise ValueError("invalid shard")
    parts = []
    for pool in pools:
        gs = [g for g in manifest["games"] if g["pool"] == pool]
        if stage == "pilot":
            gs = gs[:256]
        parts.extend((pool, i // 64, gs[i:i + 64]) for i in range(0, len(gs), 64))
    parts = parts[shard::nshards]
    torch.set_num_threads(1)
    model, _ = load_humansl(checkpoint, "cuda")
    model.eval()
    loaded_at = time.monotonic()
    measured = []
    # Spawn avoids fork-after-CUDA and bounds queued data to one 64-game shard.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=te.init_worker, initargs=(config,)) as ex:
        for pool, idx, gs in parts:
            part = f"{pool}_p{idx:04d}"
            metadata = {"run_id": manifest["run_id"], "pool": pool, "part": part}
            path = Path(run_dir) / "moments" / (part + ".npz")
            if valid_shard(path, metadata):
                print(f"resume {part}", flush=True)
                continue
            tick = time.monotonic()
            replays = list(ex.map(_replay_game, gs, chunksize=1))
            replay_seconds = time.monotonic() - tick
            outputs = [extract_game(model, r, projection, pool=pool, key=g["original_key"], audit_times=selected.get(g["key"], ())) for g, r in zip(gs, replays)]
            torch.cuda.synchronize()
            forward_seconds = time.monotonic() - tick - replay_seconds
            arrays = {k: np.stack([o[k] for o in outputs]) for k in ("U0", "F0", "U2", "F2", "U2perm", "F2perm", "n")}
            arrays["keys"] = np.array([g["key"] for g in gs])
            for k in outputs[0]["audit"]:
                arrays["audit_" + k] = np.concatenate([o["audit"][k] for o in outputs])
            for name, arr in arrays.items():
                if name.startswith(("U", "F")) and not np.isfinite(arr).all():
                    raise ValueError(f"nonfinite {part}/{name}")
            raw_bytes = sum(a.nbytes for a in arrays.values())
            used = sum(p.stat().st_size for p in Path(run_dir).rglob("*") if p.is_file())
            if used + raw_bytes > manifest["settings"]["max_bytes"]:
                raise RuntimeError("budget_exceeded: projected disk")
            write_shard(run_dir, part, arrays, metadata)
            row = {"part": part, "games": len(gs), "positions": int(arrays["n"].sum()), "replay_seconds": replay_seconds,
                   "forward_moments_seconds": forward_seconds, "seconds": time.monotonic() - tick, "bytes": path.stat().st_size}
            measured.append(row)
            print(json.dumps(row), flush=True)
    report = {"stage": stage, "job_id": os.environ["SLURM_JOB_ID"], "run_id": manifest["run_id"], "shard": shard, "parts": measured,
              "load_seconds": loaded_at - start, "elapsed_seconds": time.monotonic() - start,
              "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "legality": manifest["settings"]["legality"]}
    atomic_json(Path(run_dir) / f"{stage}_{shard}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=("pilot", "calibration", "official"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    extract(**vars(a))
