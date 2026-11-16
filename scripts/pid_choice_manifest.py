#!/usr/bin/env python
"""Immutable candidate-only input manifest for the conditional-choice study."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aicup_pid import v2
from aicup_pid.paths import AICUP1, FOLDS_CSV, TRAIN_DIR


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_record(path):
    p = Path(path).absolute()
    before = p.stat()
    sha = digest_file(p)
    after = p.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed while hashing: {p}")
    return {"path": str(p), "realpath": str(p.resolve()), "sha256": sha, "size": after.st_size}


def verify_source(record):
    p = Path(record["path"])
    if str(p.resolve()) != record["realpath"] or not p.is_file() or p.stat().st_size != record["size"] or digest_file(p) != record["sha256"]:
        raise ValueError(f"source changed: {p}")


def namespaced_key(pool, key):
    if "|" in pool or "|" in key:
        raise ValueError("ambiguous key")
    return pool + "|" + key


def validate_projection(path):
    with np.load(path, allow_pickle=False) as p:
        if p["blocks"].tolist() != ["g", "x", "e"] or p["fit_folds"].tolist() != [3, 4] or p["W"].shape != (1152, 64) or p["sd"].shape != (1152,):
            raise ValueError("unexpected projection provenance/shape")
        if not np.isfinite(p["W"]).all() or not np.isfinite(p["sd"]).all() or (p["sd"] <= 0).any():
            raise ValueError("invalid projection values")
        return np.asarray(p["W"][384:768, :32] / p["sd"][384:768, None], dtype=np.float32)


def atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with open(tmp, "x") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def seed_split(manifest, pool, seed):
    rows = [g for g in manifest["games"] if g["pool"] == pool]
    cands = {}
    for g in rows:
        cands.setdefault(g["owner"], []).append((g["original_key"], g["color"]))
    if any(len(g) != 10 for g in cands.values()) or not cands:
        raise ValueError("expected ten games per candidate")
    hold = v2.candidate_holdout(cands, 3, seed)
    held = {k for gs in hold.values() for k, _ in gs}
    order = sorted(cands, key=lambda p: stable_hash("choice-half-v1|" + pool + "|" + p))
    return {"reference": {p: [namespaced_key(pool, k) for k, _ in gs if k not in held] for p, gs in cands.items()},
            "heldout": {p: [namespaced_key(pool, k) for k, _ in gs] for p, gs in hold.items()},
            "halves": {p: int(i >= len(order) // 2) for i, p in enumerate(order)}}


def game_record(pool, original, owner, color, sgf, **extra):
    if color not in ("B", "W"):
        raise ValueError("invalid player color")
    return {"pool": pool, "key": namespaced_key(pool, original), "original_key": original, "owner": owner,
            "color": color, "sgf": sgf, "sgf_sha256": stable_hash(sgf), **extra}


def select_calibration(train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):
    folds = {r["player_id"]: int(r["fold"]) for r in csv.DictReader(open(folds_csv))}
    by = {}
    sources = [source_record(folds_csv)]
    for p in sorted(Path(train_dir).glob("train_*.csv")):
        sources.append(source_record(p))
        with open(p) as f:
            for r in csv.DictReader(f):
                owner = r["player_id"]
                if folds.get(owner) not in (3, 4) or r["sgf_content"].count(";" + r["color"] + "[") < 10:
                    continue
                key = r["rank"] + ":" + r["game_id"]
                g = game_record("calibration", key, owner, r["color"], r["sgf_content"], fold=folds[owner])
                bucket = by.setdefault(owner, [])
                bucket.append(g)
                if len(bucket) > 10:
                    bucket.sort(key=lambda x: stable_hash("choice-v1|" + x["original_key"]))
                    bucket.pop()
    eligible = sorted((p for p, games in by.items() if len(games) == 10), key=lambda p: stable_hash("choice-player-v1|" + p))[:100]
    if len(eligible) != 100:
        raise ValueError("fewer than 100 eligible calibration players")
    games = []
    for p in eligible:
        gs = sorted(by[p], key=lambda x: stable_hash("choice-v1|" + x["original_key"]))
        games.extend(g | {"role": "reference" if i < 7 else "heldout"} for i, g in enumerate(gs))
    return sorted(games, key=lambda x: stable_hash("choice-v1|" + x["original_key"])), sources


def build_manifest(run_root, *, fusion, projection, checkpoint, source_root):
    validate_projection(projection)
    source_root = Path(source_root).resolve()
    games, sources = select_calibration()
    for tag, pool in [("public", "pub2"), ("private", "priv2")]:
        p = source_root / "data/tests_1111" / f"player_identification_candidates2_{tag}.csv"
        sources.append(source_record(p))
        with open(p) as f:
            rows = list(csv.DictReader(f))
        gs = [game_record(pool, f"c:{i}:{r['game_id']}", r["player_id"], r["color"], r["sgf_content"], row_index=i) for i, r in enumerate(rows)]
        if len(gs) != 6000 or len({g["owner"] for g in gs}) != 600:
            raise ValueError(f"unexpected {pool} candidate coverage")
        games.extend(gs)
    assets = {k: source_record(p) for k, p in {"fusion": fusion, "projection": projection, "checkpoint": checkpoint}.items()}
    # Snapshot the code determining extraction and interpretation, never query CSVs.
    code_paths = [ROOT / "scripts/pid_choice_manifest.py", ROOT / "scripts/pid_choice_extract.py", ROOT / "src/aicup_pid/choice_signal.py",
                  ROOT / "src/aicup_pid/v2.py", AICUP1 / "scripts/trunk_extract.py", AICUP1 / "src/aicup/humansl.py", AICUP1 / "src/aicup/l1b.py"]
    code = [source_record(p) for p in code_paths]
    body = {"schema_version": 1, "source_root": str(source_root), "games": games, "sources": sources, "assets": assets, "code": code,
            "settings": {"max_ply": 256, "part_size": 64, "seeds": list(range(6)), "ridges": [.1, 1., 10.], "alphas": [0., .05, .1, .2, .4],
                         "legality": "board_simple_ko_no_superko", "max_gpuh": 6., "max_bytes": 1073741824}}
    run_id = stable_hash(json.dumps(body, sort_keys=True, ensure_ascii=False))[:12]
    out = Path(run_root) / run_id
    body["run_id"] = run_id
    if (out / "manifest.json").exists():
        if json.loads((out / "manifest.json").read_text()) != body:
            raise ValueError("run collision")
    else:
        atomic_json(out / "manifest.json", body)
    for pool in ("pub2", "priv2"):
        for seed in range(6):
            seed_split(body, pool, seed)
    return out


def read_manifest(run_dir, verify=True):
    m = json.loads((Path(run_dir) / "manifest.json").read_text())
    body = {k: v for k, v in m.items() if k != "run_id"}
    if m["schema_version"] != 1 or stable_hash(json.dumps(body, sort_keys=True, ensure_ascii=False))[:12] != m["run_id"]:
        raise ValueError("manifest checksum/schema mismatch")
    keys = [g["key"] for g in m["games"]]
    if len(set(keys)) != len(keys) or any(stable_hash(g["sgf"]) != g["sgf_sha256"] for g in m["games"]):
        raise ValueError("duplicate key or modified SGF")
    if verify:
        for rec in [*m["sources"], *m["assets"].values(), *m["code"]]:
            verify_source(rec)
    return m


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--run-root", required=True)
    a.add_argument("--source-root", default="/home/u2507115/BK_AICUP_2026-2")
    a.add_argument("--fusion", default="/home/u2507115/BK_AICUP_2026-2/results/pid_fusion_v2bgm_d2.json")
    a.add_argument("--projection", default="/work/u2507115/pid_v2/probep/proj3.npz")
    a.add_argument("--checkpoint", default=str(AICUP1 / "models/katago/b18c384nbt-humanv0.ckpt"))
    args = a.parse_args()
    print(build_manifest(**vars(args)), flush=True)
