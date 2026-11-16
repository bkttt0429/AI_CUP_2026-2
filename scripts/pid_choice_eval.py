#!/usr/bin/env python
"""Candidate-only baseline reproduction and cross-fitted choice augmentation."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aicup_pid import v2
from aicup_pid.choice_signal import fit_candidate, normalise_moments, score_candidates
from aicup_pid.choice_eval import GridPoint, GridScores, add_evidence, bootstrap_procedure, crossfit_grid, ranks, rank_scores
from aicup_pid.components import parse_all
from aicup_pid.embed import PlayerEmbedding
from aicup_pid.fusion import standardise
from aicup_pid.trunkstyle import load_store
from pid_choice_manifest import atomic_json, digest_file, read_manifest, seed_split, source_record
from pid_choice_extract import load_moments
from pid_choice_audit import load_calibration


def ridge_grid(arm, accepted, fixed):
    return [None] if arm == "R" else list(fixed) if arm == "A0" else list(accepted)


def load_vectors(path, keys):
    with np.load(path, allow_pickle=False) as z:
        if len(set(z["key"].tolist())) != len(z["key"]):
            raise ValueError("duplicate embedding keys")
        lookup = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
    if any(k not in lookup for k in keys):
        raise ValueError(f"missing reference-only vectors in {path}")
    out = np.stack([lookup[k] for k in keys])
    if not np.isfinite(out).all():
        raise ValueError("nonfinite reference-only vectors")
    return out


def validate_candidate_rows(rows, pool):
    gs = [g for g in rows if g["pool"] == pool]
    if not gs or any(not g["original_key"].startswith("c:") for g in gs):
        raise ValueError("only candidate rows are allowed")
    return gs


def questions_from_split(split, owners):
    questions, truth, sizes = [], [], []
    for ci, owner in enumerate(owners):
        h = split["heldout"][owner]
        qs = [[k] for k in h] + [h[:2], h[:3]]
        questions.extend(qs); truth.extend([ci] * len(qs)); sizes.extend(map(len, qs))
    return questions, np.array(truth), np.array(sizes)


def owner_size_scores(score, owner, sizes, count):
    out = np.empty((count, 3))
    for j, n in enumerate((1, 2, 3)):
        mask = sizes == n
        den = np.bincount(owner[mask], minlength=count)
        if np.any(den == 0):
            raise ValueError("incomplete owner/type coverage")
        out[:, j] = np.bincount(owner[mask], weights=score[mask], minlength=count) / den
    return out


def question_groups(questions, games, moments):
    colors, lengths = [], []
    for q in questions:
        cs = {games[k]["color"] for k in q}
        colors.append(next(iter(cs)) if len(cs) == 1 else "mixed")
        n = np.mean([moments[k]["n"] for k in q])
        lengths.append("<64" if n < 64 else "64-95" if n < 96 else "96+")
    return {"color": np.array(colors), "mean_own_moves": np.array(lengths), "size": np.array([str(len(q)) for q in questions])}


def group_diagnostics(baseline_rank, new_rank, groups):
    sc = lambda r: np.where(r < 5, np.exp(-r.astype(float)), 0.)
    delta = sc(new_rank) - sc(baseline_rank)
    return {name: {label: {"questions": int((labels == label).sum()),
                          "rank_score_delta": float(delta[labels == label].mean()),
                          "recovered_top1": int(((labels == label) & (baseline_rank > 0) & (new_rank == 0)).sum()),
                          "broken_top1": int(((labels == label) & (baseline_rank == 0) & (new_rank > 0)).sum())}
                   for label in sorted(set(labels.tolist()))} for name, labels in groups.items()}


def load_baseline_inputs(manifest, pool, seed):
    """Deliberately does not use Pool, official_pool or any question loader."""
    root = Path(manifest["source_root"])
    doc = json.loads(Path(manifest["assets"]["fusion"]["path"]).read_text())
    games = validate_candidate_rows(manifest["games"], pool)
    by_key = {g["key"]: g for g in games}
    split = seed_split(manifest, pool, seed)
    owners = sorted(split["reference"])
    qs, truth, sizes = questions_from_split(split, owners)
    held = {k for ks in split["heldout"].values() for k in ks}
    if held.intersection(k for ks in split["reference"].values() for k in ks):
        raise ValueError("reference/query overlap")
    convert = lambda ks: [(by_key[k]["original_key"], by_key[k]["color"]) for k in ks]
    ref = {p: convert(split["reference"][p]) for p in owners}
    queries = [convert(ks) for ks in qs]
    sgfs = {g["original_key"]: g["sgf"] for g in games}
    store_path = root / doc["trunk_store"]
    store = load_store(str(store_path))
    if any(k not in store for k in sgfs) or any(not np.isfinite(store[k]).all() for k in sgfs):
        raise ValueError("missing/nonfinite candidate trunk rows")
    emb_path = root / doc["emb_model"]
    emb = PlayerEmbedding.from_state(torch.load(emb_path, map_location="cpu", weights_only=False), device="cpu")
    moves = parse_all(sgfs, workers=10)
    other = [c for c in doc["components"] if c != "seq"]
    S = v2.components(ref, queries, moves, sgfs, doc["cfg"], store, emb, which=other)
    keys = [g["original_key"] for g in games]
    row = {k: i for i, k in enumerate(keys)}
    def membership(sets):
        rr = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        cc = np.array([row[k] for s in sets for k, _ in s])
        return sparse.csr_matrix((np.ones(len(cc), np.float32), (rr, cc)), shape=(len(sets), len(keys)))
    A, B = membership([ref[p] for p in owners]), membership(queries)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    members = Counter(tuple(x) if isinstance(x, list) else (x,) for x in doc["seq_official"])
    acc, weight = 0., 0
    sources = [source_record(emb_path)]
    sources.extend(source_record(p) for p in sorted(store_path.glob("*.npz")) if not p.name.endswith(".tmp.npz"))
    for names, w in members.items():
        match = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(names[0]).name)
        if match is None:
            raise ValueError("unexpected baseline member")
        v, tag = match.groups()
        path = root / "data/pid_v2_seq_ft_offsim" / f"offsim_{pool}_h3s{seed}_net{v}{tag}_t50.npz"
        V = load_vectors(path, keys)
        sources.append(source_record(path))
        acc += w * standardise(unit(np.asarray(B @ V, np.float64)) @ unit(np.asarray(A @ V, np.float64)).T)
        weight += w
    S["seq"] = acc / weight
    if set(S) != set(doc["components"]) or any(not np.isfinite(s).all() for s in S.values()):
        raise ValueError("missing/nonfinite baseline component; fallback forbidden")
    baseline = sum(w * standardise(S[c]) for w, c in zip(doc["weights"], doc["components"]))
    return {"baseline": baseline, "questions": qs, "truth": truth, "sizes": sizes, "owners": owners,
            "halves": np.array([split["halves"][p] for p in owners]), "references": split["reference"], "sources": sources}


def baseline_check(run_dir, seeds=(0, 1, 2, 3)):
    start = time.monotonic()
    m = read_manifest(run_dir)
    expected = json.loads((Path(m["source_root"]) / "results/final/v2_probe_bottleneck_v2bgm_d2.json").read_text())
    out = {"run_id": m["run_id"], "pools": {}, "code_sha256": digest_file(__file__)}
    for pool in ("pub2", "priv2"):
        scores, hits = [], []
        for seed in seeds:
            b = load_baseline_inputs(m, pool, seed)
            sc, rr = rank_scores(b["baseline"], b["truth"]), ranks(b["baseline"], b["truth"])
            mask = b["sizes"] == 1
            scores.extend(sc[mask]); hits.extend(rr[mask] == 0)
            path = Path(run_dir) / f"baseline_{pool}_s{seed}.npz"
            with open(path.with_suffix(".tmp"), "wb") as f:
                np.savez_compressed(f, scores=b["baseline"], truth=b["truth"], sizes=b["sizes"], owners=np.array(b["owners"]), halves=b["halves"])
                f.flush(); os.fsync(f.fileno())
            os.replace(path.with_suffix(".tmp"), path)
            atomic_json(path.with_suffix(".json"), {"run_id": m["run_id"], "sha256": digest_file(path), "sources": b["sources"], "code_sha256": digest_file(__file__)})
            print(f"baseline {pool} seed {seed}: {np.mean(sc[mask]):.12f}, Top1 {np.mean(rr[mask] == 0):.12f}", flush=True)
        measured = {"score": float(np.mean(scores)), "top1": float(np.mean(hits))}
        if tuple(seeds) == (0, 1, 2, 3) and any(abs(measured[k] - expected["pools"][pool][k]) > 1e-9 for k in measured):
            raise ValueError(f"baseline reproduction failed {pool}: {measured}")
        out["pools"][pool] = measured
    out.update(verdict="passed", elapsed_seconds=time.monotonic() - start)
    out["seeds"] = list(seeds)
    filename = "baseline_check.json" if tuple(seeds) == (0, 1, 2, 3) else "baseline_extra.json"
    atomic_json(Path(run_dir) / filename, out)
    return out


def _sum_moments(rows, sets, suffix, scale):
    U = np.stack([sum(rows[k]["U" + suffix].astype(np.float64) for k in ks) for ks in sets])
    F = np.stack([sum(rows[k]["F" + suffix].astype(np.float64) for k in ks) for ks in sets])
    n = np.array([sum(int(rows[k]["n"]) for k in ks) for ks in sets])
    u, f = normalise_moments(U, F, scale)
    return u, f, n


def evaluate_seed(run_dir, manifest, pool, seed):
    root = Path(run_dir)
    input_gate = json.loads((root / "audit_input.json").read_text())
    if input_gate["verdict"] != "passed" or input_gate["run_id"] != manifest["run_id"] or input_gate["games"] != len(manifest["games"]):
        raise ValueError("complete input turn-order audit must pass")
    if json.loads((root / "baseline_check.json").read_text())["verdict"] != "passed":
        raise ValueError("baseline reproduction must pass")
    if json.loads((root / f"audit_reference_{pool}_s{seed}.json").read_text())["verdict"] != "passed":
        raise ValueError("reference numerical audit must pass")
    scale, fbar, ridges = load_calibration(root)
    rows, _ = load_moments(root, manifest, pool)
    split = seed_split(manifest, pool, seed)
    owners = sorted(split["reference"])
    qs, truth, sizes = questions_from_split(split, owners)
    path = root / f"baseline_{pool}_s{seed}.npz"
    provenance = json.loads(path.with_suffix(".json").read_text())
    if provenance["run_id"] != manifest["run_id"] or digest_file(path) != provenance["sha256"]:
        raise ValueError("baseline provenance mismatch")
    with np.load(path) as z:
        base = z["scores"]
        if z["owners"].tolist() != owners or not np.array_equal(z["truth"], truth) or not np.array_equal(z["sizes"], sizes):
            raise ValueError("baseline question ordering differs")
        halves = z["halves"]
    configs, score_grid, rank_grid = [], [], []
    for arm, suffix in [("R", "0"), ("A0", "0"), ("A1", "0"), ("A2", "2"), ("A2perm", "2perm")]:
        d = scale if suffix == "0" else np.tile(scale, 2)
        ur, fr, nr = _sum_moments(rows, [split["reference"][p] for p in owners], suffix, d)
        uq, fq, nq = _sum_moments(rows, qs, suffix, d)
        for ridge in ridge_grid(arm, ridges, manifest["settings"]["ridges"]):
            if arm == "R":
                unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
                evidence = unit(uq / nq[:, None]) @ unit(ur / nr[:, None]).T
            else:
                fr_used = nr[:, None, None] * fbar if arm == "A0" else fr
                fq_used = nq[:, None, None] * fbar if arm == "A0" else fq
                z = np.stack([fit_candidate(u, f, n, ridge) for u, f, n in zip(ur, fr_used, nr)])
                evidence = score_candidates(uq, fq_used, z)
            for alpha in manifest["settings"]["alphas"]:
                new = add_evidence(base, evidence, alpha)
                configs.append(GridPoint(arm, ridge, alpha))
                score_grid.append(owner_size_scores(rank_scores(new, truth), truth, sizes, len(owners)))
                rank_grid.append(ranks(new, truth).astype(np.int16))
        print(f"grid {pool} seed {seed}: {arm} complete", flush=True)
    base_sc = owner_size_scores(rank_scores(base, truth), truth, sizes, len(owners))
    data = GridScores(pool, (seed,), tuple(owners), halves, tuple(configs), np.stack(score_grid)[None], base_sc[None])
    selected = crossfit_grid(data)
    locks = {arm: [] for arm in selected["selection"]}
    for arm, sels in selected["selection"].items():
        for h, index in enumerate(sels[0]):
            source = data.score[0, :, data.halves != h, :]
            locks[arm].append({"evaluated_half": h, "config": asdict(configs[index]), "grid_index": index,
                               "source_grid_sha256": hashlib.sha256(source.tobytes()).hexdigest()})
    atomic_json(root / f"selection_{pool}_s{seed}.json", {"run_id": manifest["run_id"], "pool": pool, "seed": seed, "locks": locks, "code_sha256": digest_file(__file__)})
    raw_rank = np.stack(rank_grid)
    baseline_rank = ranks(base, truth)
    groups = question_groups(qs, {g["key"]: g for g in manifest["games"]}, rows)
    diag = {}
    for arm, sels in selected["selection"].items():
        rr = np.empty_like(baseline_rank)
        for h, index in enumerate(sels[0]):
            mask = halves[truth] == h
            rr[mask] = raw_rank[index, mask]
        diag[arm] = {"recovered_top1": int(((baseline_rank > 0) & (rr == 0)).sum()), "broken_top1": int(((baseline_rank == 0) & (rr > 0)).sum()),
                     "strata": group_diagnostics(baseline_rank, rr, groups),
                     "by_original_rank": {str(lo) + "-" + str(hi): {"questions": int(((baseline_rank >= lo) & (baseline_rank <= hi)).sum()),
                         "new_top1": int(((baseline_rank >= lo) & (baseline_rank <= hi) & (rr == 0)).sum())} for lo, hi in [(0, 0), (1, 4), (5, 19), (20, 599)]}}
    atomic_json(root / f"diagnostics_{pool}_s{seed}.json", diag)
    with open(root / f"grid_{pool}_s{seed}.npz", "wb") as f:
        np.savez_compressed(f, score=data.score, base=data.base, halves=halves, owners=np.array(owners), configs=np.array(json.dumps([asdict(c) for c in configs])), ranks=raw_rank, baseline_rank=baseline_rank)
    return data


def robustness_verdict(primary, report):
    agrees = all(p["by_seed"][str(seed)]["A2"] > 0 for p in report["pools"].values() for seed in (4, 5))
    return "promising" if primary["verdict"] == "promising_pending_robustness" and agrees else "robustness_not_confirmed"


def evaluate(run_dir, seeds, draws=5000):
    m = read_manifest(run_dir)
    pools = {}
    for pool in ("pub2", "priv2"):
        pieces = [evaluate_seed(run_dir, m, pool, s) for s in seeds]
        p = pieces[0]
        pools[pool] = GridScores(pool, tuple(seeds), p.owners, p.halves, p.configs, np.concatenate([x.score for x in pieces]), np.concatenate([x.base for x in pieces]))
    report = bootstrap_procedure(pools, seeds=tuple(seeds), draws=draws)
    report["run_id"] = m["run_id"]
    report["code_sha256"] = digest_file(__file__)
    if tuple(seeds) == (4, 5):
        primary = json.loads((Path(run_dir) / "eval_s2_3.json").read_text())
        report["robustness_role"] = "same players and games, new holdout splits only"
        report["verdict"] = robustness_verdict(primary, report)
    atomic_json(Path(run_dir) / ("eval_s" + "_".join(map(str, seeds)) + ".json"), report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=("baseline-check", "baseline-extra", "evaluate"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    p.add_argument("--draws", type=int, default=5000)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("use Slurm")
    torch.set_num_threads(1)
    result = baseline_check(a.run_dir) if a.stage == "baseline-check" else baseline_check(a.run_dir, a.seeds) if a.stage == "baseline-extra" else evaluate(a.run_dir, a.seeds, a.draws)
    print(json.dumps(result), flush=True)
