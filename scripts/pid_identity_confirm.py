"""Frozen, post-discovery confirmation of the complete paired Fisher margin on splits 0 and 2."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, PRIOR, ROOT, RUN, POOLS, atomic_json, atomic_npz, digest_file, guard, read_manifest
from pid_identity_finalists import ARMS, build_kernel, rank_report, score
from pid_choice_eval import load_baseline_inputs, ranks, rank_scores
from pid_tactical_score import competition

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-identity-confirmations.md"
OUT = ROOT / "results/final/identity_fisher_confirmation_20261118.json"


def effects(rows, mask, boot):
    r = rows[mask]
    # Columns: pooled owner, base rank, Fisher rank, isotropic rank.
    ids, base, fisher, iso = r.T
    den = boot @ np.bincount(ids, minlength=1200)
    out = {"rows": len(ids), "owners": len(set(ids)), "rescued": int(np.sum((base > 0) & (fisher == 0))),
           "broken": int(np.sum((base == 0) & (fisher > 0)))}
    for label, cmp in (("vs_base", base), ("vs_isotropic", iso)):
        delta = competition(fisher) - competition(cmp)
        sums = np.bincount(ids, weights=delta, minlength=1200)
        out[label] = {"mean": float(delta.mean()), "ci95": np.quantile((boot @ sums) / den, [.025, .975]).tolist()}
    return out


def main():
    guard()
    torch.set_num_threads(3)
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    doc_path = source / "results/pid_fusion_v2fin4_f.json"
    doc = json.loads(doc_path.read_text())
    old = json.loads((PRIOR / "baseline.json").read_text())
    previous = json.loads((ROOT / "results/final/identity_finalists_20261118.json").read_text())
    if previous["ensemble"] != digest_file(doc_path) or previous["code"] != digest_file(ROOT / "scripts/pid_identity_finalists.py"):
        raise ValueError("primary experiment changed")
    seen = set()
    for p in POOLS:
        for s in (1, 3):
            path = CACHE / f"identity_{p}_{s}.npz"
            if digest_file(path) != previous["splits"][str(s)]["archives"][p]:
                raise ValueError("exploratory query roles changed")
            with np.load(path, allow_pickle=False) as z:
                seen.update(z["keys"].tolist())
    rows, masks, new_keys, per_seed, fingerprints = [], [], set(), {}, {}
    with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
        root = Path(temp)
        (root / "data").mkdir()
        for link, target in (("results", "results"), ("data/trunk", "data/trunk"), ("data/pid_v2_seq_ft_offsim", "data/pid_v2_f0tmp/ft")):
            (root / link).symlink_to(source / target)
        view = {**m, "source_root": str(root), "assets": {**m["assets"], "fusion": {"path": str(doc_path)}}}
        for seed in (0, 2):
            all_records = []
            for pi, p in enumerate(POOLS):
                collected = load_baseline_inputs(view, p, seed)
                only = collected["sizes"] == 1
                s, truth = collected["baseline"][only], collected["truth"][only]
                if abs(rank_scores(s, truth).mean() - old["pools"][p]["seed_means"][seed]) > 1e-9:
                    raise ValueError("cannot reproduce accepted single-game scores")
                owners = np.array(collected["owners"])
                b = {"scores": s, "truth": truth, "rank": ranks(s, truth), "owners": owners,
                     "refs": np.array([collected["references"][a] for a in owners]),
                     "keys": np.array([q[0] for q in collected["questions"] if len(q) == 1])}
                keys = np.r_[b["refs"].ravel(), b["keys"]]
                if len(set(keys)) != 6000 or not all(k.startswith(p + "|c:") for k in keys):
                    raise ValueError("incompatible roles")
                gram = build_kernel(source, doc, p, seed, [k.split("|", 1)[1] for k in keys], old["pools"][p]["sources"][str(seed)])
                rec, _ = score(p, b, gram)
                del gram
                test = np.arange(len(truth)) % 3 != 0
                eval_keys = b["keys"][test]
                keep = []
                for k in eval_keys:
                    fresh = k not in seen and k not in new_keys
                    keep.append(fresh)
                    if fresh:
                        new_keys.add(k)
                x = np.column_stack([truth[test] + 600 * pi, rec["base_rank"], rec["new_rank"][:, ARMS.index("fisher")],
                                     rec["new_rank"][:, ARMS.index("isotropic")]])
                rows.append(x)
                masks.extend(keep)
                archive = CACHE / f"confirm_{p}_{seed}.npz"
                atomic_npz(archive, keys=eval_keys, rows=x, fresh=np.array(keep), order=rec["pair"][test],
                           fitted_gaps=rec["gap"], raw=rec["raw"][test])
                fingerprints[f"{p}/{seed}"] = digest_file(archive)
                all_records.append(rec)
                print("confirmed split", seed, p, flush=True)
            t = rank_report(all_records)
            per_seed[str(seed)] = {k: t["arms"][k] for k in ("fisher", "isotropic")}
    rng = np.random.default_rng(111912)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1 / 600), 5000) for _ in POOLS])
    rows = np.vstack(rows)
    a, b = effects(rows, np.ones(len(rows), bool), boot), effects(rows, np.asarray(masks), boot)
    atomic_json(OUT, {"plan": digest_file(PLAN), "code": digest_file(__file__), "archives": fingerprints,
                      "per_split": per_seed, "combined": a, "unique_new_as_queries": b,
                      "screen_pass": bool(a["vs_base"]["mean"] >= .002 and a["vs_base"]["ci95"][0] > 0 and b["vs_base"]["ci95"][0] > 0)})


if __name__ == "__main__":
    main()
