"""Verify every base rank against the window audit, then describe the fixed hard-pair groups."""
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_local_choice_data import DEST, LEDGER, ROOT, RUN, atomic_json, check, source_record, verify_source


def verify():
    out = DEST / "transfer_v2fin3a"
    summary = json.loads((out / "summary.json").read_text())
    audit_summary = json.loads((DEST / "window_rescues.json").read_text())
    source = DEST / "simple_ko_v2"
    pilot = json.loads((source / "report.json").read_text())
    pilot_scores = json.loads((source / "checked.json").read_text())["scores"]
    for item in [*pilot["checkpoints"], pilot_scores]:
        verify_source(item)
    prior = json.loads((DEST / "window_rescue_rows.json").read_text())
    lookup = {(r["seed"], r["key"]): r for r in prior}
    if len(lookup) != 7200:
        raise ValueError("missing audit query rows")
    rank0, rank1, groups, owner, pairs, flags = [], [], [], [], [], []
    used = set()
    for pool_number, pool in enumerate(("pub2", "priv2")):
        info = summary["sources"][pool]
        verify_source(info["sources"])
        with np.load(info["sources"]["path"], allow_pickle=False) as score:
            for seed in (2, 3):
                prefix = f"s{seed}_"
                expected_fingerprint = audit_summary["candidate_sources_sha256"][f"v2fin3_a/{pool}/{seed}"]
                if str(score[prefix + "base_sources"]) != expected_fingerprint:
                    raise ValueError("baseline vector sources differ from original paired audit")
                for i, key in enumerate(score[prefix + "qkeys"].tolist()):
                    k = (seed, key)
                    audit = lookup[k]
                    cluster = int(score[prefix + "truth"][i]) + pool_number * 600
                    if audit["pool"] != pool or audit["rank1"] != score[prefix + "base"][i] or audit["cluster"] != cluster or k in used:
                        raise ValueError("unpaired query or base rank differs from independent audit")
                    used.add(k)
                    owner.append(cluster)
                    rank0.append(audit["rank1"])
                    rank1.append(int(score[prefix + "L"][i]))
                    groups.append(audit["selected_rival"] >= 0)
                    flags.append(score[prefix + "small_margin"][i])
                    pairs.append(score[prefix + "L_pair_won"][i])
    assert used == set(lookup)
    a, b = np.asarray(rank1), np.asarray(rank0)
    cluster = np.asarray(owner)
    score = lambda r: np.exp(-r.astype(float)) * (r < 5)
    delta = score(a) - score(b)
    primary = summary["comparisons"]["L minus base"]
    np.testing.assert_allclose(delta.mean(), primary["difference"], atol=1e-10)
    assert primary["rescued"] == int(((a == 0) & (b > 0)).sum())
    assert primary["broken"] == int(((b == 0) & (a > 0)).sum())
    rng = np.random.default_rng(1811)
    draws = np.column_stack([rng.integers(off, off + 600, (5000, 600)) for off in (0, 600)])

    def group_stats(mask, values):
        den = np.bincount(cluster[mask], minlength=1200)[draws].sum(1)
        num = np.bincount(cluster[mask], values[mask], minlength=1200)[draws].sum(1)
        return {"n": int(mask.sum()), "pool_owner_units": len(set(cluster[mask])), "difference": float(values[mask].mean()),
                "ci95": np.quantile(num[den > 0] / den[den > 0], [.025, .975]).tolist()}

    complete = group_stats(np.ones(len(a), dtype=bool), delta)
    np.testing.assert_allclose(complete["ci95"], primary["ci95"], atol=1e-10)
    marked = np.asarray(groups)
    small = np.asarray(flags)
    in2 = b < 2
    report = {"primary_verified": True, "protocol": summary["protocol"], "all": complete,
              "L_minus_base_by_pool": {name: group_stats(cluster // 600 == i, delta) for i, name in enumerate(("pub2", "priv2"))},
              "previously_confused_owner_subset": {**group_stats(marked, delta),
                  "rescued": int((marked & (a == 0) & (b > 0)).sum()), "broken": int((marked & (b == 0) & (a > 0)).sum()),
                  "scope": "Different query games, shared enrollment evidence with rival discovery"},
              "rescued_from_base_rank2_to_5": int(((b > 0) & (b < 5) & (a == 0)).sum()),
              "top_two_specialist_minus_base_accuracy": group_stats(in2 & small, np.asarray(pairs, float) - (b == 0)),
              "baseline_and_checkpoint_sources_verified": True,
              "audit_rows": source_record(DEST / "window_rescue_rows.json"), "job": os.environ["SLURM_JOB_ID"]}
    atomic_json(out / "checked.json", report)
    atomic_json(ROOT / "results/final/local_choice_transfer_check_20261118.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("use production Slurm")
    verify()
    check(RUN, LEDGER)
