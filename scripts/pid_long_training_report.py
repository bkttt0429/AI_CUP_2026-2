"""Verify official scores and measure the paired checkpoint and family-weight effects."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_long_training_inputs import ARMS, CACHE, DOCUMENTS, PLAN, POOLS, ROOT, RUN, atomic_json, digest_file, guard, read_manifest
from pid_reference_policy import physical_signature
from pid_tactical_score import competition

OUT = ROOT / "results/final/long_training_attribution_20261119.json"


class OwnerBootstrap:
    def __init__(self):
        rng = np.random.default_rng(111930)
        self.weights = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS]).astype(float)

    def means(self, ids, values):
        x = np.asarray(values, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) != len(ids) or not len(ids) or not np.isfinite(x).all():
            raise ValueError("invalid bootstrap observations")
        totals = np.stack([np.bincount(ids, weights=c, minlength=1200) for c in x.T], axis=1)
        den = self.weights @ np.bincount(ids, minlength=1200)
        bs = (self.weights @ totals) / den[:, None]
        intervals = np.quantile(bs, [.025, .975], axis=0).T
        return [{"mean": float(mu), "ci95": ci.tolist()} for mu, ci in zip(x.mean(0), intervals)]


def official_checks(source, recs, seeds):
    suffix = "_".join(map(str, seeds))
    checks = {}
    for arm in ("replace", "add"):
        doc = DOCUMENTS[arm]
        path = source / f"results/final/v2_offsim_{doc}_vs_zz_r4f_s{suffix}.json"
        published = json.loads(path.read_text())
        if published["seeds"] != list(seeds) or published["document"] != "pid_fusion_zz_r4f.json":
            raise ValueError("wrong published evaluation")
        for key in (*POOLS, "both"):
            rs = [v for (p, s), v in recs.items() if s in seeds and (p == key or key == "both")]
            n = np.concatenate([r["sizes"] for r in rs])
            ranks = np.concatenate([r["ranks"] for r in rs])
            for tag, j in (("50", 0), ("alt", ARMS.index(arm))):
                sc = competition(ranks[:, j])
                got = [sc[n == m].mean() for m in (1, 2, 3)]
                want = published["scores"][key][tag]
                np.testing.assert_allclose([*got, np.mean(got)], [want["1"], want["2"], want["3"], want["mix"]], atol=1e-9, rtol=0)
        checks[arm] = {"sha": digest_file(path), "effect": published["steps_diff"]["both: talt vs t50"]}
    return checks


def factorial_values(sc):
    b, x, a, c, m = sc.T
    checkpoint = .5*((m-b)+(a-c))
    weight = .5*((c-b)+(a-m))
    np.testing.assert_allclose(checkpoint+weight, a-b, atol=1e-14)
    return {"replace_minus_base": x-b, "add_minus_base": a-b, "old_weight_only": c-b, "mix_at_original_weight": m-b,
            "late_at_double_weight": a-c, "weight_at_mixed_checkpoints": a-m,
            "attributed_checkpoint_mix": checkpoint, "attributed_family_weight": weight}


def factorial(ids, sizes, scores, boot, one_only=False):
    contrasts = factorial_values(scores)
    data = np.column_stack(list(contrasts.values()))
    out = {}
    # Each owner in each split has three single queries and one of each larger size.
    for name, mask in (("single", sizes == 1), ("two", sizes == 2), ("three", sizes == 3)):
        if name != "single" and one_only:
            continue
        out[name] = dict(zip(contrasts, boot.means(ids[mask], data[mask])))
    if not one_only:
        per_owner = np.zeros((1200, data.shape[1]))
        for n in (1, 2, 3):
            use = sizes == n
            cnt = np.bincount(ids[use], minlength=1200)
            if not np.all(cnt == cnt[0]) or cnt[0] == 0:
                raise ValueError("unbalanced owner/query sizes")
            np.add.at(per_owner, ids[use], data[use] / (3*cnt[0]))
        out["mix"] = dict(zip(contrasts, boot.means(np.arange(1200), per_owner)))
    return out


def outcome(ids, ranks, wins, shifts, boot):
    old = ranks[:, 0]
    wrong = old == 1
    out = {}
    for arm in ("replace", "add", "weight", "mix"):
        i = ARMS.index(arm)
        new = ranks[:, i]
        pass_rival = wins[:, i]
        if np.any((new == 0) & ~pass_rival) or np.any(wrong & wins[:, 0]):
            raise ValueError("candidate order contradicts full ranking")
        top = wrong & (new == 0)
        third = wrong & pass_rival & (new > 0)
        behind = wrong & ~pass_rival
        diff = shifts["change"][:, i]
        effect = {"rescued_any_rank": int(((old > 0) & (new == 0)).sum()),
                  "broken_first": int(((old == 0) & (new > 0)).sum()), "old_second": int(wrong.sum()),
                  "old_second_owners": len(np.unique(ids[wrong])), "from_second_to_first": int(top.sum()), "passed_rival_third_wins": int(third.sum()),
                  "still_behind_rival": int(behind.sum()), "rescued_from_below_second": int(((old > 1) & (new == 0)).sum())}
        if effect["from_second_to_first"] + effect["passed_rival_third_wins"] + effect["still_behind_rival"] != effect["old_second"]:
            raise ValueError("old rank-2 outcomes fail to partition")
        paired = {"recovery_rate": (new[wrong] == 0), "pass_old_rival_rate": pass_rival[wrong]}
        for name, matrices in shifts.items():
            delta = matrices[wrong, i]
            paired.update({name+"_true": delta[:, 0], name+"_rival_fall": -delta[:, 1],
                           name+"_margin": delta[:, 0]-delta[:, 1]})
        paired["current_minus_sibling_margin"] = paired["change_margin"]-paired["sibling_margin"]
        paired["rival_fall_minus_true_rise"] = paired["change_rival_fall"]-paired["change_true"]
        effect["on_old_second"] = dict(zip(paired, boot.means(ids[wrong], np.column_stack(list(paired.values())))))
        effect["true_lift_all_singles"] = boot.means(ids, diff[:, 0])[0]
        out[arm] = effect
    return out


def summarize(recs, seeds, boot, masks=None):
    items = [(p, s, recs[p, s]) for s in seeds for p in POOLS]
    stack = lambda k: np.concatenate([r[k] for _, _, r in items])
    ids = np.concatenate([r["truth"]+600*POOLS.index(p) for p, _, r in items])
    sizes, ranks, keys = (stack(k) for k in ("sizes", "ranks", "keys"))
    one = sizes == 1
    qids, qranks = ids[one], ranks[one]
    y = np.ones(len(qids), bool) if masks is None else np.concatenate([masks[p, s] for p, s, _ in items])
    shifts = {k: stack(k)[y] for k in ("change", "common", "sibling")}
    counter = outcome(qids[y], qranks[y], stack("wins")[y], shifts, boot)
    factors = (factorial(ids, sizes, competition(ranks), boot) if masks is None else
               factorial(qids[y], np.ones(y.sum()), competition(qranks[y]), boot, one_only=True))
    return {"observations": int(y.sum()), "distinct_single_games": len(set(keys[y])), "query_owners": len(set(qids[y])),
            "baseline_single": float(competition(qranks[y, 0]).mean()), "factorial": factors, "transitions": counter}


def main():
    budget = guard()
    m = read_manifest(RUN, verify=False)
    source = Path(m["source_root"])
    games = [g for g in m["games"] if g["pool"] in POOLS]
    if len({physical_signature(g) for g in games}) != 12000:
        raise ValueError("ambiguous physical candidate games")
    recs, provenance = {}, {}
    for p in POOLS:
        path = CACHE / f"inputs_{p}.json"
        info = json.loads(path.read_text())
        if info["plan"] != digest_file(PLAN) or info["manifest"] != digest_file(RUN / "manifest.json"):
            raise ValueError("protocol or source manifest changed")
        provenance[p] = digest_file(path)
        for s in range(6):
            row = CACHE / f"{p}_s{s}.npz"
            if digest_file(row) != info["splits"][str(s)]["records"]:
                raise ValueError("stored decisions changed")
            with np.load(row) as z:
                recs[p, s] = {k: z[k] for k in ("truth","sizes","ranks","keys","rival","leaders","wins","change","common","sibling")}
    old, new = range(4), (4, 5)
    checked = {"0_3": official_checks(source, recs, old), "4_5": official_checks(source, recs, new)}
    seen = {k for s in old for p in POOLS for k in recs[p,s]["keys"]}
    masks = {}
    for s in new:
        for p in POOLS:
            keys = recs[p,s]["keys"]
            masks[p,s] = np.array([k not in seen for k in keys])
            seen.update(keys)
    boot = OwnerBootstrap()
    report = {"published": checked, "0_3": summarize(recs, old, boot), "4_5": summarize(recs, new, boot),
              "unique_new": summarize(recs, new, boot, masks), "physical_candidates": len(games),
              "plan": digest_file(PLAN), "inputs": provenance, "budget": budget}
    atomic_json(OUT, report)
    print("verified all published size scores", report["4_5"]["observations"], report["unique_new"]["observations"], flush=True)


if __name__ == "__main__":
    main()
