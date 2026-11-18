"""Owner-clustered ranking effects and margin-matched within-game diagnostics."""
import numpy as np

from pid_tactical_score import competition


def summarize(records, arms):
    def cat(k):
        return np.concatenate([r[k] for r in records])

    base, new = cat("base"), cat("new")
    ids, seed, raw = cat("cluster"), cat("seed"), cat("raw")
    keys, bins = cat("keys"), cat("bin")
    views = np.concatenate([r["nominees"] for r in records], axis=1)
    gaps = views[:, :, :, 0] - views[:, :, :, 1]
    group = ids // 600
    all_rows = np.ones(len(ids), dtype=bool)
    rng = np.random.default_rng(111819)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1/600), size=5000) for _ in range(2)])
    def cluster(weight, mask):
        return np.bincount(ids[mask], weights=np.asarray(weight)[mask], minlength=1200)
    def interval(x, mask):
        if not mask.any():
            return {"rows": 0}
        den = boot @ cluster(np.ones(len(ids)), mask)
        total = boot @ cluster(x, mask)
        return {"mean": float(np.mean(x[mask])), "ci95": np.quantile(total[den>0] / den[den>0], [.025, .975]).tolist(),
                "rows": int(mask.sum())}

    def arm_stats(mask):
        result = {}
        for j, arm in enumerate(arms):
            delta = competition(new[:, j]) - competition(base)
            result[arm] = {**interval(delta, mask), "rescued": int(np.sum(mask & (base == 1) & (new[:, j] == 0))),
                           "broken": int(np.sum(mask & (base == 0) & (new[:, j] == 1))),
                           "top1": float(np.mean(new[mask, j] == 0)), "rank_score": float(competition(new[mask, j]).mean())}
        return result

    combined = arm_stats(all_rows)
    split_stats = {str(s): arm_stats(seed == s) for s in np.unique(seed)}
    controls = {}
    for real, ctrl in (("floor", "sham_floor"), ("persistent", "sham_persistent"), ("template", "sham_template"),
                       ("persistent", "intact"), ("sham_persistent", "intact")):
        controls[real + "_minus_" + ctrl] = interval(competition(new[:, arms.index(real)]) -
                                                      competition(new[:, arms.index(ctrl)]), all_rows)
    # Within shared cells, match correct nominations to the observed error distribution.
    collapse = raw[:, arms.index("sham_floor")] - raw[:, arms.index("floor")]
    cells = {}
    for s in np.unique(seed):
        for p in range(2):
            for q in range(5):
                cell = (group == p) & (seed == s) & (bins == q)
                err, ok = cell & (base == 1), cell & (base == 0)
                if err.any() and ok.any():
                    cells[f"{s}/{p}/{q}"] = [cluster(x, mask) for mask in (err, ok)
                                             for x in (np.ones(len(base)), collapse)]
    counts = np.array(list(cells.values()))
    def matched(weights):
        ne, se, nc, sc = (weights @ counts[:, i, :].T for i in range(4))
        valid = (ne > 0) & (nc > 0)
        n = np.maximum((ne * valid).sum(1), 1.)
        error_mean = (se * valid).sum(1) / n
        correct_mean = ((sc / np.maximum(nc, 1)) * ne * valid).sum(1) / n
        return np.stack([error_mean, correct_mean, error_mean-correct_mean], axis=1)
    point = matched(np.ones((1, 1200)))[0]
    ci = np.quantile(matched(boot), [.025, .975], axis=0).T
    diagnosis = {"counts": {k: [int(v[0].sum()), int(v[2].sum())] for k, v in cells.items()},
                 "errors_total": int((base == 1).sum()), "errors_matched": int(counts[:, 0].sum()),
                 "correct_matched": int(counts[:, 2].sum()), "correct_total": int((base == 0).sum()),
                 **{label: {"mean": float(x), "ci95": ci[i].tolist()}
                    for i, (label, x) in enumerate(zip(("errors", "matched_correct", "difference"), point))}}

    def flag_stats(mask):
        return {"rows": int(mask.sum()), "runner_is_true": int((mask & (base == 1)).sum()),
                "winner_is_true": int((mask & (base == 0)).sum()), "neither": int((mask & (base > 1)).sum()),
                "runner_rate": interval((base == 1).astype(float), mask)}
    flags = {"intact_both_runner": flag_stats((gaps[:, :, 0] < 0).all(0)),
             "quarters_both_runner": flag_stats((gaps[:, :, 1:5] < 0).all(axis=(0, 2))),
             "sham_both_runner": flag_stats((gaps[:, :, 5:9] < 0).all(axis=(0, 2)))}
    standalone = {}
    for arm in arms:
        fr = raw[:, arms.index(arm)]
        hits = np.where(fr == 0, .5, ((fr > 0) == (base == 0)).astype(float))
        standalone[arm] = {k: interval(hits, m) for k, m in (("inside", base < 2), ("error", base == 1), ("correct", base == 0))}
    passed = [k for k in ("floor", "persistent", "template", "intact") if combined[k]["mean"] >= .002 and combined[k]["ci95"][0] > 0]
    return {"rows": len(ids), "unique_games": len(set(keys)), "owners": len(set(ids)), "accepted_rank_score": float(competition(base).mean()),
            "accepted_top1": float((base == 0).mean()), "combined": combined, "per_seed": split_stats, "controls": controls,
            "matched_collapse": diagnosis, "unanimous": flags, "standalone": standalone, "screen_passed": passed}
