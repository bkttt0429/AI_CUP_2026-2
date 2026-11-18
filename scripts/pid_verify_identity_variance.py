"""Independent reconstruction of published rank, uncertainty, and equal-budget contrasts."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, ROOT, POOLS, baseline_path, atomic_json, digest_file, guard

OUT = ROOT / "results/final"


def read_store(path, sha=None):
    if sha is not None and digest_file(path) != sha:
        raise ValueError(f"source changed: {path}")
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=2e-9)


def scale(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def win(x):
    x = np.asarray(x)
    return (x > 0).astype(float) + .5 * (x == 0)


def grade(r):
    return np.select([r == k for k in range(5)], np.exp(-np.arange(5)), default=0.)


def weights(groups, seed):
    rng = np.random.default_rng(seed)
    return np.hstack([rng.multinomial(n, np.full(n, 1 / n), 5000) for n in groups])


def interval(x, idx, w, record):
    n = np.bincount(idx, minlength=w.shape[1])
    s = np.bincount(idx, weights=x, minlength=w.shape[1])
    close(record["mean"], np.mean(x))
    close(record["ci95"], np.percentile((w @ s) / (w @ n), [2.5, 97.5]))


def direct_margin(matrix):
    eigen, axes = np.linalg.eigh(matrix)
    if eigen.min() < -1e-8:
        raise ValueError("indefinite saved Gram matrix")
    x = axes * np.sqrt(np.maximum(eigen, 0.))
    left, right = x[:7].mean(0), x[7:14].mean(0)
    vec, query = left - right, x[-1] - (left + right) * .5
    iso = query @ scale(vec)
    out = []
    for start in (0, 14):
        z = x[start:start + 14].reshape(2, 7, -1)
        residual = (z - z.mean(1, keepdims=True)).reshape(14, -1)
        scatter = residual.T @ residual
        w = np.linalg.solve(np.eye(len(vec)) + scatter / (np.trace(scatter) / 12), vec)
        out.append(query @ scale(w))
    return [out[0] - iso, out[1] - iso, iso, out[0]]


def accepted(final):
    historic = set()
    rank_count = 0
    for s in (1, 3):
        r = final["splits"][str(s)]
        d, ids, pair_rows, sizes = [], [], [], []
        for pi, p in enumerate(POOLS):
            archive = read_store(CACHE / f"identity_{p}_{s}.npz", r["archives"][p])
            b = read_store(baseline_path(p, s))
            historic.update(b["keys"])
            use = np.flatnonzero(np.arange(len(b["truth"])) % 3 != 0)
            full_order = np.argsort(-b["scores"], kind="stable")
            np.testing.assert_array_equal(full_order[:, :2], archive["pair"])
            hold = archive["selected"]
            assert len(set(hold.ravel())) == hold.size
            ordered = [(int(b["truth"][i]), int(full_order[i, 0])) for i in range(0, len(b["truth"]), 3) if full_order[i, 1] == b["truth"][i]]
            ordered.sort(key=lambda edge: hashlib.sha256(f"identity-pair-v1|{p}|{b['owners'][edge[0]]}|{b['owners'][edge[1]]}".encode()).digest())
            used, actual = set(), []
            for a, c in ordered:
                if a not in used and c not in used:
                    used.update([a, c])
                    actual.append((a, c))
            np.testing.assert_array_equal(actual, hold)
            names = archive["pair_names"].tolist()
            for j, (a, c) in enumerate(hold):
                aq = 3 * a + np.arange(1, 3)
                cq = 3 * c + np.arange(1, 3)
                a_rank = np.argmax(full_order[aq] == a, axis=1)
                rival_rank = np.argmax(full_order[aq] == c, axis=1)
                fresh = a_rank < rival_rank
                ck = {"new_A_beats_fixed_B": fresh.mean(), "full_roster_A_top1": np.mean(a_rank == 0),
                      "same_wrong_B_still_top1": np.mean(rival_rank == 0),
                      "different_wrong_player_top1": np.mean((a_rank != 0) & (rival_rank != 0)),
                      "new_B_beats_fixed_A": np.mean(np.argmax(full_order[cq] == c, axis=1) < np.argmax(full_order[cq] == a, axis=1))}
                for key, val in ck.items():
                    close(archive["pair_values"][j, names.index(key)], val)
            sizes.append(len(hold))
            pair_rows.extend(archive["pair_values"])
            for i in (0, 123, 601, 1118, 1799):
                close(direct_margin(archive[f"gram_{i}"]), archive["raw"][i])
            base = np.argmax(full_order[use] == b["truth"][use, None], axis=1)
            np.testing.assert_array_equal(base, archive["base_rank"])
            diff = []
            for j in range(4):
                order = full_order[use].copy()
                swap = archive["gap"][:, j] < 0
                order[swap, :2] = order[swap, 1::-1]
                rank = np.argmax(order == b["truth"][use, None], axis=1)
                np.testing.assert_array_equal(rank, archive["new_rank"][:, j])
                diff.append(grade(rank) - grade(base))
            d.extend(np.stack(diff, axis=1))
            ids.extend(b["truth"][use] + 600 * pi)
            rank_count += 4 * len(use)
        d, ids = np.asarray(d), np.array(ids)
        w = weights([600, 600], 111901)
        for j, arm in enumerate(("residual", "placebo", "isotropic", "fisher")):
            interval(d[:, j], ids, w, r["ranking"]["arms"][arm])
        pw = weights(sizes, 111901)
        for j, k in enumerate(names):
            interval(np.asarray(pair_rows)[:, j], np.arange(sum(sizes)), pw, r["persistence"][k])
    return historic, rank_count


def independent(r):
    v = read_store(CACHE / "crossed.npz", r["archive"])
    n = len(v["owners"])
    assert len(set(v["signatures"].ravel())) == 24 * n and len(set(v["pairs"].ravel())) == v["pairs"].size
    raw = read_store(Path(r["audit"]["embedding_path"]), r["audit"]["embedding_sha"])
    lookup = dict(zip(raw["key"], raw["emb"]))
    enc = scale(np.array([[lookup[k] for k in row] for row in v["keys"]], dtype=float))
    ds, dr, gap_delta = [], [], []
    geom = []
    for j, (a, b) in enumerate(v["pairs"]):
        rr = np.stack([enc[[a,b], sl].mean(1) for sl in (slice(8, 15), slice(15, 22))])
        gap = np.empty((2, 2, 2))
        for t, who in enumerate((a, b)):
            for panel in range(2):
                p, q = scale(rr[panel])
                for game in range(2):
                    gap[t, panel, game] = ((enc[who, 22 + game] @ p) - (enc[who, 22 + game] @ q)) * (1 - 2*t)
        close(gap, v["margins"][j])
        ref = (np.sign(gap[:, 1]) != np.sign(gap[:, 0])).mean()
        query = (np.sign(gap[..., 1]) != np.sign(gap[..., 0])).mean()
        ds.append(query)
        dr.append(ref)
        gap_delta.append(query - ref)
        geom.append((rr[0,0] - rr[0,1]) @ (rr[1,0] - rr[1,1]))
    w = weights([len(ds)], 111901)
    for name, values in (("query_flip", ds), ("reference_flip", dr), ("query_minus_ref_flip", gap_delta), ("cross_split_separation", geom)):
        interval(np.asarray(values), np.arange(len(ds)), w, r["stats"][name])
    return v


def sessions(crossed, summary, all_roster=False):
    design = summary["full_roster_whitening_sensitivity"] if all_roster else summary
    v = read_store(CACHE / ("sessions_all_roster.npz" if all_roster else "sessions.npz"), design["archive"])
    zp = read_store(Path(summary["phase_source"]["path"]), summary["phase_source"]["sha"])
    table = dict(zip(zp["key"], zp["PT"]))
    x = np.array([[table[k][:256] for k in row] for row in crossed["keys"]], dtype=float).reshape(*crossed["keys"].shape, 2, 128)
    # Confirm the whitening covariance using a different accumulation order.
    ok = (v["counts"] == [30, 45]).all(-1)
    sums = np.zeros((2,128,128))
    df = 0
    roster = np.arange(len(x)) if all_roster else np.unique(crossed["pairs"])
    assert len(roster) == design["whitening_owners"]
    for i in roster:
        block = x[i, :7]
        ref = block[ok[i, :7]]
        if len(ref) >= 2:
            df += len(ref) - 1
            residual = ref - ref.mean(0)
            sums += np.einsum("gpd,gpe->pde", residual, residual)
    for phase in (0,1):
        cov = sums[phase] / df
        reg = .9 * cov + .1 * np.trace(cov) / 128 * np.eye(128)
        w = v["whitener"][phase]
        close(w.T @ reg @ w, np.eye(128))
    z = np.einsum("ngpi,pij->ngpj", x, v["whitener"], optimize=True)
    for i, (pair, panel, owner, colour) in enumerate(v["details"]):
        a, b = crossed["pairs"][pair]
        sl = slice(8 + 7*panel, 15 + 7*panel)
        h = [z[u, sl][ok[u, sl]].mean(0) for u in (a,b)]
        axis = scale(h[0] - h[1])
        gap = np.einsum("gpd,pd->gp", z[owner, 22:24] - (h[0] + h[1]) / 2, axis)
        close(gap if owner == a else -gap, v["raw"][i])
        assert ok[owner, 22:24].all()
    raw = v["raw"]
    within = win(raw.sum(-1)).mean(1)
    across = win(raw[:, :, 0] + raw[:, ::-1, 1]).mean(1)
    close(within, v["data"][:,0])
    close(across, v["data"][:,1])
    for mask, scope in ((np.ones(len(raw), bool), "all"), (v["details"][:,-1] == 1, "same_colour")):
        clusters, ids = np.unique(v["details"][mask,0], return_inverse=True)
        interval((across - within)[mask], ids, weights([len(clusters)],111917), design[scope]["gain"])
    return len(raw)


def confirm(historic, c):
    seen = set(historic)
    items, include = [], []
    for s in (0,2):
        for p in POOLS:
            z = read_store(CACHE / f"confirm_{p}_{s}.npz", c["archives"][f"{p}/{s}"])
            for i, key in enumerate(z["keys"]):
                base, new = z["rows"][i, 1:3]
                fresh = key not in seen
                assert fresh == z["fresh"][i]
                seen.add(key)
                truth = z["rows"][i, 0] % 600
                top = z["order"][i]
                assert (base < 2) == (truth in top)
                if base < 2:
                    assert top[base] == truth
                assert new == (1 - base if base < 2 and z["fitted_gaps"][i,3] < 0 else base)
                include.append(fresh)
            items.extend(z["rows"])
    items, include = np.array(items), np.array(include)
    w = weights([600,600],111912)
    for vals, name in ((items, "combined"), (items[include], "unique_new_as_queries")):
        ids, base, fisher, iso = vals.T
        for key, ref in (("vs_base", base), ("vs_isotropic", iso)):
            interval(grade(fisher)-grade(ref), ids, w, c[name][key])
        assert c[name]["rows"] == len(ids)
    return len(items), int(include.sum())


def shifts(report, original):
    for seed in (1, 3):
        setting = report["splits"][str(seed)]
        groups, components = [], []
        for pool in POOLS:
            b = read_store(baseline_path(pool, seed))["scores"]
            hold = read_store(CACHE / f"identity_{pool}_{seed}.npz", original["splits"][str(seed)]["archives"][pool])["selected"]
            groups.append(len(hold))
            for a, other in hold:
                vals = []
                for q in range(3):
                    row = b[3*a + q]
                    background = np.delete(row, np.sort([a, other]))
                    vals.append([(row[i] - np.mean(background)) / np.std(background) for i in (a, other)])
                original_a, original_b = vals[0]
                records = []
                for new_a, new_b in vals[1:]:
                    both = int(new_a > new_b or (new_a == new_b and a < other))
                    a1 = int(new_a > original_b or (new_a == original_b and a < other))
                    b1 = int(original_a > new_b or (original_a == new_b and a < other))
                    records.append([new_a - original_a, original_b - new_b, a1, b1, both, (a1 + both - b1) / 2, (b1 + both - a1) / 2])
                components.append(np.mean(records, axis=0))
        columns = ("true_score_rise", "rival_score_fall", "only_true_replaced_wins", "only_rival_replaced_wins",
                   "both_replaced_wins", "true_recovery_share", "rival_recovery_share")
        for k, x in zip(columns, np.asarray(components).T):
            interval(x, np.arange(len(x)), weights(groups, 111920), setting["stats"][k])


def main():
    budget = guard()
    files = [OUT / f"identity_{name}_20261118.json" for name in ("crossed","finalists","sessions","fisher_confirmation","score_shift")]
    cr, fi, se, co, sh = [json.loads(f.read_text()) for f in files]
    hist, nr = accepted(fi)
    new = independent(cr)
    panels = sessions(new, se)
    assert sessions(new, se, all_roster=True) == panels
    rows, unique = confirm(hist, co)
    shifts(sh, fi)
    out = {"passed": True, "full_orders_recomputed": nr, "direct_gram_reconstructions": 20, "disjoint_external_pairs": len(new["pairs"]),
           "equal_budget_player_panels": panels, "confirmation_queries": rows, "newly_queried_unique_games": unique,
           "report_hashes": {f.name:digest_file(f) for f in files}, "budget": budget}
    atomic_json(OUT / "identity_variance_check_20261118.json", out)
    print("INDEPENDENT AUDIT PASSED", nr, rows, unique)


if __name__ == "__main__":
    main()
