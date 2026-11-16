"""Retrospective, context-matched event-response identity kernels."""
import hashlib

import numpy as np

LAGS = (1, 2, 4, 8)


def basis(a):
    t, v, h = a["t"], a["value"], a["entropy"]
    return np.column_stack((np.ones(len(t)), a["mover"], (t >= 60) & (t < 150), t >= 150, v, v * v, h, h * h)).astype(float)


def fit_nuisance(references):
    x = np.concatenate([basis(a) for a in references])
    y = np.concatenate([np.column_stack((a["surprisal"], a["residual"])) for a in references]).astype(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite calibration")
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    mu[0], sd[0] = 0., 1.
    x = (x - mu) / sd
    penalty = np.eye(x.shape[1]) * (1e-3 * len(x)); penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    rms = np.maximum(np.sqrt(np.mean((y - x @ coef) ** 2, axis=0)), 1e-6)
    return {"mu": mu, "sd": sd, "coef": coef, "rms": rms,
            "entropy_median": np.median(np.concatenate([a["entropy"] for a in references]))}


def adjust_record(a, fit):
    y = np.column_stack((a["surprisal"], a["residual"])).astype(float)
    adjusted = (y - ((basis(a) - fit["mu"]) / fit["sd"]) @ fit["coef"]) / fit["rms"]
    if not np.isfinite(adjusted).all():
        raise ValueError("nonfinite adjusted signal")
    return dict(a, shock=np.clip(adjusted[:, 0], -3., 3.), adjusted_r=adjusted[:, 1:])


def event_indices(a, lag):
    if lag < 1:
        raise ValueError("positive lag required")
    own = np.flatnonzero(a["mover"] == a["subject"])
    opponent = np.flatnonzero(a["mover"] != a["subject"])
    out = []
    for event in opponent:
        i = np.searchsorted(own, event)
        if i == 0 or i + lag > len(own):
            continue
        if own[i] != event + 1 or a["t"][own[i]] != a["t"][event] + 1:
            raise ValueError("opponent event must immediately precede subject turn")
        out.append((event, own[i - 1], own[i + lag - 1]))
    return np.array(out, dtype=int).reshape(-1, 3)


def permute_shock(shock, cells, key):
    seed = int.from_bytes(hashlib.sha256(("event-response-perm-v1|" + key).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    out, eligible, moved = shock.copy(), 0, 0
    for c in np.unique(cells):
        ix = np.flatnonzero(cells == c)
        if len(ix) > 1:
            perm = rng.permutation(ix)
            out[ix] = shock[perm]
            eligible += len(ix); moved += int((perm != ix).sum())
    return out, {"eligible": eligible, "moved": moved}


def response_moments(a, key, entropy_median):
    r = np.asarray(a["adjusted_r"], float)
    own = np.flatnonzero(a["mover"] == a["subject"])
    if not len(own):
        raise ValueError("no subject moves")
    # A COMMON observed horizon avoids different censoring or permutations
    # changing the joint four-lag signal between E and its negative control.
    events = event_indices(a, max(LAGS))
    e, prior = events[:, 0], events[:, 1]
    cells = np.digitize(a["t"][e], [60, 150]) * 6 + np.digitize(a["value"][e], [-1 / 3, 1 / 3]) * 2 + (a["entropy"][e] >= entropy_median)
    shock = np.asarray(a["shock"][e], float).copy()
    for c in np.unique(cells):
        ix = cells == c
        shock[ix] -= shock[ix].mean()
    perm, meta = permute_shock(shock, cells, key)
    pre_index = np.searchsorted(own, prior)
    delta = np.stack([r[own[pre_index + lag]] - r[prior] for lag in LAGS])
    n = np.full(len(LAGS), len(e), dtype=int)
    return {"E_num": np.einsum("e,led->ld", shock, delta), "P_num": np.einsum("e,led->ld", perm, delta),
            "E_den": np.full(len(LAGS), shock @ shock), "P_den": np.full(len(LAGS), perm @ perm),
            "D_num": delta.sum(1), "N": n, "M_sum": r[own].sum(0), "M_n": len(own),
            "events": len(e), "permutable": meta["eligible"], "moved": meta["moved"],
            "shock_energy": float(shock @ shock)}


def pool_kernel(rows, arm):
    if not rows:
        raise ValueError("empty kernel pool")
    if arm == "M":
        return sum(r["M_sum"] for r in rows) / sum(r["M_n"] for r in rows)
    n = sum(r["N"] for r in rows)
    if arm == "D":
        return (sum(r["D_num"] for r in rows) / np.maximum(n[:, None], 1)).ravel()
    if arm not in ("E", "P"):
        raise ValueError("unknown event arm")
    den = sum(r[arm + "_den"] for r in rows) + n
    return (sum(r[arm + "_num"] for r in rows) / np.maximum(den[:, None], 1)).ravel()


def evaluate_repeatability(games, features, draws=5000):
    owners = sorted({g["owner"] for g in games})
    ref = {p: [g["key"] for g in games if g["owner"] == p and g["role"] == "reference"] for p in owners}
    held = {p: [g["key"] for g in games if g["owner"] == p and g["role"] == "heldout"] for p in owners}
    if any(len(ref[p]) != 7 or len(held[p]) != 3 for p in owners):
        raise ValueError("need fixed 7/3 calibration split")
    reference_keys = [k for ks in ref.values() for k in ks]
    scale = {}
    for arm in ("M", "D", "E", "P"):
        sample = np.stack([pool_kernel([features[k]], arm) for k in reference_keys])
        scale[arm] = np.mean(sample ** 2, axis=0)
    ep = np.sqrt(np.maximum((scale["E"] + scale["P"]) / 2, 1e-12))
    scale = {a: ep if a in ("E", "P") else np.sqrt(np.maximum(s, 1e-12)) for a, s in scale.items()}
    questions, truth, sizes = [], [], []
    for i, owner in enumerate(owners):
        h = held[owner]
        qs = [[k] for k in h] + [h[:2], h]
        questions.extend(qs); truth.extend([i] * 5); sizes.extend(map(len, qs))
    truth, sizes = np.array(truth), np.array(sizes)
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    measures, per_owner, per_margin, curves, curve_deltas, query_deltas = {}, {}, {}, {}, {}, {}
    for arm in ("M", "D", "E", "P"):
        x = unit(np.stack([pool_kernel([features[k] for k in ref[p]], arm) for p in owners]) / scale[arm])
        y = unit(np.stack([pool_kernel([features[k] for k in q], arm) for q in questions]) / scale[arm])
        score = y @ x.T
        own = score[np.arange(len(score)), truth]
        rank = ((score > own[:, None]) | ((score == own[:, None]) & (np.arange(len(owners))[None] < truth[:, None]))).sum(1)
        rs = np.where(rank < 5, np.exp(-rank.astype(float)), 0.)
        margin = own - (score.sum(1) - own) / (len(owners) - 1)
        def owner_balanced(values):
            return np.stack([np.array([values[(truth == i) & (sizes == n)].mean() for n in (1, 2, 3)]) for i in range(len(owners))]).mean(1)
        per_owner[arm], per_margin[arm] = owner_balanced(rs), owner_balanced(margin)
        measures[arm] = {"rank_score": float(per_owner[arm].mean()), "top1": float(owner_balanced(rank == 0).mean()), "identity_margin": float(per_margin[arm].mean()),
                         "by_size": {str(n): {"rank_score": float(rs[sizes == n].mean()), "top1": float((rank[sizes == n] == 0).mean()), "identity_margin": float(margin[sizes == n].mean())} for n in (1, 2, 3)}}
        curves[arm] = {"7": {"rank_score": measures[arm]["rank_score"], "top1": measures[arm]["top1"]}}
        query_deltas[arm] = np.array([rs[(truth == i) & (sizes == 3)].mean() - rs[(truth == i) & (sizes == 1)].mean() for i in range(len(owners))])
        for k in (1, 3):
            xx = unit(np.stack([pool_kernel([features[key] for key in ref[p][:k]], arm) for p in owners]) / scale[arm])
            ss = y @ xx.T
            oo = ss[np.arange(len(ss)), truth]
            rr = ((ss > oo[:, None]) | ((ss == oo[:, None]) & (np.arange(len(owners))[None] < truth[:, None]))).sum(1)
            pp = owner_balanced(np.where(rr < 5, np.exp(-rr.astype(float)), 0.))
            curves[arm][str(k)] = {"rank_score": float(pp.mean()), "top1": float(owner_balanced(rr == 0).mean())}
            if k == 1:
                curve_deltas[arm] = per_owner[arm] - pp
    rng = np.random.default_rng(20261117)
    ix = rng.integers(len(owners), size=(draws, len(owners)))
    difference = per_owner["E"] - per_owner["P"]
    ci = np.quantile(difference[ix].mean(1), [.025, .975]).tolist()
    mci = np.quantile(per_margin["E"][ix].mean(1), [.025, .975]).tolist()
    rows = list(features.values())
    coverage = float(np.mean([r["events"] >= 20 for r in rows]))
    events = sum(r["events"] for r in rows)
    fraction = sum(r["permutable"] for r in rows) / max(events, 1)
    passed = coverage >= .95 and fraction >= .60 and ci[0] > 0 and mci[0] > 0
    learning = {a: {"reference_counts": curves[a], "reference7_minus1": {"diff": float(curve_deltas[a].mean()), "ci95": np.quantile(curve_deltas[a][ix].mean(1), [.025, .975]).tolist()},
                    "query3_minus1": {"diff": float(query_deltas[a].mean()), "ci95": np.quantile(query_deltas[a][ix].mean(1), [.025, .975]).tolist()}} for a in measures}
    return {"arms": measures, "E_minus_P": {"diff": float(difference.mean()), "ci95": ci}, "E_identity_margin_ci95": mci,
            "learning_curves": learning,
            "coverage_20_events": coverage, "permutable_fraction": fraction, "events": events,
            "moved_fraction": sum(r["moved"] for r in rows) / max(events, 1),
            "median_events": float(np.median([r["events"] for r in rows])),
            "zero_shock_energy_games": sum(r["shock_energy"] < 1e-12 for r in rows),
            "verdict": "pilot_repeatability_supported" if passed else "pilot_repeatability_not_established",
            "scope": "100 calibration players used in projection training; owner bootstrap conditional on fixed reference bank/nuisance/metric; not baseline complementarity or causal response", "draws": draws}
