"""Compare the same 75 decisions drawn from one or two games of the same anonymous player."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_identity_inputs import CACHE, ROOT, RUN, atomic_json, atomic_npz, digest_file, guard, read_manifest
from aicup_pid.paths import AICUP1

PLAN = ROOT / "docs/superpowers/plans/2026-11-18-identity-confirmations.md"
OUT = ROOT / "results/final/identity_sessions_20261118.json"


def exchange(gaps):
    """Rows are two games; columns are early and middle evidence along fixed phase axes."""
    x = np.asarray(gaps)
    straight = x.sum(1)
    crossed = x[:, 0] + x[::-1, 1]
    if not np.isclose(straight.sum(), crossed.sum(), atol=1e-10):
        raise ValueError("movement evidence not balanced")
    win = lambda v: float(np.mean((v > 0) + .5 * (v == 0)))
    return {"same_game": win(straight), "two_games": win(crossed), "gain": win(crossed) - win(straight),
            "excess_covariance": float(.5 * (x[0, 0] - x[1, 0]) * (x[0, 1] - x[1, 1]))}


def whitening(z, valid):
    r = []
    for x, take in zip(z, valid):
        if take.sum() >= 2:
            y = x[take]
            r.append(y - y.mean(0))
    if len(r) < 2:
        raise ValueError("insufficient discovery owners")
    df = sum(len(x) - 1 for x in r)
    centered = np.vstack(r)
    mats = []
    for x in centered.transpose(1, 0, 2):
        cov = (x.T @ x) / df
        reg = .9 * cov + .1 * cov.trace() / cov.shape[0] * np.eye(cov.shape[0])
        eigen, axes = np.linalg.eigh(reg)
        if eigen.min() <= 0:
            raise ValueError("singular within-player covariance")
        mats.append((axes / np.sqrt(eigen)[None, :]) @ axes.T)
    return np.array(mats)


def source_rows(keys, phase_path, owners):
    needed = {k: i for i, k in enumerate(keys.ravel())}
    count = np.full((keys.size, 2), -1, dtype=int)
    colours = np.full(keys.size, "", dtype="U1")
    for directory in ("trunk_vec", "trunk_vec_x40", "trunk_vec_x80"):
        fold = AICUP1 / "results" / directory / "fold2"
        for tab in sorted(fold.glob("games_s*_p*.csv")):
            with open(tab) as f:
                meta = list(csv.DictReader(f))
            hits = [(i, needed[x["game_key"]]) for i, x in enumerate(meta) if x["game_key"] in needed]
            if not hits:
                continue
            with np.load(tab.with_name(tab.name.replace("games_", "moves_").replace(".csv", ".npz")), allow_pickle=False) as z:
                subject = z["target"].astype(bool)
                game, t = z["game_row"][subject], z["t"][subject]
            early = np.bincount(game[t < 60], minlength=len(meta))
            middle = np.bincount(game[(t >= 60) & (t < 150)], minlength=len(meta))
            for i, j in hits:
                if count[j, 0] >= 0 or meta[i]["player_id"] != owners[j // keys.shape[1]]:
                    raise ValueError("duplicate or misattributed phase metadata")
                count[j] = [early[i], middle[i]]
                colours[j] = meta[i]["color"]
    if np.any(count < 0) or not np.isin(colours, ["B", "W"]).all():
        raise ValueError("missing phase metadata")
    with np.load(phase_path, allow_pickle=False) as z:
        lookup = dict(zip(z["key"], range(len(z["key"]))))
        if len(lookup) != len(z["key"]):
            raise ValueError("nonunique phase vector keys")
        idx = [lookup[k] for k in keys.ravel()]
        obs = z["PT"][idx, :256].astype(float).reshape(*keys.shape, 2, 128)
    return obs, count.reshape(*keys.shape, 2), colours.reshape(keys.shape)


def estimate(data, pair_ids, keep):
    x = data[keep]
    ids = pair_ids[keep]
    unique, groups = np.unique(ids, return_inverse=True)
    if len(unique) < 10:
        return {"rows": len(x), "pairs": len(unique)}
    rng = np.random.default_rng(111917)
    weight = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), 5000)
    den = weight @ np.bincount(groups)
    names = ("same_game", "two_games", "gain", "excess_covariance")
    result = {"player_panel_rows": len(x), "pairs": len(unique)}
    for j, name in enumerate(names):
        estimates = (weight @ np.bincount(groups, weights=x[:, j])) / den
        result[name] = {"mean": float(x[:, j].mean()), "ci95": np.quantile(estimates, [.025, .975]).tolist()}
    return result


def main():
    guard()
    parent = json.loads((ROOT / "results/final/identity_crossed_20261118.json").read_text())
    if parent["archive"] != digest_file(CACHE / "crossed.npz"):
        raise ValueError("independent roster changed")
    with np.load(CACHE / "crossed.npz", allow_pickle=False) as z:
        pairs, keys, owners = z["pairs"], z["keys"], z["owners"]
    source = Path(read_manifest(RUN, verify=False)["source_root"])
    phase = source / "data/pid_v2_phase_rows/stored_f2.npz"
    obs, n, colour = source_rows(keys, phase, owners)
    valid = (n == [30, 45]).all(-1)
    whitener = whitening(obs[:, :7], valid[:, :7])
    whitened = np.einsum("ngpd,pde->ngpe", obs, whitener, optimize=True)
    raw, details, results = [], [], []
    for u, (a, b) in enumerate(pairs):
        for panel, sl in enumerate((slice(8, 15), slice(15, 22))):
            good = valid[[a, b], sl]
            if np.any(good.sum(1) < 3):
                continue
            refs = whitened[[a, b], sl]
            m, k = [x[g].mean(0) for x, g in zip(refs, good)]
            axis = m - k
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            for i, owner in enumerate((a, b)):
                if not valid[owner, 22:24].all():
                    continue
                q = whitened[owner, 22:24] - (m + k) / 2.
                gaps = (q * axis[None]).sum(-1) * (1. if i == 0 else -1.)
                measurement = exchange(gaps)
                raw.append(gaps)
                details.append([u, panel, owner, int(colour[owner, 22] == colour[owner, 23])])
                results.append([measurement[k] for k in ("same_game", "two_games", "gain", "excess_covariance")])
    data, details = np.asarray(results), np.asarray(details)
    raw = np.asarray(raw)
    full = estimate(data, details[:, 0], np.ones(len(data), bool))
    same_colour = estimate(data, details[:, 0], details[:, 3] == 1)
    path = CACHE / "sessions.npz"
    atomic_npz(path, raw=raw, details=details, data=data, counts=n, colour=colour, whitener=whitener)
    atomic_json(OUT, {"plan": digest_file(PLAN), "code": digest_file(__file__), "archive": digest_file(path),
                      "phase_source": {"path": str(phase), "sha": digest_file(phase)}, "valid_games": int(valid.sum()),
                      "owners_with_evaluation": len(set(details[:, 2])), "all": full, "same_colour": same_colour,
                      "early_decisions": 30, "middle_decisions": 45})
    print("equal-budget phase comparison", full, flush=True)


if __name__ == "__main__":
    main()
