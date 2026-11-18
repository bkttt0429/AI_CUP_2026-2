"""Independent checks of candidate indexing, reported decisions and temporal projections."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_corroboration_score import ARMS, BASE_INFO, CACHE, POOLS, ROOT, baseline_path, read_views, score, digest_file, atomic_json, guard
from pid_corroboration_views import RUN, read_manifest
from pid_choice_manifest import seed_split, v2
from pid_tactical_score import competition

OUT = ROOT / "results/final/corroboration_check_20261118.json"


def direct(q, a, b):
    """Explicitly project each coordinate and sum off-diagonal reference-game pairs."""
    def normalized(v):
        return v / np.linalg.norm(v)
    d = normalized(normalized(a[:, 0].mean(0)) - normalized(b[:, 0].mean(0)))
    transform = np.eye(len(d)) - np.outer(d, d)
    def flatten(x):
        centered = x - x.mean(axis=-2, keepdims=True)
        return (centered @ transform).reshape((*x.shape[:-2], -1))
    out = []
    for sl in (slice(1, 5), slice(5, 9)):
        y = flatten(q[sl])
        def individual(ref):
            rr = flatten(ref[:, sl])
            cross = sum(rr[i] @ rr[j] for i in range(7) for j in range(i))
            return (rr @ y).mean() - .5 * cross / 21
        out.append(individual(a)-individual(b))
    return out


def template_direct(q, a, b):
    scores = []
    for sl in (slice(1, 5), slice(5, 9)):
        margins = []
        for i in range(7):
            keep = np.arange(7) != i
            ca, cb = a[keep, 0].mean(0), b[keep, 0].mean(0)
            d = ca / np.linalg.norm(ca) - cb / np.linalg.norm(cb)
            x, aa, bb = [v - v.mean() for v in (q[sl] @ d, a[i, sl] @ d, b[i, sl] @ d)]
            margins.append((x-(aa+bb)*.5) @ (aa-bb))
        scores.append(np.mean(margins))
    return scores


def main():
    guard()
    banks, _ = read_views()
    manifest = read_manifest(RUN, verify=False)
    final = ROOT / "results/final"
    def get(name):
        return json.loads((final / f"corroboration{name}_20261118.json").read_text())
    screen, rep, comp, rival = [get(k) for k in ("", "_replication", "_complement", "_rivals")]
    hashes, counts, checked = {}, {}, 0
    second = []
    for pi, pool in enumerate(POOLS):
        for seed in (1, 3, 0, 2):
            split = seed_split(manifest, pool, seed)
            players = sorted(split["reference"])
            ref = np.array([split["reference"][p] for p in players])
            hold = np.array([split["heldout"][p] for p in players])
            hashes[f"{pool}/{seed}"] = hashlib.sha256("\n".join(ref.ravel()).encode()).hexdigest()
            if seed in (1, 3):
                with np.load(baseline_path(pool, seed)) as z:
                    b = dict(z)
                np.testing.assert_array_equal(ref, b["refs"])
                np.testing.assert_array_equal(hold.ravel(), b["keys"])
                rebuilt, _ = score(banks, b, pool, seed)
                with np.load(CACHE / f"score_{pool}_{seed}.npz") as z:
                    for k, value in rebuilt.items():
                        if np.issubdtype(value.dtype, np.inexact):
                            np.testing.assert_allclose(value, z[k], rtol=1e-6, atol=1e-7)
                        else:
                            np.testing.assert_array_equal(value, z[k])
                for name, cols in (("score", ARMS), ("complement", comp["arms"])):
                    with np.load(CACHE / f"{name}_{pool}_{seed}.npz") as z:
                        ranks, pred, base = (z[k] for k in ("new", "fitted", "base"))
                        for j, arm in enumerate(cols):
                            want = base.copy()
                            change = (pred[:, j] < 0) & (base < 2)
                            want[change] = 1-base[change]
                            np.testing.assert_array_equal(want, ranks[:, j])
                            tally = counts.setdefault(name+"/"+arm, np.zeros(3))
                            tally += [(want == 0).sum() - ((want == 0) & (base == 0)).sum(),
                                      (base == 0).sum() - ((want == 0) & (base == 0)).sum(),
                                      (competition(want)-competition(base)).sum()]
                continue
            with np.load(CACHE / f"rivals_{pool}_{seed}.npz") as z:
                raw, qr, rr, tid, rank = (z[k] for k in ("raw", "keys", "rivals", "cluster", "rank"))
            path = BASE_INFO / f"confirm_{pool}_{seed}.npz"
            with np.load(path) as z:
                archive_keys, archive_order, fresh = z["keys"], z["order"], z["fresh"]
            np.testing.assert_array_equal(hold[:, 1:].ravel(), archive_keys)
            aidx = {k: i for i, k in enumerate(archive_keys)}
            selected = np.array([aidx[k] for k in qr])
            assert fresh[selected].all()
            np.testing.assert_array_equal(rr[:, 0], np.array([p[0] if p[0] != own % 600 else p[1]
                                                               for p, own in zip(archive_order[selected], tid)]))
            np.testing.assert_array_equal(rr[:, 2], np.array([p[0] if p[0] != own % 600 else p[1]
                                                               for p, own in zip(archive_order[selected ^ 1], tid)]))
            with np.load(CACHE / f"fresh_{pool}_{seed}.npz") as z:
                scalar = {k: t for k, t in zip(z["keys"], z["signed"])}
            # Independently reconstruct both temporal classifiers with literal projections.
            for n in np.linspace(0, len(qr)-1, 12, dtype=int):
                owner = int(tid[n] % 600)
                xs, ys = [], []
                for ix, v in banks:
                    a = v[[ix[k] for k in ref[owner]]].astype(float)
                    q = v[ix[qr[n]]].astype(float)
                    bs = [v[[ix[k] for k in ref[o]]].astype(float) for o in rr[n]]
                    xs.append([direct(q, a, b) for b in bs])
                    ys.append(template_direct(q, a, bs[0]))
                np.testing.assert_allclose(raw[n], np.mean(xs, axis=0), rtol=4e-4, atol=4e-7)
                np.testing.assert_allclose(scalar[qr[n]], np.mean(ys, axis=0), rtol=4e-4, atol=4e-7)
                checked += 1
            second.extend(zip(tid[rank == 1], raw[rank == 1, :, 0]))
    for key, (rescued, broken, total) in counts.items():
        which, arm = key.split("/")
        target = (screen if which == "score" else comp)["combined"][arm]
        assert (rescued, broken) == (target["rescued"], target["broken"])
        np.testing.assert_allclose(total / 4800, target["mean"], atol=1e-12)
    ids = np.array([v[0] for v in second])
    views = np.array([v[1] for v in second])
    hits = np.sign(views)*.5+.5
    contrast = hits[:, 1]-hits[:, 0]
    rng = np.random.default_rng(111828)
    boot = np.hstack([rng.multinomial(600, np.full(600, 1/600), 5000) for _ in POOLS])
    avg = (boot @ np.bincount(ids, weights=contrast, minlength=1200)) / (boot @ np.bincount(ids, minlength=1200))
    np.testing.assert_allclose(np.quantile(avg, [.025, .975]), rival["true_second"]["nearest_minus_actual"]["ci95"])
    np.testing.assert_allclose(hits.mean(0), [rival["true_second"][k]["mean"] for k in ("actual", "nearest", "other_game")])
    seen = set()
    for p in POOLS:
        with np.load(CACHE / f"bilateral_{p}.npz") as z:
            assert len(set(z["players"].ravel())) == z["players"].size
            assert len(set(z["keys"].ravel())) == z["keys"].size
            seen.update(z["keys"].ravel())
    assert len(seen) == rep["overall"]["players"]
    functions = {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in (seed_split, v2.candidate_holdout)}
    paths = sorted((ROOT / "scripts").glob("pid_corroboration_*.py")) + [ROOT / "src/aicup_pid/corroboration.py"]
    atomic_json(OUT, {"recomputed_ranks": 4800*len(counts), "reconstructed_scalar_and_vector_queries": checked,
                     "same_query_cases": len(second), "paired_games": len(seen), "ordered_reference_sha": hashes,
                     "split_function_sha": functions, "code_sha": {p.name: digest_file(p) for p in paths}})
    print("verified", len(counts)*4800, "ranking decisions and", checked, "direct query reconstructions", flush=True)


if __name__ == "__main__":
    main()
