"""METHOD_NOTES 2.171: do the EARLY readings of a training run predict the family's final single-network score?

Every training log prints, each 5,000 steps, the top-1 of held-out games of the training players ("seen").  The decision
reading of a family at the single-network level is the official-candidate hold-out score (1-game questions, mean over the
family's networks; results/final/v2_offcand2_*.json, "single_by_group").  For every family that has both: the seen top-1
at 5,000 / 10,000 steps and at the end against that score; then the pairs of families that differ in ONE thing.
Nothing is selected here; it only tells how much an early stop can be trusted.
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

R2 = Path("/home/u2507115/BK_AICUP_2026-2")
LOGS = Path("/home/u2507115/BK_AICUP_2026-1/logs/nano4")
PAT = re.compile(r"step (\d+) \[ema\]: held-out own games among \d+ players: seen \(\d+\) top-1 / top-5 / top-20 ([0-9.]+)")
STEP = re.compile(r"step +(\d+)/(\d+): loss ([0-9.]+)")
MAP = {"o22F": ["o22S", "o22SM"], "o24F": ["o24S", "o24SM"], "o26F": ["o26S", "o26SM"], "o27F": ["o27S", "o27SM"], "o31H": ["o31S"], "o22H": ["o22S"]}


def tags_of(group):
    if group in MAP:
        return MAP[group]
    base = group[:-1] if group.endswith("F") else group
    return [base + "A", base + "C"] if group.endswith("F") else [base]


def read_logs(tag):
    """{network: {step: seen top-1}}, {network: {step: loss}} from the newest log of every network of a tag."""
    seen, loss = {}, {}
    files = defaultdict(list)
    for f in glob.glob(str(LOGS / f"pidseq_-1_{tag}s*-*.log")):
        m = re.match(rf"pidseq_-1_({re.escape(tag)}s\d+(?:k\d+)?)-(\d+)\.log$", Path(f).name)
        if m:
            files[m.group(1)].append((int(m.group(2)), f))
    for net, fs in files.items():
        text = Path(max(fs)[1]).read_text(errors="replace")
        s = {int(a): float(b) for a, b in PAT.findall(text)}
        if s:
            seen[net] = s
            loss[net] = {int(a): float(c) for a, _, c in STEP.findall(text)}
    return seen, loss


def main():
    single = {}
    for f in sorted(glob.glob(str(R2 / "results/final/v2_offcand2_*.json")), key=lambda p: Path(p).stat().st_mtime):
        d = json.load(open(f))
        try:
            a, b = d["pools"]["pub2"]["1"]["single_by_group"], d["pools"]["priv2"]["1"]["single_by_group"]
        except (KeyError, TypeError):
            continue
        for g in a:
            if g in b:
                single[g] = (0.5 * (a[g] + b[g]), Path(f).name)  # the newest file that holds the group wins
    rows = []
    for g, (sc, src) in sorted(single.items()):
        if not g.startswith("o"):
            continue
        seen = {}
        for t in tags_of(g):
            seen.update(read_logs(t)[0])
        if not seen:
            continue
        last = [max(s) for s in seen.values()]
        at = lambda k: float(np.mean([s[k] for s in seen.values() if k in s])) if any(k in s for s in seen.values()) else float("nan")  # noqa: E731
        rows.append((g, len(seen), int(np.median(last)), at(5000), at(10000), float(np.mean([s[max(s)] for s in seen.values()])), sc, src))
    print("family | networks with a log | steps | seen top-1 at 5k | at 10k | at the end | official-candidate single-network score (1 game)")
    for r in rows:
        print(f"  {r[0]:6s} {r[1]:3d} {r[2]:6d}   {r[3]:.3f}  {r[4]:.3f}  {r[5]:.3f}   {r[6]:.4f}   ({r[7]})")
    A = np.array([[r[3], r[4], r[5], r[6]] for r in rows], float)
    for j, name in enumerate(("5k", "10k", "end")):
        ok = np.isfinite(A[:, j])
        rho, p = stats.spearmanr(A[ok, j], A[ok, 3])
        r_, _ = stats.pearsonr(A[ok, j], A[ok, 3])
        print(f"across {int(ok.sum())} families: seen top-1 at {name} vs the official-candidate score: Spearman {rho:+.2f} (p {p:.3f}), Pearson {r_:+.2f}")
    print("\npairs of families that differ in one thing (second minus first): seen top-1 at 5k / 10k / end, and the official-candidate score")
    by = {r[0]: r for r in rows}
    for a, b, what in (("o22F", "o24F", "512x10 instead of 384x8 (plain inputs)"), ("o19F", "o21F", "512x10 instead of 384x8 (second feature set)"),
                       ("o22F", "o23F", "second feature set instead of plain inputs"), ("o23F", "o25F", "hidden openings / windows / dropout on top"),
                       ("o22F", "o26F", "one competition fold fewer (the seen set differs: not comparable early)"), ("o16F", "o17F", "o17 vs o16"),
                       ("o17F", "o18F", "o18 vs o17"), ("o18F", "o19F", "by-games sampling (o19) vs o18"), ("o13F", "o16F", "second feature set (o16) vs first (o13)"),
                       ("o3S", "o4S", "182k external accounts instead of 116k"), ("o3S", "o7F", "longer training + passed recipe")):
        if a in by and b in by:
            x, y = by[a], by[b]
            print(f"  {b} - {a} ({what}): {y[3] - x[3]:+.3f} / {y[4] - x[4]:+.3f} / {y[5] - x[5]:+.3f}   official {y[6] - x[6]:+.4f}")
    # the running pair: o31 (window) against o22, same seeds
    s31, l31 = read_logs("o31S")
    s22, l22 = read_logs("o22S")
    print("\nrunning: o31S (window input) minus o22S, same seeds; seen top-1 and training loss at each evaluation step so far")
    for k in sorted({k for s in s31.values() for k in s}):
        d = [s31[n][k] - s22[n.replace('o31', 'o22')][k] for n in s31 if k in s31[n] and n.replace("o31", "o22") in s22 and k in s22[n.replace("o31", "o22")]]
        dl = [l31[n][k] - l22[n.replace('o31', 'o22')][k] for n in l31 if k in l31[n] and n.replace("o31", "o22") in l22 and k in l22[n.replace("o31", "o22")]]
        if d:
            print(f"  step {k}: seen top-1 {np.mean(d):+.4f} (per seed {' '.join(f'{v:+.3f}' for v in d)}); loss {np.mean(dl):+.4f}" if dl else f"  step {k}: seen top-1 {np.mean(d):+.4f}")
    # seed-to-seed spread of the baseline at the same step (what a paired difference must beat)
    for k in (5000, 10000, 20000, 40000):
        v = [s[k] for s in s22.values() if k in s]
        if len(v) > 1:
            print(f"  o22S seed spread at step {k}: sd {np.std(v, ddof=1):.4f} over {len(v)} networks")


if __name__ == "__main__":
    main()
