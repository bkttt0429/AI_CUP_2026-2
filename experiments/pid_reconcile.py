#!/usr/bin/env python
"""Reconcile pid_check_files.py's main-line Top-1 with submission_player.csv before an upload (fork plan section 6), and
score the submission against truth files when there are any (the mock rehearsal).

The check script and predict_pid.py fit the same models with the same weights, so their Top-1 must agree on every
question that had a usable game; a disagreement means the two runs did not see the same inputs. Exit 1 when any Top-1
differs or a question is missing on either side, 2 on input errors, else 0.

  python -I scripts/pid_reconcile.py --checks results/checks/official_checks.json --submission results/submission/submission_player.csv --out results/checks/official_reconcile.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

MAX_EXAMPLES = 20


def reconcile(checks, submission, truth=None):
    top1_checks = {q: c for pool in checks["pools"] for q, c in pool["main_line"]["top1"]}
    sub = {r.question_id: [getattr(r, f"top_{i}") for i in range(1, pools.TOP_K + 1)] for r in submission.itertuples()}
    compared = [q for q, c in top1_checks.items() if c is not None and q in sub]
    diffs = [{"question_id": q, "checks": top1_checks[q], "submission": sub[q][0]} for q in compared if sub[q][0] != top1_checks[q]]
    out = {"n_questions_checks": len(top1_checks), "n_questions_submission": len(sub), "n_compared": len(compared),
           "n_top1_equal": len(compared) - len(diffs), "n_top1_differ": len(diffs), "differences": diffs[:MAX_EXAMPLES],
           "checks_without_top1": [q for q, c in top1_checks.items() if c is None],
           "missing_in_submission": [q for q in top1_checks if q not in sub], "missing_in_checks": [q for q in sub if q not in top1_checks],
           "truth_score": None}
    if truth is not None:
        t = truth[truth["question_id"].isin(sub)]
        tops = [sub[q] for q in t["question_id"]]
        out["truth_score"] = {**pools.summarise(tops, t["player_id"].tolist(), t["n_games"].to_numpy(dtype=int)), "n_truth_rows": int(len(truth))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks", required=True, help="pid_check_files.py --out JSON")
    ap.add_argument("--submission", required=True, help="submission_player.csv")
    ap.add_argument("--truth", nargs="*", default=[], help="truth CSVs of pid_mock_official.py (mock rehearsal only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        checks = json.loads(Path(a.checks).read_text())
        submission = pd.read_csv(a.submission, dtype=str, keep_default_na=False)
        truth = pd.concat([pd.read_csv(p, dtype={"question_id": str, "player_id": str}) for p in a.truth], ignore_index=True) if a.truth else None
        res = reconcile(checks, submission, truth)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    res.update({"checks": str(a.checks), "submission": str(a.submission), "truth": list(a.truth), **pid_environment("cpu")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(res, indent=1) + "\n")
    print(f"top_1 agree {res['n_top1_equal']}/{res['n_compared']} (differ {res['n_top1_differ']}; checks without a Top-1 {len(res['checks_without_top1'])}; "
          f"missing in submission {len(res['missing_in_submission'])}, in checks {len(res['missing_in_checks'])})")
    if res["truth_score"]:
        s = res["truth_score"]
        print(f"against truth: score {s['score']:.4f} top1 {s['top1']:.3f} top5 {s['top5']:.3f} ({s['n']} questions); by n " +
              " ".join(f"{n}:{b['score']:.3f}" for n, b in s["by_n"].items()))
    print(f"wrote {a.out}")
    bad = res["n_top1_differ"] or res["missing_in_submission"] or res["missing_in_checks"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
