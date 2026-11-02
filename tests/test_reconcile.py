import json
import math

import pandas as pd

import pid_reconcile


def _files(tmp_path, top1_sub=("pub_p_a", "pub_p_b", "pub_p_c"), none_second=True):
    checks = {"pools": [{"prefix": "pub_", "main_line": {"top1": [["pub_q_0001", "pub_p_a"], ["pub_q_0002", None if none_second else "pub_p_b"], ["pub_q_0003", "pub_p_c"]]}}]}
    cp = tmp_path / "checks.json"
    cp.write_text(json.dumps(checks))
    rows = []
    for i, t in enumerate(top1_sub):
        others = [p for p in ("pub_p_a", "pub_p_b", "pub_p_c", "pub_p_d", "pub_p_e", "pub_p_f") if p != t][:4]
        rows.append({"question_id": f"pub_q_{i + 1:04d}", "top_1": t, **{f"top_{j + 2}": o for j, o in enumerate(others)}})
    sp = tmp_path / "submission_player.csv"
    pd.DataFrame(rows).to_csv(sp, index=False)
    tp = tmp_path / "truth_public.csv"
    pd.DataFrame({"question_id": ["pub_q_0001", "pub_q_0002", "pub_q_0003"], "player_id": ["pub_p_a", "pub_p_b", "pub_p_d"],
                  "train_player_id": ["a", "b", "d"], "rank": ["D", "C", "B"], "n_games": [5, 10, 5]}).to_csv(tp, index=False)
    return cp, sp, tp


def test_reconcile_agrees_and_scores(tmp_path, capsys):
    cp, sp, tp = _files(tmp_path)
    out = tmp_path / "rec.json"
    assert pid_reconcile.main(["--checks", str(cp), "--submission", str(sp), "--truth", str(tp), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["n_questions_checks"] == 3 and r["n_questions_submission"] == 3 and r["n_compared"] == 2
    assert r["n_top1_equal"] == 2 and r["n_top1_differ"] == 0 and r["checks_without_top1"] == ["pub_q_0002"] and r["missing_in_submission"] == []
    s = r["truth_score"]
    assert s["n"] == 3 and abs(s["score"] - (1.0 + 1.0 + math.exp(-3)) / 3) < 1e-9 and s["top1"] == 2 / 3 and set(s["by_n"]) == {"5", "10"}  # q3: truth d is top_4
    assert "pid_git_head" in r and "top_1 agree 2/2" in capsys.readouterr().out


def test_reconcile_flags_disagreement(tmp_path):
    cp, sp, tp = _files(tmp_path, top1_sub=("pub_p_a", "pub_p_b", "pub_p_e"))
    out = tmp_path / "rec.json"
    assert pid_reconcile.main(["--checks", str(cp), "--submission", str(sp), "--out", str(out)]) == 1
    r = json.loads(out.read_text())
    assert r["n_top1_differ"] == 1 and r["differences"] == [{"question_id": "pub_q_0003", "checks": "pub_p_c", "submission": "pub_p_e"}] and r["truth_score"] is None


def test_reconcile_input_errors(tmp_path):
    assert pid_reconcile.main(["--checks", str(tmp_path / "no.json"), "--submission", str(tmp_path / "no.csv"), "--out", str(tmp_path / "r.json")]) == 2
