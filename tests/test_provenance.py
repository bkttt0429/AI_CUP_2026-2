from aicup_pid import provenance


def test_pid_environment_records_both_repositories():
    env = provenance.pid_environment("cpu")
    assert {"git_head", "torch_version", "device_name", "pid_git_head", "pid_git_dirty"} <= set(env)
    assert env["pid_git_head"] is None or (isinstance(env["pid_git_head"], str) and len(env["pid_git_head"]) >= 7)
    assert env["pid_git_dirty"] in (None, True, False)
    assert provenance._git("rev-parse", "--is-inside-work-tree") in ("true", None)
