import importlib


def test_signature_distinguishes_subject_setup_and_omitted_tail():
    m = importlib.import_module("pid_observation_audit")
    a = "(;SZ[19]HD[0];B[dd];W[pp];B[dp])"
    b = "(;SZ[19]HD[0];B[dd];W[pp];B[pd])"
    assert m.signature(a, "B", limit=2) == m.signature(b, "B", limit=2)
    assert m.signature(a, "B") != m.signature(b, "B")
    assert m.signature(a, "B") != m.signature(a, "W")
    assert m.signature(a, "B") != m.signature(a.replace("HD[0]", "HD[2]"), "B")


def test_symmetry_equivalence_is_separate_from_raw_equality():
    m = importlib.import_module("pid_observation_audit")
    a = "(;SZ[19];B[dd];W[pp];B[dp];W[])"
    b = "(;SZ[19];B[pd];W[dp];B[pp];W[])"
    assert m.signature(a, "B") != m.signature(b, "B")
    assert m.signature(a, "B", symmetric=True) == m.signature(b, "B", symmetric=True)
