"""Exercises the `compare` subcommand's gate end-to-end, over the real CLI
entrypoint against hand-written fixture files — not the contract package's
tests, which a review is running against separately. Run directly:
    uv run pytest packages/pyspacer-function/tests/test_beta_equivalence.py
"""

import json
import subprocess
import sys
from pathlib import Path

# scripts/beta_equivalence.py is not an installed module — a plain import
# would need it on sys.path — so it is located by an explicit path relative
# to this test file and run as a subprocess instead.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "beta_equivalence.py"


def _point(image, row, col, scores):
    return {"image": image, "row": row, "col": col, "scores": scores}


def _result(*points):
    return {"provenance": {"mode": "test", "images": ["img.jpg"]}, "points": list(points)}


def _run_compare(tmp_path, baseline, candidate):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "compare", str(baseline_path), str(candidate_path)],
        capture_output=True,
        text=True,
    )


def test_identical_scores_pass_every_criterion(tmp_path):
    scores = {"a::": 0.7, "b::": 0.2, "c::": 0.1}
    baseline = _result(_point("img.jpg", 10, 20, dict(scores)))
    candidate = _result(_point("img.jpg", 10, 20, dict(scores)))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 0
    assert "RESULT: PASS" in proc.stdout


def test_flipped_top1_fails_and_is_reported(tmp_path):
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.2, "c::": 0.1}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.2, "b::": 0.7, "c::": 0.1}))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 1
    assert "[FAIL] top1_label_identical" in proc.stdout
    assert "[FAIL] top3_label_set_identical" not in proc.stdout


def test_changed_top3_set_fails_and_is_reported(tmp_path):
    # c and d are a near-tie that swaps places (delta 6e-6, under tolerance),
    # so 3rd place changes without either individual score moving much.
    baseline = _result(
        _point("img.jpg", 10, 20, {"a::": 0.5, "b::": 0.3, "c::": 0.100003, "d::": 0.099997})
    )
    candidate = _result(
        _point("img.jpg", 10, 20, {"a::": 0.5, "b::": 0.3, "c::": 0.099997, "d::": 0.100003})
    )

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 1
    assert "[PASS] top1_label_identical" in proc.stdout
    assert "[PASS] max_delta_under_tolerance" in proc.stdout
    assert "[FAIL] top3_label_set_identical" in proc.stdout


def test_delta_over_tolerance_fails_without_flipping_rank_or_threshold(tmp_path):
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.29999, "c::": 0.00001}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.29899, "c::": 0.00101}))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 1
    assert "[FAIL] max_delta_under_tolerance" in proc.stdout
    assert "[FAIL] top1_label_identical" not in proc.stdout
    assert "[FAIL] no_threshold_crossings" not in proc.stdout


def test_threshold_crossing_within_tolerance_still_fails(tmp_path):
    # "a::" stays dominant and "c::" stays lowest in both runs, so rank and
    # top-3 membership are untouched; only "b::" moves, by 3e-6 — comfortably
    # under the 1e-5 tolerance — yet that sliver crosses 0.5. A tolerance-only
    # check would wrongly pass this.
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.9, "b::": 0.499998, "c::": 0.1}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.9, "b::": 0.500001, "c::": 0.1}))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 1
    assert "[PASS] top1_label_identical" in proc.stdout
    assert "[PASS] top3_label_set_identical" in proc.stdout
    assert "[PASS] max_delta_under_tolerance" in proc.stdout
    assert "[FAIL] no_threshold_crossings" in proc.stdout


def test_mismatched_points_reported_as_uncomparable_not_a_gate_failure(tmp_path):
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.3}))
    candidate = _result(_point("img.jpg", 99, 99, {"a::": 0.7, "b::": 0.3}))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 2
    assert "cannot compare" in proc.stderr
