"""Exercises the `compare` subcommand's gate against hand-written fixture
files — not the contract package's tests, which a review is running against
separately. Criterion logic runs in-process against `compare()`'s return
value; the CLI entrypoint is exercised separately for exit codes, the
PASS/FAIL result line, and the per-criterion report rendering. Run directly:
    uv run pytest packages/pyspacer-function/tests/test_beta_equivalence.py
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

# scripts/beta_equivalence.py is not an installed module — a plain import
# would need it on sys.path — so it is located by an explicit path relative
# to this test file, run as a subprocess for the CLI-level tests below and
# loaded in-process (see `compare` below) for the criterion-logic tests.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "beta_equivalence.py"

_spec = importlib.util.spec_from_file_location("beta_equivalence", _SCRIPT)
beta_equivalence = importlib.util.module_from_spec(_spec)
# dataclass() resolves postponed annotations via sys.modules[__module__], so
# the module must be registered there before exec_module runs its class bodies.
sys.modules[_spec.name] = beta_equivalence
_spec.loader.exec_module(beta_equivalence)
compare = beta_equivalence.compare


def _point(image, row, col, scores):
    return {"image": image, "row": row, "col": col, "scores": scores}


def _result(*points):
    return {"provenance": {"mode": "test", "images": ["img.jpg"]}, "points": list(points)}


def _failed(report) -> set[str]:
    """The set of criterion names that did not pass, for hand-checked
    literals that pin down exactly which criteria fired."""
    return {c.name for c in report.criteria if not c.passed}


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

    report = compare(baseline, candidate)
    assert report.passed is True
    assert _failed(report) == set()

    # The CLI's exit code and PASS message are a genuinely CLI-level concern
    # (argument parsing, cmd_compare's return value), so prove those too.
    proc = _run_compare(tmp_path, baseline, candidate)
    assert proc.returncode == 0
    assert "RESULT: PASS" in proc.stdout


def test_flipped_top1_fails_and_is_reported(tmp_path):
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.2, "c::": 0.1}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.2, "b::": 0.7, "c::": 0.1}))

    report = compare(baseline, candidate)
    # top3 membership is untouched (both labels stay among the only 3), but
    # swapping 0.7 and 0.2 also blows the tolerance and crosses the threshold.
    assert _failed(report) == {
        "top1_label_identical",
        "max_delta_under_tolerance",
        "no_threshold_crossings",
    }

    proc = _run_compare(tmp_path, baseline, candidate)
    assert proc.returncode == 1
    # The report line names the criterion that failed and one that passed,
    # so an operator can see which check burned the tag without re-deriving
    # it from the raw scores.
    assert "[FAIL] top1_label_identical" in proc.stdout
    assert "[PASS] top3_label_set_identical" in proc.stdout


def test_changed_top3_set_fails_and_is_reported():
    # c and d are a near-tie that swaps places (delta 6e-6, under tolerance),
    # so 3rd place changes without either individual score moving much.
    baseline = _result(
        _point("img.jpg", 10, 20, {"a::": 0.5, "b::": 0.3, "c::": 0.100003, "d::": 0.099997})
    )
    candidate = _result(
        _point("img.jpg", 10, 20, {"a::": 0.5, "b::": 0.3, "c::": 0.099997, "d::": 0.100003})
    )

    report = compare(baseline, candidate)

    assert _failed(report) == {"top3_label_set_identical"}


def test_delta_over_tolerance_fails_without_flipping_rank_or_threshold():
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.29999, "c::": 0.00001}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.29899, "c::": 0.00101}))

    report = compare(baseline, candidate)

    assert _failed(report) == {"max_delta_under_tolerance"}


def test_threshold_crossing_within_tolerance_still_fails():
    # "a::" stays dominant and "c::" stays lowest in both runs, so rank and
    # top-3 membership are untouched; only "b::" moves, by 3e-6 — comfortably
    # under the 1e-5 tolerance — yet that sliver crosses 0.5. A tolerance-only
    # check would wrongly pass this.
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.9, "b::": 0.499998, "c::": 0.1}))
    candidate = _result(_point("img.jpg", 10, 20, {"a::": 0.9, "b::": 0.500001, "c::": 0.1}))

    report = compare(baseline, candidate)

    assert _failed(report) == {"no_threshold_crossings"}


def test_mismatched_points_reported_as_uncomparable_not_a_gate_failure(tmp_path):
    baseline = _result(_point("img.jpg", 10, 20, {"a::": 0.7, "b::": 0.3}))
    candidate = _result(_point("img.jpg", 99, 99, {"a::": 0.7, "b::": 0.3}))

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 2
    assert "cannot compare" in proc.stderr


def test_zero_points_reported_as_uncomparable_not_a_gate_pass(tmp_path):
    # Both files are well-formed with an empty points list — every criterion
    # is vacuously true of an empty set, so this must not read as a PASS.
    baseline = _result()
    candidate = _result()

    proc = _run_compare(tmp_path, baseline, candidate)

    assert proc.returncode == 2
    assert "cannot compare" in proc.stderr
    assert "RESULT: PASS" not in proc.stdout


def test_producing_mode_on_directory_with_no_usable_images_fails_and_names_skips(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "notes.txt").write_text("not an image")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_path = tmp_path / "baseline.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "baseline",
            "--images-dir",
            str(images_dir),
            "--model-dir",
            str(model_dir),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert not out_path.exists()
    assert "notes.txt" in proc.stdout + proc.stderr
