"""Pre-release score-equivalence gate for the Beta (legacy) lane.

ECR tags are immutable and the release workflow pushes a matching git tag, so
a bad image burns a `v1-K` and another release cycle. The Beta cutover moves
three numeric axes at once — architecture (ECS worker x86_64, inference
Lambda arm64), interpreter (prod Python 3.13.15, Lambda base Python 3.12) and
thread count (Lambda sets `torch.set_num_threads(6)`, the ECS worker takes
torch's default) — so before a tag is burnt this proves the Lambda path
reproduces the in-process scores it replaces.

Three subcommands so the two sides can be produced in different containers
and compared afterwards:

  baseline   score images the way mermaid-api's `_classify_image` does
             in-process, via spacer.tasks.extract_features /
             classify_features. Imports only pyspacer, numpy and the
             standard library, so it runs inside the pulled production API
             image, which has neither this repo nor pyspacer_function
             installed.
  candidate  score the same images through pyspacer_function.classify.classify(),
             the call the Lambda handler makes. Runs inside the built legacy
             image.
  compare    read a baseline and a candidate result file and apply the gate.

Runbook:

    # baseline, inside the pulled prod API image (x86_64):
    docker run --rm --platform linux/amd64 -v <images>:/imgs:ro -v <models/beta_model>:/m:ro \\
        -v <script>:/s.py:ro --entrypoint python <prod-asset-image> /s.py baseline \\
        --images-dir /imgs --model-dir /m --out /imgs/baseline.json

    # candidate, inside the built legacy image (arm64):
    docker run --rm -v <images>:/imgs:ro -v <models/beta_model>:/m:ro \\
        -v <script>:/s.py:ro --entrypoint python pyspacer-inference:beta-dev /s.py candidate \\
        --images-dir /imgs --model-dir /m --out /imgs/candidate.json

    # then, on the host (no pyspacer/torch required):
    python beta_equivalence.py compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

DEFAULT_NUM_POINTS = 25
DEFAULT_TOLERANCE = 1e-5
DEFAULT_THRESHOLD = 0.5
TOP_N = 3
# PIL (pyspacer's load_image) reads PNG natively and both producing modes glob
# identically, so including it costs nothing in comparison soundness while
# scoring images that .jpg-only would drop.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def generate_points(
    height: int, width: int, num_points: int, margin: tuple[int, int] = (0, 0)
) -> list[tuple[int, int]]:
    """Deterministic (row, col) grid, ported from mermaid-api's
    api.utils.classification.generate_points (same formula, parametrised by
    plain dimensions instead of a Django Image) so both producing modes score
    the identical points mermaid-api would for the same image size."""
    points_per_side = math.ceil(math.sqrt(num_points)) + 1
    shift_y = (height - 2 * margin[0]) / points_per_side
    shift_x = (width - 2 * margin[1]) / points_per_side
    points_per_side -= 1

    start_x = margin[1] + shift_x
    start_y = margin[0] + shift_y
    coords = []
    for y in range(points_per_side):
        cur_y = int(start_y + (shift_y * y))
        for x in range(points_per_side):
            coords.append((cur_y, int(start_x + (shift_x * x))))
    return coords


def _discover_images(images_dir: Path) -> list[Path]:
    """Every file under images_dir is accounted for: a recognized extension is
    scored, anything else is counted and named so a mismatched dataset is
    never dropped without a trace."""
    entries = sorted(p for p in images_dir.iterdir() if p.is_file())
    images = [p for p in entries if p.suffix.lower() in _IMAGE_SUFFIXES]
    skipped = [p for p in entries if p.suffix.lower() not in _IMAGE_SUFFIXES]
    if skipped:
        print(
            f"skipping {len(skipped)} file(s) with an unrecognized extension: "
            f"{', '.join(p.name for p in skipped)}",
            file=sys.stderr,
        )
    if not images:
        suffixes = "/".join(sorted(_IMAGE_SUFFIXES))
        raise FileNotFoundError(f"no usable images ({suffixes}) found under {images_dir}")
    return images


def _set_thread_count(threads: int | None) -> int:
    """Apply an explicit torch thread count when given, else leave torch's own
    default in place. Either way, returns the count actually in effect so it
    can be recorded in provenance."""
    import torch

    if threads is not None:
        torch.set_num_threads(threads)
    return torch.get_num_threads()


def _provenance(mode: str, num_points: int, num_threads: int, images: list[str]) -> dict:
    """A comparison between two runs whose libraries, architecture or thread
    count are unknown proves nothing, so every producing run stamps its own."""
    return {
        "mode": mode,
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": _pkg_version("torch"),
        "numpy_version": np.__version__,
        "sklearn_version": _pkg_version("scikit-learn"),
        "pyspacer_version": _pkg_version("pyspacer"),
        "num_threads": num_threads,
        "num_points": num_points,
        "images": images,
    }


def _run_baseline(images_dir: Path, model_dir: Path, num_points: int, threads: int | None) -> dict:
    """Reproduce mermaid-api's in-process scoring: one ExtractFeaturesMsg /
    ClassifyFeaturesMsg pair per image through spacer.tasks, predict_proba
    applied one point at a time — the exact baseline pyspacer_function's
    legacy predict path must match."""
    import tempfile

    from spacer.extractors import EfficientNetExtractor
    from spacer.messages import ClassifyFeaturesMsg, DataLocation, ExtractFeaturesMsg
    from spacer.storage import load_image
    from spacer.tasks import classify_features, extract_features

    num_threads = _set_thread_count(threads)
    weights_loc = DataLocation("filesystem", str(model_dir / "efficientnet_weights.pt"))
    classifier_loc = DataLocation("filesystem", str(model_dir / "classifier.pkl"))
    images = _discover_images(images_dir)

    points_out = []
    with tempfile.TemporaryDirectory() as tmp:
        for image_path in images:
            image_loc = DataLocation("filesystem", str(image_path))
            width, height = load_image(image_loc).size
            points = generate_points(height, width, num_points)
            feature_loc = DataLocation(
                "filesystem", str(Path(tmp) / f"{image_path.stem}.featurevector")
            )

            extract_msg = ExtractFeaturesMsg(
                job_token=image_path.name,
                extractor=EfficientNetExtractor(data_locations=dict(weights=weights_loc)),
                rowcols=points,
                image_loc=image_loc,
                feature_loc=feature_loc,
            )
            extract_features(extract_msg)

            classify_msg = ClassifyFeaturesMsg(
                job_token=extract_msg.job_token,
                feature_loc=extract_msg.feature_loc,
                classifier_loc=classifier_loc,
            )
            result = classify_features(classify_msg)
            labels = [str(c) for c in result.classes]
            for row, col, scores in result.scores:
                points_out.append(
                    {
                        "image": image_path.name,
                        "row": row,
                        "col": col,
                        "scores": dict(zip(labels, (float(s) for s in scores))),
                    }
                )

    return {
        "provenance": _provenance("baseline", num_points, num_threads, [p.name for p in images]),
        "points": points_out,
    }


def _run_candidate(images_dir: Path, model_dir: Path, num_points: int, threads: int | None) -> dict:
    """Score every image through pyspacer_function.classify.classify() over
    the legacy artifact pair — the same call the Lambda handler makes."""
    from spacer.messages import DataLocation
    from spacer.storage import load_image

    from pyspacer_function.classify import classify
    from pyspacer_function.resolver import LegacyModelFiles

    num_threads = _set_thread_count(threads)
    files = LegacyModelFiles.in_dir(model_dir)
    images = _discover_images(images_dir)

    points_out = []
    for image_path in images:
        image_loc = DataLocation("filesystem", str(image_path))
        width, height = load_image(image_loc).size
        points = generate_points(height, width, num_points)

        results, _valid = classify(image_loc, files, points)
        for point in results:
            points_out.append(
                {
                    "image": image_path.name,
                    "row": point.row,
                    "col": point.col,
                    "scores": {s.label: s.score for s in point.scores},
                }
            )

    return {
        "provenance": _provenance("candidate", num_points, num_threads, [p.name for p in images]),
        "points": points_out,
    }


def _top_labels(scores: dict[str, float], n: int) -> list[str]:
    """The n highest-scoring labels, ties broken by label name so the ranking
    is deterministic when two scores are exactly equal."""
    return [label for label, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


@dataclass
class Criterion:
    name: str
    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    criteria: list[Criterion]
    worst_delta: float
    worst_delta_point: str | None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)


def compare(
    baseline: dict,
    candidate: dict,
    tolerance: float = DEFAULT_TOLERANCE,
    threshold: float = DEFAULT_THRESHOLD,
) -> ComparisonReport:
    """Apply the four-criterion score-equivalence gate: identical top-1 label,
    identical top-3 label set, max |delta| under tolerance, and zero points
    where a score crosses `threshold` in one run and not the other — the last
    one matters because mermaid-api's CLASSIFIED_THRESHOLD decides whether an
    Annotation row is written at all, so a near-tie flip changes stored data
    even when every delta looks tiny. Raises ValueError when the two runs did
    not score the same points — including when neither scored any — which is
    a precondition failure distinct from a gate failure: every criterion here
    is a "no violations found" check, trivially true of an empty comparison,
    so a vacuous run must never be reported as a pass."""
    base_by_key = {(p["image"], p["row"], p["col"]): p["scores"] for p in baseline["points"]}
    cand_by_key = {(p["image"], p["row"], p["col"]): p["scores"] for p in candidate["points"]}
    if set(base_by_key) != set(cand_by_key):
        mismatched = sorted(set(base_by_key) ^ set(cand_by_key))
        raise ValueError(
            f"baseline and candidate score different points — {len(mismatched)} "
            f"mismatched (image, row, col) keys, e.g. {mismatched[:5]}"
        )
    if not base_by_key:
        raise ValueError("baseline and candidate both score zero points — nothing to compare")

    top1_violations: list[str] = []
    top3_violations: list[str] = []
    delta_violations: list[str] = []
    crossing_violations: list[str] = []
    worst_delta = 0.0
    worst_delta_point: str | None = None

    for key in sorted(base_by_key):
        image, row, col = key
        b_scores, c_scores = base_by_key[key], cand_by_key[key]
        if set(b_scores) != set(c_scores):
            raise ValueError(f"{image} ({row},{col}): baseline and candidate disagree on labels")

        b_top1, c_top1 = _top_labels(b_scores, 1)[0], _top_labels(c_scores, 1)[0]
        if b_top1 != c_top1:
            top1_violations.append(
                f"{image} ({row},{col}): baseline={b_top1!r} candidate={c_top1!r}"
            )

        b_top3, c_top3 = set(_top_labels(b_scores, TOP_N)), set(_top_labels(c_scores, TOP_N))
        if b_top3 != c_top3:
            top3_violations.append(
                f"{image} ({row},{col}): baseline={sorted(b_top3)} candidate={sorted(c_top3)}"
            )

        for label, b_score in b_scores.items():
            c_score = c_scores[label]
            delta = abs(b_score - c_score)
            if delta > worst_delta:
                worst_delta, worst_delta_point = delta, f"{image} ({row},{col}) label={label!r}"
            if delta >= tolerance:
                delta_violations.append(
                    f"{image} ({row},{col}) label={label!r}: delta={delta:.3g} "
                    f"(baseline={b_score:.6f} candidate={c_score:.6f})"
                )
            if (b_score >= threshold) != (c_score >= threshold):
                crossing_violations.append(
                    f"{image} ({row},{col}) label={label!r}: baseline={b_score:.6f} "
                    f"candidate={c_score:.6f} threshold={threshold}"
                )

    criteria = [
        Criterion("top1_label_identical", not top1_violations, top1_violations),
        Criterion("top3_label_set_identical", not top3_violations, top3_violations),
        Criterion("max_delta_under_tolerance", not delta_violations, delta_violations),
        Criterion("no_threshold_crossings", not crossing_violations, crossing_violations),
    ]
    return ComparisonReport(criteria, worst_delta, worst_delta_point)


def _print_report(report: ComparisonReport) -> None:
    for criterion in report.criteria:
        status = "PASS" if criterion.passed else "FAIL"
        print(f"[{status}] {criterion.name} ({len(criterion.violations)} violation(s))")
        for violation in criterion.violations:
            print(f"    {violation}")
    where = f" at {report.worst_delta_point}" if report.worst_delta_point else ""
    print(f"worst |delta|: {report.worst_delta:.3g}{where}")
    print("RESULT: " + ("PASS" if report.passed else "FAIL"))


def cmd_baseline(args: argparse.Namespace) -> int:
    result = _run_baseline(
        Path(args.images_dir), Path(args.model_dir), args.num_points, args.threads
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    n_images = len(result["provenance"]["images"])
    print(f"wrote {len(result['points'])} point scores across {n_images} images to {args.out}")
    return 0


def cmd_candidate(args: argparse.Namespace) -> int:
    result = _run_candidate(
        Path(args.images_dir), Path(args.model_dir), args.num_points, args.threads
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    n_images = len(result["provenance"]["images"])
    print(f"wrote {len(result['points'])} point scores across {n_images} images to {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    try:
        report = compare(baseline, candidate, tolerance=args.tolerance, threshold=args.threshold)
    except ValueError as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 0 if report.passed else 1


def _add_producer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--images-dir", required=True, help="directory of local .jpg/.jpeg/.png images"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="directory with efficientnet_weights.pt and classifier.pkl",
    )
    parser.add_argument("--out", required=True, help="path to write the JSON result file")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads value; omit to leave torch's own default",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_p = subparsers.add_parser(
        "baseline", help="score images via spacer.tasks, mirroring mermaid-api in-process"
    )
    _add_producer_args(baseline_p)
    baseline_p.set_defaults(func=cmd_baseline)

    candidate_p = subparsers.add_parser(
        "candidate", help="score images via pyspacer_function.classify.classify()"
    )
    _add_producer_args(candidate_p)
    candidate_p.set_defaults(func=cmd_candidate)

    compare_p = subparsers.add_parser(
        "compare", help="apply the score-equivalence gate to two result files"
    )
    compare_p.add_argument("baseline", help="baseline result JSON path")
    compare_p.add_argument("candidate", help="candidate result JSON path")
    compare_p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    compare_p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    compare_p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
