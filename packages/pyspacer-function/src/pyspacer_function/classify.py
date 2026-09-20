"""Storage-agnostic classify core: extract features with pyspacer's EfficientNet
extractor, then predict. Both serving formats share the extraction half and
branch only on the predict half — "graph" runs the portable TorchScript head via
load_predictor, "legacy" runs the pickled scikit-learn classifier through
pyspacer's own loader. Takes pyspacer DataLocations, so it runs over
s3/filesystem/memory storage identically — which is what makes it testable
without AWS.

A head is fitted to feature vectors, so on the graph lane the extractor is
checked against the one model.json records before its output reaches the head.
Extracting with a different geometry produces different labels at comparable
confidence and raises nothing, so this is the only place that failure is
visible. An artifact cut before that block existed is served with a warning
rather than refused, so adding the check does not retire the versions already
released."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from spacer.data_classes import ImageFeatures
from spacer.extractors import EfficientNetExtractor
from spacer.messages import DataLocation
from spacer.storage import load_classifier, load_image
from spacer.task_utils import check_extract_inputs

from mermaid_inference_contract import PointResult, PointScore

from pyspacer_function.resolver import LegacyModelFiles, ModelFiles

if TYPE_CHECKING:
    from mermaid_classifier.pyspacer.inference import ExtractorSpec

logger = logging.getLogger(__name__)


class ClassifyOutcome(NamedTuple):
    point_results: list[PointResult]
    valid_rowcol: bool
    feature_stored: bool  # False when no location was requested, or the write failed


def classify(
    image_loc: DataLocation,
    files: ModelFiles | LegacyModelFiles,
    points: Sequence[Sequence[int]],
    *,
    extractor: EfficientNetExtractor | None = None,
    feature_output_loc: DataLocation | None = None,
) -> ClassifyOutcome:
    """Classify each point. When feature_output_loc is given, the extracted
    features are persisted there once predict succeeds; nothing downstream
    reads that artifact back, so a failed write is logged and reported through
    feature_stored rather than raised."""
    points = [(int(r), int(c)) for r, c in points]
    spec = _recorded_extractor(files)

    image = load_image(image_loc)
    check_extract_inputs(image, points, image_loc.key)  # pyright: ignore[reportArgumentType]  # pyspacer stubs use PIL.Image module, not PIL.Image.Image

    if extractor is None:
        extractor = EfficientNetExtractor(
            data_locations=dict(weights=DataLocation("filesystem", str(files.efficientnet_pt)))
        )
    if spec is not None:
        spec.check_extractor(extractor)

    features, _ = extractor(image, points)  # pyright: ignore[reportArgumentType]  # same PIL stub issue
    if spec is not None:
        spec.check_feature_dim(features.feature_dim)

    if isinstance(files, LegacyModelFiles):
        labels, proba = _predict_legacy(files, features, points)
    else:
        labels, proba = _predict_graph(files, features, points)

    # Persisted only once predict succeeds, so a failed classify leaves no
    # orphan feature-vector object — store() writes the same np.savez_compressed
    # archive (meta/rows/cols/feat) that ImageFeatures.load() reads back.
    feature_stored = False
    if feature_output_loc is not None:
        try:
            features.store(feature_output_loc)
            feature_stored = True
        except Exception as exc:
            # Stable marker for the CloudWatch Logs metric filter + alarm
            # mermaid-{env}-inference-feature-store-errors: the write failure is
            # swallowed and reported as success, so it never increments Lambda Errors.
            # Keep token in sync with mermaid-api InferenceStack.
            logger.exception(
                "[classify.feature_store_error] failed to store feature vector"
                " error=%s bucket=%r key=%r",
                type(exc).__name__,
                feature_output_loc.bucket_name,
                feature_output_loc.key,
            )

    results = []
    for (row, col), point_proba in zip(points, proba):
        scored = sorted(zip(labels, point_proba), key=itemgetter(1), reverse=True)
        results.append(
            PointResult(
                row=row,
                col=col,
                scores=[PointScore(label=label, score=float(s)) for label, s in scored],
            )
        )
    return ClassifyOutcome(
        point_results=results, valid_rowcol=features.valid_rowcol, feature_stored=feature_stored
    )


def _recorded_extractor(files: ModelFiles | LegacyModelFiles) -> ExtractorSpec | None:
    """The extractor model.json records, or None when nothing records one.

    None on the legacy lane, whose pickle carries no manifest, and on a graph
    artifact cut before the block existed — v2 is one, so refusing would take
    a released version out of service to add a check it predates. The gap is
    logged rather than guessed at. A block that is present but malformed still
    raises, as does one that disagrees with the live extractor.

    Imported inside the branch, as with load_predictor below: the legacy image
    installs no mermaid-classifier.
    """
    if not isinstance(files, ModelFiles):
        return None

    from mermaid_classifier.pyspacer.inference import MANIFEST_KEY, ExtractorSpec

    manifest = json.loads(files.model_json.read_text())
    if MANIFEST_KEY not in manifest:
        # Stable marker, like the two tokens handler.py owns: a CloudWatch
        # metric filter can count artifacts still serving unverified.
        logger.warning(
            "[classify.unverified_extractor] model.json has no %r block;"
            " serving without checking the extractor that produced its"
            " training features",
            MANIFEST_KEY,
        )
        return None
    return ExtractorSpec.from_manifest(manifest)


def _predict_graph(
    files: ModelFiles, features: ImageFeatures, points: list[tuple[int, int]]
) -> tuple[list[str], list[list[float]]]:
    """Batched scoring through the portable artifact's TorchScript head."""
    # mermaid-classifier is absent from the legacy image, so this loader is
    # imported inside the branch that uses it.
    from mermaid_classifier.pyspacer.inference import load_predictor

    predictor = load_predictor(files.model_pt, files.model_json)
    batch = np.vstack([features.get_array((row, col)) for row, col in points])
    return list(predictor.classes), predictor.predict_proba(batch).tolist()


def _predict_legacy(
    files: LegacyModelFiles, features: ImageFeatures, points: list[tuple[int, int]]
) -> tuple[list[str], list[list[float]]]:
    """One predict_proba per point over its own (1, 1280) float64 row, as
    spacer.tasks.classify_features does: batch size changes BLAS reduction
    order, and these scores must reproduce the in-process baseline to within
    1e-5. load_classifier is lru_cached, so warm invocations reuse the
    classifier."""
    clf = load_classifier(DataLocation("filesystem", str(files.classifier_pkl)))
    proba = [clf.predict_proba(features.get_array((row, col))).tolist()[0] for row, col in points]
    classes = clf.classes_
    assert (
        classes is not None
    )  # a stored classifier is always fitted (store_classifier requires it)
    return [str(c) for c in classes], proba
