"""Storage-agnostic classify core: extract features with pyspacer's EfficientNet
extractor, then predict. Both serving formats share the extraction half and
branch only on the predict half — "graph" runs the portable TorchScript head via
load_predictor, "legacy" runs the pickled scikit-learn classifier through
pyspacer's own loader. Takes pyspacer DataLocations, so it runs over
s3/filesystem/memory storage identically — which is what makes it testable
without AWS."""
from __future__ import annotations

import logging
from operator import itemgetter
from typing import NamedTuple

import numpy as np
from spacer.extractors import EfficientNetExtractor
from spacer.messages import DataLocation
from spacer.storage import load_classifier, load_image
from spacer.task_utils import check_extract_inputs

from mermaid_inference_contract import PointResult, PointScore

from pyspacer_function.resolver import LegacyModelFiles, ModelFiles

logger = logging.getLogger(__name__)


class ClassifyOutcome(NamedTuple):
    point_results: list[PointResult]
    valid_rowcol: bool
    feature_stored: bool  # False when no location was requested, or the write failed


def classify(
    image_loc,
    files: ModelFiles | LegacyModelFiles,
    points,
    *,
    extractor=None,
    feature_output_loc: DataLocation | None = None,
) -> ClassifyOutcome:
    """Classify each point. When feature_output_loc is given, the extracted
    features are persisted there once predict succeeds; nothing downstream
    reads that artifact back, so a failed write is logged and reported through
    feature_stored rather than raised."""
    points = [(int(r), int(c)) for r, c in points]

    image = load_image(image_loc)
    check_extract_inputs(image, points, image_loc.key)

    if extractor is None:
        extractor = EfficientNetExtractor(
            data_locations=dict(
                weights=DataLocation("filesystem", str(files.efficientnet_pt))
            )
        )
    features, _ = extractor(image, points)

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


def _predict_graph(files: ModelFiles, features, points):
    """Batched scoring through the portable artifact's TorchScript head."""
    # mermaid-classifier is absent from the legacy image, so this loader is
    # imported inside the branch that uses it.
    from mermaid_classifier.pyspacer.inference import load_predictor

    predictor = load_predictor(files.model_pt, files.model_json)
    batch = np.vstack([features.get_array((row, col)) for row, col in points])
    return list(predictor.classes), predictor.predict_proba(batch).tolist()


def _predict_legacy(files: LegacyModelFiles, features, points):
    """One predict_proba per point over its own (1, 1280) float64 row, as
    spacer.tasks.classify_features does: batch size changes BLAS reduction
    order, and these scores must reproduce the in-process baseline to within
    1e-5. load_classifier is lru_cached, so warm invocations reuse the
    classifier."""
    clf = load_classifier(DataLocation("filesystem", str(files.classifier_pkl)))
    proba = [
        clf.predict_proba(features.get_array((row, col))).tolist()[0] for row, col in points
    ]
    return [str(c) for c in clf.classes_], proba
