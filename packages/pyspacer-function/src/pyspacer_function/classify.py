"""Storage-agnostic classify core: extract features with pyspacer's EfficientNet
extractor, then predict with the portable TorchScript head via load_predictor.
Mirrors the PR #64 recipe but imports only the [inference] loader (never
annotation.py). Takes pyspacer DataLocations, so it runs over s3/filesystem/
memory storage identically — which is what makes it testable without AWS."""
from __future__ import annotations

from operator import itemgetter

import numpy as np
from spacer.data_classes import DataLocation
from spacer.extractors import EfficientNetExtractor
from spacer.storage import load_image
from spacer.task_utils import check_extract_inputs

from mermaid_classifier.pyspacer.inference import load_predictor
from mermaid_inference_contract import PointResult, PointScore

from pyspacer_function.resolver import ModelFiles


def classify(image_loc, files: ModelFiles, points, *, extractor=None):
    """Classify each point. Returns (point_results, valid_rowcol)."""
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

    predictor = load_predictor(files.model_pt, files.model_json)
    batch = np.vstack([features.get_array((r, c)) for r, c in points])
    proba = predictor.predict_proba(batch)  # (N, n_classes)

    results = []
    for i, (row, col) in enumerate(points):
        scored = sorted(
            zip(predictor.classes, proba[i].tolist()),
            key=itemgetter(1),
            reverse=True,
        )
        results.append(
            PointResult(
                row=row,
                col=col,
                scores=[PointScore(label=label, score=float(s)) for label, s in scored],
            )
        )
    return results, features.valid_rowcol
