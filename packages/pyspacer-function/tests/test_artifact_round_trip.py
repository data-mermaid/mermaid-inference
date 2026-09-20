"""The producer's artifact, served by the real consumer.

Every other test here hand-writes model.json, so none of them would notice
mermaid-classifier changing what export_artifact emits. This one exports a
real artifact through the real export path and serves it through classify(),
which is the only place the two repos' understanding of the manifest is
checked against each other.
"""

import numpy as np
import pytest
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from spacer.data_classes import DataLocation as _DataLocation
from spacer.extractors import EfficientNetExtractor
from spacer.messages import DataLocation

from mermaid_classifier.pyspacer.inference import (
    ExtractorMismatchError,
    ExtractorSpec,
    export_artifact,
)
from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier
from pyspacer_function.classify import classify
from pyspacer_function.resolver import ModelFiles

IN_DIM = 4
CLASSES = ["ba1::", "ba2::gf2", "ba3::"]
WEIGHTS_SHA256 = "a" * 64
# One separable centre per class; a probe sitting on a centre must come back
# as that class, which is what makes the served scores meaningful here.
CENTRES = (np.eye(len(CLASSES), IN_DIM) * 5).astype(np.float32)


def _calibrated_model():
    """A small prefit CalibratedClassifierCV over a TorchMLPClassifier.

    The inner estimator has to be TorchMLPClassifier: build_calibrated_head
    reads its torch module to freeze the graph, so sklearn's own MLPClassifier
    would not be exportable.
    """
    per_class = 20
    noise = np.random.RandomState(0).normal(0, 0.05, (len(CLASSES) * per_class, IN_DIM))
    features = (np.repeat(CENTRES, per_class, axis=0) + noise).astype(np.float32)
    labels = np.array([CLASSES[i // per_class] for i in range(len(features))])

    clf = TorchMLPClassifier(hidden_layer_sizes=(8,), random_state=0)
    for _ in range(40):
        clf.partial_fit(features, labels, classes=list(CLASSES))
    return CalibratedClassifierCV(clf, cv="prefit").fit(features, labels), features


def _spec() -> ExtractorSpec:
    """Derived off the real extractor, as the training pipeline derives it."""
    extractor = EfficientNetExtractor(
        data_locations={"weights": _DataLocation("filesystem", "unused.pt")}
    )
    return ExtractorSpec.from_extractor(
        extractor, weights_uri="s3://bucket/efficientnet.pt", weights_sha256=WEIGHTS_SHA256
    )


@pytest.fixture
def exported_model_files(tmp_path) -> ModelFiles:
    model, features = _calibrated_model()
    dest = tmp_path / "vX"
    dest.mkdir()
    # feature_dim has to match the head's input width, which is what makes the
    # spec a claim about this model rather than a free-floating record.
    spec = ExtractorSpec.from_dict({**_spec().to_dict(), "feature_dim": IN_DIM})
    export_artifact(model, dest, features, extractor=spec)
    (dest / "efficientnet.pt").write_bytes(b"")  # unused: extractor is injected
    return ModelFiles.in_dir(dest)


def _image_and_points(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(img_path)
    return DataLocation("filesystem", str(img_path)), [(10, 10)]


def test_an_exported_artifact_serves(tmp_path, exported_model_files, fake_extractor_cls):
    image_loc, points = _image_and_points(tmp_path)

    for index, label in enumerate(CLASSES):
        results, valid, _ = classify(
            image_loc,
            exported_model_files,
            points,
            extractor=fake_extractor_cls({(10, 10): CENTRES[index].tolist()}),
        )
        assert valid is True
        assert results[0].scores[0].label == label
        assert {s.label for s in results[0].scores} == set(CLASSES)


def test_the_exported_manifest_is_what_the_serving_gate_reads(
    tmp_path, exported_model_files, fake_extractor_cls
):
    # Not a restatement of test_classify's crop check: that one uses a
    # hand-written manifest, so it cannot catch export_artifact writing the
    # block under a different key or shape.
    class _SmallCrop(fake_extractor_cls):
        CROP_SIZE = 64

    image_loc, points = _image_and_points(tmp_path)
    with pytest.raises(ExtractorMismatchError, match="crop_size"):
        classify(
            image_loc,
            exported_model_files,
            points,
            extractor=_SmallCrop({(10, 10): [1.0, 0.0, 0.0, 0.0]}),
        )
