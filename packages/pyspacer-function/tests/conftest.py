"""Shared fixtures: a tiny TorchScript model.pt + manifest, a real prefit
classifier.pkl for the legacy format, and a fake extractor — so classify/handler
tests need neither real EfficientNet weights nor S3."""
import json
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from spacer.data_classes import ImageFeatures, PointFeatures
from spacer.messages import DataLocation
from spacer.storage import store_classifier

from pyspacer_function.resolver import LegacyModelFiles, ModelFiles

_IN_DIM = 4
_CLASSES = ["a::", "b::", "c::"]
# Beta's labels are ba_uuid::gf_uuid, and are read off the pickled classifier —
# distinct from _CLASSES so a legacy result cannot pass on graph labels.
_LEGACY_CLASSES = ["1111::", "2222::3333", "4444::"]


class _ProbaHead(torch.nn.Module):
    """Maps (N, in_dim) features to (N, n_classes) probabilities via softmax —
    a stand-in for the calibrated head; enough to exercise load_predictor."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return torch.softmax(self.lin(x), dim=1)


def write_model_files(dest: Path) -> ModelFiles:
    """Write efficientnet.pt (stub) + model.pt (TorchScript) + model.json into
    dest/, returning the ModelFiles."""
    dest.mkdir(parents=True, exist_ok=True)
    head = _ProbaHead(_IN_DIM, len(_CLASSES))
    head.eval()
    torch.jit.script(head).save(str(dest / "model.pt"))
    (dest / "model.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "pyspacer_mlp_classifier",
                "classes": _CLASSES,
                "input_dim": _IN_DIM,
                "config": {"patch_size": 224},
                "trained_with": {
                    "torch": torch.__version__,
                    "sklearn": _pkg_version("scikit-learn"),
                    "pyspacer": _pkg_version("pyspacer"),
                },
            }
        )
    )
    (dest / "efficientnet.pt").write_bytes(b"")  # unused: extractor is injected
    return ModelFiles.in_dir(dest)


def write_legacy_model_files(dest: Path) -> LegacyModelFiles:
    """Write efficientnet_weights.pt (stub) + classifier.pkl into dest/. The
    pickle is a prefit CalibratedClassifierCV stored through spacer: pyspacer's
    ClassifierUnpickler rejects anything else on load."""
    dest.mkdir(parents=True, exist_ok=True)
    per_class = 10
    centres = np.eye(len(_LEGACY_CLASSES), _IN_DIM)
    noise = np.random.RandomState(0).normal(0, 0.05, (len(_LEGACY_CLASSES) * per_class, _IN_DIM))
    features = np.repeat(centres, per_class, axis=0) + noise
    labels = [_LEGACY_CLASSES[i // per_class] for i in range(len(features))]
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0)
    mlp.fit(features, labels)
    clf = CalibratedClassifierCV(mlp, cv="prefit").fit(features, labels)
    store_classifier(DataLocation("filesystem", str(dest / "classifier.pkl")), clf)
    (dest / "efficientnet_weights.pt").write_bytes(b"")  # unused: extractor is injected
    return LegacyModelFiles.in_dir(dest)


class FakeExtractor:
    """Returns precomputed feature vectors keyed by (row, col), mimicking
    EfficientNetExtractor's (ImageFeatures, return_msg) signature."""

    def __init__(self, vectors: dict[tuple[int, int], list[float]]):
        self.vectors = vectors

    def __call__(self, image, rowcols):
        pfs = [
            PointFeatures(row=r, col=c, data=self.vectors[(r, c)]) for r, c in rowcols
        ]
        feats = ImageFeatures(
            point_features=pfs,
            valid_rowcol=True,
            feature_dim=len(pfs[0].data),
            npoints=len(pfs),
        )
        return feats, None


@pytest.fixture
def model_files(tmp_path) -> ModelFiles:
    return write_model_files(tmp_path / "m")


@pytest.fixture
def make_model_dir():
    """Returns the write_model_files helper, so tests can lay out a
    <root>/<version>/ model directory without a cross-module import."""
    return write_model_files


@pytest.fixture
def fake_extractor_cls():
    """Returns the FakeExtractor class (request it instead of importing it, so
    pytest's import mode never matters)."""
    return FakeExtractor


@pytest.fixture
def legacy_model_files(tmp_path) -> LegacyModelFiles:
    return write_legacy_model_files(tmp_path / "legacy")


@pytest.fixture
def make_legacy_model_dir():
    """Returns the write_legacy_model_files helper, so tests can lay out a
    <root>/<version>/ legacy model directory without a cross-module import."""
    return write_legacy_model_files


@pytest.fixture
def legacy_classes() -> list[str]:
    """The label set trained into the legacy fixture's classifier.pkl."""
    return list(_LEGACY_CLASSES)
