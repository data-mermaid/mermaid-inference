"""Shared fixtures: a tiny TorchScript model.pt + manifest, and a fake
extractor — so classify/handler tests need neither real EfficientNet weights
nor S3."""
import json
from pathlib import Path

import pytest
import torch
from spacer.data_classes import ImageFeatures, PointFeatures

from pyspacer_function.resolver import ModelFiles

_IN_DIM = 4
_CLASSES = ["a::", "b::", "c::"]


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
                "trained_with": {"torch": torch.__version__, "sklearn": "1.5.2"},
            }
        )
    )
    (dest / "efficientnet.pt").write_bytes(b"")  # unused: extractor is injected
    return ModelFiles.in_dir(dest)


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
