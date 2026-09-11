"""Resolve a classifier_version to local model files. Which files depends on
the deploy-pinned serving format: "graph" is the efficientnet.pt / model.pt /
model.json triple, "legacy" the efficientnet_weights.pt / classifier.pkl pair.
Two backends: S3 (prod, caches to /tmp/<version>/) and Local (dev/test). No
torch/pyspacer here; boto3 is imported lazily."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspacer_function import config

_FILENAMES = {
    "graph": ("efficientnet.pt", "model.pt", "model.json"),
    "legacy": ("efficientnet_weights.pt", "classifier.pkl"),
}


def _paths_in(directory: Path, model_format: str) -> tuple[Path, ...]:
    """The format's files under directory, in the dataclass's field order."""
    return tuple(directory / name for name in _FILENAMES[model_format])


@dataclass(frozen=True)
class ModelFiles:
    efficientnet_pt: Path
    model_pt: Path
    model_json: Path

    @classmethod
    def in_dir(cls, directory: Path) -> "ModelFiles":
        return cls(
            efficientnet_pt=directory / "efficientnet.pt",
            model_pt=directory / "model.pt",
            model_json=directory / "model.json",
        )


@dataclass(frozen=True)
class LegacyModelFiles:
    """The Beta artifact pair: extractor weights plus a pickled
    CalibratedClassifierCV. Field order matches _FILENAMES["legacy"]."""

    efficientnet_pt: Path
    classifier_pkl: Path

    @classmethod
    def in_dir(cls, directory: Path) -> "LegacyModelFiles":
        return cls(*_paths_in(directory, "legacy"))


_MODEL_FILES = {"graph": ModelFiles, "legacy": LegacyModelFiles}


class LocalBackend:
    """Reads $LOCAL_MODELS_DIR/<version>/ in place (no copy)."""

    def __init__(self, root: Path, model_format: str = "graph"):
        self.root = Path(root)
        self.model_format = model_format

    def resolve(self, version: str) -> ModelFiles | LegacyModelFiles:
        directory = self.root / version
        for path in _paths_in(directory, self.model_format):
            if not path.exists():
                raise FileNotFoundError(path)
        return _MODEL_FILES[self.model_format].in_dir(directory)


class S3Backend:
    """Downloads s3://<bucket>/classifier/<version>/* to <cache_root>/<version>/
    once; warm invocations reuse the cached files."""

    def __init__(
        self,
        bucket: str,
        cache_root: Path = Path("/tmp"),
        client=None,
        model_format: str = "graph",
    ):
        self.bucket = bucket
        self.cache_root = Path(cache_root)
        self._client = client
        self.model_format = model_format

    def _s3(self):
        if self._client is None:
            import boto3  # lazy: keep module import light

            self._client = boto3.client("s3")
        return self._client

    def resolve(self, version: str) -> ModelFiles | LegacyModelFiles:
        dest_dir = self.cache_root / version
        for name in _FILENAMES[self.model_format]:
            dest = dest_dir / name
            if not dest.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                self._s3().download_file(
                    self.bucket, f"classifier/{version}/{name}", str(dest)
                )
        return _MODEL_FILES[self.model_format].in_dir(dest_dir)


def get_resolver():
    """Env-selected resolver: CONFIG_BUCKET -> S3Backend, else LOCAL_MODELS_DIR
    -> LocalBackend. Both serve the deploy-pinned CLASSIFIER_FORMAT."""
    model_format = config.classifier_format()
    bucket = config.config_bucket()
    if bucket:
        return S3Backend(bucket, model_format=model_format)
    local = config.local_models_dir()
    if local:
        return LocalBackend(Path(local), model_format=model_format)
    raise RuntimeError(
        "No model source configured: set CONFIG_BUCKET (prod) or "
        "LOCAL_MODELS_DIR (dev)."
    )
