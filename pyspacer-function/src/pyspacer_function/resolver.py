"""Resolve a classifier_version to local model files (efficientnet.pt,
model.pt, model.json). Two backends: S3 (prod, caches to /tmp/<version>/) and
Local (dev/test). No torch/pyspacer here; boto3 is imported lazily."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspacer_function import config

_FILENAMES = ("efficientnet.pt", "model.pt", "model.json")


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


class LocalBackend:
    """Reads $LOCAL_MODELS_DIR/<version>/ in place (no copy)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def resolve(self, version: str) -> ModelFiles:
        files = ModelFiles.in_dir(self.root / version)
        for path in (files.efficientnet_pt, files.model_pt, files.model_json):
            if not path.exists():
                raise FileNotFoundError(path)
        return files


class S3Backend:
    """Downloads s3://<bucket>/classifier/<version>/* to <cache_root>/<version>/
    once; warm invocations reuse the cached files."""

    def __init__(self, bucket: str, cache_root: Path = Path("/tmp"), client=None):
        self.bucket = bucket
        self.cache_root = Path(cache_root)
        self._client = client

    def _s3(self):
        if self._client is None:
            import boto3  # lazy: keep module import light

            self._client = boto3.client("s3")
        return self._client

    def resolve(self, version: str) -> ModelFiles:
        dest_dir = self.cache_root / version
        files = ModelFiles.in_dir(dest_dir)
        for name in _FILENAMES:
            dest = dest_dir / name
            if not dest.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                self._s3().download_file(
                    self.bucket, f"classifier/{version}/{name}", str(dest)
                )
        return files


def get_resolver():
    """Env-selected resolver: CONFIG_BUCKET -> S3Backend, else LOCAL_MODELS_DIR
    -> LocalBackend."""
    bucket = config.config_bucket()
    if bucket:
        return S3Backend(bucket)
    local = config.local_models_dir()
    if local:
        return LocalBackend(Path(local))
    raise RuntimeError(
        "No model source configured: set CONFIG_BUCKET (prod) or "
        "LOCAL_MODELS_DIR (dev)."
    )
