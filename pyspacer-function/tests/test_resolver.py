from pathlib import Path

import pytest

from pyspacer_function.resolver import (
    LocalBackend,
    ModelFiles,
    S3Backend,
    get_resolver,
)


def _make_version_dir(root: Path, version: str) -> Path:
    d = root / version
    d.mkdir(parents=True)
    (d / "efficientnet.pt").write_bytes(b"eff")
    (d / "model.pt").write_bytes(b"pt")
    (d / "model.json").write_text("{}")
    return d


def test_local_backend_resolves_existing_files(tmp_path):
    _make_version_dir(tmp_path, "v1")
    files = LocalBackend(tmp_path).resolve("v1")
    assert isinstance(files, ModelFiles)
    assert files.model_pt == tmp_path / "v1" / "model.pt"
    assert files.model_json.read_text() == "{}"


def test_local_backend_missing_version_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalBackend(tmp_path).resolve("nope")


class _FakeS3:
    """Records download_file calls and writes a stub file at dest."""

    def __init__(self):
        self.calls = []

    def download_file(self, bucket, key, dest):
        self.calls.append((bucket, key, dest))
        Path(dest).write_text(f"{bucket}/{key}")


def test_s3_backend_downloads_to_tmp_then_caches(tmp_path):
    fake = _FakeS3()
    backend = S3Backend("cfg-bucket", cache_root=tmp_path, client=fake)

    files = backend.resolve("v2")
    assert files.model_pt == tmp_path / "v2" / "model.pt"
    assert files.model_pt.read_text() == "cfg-bucket/classifier/v2/model.pt"
    assert len(fake.calls) == 3  # efficientnet.pt, model.pt, model.json

    # Second resolve: files already cached -> no new downloads.
    backend.resolve("v2")
    assert len(fake.calls) == 3


def test_get_resolver_prefers_s3_when_config_bucket_set(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_BUCKET", "cfg-bucket")
    monkeypatch.delenv("LOCAL_MODELS_DIR", raising=False)
    assert isinstance(get_resolver(), S3Backend)


def test_get_resolver_uses_local_when_only_local_set(monkeypatch, tmp_path):
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))
    assert isinstance(get_resolver(), LocalBackend)


def test_get_resolver_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.delenv("LOCAL_MODELS_DIR", raising=False)
    with pytest.raises(RuntimeError):
        get_resolver()
