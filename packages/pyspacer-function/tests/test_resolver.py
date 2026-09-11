from pathlib import Path

import pytest

from pyspacer_function.resolver import (
    LegacyModelFiles,
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


def _make_legacy_version_dir(root: Path, version: str) -> Path:
    d = root / version
    d.mkdir(parents=True)
    (d / "efficientnet_weights.pt").write_bytes(b"eff")
    (d / "classifier.pkl").write_bytes(b"pkl")
    return d


def test_local_backend_legacy_resolves_the_legacy_pair(tmp_path):
    _make_legacy_version_dir(tmp_path, "v1")
    files = LocalBackend(tmp_path, model_format="legacy").resolve("v1")
    assert isinstance(files, LegacyModelFiles)
    assert files.efficientnet_pt == tmp_path / "v1" / "efficientnet_weights.pt"
    assert files.classifier_pkl.read_bytes() == b"pkl"


def test_local_backend_legacy_missing_classifier_raises(tmp_path):
    d = tmp_path / "v1"
    d.mkdir(parents=True)
    (d / "efficientnet_weights.pt").write_bytes(b"eff")
    with pytest.raises(FileNotFoundError):
        LocalBackend(tmp_path, model_format="legacy").resolve("v1")


def test_local_backend_legacy_rejects_a_graph_directory(tmp_path):
    _make_version_dir(tmp_path, "v1")  # efficientnet.pt/model.pt/model.json only
    with pytest.raises(FileNotFoundError):
        LocalBackend(tmp_path, model_format="legacy").resolve("v1")


def test_s3_backend_legacy_downloads_exactly_the_legacy_pair(tmp_path):
    fake = _FakeS3()
    backend = S3Backend("cfg-bucket", cache_root=tmp_path, client=fake, model_format="legacy")

    files = backend.resolve("v1")

    assert isinstance(files, LegacyModelFiles)
    keys = [key for _, key, _ in fake.calls]
    assert keys == [
        "classifier/v1/efficientnet_weights.pt",
        "classifier/v1/classifier.pkl",
    ]
    # classifier_labels.pkl sits in the same prefix but holds a stale CoralNet-era
    # label set: the served labels come from the pickled classifier's classes_.
    assert "classifier/v1/classifier_labels.pkl" not in keys
    assert files.classifier_pkl.read_text() == "cfg-bucket/classifier/v1/classifier.pkl"


def test_get_resolver_honours_legacy_format(monkeypatch, tmp_path):
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("CLASSIFIER_FORMAT", "legacy")
    _make_legacy_version_dir(tmp_path, "v1")

    assert isinstance(get_resolver().resolve("v1"), LegacyModelFiles)


def test_get_resolver_defaults_to_graph_format(monkeypatch, tmp_path):
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.delenv("CLASSIFIER_FORMAT", raising=False)
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))
    _make_version_dir(tmp_path, "v1")

    assert isinstance(get_resolver().resolve("v1"), ModelFiles)
