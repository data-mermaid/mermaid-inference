from pyspacer_function import config


def test_image_version_reads_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_IMAGE_VERSION", "0.2.0")
    assert config.image_version() == "0.2.0"


def test_image_version_defaults_to_unknown(monkeypatch):
    monkeypatch.delenv("INFERENCE_IMAGE_VERSION", raising=False)
    assert config.image_version() == "unknown"
