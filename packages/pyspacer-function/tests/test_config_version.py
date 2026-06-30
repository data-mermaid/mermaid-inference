import pytest

from pyspacer_function import config


def test_classifier_version_reads_env(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    assert config.classifier_version() == "v2"


def test_classifier_version_required(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_VERSION", raising=False)
    with pytest.raises(RuntimeError):
        config.classifier_version()
