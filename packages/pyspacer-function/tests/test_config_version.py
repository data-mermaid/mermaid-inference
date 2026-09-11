import pytest

from pyspacer_function import config


def test_classifier_version_reads_env(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    assert config.classifier_version() == "v2"


def test_classifier_version_required(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_VERSION", raising=False)
    with pytest.raises(RuntimeError):
        config.classifier_version()


def test_classifier_format_defaults_to_graph(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_FORMAT", raising=False)
    assert config.classifier_format() == "graph"


def test_classifier_format_reads_env(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_FORMAT", "legacy")
    assert config.classifier_format() == "legacy"


def test_classifier_format_rejects_unknown(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_FORMAT", "segmentation")
    with pytest.raises(RuntimeError, match="segmentation"):
        config.classifier_format()
