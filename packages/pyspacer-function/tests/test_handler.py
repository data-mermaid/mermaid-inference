import ast
from pathlib import Path

import pyspacer_function.classify as classify_mod
import pyspacer_function.handler as handler_mod
from mermaid_inference_contract import PointResult, PointScore
from pyspacer_function.handler import handler


def _event(version="v1", traceparent="tp-1"):
    return {
        "classifier_type": "pyspacer",
        "classifier_version": version,
        "image": {"bucket": "b", "key": "k.jpg"},
        "points": [[10, 10]],
        "traceparent": traceparent,
    }


def test_handler_returns_pyspacer_response(monkeypatch, tmp_path, make_model_dir):
    root = tmp_path / "models"
    make_model_dir(root / "v1")
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    # Avoid real extraction/S3: stub the classify core (imported lazily in handler).
    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=10, col=10, scores=[PointScore(label="a::", score=1.0)])],
            True,
        ),
    )

    out = handler(_event())
    assert out["classifier_type"] == "pyspacer"
    assert out["classifier_version"] == "v1"
    assert out["valid_rowcol"] is True
    assert out["traceparent"] == "tp-1"
    assert out["point_results"][0]["scores"][0]["label"] == "a::"


def test_handler_validation_error_on_bad_payload():
    out = handler({"classifier_type": "pyspacer"})  # missing required fields
    assert out["error_code"] == "validation_error"


def test_handler_validation_error_echoes_raw_traceparent():
    out = handler({"classifier_type": "pyspacer", "traceparent": "tp-val"})  # missing required fields
    assert out["error_code"] == "validation_error"
    assert out["traceparent"] == "tp-val"


def test_handler_processing_error_carries_traceparent(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))  # no version dir present
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    out = handler(_event(version="missing", traceparent="tp-2"))
    assert out["error_code"] == "processing_error"
    assert out["traceparent"] == "tp-2"
    assert out["classifier_version"] == "missing"


def test_handler_processing_error_logs_metric_filter_marker(monkeypatch, tmp_path, caplog):
    # The marker drives the CloudWatch Logs metric filter + alarm in mermaid-api.
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))  # no version dir present
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    with caplog.at_level("ERROR"):
        out = handler(_event(version="missing"))
    assert out["error_code"] == "processing_error"
    assert "[classify.processing_error]" in caplog.text


def test_handler_validation_error_does_not_log_processing_marker(caplog):
    # Client-side validation failures must NOT match the processing-error filter.
    with caplog.at_level("ERROR"):
        out = handler({"classifier_type": "pyspacer"})  # missing required fields
    assert out["error_code"] == "validation_error"
    assert "[classify.processing_error]" not in caplog.text


def test_handler_module_has_no_backend_import_at_module_scope():
    tree = ast.parse(Path(handler_mod.__file__).read_text())
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"torch", "spacer"}
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in {"torch", "spacer"}
            assert node.module != "pyspacer_function.classify"
