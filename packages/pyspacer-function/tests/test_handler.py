import ast
import json
from pathlib import Path

import pyspacer_function.classify as classify_mod
import pyspacer_function.compat as compat_mod
import pyspacer_function.handler as handler_mod
from mermaid_inference_contract import PointResult, PointScore
from pyspacer_function.handler import handler


def _event(traceparent="tp-1"):
    return {
        "classifier_type": "pyspacer",
        "image": {"bucket": "b", "key": "k.jpg"},
        "points": [[10, 10]],
        "traceparent": traceparent,
    }


def test_handler_returns_pyspacer_response(monkeypatch, tmp_path, make_model_dir):
    root = tmp_path / "models"
    make_model_dir(root / "v2")  # fixture writes a model.json whose trained_with matches runtime
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
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
    assert out["classifier_version"] == "v2"  # provenance from deploy config
    assert out["valid_rowcol"] is True
    assert out["traceparent"] == "tp-1"
    assert out["point_results"][0]["scores"][0]["label"] == "a::"


def test_handler_reports_feature_vector_output_when_requested(
    monkeypatch, tmp_path, make_model_dir
):
    root = tmp_path / "models"
    make_model_dir(root / "v2")
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=10, col=10, scores=[PointScore(label="a::", score=1.0)])],
            True,
        ),
    )

    event = _event()
    event["feature_vector_output"] = {"bucket": "fb", "key": "features/out.featurevector"}

    out = handler(event)
    assert out["feature_vector_output"] == {"bucket": "fb", "key": "features/out.featurevector"}


def test_handler_leaves_feature_vector_output_none_when_not_requested(
    monkeypatch, tmp_path, make_model_dir
):
    root = tmp_path / "models"
    make_model_dir(root / "v2")
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=10, col=10, scores=[PointScore(label="a::", score=1.0)])],
            True,
        ),
    )

    out = handler(_event())  # no feature_vector_output in the request
    assert out["feature_vector_output"] is None


def test_handler_feature_vector_write_failure_surfaces_as_processing_error(
    monkeypatch, tmp_path, make_model_dir, caplog
):
    root = tmp_path / "models"
    make_model_dir(root / "v2")
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    def _classify_with_failing_store(*a, feature_output_loc=None, **k):
        # The handler must forward the request's location through unchanged.
        assert feature_output_loc is not None
        assert feature_output_loc.bucket_name == "fb"
        assert feature_output_loc.key == "features/out.featurevector"
        raise RuntimeError("feature vector store failed")

    monkeypatch.setattr(classify_mod, "classify", _classify_with_failing_store)

    event = _event(traceparent="tp-fv-fail")
    event["feature_vector_output"] = {"bucket": "fb", "key": "features/out.featurevector"}

    with caplog.at_level("ERROR"):
        out = handler(event)

    assert out["error_code"] == "processing_error"
    assert out["traceparent"] == "tp-fv-fail"
    # Confirms the RuntimeError from the store attempt propagated, not an
    # earlier failure from a wrong/missing feature_output_loc forward.
    assert out["message"] == "feature vector store failed"
    assert "[classify.processing_error]" in caplog.text


def test_handler_validation_error_on_bad_payload():
    out = handler({"classifier_type": "pyspacer"})  # missing required fields
    assert out["error_code"] == "validation_error"


def test_handler_validation_error_echoes_raw_traceparent():
    out = handler({"classifier_type": "pyspacer", "traceparent": "tp-val"})  # missing required fields
    assert out["error_code"] == "validation_error"
    assert out["traceparent"] == "tp-val"


def test_handler_processing_error_carries_traceparent(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))  # no version dir present
    monkeypatch.setenv("CLASSIFIER_VERSION", "missing")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    out = handler(_event(traceparent="tp-2"))
    assert out["error_code"] == "processing_error"
    assert out["traceparent"] == "tp-2"
    assert out["classifier_version"] == "missing"


def test_handler_processing_error_logs_metric_filter_marker(monkeypatch, tmp_path, caplog):
    # The marker drives the CloudWatch Logs metric filter + alarm in mermaid-api.
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path))  # no version dir present
    monkeypatch.setenv("CLASSIFIER_VERSION", "missing")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    with caplog.at_level("ERROR"):
        out = handler(_event(traceparent="tp-marker"))
    assert out["error_code"] == "processing_error"
    assert "[classify.processing_error]" in caplog.text
    # The marker line must carry the trace id so a processing failure is
    # joinable to the API worker's log (see inference-traceparent-tracing spec).
    assert "tp-marker" in caplog.text


def test_handler_logs_traceparent_on_success(monkeypatch, tmp_path, make_model_dir, caplog):
    root = tmp_path / "models"
    make_model_dir(root / "v2")  # fixture writes a model.json whose trained_with matches runtime
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=10, col=10, scores=[PointScore(label="a::", score=1.0)])],
            True,
        ),
    )

    with caplog.at_level("INFO"):
        out = handler(_event(traceparent="tp-success"))
    assert out["traceparent"] == "tp-success"
    traceparent_records = [
        r for r in caplog.records if getattr(r, "traceparent", None) == "tp-success"
    ]
    # Expect at least the handler-entry log plus a success-completion log, both
    # carrying the same trace id via structured `extra`.
    assert len(traceparent_records) >= 2


def test_handler_logs_traceparent_at_entry_on_validation_error(caplog):
    with caplog.at_level("INFO"):
        out = handler({"classifier_type": "pyspacer", "traceparent": "tp-val-log"})
    assert out["error_code"] == "validation_error"
    assert any(getattr(r, "traceparent", None) == "tp-val-log" for r in caplog.records)


def test_handler_validation_error_does_not_log_processing_marker(caplog):
    # Client-side validation failures must NOT match the processing-error filter.
    with caplog.at_level("ERROR"):
        out = handler({"classifier_type": "pyspacer"})  # missing required fields
    assert out["error_code"] == "validation_error"
    assert "[classify.processing_error]" not in caplog.text


def test_handler_stamps_contract_version(monkeypatch, tmp_path, make_model_dir):
    import mermaid_inference_contract as contract

    root = tmp_path / "models"
    make_model_dir(root / "v2")  # fixture writes a model.json whose trained_with matches runtime
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v2")
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
    assert out["contract_version"] == contract.__version__


def test_handler_validation_error_stamps_contract_version():
    import mermaid_inference_contract as contract

    out = handler({"classifier_type": "pyspacer"})  # missing required fields
    assert out["contract_version"] == contract.__version__


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


def test_handler_legacy_format_succeeds_without_model_json(
    monkeypatch, tmp_path, make_legacy_model_dir
):
    root = tmp_path / "models"
    make_legacy_model_dir(root / "v-legacy")
    # No model.json anywhere under the version dir: a run through the graph
    # gate would blow up on files.model_json, which LegacyModelFiles has no
    # such field for.
    assert not (root / "v-legacy" / "model.json").exists()

    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v-legacy")
    monkeypatch.setenv("CLASSIFIER_FORMAT", "legacy")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)

    # legacy_pins.txt carries placeholder production versions that need not
    # match this dev environment; the pin-matching itself is compat.py's own
    # test coverage, not the dispatch behavior under test here.
    monkeypatch.setattr(compat_mod, "check_legacy_pins", lambda: None)
    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=10, col=10, scores=[PointScore(label="1111::", score=1.0)])],
            True,
        ),
    )

    out = handler(_event(traceparent="tp-legacy"))
    assert "error_code" not in out  # a processing_error means the graph gate ran instead
    assert out["point_results"][0]["scores"][0]["label"] == "1111::"
    assert out["classifier_version"] == "v-legacy"
    assert out["traceparent"] == "tp-legacy"


def test_handler_graph_format_runs_check_compatibility_on_mismatch(
    monkeypatch, tmp_path, make_model_dir
):
    root = tmp_path / "models"
    make_model_dir(root / "v3")
    manifest_path = root / "v3" / "model.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trained_with"]["pyspacer"] = "0.0.1"  # force a mismatch against the runtime
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v3")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.delenv("CLASSIFIER_FORMAT", raising=False)  # default: graph

    out = handler(_event(traceparent="tp-mismatch"))
    assert out["error_code"] == "processing_error"
    assert out["traceparent"] == "tp-mismatch"
    # Names check_compatibility's own mismatch message, so this fails if the
    # graph path is ever routed to check_legacy_pins instead.
    assert "pyspacer: model built with 0.0.1" in out["message"]
