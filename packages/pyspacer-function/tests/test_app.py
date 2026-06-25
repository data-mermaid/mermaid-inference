import pyspacer_function.classify as classify_mod
from fastapi.testclient import TestClient
from mermaid_inference_contract import PointResult, PointScore


def test_post_classify_returns_response(monkeypatch, tmp_path, make_model_dir):
    root = tmp_path / "models"
    make_model_dir(root / "v1")
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(root))
    monkeypatch.setenv("CLASSIFIER_VERSION", "v1")
    monkeypatch.delenv("CONFIG_BUCKET", raising=False)
    monkeypatch.setattr(
        classify_mod,
        "classify",
        lambda *a, **k: (
            [PointResult(row=1, col=1, scores=[PointScore(label="a::", score=1.0)])],
            True,
        ),
    )

    from pyspacer_function.app import app

    client = TestClient(app)
    resp = client.post(
        "/classify",
        json={
            "classifier_type": "pyspacer",
            "image": {"bucket": "b", "key": "k.jpg"},
            "points": [[1, 1]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier_version"] == "v1"
    assert body["point_results"][0]["scores"][0]["label"] == "a::"


def test_post_classify_bad_payload_returns_validation_envelope(monkeypatch):
    from pyspacer_function.app import app

    client = TestClient(app)
    resp = client.post("/classify", json={"classifier_type": "pyspacer"})
    assert resp.status_code == 200
    assert resp.json()["error_code"] == "validation_error"
