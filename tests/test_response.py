import pytest
from pydantic import ValidationError

from mermaid_inference_contract.response import (
    PointResult,
    PointScore,
    PyspacerResponse,
    parse_classify_response,
)


def _payload():
    return {
        "classifier_type": "pyspacer",
        "classifier_version": "v4",
        "point_results": [
            {
                "row": 100,
                "col": 150,
                "scores": [
                    {"label": "ba_uuid::gf_uuid", "score": 0.9},
                    {"label": "ba_uuid", "score": 0.1},
                ],
            }
        ],
        "valid_rowcol": True,
    }


def test_label_is_opaque_string():
    s = PointScore(label="ba_uuid::gf_uuid", score=0.42)
    assert s.label == "ba_uuid::gf_uuid"


def test_response_is_flat_no_result_wrapper():
    resp = PyspacerResponse.model_validate(_payload())
    assert isinstance(resp.point_results[0], PointResult)
    assert resp.valid_rowcol is True
    assert resp.traceparent is None
    # flat: point_results sits at top level, not under a `result` key
    assert "result" not in resp.model_dump()


def test_roundtrips_through_json():
    resp = PyspacerResponse.model_validate(_payload())
    assert PyspacerResponse.model_validate_json(resp.model_dump_json()) == resp


def test_parse_routes_pyspacer():
    resp = parse_classify_response(_payload())
    assert isinstance(resp, PyspacerResponse)


def test_parse_rejects_unknown_classifier_type():
    bad = _payload() | {"classifier_type": "segmentation"}
    with pytest.raises(ValidationError):
        parse_classify_response(bad)
