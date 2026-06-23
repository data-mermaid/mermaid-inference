import pytest
from pydantic import ValidationError

from mermaid_inference_contract.locations import S3Location
from mermaid_inference_contract.request import PyspacerRequest, parse_classify_request


def _payload():
    return {
        "classifier_type": "pyspacer",
        "classifier_version": "v4",
        "image": {"bucket": "imgs", "key": "a.png"},
        "points": [[100, 150], [200, 300]],
    }


def test_defaults_feature_vector_and_traceparent_none():
    req = PyspacerRequest(
        classifier_type="pyspacer",
        classifier_version="v4",
        image=S3Location(bucket="imgs", key="a.png"),
        points=[(1, 2)],
    )
    assert req.feature_vector_output is None
    assert req.traceparent is None


def test_points_coerced_to_tuples():
    req = PyspacerRequest.model_validate(_payload())
    assert req.points == [(100, 150), (200, 300)]


def test_roundtrips_through_json():
    req = PyspacerRequest.model_validate(_payload())
    assert PyspacerRequest.model_validate_json(req.model_dump_json()) == req


def test_parse_routes_pyspacer():
    req = parse_classify_request(_payload())
    assert isinstance(req, PyspacerRequest)
    assert req.classifier_version == "v4"


def test_parse_accepts_json_string():
    import json

    req = parse_classify_request(json.dumps(_payload()))
    assert isinstance(req, PyspacerRequest)


def test_parse_rejects_unknown_classifier_type():
    bad = _payload() | {"classifier_type": "segmentation"}
    with pytest.raises(ValidationError):
        parse_classify_request(bad)
