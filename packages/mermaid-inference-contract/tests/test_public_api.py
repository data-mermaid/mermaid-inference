import mermaid_inference_contract as mic


def test_public_symbols_importable_from_root():
    expected = {
        "S3Location",
        "PyspacerRequest",
        "ClassifyRequest",
        "parse_classify_request",
        "PointScore",
        "PointResult",
        "PyspacerResponse",
        "ClassifyResponse",
        "parse_classify_response",
        "ErrorCode",
        "ErrorEnvelope",
        "Traceparent",
        "parse_traceparent",
        "format_traceparent",
        "new_traceparent",
    }
    assert set(mic.__all__) - {"__version__"} == expected
    for name in expected:
        assert hasattr(mic, name), name


def test_can_build_request_via_root_import():
    req = mic.parse_classify_request(
        {
            "classifier_type": "pyspacer",
            "image": {"bucket": "b", "key": "k"},
            "points": [[1, 2]],
        }
    )
    assert isinstance(req, mic.PyspacerRequest)
