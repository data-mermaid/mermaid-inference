from mermaid_inference_contract.errors import ErrorCode, ErrorEnvelope


def test_validation_error_is_non_retryable_by_default():
    env = ErrorEnvelope(error_code=ErrorCode.VALIDATION_ERROR, message="bad payload")
    assert env.retryable is False
    assert env.classifier_type is None
    assert env.traceparent is None


def test_envelope_roundtrips_through_json():
    env = ErrorEnvelope(
        error_code=ErrorCode.PROCESSING_ERROR,
        message="boom",
        retryable=True,
        classifier_type="pyspacer",
        classifier_version="v4",
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )
    again = ErrorEnvelope.model_validate_json(env.model_dump_json())
    assert again == env
    assert again.error_code is ErrorCode.PROCESSING_ERROR


def test_error_code_serializes_as_string_value():
    env = ErrorEnvelope(error_code=ErrorCode.VALIDATION_ERROR, message="x")
    assert '"validation_error"' in env.model_dump_json()
