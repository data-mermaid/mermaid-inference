from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"  # bad payload — non-retryable
    PROCESSING_ERROR = "processing_error"  # inference failure


class ErrorEnvelope(BaseModel):
    """Structured error an inference function returns (rather than raising) for
    validation or processing failures; the worker surfaces it into
    ClassificationStatus. Retry policy for retryable errors lives in the SQS
    job layer, outside this package."""

    error_code: ErrorCode
    message: str
    retryable: bool = False
    classifier_type: str | None = None
    classifier_version: str | None = None
    traceparent: str | None = None
