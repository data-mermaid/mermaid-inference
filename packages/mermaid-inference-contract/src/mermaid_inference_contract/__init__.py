"""mermaid-inference-contract: the request/response schema shared by the
MERMAID API worker and ML inference. Pure pydantic — no torch, no boto3."""

from mermaid_inference_contract.errors import ErrorCode, ErrorEnvelope
from mermaid_inference_contract.locations import S3Location
from mermaid_inference_contract.request import (
    ClassifyRequest,
    PyspacerRequest,
    parse_classify_request,
)
from mermaid_inference_contract.response import (
    ClassifyResponse,
    PointResult,
    PointScore,
    PyspacerResponse,
    parse_classify_response,
)
from mermaid_inference_contract.tracing import (
    Traceparent,
    format_traceparent,
    new_traceparent,
    parse_traceparent,
)

__version__ = "0.3.0"

__all__ = [
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
    "__version__",
]
