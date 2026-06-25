"""Lambda entrypoint. Module scope imports ONLY stdlib + the pure-pydantic
contract + the (torch-free) resolver/config — the torch/pyspacer backend and
the classify core are imported lazily inside handler() so the multi-second
import stays out of Lambda's 10 s INIT phase."""
import logging
import os

# /tmp is the only writable path on Lambda's read-only filesystem.
for _var in ("SPACER_EXTRACTORS_CACHE_DIR", "TORCH_HOME", "HOME", "MPLCONFIGDIR"):
    os.environ.setdefault(_var, "/tmp")

from pydantic import ValidationError

from mermaid_inference_contract import (
    ErrorCode,
    ErrorEnvelope,
    PyspacerResponse,
    parse_classify_request,
)

from pyspacer_function.config import image_version, num_threads
from pyspacer_function.resolver import get_resolver

logger = logging.getLogger(__name__)
logger.info("pyspacer-function image version: %s", image_version())


def _event_traceparent(event) -> str | None:
    return event.get("traceparent") if isinstance(event, dict) else None


def handler(event, context=None) -> dict:
    try:
        req = parse_classify_request(event)
    except ValidationError as exc:
        return ErrorEnvelope(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=str(exc),
            classifier_type="pyspacer",
            traceparent=_event_traceparent(event),
        ).model_dump(mode="json")

    try:
        import torch

        torch.set_num_threads(num_threads())
        from spacer.data_classes import DataLocation

        from pyspacer_function.classify import classify

        files = get_resolver().resolve(req.classifier_version)
        image_loc = DataLocation("s3", key=req.image.key, bucket_name=req.image.bucket)
        results, valid = classify(image_loc, files, [tuple(p) for p in req.points])

        return PyspacerResponse(
            classifier_type="pyspacer",
            classifier_version=req.classifier_version,
            point_results=results,
            valid_rowcol=valid,
            traceparent=req.traceparent,
        ).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 — surface as a processing-error envelope
        # Stable marker for the CloudWatch Logs metric filter + alarm: these
        # failures are RETURNED as PROCESSING_ERROR envelopes, so they never
        # increment the Lambda Errors metric. Keep the token in sync with the
        # MetricFilter pattern in mermaid-api InferenceStack.
        logger.exception(
            "[classify.processing_error] classify failed (classifier_version=%s)",
            req.classifier_version,
        )
        return ErrorEnvelope(
            error_code=ErrorCode.PROCESSING_ERROR,
            message=str(exc),
            classifier_type="pyspacer",
            classifier_version=req.classifier_version,
            traceparent=req.traceparent,
        ).model_dump(mode="json")
