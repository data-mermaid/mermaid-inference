from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from mermaid_inference_contract.locations import S3Location


class PyspacerRequest(BaseModel):
    classifier_type: Literal["pyspacer"]  # discriminator + lane key
    classifier_version: str  # the ONLY model selector the API sends
    image: S3Location
    points: list[tuple[int, int]]  # (row, col) pairs
    feature_vector_output: S3Location | None = None  # in contract; None this round
    traceparent: str | None = None  # W3C trace-context, carried in the envelope


# Discriminated union on classifier_type. A `segmentation` member is added here
# when that lane lands — no change to the existing member's fields.
ClassifyRequest = Annotated[
    Union[PyspacerRequest],
    Field(discriminator="classifier_type"),
]

_request_adapter: TypeAdapter = TypeAdapter(ClassifyRequest)


def parse_classify_request(data: dict | str | bytes) -> PyspacerRequest:
    """Validate a raw request and route it by classifier_type. Raises
    pydantic ValidationError on an unknown/missing classifier_type or bad fields."""
    if isinstance(data, (str, bytes)):
        return _request_adapter.validate_json(data)
    return _request_adapter.validate_python(data)
