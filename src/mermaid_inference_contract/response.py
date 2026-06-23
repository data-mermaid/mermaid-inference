from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class PointScore(BaseModel):
    label: str  # "ba_uuid::gf_uuid" or "ba_uuid" — opaque to the contract
    score: float  # calibrated probability


class PointResult(BaseModel):
    row: int
    col: int
    scores: list[PointScore]  # full per-class list, ordered by score descending


class PyspacerResponse(BaseModel):
    classifier_type: Literal["pyspacer"]
    classifier_version: str
    point_results: list[PointResult]
    valid_rowcol: bool  # from pyspacer ClassifyReturnMsg.valid_rowcol
    traceparent: str | None = None


ClassifyResponse = Annotated[
    Union[PyspacerResponse],
    Field(discriminator="classifier_type"),
]

_response_adapter: TypeAdapter = TypeAdapter(ClassifyResponse)


def parse_classify_response(data: dict | str | bytes) -> PyspacerResponse:
    """Validate a raw response and route it by classifier_type. Raises
    pydantic ValidationError on an unknown/missing classifier_type or bad fields."""
    if isinstance(data, (str, bytes)):
        return _response_adapter.validate_json(data)
    return _response_adapter.validate_python(data)
