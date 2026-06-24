import re
import secrets

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

_HEX = re.compile(r"\A[0-9a-f]+\Z")


def _check_hex(value: str, length: int, name: str, *, nonzero: bool) -> str:
    if len(value) != length or not _HEX.match(value):
        raise ValueError(f"{name} must be {length} lowercase hex chars, got {value!r}")
    if nonzero and value == "0" * length:
        raise ValueError(f"{name} must not be all zeroes")
    return value


class Traceparent(BaseModel):
    """W3C trace-context id: version-trace_id-parent_id-flags."""

    model_config = ConfigDict(extra="forbid")

    version: str = "00"
    trace_id: str
    parent_id: str
    flags: str = "01"

    @field_validator("version", "flags")
    @classmethod
    def _two_hex(cls, v: str, info: ValidationInfo) -> str:
        return _check_hex(v, 2, info.field_name, nonzero=False)

    @field_validator("trace_id")
    @classmethod
    def _trace_id_hex(cls, v: str) -> str:
        return _check_hex(v, 32, "trace_id", nonzero=True)

    @field_validator("parent_id")
    @classmethod
    def _parent_id_hex(cls, v: str) -> str:
        return _check_hex(v, 16, "parent_id", nonzero=True)


def parse_traceparent(s: str) -> Traceparent:
    parts = s.split("-")
    if len(parts) != 4:
        raise ValueError(f"traceparent must have 4 dash-separated fields, got {s!r}")
    version, trace_id, parent_id, flags = parts
    return Traceparent(version=version, trace_id=trace_id, parent_id=parent_id, flags=flags)


def format_traceparent(tp: Traceparent) -> str:
    return f"{tp.version}-{tp.trace_id}-{tp.parent_id}-{tp.flags}"


def new_traceparent(sampled: bool = True) -> Traceparent:
    return Traceparent(
        version="00",
        trace_id=secrets.token_hex(16),
        parent_id=secrets.token_hex(8),
        flags="01" if sampled else "00",
    )
