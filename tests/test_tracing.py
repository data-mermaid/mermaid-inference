import pytest

from mermaid_inference_contract.tracing import (
    Traceparent,
    format_traceparent,
    new_traceparent,
    parse_traceparent,
)

VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_parse_then_format_roundtrips():
    tp = parse_traceparent(VALID)
    assert tp.version == "00"
    assert tp.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert tp.parent_id == "00f067aa0ba902b7"
    assert tp.flags == "01"
    assert format_traceparent(tp) == VALID


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not-a-traceparent",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",  # missing flags
        "00-XYZ-00f067aa0ba902b7-01",  # non-hex trace_id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",  # 1-char flags
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # all-zero trace_id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # all-zero parent_id
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_traceparent(bad)


def test_new_traceparent_is_parseable_and_unique():
    a = new_traceparent()
    b = new_traceparent()
    assert a.flags == "01"
    assert new_traceparent(sampled=False).flags == "00"
    # round-trips through string form
    assert parse_traceparent(format_traceparent(a)) == a
    # randomness: two fresh ids differ
    assert a.trace_id != b.trace_id


def test_traceparent_model_roundtrips_json():
    tp = parse_traceparent(VALID)
    assert Traceparent.model_validate_json(tp.model_dump_json()) == tp
