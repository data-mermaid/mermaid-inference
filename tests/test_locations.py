import pytest
from pydantic import ValidationError

from mermaid_inference_contract.locations import S3Location


def test_s3location_has_only_bucket_and_key():
    loc = S3Location(bucket="b", key="k")
    assert loc.bucket == "b"
    assert loc.key == "k"
    assert set(loc.model_dump().keys()) == {"bucket", "key"}


def test_s3location_roundtrips_through_json():
    loc = S3Location(bucket="b", key="k")
    again = S3Location.model_validate_json(loc.model_dump_json())
    assert again == loc


def test_s3location_rejects_credential_fields():
    with pytest.raises(ValidationError):
        S3Location(bucket="b", key="k", access_key="AK")
