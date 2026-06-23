from mermaid_inference_contract.locations import S3Location


def test_s3location_minimal_defaults_creds_none():
    loc = S3Location(bucket="b", key="k")
    assert loc.bucket == "b"
    assert loc.key == "k"
    assert loc.access_key is None
    assert loc.secret_key is None
    assert loc.session_token is None


def test_s3location_roundtrips_through_json():
    loc = S3Location(bucket="b", key="k", access_key="AK", secret_key="SK", session_token="ST")
    again = S3Location.model_validate_json(loc.model_dump_json())
    assert again == loc
