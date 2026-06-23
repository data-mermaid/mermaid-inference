from mermaid_inference_contract.credentials import resolve_credentials
from mermaid_inference_contract.locations import S3Location


def test_returns_none_when_no_explicit_creds():
    assert resolve_credentials(S3Location(bucket="b", key="k")) is None


def test_returns_kwargs_without_session_token():
    loc = S3Location(bucket="b", key="k", access_key="AK", secret_key="SK")
    assert resolve_credentials(loc) == {
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
    }


def test_includes_session_token_when_present():
    loc = S3Location(bucket="b", key="k", access_key="AK", secret_key="SK", session_token="ST")
    assert resolve_credentials(loc) == {
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
        "aws_session_token": "ST",
    }


def test_partial_creds_treated_as_none():
    # access_key without secret_key is not usable -> ambient role
    assert resolve_credentials(S3Location(bucket="b", key="k", access_key="AK")) is None
