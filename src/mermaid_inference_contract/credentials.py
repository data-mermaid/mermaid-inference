from mermaid_inference_contract.locations import S3Location


def resolve_credentials(loc: S3Location) -> dict | None:
    """Decide which credentials a caller should pass to its own boto3 client.

    Returns boto3 credential kwargs when the location carries an explicit
    access/secret pair (cross-account case), else None to signal 'use the
    ambient IAM role'. This function performs no AWS calls and imports no boto3.
    """
    if loc.access_key and loc.secret_key:
        creds = {
            "aws_access_key_id": loc.access_key,
            "aws_secret_access_key": loc.secret_key,
        }
        if loc.session_token:
            creds["aws_session_token"] = loc.session_token
        return creds
    return None
