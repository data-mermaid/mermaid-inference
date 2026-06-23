from pydantic import BaseModel


class S3Location(BaseModel):
    """An S3-addressed reference to a payload (image, feature vector, model files).
    The contract passes locations, not bytes. Omit the credential fields to use
    the caller's ambient IAM role."""

    bucket: str
    key: str
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None  # for STS temporary credentials
