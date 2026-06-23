from pydantic import BaseModel, ConfigDict


class S3Location(BaseModel):
    """An S3-addressed reference to a payload (image, feature vector, model
    files). The contract passes locations, not bytes. Credentials are NOT part
    of the contract: inference uses its execution role's default boto3
    credential chain (cross-account access is an IAM concern)."""

    model_config = ConfigDict(extra="forbid")

    bucket: str
    key: str
