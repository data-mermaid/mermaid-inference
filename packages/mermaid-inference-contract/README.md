# mermaid-inference-contract

The pure-pydantic request/response schema shared by the MERMAID API worker and ML
inference — a discriminated union on `classifier_type` (the `pyspacer` member only;
a `segmentation` member is reserved), plus error-envelope and `traceparent` helpers.
No torch, no boto3. `S3Location` carries only `{bucket, key}`; credentials are an
IAM concern resolved by the caller's default boto3 credential chain.

Part of the [`mermaid-inference`](../../README.md) workspace.

## Install (pinned by git tag)

```bash
pip install "git+https://github.com/data-mermaid/mermaid-inference@<tag>#subdirectory=packages/mermaid-inference-contract"
```

## Usage

```python
from mermaid_inference_contract import parse_classify_request, PyspacerResponse

req = parse_classify_request(event_body)          # validates + routes on classifier_type
resp = PyspacerResponse(
    classifier_type="pyspacer",
    classifier_version=req.classifier_version,
    point_results=[...],
    valid_rowcol=True,
)
```

## Tests

```bash
uv run pytest packages/mermaid-inference-contract/tests/
```
