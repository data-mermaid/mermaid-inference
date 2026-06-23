# mermaid-inference

The contract between the MERMAID API and ML inference.

## mermaid-inference-contract

Pure-pydantic request/response schema (discriminated union on `classifier_type`,
pyspacer member only) plus credential/error/traceparent helpers. No torch, no boto3.

### Install (pinned by git tag)

```bash
pip install "git+https://github.com/data-mermaid/mermaid-inference@v0.1.0"
```

### Usage

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
