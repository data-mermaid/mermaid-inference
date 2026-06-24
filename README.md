# mermaid-inference

The contract between the MERMAID API and ML inference — a pure-pydantic
request/response schema plus the inference functions that honour it. The
framework is the contract, not the compute: the schema is durable and
independent of where inference code runs (see the root
`docs/adr/0001-per-model-compute-lanes-via-lambda.md`).

The repo is a uv workspace: a virtual umbrella root with the shipped packages
under `packages/`.

| Package | What it is | Deps |
|---|---|---|
| `packages/mermaid-inference-contract` | The request/response schema + helpers. Pinned into mermaid-api by git tag + subdirectory. | pydantic only |
| `packages/pyspacer-function` | The pyspacer classifier **compute lane** — a Lambda container image, plus a FastAPI local-dev harness. | torch / pyspacer (via `mermaid-classifier[inference]`) |

## `mermaid-inference-contract`

Pure-pydantic request/response schema (a discriminated union on
`classifier_type`, `pyspacer` member only) plus error-envelope and
`traceparent` helpers. No torch, no boto3. `S3Location` carries only
`{bucket, key}` — credentials are an IAM concern resolved by the caller's
default boto3 credential chain, not part of the contract.

### Install (pinned by git tag)

```bash
pip install "git+https://github.com/data-mermaid/mermaid-inference@<tag>#subdirectory=packages/mermaid-inference-contract"
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

## `pyspacer-function`

The pyspacer classifier compute lane. The handler validates a `PyspacerRequest`,
**lazily** imports the torch/pyspacer backend (keeping the multi-second import
out of Lambda's INIT phase), runs pyspacer EfficientNet extraction, loads the
version's `model.pt` head via the shared `load_predictor()`, runs
`predict_proba`, and returns a `PyspacerResponse`. Extractor weights +
`model.pt`/`model.json` are resolved from `classifier_version` alone and cached
in `/tmp` keyed by version. A thin FastAPI wrapper exposes the same handler for
local-dev and manual checks.

It depends on `mermaid-classifier[inference]` (the portable-artifact loader) and
pyspacer (feature extraction), and is deployed as a Lambda container image
(provisioning lives in mermaid-api's CDK). See `packages/pyspacer-function/` for
the handler, the model resolver, the Dockerfile, and run instructions.

## Development

```bash
uv sync                                                  # resolve the workspace
uv run pytest packages/mermaid-inference-contract/tests/ # contract tests (pydantic-only)
uv run pytest packages/pyspacer-function/tests/          # inference-function tests (torch)
```

Run the two suites as **separate invocations**, not in one `pytest` process: the
function pulls torch, and torch cannot initialise twice in a single process, so
collecting both trees at once fails by design. Bare `uv run pytest` at the repo
root is scoped to the contract tests.

## CI / image publishing

The `.github/workflows/build-push.yml` workflow builds and pushes the `pyspacer-function` Lambda image to ECR (`mermaid-inference-pyspacer`) on push to `main` and on version tags (`v*`). It requires two GitHub repo configs:

- **Variable** `MERMAID_CLASSIFIER_REF`: the git tag of `mermaid-classifier` that the image pins (passed to the Dockerfile as `MERMAID_CLASSIFIER_REF` build arg).
- **Secret** `AWS_ACCOUNT_ID`: the AWS account ID for OIDC role assumption (assumes `mermaid-inference-image-push-role`).

The workflow pushes immutable semver (`:0.2.0`) and SHA (`:abc1234`) tags to ECR. To release a new image, bump the `version` field in the root `pyproject.toml`.
