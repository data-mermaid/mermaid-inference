# pyspacer-function

The pyspacer classifier inference function — the classifier **compute lane**
(see root `docs/adr/0001`). Validates a `PyspacerRequest`, lazily imports the
torch/pyspacer backend, runs EfficientNet extraction + the portable TorchScript
head, and returns a `PyspacerResponse`. Deployed as a Lambda container image
(provisioning lives in mermaid-api's CDK, issue #53).

## Run the tests

```bash
uv run pytest packages/pyspacer-function/tests/ -v
```

## Run the local-dev HTTP harness

The harness serves the same `handler()` over HTTP. Point it at a local model
directory laid out as `<LOCAL_MODELS_DIR>/<version>/{efficientnet.pt,model.pt,model.json}`:

```bash
export LOCAL_MODELS_DIR=/path/to/models
uv run uvicorn pyspacer_function.app:app --reload

curl -s localhost:8000/classify -H 'content-type: application/json' -d '{
  "classifier_type": "pyspacer",
  "classifier_version": "v1",
  "image": {"bucket": "my-bucket", "key": "img.jpg"},
  "points": [[100, 100], [200, 200]]
}'
```

> The image `S3Location` is read via boto3's default credential chain. To run
> fully offline, exercise the storage-agnostic `classify()` core directly with a
> `filesystem`/`memory` `DataLocation` (see `tests/test_classify.py`).

## Model resolution

`classifier_version` alone selects the model. On Lambda, set `CONFIG_BUCKET` and
the function fetches `s3://$CONFIG_BUCKET/classifier/<version>/*` into
`/tmp/<version>/` (warm-cached). Locally, set `LOCAL_MODELS_DIR` instead.

## Build the container image

From the repo root (set the `mermaid-classifier` tag in the Dockerfile first):

```bash
docker build -f packages/pyspacer-function/Dockerfile -t pyspacer-inference:dev .
```
