# pyspacer-function

The pyspacer classifier inference function — the classifier **compute lane**. Validates a `PyspacerRequest`, lazily imports the
torch/pyspacer backend, runs EfficientNet extraction, predicts through one of
two serving formats (see below), and returns a `PyspacerResponse`. Deployed as
a Lambda container image (provisioning lives in mermaid-api's CDK,
mermaid-classifier issue #53).

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

## Serving formats

The image is built to serve exactly one artifact shape, selected by the
build-time-baked `CLASSIFIER_FORMAT` (default `graph`):

- **`graph`** — `efficientnet.pt` + `model.pt` + `model.json`, predicted
  through `mermaid-classifier`'s `load_predictor`. Gated by
  `check_compatibility()` against the manifest's `trained_with`.
- **`legacy`** — `efficientnet_weights.pt` + `classifier.pkl`, the Beta model
  (published as `v1`; not the model the team calls "V1", which is `v2`)
  loaded straight from its scikit-learn pickle via pyspacer's own
  `load_classifier`. Gated by `check_legacy_pins()` against `legacy_pins.txt`,
  since the pickle carries no manifest. Built from `Dockerfile.legacy`, which
  installs no `mermaid-classifier` at all.

## Model resolution

`classifier_version` selects the model; the deploy-pinned `CLASSIFIER_FORMAT`
selects which files that version's directory must hold. On Lambda, set
`CONFIG_BUCKET` and the function fetches `s3://$CONFIG_BUCKET/classifier/<version>/*`
(the graph triple or the legacy pair, per the format) into `/tmp/<version>/`
(warm-cached). Locally, set `LOCAL_MODELS_DIR` instead.

## Build the container image

From the repo root:

```bash
# graph (default) — needs an exact mermaid-classifier tag/sha
docker build -f packages/pyspacer-function/Dockerfile \
    --build-arg MERMAID_CLASSIFIER_REF=<classifier git tag> \
    --build-arg CLASSIFIER_VERSION=<vN> \
    -t pyspacer-inference:dev .

# legacy — installs no mermaid-classifier, so no MERMAID_CLASSIFIER_REF
docker build -f packages/pyspacer-function/Dockerfile.legacy \
    --build-arg CLASSIFIER_VERSION=<vN> \
    -t pyspacer-inference:beta-dev .
```

## Score-equivalence gate (legacy)

**Release precondition — contract version before image:** `PyspacerResponse`
forbids extra fields and the handler dumps every field including defaults, so
an image built from a tree that adds a response field puts that field on the
wire for every response, feature-vector requests or not. A consumer still
pinned to the older contract version rejects the unknown key, which fails
every classification through the image. Cut the contract git tag and land
mermaid-api's pin bump **before** pointing any such image at an environment —
nothing in code enforces that order, and, as with the ECR tag below, getting
it backwards is not something you can take back.

Before a legacy-format release burns a `vN-K` ECR tag, run
`scripts/beta_equivalence.py` — its module docstring carries the full runbook,
including the exact `docker run` invocations for its `baseline`, `candidate`,
and `compare` subcommands. Run it **before** `build-push.yml`: ECR tags are
immutable, so a bad image burns a version-build number and another release
cycle.

The gate is not a formality: the Beta cutover moves three numeric axes at
once — the mermaid-api ECS worker is x86_64/Python 3.13, the inference Lambda
is arm64/Python 3.12, and the Lambda sets 6 torch threads where the worker
takes the default — and any one of those can reorder a floating-point
reduction and change a score.

`legacy_pins.txt`'s own versions are numeric-stack-sensitive for the same
reason: they sit two minor versions behind the graph image on torch and
pyspacer so this lane mirrors the production API image. A pin there cannot be
bumped routinely, security patches included — changing one means
re-capturing the baseline from a running production API task and re-running
this gate, since a different numeric stack invalidates the measurement it is
built to make.
