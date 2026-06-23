# arm64 (Graviton) builds natively on Apple-Silicon dev machines; torch's
# aarch64 wheels are CPU-only. For x86_64: change the platform to
# linux/amd64, set Architectures: [x86_64] in the CDK function (issue #53),
# and install torch from the CPU index.
FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.12

# Heavy, stable layers first -> lazy container loading pulls fewer bytes on cold start.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0

# mermaid-classifier[inference] -> pyspacer==0.14.0 (scikit-learn 1.5.2, Pillow,
# numpy, boto3) + the load_predictor graph loader. Pinned by git tag; bump the
# tag here and rebuild when shipping a new classifier/library version.
RUN pip install --no-cache-dir \
    "mermaid-classifier[inference] @ git+https://github.com/data-mermaid/mermaid-classifier@<TAG>"

# FastAPI/uvicorn are only used by the local-dev harness, but cheap to include.
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29"

# Contract (repo root) + function package, installed from the build context.
# --no-deps: torch/pyspacer/fastapi are already present from the layers above.
COPY . /build
RUN pip install --no-cache-dir --no-deps /build \
    && pip install --no-cache-dir --no-deps /build/pyspacer-function

ENV SPACER_EXTRACTORS_CACHE_DIR=/tmp \
    TORCH_HOME=/tmp \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp

CMD ["pyspacer_function.handler.handler"]
