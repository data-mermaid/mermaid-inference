"""Environment-driven settings for the pyspacer inference function."""
import os


def config_bucket() -> str | None:
    """The same-account S3 bucket holding classifier/<version>/ artifacts.
    Set on Lambda; unset locally."""
    return os.environ.get("CONFIG_BUCKET") or None


def local_models_dir() -> str | None:
    """A local directory holding <version>/ model dirs, for dev/test only."""
    return os.environ.get("LOCAL_MODELS_DIR") or None


def num_threads() -> int:
    """Torch CPU threads: the configured vCPU count, else detected cores."""
    return int(os.environ.get("INFERENCE_NUM_THREADS") or 0) or (os.cpu_count() or 1)


def image_version() -> str:
    """Function image version baked at build time (mermaid-inference semver).

    Logged on cold start so any CloudWatch log line is traceable to a build.
    """
    return os.environ.get("INFERENCE_IMAGE_VERSION") or "unknown"


def classifier_version() -> str:
    """The model version this function is deployed to serve (baked into the
    image at build as CLASSIFIER_VERSION). Required — the function resolves
    classifier/<version>/ from S3 and cannot run without it."""
    version = os.environ.get("CLASSIFIER_VERSION")
    if not version:
        raise RuntimeError(
            "CLASSIFIER_VERSION is not set: the image must bake the model "
            "version it serves (build arg CLASSIFIER_VERSION)."
        )
    return version
