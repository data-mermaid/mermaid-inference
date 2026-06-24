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
