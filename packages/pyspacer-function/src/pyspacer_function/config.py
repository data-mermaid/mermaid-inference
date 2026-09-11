"""Environment-driven settings for the pyspacer inference function."""
import os

_FORMATS = ("graph", "legacy")
_DEFAULT_FORMAT = "graph"


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


def classifier_format() -> str:
    """The artifact shape this function is deployed to serve (baked into the
    image at build as CLASSIFIER_FORMAT): "graph" for the efficientnet.pt /
    model.pt / model.json triple, "legacy" for a pickled scikit-learn
    classifier. Unset means "graph" — images built before the legacy format
    existed set nothing and must keep serving unchanged."""
    fmt = os.environ.get("CLASSIFIER_FORMAT") or _DEFAULT_FORMAT
    if fmt not in _FORMATS:
        raise RuntimeError(
            f"CLASSIFIER_FORMAT={fmt!r} is not a recognised serving format: "
            f"expected one of {', '.join(_FORMATS)}."
        )
    return fmt
