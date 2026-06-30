import json
from pathlib import Path

import pytest

from pyspacer_function import compat


def _manifest(tmp_path, trained_with):
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"schema_version": 1, "trained_with": trained_with}))
    return p


def _runtime_versions():
    from importlib.metadata import version
    import torch

    return {"torch": torch.__version__, "sklearn": version("scikit-learn"), "pyspacer": version("pyspacer")}


def test_compatible_manifest_passes(tmp_path):
    compat.check_compatibility(_manifest(tmp_path, _runtime_versions()))  # no raise


def test_mismatched_pyspacer_raises(tmp_path):
    rv = _runtime_versions()
    rv["pyspacer"] = "0.0.1"
    with pytest.raises(RuntimeError, match="pyspacer"):
        compat.check_compatibility(_manifest(tmp_path, rv))


def test_missing_trained_with_raises(tmp_path):
    with pytest.raises(RuntimeError):
        compat.check_compatibility(_manifest(tmp_path, {}))
