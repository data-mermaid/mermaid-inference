"""Fail-loud check that the serving runtime matches the libraries a model was
built with. A model.pt (TorchScript head) + its calibrated probabilities only
reproduce under the same torch / scikit-learn / pyspacer; a mismatch silently
mis-scores, so we refuse to serve it."""
from __future__ import annotations

import json
from importlib.metadata import version as _pkg_version
from pathlib import Path

_KEYS = ("torch", "sklearn", "pyspacer")


def _runtime() -> dict[str, str]:
    import torch  # already imported by the handler before this runs

    return {
        "torch": torch.__version__,
        "sklearn": _pkg_version("scikit-learn"),
        "pyspacer": _pkg_version("pyspacer"),
    }


def check_compatibility(model_json_path: Path) -> None:
    manifest = json.loads(Path(model_json_path).read_text())
    trained_with = manifest.get("trained_with") or {}
    runtime = _runtime()
    mismatches = []
    for key in _KEYS:
        want = trained_with.get(key)
        have = runtime[key]
        if want is None:
            mismatches.append(f"{key}: manifest missing (runtime {have})")
        elif want != have:
            mismatches.append(f"{key}: model built with {want}, runtime has {have}")
    if mismatches:
        raise RuntimeError(
            "Inference image is incompatible with the deployed model artifact — "
            + "; ".join(mismatches)
        )
