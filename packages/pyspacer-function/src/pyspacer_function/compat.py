"""Fail-loud check that the serving runtime matches the libraries a model was
built with. A calibrated classifier's probabilities only reproduce under the
same torch / scikit-learn / pyspacer; a mismatch silently mis-scores, so we
refuse to serve it. The graph format matches its model.json manifest; the
legacy format, which has no manifest, matches legacy_pins.txt instead."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

_KEYS = ("torch", "sklearn", "pyspacer")
_LEGACY_PINS = Path(__file__).with_name("legacy_pins.txt")
_INCOMPATIBLE = "Inference image is incompatible with the deployed model artifact — "


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
        raise RuntimeError(_INCOMPATIBLE + "; ".join(mismatches))


def _parse_pins(pins_path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in pins_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise RuntimeError(f"{pins_path}: expected 'package==version', got {raw!r}")
        name, _, want = line.partition("==")
        pins[name.strip()] = want.strip()
    return pins


def check_legacy_pins(pins_path: Path | None = None) -> None:
    """Gate the legacy format on legacy_pins.txt, the same file the image
    installs from — the Beta pickle carries no manifest to match against."""
    path = Path(pins_path) if pins_path is not None else _LEGACY_PINS
    mismatches = []
    for name, want in _parse_pins(path).items():
        try:
            have = _pkg_version(name)
        except PackageNotFoundError:
            mismatches.append(f"{name}: pinned at {want}, not installed")
            continue
        if have != want:
            mismatches.append(f"{name}: pinned at {want}, runtime has {have}")
    if mismatches:
        raise RuntimeError(_INCOMPATIBLE + "; ".join(mismatches))
