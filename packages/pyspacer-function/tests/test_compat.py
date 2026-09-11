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


def _pins_file(tmp_path: Path, pins: dict[str, str]) -> Path:
    p = tmp_path / "legacy_pins.txt"
    body = "\n".join(f"{name}=={want}" for name, want in pins.items())
    p.write_text(f"# placeholder pins\n{body}\n")
    return p


def _installed(*names: str) -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in names}


def test_matching_legacy_pins_pass(tmp_path):
    pins = _installed("pyspacer", "scikit-learn", "torch", "numpy", "Pillow")
    compat.check_legacy_pins(_pins_file(tmp_path, pins))  # no raise


def test_mismatched_legacy_pin_raises_naming_the_package(tmp_path):
    pins = _installed("pyspacer", "scikit-learn")
    pins["scikit-learn"] = "0.0.1"
    with pytest.raises(RuntimeError, match="scikit-learn"):
        compat.check_legacy_pins(_pins_file(tmp_path, pins))


def test_legacy_pins_name_every_offender(tmp_path):
    pins = _installed("pyspacer", "numpy", "Pillow")
    pins["pyspacer"] = "0.0.1"
    pins["numpy"] = "0.0.2"
    with pytest.raises(RuntimeError) as exc:
        compat.check_legacy_pins(_pins_file(tmp_path, pins))
    assert "pyspacer" in str(exc.value)
    assert "numpy" in str(exc.value)


def test_legacy_pin_on_an_uninstalled_package_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not-a-real-package"):
        compat.check_legacy_pins(_pins_file(tmp_path, {"not-a-real-package": "1.0"}))


def test_check_legacy_pins_default_path_resolves_and_parses_the_shipped_file():
    # Whether the pinned versions match the runtime depends on which image runs
    # the suite (only the legacy image's install matches); resolution and
    # parsing of the default path hold in every environment.
    assert compat._LEGACY_PINS.exists()
    pins = compat._parse_pins(compat._LEGACY_PINS)
    assert pins  # the shipped file pins at least one package
