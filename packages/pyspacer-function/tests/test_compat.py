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


def test_malformed_pin_line_raises_naming_the_line(tmp_path):
    p = tmp_path / "legacy_pins.txt"
    p.write_text("# placeholder pins\npyspacer==1.0\nnot-a-valid-line\n")
    with pytest.raises(RuntimeError, match="not-a-valid-line"):
        compat._parse_pins(p)


def test_check_legacy_pins_default_path_resolves_and_parses_the_shipped_file():
    # Whether the pinned versions match the runtime depends on which image runs
    # the suite (only the legacy image's install matches); resolution and
    # parsing of the default path hold in every environment.
    assert compat._LEGACY_PINS.exists()
    pins = compat._parse_pins(compat._LEGACY_PINS)
    assert pins  # the shipped file pins at least one package


# ---- PEP 440 local segments ------------------------------------------------
#
# Training builds linux/amd64 and takes torch from PyTorch's CPU wheel index,
# which stamps +cpu. This image builds linux/arm64, where PyPI's aarch64 wheels
# are already CPU-only and carry no local segment. Comparing the two as strings
# refuses every SageMaker-trained artifact.


def test_a_local_segment_is_not_a_mismatch(tmp_path):
    rv = _runtime_versions()
    rv["torch"] = rv["torch"].split("+", 1)[0] + "+cpu"
    compat.check_compatibility(_manifest(tmp_path, rv))  # no raise


def test_the_released_v2_provenance_shape_passes(tmp_path):
    # v2's released model.json records torch "2.8.0+cpu"; once the pyspacer key
    # the release gate now requires is present, that artifact has to serve on
    # the arm64 runtime rather than be refused for its wheel tag.
    rv = _runtime_versions()
    rv["torch"] = "2.8.0+cpu"
    if rv["torch"].split("+", 1)[0] != _runtime_versions()["torch"].split("+", 1)[0]:
        pytest.skip("runtime torch is no longer 2.8.x; the v2 shape no longer applies")
    compat.check_compatibility(_manifest(tmp_path, rv))  # no raise


def test_a_different_public_version_still_raises(tmp_path):
    # Only the local segment is ignored; this must not have become "anything
    # goes".
    rv = _runtime_versions()
    rv["torch"] = "1.0.0+cpu"
    with pytest.raises(RuntimeError, match="torch"):
        compat.check_compatibility(_manifest(tmp_path, rv))


def test_legacy_pins_ignore_a_local_segment_too(tmp_path):
    pins = _installed("scikit-learn")
    pins["scikit-learn"] = pins["scikit-learn"] + "+cpu"
    compat.check_legacy_pins(_pins_file(tmp_path, pins))  # no raise
