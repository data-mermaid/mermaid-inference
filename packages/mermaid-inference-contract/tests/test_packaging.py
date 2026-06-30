import importlib
import sys


def test_package_imports():
    mod = importlib.import_module("mermaid_inference_contract")
    assert mod.__version__ == "0.3.0"


def test_no_heavy_deps_imported():
    """Importing the contract must not pull torch or boto3 into the process."""
    for name in ("torch", "boto3"):
        sys.modules.pop(name, None)
    importlib.reload(importlib.import_module("mermaid_inference_contract"))
    assert "torch" not in sys.modules
    assert "boto3" not in sys.modules
