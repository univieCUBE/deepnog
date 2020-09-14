import builtins
from importlib import import_module
import pytest
import sys


@pytest.fixture
def hide_available_torch(monkeypatch):
    """ Pretend PyTorch was not installed. """
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if 'torch' in name:
            raise ImportError("Monkeypatched import hiding torch")
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


@pytest.mark.parametrize("pkg_module", [
    ["deepnog", ".data"],
    ["deepnog.data", ".dataset"],
    ["deepnog", ".learning"],
    ["deepnog.learning", ".inference"],
    ["deepnog.learning", ".training"],
    ["deepnog", ".models"],
    ["deepnog.models", ".deepencoding"],
    ["deepnog.models", ".deepfam"],
    ["deepnog.models", ".deepnog"],
    ]
)
def test_missing_torch_error_message(hide_available_torch, pkg_module):
    expected_msg = "conda install pytorch -c pytorch"
    pkg, module = pkg_module
    with pytest.raises(ImportError, match=expected_msg):
        import_module(name=module, package=pkg)
