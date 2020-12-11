import builtins
from importlib import import_module
import pytest


@pytest.fixture
def hide_available_torch(monkeypatch):
    """ Pretend PyTorch was not installed. """
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError("Monkeypatched import hiding torch")
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


@pytest.mark.parametrize("pkg_module", [
    ["deepnog", ".data"],
    ["deepnog.data", ".dataset"],
    ["deepnog.data", "ProteinDataset"],
    ["deepnog.data.dataset", "ProteinIterableDataset"],
    ["deepnog", ".learning"],
    ["deepnog.learning", ".inference"],
    ["deepnog.learning", ".training"],
    ["deepnog.learning", "fit"],
    ["deepnog.learning", "predict"],
    ["deepnog", ".models"],
    ["deepnog.models", ".deepfam"],
    ["deepnog.models.deepfam", "DeepFam"],
    ["deepnog.models", ".deepnog"],
    ["deepnog.models.deepnog", "DeepNOG"],
    ]
)
@pytest.mark.hide_torch  # needs pytest --hide-torch option to run
def test_missing_torch_error_message(hide_available_torch, pkg_module):
    expected_msg = "conda install pytorch -c pytorch"
    pkg, module = pkg_module
    with pytest.raises(ImportError, match=expected_msg):
        try:
            # import a module from (sub)package
            import_module(name=module, package=pkg)
        except ModuleNotFoundError:
            # import a class/attribute from module
            getattr(import_module(pkg), module)
