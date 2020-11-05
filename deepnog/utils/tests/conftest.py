import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hide-torch",
        action='store_true',
        default=False,
        help="Run tests hiding torch."
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "hide_torch: mark test as hiding torch")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hide-torch"):
        # --hide-torch given in cli: do not skip the tests mocking torch
        return
    skip_torch = pytest.mark.skip(reason="needs --hide-torch option to run")
    for item in items:
        if "hide_torch" in item.keywords:
            item.add_marker(skip_torch)
