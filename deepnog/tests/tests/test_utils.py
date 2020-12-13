from deepnog.tests.utils import get_deepnog_root


def test_get_deepnog_root():
    deepnog_root = get_deepnog_root()

    # Ensure this is the correct directory will all the subpackages
    subpackages = ['client', 'config', 'data', 'learning',
                   'models', 'tests', 'utils',
                   ]
    for pkg in subpackages:
        d = deepnog_root/pkg
        assert d.is_dir()

    # Also check presence of some files
    some_files = ["tests/data/test_inference_short.csv",
                  "tests/parameters/test_deepnog.pthsmall",
                  "config/deepnog_config.yml",
                  ]
    for file_ in some_files:
        f = deepnog_root/file_
        assert f.is_file()

    # Ensure certain things are not present in deepnog
    unexpected_files = ["me/no/think.so",
                        "plagiarism/manuscript.tex",
                        "bugs",
                        ]
    for file_ in unexpected_files:
        f = deepnog_root/file_
        assert not f.exists()
