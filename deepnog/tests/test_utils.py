from deepnog.tests.utils import get_deepnog_root


def test_get_deepnog_root():
    deepnog_root = get_deepnog_root()
    subpackages = ['client', 'config', 'data', 'learning',
                   'models', 'tests', 'utils',
                   ]
    for pkg in subpackages:
        d = deepnog_root/pkg
        assert d.is_dir()

    some_files = ["tests/data/test_inference_short.csv",
                  "tests/parameters/test_deepencoding.pthsmall",
                  "config/deepnog_config.yml",
                  ]
    for file_ in some_files:
        f = deepnog_root/file_
        assert f.is_file()

    unexpected_files = ["me/no/think.so",
                        "plagiarism/manuscript.tex",
                        "bugs",
                        ]
    for file_ in unexpected_files:
        f = deepnog_root/file_
        assert not f.exists()
