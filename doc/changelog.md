# Changelog

## [Next release]
...


## [1.1.0] - 2020-02-28

### Added
- EggNOG5 root (tax 1) prediction

### Changed
- Package structure changed for higher modularity. This will require changes
  in downstream usages.
- Remove network weights from the repository, because files are too large for
  github and/or PyPI. `deepnog` automatically downloads these from
  [CUBE](https://cube.univie.ac.at) servers, and caches them locally.
- More robust inter-process communication in data loading

### Fixes
- Fix error on very short amino acid sequences
- Fix error on unrecognized symbols in sequences (stop codons etc.)
- Fix multiprocess data loading from gzipped files
- Fix type mismatch in deepencoding embedding layer (Windows only)

### Maintenance
- Continuous integration on
  - [Travis](https://travis-ci.com/VarIr/deepnog/) (Linux, MacOS)
  - [AppVeyor](https://ci.appveyor.com/project/VarIr/deepnog) (Windows)
- [Codecov](https://codecov.io/gh/VarIr/deepnog/) coverage reports
- [LGTM](https://lgtm.com/projects/g/VarIr/deepnog) code quality/security reports
- Documentation on [ReadTheDocs](https://deepnog.readthedocs.io)
- Upload to [PyPI](https://pypi.org/project/deepnog/), thus enabling
  `$ pip install deepnog`.


## [1.0.0] - 2019-10-18

The first release of `deepnog` to appear in this changelog.
It already contains the following features:

- EggNOG5 bacteria (tax 2) prediction
- DeepEncoding architecture
- CPU and GPU support
- Runs on all major platforms (Linux, MacOS, Windows)

[Next release]: https://github.com/VarIr/deepnog/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/VarIr/deepnog/releases/tag/v1.1.0
[1.0.0]: https://github.com/VarIr/deepnog/releases/tag/v1.0.0final

[//]: # "Sections: Added, Fixed, Changed, Removed"
