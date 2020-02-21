# Changelog

## [Next release]

### Added
- EggNOG5 root (tax 1) prediction

### Changed
- Package structure changed for higher modularity. This will require changes
  in downstream usages.
- Remove network weights from the repository, because files are too large for
  github and/or PyPI. `deepnog` automatically downloads these from
  [CUBE](https://cube.univie.ac.at) servers, and caches them locally.

### Fixes
- Fix error on very short amino acid sequences
- Fix error on unrecognized symbols in sequences (stop codons etc.)

### Maintenance
- Continuous integration on
  - Travis (Linux, MacOS)
  - AppVeyor (Windows)
- Codecov coverage reports
- LGTM code quality/security reports
- Documentation on readthedocs


## [1.0.0] - 2019-10-18

The first release of `deepnog` to appear in this changelog.
It already contains the following features:

- EggNOG5 bacteria (tax 2) prediction
- DeepEncoding architecture
- CPU and GPU support
- Runs on all major platforms (Linux, MacOS, Windows)

[Next release]: https://github.com/VarIr/deepnog/compare/v1.0.0final...HEAD
[1.0.0]:   https://github.com/VarIr/deepnog/releases/tag/v1.0.0final

[//]: # "Sections: Added, Fixed, Changed, Removed"
