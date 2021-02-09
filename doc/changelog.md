# Changelog

## [Next release]
...


## [1.2.3] - 2021-02-09

### Added in 1.2.3
- Add citation for paper published in *Bioinformatics* ([doi](https://doi.org/10.1093/bioinformatics/btaa1051))
### Changes in 1.2.3
- CI with Github Actions (Linux, macOS)

### Fixes in 1.2.3
- Fixes a bug where custom trained models would not use dropout correctly
  [see #44](https://github.com/univieCUBE/deepnog/issues/44)
- Fixes data usage in training with an iterable dataset without shuffling
  [see #43](https://github.com/univieCUBE/deepnog/pull/43)
- Fixes several non-critical warnings
  [see #50](https://github.com/univieCUBE/deepnog/pull/50)
- Several small fixes regarding updated libraries etc.


## [1.2.2] - 2020-12-10

### Added in 1.2.2
- Install from bioconda
- Support for 109 taxonomic levels in eggNOG 5 (was three before)
  (e.g. `deepnog infer -db eggnog5 -t 1239` for Firmicutes)
- Support for COG2020 (use `deepnog infer -db cog2020 -t 1`)

### Fixes/changes in 1.2.2
- Requirement PyYAML
- Test class imports
- Exit on requesting unavailable device (instead of raising an error) 

## [1.2.1] - 2020-08-28

### Added in 1.2.1
- Training custom models: Users can now train additional models for further
  tax. levels of eggNOG 5 or even different orthology databases
- TensorBoard status reports: Follow training/validation loss online
- Support for configuration file (``deepnog_config.yml``)
- Model quality assessment

### Changed in 1.2.1
- The command line invocation now uses two subcommands:
  * ``deepnog train`` for training new models, and
  * ``deepnog infer`` for general orthologous group assignment
    (and model quality assessment)

### Fixed in 1.2.1
- Fixed packaging issue in 1.2.0 (which was subsequently removed altogether)
- Several additional bug fixes and smaller changes


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
  - [Travis](https://travis-ci.com/univieCUBE/deepnog/) (Linux, MacOS)
  - [AppVeyor](https://ci.appveyor.com/project/VarIr/deepnog) (Windows)
- [Codecov](https://codecov.io/gh/univieCUBE/deepnog/) coverage reports
- [LGTM](https://lgtm.com/projects/g/univieCUBE/deepnog) code quality/security reports
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

[Next release]: https://github.com/univieCUBE/deepnog/compare/v1.2.3...HEAD
[1.2.3]: https://github.com/univieCUBE/deepnog/releases/tag/v1.2.3
[1.2.2]: https://github.com/univieCUBE/deepnog/releases/tag/v1.2.2
[1.2.1]: https://github.com/univieCUBE/deepnog/releases/tag/v1.2.1
[1.1.0]: https://github.com/univieCUBE/deepnog/releases/tag/v1.1.0
[1.0.0]: https://github.com/univieCUBE/deepnog/releases/tag/v1.0.0final

[//]: # "Sections: Added, Fixed, Changed, Removed"
