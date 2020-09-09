[![Linux/MacOS builds on Travis](
  https://travis-ci.com/univieCUBE/deepnog.svg?branch=master)](
  https://travis-ci.com/univieCUBE/deepnog)
[![Windows builds on AppVeyor](
  https://ci.appveyor.com/api/projects/status/ccdysyv0o2gey6iu/branch/master?svg=true)](
  https://ci.appveyor.com/project/VarIr/deepnog/branch/master)
[![codecov](
  https://codecov.io/gh/univieCUBE/deepnog/branch/master/graph/badge.svg)](
  https://codecov.io/gh/univieCUBE/deepnog)
[![Language grade: Python](
  https://img.shields.io/lgtm/grade/python/g/univieCUBE/deepnog.svg?logo=lgtm&logoWidth=18)](
  https://lgtm.com/projects/g/univieCUBE/deepnog/context:python)
[![Documentation Status](
  https://readthedocs.org/projects/deepnog/badge/?version=latest)](
  https://deepnog.readthedocs.io/en/latest/?badge=latest)


# DeepNOG: protein orthologous groups assignment

Assign proteins to orthologous groups (eggNOG 5) on CPUs or GPUs with deep networks.
DeepNOG is much faster than alignment-based methods,
while being as accurate as HMMER.

The `deepnog` command line tool is written in Python 3.7+. 


## Installation guide

The easiest way to install DeepNOG is to obtain it from PyPI:

```pip install deepnog```

Alternatively, you can clone or download bleeding edge versions
from GitHub and run

```pip install /path/to/DeepNOG```

If you plan to extend DeepNOG as a developer, run

```pip install -e /path/to/DeepNOG```

instead.

## Usage

DeepNOG can be used through calling the above installed `deepnog`
command with a protein sequence file (FASTA). 

Example usages: 

*  `deepnog infer proteins.faa`
    * Predicted groups of proteins in proteins.faa will be written to the console.
      By default, eggNOG5 bacteria level is used.
*  `deepnog infer proteins.faa --out prediction.csv`
    * Write into prediction.csv instead
*  `deepnog infer proteins.faa -db eggNOG5 -t 1236 -V 3 -c 0.99`
    * Predict EggNOG5 Gammaproteobacteria (tax 1236) groups
    * discard individual predictions below 99 % confidence
    * Show detailed progress report (-V 3)
*  `deepnog train train.fa val.fa train.csv val.csv -a deepnog -e 15 --shuffle
                 -r 123 -db eggNOG5 -t 3 -o /path/to/outdir`
    * Train a model for the (hypothetical) tax level 3 of eggNOG5 with a fixed
      random seed for reproducible results.


The individual models for OG predictions are not stored on GitHub or PyPI,
because they exceed file size limitations (up to 200M).
`deepnog` automatically downloads the models, and puts them into a
cache directory (default `~/deepnog_data/`). You can change this directory
by setting the `DEEPNOG_DATA` environment variable.

For help and advanced options, call `deepnog --help`,
and `deepnog infer --help` or `deepnog train --help` for specific options
for inference or training, respectively.
See also the [user & developer guide](doc/guide.pdf).

## File formats supported

Preferred: FASTA (raw or gzipped)

DeepNOG supports protein sequences stored in all file formats listed in
https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
only.

## Databases currently supported

- eggNOG 5.0, taxonomic level 1 (root level)
- eggNOG 5.0, taxonomic level 2 (bacteria level)
- eggNOG 5.0, taxonomic level 1236 (Gammaproteobacteria)
- (for additional levels, please create an issue on Github, or train a model yourself---new in v1.2)

## Deep network architectures currently supported

* DeepNOG
* DeepFam (no precomputed model currently available)


## Required packages (and minimum version)

*  PyTorch 1.2.0
*  NumPy 1.16.4
*  pandas 0.25.1
*  scikit-learn
*  tensorboard
*  Biopython 1.74
*  PyYAML
*  tqdm 4.35.0
*  pytest 5.1.2 (for tests only)

See also `requirements/*.txt` for platform-specific recommendations
(sometimes, specific versions might be required due to platform-specific
bugs in the deepnog requirements)

## Acknowledgements
This research is supported by the Austrian Science Fund (FWF): P27703, P31988;
and by the GPU grant program of Nvidia corporation.

## Citation
A research article is currently under review.
