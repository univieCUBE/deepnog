![Linux/macOS builds on Actions](
  https://github.com/univieCUBE/deepnog/workflows/deepnog%20CI/badge.svg)
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
[![PyPI version](
  https://badge.fury.io/py/deepnog.svg)](
  https://badge.fury.io/py/deepnog)
[![Anaconda-Server Badge](
  https://anaconda.org/bioconda/deepnog/badges/version.svg)](
  https://anaconda.org/bioconda/deepnog)
![PyPI - Python Version](
  https://img.shields.io/pypi/pyversions/deepnog?style=flat-square)


# DeepNOG: protein orthologous groups assignment

Assign proteins to orthologous groups (eggNOG 5) on CPUs or GPUs with deep networks.
DeepNOG is much faster than alignment-based methods,
providing accuracy similar to HMMER.


## Installation guide

The easiest way to install DeepNOG is to obtain it from PyPI:
``` bash
pip install deepnog
```

Alternatively, you can clone or download bleeding edge versions
from GitHub and run
``` bash
pip install /path/to/DeepNOG
```

If you plan to extend DeepNOG as a developer, run
``` bash
pip install -e /path/to/DeepNOG
```

instead.

``deepnog`` can also be installed from bioconda like this:
``` bash
conda config --add channels pytorch
conda install pytorch deepnog
```

## Usage

Call the `deepnog` command line tool with a
protein sequence file in FASTA format.
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

Preferred: FASTA (raw, .gz, or .xz)

DeepNOG supports protein sequences stored in all file formats listed in
[https://biopython.org/wiki/SeqIO](https://biopython.org/wiki/SeqIO),
but is tested for the FASTA-file format
only.

## Databases currently supported

- eggNOG 5.0
  * taxonomic level 1 (root level)
  * taxonomic level 2 (bacteria level)
  * For >100 additional eggNOG 5.0 levels, consult the
  [docs](https://deepnog.readthedocs.io/en/latest/documentation/models.html).
- COG 2020
- (for additional databases/levels, please create an issue on Github,
   or train a model yourself---new in v1.2)

## Deep network architectures currently supported

* DeepNOG
* DeepFam (no precomputed model currently available)


## Required packages

``deepnog`` builds upon the following packages:
*  PyTorch
*  NumPy
*  pandas
*  scikit-learn
*  tensorboard
*  Biopython
*  PyYAML
*  tqdm
*  pytest (for tests only)

See also `requirements/*.txt` for platform-specific recommendations
(sometimes, specific versions might be required due to platform-specific
bugs in the deepnog requirements)

## Acknowledgements
This research is supported by the Austrian Science Fund (FWF): P27703, P31988;
and by the GPU grant program of Nvidia corporation.

## Citation
If you use DeepNOG, please consider citing our research article ([click here for bibtex](https://academic.oup.com/Citation/Download?resourceId=6050698&resourceType=3&citationFormat=2)):

Roman Feldbauer, Lukas Gosch, Lukas Lüftinger, Patrick Hyden,
Arthur Flexer, Thomas Rattei,
DeepNOG: Fast and accurate protein orthologous group assignment,
*Bioinformatics*, 2020, btaa1051, https://doi.org/10.1093/bioinformatics/btaa1051
