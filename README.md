[![Linux/MacOS builds on Travis](
  https://travis-ci.com/VarIr/deepnog.svg?token=Pv7ns6A7X34baaBVUTz8&branch=master)](
  https://travis-ci.com/VarIr/deepnog)
[![Windows builds on AppVeyor](
  https://ci.appveyor.com/api/projects/status/xxxxxxxxxxxxxx/branch/master?svg=true)](
  https://ci.appveyor.com/project/VarIr/deepnog/branch/master)
[![codecov](
  https://codecov.io/gh/VarIr/deepnog/branch/master/graph/badge.svg?token=aP6UBdQDmk)](
  https://codecov.io/gh/VarIr/deepnog)
[![Language grade: Python](
  https://img.shields.io/lgtm/grade/python/g/VarIr/deepnog.svg?logo=lgtm&logoWidth=18)](
  https://lgtm.com/projects/g/VarIr/deepnog/context:python)
[![Documentation Status](
  https://readthedocs.org/projects/deepnog/badge/?version=latest)](
  https://deepnog.readthedocs.io/en/latest/?badge=latest)


# DeepNOG: protein orthologous groups prediction

Predict orthologous groups of proteins on CPUs or GPUs with deep networks.
DeepNOG is both faster and more accurate than assigning OGs with HMMER.

The `deepnog` command line tool is written in Python 3.7+. 

Current version: 1.0.4

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

*  deepnog proteins.faa 
    * OGs prediction of proteins in proteins.faa will be written into out.csv
*  deepnog proteins.faa --out prediction.csv
    * Write into prediction.csv instead
*  deepnog proteins.faa --tab
    * Instead of semicolon (;) separated, generate tab separated output-file

The individual models for OG predictions are not stored on GitHub or PyPI,
because they exceed file size limitations (up to 200M).
`deepnog` automatically downloads the models, and puts them into a
cache directory (default `~/deepnog_data/`). You can change this directory
by setting the `DEEPNOG_DATA` environment variable.

For help and advanced options, call `deepnog --help`,
or see the [user & developer guide](doc/guide.pdf).

## File formats supported

Preferred: FASTA (raw or gzipped)

DeepNOG supports protein sequences stored in all file formats listed in
https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
only.

## Databases supported

- eggNOG 5.0, taxonomic level 1 (root level)
- eggNOG 5.0, taxonomic level 2 (bacteria level)
- (for additional level, please create an issue)

## Neural network architectures supported

*  DeepEncoding (=DeepNOG in the research article)


## Required packages (and minimum version)

*  PyTorch 1.2.0
*  NumPy 1.16.4
*  pandas 0.25.1
*  Biopython 1.74
*  tqdm 4.35.0
*  pytest 5.1.2 (for tests only)

## Acknowledgements
This research is supported by the Austrian Science Fund (FWF): P27703, P31988,
and by the GPU grant program of Nvidia corporation.

## Citation
A research article is currently in preparation.
