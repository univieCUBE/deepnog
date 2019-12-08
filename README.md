# DeepNOG - predicting orthologous groups (OGs) of proteins

Deep learning based command line tool written in Python 3 (3.7.4). 

Predicts the OGs of given protein sequences based on pretrained neural
networks.

Version: 1.0.0

## Required packages

*  PyTorch 1.2.0
*  NumPy 1.16.4
*  pandas 0.25.1
*  Biopython 1.74
*  tqdm 4.35.0
*  pytest 5.1.2 (for tests only)

## Installation guide

Clone or download the source code of the project and run

```pip install /path/to/DeepNOG```

If you plan to extend DeepNOG as a developer, run

```pip install -e /path/to/DeepNOG```

instead.

## Usage

DeepNOG can be used through calling the above installed deepnog command with a protein data file. 

Example usages: 

*  deepnog proteins.faa 
    * OGs prediction of proteins in proteins.faa will be written into out.csv
*  deepnog proteins.faa --out prediction.csv
    * Write into prediction.csv instead
*  deepnog proteins.faa --tab
    * Instead of semicolon (;) separated, generate tab separated output-file

For help and advanced options, call deepnog --help or see the [user & developer guide.](../doc/guide.pdf)

## File formats supported

Prefered: FASTA (raw or gzipped)

DeepNOG supports protein sequences stored in all file formats listed in
https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
only.

## Databases supported

eggNOG 5.0, taxonomic level 2 (bacteria level)

## Neural network architectures supported

*  DeepEncoding, for details, see the [report](../doc/report.pdf). (ToDo: change to paper)


    