# DeepNOG - predicting orthologous groups (OGs) of proteins

Deep learning based command line tool written in Python 3 (3.7.4). 

Predicts the OGs of given protein sequences based on pretrained neural
networks.

## Required packages

*  PyTorch 1.1+
*  NumPy
*  pandas
*  Biopython
*  tqdm
*  pytest (for tests only)

## Usage

To use DeepNOG, clone project or download tool-folder and call deepnog.py with a protein data file. 

Example usages: 

*  python deepnog.py proteins.faa 
    * OGs prediction of proteins in proteins.faa will be written into out.csv
*  python deepnog.py proteins.faa --out prediction.csv
    * Write into prediction.csv instead
*  python deepnog.py proteins.faa --tab
    * Instead of semicolon (;) separated, generated tab separated output-file

For help and advanced options, call python deepnog.py --help

## File formats supported

Prefered: FASTA

DeepNOG supports protein sequences stored in all file formats listed in
https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
only.

## Databases supported

eggNOG 5.0, taxonomic level 2

## Neural network architectures supported

*  DeepEncoding
    * Details to follow

    