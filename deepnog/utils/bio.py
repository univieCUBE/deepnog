from functools import partial
import gzip
import lzma
from pathlib import Path
from typing import Iterator
import warnings

__all__ = ['EXTENDED_IUPAC_PROTEIN_ALPHABET',
           'SeqIO',
           'parse',
           ]

# Bio.Alphabet.ExtendendIUPACProtein (deprecated in 2020)
EXTENDED_IUPAC_PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'

# Biopython warns about Alphabet, even if you don't use Alphabet...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    from Bio import SeqIO


def parse(p: Path, fformat: str = 'fasta', alphabet=None) -> Iterator:
    """ Parse a possibly compressed sequence file.

    Parameters
    ----------
    p : Path or str
        Path to sequence file
    fformat : str
        File format supported by Biopython.SeqIO.parse, e.g "fasta"
    alphabet : any
        Pass alphabet to SeqIO.parse

    Returns
    -------
    it : Iterator
        The SeqIO.parse iterator yielding SeqRecords
    """
    p = Path(p)
    if p.suffix in ['.gz', '.gzip']:
        _open = partial(gzip.open, mode='rt')
    elif p.suffix in ['.xz', '.lzma']:
        _open = partial(lzma.open, mode='rt')
    else:
        _open = open
    return SeqIO.parse(_open(str(p)), format=fformat, alphabet=alphabet)
