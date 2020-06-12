import warnings

__all__ = ['EXTENDED_IUPAC_PROTEIN_ALPHABET',
           'SeqIO',
           ]

# Bio.Alphabet.ExtendendIUPACProtein (deprecated in 2020)
EXTENDED_IUPAC_PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'

# Biopython warns about Alphabet, even if you don't use Alphabet...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    from Bio import SeqIO
