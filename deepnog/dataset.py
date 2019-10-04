"""
Author: Lukas Gosch
Date: 3.10.2019
Description:
    Dataset classes and helper functions for usage with neural network models 
    written in PyTorch.
"""
import os
from itertools import islice
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate
from Bio import SeqIO
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein

# Class of data-minibatches
collated_sequences = namedtuple('collated_sequences',
                                ['indices', 'ids', 'sequences'])


def collate_sequences(batch, zero_padding=True):
    """ Collate and zero-pad encoded sequence. 

    Parameters
    ----------
    batch : list[namedtuple] or namedtuple
        Batch of protein sequences to classify stored as a namedtuple-class
        sequence (see ProteinDataset).
    zero_padding : bool
        If True, zero-pads protein sequences through appending zeros until
        every sequence is as long as the longest sequences in batch. If False
        raise NotImplementedError.

    Returns
    -------
    batch : namedtuple
        Input batch zero-padded and stored in namedtuple-class 
        collated_sequences.
    """
    # Check if an individual sample or a batch was given
    if not isinstance(batch, list):
        batch = [batch]

    # Find the longest sequence, in order to zero pad the others
    max_len = 0
    n_data = 0
    for seq in batch:
        query = seq.encoded
        n_data += 1
        sequence_len = len(query)
        if sequence_len > max_len:
            max_len = sequence_len

    # Collate the sequences
    if zero_padding:
        sequences = np.zeros((n_data, max_len,), dtype=np.int)
        for i, seq in enumerate(batch):
            query = np.array(seq.encoded)
            start = 0
            end = len(query)
            # Zero pad
            sequences[i, start:end] = query[:].T
        # Convert NumPy array to PyToch Tensor
        sequences = default_collate(sequences)
    else:
        # no zero-padding, must use minibatches of size 1 downstream!
        raise NotImplementedError('Batching requires zero padding!')

    # Collate the ids
    ids = [seq.id for seq in batch]

    # Collate the indices
    indices = [seq.index for seq in batch]

    return collated_sequences(indices=indices,
                              ids=ids,
                              sequences=sequences)


def gen_amino_acid_vocab(alphabet=None):
    """ Create vocabulary for protein sequences. 

    A vocabulary is defined as a mapping from the amino-acid letters in the
    alphabet to numbers. As this mapping is aware of zero-padding, it maps
    the first letter in the alphabet to 1 instead of 0.
        
    Parameters
    ----------
    alphabet : Bio.Alphabet.ProteinAlphabet
        Alphabet to use for vocabulary. If None, uses 
        ExtendedIUPACProtein.

    Returns
    -------
    vocab : dict
        Mapping of amino acid characters to numbers.
    """
    if alphabet is None:
        # Use all 26 letters
        alphabet = ExtendedIUPACProtein()

    # In case of ExtendendIUPACProtein: Map 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
    # to [1, 26] so that zero padding does not interfere.
    aminoacid_to_ix = {}
    for i, aa in enumerate(alphabet.letters):
        # Map both upper case and lower case to the same embedding
        for key in [aa.upper(), aa.lower()]:
            aminoacid_to_ix[key] = i + 1
    vocab = aminoacid_to_ix
    return vocab


def consume(iterator, n=None):
    """ Advance the iterator n-steps ahead. If n is None, consume entirely.

    Function from Itertools Recipes in official Python 3.7.4. docs.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


class ProteinIterator():
    """ Iterator allowing for multiprocess dataloading of a sequence file.

    ProteinIterator is a wrapper for the iterator returned by
    Biopythons Bio.SeqIO class when parsing a sequence file. It
    specifies custom __next__() method to support multiprocess data
    loading. It does so by each worker skipping num_worker - 1 data
    samples for each call to __next__(). Furthermore, each worker skips
    worker_id data samples in the initialization.

    It also makes sure that a unique ID is set for each SeqRecord
    optained from the data-iterator. This allows unambiguous handling
    of large protein datasets which may have duplicate IDs from merging
    multiple sources or may have no IDs at all. For easy and efficient
    sorting of batches of sequences as well as for direct access to the
    original IDs, the index is stored separately.

    Parameters
    ----------
    iterator : iterator
        Iterator over sequence file returned by Biopythons
        Bio.SeqIO.parse() function.
    aa_vocab : dict
        Amino-acid vocabulary mapping letters to integers
    num_workers : int
        Number of workers set in DataLoader
    worker_id : int
        ID of worker this iterator belongs to

    Variables
    ---------
    sequence : namedtuple
        Tuple subclass named sequence holding all relevant information
        DeepNOG needs to correctly perform and store protein predictions
        for one protein sequence.
    """

    def __init__(self, iterator, aa_vocab, num_workers=1, worker_id=0):
        self.iterator = iterator
        self.vocab = aa_vocab
        # Start position
        self.start = worker_id
        self.pos = None
        # Number of sequences to skip for each next() call.
        self.step = num_workers - 1
        # Make Dataset return namedtuple
        self.sequence = namedtuple('sequence',
                                   ['index', 'id', 'string', 'encoded'])

    def __iter__(self):
        return self

    def __next__(self):
        """ Return next protein sequence in datafile as sequence object.

        Returns element at current + step + 1 position or start
        position. Fruthermore prefixes element with unique sequential
        ID.
        """
        # Check if iterator has been positioned correctly.
        if self.pos is not None:
            consume(self.iterator, n=self.step)
            self.pos += self.step + 1
        else:
            consume(self.iterator, n=self.start)
            self.pos = self.start + 1
        next_seq = next(self.iterator)
        # If sequences has no identifier, skip it
        while next_seq.id == '':
            consume(self.iterator, n=self.start)
            self.pos = self.start + 1
            next_seq = next(self.iterator)
        # Generate sequence object from SeqRecord
        sequence = self.sequence(index=self.pos,
                                 id=f'{next_seq.id}',
                                 string=str(next_seq.seq),
                                 encoded=[self.vocab[c] for c in next_seq.seq])
        return sequence


class ProteinDataset(IterableDataset):
    """ Protein dataset holding the proteins to classify.

    Does not load and store all proteins from a given sequence file but only
    holds an iterator to the next sequence to load. 

    Thread safe class allowing for multi-worker loading of sequences 
    from a given datafile.

    Parameters
    ----------
    file : str
        Path to file storing the protein sequences.
    f_format : str
        File format in which to expect the protein sequences. Must
        be supported by Biopythons Bio.SeqIO class.
    max_length : int
        If only proteins up to a certain length should be loaded.
        Defaults to None, meaning no length constraint
    zero_padding : bool
        Default behaviour is to zero pad all sequences up to
        the length of the longest one by appending zeros at the end.
        If max_length is set, zero pads all sequences up to
        max_length. False deactivates any zero padding.
    """

    def __init__(self, file, f_format='fasta', max_length=None,
                 zero_padding=True):
        """ Initialize iterator over sequences in file."""
        # Generate amino-acid vocabulary
        self.alphabet = ExtendedIUPACProtein()
        self.vocab = gen_amino_acid_vocab(self.alphabet)
        # Generate file-iterator
        if os.path.isfile(file):
            self.iter = SeqIO.parse(file, format=f_format,
                                    alphabet=self.alphabet)
        else:
            raise ValueError('Given file does not exist or is not a file.')

    def __iter__(self):
        """ Return iterator over sequences in file. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return ProteinIterator(self.iter, self.vocab)
        else:
            return ProteinIterator(self.iter, self.vocab,
                                   worker_info.num_workers,
                                   worker_info.id)
