"""
Author: Lukas Gosch

Date: 2019-10-03

Description:

    Dataset classes and helper functions for usage with deep network models
    written in PyTorch.
"""
from collections import namedtuple, deque
import gzip
from itertools import islice
import os
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from . import sync
from .utils import EXTENDED_IUPAC_PROTEIN_ALPHABET, SeqIO

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
    max_len = 36
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
        # Convert NumPy array to PyTorch Tensor
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
    alphabet to numbers. As this mapping is aware of zero-padding,
    it maps the first letter in the alphabet to 1 instead of 0.

    Parameters
    ----------
    alphabet : str
        Alphabet to use for vocabulary.
        If None, use 'ACDEFGHIKLMNPQRSTVWYBXZJUO' (equivalent to deprecated
        Biopython's ExtendedIUPACProtein).

    Returns
    -------
    vocab : dict
        Mapping of amino acid characters to numbers.
    """
    if alphabet is None:
        # Use all 26 letters from Bio.Alphabet.ExtendendIUPACProtein
        # (deprecated in 2020)
        alphabet = EXTENDED_IUPAC_PROTEIN_ALPHABET

    # In case of ExtendendIUPACProtein: Map 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
    # to [1, 26] so that zero padding does not interfere.
    aminoacid_to_ix = {}
    for i, aa in enumerate(alphabet):
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


class ProteinIterator:
    """ Iterator allowing for multiprocess data loading of a sequence file.

    ProteinIterator is a wrapper for the iterator returned by
    Biopython's Bio.SeqIO class when parsing a sequence file. It
    specifies custom __next__() method to support single- and multi-process
    data loading.

    In the single-process loading case, nothing special happens,
    the ProteinIterator sequentially iterates over the data file. In the end,
    it informs the main module about the number of skipped sequences (due to
    empty ids) through setting a global variable in the main module.

    In the multi-process loading case, each ProteinIterator loads a sequence
    and then skips the next few sequences dedicated to the other workers.
    This works by each worker skipping num_worker - 1 data samples
    for each call to __next__(). Furthermore, each worker skips
    worker_id data samples in the initialization. At the end of the
    workers lifetime, it sends the number of skipped sequences back to the
    main process through a pipe the main process created.

    The ProteinIterator class also makes sure that a unique ID is set for each
    SeqRecord obtained from the data-iterator. This allows unambiguous handling
    of large protein datasets which may have duplicate IDs from merging
    multiple sources or may have no IDs at all. For easy and efficient
    sorting of batches of sequences as well as for direct access to the
    original IDs, the index is stored separately.

    Parameters
    ----------
    file_ : str
        Path to sequence file, from which an iterator over the sequences
        will be created with Biopython's Bio.SeqIO.parse() function.
    aa_vocab : dict
        Amino-acid vocabulary mapping letters to integers
    f_format : str
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    num_workers : int
        Number of workers set in DataLoader or one if no workers set.
        If bigger or equal to two, the multi-process loading case happens.
    worker_id : int
        ID of worker this iterator belongs to

    Attributes
    ----------
    sequence : namedtuple
        Sequence data and metadata: All relevant information DeepNOG needs to
        perform and store protein predictions for one protein sequence
    """

    def __init__(self, file_, aa_vocab, f_format, num_workers=1, worker_id=0):
        # Generate file-iterator
        if Path(file_).suffix == '.gz':
            f = gzip.open(file_, 'rt')
            iterator = SeqIO.parse(f, format=f_format, )
        else:
            iterator = SeqIO.parse(file_, format=f_format, )
        self.iterator = iterator
        self.vocab = aa_vocab
        self.n_skipped = 0
        self.communicated = False
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

        If last protein sequences was read, communicates number of
        skipped protein sequences due to empty ids back to main process.

        Returns
        -------
        sequence : namedtuple
            Element at current + step + 1 position or start position.
            Furthermore prefixes element with unique sequential ID.
        """
        # Check if iterator has been positioned correctly.
        if self.pos is not None:
            consume(self.iterator, n=self.step)
            self.pos += self.step + 1
        else:
            consume(self.iterator, n=self.start)
            self.pos = self.start + 1
        try:
            next_seq = next(self.iterator)
            # If sequences has no identifier, skip it
            while next_seq.id == '':
                self.n_skipped += 1
                consume(self.iterator, n=self.step)
                self.pos += self.step + 1
                next_seq = next(self.iterator)
            # Generate sequence object from SeqRecord
            sequence = self.sequence(index=self.pos,
                                     id=f'{next_seq.id}',
                                     string=str(next_seq.seq),
                                     encoded=[self.vocab.get(c, 0)
                                              for c in next_seq.seq])
        except StopIteration:
            # Check if skipped sequences have been communicated back to main
            # process
            if not self.communicated:
                # Check if subprocesses (workers) were created to read data
                if self.step == 0:
                    # If no workers were used to read the data, ProteinIterator
                    # runs in same process as main module and can access its
                    # namespace.
                    sync.n_skipped = self.n_skipped
                else:
                    try:
                        # Close reading pipe dedicated for this worker process
                        os.close(sync.rpipe_l[self.start])
                        # Prepare message to send to main process
                        msg = f'{self.n_skipped}'.encode(encoding='utf-8')
                        # Send message to main process
                        os.write(sync.wpipe_l[self.start], msg)
                    except (IndexError,  # if sync.pipes are not set up
                            IOError):    # general errors
                        warnings.warn(f'Interprocess communication failed. '
                                      f'The reported number of problematic '
                                      f'sequences will be unreliable. ')
                self.communicated = True

            # Close the file handle
            self.iterator.close()

            raise StopIteration

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
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    """

    def __init__(self, file, f_format='fasta'):
        """ Initialize sequence dataset from file."""
        self.file = file
        self.f_format = f_format

        # Generate amino-acid vocabulary
        self.alphabet = EXTENDED_IUPAC_PROTEIN_ALPHABET
        self.vocab = gen_amino_acid_vocab(self.alphabet)

    def __iter__(self):
        """ Return iterator over sequences in file. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return ProteinIterator(self.file, self.vocab, self.f_format)
        else:
            return ProteinIterator(self.file, self.vocab, self.f_format,
                                   worker_info.num_workers,
                                   worker_info.id)
