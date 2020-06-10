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
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from deepnog.sync import SynchronizedCounter
from deepnog.utils import EXTENDED_IUPAC_PROTEIN_ALPHABET, SeqIO

# Class of data-minibatches
collated_sequences = namedtuple('collated_sequences',
                                ['indices', 'ids', 'sequences', 'labels'])


def collate_sequences(batch: Union[List[namedtuple], namedtuple],
                      zero_padding: bool = True, min_length: int = 36):
    """ Collate and zero-pad encoded sequence.

    Parameters
    ----------
    batch : namedtuple, or list of namedtuples
        Batch of protein sequences to classify stored as a namedtuple
        sequence (see ProteinDataset).
    zero_padding : bool
        Zero-pad protein sequences, that is, append zeros until every sequence
        is as long as the longest sequences in batch.
    min_length : int, optional
        Zero-pad sequences to at least ``min_length``.
        By default, this is set to 36, which is the largest kernel size in the
        default DeepNOG/DeepEncoding architecture.

    Returns
    -------
    batch : namedtuple
        Input batch zero-padded and stored in namedtuple
        collated_sequences.
    """
    # Check if an individual sample or a batch was given
    if not isinstance(batch, list):
        batch = [batch]

    # Find the longest sequence, in order to zero pad the others
    max_len = min_length
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

    # Collate the labels
    try:
        labels = np.array([b.label for b in batch], dtype=np.int)
        labels = default_collate(labels)
    except AttributeError:
        labels = None

    return collated_sequences(indices=indices,
                              ids=ids,
                              sequences=sequences,
                              labels=labels)


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
    worker_id data samples in the initialization.

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
    labels : pd.DataFrame
        Dataframe storing labels associated to the sequences.
        This is required for training, and ignored during inference.
        Must contain 'protein_id' and 'label_num' columns providing
        identifiers and numerical labels.
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
    """

    def __init__(self, file_, labels: pd.DataFrame, aa_vocab, f_format,
                 n_skipped: Union[int, SynchronizedCounter] = 0,
                 num_workers=1, worker_id=0):
        # Generate file-iterator
        if Path(file_).suffix == '.gz':
            f = gzip.open(file_, 'rt')
            iterator = SeqIO.parse(f, format=f_format, )
        else:
            iterator = SeqIO.parse(file_, format=f_format, )
        self.iterator = iterator

        if labels is None:
            self.label_from_id = {}
        else:
            # NOTE: this only supports single-label experiments!
            # In case of multi-labels, the last label will overwrite
            # any previously seen labels.
            self.label_from_id = {row.protein_id: row.label_num
                                  for row in labels.itertuples()}

        self.vocab = aa_vocab
        self.n_skipped = n_skipped

        # Start position
        self.start = worker_id
        self.pos = None

        # Number of sequences to skip for each next() call.
        self.step = num_workers - 1

        # Make Dataset return namedtuple
        self.sequence = namedtuple('sequence',
                                   ['index', 'id', 'string', 'encoded', 'label'])

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
            Contains sequence data and metadata, i.e. all relevant
            information deepnog needs to perform and store predictions
            for one protein sequence.
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
            # If sequence has no identifier, skip it
            while next_seq.id == '':
                self.n_skipped += 1
                consume(self.iterator, n=self.step)
                self.pos += self.step + 1
                next_seq = next(self.iterator)
            # Generate sequence object from SeqRecord
            sequence_id = f'{next_seq.id}'
            sequence = self.sequence(index=self.pos,
                                     id=sequence_id,
                                     string=str(next_seq.seq),
                                     encoded=[self.vocab.get(c, 0)
                                              for c in next_seq.seq],
                                     label=self.label_from_id.get(sequence_id))
        except StopIteration:
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
    labels_file : str, optional
        Path to file storing labels associated to the sequences.
        This is required for training, and ignored during inference.
        Must be in CSV format with header line, that is, compatible to be
        read by pandas.read_csv(). The labels are expected in the last column.
    f_format : str
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    """

    def __init__(self, file, labels_file: str = None, f_format='fasta'):
        """ Initialize sequence dataset from file."""
        self.file = file
        self.f_format = f_format

        # Read labels, if available
        self.labels_file = labels_file
        if self.labels_file is None:
            self.labels = None
        else:
            self.labels = pd.read_csv(labels_file)
            self.label_encoder = LabelEncoder()
            self.labels['label_num'] = self.label_encoder.fit_transform(
                self.labels.iloc[:, -1])

        # Generate amino-acid vocabulary
        self.alphabet = EXTENDED_IUPAC_PROTEIN_ALPHABET
        self.vocab = gen_amino_acid_vocab(self.alphabet)

        self.n_skipped = SynchronizedCounter(init=0)

    def __iter__(self):
        """ Return iterator over sequences in file. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return ProteinIterator(self.file, self.labels, self.vocab,
                                   self.f_format, n_skipped=0)
        else:
            return ProteinIterator(self.file, self.labels, self.vocab,
                                   self.f_format, n_skipped=self.n_skipped,
                                   num_workers=worker_info.num_workers,
                                   worker_id=worker_info.id)

    def __len__(self):
        try:
            return self.labels.label_num.size
        except AttributeError:
            raise TypeError(f"object of type 'ProteinDataset' has no len(), "
                            f"unless a label file is provided during its "
                            f"construction.") from None
