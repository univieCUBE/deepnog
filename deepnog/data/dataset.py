"""
Author: Lukas Gosch

Date: 2019-10-03

Description:

    Dataset classes and helper functions for usage with deep network models
    written in PyTorch.
"""
import gzip
from itertools import islice
from pathlib import Path
from typing import List, Union, NamedTuple, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Bio.SeqRecord import SeqRecord

from ..utils import get_logger, SynchronizedCounter
from ..utils import EXTENDED_IUPAC_PROTEIN_ALPHABET, parse, SeqIO
from ..utils import try_import_pytorch

torch = try_import_pytorch()
from torch.utils.data import Dataset, IterableDataset  # noqa
from torch.utils.data.dataloader import default_collate  # noqa

__all__ = ['collate_sequences',
           'gen_amino_acid_vocab',
           'ProteinDataset',
           'ProteinIterableDataset',
           'ProteinIterator',
           'ShuffledProteinIterableDataset',
           ]

# Make Dataset return namedtuple
sequence_tuple = NamedTuple('sequence',
                            [('index', int),
                             ('id', str),
                             ('string', str),
                             ('encoded', list),
                             ('label', torch.Tensor)])

# Class of data-minibatches
collated_sequences = NamedTuple('collated_sequences',
                                [('indices', List[int]),
                                 ('ids', List[str]),
                                 ('sequences', torch.Tensor),
                                 ('labels', torch.Tensor)])

UNINITIALIZED_POS = -999


def collate_sequences(batch: Union[List[sequence_tuple], sequence_tuple],
                      zero_padding: bool = True, min_length: int = 36,
                      random_padding: bool = False) -> collated_sequences:
    """ Collate and zero-pad encoded sequence.

    Parameters
    ----------
    batch : namedtuple, or list of namedtuples
        Batch of protein sequences to classify stored as a namedtuple
        sequence.
    zero_padding : bool
        Zero-pad protein sequences, that is, append zeros until every sequence
        is as long as the longest sequences in batch.
        NOTE: currently unused. Zero-padding is always performed.
    min_length : int, optional
        Zero-pad sequences to at least ``min_length``.
        By default, this is set to 36, which is the largest kernel size in the
        default DeepNOG architecture.
    random_padding : bool, optional
        Zero pad sequences by prepending and appending zeros. The fraction
        is determined randomly. This may counter detrimental effects, when
        short sequences would always have long zero-tails, otherwise.
    Returns
    -------
    batch : NamedTuple
        Input batch zero-padded and stored in namedtuple
        collated_sequences.
    """
    if not zero_padding:
        warnings.warn(f"Called collate_sequences(zero_padding={zero_padding}). "
                      f"However, all sequences will currently be zero-padded.")
    # Check if an individual sample or a batch was given
    if not isinstance(batch, list):
        batch = [batch]

    # Find the longest sequence, in order to zero pad the others
    max_len = min_length
    n_data = 0
    for seq in batch:
        sequence = seq.encoded
        n_data += 1
        sequence_len = len(sequence)
        if sequence_len > max_len:
            max_len = sequence_len

    # Collate the sequences
    sequences = np.zeros((n_data, max_len,), dtype=np.int32)
    for i, seq in enumerate(batch):
        sequence = np.array(seq.encoded)
        # If selected, choose randomly, where to insert zeros
        if random_padding and len(sequence) < max_len:
            n_zeros = max_len - len(sequence)
            start = np.random.choice(n_zeros + 1)
            end = start + len(sequence)
        else:
            start = 0
            end = len(sequence)
        # Zero pad
        sequences[i, start:end] = sequence[:].T
    # Convert NumPy array to PyTorch Tensor
    sequences = default_collate(sequences)

    # Collate the protein ids (str)
    ids = [seq.id for seq in batch]

    # Collate the numerical indices
    indices: List[int] = [seq.index for seq in batch]

    # Collate the labels
    try:
        labels = np.array([b.label for b in batch], dtype=np.int32)
        labels = default_collate(labels)
    except (AttributeError, TypeError):
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


def _consume(iterator, n: int):
    """ Advance the iterator n-steps ahead. If n is None, consume entirely.

    Function from Itertools Recipes in official Python 3.7.4. docs.
    """
    # Use functions that consume iterators at C speed.
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
            self.has_labels = False
            self.label_from_id = {}
        else:
            # NOTE: this only supports single-label experiments!
            # In case of multi-labels, the last label will overwrite
            # any previously seen labels.
            self.label_from_id = {row.protein_id: row.label_num
                                  for row in labels.itertuples()}
            self.has_labels = True

        self.vocab = aa_vocab
        self.n_skipped = n_skipped

        # Start position
        self.start: int = worker_id
        self.pos: int = UNINITIALIZED_POS

        # Number of sequences to skip for each next() call.
        self.step: int = num_workers - 1

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
        if self.pos == UNINITIALIZED_POS:
            _consume(self.iterator, n=self.start)
            self.pos = self.start + 1
        else:
            _consume(self.iterator, n=self.step)
            self.pos += self.step + 1

        next_seq = next(self.iterator)
        # If sequence has no identifier, skip it.
        # Also skip sequences that should have labels, but don't.
        sequence_id: str = f'{next_seq.id}'
        label = self.label_from_id.get(sequence_id)
        while sequence_id == '' or (self.has_labels and label is None):
            self.n_skipped += 1
            _consume(self.iterator, n=self.step)
            self.pos += self.step + 1
            next_seq = next(self.iterator)
            sequence_id: str = f'{next_seq.id}'
            label = self.label_from_id.get(sequence_id)
        # Generate sequence object from SeqRecord
        encoded = [self.vocab.get(c, 0) for c in next_seq.seq]
        sequence = sequence_tuple(index=self.pos,
                                  id=sequence_id,
                                  string=str(next_seq.seq),
                                  encoded=encoded,
                                  label=label)

        return sequence


class ProteinIterableDataset(IterableDataset):
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
        Must be in CSV format with header line and index column, that is,
        compatible to be read by pandas.read_csv(..., index_col=0).
        The labels are expected in a column named "eggnog_id" or
        in the last column.
    f_format : str
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    label_encoder : LabelEncoder, optional
        The label encoder maps str class names to numerical labels.
        Provide a label encoder during validation.
    """

    def __init__(self, file, labels_file: str = None, f_format='fasta',
                 label_encoder: LabelEncoder = None):
        """ Initialize sequence dataset from file."""
        self.file = file
        self.f_format = f_format

        # Read labels, if available
        self.labels_file = labels_file
        if self.labels_file is None:
            self.labels = None
        else:
            self.labels = pd.read_csv(labels_file,
                                      index_col=0,
                                      compression='infer',
                                      dtype=str,
                                      )

            # Sequence IDs and labels are assumed in named columns,
            # but if not, let's try a specific order and hope for the best
            self.labels = _rename_labels_columns(self.labels)

            # Transform class names to numerical labels
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.labels['label_num'] = self.label_encoder.fit_transform(
                    self.labels.eggnog_id)
            else:
                self.label_encoder = label_encoder
                self.labels['label_num'] = self.label_encoder.transform(
                    self.labels.eggnog_id)

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
            raise TypeError(f"object of type {type(self)} has no len(), "
                            f"unless a label file is provided during its "
                            f"construction.") from None


class ShuffledProteinIterableDataset(ProteinIterableDataset):
    """ Shuffle an iterable ProteinDataset by introducing a shuffle buffer.

    Parameters
    ----------
    file : str
        Path to file storing the protein sequences.
    labels_file : str, optional
        Path to file storing labels associated to the sequences.
        This is required for training, and ignored during inference.
        Must be in CSV format with header line and index column, that is,
        compatible to be read by pandas.read_csv(..., index_col=0).
        The labels are expected in a column named "eggnog_id" or
        in the last column.
    f_format : str
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    label_encoder : LabelEncoder, optional
        The label encoder maps str class names to numerical labels.
        Provide a label encoder during validation.
    buffer_size : int
        How many objects will be buffered, i.e. are available to choose from.

    References
    ----------
    Adapted from code by Sharvil Nanavati, see
    https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/5
    """
    def __init__(self, file, labels_file: str = None, f_format='fasta',
                 label_encoder: LabelEncoder = None, buffer_size: int = 1000):
        super().__init__(file=file, labels_file=labels_file, f_format=f_format,
                         label_encoder=label_encoder)
        self.dataset = self
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            dataset_iter = ProteinIterator(self.file, self.labels, self.vocab,
                                           self.f_format, n_skipped=0)
        else:
            dataset_iter = ProteinIterator(self.file, self.labels, self.vocab,
                                           self.f_format, n_skipped=self.n_skipped,
                                           num_workers=worker_info.num_workers,
                                           worker_id=worker_info.id)
        try:
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = np.random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def _rename_labels_columns(df):
    """ Sequence IDs and labels are assumed in named columns,
    but if not, let's try a specific order and hope for the best
    """
    try:
        df.eggnog_id
    except AttributeError:
        df = df.rename(columns={df.columns[-1]: 'eggnog_id'})
    try:
        df.protein_id
    except AttributeError:
        df = df.rename(columns={df.columns[0]: 'protein_id'})
    return df


class ProteinDataset(Dataset):
    """ Protein dataset with sequences and labels for training.

    If sequences and labels are provided as files rather than objects,
    loads and stores all proteins from input files during construction.
    While this comes at the price of some delay, it allows to truly shuffle
    the complete dataset during training.

    Parameters
    ----------
    sequences : list, str, Path
        Protein sequences as list of Biopython Seq, or path to fasta file
        containing the sequences.
    labels : DataFrame, str, Path, optional
        Protein orthologous group labels as DataFrame,
        or str to CSV file containing such a dataframe.
        This is required for training, and ignored during inference.
        Must be in CSV format with header line and index column, that is,
        compatible to be read by pandas.read_csv(..., index_col=0).
        The labels are expected in a column named "eggnog_id" or
        in the last column, and sequence IDs in a column "protein_id".
    f_format : str, optional
        File format in which to expect the protein sequences.
        Must be supported by Biopython's Bio.SeqIO class.
    label_encoder : LabelEncoder, optional
        The label encoder maps str class names to numerical labels.
        Provide a label encoder during validation.
    verbose: int, optional
        Control verbosity of logging.
    """
    def __init__(self, sequences: Union[Sequence[SeqRecord], str, Path],
                 labels: Union[pd.DataFrame, str, Path, None] = None,
                 f_format: str = 'fasta',
                 label_encoder: Union[LabelEncoder, None] = None,
                 verbose: int = 0,
                 ):
        self.sequences = sequences
        self.labels = labels
        self.f_format = f_format
        self.label_encoder = label_encoder
        self.verbose = verbose
        self.logger = get_logger(__name__, verbose=self.verbose)

        # Read labels, if available
        if self.labels is None:
            self.logger.info('Not using labels')
            self.label_from_id = {}
        else:
            self.logger.info('Loading labels')
            try:
                self.labels = pd.read_csv(self.labels,
                                          index_col=0,
                                          compression='infer',
                                          dtype=str,
                                          )
            except (ValueError, TypeError):
                pass
            if not isinstance(self.labels, pd.DataFrame):
                raise ValueError('Invalid labels, must be .csv file or DataFrame')

            # Try to rename columns, if not already correctly named
            self.labels = _rename_labels_columns(self.labels)

            # Transform class names to numerical labels
            if self.label_encoder is None:
                self.logger.info('Setting up new LabelEncoder')
                self.label_encoder = LabelEncoder()
                self.labels['label_num'] = self.label_encoder.fit_transform(
                    self.labels.eggnog_id)
            else:
                self.logger.info('Using provided LabelEncoder')
                try:
                    self.labels['label_num'] = self.label_encoder.transform(
                        self.labels.eggnog_id)
                except ValueError:
                    n_before = self.labels.shape[0]
                    df_available = pd.DataFrame(self.label_encoder.classes_, columns=['eggnog_id'])
                    self.labels = self.labels.merge(df_available)
                    self.logger.warning(f'Removed {n_before - self.labels.shape[0]} '
                                        f'sequences of unknown classes.')
                    self.labels['label_num'] = self.label_encoder.transform(
                        self.labels.eggnog_id)
            self.label_from_id = {row.protein_id: row.label_num
                                  for row in self.labels.itertuples()}

        self.logger.info('Loading sequences')
        try:
            if self.labels is None:
                self.sequences = list(parse(self.sequences))
            else:
                sequences = []
                n_skipped: int = 0
                for record in parse(self.sequences):
                    if self.label_from_id.get(record.id, None) is None:
                        n_skipped += 1
                    else:
                        sequences.append(record)
                self.sequences = sequences
                if n_skipped:
                    self.logger.warning(f'{n_skipped}/{len(sequences) + n_skipped} '
                                        f'sequences without labels were skipped '
                                        f'({len(self.label_from_id)} total labels).')
        except TypeError:
            pass  # A list of SeqRecords may have been passed
        if not isinstance(self.sequences[0], SeqRecord):
            raise ValueError(f'Invalid sequences, must be FASTA file or '
                             f'a list/tuple of {SeqRecord}')

        # Generate amino-acid vocabulary
        self.alphabet = EXTENDED_IUPAC_PROTEIN_ALPHABET
        self.vocab = gen_amino_acid_vocab(self.alphabet)

        self.n_skipped = SynchronizedCounter(init=0)
        self.logger.debug('Dataset init complete')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        seq = self.sequences[item]
        sequence_id: str = f'{seq.id}'
        label = self.label_from_id.get(sequence_id, None)
        encoded = [self.vocab.get(c, 0) for c in seq]
        sequence = sequence_tuple(index=item,
                                  id=sequence_id,
                                  string=str(seq.seq),
                                  encoded=encoded,
                                  label=label)
        return sequence
