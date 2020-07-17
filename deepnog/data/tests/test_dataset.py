"""
Author: Lukas Gosch
        Roman Feldbauer
Date: 2019-10-03
Description:
    Test dataset module.
"""
from itertools import repeat
from functools import partial
import pytest

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from deepnog.data import dataset as ds
from deepnog.tests.utils import get_deepnog_root

TESTS = get_deepnog_root()/"tests"
test_file = TESTS/"data/GCF_000007025.1.faa"
test_file_gzip = TESTS/"data/GCF_000007025.1.faa.gz"
TRAINING_FASTA = TESTS/"data/test_training_dummy.faa"
TRAINING_LABELS = TESTS/"data/test_training_dummy.faa.csv"
EXPECTED_IDS_WITH_LABEL = [f'test_all_A{x}' for x in range(11)] \
                          + [f'test_all_C{x}' for x in range(11)] \
                          + [f'M{x:02d}' for x in range(1, 9)]
EXPECTED_IDS = [f'test_all_A{x}' for x in range(12)] \
               + [f'test_all_C{x}' for x in range(12)] \
               + [f'M{x:02d}' for x in range(1, 11)]
LABELS_WRONG_COL_NAMES = TESTS/"data/test_inference_short_wrong_column_names.csv"


@pytest.mark.parametrize("f", [test_file, test_file_gzip, ])
@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_multiprocess_data_loading(f, num_workers, f_format='fasta'):
    """ Test if different workers produce different sequences and process
        whole sequence file.
    """
    dataset = ds.ProteinIterableDataset(f, f_format=f_format)
    data_loader = DataLoader(dataset,
                             num_workers=num_workers,
                             batch_size=1,
                             collate_fn=ds.collate_sequences, )
    s = set()
    i = 0
    for i, batch in enumerate(data_loader):
        # Check uniqueness of IDs for loaded data
        for identifier in batch.ids:
            assert(identifier not in s)
            s.add(identifier)
    # Check if all data was loaded
    assert(i == 1288)
    assert len(s) == 1289


@pytest.mark.parametrize("batch_size", [None, 1, 16, 32])
def test_correct_collating_sequences(batch_size, f_format='fasta'):
    """ Test if a batch of correct size is produced. """
    dataset = ds.ProteinIterableDataset(test_file, f_format=f_format)
    for i, batch in enumerate(DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         collate_fn=ds.collate_sequences)):
        if batch_size is None:
            assert(batch.sequences.shape[0] == 1)
            assert(len(batch.ids) == 1)
            assert(len(batch.indices) == 1)
        else:
            assert(batch.sequences.shape[0] == batch_size)
            assert(len(batch.ids) == batch_size)
            assert(len(batch.indices) == batch_size)
        break


@pytest.mark.parametrize('random_padding', [False, True])
def test_zero_padding(random_padding: bool, f_format='fasta'):
    """ Test correct zeroPadding. """
    pad_file = TESTS/"data/test_zeroPadding.faa"
    dataset = ds.ProteinIterableDataset(pad_file, f_format=f_format)
    for batch in DataLoader(dataset,
                            batch_size=2,
                            num_workers=0,
                            collate_fn=partial(ds.collate_sequences,  # noqa
                                               random_padding=random_padding)):
        # Test correct shape (seq1: 56aa, seq2: 112aa)
        assert batch.sequences.shape[1] == 112
        # Test correctly zeros inserted
        if random_padding:
            n_zeros = (batch.sequences[0] == 0).sum()
            assert n_zeros == 56
        else:
            assert sum(batch.sequences[0, 56:]) == 0
        assert sum(batch.sequences[1, :] == 0) == 0

    with pytest.warns(UserWarning, match="all sequences will currently be zero-padded"):
        for _ in DataLoader(dataset,  # noqa
                            collate_fn=partial(ds.collate_sequences, zero_padding=False)):
            pass


def test_correct_encoding():
    """ Test correct amino acid to integer encoding. """
    # Default alphabet is ExtendedIUPACProtein mapped to [1,26]
    vocab = ds.gen_amino_acid_vocab()
    test_string = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
    test_encoded = [vocab[c] for c in test_string]
    for i, batch in enumerate(test_encoded):
        assert((i+1) == batch)


@pytest.mark.parametrize('batch_size', [1, 2, 10])
@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('buffer_size', [2, 10, 30])
@pytest.mark.parametrize('labels', [None, TRAINING_LABELS])
def test_shuffled_dataset(batch_size, num_workers, buffer_size, labels):
    dataset = ds.ShuffledProteinIterableDataset(TRAINING_FASTA,
                                                labels_file=labels,
                                                buffer_size=buffer_size)
    observed_ids = list()
    for i, batch in enumerate(DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=ds.collate_sequences)):
        for id_ in batch.ids:
            observed_ids.append(id_)

    observed_ids_set = set(observed_ids)
    assert len(observed_ids) == len(observed_ids_set), 'Sequences lost or duplicated during shuffle'

    if dataset.labels is not None:
        expected = EXPECTED_IDS_WITH_LABEL
    else:
        expected = EXPECTED_IDS
    intersection = observed_ids_set.intersection(set(expected))
    assert len(intersection) == len(expected), \
        'Observed protein IDs do not match the expected ID list'


def test_rename_cols_in_iterable_dataset():
    df = read_csv(LABELS_WRONG_COL_NAMES, index_col=0)
    assert 'eggnog_id' not in df.columns
    assert 'protein_id' not in df.columns

    dataset = ds.ProteinIterableDataset(test_file, labels_file=str(LABELS_WRONG_COL_NAMES))
    assert 'eggnog_id' in dataset.labels.columns
    assert 'protein_id' in dataset.labels.columns


def test_protein_dataset():
    dataset = ds.ProteinDataset(TRAINING_FASTA, labels=None)
    assert not dataset.label_from_id

    df = read_csv(TRAINING_LABELS, index_col=0)
    dataset = ds.ProteinDataset(TRAINING_FASTA, labels=df)
    assert dataset.labels is df

    wrong_labels = np.random.randint(0, 10, len(df))
    with pytest.raises(ValueError, match='Invalid labels'):
        _ = ds.ProteinDataset(TRAINING_FASTA, labels=wrong_labels)

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df['eggnog_id'].values[:22])  # omit last class
    dataset = ds.ProteinDataset(TRAINING_FASTA,
                                labels=TRAINING_LABELS,
                                label_encoder=label_encoder)
    assert dataset.labels.shape[0] == 22
    assert all(dataset.labels.label_num.value_counts() == [11, 11])

    seq_records = list(repeat(SeqRecord(Seq('MATTAC'), id='seq1', name='seq1'), 22))
    dataset = ds.ProteinDataset(seq_records, labels=TRAINING_LABELS)
    for i in range(22):
        assert dataset[i].id == 'seq1'
        assert dataset[i].index == i
        assert dataset[i].string == 'MATTAC'
        assert dataset[i].label is None
        assert len(dataset[i].encoded) == 6

    # list(repeat(...)) w/o times: traust di nie
    sequences = list(repeat(Seq('MATTAC'), times=10))
    with pytest.raises(ValueError, match="must be FASTA file or a list/tuple "
                                         "of <class 'Bio.SeqRecord.SeqRecord'>"):
        _ = ds.ProteinDataset(sequences, labels=TRAINING_LABELS)
