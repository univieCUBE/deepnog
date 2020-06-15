"""
Author: Lukas Gosch
Date: 2019-10-03
Description:
    Test dataset module.
"""
from pathlib import Path
import pytest

from torch.utils.data import DataLoader
from deepnog.data import dataset as ds

test_file = Path(__file__).parent.absolute() / "data/GCF_000007025.1.faa"
test_file_gzip = Path(__file__).parent.absolute()/"data/GCF_000007025.1.faa.gz"
TRAINING_FASTA = Path(__file__).parent.absolute()/"data/test_training_dummy.faa"
TRAINING_LABELS = Path(__file__).parent.absolute()/"data/test_training_dummy.faa.csv"
EXPECTED_IDS_WITH_LABEL = [f'test_all_A{x}' for x in range(11)] \
                          + [f'test_all_C{x}' for x in range(11)] \
                          + [f'M{x:02d}' for x in range(1, 9)]
EXPECTED_IDS = [f'test_all_A{x}' for x in range(12)] \
               + [f'test_all_C{x}' for x in range(12)] \
               + [f'M{x:02d}' for x in range(1, 11)]


@pytest.mark.parametrize("f", [test_file, test_file_gzip, ])
@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_multiprocess_data_loading(f, num_workers, f_format='fasta'):
    """ Test if different workers produce different sequences and process
        whole sequence file.
    """
    dataset = ds.ProteinDataset(f, f_format=f_format)
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
    dataset = ds.ProteinDataset(test_file, f_format=f_format)
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


def test_zero_padding(f_format='fasta'):
    """ Test correct zeroPadding. """
    pad_file = Path(__file__).parent.absolute()/"data/test_zeroPadding.faa"
    dataset = ds.ProteinDataset(pad_file, f_format=f_format)
    for i, batch in enumerate(DataLoader(dataset,
                                         batch_size=2,
                                         num_workers=0,
                                         collate_fn=ds.collate_sequences)):
        # Test correct shape
        assert(batch.sequences.shape[1] == 112)
        # Test correctly zeros inserted
        assert(sum(batch.sequences[0, 56:]) == 0)


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
    dataset = ds.ShuffledProteinDataset(TRAINING_FASTA,
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
