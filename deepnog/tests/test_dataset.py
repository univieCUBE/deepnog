"""
Author: Lukas Gosch
Date: 2019-10-03
Description:
    Test dataset module.
"""
from collections import namedtuple
import os
from pathlib import Path
import pytest

from torch.utils.data import DataLoader
from deepnog import dataset as ds
from deepnog import sync

test_file = Path(__file__).parent.absolute() / "data/GCF_000007025.1.faa"
test_file_gzip = Path(__file__).parent.absolute()/"data/GCF_000007025.1.faa.gz"


@pytest.mark.parametrize("f", [test_file, test_file_gzip, ])
@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_multiprocess_data_loading(f, num_workers, f_format='fasta'):
    """ Test if different workers produce different sequences and process
        whole sequence file.
    """
    # Need to set up the pipes here, as otherwise done in inference.predict()
    sync.init()
    for pipes in range(num_workers):
        r, w = os.pipe()
        os.set_inheritable(r, True)  # Compatibility with Windows
        os.set_inheritable(w, True)  # Compatibility with Windows
        sync.rpipe_l.append(r)
        sync.wpipe_l.append(w)

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
