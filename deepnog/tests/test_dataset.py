"""
Author: Lukas Gosch
Date: 3.10.2019
Description:
    Test dataset module.
"""

from torch.utils.data import DataLoader
import pytest
import os
import sys
import inspect

from .. import dataset as ds


class TestDataset:
    """ Class grouping tests for dataset module. """

    @pytest.mark.parametrize("num_workers", [None, 2, 3, 4])
    def test_multiprocessDataLoading(self, num_workers, f_format='fasta'):
        """ Test if different workers produce different sequences and process
            whole sequence file.
        """
        test_file = "./tests/data/GCF_000007025.1.faa"
        dataset = ds.ProteinDataset(test_file, f_format=f_format)
        s = set()
        for i, batch in enumerate(DataLoader(dataset, num_workers=0,
                                             batch_size=1,
                                             collate_fn=ds.collate_sequences)):
            # Check uniqueness of IDs for loaded data
            for identifier in batch.ids:
                assert(identifier not in s)
                s.add(identifier)
        # Check if all data was loaded
        assert(i == 1288)

    @pytest.mark.parametrize("batch_size", [None, 1, 16, 32])
    def test_correctCollatingSequences(self, batch_size, f_format='fasta'):
        """ Test if a batch of correct size is produced. """
        test_file = "./tests/data/GCF_000007025.1.faa"
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

    def test_zeroPadding(self, f_format='fasta'):
        """ Test correct zeroPadding. """
        test_file = "./tests/data/test_zeroPadding.faa"
        dataset = ds.ProteinDataset(test_file, f_format=f_format)
        for i, batch in enumerate(DataLoader(dataset,
                                             batch_size=2,
                                             num_workers=0,
                                             collate_fn=ds.collate_sequences)):
            # Test correct shape
            assert(batch.sequences.shape[1] == 112)
            # Test correctly zeros inserted
            assert(sum(batch.sequences[0, 56:]) == 0)

    def test_correctEncoding(self, f_format='fasta'):
        """ Test correct amino acid to integer encoding. """
        # Default alphabet is ExtendedIUPACProtein mapped to [1,26]
        vocab = ds.gen_amino_acid_vocab()
        test_string = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
        test_encoded = [vocab[c] for c in test_string]
        for i, batch in enumerate(test_encoded):
            assert((i+1) == batch)
