"""
Author: Lukas Gosch
Date: 5.9.2019
Description:
    Test dataset module.
"""

#######
# TODO: Package project and replace encapsulated code with relative imports!
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from dataset import ProteinDataset
from dataset import collate_sequences
from dataset import AminoAcidWordEmbedding
#######

import pytest
from torch.utils.data import DataLoader

class TestDataset:
    """ Class grouping tests for dataset module. """

    @pytest.mark.parametrize("num_workers", [None, 2, 3, 4])
    def test_multiprocessDataLoading(self, num_workers, f_format='fasta'):
        """ Test if different workers produce different sequences and process
            whole sequence file.
        """
        test_file = "./tests/data/GCF_000007025.1.faa"
        dataset = ProteinDataset(test_file, f_format=f_format)
        s = set()
        for i, batch in enumerate(DataLoader(dataset, num_workers=0, 
                                          batch_size = 1, 
                                          collate_fn=collate_sequences)):
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
        dataset = ProteinDataset(test_file, f_format=f_format)
        for i, batch in enumerate(DataLoader(dataset, 
                                        batch_size=batch_size, 
                                        num_workers=4, 
                                        collate_fn=collate_sequences)):
            if batch_size is None:
                assert(batch.sequences.shape[0] == 1)
                assert(len(batch.ids) == 1)
            else:
                assert(batch.sequences.shape[0] == batch_size)
                assert(len(batch.ids) == batch_size)
            break

    def test_zeroPadding(self, f_format='fasta'):
        """ Test correct zeroPadding. """
        test_file = "./tests/data/test_zeroPadding.faa"
        dataset = ProteinDataset(test_file, f_format=f_format)
        for i, batch in enumerate(DataLoader(dataset, 
                                        batch_size=2, 
                                        num_workers=0, 
                                        collate_fn=collate_sequences)):
            # Test correct shape
            assert(batch.sequences.shape[1] == 112)
            # Test correctly zeros inserted
            assert(sum(batch.sequences[0,56:]) == 0)

    def test_correctEncoding(self, f_format='fasta'):
        """ Test correct amino acid to integer encoding. """
        # Default alphabet is ExtendedIUPACProtein mapped to [1,26]
        vocab = AminoAcidWordEmbedding.gen_amino_acid_vocab()
        test_string = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
        test_encoded = [vocab[c] for c in test_string]
        for i, batch in enumerate(test_encoded):
            assert((i+1) == batch)