"""
Author: Lukas Gosch
Date: 5.9.2019
Description:
    Functions to preprocess proteins for classification.
"""
import os
from itertools import islice
import torch
from torch.utils.data import IterableDataset
from Bio import SeqIO
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein

class ProteinIterator():
    """ Iterator over sequence file used by ProteinDataset.

        ProteinIterator wraps the iterator returned by the Bio.SeqIO.parse()
        function. It makes sure that a unique ID is set in each SeqRecord 
        optained from the data-iterator. The id attribute in each SeqRecord 
        is prefixed by an index i which directly corresponds to the i-th 
        sequence in the sequence file. 

        Parameters
        ----------
        iterator
            Iterator over sequence file returned by Biopythons
            Bio.SeqIO.parse() function.
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        """ Return next SeqRecrod in iterator and prefix id. """
        next_seq = next(self.iterator)
        self.position += 1
        next_seq.id = f'{self.position}_{next_seq.id}'
        return next_seq

class ProteinDataset(IterableDataset):
    """ Protein dataset holding the proteins to classify. 
    
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

    def __init__(self, file, f_format = 'fasta', max_length = None,
                 zero_padding = True):
        """ Initialize iterator over sequences in file."""
        if os.path.isfile(file):
            self.iter = ProteinIterator(SeqIO.parse(file, format = f_format, 
                                        alphabet = ExtendedIUPACProtein()))
        else:
            raise ValueError('Given file does not exist or is not a file.')
        
    def __iter__(self):
        """ Return iterator over sequences in file. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.iter
        else:
            raise NotImplementedError('Multiprocess-dataloading not supported.')

def consume(iterator, n=None):
    """ Advance the iterator n-steps ahead. If n is None, consume entirely.

        Function from Itertools Recipes in official Python 3.7.4. docs.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

class MPProteinIterator():
    """ Iterator allowing for multiprocess dataloading of a sequence file. 
        
        MPProteinIterator is a wrapper for the iterator returned by 
        Biopythons Bio.SeqIO class when parsing a sequence file. It 
        specifies custom __next__() method to support multiprocess data 
        loading. It does so by each worker skipping num_worker - 1 data 
        samples for each call to __next__(). Furthermore, each worker skips
        worker_id data samples in the initialization.

        Note: As of now not used as multithreading breaks sequence order
              as defined by the sequence file.
    """
    def __init__(self, iterator, num_workers=1, worker_id=0):
        # Advance iterator worker_id times.
        self.iterator = iterator
        # Start position
        self.start = worker_id
        self.started = False
        # Number of sequences to skip for each next() call.
        self.step = num_workers - 1

    def __iter__(self):
        return self

    def __next__(self):
        """ Return element at current + step + 1 position or start position. """
        # Check if iterator has been positioned correctly.
        if self.started:
            consume(self.iterator, n=self.step)
        else:
            consume(self.iterator, n=self.start)
            self.started = True
        return next(self.iterator)        