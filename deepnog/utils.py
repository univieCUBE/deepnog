"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

     Various utility functions
"""
# SPDX-License-Identifier: BSD-3-Clause
import warnings

import torch


__all__ = ['EXTENDED_IUPAC_PROTEIN_ALPHABET',
           'set_device',
           'SeqIO',
           ]

# Bio.Alphabet.ExtendendIUPACProtein (deprecated in 2020)
EXTENDED_IUPAC_PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'

# Biopython warns about Alphabet, even if you don't use Alphabet...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    from Bio import SeqIO


def set_device(device):
    """ Set device (CPU/GPU) depending on user choice and availability.

    Parameters
    ----------
    device : str
        Device set by user as an argument to DeepNOG call.

    Returns
    -------
    device : torch.device
        Object containing the device type to be used for prediction
        calculations.
    """
    if device == 'auto':
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
    elif device == 'gpu':
        cuda = torch.cuda.is_available()
        if cuda:
            device = torch.device('cuda')
        else:
            raise RuntimeError('Device set to gpu but no cuda-enabled gpu '
                               'is available on this machine.')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(f'Unknown device "{device}". Try "auto".')
    return device
