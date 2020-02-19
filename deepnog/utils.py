"""
Author: Roman Feldbauer
Date: 2020-02-19
"""
# SPDX-License-Identifier: BSD-3-Clause
import torch


__all__ = ['EXTENDED_IUPAC_PROTEIN_ALPHABET',
           'set_device',
           ]

# Bio.Alphabet.ExtendendIUPACProtein (deprecated in 2020)
EXTENDED_IUPAC_PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'


def set_device(user_choice):
    """ Sets calc. device depending on users choices and availability.

    Parameters
    ----------
    user_choice : str
        Device set by user as an argument to DeepNOG call.

    Returns
    -------
    device : torch.device
        Object containing the device type to be used for prediction
        calculations.
    """
    if user_choice == 'auto':
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
    elif user_choice == 'gpu':
        cuda = torch.cuda.is_available()
        if cuda:
            device = torch.device('cuda')
        else:
            raise RuntimeError('Device set to gpu but no cuda-enabled gpu '
                               'available on machine!')
    else:
        device = torch.device('cpu')
    return device
