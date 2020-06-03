"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

     Various utility functions
"""
# SPDX-License-Identifier: BSD-3-Clause
from importlib import import_module
import sys
from typing import Union
import warnings

import torch


__all__ = ['EXTENDED_IUPAC_PROTEIN_ALPHABET',
           'set_device',
           'count_parameters',
           'eprint',
           'load_nn',
           'SeqIO',
           ]

# Bio.Alphabet.ExtendendIUPACProtein (deprecated in 2020)
EXTENDED_IUPAC_PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'

# Biopython warns about Alphabet, even if you don't use Alphabet...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    from Bio import SeqIO


def count_parameters(model):
    """ Count the number of trainable parameters in model.

    Reference
    ---------
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_device(device: Union[str, torch.device]) -> torch.device:
    """ Set device (CPU/GPU) depending on user choice and availability.

    Parameters
    ----------
    device : [str, torch.device]
        Device set by user as an argument to DeepNOG call.

    Returns
    -------
    device : torch.device
        Object containing the device type to be used for prediction
        calculations.
    """
    if isinstance(device, torch.device):
        pass
    elif device == 'auto':
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


def load_nn(architecture, model_dict=None, phase='eval', device='cpu'):
    """ Import NN architecture and set loaded parameters.

    Parameters
    ----------
    architecture : str
        Name of neural network module and class to import.
    model_dict : dict, optional
        Dictionary holding all parameters and hyper-parameters of the model.
        Required during inference, optional for training.
    phase : ['train', 'eval']
        Set network in training or evaluation mode with effects on
        storing gradients, dropout, etc.
    device : [str, torch.device]
        Device to load the model into.

    Returns
    -------
    model : torch.nn.Module
        Neural network object of type architecture with parameters
        loaded from model_dict and moved to device.
    """
    # Import and instantiate neural network class
    model_module = import_module(f'.models.{architecture}', 'deepnog')
    model_class = getattr(model_module, architecture)
    model = model_class(model_dict)
    # Set trained parameters of model
    if model_dict is not None:
        model.load_state_dict(model_dict['model_state_dict'])
    # Move to GPU, if selected
    model.to(device)
    # Inform neural network layers to be in evaluation or training mode
    if phase.lower().startswith('train'):
        model = model.train()
    elif phase.lower().startswith('eval'):
        model = model.eval()
    else:
        raise ValueError(f'Unknown phase "{phase}". '
                         f'Must be "train", or "eval".')
    return model


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
