"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

     Various utility functions
"""
# SPDX-License-Identifier: BSD-3-Clause
from importlib import import_module
from pathlib import Path
from typing import Sequence, Union

from . import get_logger
from . import try_import_pytorch

torch = try_import_pytorch()

__all__ = ['set_device',
           'count_parameters',
           'load_nn',
           ]


def count_parameters(model, tunable_only: bool = True) -> int:
    """ Count the number of parameters in the given model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model (deep network)
    tunable_only : bool, optional
        Count only tunable network parameters

    References
    ----------
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    if tunable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


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
    err_msg = None
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
            err_msg = ('Device set to "gpu", but could not access '
                       'any CUDA-enabled GPU. Please make sure that '
                       'a GPU is available and CUDA is installed '
                       'on this machine.')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        err_msg = f'Unknown device "{device}". Try "auto".'
    if err_msg is not None:
        logger = get_logger(__name__, verbose=0)
        logger.error(f'Unknown device "{device}". Try "auto".')
        import sys
        sys.exit(1)
    return device


def load_nn(architecture: Union[str, Sequence[str]], model_dict: dict = None, phase: str = 'eval',
            device: Union[torch.device, str] = 'cpu', verbose: int = 0):
    """ Import NN architecture and set loaded parameters.

    Parameters
    ----------
    architecture : str or list-like of two str
        If single string: name of neural network module and class to import.
        E.g. 'deepnog' will load deepnog.models.deepnog.deepnog.
        Otherwise, separate module and class name of deep network to import.
        E.g. ('deepthought', 'DeepNettigkeit') will load deepnog.models.deepthought.DeepNettigkeit.
    model_dict : dict, optional
        Dictionary holding all parameters and hyper-parameters of the model.
        Required during inference, optional for training.
    phase : ['train', 'infer', 'eval']
        Set network in training or inference=evaluation mode with effects on
        storing gradients, dropout, etc.
    device : [str, torch.device]
        Device to load the model into.
    verbose : int
        Increasingly verbose logging

    Returns
    -------
    model : torch.nn.Module
        Neural network object of type architecture with parameters
        loaded from model_dict and moved to device.
    """
    logger = get_logger(__name__, verbose=verbose)

    if isinstance(architecture, (str, Path)):
        module = str(architecture)
        cls = str(architecture)
    else:
        module, cls = [str(x) for x in architecture]
    # Import and instantiate neural network class
    model_module = import_module(f'.models.{module}', 'deepnog')
    model_class = getattr(model_module, cls)
    model = model_class(model_dict)
    # Set trained parameters of model
    try:
        model.load_state_dict(model_dict['model_state_dict'])
        logger.debug('Loaded trained network weights.')
    except KeyError as e:
        logger.debug('Did not load any trained network weights.')
        if not phase.lower().startswith('train'):
            raise RuntimeError('No trained weights available '
                               'during inference.') from e
    # Move to GPU, if selected
    model.to(device)
    # Inform neural network layers to be in evaluation or training mode
    if phase.lower().startswith('train'):
        logger.debug('Setting model.train() mode')
        model = model.train()
    elif phase.lower().startswith('eval') or phase.lower().startswith('infer'):
        logger.debug('Setting model.eval() mode')
        model = model.eval()
    else:
        raise ValueError(f'Unknown phase "{phase}". '
                         f'Must be "train", "infer", or "eval".')
    return model
