"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Predict orthologous groups of protein sequences.
"""
# SPDX-License-Identifier: BSD-3-Clause
from importlib import import_module
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import collate_sequences

__all__ = ['load_nn', 'predict', ]


def load_nn(architecture, model_dict, device='cpu'):
    """ Import NN architecture and set loaded parameters.

    Parameters
    ----------
    architecture : str
        Name of neural network module and class to import.
    model_dict : dict
        Dictionary holding all parameters and hyper-parameters of the model.
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
    model.load_state_dict(model_dict['model_state_dict'])
    # Move to GPU, if available
    model.to(device)
    # Inform neural network layers to be in evaluation mode
    model = model.eval()
    return model


def predict(model, dataset, device='cpu', batch_size=16, num_workers=4,
            verbose=3):
    """ Use model to predict zero-indexed labels of dataset.

    Also handles communication with ProteinIterators used to load data to
    log how many sequences have been skipped due to having empty sequence ids.

    Parameters
    ----------
    model : nn.Module
        Trained neural network model.
    dataset : ProteinDataset
        Data to predict protein families for.
    device : [str, torch.device]
        Device of model.
    batch_size : int
        Forward batch_size proteins through neural network at once.
    num_workers : int
        Number of workers for data loading.
    verbose : int
        Define verbosity.

    Returns
    -------
    preds : torch.Tensor, shape (n_samples,)
        Stores the index of the output-node with the highest activation
    confs : torch.Tensor, shape (n_samples,)
        Stores the confidence in the prediction
    ids : list[str]
        Stores the (possible empty) protein labels extracted from data
        file.
    indices : list[int]
        Stores the unique indices of sequences mapping to their position
        in the file
    """
    pred_l = []
    conf_l = []
    ids = []
    indices = []

    if num_workers < 2:
        num_workers = 0
    # Create data-loader for protein dataset
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_sequences,
                             )
    # Disable tracking of gradients to increase performance
    with torch.no_grad():
        # Do prediction calculations
        disable_tqdm = verbose < 3
        for i, batch in enumerate(tqdm(data_loader,
                                       disable=disable_tqdm,
                                       desc="Predicting batch")):
            # Push sequences on correct device
            sequences = batch.sequences.to(device)
            # Predict protein families
            output = model(sequences)
            conf, pred = torch.max(output, 1)
            # Store predictions
            pred_l.append(pred)
            conf_l.append(conf)
            ids.extend(batch.ids)
            indices.extend(batch.indices)

    # Collect skipped-sequences messages from workers in the case of
    # multi-process data-loading
    n_skipped = dataset.n_skipped
    # Check if sequences were skipped due to empty id
    if verbose > 0 and n_skipped > 0:
        warnings.warn(f'Skipped {n_skipped} sequences as no sequence id '
                      f'could be detected.')
    if len(pred_l) == 0:
        warnings.warn(f'Skipped all sequences. No output will be provided.')
        return None, None, None, None
    else:
        # Merge individual output tensors
        preds = torch.cat(pred_l)
        confs = torch.cat(conf_l)
        return preds, confs, ids, indices
