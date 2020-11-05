"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Predict orthologous groups of protein sequences.
"""
# SPDX-License-Identifier: BSD-3-Clause
from os import environ
from typing import List
import warnings

from tqdm import tqdm

from ..data.dataset import collate_sequences
from ..utils import get_logger, try_import_pytorch

torch = try_import_pytorch()
from torch.utils.data import DataLoader  # noqa

__all__ = ['predict', ]


def predict(model, dataset, device='cpu', batch_size=16, num_workers=4,
            verbose=3) -> (torch.Tensor, torch.Tensor, List[str], List[str]):
    """ Use model to predict zero-indexed labels of dataset.

    Also handles communication with ProteinIterators used to load data to
    log how many sequences have been skipped due to having empty sequence ids.

    Parameters
    ----------
    model : nn.Module
        Trained neural network model.
    dataset : ProteinIterableDataset
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
    logger = get_logger(__name__, verbose=verbose)

    logger.info(f'Inference device: {device}')

    debug_force_mode = environ.get("DEEPNOG_FORCE_MODE", default=None)
    if debug_force_mode is not None and debug_force_mode.lower() == 'train':
        logger.warning('forcing model.train(), with possible effects on dropout and batchnorm')
        model.train()

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
    try:
        n_sequences = len(dataset)
    except TypeError:
        n_sequences = None

    with torch.no_grad():
        # Do prediction calculations
        disable_tqdm = verbose < 3
        with tqdm(desc='deepnog inference',
                  total=n_sequences,
                  mininterval=1.,
                  disable=disable_tqdm,
                  unit='seq',
                  unit_scale=True) as pbar:
            for i, batch in enumerate(data_loader,):
                # Push sequences on correct device
                sequences = batch.sequences.to(device)
                # Predict protein families
                output = model(sequences)
                output = model.softmax(output)

                conf, pred = torch.max(output, 1)
                # Store predictions
                pred_l.append(pred)
                conf_l.append(conf)
                ids.extend(batch.ids)
                indices.extend(batch.indices)
                # Update progress bar by batch size
                pbar.update(n=len(sequences))

    logger.info('Inference complete.')
    # Collect skipped-sequences messages from workers in the case of
    # multi-process data-loading
    n_skipped = dataset.n_skipped
    # Check if sequences were skipped due to empty id
    if n_skipped > 0:
        warnings.warn(f'Skipped {n_skipped} sequences as no sequence id '
                      f'could be detected.')
    if len(pred_l) == 0:
        logger.error('Skipped all sequences. No output will be provided. '
                     'Sequences might have had no sequence IDs in the '
                     'input file.')
        return None, None, None, None
    else:
        # Merge individual output tensors
        preds = torch.cat(pred_l)
        confs = torch.cat(conf_l)
        return preds, confs, ids, indices
