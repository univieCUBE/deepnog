"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Input/output helper functions
"""
# SPDX-License-Identifier: BSD-3-Clause
from os import environ
from pathlib import Path
import shutil
from typing import List
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import urljoin
import warnings

import pandas as pd

from deepnog.utils.logger import get_logger
from .imports import try_import_pytorch

torch = try_import_pytorch()
from torch import Tensor  # noqa

__all__ = ['create_df',
           'get_data_home',
           'get_weights_path',
           ]

DEEPNOG_REMOTE_DEFAULT = ('https://fileshare.csb.univie.ac.at/'
                          'deepnog/parameters/')


def create_df(class_labels: list, preds: Tensor, confs: Tensor, ids: List[str],
              indices: List[int], threshold: float = None):
    """ Creates one dataframe storing all relevant prediction information.

    The rows in the returned dataframe have the same order as the
    original sequences in the data file. First column of the dataframe
    represents the position of the sequence in the datafile.

    Parameters
    ----------
    class_labels : list
        Store class name corresponding to an output node of the network.
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
    threshold : float
        If given, prediction labels and confidences are set to '' if
        confidence in prediction is not at least threshold.

    Returns
    -------
    df : pandas.DataFrame
        Stores prediction information about the input protein sequences.
        Duplicates (defined by their sequence_id) have been removed from df.
    """
    confs = confs.cpu().numpy()
    if threshold is not None:
        # Set empty label and confidence if prediction confidence below
        # threshold of model
        labels = [class_labels[pred] if conf >= threshold
                  else '' for pred, conf in zip(preds, confs)]
        confs = [str(conf) if conf >= threshold
                 else '' for conf in confs]
    else:
        labels = [class_labels[int(pred)] for pred in preds]
    # Create prediction results frame
    df = pd.DataFrame(data={'index': indices,
                            'sequence_id': ids,
                            'prediction': labels,
                            'confidence': confs})
    df.sort_values(by='index', axis=0, inplace=True)
    # Remove duplicate sequences
    duplicate_mask = df.sequence_id.duplicated(keep='first')
    n_duplicates = sum(duplicate_mask)
    if n_duplicates > 0:
        warnings.warn(f'Detected {n_duplicates} duplicate sequences based on '
                      f'their extracted sequence id. Keeping the first '
                      f'sequence among the duplicates when writing prediction '
                      f'output file.')
        df = df[~duplicate_mask]
    return df


def get_data_home(data_home: str = None, verbose: int = 0) -> Path:
    """Return the path of the deepnog data dir.

    This folder is used for large files that cannot go into the Python package
    on PyPI etc. For example, the network parameters (weights) files may be
    larger than 100MiB.
    By default the data dir is set to a folder named 'deepnog_data' in the
    user home folder.
    Alternatively, it can be set by the 'DEEPNOG_DATA' environment
    variable or programmatically by giving an explicit folder path.
    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to deepnog data dir.
    verbose : int
        Log or not.

    Notes
    -----
    Adapted from SKLEARN_DATAHOME_.

    .. _SKLEARN_DATAHOME: https://github.com/scikit-learn/scikit-learn/blob/0.22.X/sklearn/datasets/_base.py
    """  # noqa
    logger = get_logger(__name__, verbose=verbose)
    if data_home is None:
        data_home = environ.get('DEEPNOG_DATA',
                                Path.home()/'deepnog_data')
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    logger.debug(f'DEEPNOG_DATA = {data_home}')
    return data_home


def get_weights_path(database: str, level: str, architecture: str,
                     data_home: str = None, download_if_missing: bool = True,
                     verbose: int = 0) -> Path:
    """ Get path to neural network weights.

    This is a path on local storage. If the corresponding files are not
    present, download from remote storage. The default remote URL can be
    overridden by setting the environment variable DEEPNOG_REMOTE.

    Parameters
    ----------
    database : str
        The orthologous groups database. Example: eggNOG5
    level: str
        The taxonomic level within the database. Example: 2 (for bacteria)
    architecture: str
        Network architecture. Example: deepnog
    data_home : str, optional
        Specify another download and cache folder for the weights.
        By default all deepnog data is stored in '$HOME/deepnog_data' subfolders.
    download_if_missing : boolean, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.
    verbose : int
        Log or not

    Returns
    -------
    weights_path : Path
        Path to file of network weights
    """
    logger = get_logger(__name__, verbose=verbose)
    data_home = get_data_home(data_home=data_home)
    weights_dir = data_home/f'{database}'/f'{level}'
    weights_file = weights_dir/f'{architecture}.pth'
    available = weights_file.exists()
    logger.debug(f'Weights file available: {weights_file}')

    model_name = f"{database} (tax {level}) {architecture} model"
    if not available and download_if_missing:
        weights_dir.mkdir(parents=True, exist_ok=True)
        remote_url = urljoin(
            environ.get('DEEPNOG_REMOTE', DEEPNOG_REMOTE_DEFAULT),
            f"{database}/{level}/{architecture}.pth")
        logger.info(f"{model_name}: Downloading {remote_url}")
        try:
            with urlopen(remote_url) as response, weights_file.open('wb') as f:
                logger.info(f'Saving to {weights_file}')
                shutil.copyfileobj(response, f)
        except URLError as e:
            logger.error(f'{model_name}: Download failed. '
                         f'Try downloading {remote_url} and '
                         f'saving to {weights_file} manually.\nGot error:\n{e}')
    elif not available and not download_if_missing:
        raise IOError(f"{model_name}: Data not found and "
                      f"`download_if_missing` is False")

    return weights_file
