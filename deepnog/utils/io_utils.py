"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Input/output helper functions
"""
# SPDX-License-Identifier: BSD-3-Clause
import logging as log
from os import environ
from pathlib import Path
import shutil
from typing import List
from urllib.request import urlopen
from urllib.parse import urljoin
import warnings

import pandas as pd
from torch import Tensor

__all__ = ['create_df',
           'get_data_home',
           'get_logger',
           'get_weights_path',
           'init_global_logger',
           'logging',
           ]

DEEPNOG_REMOTE_DEFAULT = ('https://fileshare.csb.univie.ac.at/'
                          'deepnog/parameters/')

log.addLevelName(log.DEBUG, "\033[1;32m%s\033[1;0m" % log.getLevelName(log.DEBUG))
log.addLevelName(log.INFO, "\033[1;34m%s\033[1;0m" % log.getLevelName(log.INFO))
log.addLevelName(log.WARNING, "\033[1;33m%s\033[1;0m" % log.getLevelName(log.WARNING))
log.addLevelName(log.ERROR, "\033[1;41m%s\033[1;0m" % log.getLevelName(log.ERROR))

global logging


def init_global_logger(logger_name, verbose):
    global logging
    logging = get_logger(logger_name, verbose)


def get_logger(initname: str = 'deepnog', verbose: int = 0) -> log.Logger:
    """
    This function provides a nicely formatted logger.

    Parameters
    ----------
    initname : str
        The name of the logger to show up in log.
    verbose : int
        Increasing levels of verbosity

    References
    ----------
    Shamelessly stolen from phenotrex
    """
    logger = log.getLogger(initname)
    if type(verbose) is bool:
        logger.setLevel(log.INFO if verbose else log.WARNING)
    else:
        logger.setLevel(verbose)
    ch = log.StreamHandler()
    if verbose <= 0:
        ch.setLevel(log.ERROR)
    elif verbose == 1:
        ch.setLevel(log.WARNING)
    elif verbose == 2:
        ch.setLevel(log.INFO)
    else:
        ch.setLevel(log.DEBUG)
    logstring = ('\033[1;32m[%(asctime)s]\033[1;0m \033[1m%(name)s'
                 '\033[1;0m - %(levelname)s - %(message)s')
    formatter = log.Formatter(logstring, '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger


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


def get_data_home(data_home: str = None) -> Path:
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

    Notes
    -----
    Adapted from SKLEARN_DATAHOME_.

    .. _SKLEARN_DATAHOME: https://github.com/scikit-learn/scikit-learn/blob/0.22.X/sklearn/datasets/_base.py
    """  # noqa
    if data_home is None:
        data_home = environ.get('DEEPNOG_DATA',
                                Path.home()/'deepnog_data')
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    logging.debug(f'DEEPNOG_DATA = {data_home}')
    return data_home


def get_weights_path(database: str, level: str, architecture: str,
                     data_home=None, download_if_missing=True) -> Path:
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
        Network architecture. Example: deepencoding
    data_home : string, optional
        Specify another download and cache folder for the weights.
        By default all deepnog data is stored in '~/deepnog_data' subfolders.
    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    weights_path : Path
        Path to file of network weights
    """
    data_home = get_data_home(data_home=data_home)
    weights_dir = data_home/f'{database}/{level}/'
    weights_file = weights_dir/f'{architecture}.pth'
    available = weights_file.exists()

    if not available and download_if_missing:
        weights_dir.mkdir(parents=True, exist_ok=True)
        remote_url = urljoin(
            environ.get('DEEPNOG_REMOTE', DEEPNOG_REMOTE_DEFAULT),
            f"{database}/{level}/{architecture}.pth")
        logging.info(f"Downloading {remote_url}")
        with urlopen(remote_url) as response, weights_file.open('wb') as f:
            logging.info(f'Saving to {weights_file}')
            shutil.copyfileobj(response, f)
    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")

    return weights_file


# Always initialize here, so that ``logging`` is always available
init_global_logger('deepnog', verbose=0)
