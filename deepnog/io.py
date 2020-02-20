"""
Author: Roman Feldbauer
Date: 2020-02-19
"""
# SPDX-License-Identifier: BSD-3-Clause
from os import environ
from pathlib import Path
import sys

import pandas as pd


def create_df(class_labels, preds, confs, ids, indices, threshold=None,
              verbose=3):
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
    threshold : int
        If given, prediction labels and confidences are set to '' if
        confidence in prediction is not at least threshold.
    verbose : int
        If bigger 0, outputs warning if duplicates detected.

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
        labels = [class_labels[pred] for pred in preds]
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
        if verbose > 0:
            print(f'WARNING: Detected {n_duplicates} duplicate sequences '
                  f'based on their extracted sequence id. Keeping the first '
                  f'sequence among the duplicates when writing prediction '
                  f'output file.',
                  file=sys.stderr)
        df = df[~duplicate_mask]
    return df


def get_data_home(data_home=None):
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
    Adapted from `https://github.com/scikit-learn/scikit-learn/blob/0.22.X/\
    sklearn/datasets/_base.py`_
    """
    if data_home is None:
        data_home = environ.get('DEEPNOG_DATA',
                                Path.home()/'scikit_learn_data')
    data_home = Path(data_home).expanduser()
    if not data_home.exists():
        data_home.mkdir(parents=True)
    return data_home
