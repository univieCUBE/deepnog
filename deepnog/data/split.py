from dataclasses import dataclass
from typing import Union
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..utils import get_logger


@dataclass
class DataSplit:
    """ Class for returned data, labels, and groups after train/val/test split. """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame
    uniref_train: Union[pd.DataFrame, None]
    uniref_val: Union[pd.DataFrame, None]
    uniref_test: Union[pd.DataFrame, None]


def train_val_test_split(df: pd.DataFrame,
                         train_ratio: float = 0.96,
                         validation_ratio: float = 0.02,
                         test_ratio: float = 0.02,
                         stratify: bool = True,
                         shuffle: bool = True,
                         random_state: int = 123,
                         verbose: int = 0) -> DataSplit:
    """ Create training/validation/test split for deepnog experiments.

    Does not take UniRef clusters into account. Do not use for UniRef50/90
    experiments.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain 'string_id', 'eggnog_id' columns
    train_ratio : float
        Fraction of total sequences for training set
    validation_ratio : float
        Fraction of total sequences for validation set
    test_ratio : float
        Fractino of total sequences for test set
    stratify : bool
        Stratify the splits according to the orthology labels
    shuffle : bool
        Shuffle the sequences
    random_state : int
        Set random state for reproducible results
    verbose : int
        Level of logging verbosity

    Returns
    -------
    data_split : DataSplit
        Split X, y, groups
    """
    logger = get_logger(__name__, verbose=verbose)
    total = train_ratio + validation_ratio + test_ratio
    if abs(1. - total) > 1e-6:
        train_ratio /= total
        validation_ratio /= total
        test_ratio /= total
        logger.warning(f'Ratios do not sum to 1. Normalizing to train/val/test = '
                       f'{train_ratio:.3f}/{validation_ratio:.3f}/{test_ratio:.3f}')

    test_size = 1 - train_ratio

    X_train, X_test, y_train, y_test = \
        train_test_split(df.string_id,
                         df.eggnog_id,
                         test_size=test_size,
                         shuffle=shuffle,
                         random_state=random_state,
                         stratify=df.eggnog_id if stratify else None,
                         )

    X_val, X_test, y_val, y_test = \
        train_test_split(X_test,
                         y_test,
                         test_size=test_ratio / (test_ratio + validation_ratio),
                         stratify=y_test if stratify else None,
                         )
    return DataSplit(X_train=X_train,
                     X_val=X_val,
                     X_test=X_test,
                     y_train=y_train,
                     y_val=y_val,
                     y_test=y_test,
                     uniref_train=None,
                     uniref_val=None,
                     uniref_test=None)


def group_train_val_test_split(df: pd.DataFrame,
                               train_ratio: float = 0.96,
                               validation_ratio: float = 0.02,
                               test_ratio: float = 0.02,
                               random_state: int = 123,
                               with_replacement: bool = True,
                               verbose: int = 0) -> DataSplit:
    """ Create training/validation/test split for deepnog experiments.

    Takes UniRef cluster IDs into account, that is, makes sure that
    sequences from the same cluster go into the same set. In other
    words, training, validation, and test sets are disjunct in terms
    of UniRef clusters.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain 'string_id', 'eggnog_id', 'uniref_id' columns
    train_ratio : float
        Fraction of total sequences for training set
    validation_ratio : float
        Fraction of total sequences for validation set
    test_ratio : float
        Fraction of total sequences for test set
    random_state : int
        Set random state for reproducible results
    with_replacement : bool
        By default, scikit-learn GroupShuffleSplit samples objects with
        replacement. Disabling replacement removes
    verbose : int
        Level of logging verbosity

    Returns
    -------
    data_split : NamedTuple
        Split X, y, groups
    """
    logger = get_logger(__name__, verbose=verbose)
    total = train_ratio + validation_ratio + test_ratio
    if abs(1. - total) > 1e-6:
        train_ratio /= total
        validation_ratio /= total
        test_ratio /= total
        logger.warning(f'Ratios do not sum to 1. Normalizing to train/val/test = '
                       f'{train_ratio:.3f}/{validation_ratio:.3f}/{test_ratio:.3f}')

    test_size = 1. - train_ratio

    gss = GroupShuffleSplit(n_splits=1,
                            train_size=train_ratio,
                            test_size=test_size,
                            random_state=random_state,
                            )
    train_ind, test_ind = next(gss.split(df.string_id, df.eggnog_id, df.uniref_id))
    # MUST use iloc, or otherwise the dataframe index might be used,
    # but sklearn returns the numpy array indices
    X_train = df.string_id.iloc[train_ind]
    X_test = df.string_id.iloc[test_ind]
    y_train = df.eggnog_id.iloc[train_ind]
    y_test = df.eggnog_id.iloc[test_ind]
    uniref_train = df.uniref_id.iloc[train_ind]
    uniref_test = df.uniref_id.iloc[test_ind]

    gss = GroupShuffleSplit(n_splits=1,
                            test_size=test_ratio / (test_ratio + validation_ratio),
                            random_state=random_state,
                            )
    val_ind, test_ind = next(gss.split(X_test, y_test, uniref_test))
    X_val = X_test.iloc[val_ind]
    X_test = X_test.iloc[test_ind]
    y_val = y_test.iloc[val_ind]
    y_test = y_test.iloc[test_ind]
    uniref_val = uniref_test.iloc[val_ind]
    uniref_test = uniref_test.iloc[test_ind]

    return DataSplit(X_train=X_train,
                     X_val=X_val,
                     X_test=X_test,
                     y_train=y_train,
                     y_val=y_val,
                     y_test=y_test,
                     uniref_train=uniref_train,
                     uniref_val=uniref_val,
                     uniref_test=uniref_test)
