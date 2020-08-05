import pytest

import numpy as np
import pandas as pd

from deepnog.data import train_val_test_split, group_train_val_test_split
from deepnog.tests.utils import get_deepnog_root

DEEPNOG_ROOT = get_deepnog_root()
TESTS = DEEPNOG_ROOT/"tests"
DATAFRAME_GROUP = TESTS/"data/test_split.csv"
DATAFRAME_SIMPLE = TESTS/"data/test_simple_split.csv"


@pytest.mark.parametrize('ratio', [[.4, .3, .3], [4, 3, 3]])
def test_simple_split(ratio):
    df = pd.read_csv(DATAFRAME_SIMPLE)
    train, val, test = ratio
    res = train_val_test_split(df,
                               train_ratio=train,
                               validation_ratio=val,
                               test_ratio=test,
                               random_state=123,
                               stratify=True,
                               shuffle=True,
                               verbose=0)

    for group in [res.uniref_train, res.uniref_val, res.uniref_test]:
        assert group is None
    assert res.X_train.shape == (4, )
    assert res.X_val.shape == (3, )
    assert res.X_test.shape == (4, )
    assert res.y_train.size == 4
    assert res.y_val.size == 3
    assert res.y_test.size == 4

    # Check THERE ARE overlaps in groups
    tr = df.merge(res.X_train)
    va = df.merge(res.X_val)
    te = df.merge(res.X_test)
    n_leak = 0
    for a, b in ((tr, va),
                 (va, te),
                 (te, tr)):
        n_leak += np.intersect1d(a.uniref_id, b.uniref_id).size
    assert n_leak

    # Check no overlaps in objects
    assert np.intersect1d(res.X_train, res.X_val).size == 0
    assert np.intersect1d(res.X_val, res.X_test).size == 0
    assert np.intersect1d(res.X_test, res.X_train).size == 0


@pytest.mark.parametrize('ratio', [[.6, .2, .2], [6, 2, 2]])
def test_group_split(ratio):
    df = pd.read_csv(DATAFRAME_GROUP)
    train, val, test = ratio
    res = group_train_val_test_split(df,
                                     train_ratio=train,
                                     validation_ratio=val,
                                     test_ratio=test,
                                     random_state=123,
                                     verbose=0)
    # Check correct size (ratio determines groups, so objects may be off)
    n_groups = df.uniref_id.unique().size
    # Also check that split works, when ratios do not sum to 1.
    sum_weights = sum(ratio)
    assert res.uniref_train.unique().size / n_groups == train / sum_weights
    assert res.uniref_val.unique().size / n_groups == val / sum_weights
    assert res.uniref_test.unique().size / n_groups == test / sum_weights
    assert res.X_train.shape == (6, )
    assert res.X_val.shape == (3, )
    assert res.X_test.shape == (1, )
    assert res.y_train.size == 6
    assert res.y_val.size == 3
    assert res.y_test.size == 1

    # Check no overlaps in groups
    assert np.intersect1d(res.uniref_train, res.uniref_val).size == 0
    assert np.intersect1d(res.uniref_val, res.uniref_test).size == 0
    assert np.intersect1d(res.uniref_test, res.uniref_train).size == 0

    # Check no overlaps in objects
    assert np.intersect1d(res.X_train, res.X_val).size == 0
    assert np.intersect1d(res.X_val, res.X_test).size == 0
    assert np.intersect1d(res.X_test, res.X_train).size == 0
