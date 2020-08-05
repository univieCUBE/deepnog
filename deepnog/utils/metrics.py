# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

__all__ = ['estimate_performance',
           ]


def _fix_column_names(df):
    # Sequence IDs and labels are assumed in named columns,
    # but if not, let's try a specific order and hope for the best
    try:
        df.label
    except AttributeError:
        df = df.rename(columns={df.columns[-1]: 'label'})
    try:
        df.sequence_id
    except AttributeError:
        df = df.rename(columns={df.columns[0]: 'sequence_id'})
    return df


def estimate_performance(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> Dict:
    """ Calculate various model performance measures.

    Parameters
    ----------
    df_true : pandas.DataFrame
        The ground truth labels. DataFrame must contain 'sequence_id'
        and 'label' columns.
    df_pred : pandas.DataFrame
        The predicted labels. DataFrame must contain 'sequence_id'
        and 'prediction' columns.

    Returns
    -------
    perf : dict
        Performance estimates:
            - macro_precision
            - micro_precision
            - macro_recall
            - micro_recall
            - macro_f1
            - micro_f1
            - accuracy
            - mcc
    """
    df_true = _fix_column_names(df_true)
    df = df_true.merge(df_pred, on='sequence_id', how='inner')
    y_true = df['label'].values.astype(str)
    y_pred = df['prediction'].values.astype(str)
    perf = dict()
    for average in ['macro', 'micro']:
        p, r, f, _ = precision_recall_fscore_support(y_true=y_true,
                                                     y_pred=y_pred,
                                                     beta=1.,  # F1 score
                                                     average=average)
        perf[f'{average}_precision'] = p
        perf[f'{average}_recall'] = r
        perf[f'{average}_f1'] = f
    perf['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred)
    perf['mcc'] = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    return perf
