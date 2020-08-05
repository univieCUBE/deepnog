import numpy as np
import pandas as pd

from deepnog.utils.metrics import estimate_performance


def test_prec_recall_accuracy():
    labels_0 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
    labels_1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    df_true = pd.DataFrame({'sequence_id': np.arange(10),
                            'label': labels_0})
    df_pred = pd.DataFrame({'sequence_id': np.arange(10),
                            'prediction': labels_1})
    res = estimate_performance(df_true=df_true,
                               df_pred=df_pred)
    assert res['accuracy'] == 0.7
    assert res['macro_precision'] == (5 / 8 + 1 / 1 + 1 / 1) / 3
    assert res['micro_precision'] == (5 + 1 + 1) / 10
    assert res['macro_recall'] == (5 / 5 + 1 / 4 + 1 / 1) / 3
    assert res['micro_recall'] == (5 + 1 + 1) / 10
