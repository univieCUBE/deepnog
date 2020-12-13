from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

from deepnog.data import ProteinIterableDataset
from deepnog.learning import fit, predict
from deepnog.utils import create_df, get_config

DEEPNOG_ROOT = Path(__file__).parent.parent.parent.absolute()
TRAINING_FASTA = DEEPNOG_ROOT/"tests/data/test_training_dummy.faa"
TRAINING_CSV = DEEPNOG_ROOT/"tests/data/test_training_dummy.faa.csv"
DEEPNOG_CONFIG = DEEPNOG_ROOT/"config/deepnog_custom_config.yml"


@pytest.mark.parametrize('architecture', ['deepnog', ])
def test_fit_model_and_predict(architecture):
    """ Fit each DeepNOG model on the dummy data, and assert inference
        on the same training data gives perfect predictions.
    """
    with TemporaryDirectory(prefix='deepnog_pytest_') as d:
        config = get_config(DEEPNOG_CONFIG)
        module = config['architecture'][architecture]['module']
        cls = config['architecture'][architecture]['class']

        result = fit(architecture=architecture,
                     module=module,
                     cls=cls,
                     training_sequences=TRAINING_FASTA,
                     validation_sequences=TRAINING_FASTA,
                     training_labels=TRAINING_CSV,
                     validation_labels=TRAINING_CSV,
                     n_epochs=2,
                     shuffle=True,
                     tensorboard_dir=None,
                     random_seed=123,
                     config_file=DEEPNOG_CONFIG,
                     verbose=0,
                     out_dir=Path(d),
                     )

        dataset = ProteinIterableDataset(TRAINING_FASTA, TRAINING_CSV, )
        preds, confs, ids, indices = predict(result.model,
                                             dataset,
                                             num_workers=0,
                                             verbose=0)
        df_pred = create_df(dataset.label_encoder.classes_,
                            preds, confs, ids, indices,
                            threshold=1e-15)
        df_true = pd.read_csv(TRAINING_CSV)
        df = df_true.merge(df_pred,
                           left_on="protein_id",
                           right_on="sequence_id")
        np.testing.assert_array_equal(df.prediction, df.eggnog_id)
