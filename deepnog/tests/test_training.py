from pathlib import Path
import tempfile
import pytest
import numpy as np
from deepnog.learning import fit

TRAINING_FASTA = Path(__file__).parent.absolute()/"data/test_training_dummy.faa"
TRAINING_CSV = Path(__file__).parent.absolute()/"data/test_training_dummy.faa.csv"
Y_TRUE = np.array([[0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
                    1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
                    1, 1, 1, 1, 1, 1, 1, 1]])


@pytest.mark.parametrize('batch_size', [4, ])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_shuffled_training(batch_size, num_workers):
    results = fit(architecture='deepencoding',
                  training_sequences=TRAINING_FASTA,
                  validation_sequences=TRAINING_FASTA,
                  training_labels=TRAINING_CSV,
                  validation_labels=TRAINING_CSV,
                  data_loader_params={'batch_size': batch_size,
                                      'num_workers': num_workers},
                  learning_rate=1e-3,
                  device='cpu',
                  verbose=0,
                  n_epochs=2,
                  shuffle=True,
                  random_seed=1,
                  tensorboard_dir=None,
                  save_each_epoch=True,
                  out_dir=Path(tempfile.mkdtemp()),
                  experiment_name='deepnog_test_epoch_save'
                  )
    for attr in ['model', 'training_dataset', 'validation_dataset', 'evaluation',
                 'y_train_true', 'y_train_pred', 'y_val_true', 'y_val_pred']:
        assert hasattr(results, attr)
    for x in [results.y_train_true, results.y_train_pred, results.y_val_true, results.y_val_pred]:
        assert x.shape == (2, 30)
    np.testing.assert_equal(results.y_train_true.sum(), Y_TRUE.sum())  # order should be different
    np.testing.assert_equal(results.y_val_pred.sum(), Y_TRUE.sum())
