from pathlib import Path
import tempfile
import pytest
import numpy as np

from deepnog.learning import fit
from deepnog.tests.utils import get_deepnog_root

DEEPNOG_TESTS = get_deepnog_root()/"tests"
TRAINING_FASTA = DEEPNOG_TESTS/"data/test_training_dummy.faa"
TRAINING_CSV = DEEPNOG_TESTS/"data/test_training_dummy.faa.csv"
Y_TRUE = np.array([[0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
                    1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
                    1, 1, 1, 1, 1, 1, 1, 1]])


@pytest.mark.parametrize("iterable_dataset", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_iterable_training(iterable_dataset: bool, shuffle: bool):
    """ Regression test for https://github.com/univieCUBE/deepnog/pull/43"""
    results = fit(
        architecture="deepnog", module="deepnog", cls="DeepNOG",
        training_sequences=DEEPNOG_TESTS/"data/test_training_COG0443.faa",
        validation_sequences=DEEPNOG_TESTS/"data/test_validation_COG0443.faa",
        training_labels=DEEPNOG_TESTS/"data/test_training_COG0443.csv",
        validation_labels=DEEPNOG_TESTS/"data/test_validation_COG0443.csv",
        iterable_dataset=iterable_dataset,
        shuffle=shuffle,
        n_epochs=1,
        random_seed=123,
    )
    training_data = results.training_dataset
    val_data = results.validation_dataset
    if iterable_dataset:
        assert training_data.file != val_data.file
        assert training_data.labels_file != val_data.labels_file
    else:
        training_ids = set([x.id for x in training_data.sequences])
        val_ids = set([x.id for x in val_data.sequences])
        assert training_ids != val_ids
    df = training_data.labels.merge(val_data.labels, how="inner", on="protein_id")
    assert df.shape[0] == 0


@pytest.mark.parametrize('batch_size', [4, ])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_shuffled_training(batch_size, num_workers):
    results = fit(architecture='deepnog',
                  module='deepnog',
                  cls='DeepNOG',
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
