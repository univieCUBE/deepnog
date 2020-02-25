"""
Author: Lukas Gosch
Date: 2019-10-18
Description:
    Test deepnog module and pretrained neural network architectures.
"""
from pathlib import Path
import pytest

import torch.nn as nn
import torch

from deepnog.dataset import ProteinDataset
from deepnog.inference import load_nn, predict
from deepnog.io import create_df


current_path = Path(__file__).parent.absolute()
weights_path = current_path/'parameters/test_deepencoding.pthsmall'
data_path = current_path/'data/test_deepencoding.faa'
data_skip_path = current_path/'data/test_skip_empty_sequences.faa'


@pytest.mark.parametrize("architecture", ['deepencoding', ])
@pytest.mark.parametrize("weights", [weights_path, ])
def test_load_nn(architecture, weights):
    """ Test loading of neural network model. """
    # Set up device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # Start test
    model_dict = torch.load(weights, map_location=device)
    model = load_nn(architecture, model_dict, device)
    assert(issubclass(type(model), nn.Module))
    assert(isinstance(model, nn.Module))


@pytest.mark.parametrize("architecture", ['deepencoding'])
@pytest.mark.parametrize("weights", [weights_path, ])
@pytest.mark.parametrize("data", [data_path, ])
@pytest.mark.parametrize("fformat", ['fasta'])
@pytest.mark.parametrize("tolerance", [2])
def test_predict(architecture, weights, data, fformat, tolerance):
    """ Test correct prediction output shapes as well as satisfying
        prediction performance.

        Prediction performance is checked through sequences from SIMAP with
        known class labels. Class labels are stored as the id in the given
        fasta file. Tolerance defines how many sequences the algorithm
        is allowed to misclassfy before the test fails.
    """
    # Set up device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # Start test
    model_dict = torch.load(weights, map_location=device)
    model = load_nn(architecture, model_dict, device)
    dataset = ProteinDataset(data, f_format=fformat)
    preds, confs, ids, indices = predict(model, dataset, device)
    # Test correct output shape
    assert(preds.shape[0] == confs.shape[0])
    assert(confs.shape[0] == len(ids))
    assert(len(ids) == len(indices))
    # Test satisfying prediction accuracy
    N = len(ids)
    ids = torch.tensor(list(map(int, ids)))
    assert(sum((ids == preds.cpu()).long()) >= N - tolerance)


@pytest.mark.parametrize("architecture", ['deepencoding'])
@pytest.mark.parametrize("weights", [weights_path, ])
@pytest.mark.parametrize("data", [data_skip_path, ])
@pytest.mark.parametrize("fformat", ['fasta'])
def test_skip_empty_sequences(architecture, weights, data, fformat):
    """ Test if sequences with empty ids are skipped and counted correctly.
    """
    # Set up device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # Start test
    model_dict = torch.load(weights, map_location=device)
    model = load_nn(architecture, model_dict, device)
    dataset = ProteinDataset(data, f_format=fformat)
    with pytest.warns(UserWarning, match='no sequence id could be detected'):
        preds, confs, ids, indices = predict(model, dataset, device)
    # Test correct output shape
    assert(preds.shape[0] == 70)
    # Test correct counted skipped sequences
    assert(int(dataset.n_skipped) == 20)


def test_create_df():
    """ Test correct creation of dataframe. """
    class_labels = ['class1', 'class2']
    preds = torch.tensor([1, 0])
    confs = torch.tensor([0.8, 0.3])
    ids = ['sequence2', 'sequence1']
    indices = [2, 1]
    df = create_df(class_labels, preds, confs, ids, indices)
    assert(df.shape == (2, 4))
    assert(sum(df['index'] == [1, 2]) == 2)
    assert(sum(df['sequence_id'] == ['sequence1', 'sequence2']) == 2)
    assert(sum(df['prediction'] == ['class1', 'class2']) == 2)
    df_confs = df['confidence'].tolist()
    assert(df_confs[0] < 0.5)
    assert(df_confs[1] > 0.5)


def test_create_df_with_duplicates():
    """ Test correct exclusion of duplicates. """
    class_labels = ['class1', 'class2']
    preds = torch.tensor([1, 0, 0, 1, 0])
    confs = torch.tensor([0.8, 0.3, 0.1, 0.6, 0.8])
    ids = ['sequence2', 'sequence1', 'sequence2', 'sequence3', 'sequence1']
    indices = [1, 2, 3, 4, 5]
    df = create_df(class_labels, preds, confs, ids, indices)
    assert(df.shape == (3, 4))
    assert(sum(df['index'] == [1, 2, 4]) == 3)
    assert(sum(df['sequence_id'] == ['sequence2', 'sequence1', 'sequence3'])
           == 3)
    assert(sum(df['prediction'] == ['class2', 'class1', 'class2']) == 3)
    df_confs = df['confidence'].tolist()
    assert(df_confs[0] > 0.5)
    assert(df_confs[1] < 0.5)
    assert(df_confs[2] > 0.5)
