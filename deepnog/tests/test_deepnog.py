"""
Author: Lukas Gosch
Date: 3.10.2019
Description:
    Test deepnog module and pretrained neural network architectures.
"""

import torch.nn as nn
import torch
import pytest
import os
import sys
import inspect

from .. import deepnog as dn
from ..dataset import ProteinDataset

class TestDeepnog:
    """ Class grouping tests for deepnog module. """

    @pytest.mark.parametrize("architecture", ['deepencoding'])
    @pytest.mark.parametrize("weights", ['tests/parameters/test_deepencoding.pth'])
    def test_load_nn(self, architecture, weights):
        """ Test loading of neural network model. """
        # Set up device
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        # Start test
        model_dict = torch.load(weights, map_location=device)
        model = dn.load_nn(architecture, model_dict, device)
        assert(issubclass(type(model), nn.Module))
        assert(isinstance(model, nn.Module))

    @pytest.mark.parametrize("architecture", ['deepencoding'])
    @pytest.mark.parametrize("weights", ['tests/parameters/test_deepencoding.pth'])
    @pytest.mark.parametrize("data", ['tests/data/test_deepencoding.faa'])
    @pytest.mark.parametrize("fformat", ['fasta'])
    @pytest.mark.parametrize("tolerance", [2])
    def test_predict(self, architecture, weights, data, fformat, tolerance):
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
        model = dn.load_nn(architecture, model_dict, device)
        dataset = ProteinDataset(data, f_format=fformat)
        preds, confs, ids, indices = dn.predict(model, dataset, device)
        # Test correct output shape
        assert(preds.shape[0] == confs.shape[0])
        assert(confs.shape[0] == len(ids))
        assert(len(ids) == len(indices))
        # Test satisfying prediction accuracy
        N = len(ids)
        ids = torch.tensor(list(map(int, ids)))
        assert(sum((ids == preds.cpu()).long()) >= N - tolerance)

    def test_create_df(self):
        """ Test correct creation of dataframe. """
        class_labels = ['class1', 'class2']
        preds = torch.tensor([1, 0])
        confs = torch.tensor([0.8, 0.3])
        ids = ['sequence2', 'sequence1']
        indices = [2, 1]
        df = dn.create_df(class_labels, preds, confs, ids, indices)
        assert(df.shape == (2, 4))
        assert(sum(df['index'] == [1, 2]) == 2)
        assert(sum(df['sequence_id'] == ['sequence1', 'sequence2']) == 2)
        assert(sum(df['prediction'] == ['class1', 'class2']) == 2)
        df_confs = df['confidence'].tolist()
        assert(df_confs[0] < 0.5)
        assert(df_confs[1] > 0.5)
