"""
Author: Lukas Gosch
Date: 2019-10-09
Description:
    Convolutional network (similar to DeepFam) for protein family prediction.
    Architecture conceived by Roman for multiclass-classification on different
    databases, adapted by Lukas to focus on single-task classification
    (using only one database). Adaption due to comparability with DeepFam
    architecture and usage of more training data.

    This networks consists of an embedding layer which learns a D-dimensional
    embedding for each amino acid. For a sequence of length L, the embedding
    has dimension DxL. A 1-D convolution with C filters of F different kernel-
    sizes K_i are performed over the embedding resulting in Cx(L-K_i-1) output
    dimension for each kernel size. SeLU activation is applied on the output
    followed by AdaptiveMaxPooling1D Layer reducing the dimension to of the
    output layer to Cx1 and resulting in the NN being sequence length
    independent. The max-pooling layer is followed up by a classic dropout
    Layer and then by a dense layer with as many output nodes as orthologous
    groups/protein families to classify.
"""
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import numpy as np

from .. import dataset as ds


__all__ = ['AminoAcidWordEmbedding',
           'deepencoding',
           ]


class AminoAcidWordEmbedding(nn.Module):
    """ PyTorch nn.Embedding where each amino acid is considered one word.

    Parameters
    ----------
    embedding_dim: int
        Embedding dimensionality.
    """

    def __init__(self, embedding_dim=10):
        super(AminoAcidWordEmbedding, self).__init__()
        # Get protein sequence vocabulary
        self.vocab = ds.gen_amino_acid_vocab()
        # Create embedding (initialized randomly)
        embeds = nn.Embedding(len(self.vocab) // 2 + 1, embedding_dim)
        self.embedding = embeds

    def forward(self, sequence):
        """ Embed a given sequence.

        Parameters
        ----------
        sequence : Tensor
            The sequence or a batch of sequences to embed. They are assumed to
            be translated to numerical values given a generated vocabulary
            (see gen_amino_acid_vocab in dataset.py)

        Returns
        -------
        x : Tensor
            The sequence (densely) embedded in a space of dimension
            embedding_dim.
        """
        x = self.embedding(sequence)
        return x


class deepencoding(nn.Module):
    """ Convolutional network for protein family prediction on eggNOG5 classes.

    Compared to DeepFam, this architecture provides:

        - learned amino acid embeddings
        - self-normalizing network with SELU
        - sequence length independence
        - stream-lined output layer

    Note on class name: using function naming style to match file name to
        dynamically load different architectures. Furthermore, NN-model
        is primarily used as a callable.

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict):
        super(deepencoding, self).__init__()

        # Read hyperparameter dictionary
        n_classes = model_dict['n_classes']
        encoding_dim = model_dict['encoding_dim']
        kernel_sizes = model_dict['kernel_size']
        n_filters = model_dict['n_filters']
        dropout = model_dict['dropout']
        pooling_layer_type = model_dict['pooling_layer_type']

        # Encoding of amino acid sequence to vector space
        self.encoding = AminoAcidWordEmbedding(embedding_dim=encoding_dim)
        # Convolutional Layers
        for i, kernel in enumerate(kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=encoding_dim,
                                   out_channels=n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layers for SELU activation
            conv_layer.weight.data.normal_(
                0.0, np.sqrt(1. / np.prod(conv_layer.kernel_size)))
            self.add_module(f'conv{i+1}', conv_layer)
        self.n_conv_layers = len(kernel_sizes)
        # Non-linearity
        self.activation1 = nn.SELU()
        # Max Pooling layer
        if 'avg' in pooling_layer_type:
            self.pool1 = nn.AdaptiveAvgPool1d(output_size=1)
        elif 'max' in pooling_layer_type:
            self.pool1 = nn.AdaptiveMaxPool1d(output_size=1)
        else:
            raise ValueError(f'Unknown pooling_layer_type: '
                             + f'{pooling_layer_type}')

        # Regularization with dropout
        self.dropout1 = nn.Dropout(p=dropout)
        # Classifcation layer
        self.classification1 = nn.Linear(
            in_features=n_filters * len(kernel_sizes),
            out_features=n_classes[0])

        # Softmax-Layer
        self.softmax = nn.Softmax(dim=1)

        # Threshold for deciding below which confidence NN should be undecided
        if 'threshold' in model_dict:
            self.threshold = model_dict['threshold']

    def forward(self, x):
        """ Forward a batch of sequences through network.

        Parameters
        ----------
        x : Tensor, shape (batch_size, sequence_len)
            Sequence or batch of sequences to classify. Assumes they are
            translated using a vocabulary. (See gen_amino_acid_vocab in
            dataset.py)

        Returns
        -------
        out : Tensor, shape (batch_size, n_classes)
            Confidence of sequence(s) being in one of the n_classes.
        """
        # Fix type mismatch on Windows
        x = x.long()

        # Amino acid embedding
        x = self.encoding(x).permute(0, 2, 1).contiguous()

        # Convolution with variable kernel sizes and adaptive max pooling
        max_pool_layer = []
        for i in range(self.n_conv_layers):
            x_conv = getattr(self, f'conv{i+1}')(x)
            x_conv = self.activation1(x_conv)
            x_conv = self.pool1(x_conv)
            max_pool_layer.append(x_conv)

        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=1)
        x = x.view(-1, x.shape[1])

        # Classification layer
        x = self.classification1(x)
        out = self.softmax(x)
        return out
