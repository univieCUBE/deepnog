"""
Author: Lukas Gosch
        Roman Feldbauer
"""
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Alphabet import ProteinAlphabet

__all__ = ['DeepFam',
           ]


class PseudoOneHotEncoding(nn.Module):
    """ Encode amino acids with DeepFam-style pseudo one hot.

    Each amino acid is encoded as a vector of length 21.
    Ambiguous characters interpolate between the possible values,
    e.g. J = 0.5 I + 0.5 L.

    Parameters
    ----------
    embedding_dim: int
        Embedding dimensionality.
    alphabet:

    See Also
    --------
    See DeepFam paper Section 2.1.1 for details on the encoding scheme at
    `<https://academic.oup.com/bioinformatics/article/34/13/i254/5045722#118270045>`_.
    """

    def __init__(self, alphabet=None, device='cpu'):
        super().__init__()
        self.device = device
        self.num_classes = 27  # i.e. 26 letters ExtendedIUPAC plus zero-padding

    def forward(self, sequence):
        """ Embedd a given sequence.

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
        x = F.one_hot(sequence, num_classes=self.num_classes)
        # Cut away one-hot encoding for zero padding as well as O & U
        x = x[:, :, 1:25].float()
        # Treat B: D or N
        B = torch.zeros((1, 24), device=self.device)
        B[0, 2] = 0.5
        B[0, 11] = 0.5
        b_found = (x[:, :, 21] == 1).nonzero()
        x[b_found[:, 0], b_found[:, 1]] = B
        # Treat Z: E or Q
        Z = torch.zeros((1, 24), device=self.device)
        Z[0, 3] = 0.5
        Z[0, 13] = 0.5
        z_found = (x[:, :, 22] == 1).nonzero()
        x[z_found[:, 0], z_found[:, 1]] = Z
        # Treat J: I or L
        J = torch.zeros((1, 24), device=self.device)
        J[0, 7] = 0.5
        J[0, 9] = 0.5
        j_found = (x[:, :, 23] == 1).nonzero()
        x[j_found[:, 0], j_found[:, 1]] = J
        # Cut away B, Z & J
        x = x[:, :, :21]

        return x


class DeepFam(nn.Module):
    """ Convolutional network for protein family prediction.

    PyTorch implementation of DeepFam architecture (original: TensorFlow).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__()

        # Read hyperparameter dictionary
        n_classes = model_dict['n_classes']
        kernel_sizes = model_dict['kernel_size']
        n_filters = model_dict['n_filters']
        dropout = model_dict['dropout']
        hidden_units = model_dict['hidden_units']

        """ Alphabet used in DeepFam-Paper. """
        self.alphabet = ProteinAlphabet()
        self.alphabet.letters = 'ACDEFGHIKLMNPQRSTVWYXBZJUO'
        self.alphabet.size = 21

        # One-Hot-Encoding Layer
        self.device = device
        self.encoding = PseudoOneHotEncoding(alphabet=self.alphabet,
                                             device=self.device)
        # Convolutional Layers
        for i, kernel in enumerate(kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.alphabet.size,
                                   out_channels=n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layer, gain = 1.0 to match tensorflow implementation
            nn.init.xavier_uniform_(conv_layer.weight, gain=1.0)
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # momentum=1-decay to port from tensorflow
            batch_layer = nn.BatchNorm1d(num_features=n_filters,
                                         eps=0.001,
                                         momentum=0.1,
                                         affine=True)
            # tensorflow implementation only updates bias term not gamma
            batch_layer.weight.requires_grad = False
            self.add_module(f'batch{i + 1}', batch_layer)
        self.n_conv_layers = i + 1

        # Max-Pooling Layer, yields same output as MaxPooling Layer for sequences of size 1000
        # as used in DeepFam but makes the NN applicable to arbitrary sequence lengths
        self.pooling1 = nn.AdaptiveMaxPool1d(output_size=1)

        self.activation1 = nn.ReLU()

        # Dropout
        self.dropout1 = nn.Dropout(p=dropout)

        # Dense NN
        # Hidden Layer
        self.linear1 = nn.Linear(in_features=n_filters * len(kernel_sizes),
                                 out_features=hidden_units)
        # Batch Normalization of Hidden Layer
        self.batch_linear = nn.BatchNorm1d(num_features=hidden_units,
                                           eps=0.001,
                                           momentum=0.1,
                                           affine=True)
        self.batch_linear.weight.requires_grad = False
        # Classifcation layer
        self.classification1 = nn.Linear(in_features=hidden_units,
                                         out_features=n_classes)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.classification1.weight, gain=1.0)
        self.linear1.bias.data.fill_(0.01)
        self.classification1.bias.data.fill_(0.01)

        # Softmax-Layer
        self.softmax = nn.Softmax(dim=1)

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
            Confidence of sequence(s) beeing in one of the n_classes.
        """
        # s.t. sum over filter size is a sum over contiguous memory blocks
        x = self.encoding(x).permute(0, 2, 1).contiguous()
        max_pool_layer = []
        for i in range(self.n_conv_layers):
            x_conv = getattr(self, f'conv{i + 1}')(x)
            x_conv = getattr(self, f'batch{i + 1}')(x_conv)
            x_conv = self.activation1(x_conv)
            x_conv = self.pooling1(x_conv)
            max_pool_layer.append(x_conv)
        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=1)
        x = x.view(-1, x.shape[1])
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.batch_linear(x)
        x = self.activation1(x)
        x = self.classification1(x)
        out = self.softmax(x)
        return out
