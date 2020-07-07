"""
Author: Lukas Gosch
        Roman Feldbauer
"""
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from .deepnog import AminoAcidWordEmbedding

__all__ = ['DeepFam',
           'DeepFamAblation1',
           'DeepFamAblation2',
           'DeepFamAblation3',
           'DeepFamAblation12',
           'DeepFamAblation13',
           'DeepFamAblation23',
           'DeepFamAblation123',
           ]


class PseudoOneHotEncoding(nn.Module):
    """ Encode amino acids with DeepFam-style pseudo one hot.

    Each amino acid is encoded as a vector of length 21.
    Ambiguous characters interpolate between the possible values,
    e.g. J = 0.5 I + 0.5 L.

    Parameters
    ----------
    device : 'cpu', 'cuda'
        Run on CPU or GPU

    See Also
    --------
    See DeepFam paper Section 2.1.1 for details on the encoding scheme at
    `<https://academic.oup.com/bioinformatics/article/34/13/i254/5045722#118270045>`_.
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.num_classes = 27  # i.e. 26 letters ExtendedIUPAC plus zero-padding
        self.alphabet_size = 21  # after encoding

    def forward(self, sequence):
        """ Embedd a given sequence.

        Parameters
        ----------
        sequence : Tensor
            The sequence or a batch of sequences to embed. They are assumed to
            be translated to numerical values given a generated vocabulary
            (see gen_amino_acid_vocab in dataset.py).
            Must correspond to the following alphabet:
            Index    0  3  6  9 12 15 18 21 24 27
            Letter   _ACDEFGHIKLMNPQRSTVWYXBZJUO

        Returns
        -------
        x : Tensor
            The sequence (densely) embedded in a space of dimension
            embedding_dim.
        """
        x = one_hot(sequence, num_classes=self.num_classes)
        # Cut away one-hot encoding for zero padding as well as O & U
        x = x[:, :, 1:25].float()
        # Indices for b, z, j arrays below:
        # Index    0  3  6  9 12 15 18 21 24
        # Letter   ACDEFGHIKLMNPQRSTVWYXBZJ
        # Treat B: D or N
        b = torch.zeros((1, 24), device=self.device)
        b[0, 2] = 0.5
        b[0, 11] = 0.5
        b_found = (x[:, :, 21] == 1).nonzero()
        x[b_found[:, 0], b_found[:, 1]] = b
        # Treat Z: E or Q
        z = torch.zeros((1, 24), device=self.device)
        z[0, 3] = 0.5
        z[0, 13] = 0.5
        z_found = (x[:, :, 22] == 1).nonzero()
        x[z_found[:, 0], z_found[:, 1]] = z
        # Treat J: I or L
        j = torch.zeros((1, 24), device=self.device)
        j[0, 7] = 0.5
        j[0, 9] = 0.5
        j_found = (x[:, :, 23] == 1).nonzero()
        x[j_found[:, 0], j_found[:, 1]] = j
        # Cut away B, Z, J to obtain
        # Index    0  3  6  9 12 15 18 21
        # Letter   ACDEFGHIKLMNPQRSTVWYX
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

        try:  # for inference these values are already available in the model
            state = model_dict['model_state_dict']
            self.n_classes = state['classification1.weight'].shape[0]
            kernel_sizes = [v.shape[-1] for k, v in state.items() if 'conv' in k and 'weight' in k]
            n_filters = state['conv1.weight'].shape[0]
            dropout = model_dict.get('dropout', 0.3)
            hidden_units = state['linear1.weight'].shape[0]
        except KeyError:  # set up the model for training
            self.n_classes = model_dict['n_classes']
            kernel_sizes = model_dict['kernel_size']
            n_filters = model_dict['n_filters']
            dropout = model_dict['dropout']
            hidden_units = model_dict['hidden_units']
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.dropout = dropout
        self.hidden_units = hidden_units

        # One-Hot-Encoding Layer
        self.device = device
        self.encoding = PseudoOneHotEncoding(device=self.device)
        # Convolutional Layers
        for i, kernel in enumerate(kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.encoding.alphabet_size,
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
        self.n_conv_layers = len(kernel_sizes)

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
                                         out_features=self.n_classes)

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
        # no softmax here
        return x


class DeepFamAblation1(DeepFam):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        # Embedding Layer
        self.device = device
        self.embedding_dim = 10
        self.encoding = AminoAcidWordEmbedding(embedding_dim=self.embedding_dim)

        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.embedding_dim,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layer, gain = 1.0 to match tensorflow implementation
            nn.init.xavier_uniform_(conv_layer.weight, gain=1.0)
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # momentum=1-decay to port from tensorflow
            batch_layer = nn.BatchNorm1d(num_features=self.n_filters,
                                         eps=0.001,
                                         momentum=0.1,
                                         affine=True)
            # tensorflow implementation only updates bias term not gamma
            batch_layer.weight.requires_grad = False
            self.add_module(f'batch{i + 1}', batch_layer)


class DeepFamAblation2(DeepFam):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        # One-Hot-Encoding Layer
        self.device = device
        self.encoding = PseudoOneHotEncoding(device=self.device)
        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.encoding.alphabet_size,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layers for SELU activation
            conv_layer.weight.data.normal_(
                0.0, np.sqrt(1. / np.prod(conv_layer.kernel_size)))
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # No batch-normalization here

        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        # Initialize linear layers for SELU
        self.linear1.weight.data.normal_(0., np.sqrt(self.linear1.in_features))
        self.classification1.weight.data.normal_(0., np.sqrt(self.classification1.in_features))

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
        # s.t. sum over filter size is a sum over contiguous memory blocks
        x = self.encoding(x).permute(0, 2, 1).contiguous()
        max_pool_layer = []
        for i in range(self.n_conv_layers):
            x_conv = getattr(self, f'conv{i + 1}')(x)
            x_conv = self.activation1(x_conv)
            x_conv = self.pooling1(x_conv)
            max_pool_layer.append(x_conv)
        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=1)
        x = x.view(-1, x.shape[1])
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.classification1(x)
        # no softmax here
        return x


class DeepFamAblation3(DeepFam):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        del self.linear1
        del self.batch_linear

        # Classification layer
        self.classification1 = nn.Linear(in_features=self.n_filters * len(self.kernel_sizes),
                                         out_features=self.n_classes)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.classification1.weight, gain=1.0)
        self.classification1.bias.data.fill_(0.01)

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
        x = self.classification1(x)
        # no softmax here
        return x


class DeepFamAblation12(DeepFamAblation2):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        # Embedding Layer
        self.device = device
        self.embedding_dim = 10
        self.encoding = AminoAcidWordEmbedding(embedding_dim=self.embedding_dim)

        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.embedding_dim,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layers for SELU activation
            conv_layer.weight.data.normal_(
                0.0, np.sqrt(1. / np.prod(conv_layer.kernel_size)))
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # No batch-normalization here

        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        # Initialize linear layers for SELU
        self.linear1.weight.data.normal_(0., np.sqrt(self.linear1.in_features))
        self.classification1.weight.data.normal_(0., np.sqrt(self.classification1.in_features))


class DeepFamAblation13(DeepFamAblation1):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        del self.linear1
        del self.batch_linear

        # Classification layer
        self.classification1 = nn.Linear(in_features=self.n_filters * len(self.kernel_sizes),
                                         out_features=self.n_classes)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.classification1.weight, gain=1.0)
        self.classification1.bias.data.fill_(0.01)

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
        x = self.classification1(x)
        # no softmax here
        return x


class DeepFamAblation23(DeepFamAblation2):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        del self.linear1
        del self.batch_linear

        # Classification layer
        self.classification1 = nn.Linear(in_features=self.n_filters * len(self.kernel_sizes),
                                         out_features=self.n_classes)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.classification1.weight, gain=1.0)
        self.classification1.bias.data.fill_(0.01)

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
            x_conv = self.activation1(x_conv)
            x_conv = self.pooling1(x_conv)
            max_pool_layer.append(x_conv)
        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=1)
        x = x.view(-1, x.shape[1])
        x = self.dropout1(x)
        x = self.activation1(x)
        x = self.classification1(x)
        # no softmax here
        return x


class DeepFamAblation123(DeepFamAblation23):
    """ Ablation study of DeepFam to DeepNOG transition.

    Change 1: WordEmbedding instead of PseudoOneHot
    Change 2: SELU instead of BN/ReLU
    Change 3: Drop the fully connected layer between ConvNet and classification
    Combinations: 12, 13, 23, 123 (=DeepNOG).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, device='cpu'):
        super().__init__(model_dict)

        # Embedding Layer
        self.device = device
        self.embedding_dim = 10
        self.encoding = AminoAcidWordEmbedding(embedding_dim=self.embedding_dim)

        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.embedding_dim,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layers for SELU activation
            conv_layer.weight.data.normal_(
                0.0, np.sqrt(1. / np.prod(conv_layer.kernel_size)))
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # No batch-normalization here
