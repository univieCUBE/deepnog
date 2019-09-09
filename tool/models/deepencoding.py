"""
Author: Lukas Gosch
Date: 6.9.2019
Description:
    Convolutional network (similar to DeepFam) for protein family prediction.
    Architecture conceived by Roman for multiclass-classification on different
    databases, adapted by Lukas to focus on single-task classification
    (using only one database). Adaption due to comparability with DeepFam 
    architecture and usage of more training data.

    This networks consists of an embedding layer which learns a D-dimensional
    embedding for each amino acid. For a sequence of length L, the embedding 
    has dimension DxL. A 1-D convolution with C filters of kernelsize K is 
    performed over the embedding resulting in Cx(L-K-1) output dimension. 
    SeLU activation is applied on the output followed by AdaptiveMaxPooling1D
    Layer reducing the dimension to of the output layer to Cx1 and resulting 
    in the NN beeing sequence length independent. The max-pooling layer is 
    followed up by a classic Dropout-Layer and then by a dense layer
    with as many output nodes as orthologous groups/protein families to 
    classify. 
"""

#######
# TODO: Package project and replace encapsulated code with relative imports!
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from dataset import AminoAcidWordEmbedding
#######

import numpy as np
import torch.nn as nn

class deepencoding(nn.Module):
    """ Convolutional network for protein family prediction on eggNOG5 classes.

    The architecture is based on DeepFam, with some changes (learned encoding,
    activation functions, output layers, etc.)
    
    Note on class name: using function naming style to match file name to
        dynamically load different architectures. Furthermore, NN-model 
        is primarily used as a callable.

    Parameters
    ----------
    model_dict : dictionary
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict):
        super(deepencoding, self).__init__()

        # Read hyperparameter dictionary
        n_classes = model_dict['n_classes']
        encoding_dim = model_dict['encoding_dim']
        kernel_size = model_dict['kernel_size']
        n_filters = model_dict['n_filters']
        dropout = model_dict['dropout']
        pooling_layer_type = model_dict['pooling_layer_type']

        # Encoding of amino acid sequence to vector space
        self.encoding = AminoAcidWordEmbedding(embedding_dim=encoding_dim)
        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=encoding_dim,
                               out_channels=n_filters,
                               kernel_size=kernel_size, )
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
        self.classification1 = nn.Linear(in_features=n_filters, 
                                         out_features=n_classes[0])

        # Initialize weights for SELU activation
        self.conv1.weight.data.normal_(0.0, 
                                np.sqrt(1. / np.prod(self.conv1.kernel_size)))

    def forward(self, x):
        x = self.encoding(x).permute(0, 2, 1).contiguous() 
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(-1, self.conv1.out_channels)
        out = self.classification1(x)
        return out