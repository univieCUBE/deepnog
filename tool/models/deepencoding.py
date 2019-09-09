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
    SeLU activation is applied on the output followed by an AlphaDropout
    Layer and an AdaptiveMaxPooling1D-Layer reducing the dimension to of
    the output layer to Cx1 and resulting in the NN beeing sequence length
    independent. The max-pooling layer is followed up by a dense layer
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

import torch.nn as nn

class DeepEncoding(nn.Module):
    """ Convolutional network for protein family prediction on eggNOG5 classes.

    The architecture is based on DeepFam, with some changes (learned encoding,
    activation functions, output layers, etc.)

    Parameters
    ----------
    ...
    """
    def __init__(self, n_classes: List[int], encoding_dim: int = 5, 
                 kernel_size: int = 3, n_filters: int = 1_000, 
                 dropout: float = 0.2, embedding_layer_type: str = 'max'):
        super(DeepEncoding, self).__init__()

        # Encoding of amino acid sequence to vector space
        self.encoding_dim = encoding_dim
        self.encoding = AminoAcidWordEmbedding(embedding_dim=encoding_dim,
                                               k=1)

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=encoding_dim,
                               out_channels=n_filters,
                               kernel_size=kernel_size, )
        # Non-linearity
        self.activation1 = nn.SELU()

        # Regularization with dropout
        self.dropout1 = nn.AlphaDropout(p=dropout)

        # Embedding layer
        if 'avg' in embedding_layer_type:
            self.embedding1 = nn.AdaptiveAvgPool1d(output_size=1)
        elif 'max' in embedding_layer_type:
            self.embedding1 = nn.AdaptiveMaxPool1d(output_size=1)
        else:
            raise ValueError(f'Unknown embedding_layer_type: '
                             + f'{embedding_layer_type}')

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
        x = self.dropout1(x)
        x = self.embedding1(x)
        x = x.view(-1, self.conv1.out_channels)
        out = self.classification1(x)
        return out