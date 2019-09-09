"""
Author: Lukas Gosch
Date: 5.9.2019
Description:
    Main
"""

import time
import argparse
import os.path
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from dataset import ProteinDataset
from dataset import collate_sequences


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File containing protein sequences for "
                        + "classification.")
    parser.add_argument("-ff", "--format", default='fasta',
                        help = "File format of protein sequences. Must be "
                        + "supported by Biopythons Bio.SeqIO class.")
    parser.add_argument("-db", "--database", default='eggNOG5',
                        help="Database to classify against.")
    parser.add_argument("-t", "--tax", type=int, default=2,
                        help="Taxonomic level to use in specified database.")
    parser.add_argument("-a", "--architecture", default='deepencoding',
                        help="Neural network architecture to use for "
                        + "classification.")
    parser.add_argument("-w", "--weights", help="Optionally specify custom "
                        + "weights filepath.")
    return parser


def load_nn(architecture, model_dict, device='cpu'):
    """ Import NN architecture and set loaded parameters. 

        architecture : str
            Name of neural network module and class to import.
        model_dict : dict
            Dictionary holding all parameters and hyperparameters of the model
        device : dict
            Device to load the model into
    """
    # Import and instantiate neural network class
    model_module = import_module(f'models.{architecture}')
    model_class = getattr(model_module, architecture)
    model = model_class(model_dict)
    # Set trained parameters of model
    model.load_state_dict(model_dict['model_state_dict'])
    # Move to GPU, if available
    model.to(device)
    # Disable tracking of gradients
    model.eval()
    return model


def predict(model, dataset):
    """ Use model to predict zero-indexed labels of dataset. 

        model : nn.Module
            Trained neural network model.
        dataset : ProteinDataset
            Data to predict protein families for.
    """
    batch_size = 16
    for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size, 
                                         num_workers=4,
                                         collate_fn=collate_sequences)):
        pred = model(batch.sequences)
        break
    return pred


def main(args = None):
    # Construct path to saved parametes of NN
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = os.path.join('parameters', 
                                    args.database, 
                                    str(args.tax),
                                    args.architecture + '.pth')
    # Set up device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'Device set to "{device}"')

    # Load neural network parameters
    print(f'Loading NN-parameters from {weights_path}')
    model_dict = torch.load(weights_path, map_location = device)
    # Load neural network model
    model = load_nn(args.architecture, model_dict, device)
    # Load class names
    class_labels = model_dict['classes']

    # Load dataset
    print(f'Accessing dataset from {args.file}')
    dataset = ProteinDataset(args.file, f_format=args.format)

    # Predict labels of given data
    pred = predict(model, dataset)
    
    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    main(args)