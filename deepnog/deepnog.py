"""
Author: Lukas Gosch
Date: 3.10.2019
Usage: python deepnog.py --help
Description:
    DeepNOG is a deep learning based command line tool which predicts the
    protein families of given protein sequences based on pretrained neural
    networks.

    File formats supported:
    Prefered: FASTA
    DeepNOG supports protein sequences stored in all file formats listed in
    https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
    only.

    Architectures supported:

    Databases supported:
        - eggNOG 5.0, taxonomic level 2
"""

import sys
import argparse
import os.path
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import deepnog
from .dataset import ProteinDataset
from .dataset import collate_sequences


def get_parser():
    """ Creates a new argument parser.

    Returns
    -------
    parser : ArgumentParser
        ArgumentParser object to parse programm arguments.
    """
    parser = argparse.ArgumentParser('DeepNOG is a deep learning based command'
                            + ' line tool which predicts the protein families'
                            + ' given protein sequences based on pretrained'
                            + ' neural networks.')
    parser.add_argument("file", help="File containing protein sequences for "
                        + "classification.")
    parser.add_argument("-o", "--out", default='out.csv', help="Path where"
                        + " to store the output-file containing the protein"
                        + " family predictions.")
    parser.add_argument("-d", "--device", default='auto', choices=['auto',
                        'cpu', 'gpu'], help="Define device for calculating "
                        + " protein sequence classification. Auto chooses gpu "
                        + " if available, otherwise cpu.")
    parser.add_argument("-ff", "--fformat", default='fasta',
                        help="File format of protein sequences. Must be "
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
    parser.add_argument("--tab", action='store_true',
                        help='If set, output will be tab-separated instead of'
                        + ' ;-separated.')
    return parser


def load_nn(architecture, model_dict, device='cpu'):
    """ Import NN architecture and set loaded parameters.

    Parameters
    ----------
    architecture : str
        Name of neural network module and class to import.
    model_dict : dict
        Dictionary holding all parameters and hyperparameters of the model.
    device : dict
        Device to load the model into.

    Returns
    -------
    model : torch.nn.Module
        Neural network object of type architecture with parameters
        loaded from model_dict and moved to device.
    """
    # Import and instantiate neural network class
    model_module = import_module(f'.models.{architecture}', 'deepnog')
    model_class = getattr(model_module, architecture)
    model = model_class(model_dict)
    # Set trained parameters of model
    model.load_state_dict(model_dict['model_state_dict'])
    # Move to GPU, if available
    model.to(device)
    # Inform neural network layers to be in evaluation mode
    model = model.eval()
    return model


def predict(model, dataset, device='cpu', batch_size=16):
    """ Use model to predict zero-indexed labels of dataset.

    Parameters
    ----------
    model : nn.Module
        Trained neural network model.
    dataset : ProteinDataset
        Data to predict protein families for.
    device : str
        Device of model.
    batch_size : int
        Forward batch_size proteins through neural network at once.

    Returns
    -------
    preds : torch.Tensor, shape (n_samples,)
        Stores the index of the output-node with the highest activation
    confs : torch.Tensor, shape (n_samples,)
        Stores the confidence in the prediciton
    ids : list[str]
        Stores the (possible empty) protein labels extracted from data 
        file.
    indices : list[int]
        Stores the unique indices of sequences mapping to their position 
        in the file
    """
    pred_l = []
    conf_l = []
    ids = []
    indices = []
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=4, collate_fn=collate_sequences)
    # Give user performance feedback
    print(f'Process {batch_size} sequences per iteration: ')
    # Disable tracking of gradients to increase performance
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # Push sequences on correct device
            sequences = batch.sequences.to(device)
            # Predict protein families
            output = model(sequences)
            conf, pred = torch.max(output, 1)
            # Store predictions
            pred_l.append(pred)
            conf_l.append(conf)
            ids.extend(batch.ids)
            indices.extend(batch.indices)
    # Merge individual output tensors
    preds = torch.cat(pred_l)
    confs = torch.cat(conf_l)
    return preds, confs, ids, indices


def create_df(class_labels, preds, confs, ids, indices, device='cpu'):
    """ Creates one dataframe storing all relevant prediction information.

    The rows in the returned dataframe have the same order as the
    original sequences in the data file. First column of the dataframe
    represents.

    Returns
    -------
    df : pandas.DataFrame
        Stores prediction information about the input protein sequences.
    """
    labels = [class_labels[pred] for pred in preds]
    confs = confs.cpu().numpy()
    df = pd.DataFrame(data={'index': indices,
                            'sequence_id': ids,
                            'prediction': labels,
                            'confidence': confs})
    df.sort_values(by='index', axis=0, inplace=True)
    return df

def set_device(user_choice):
    """ Sets calc. device depending on users choices and availability. 
        
    Returns
    -------
    device : torch.device
        Object containing the device type to be used for prediction 
        calculations.
    """
    if user_choice == 'auto':
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
    elif user_choice == 'gpu':
        cuda = torch.cuda.is_available()
        if cuda:
            device = torch.device('cuda')
        else:
            raise RuntimeError('Device set to gpu but no cuda-enabled gpu '
                               + 'available on machine!')
    else:
        device = torch.device('cpu')
    return device

def main(args=None):
    """ DeepNOG command line tool. """
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Construct path to saved parametes of NN
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = os.path.join(deepnog.__path__[0],
                                    'parameters',
                                    args.database,
                                    str(args.tax),
                                    args.architecture + '.pth')
    # Set up device
    try:
        device = set_device(args.device)
    except RuntimeError as err:
        sys.exit(f'RuntimeError: {err} \nLeaving the ship and aborting '
                 +'calculations.')
    print(f'Device set to "{device}"')

    # Load neural network parameters
    print(f'Loading NN-parameters from {weights_path} ...')
    model_dict = torch.load(weights_path, map_location=device)
    # Load neural network model
    model = load_nn(args.architecture, model_dict, device)
    # Load class names
    class_labels = model_dict['classes']

    # Load dataset
    print(f'Accessing dataset from {args.file} ...')
    dataset = ProteinDataset(args.file, f_format=args.fformat)

    # Predict labels of given data
    print(f'Predicting protein families ...')
    preds, confs, ids, indices = predict(model, dataset, device)

    # Construct pandas dataframe
    df = create_df(class_labels, preds, confs, ids, indices, device)

    # Construct path to save prediction
    if os.path.isdir(args.out):
        save_file = os.path.join(args.out, 'out.csv')
    else:
        save_file = args.out
    print(f'Writing prediction to {save_file}')
    # Write to file
    if args.tab:
        df.to_csv(save_file, sep='\t', index=False)
    else:
        df.to_csv(save_file, sep=';', index=False)

    print(f'Finished magic.')
    return


if __name__ == '__main__':
    main()
