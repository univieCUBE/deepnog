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
import deepnog.dataset as ds

# Globals for communicating number of sequences in the dataset with empty ids.
rpipe_l = []
wpipe_l = []
n_skipped = 0


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
    parser.add_argument("-ff", "--fformat", default='fasta',
                        help="File format of protein sequences. Must be "
                        + "supported by Biopythons Bio.SeqIO class.")
    parser.add_argument("-db", "--database", default='eggNOG5',
                        help="Database to classify against.")
    parser.add_argument("-t", "--tax", type=int, default=2,
                        help="Taxonomic level to use in specified database.")
    parser.add_argument("-v", "--verbose", type=int, default=3,
                        help="Define verbosity of DeepNOGs output written to "
                        + "stdout or stderr. 0 only writes errors to stderr "
                        + "which cause DeepNOG to abort and exit. 1 also "
                        + "writes warnings to stderr if e.g. a protein without"
                        + " an ID was found and skipped. 2 additionally writes"
                        + " general progress messages to stdout. 3 includes a "
                        + "dynamic progress bar of the prediction stage using "
                        + "tqdm.")
    parser.add_argument("-d", "--device", default='auto', choices=['auto',
                        'cpu', 'gpu'], help="Define device for calculating "
                        + " protein sequence classification. Auto chooses gpu "
                        + " if available, otherwise cpu.")
    parser.add_argument("-nw", "--num-workers", type=int, default=0,
                        help='Number of subprocesses (workers) to use for '
                        +'data loading. Set to a value <= 0 to use single-'
                        +'process data loading.')
    parser.add_argument("-a", "--architecture", default='deepencoding',
                        help="Neural network architecture to use for "
                        + "classification.")
    parser.add_argument("-w", "--weights", help="Optionally specify custom "
                        + "weights filepath.")
    parser.add_argument("--tab", action='store_true',
                        help='If set, output will be tab-separated instead of'
                        + ' ;-separated.')
    parser.add_argument("-bs", "--batch-size", type=int, default=16,
                        help='Batch size used for prediction. Defines how many'
                        +' sequences should be forwarded in the network at '
                        +' once. With a batch size of one, the protein '
                        +' sequences are sequentially classified by the neural'
                        +' network without the possibility of leveraging '
                        +' parallelism. Higher batch-sizes than the default '
                        +' one can speed up the prediction significantly if '
                        +' on a gpu. But If on a cpu, they can be slower as '
                        +' smaller ones due to the increased average sequence '
                        +' length in the convolution step due to zero-padding '
                        +' every sequence in each batch.')
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


def predict(model, dataset, device='cpu', batch_size=16, num_workers=4, 
            verbose=3):
    """ Use model to predict zero-indexed labels of dataset.

    Also handles communication with ProteinIterators used to load data to 
    log how many sequences have been skipped due to having empty sequence ids.

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
    num_workers : int
        Number of workers for data loading.
    verbose : int
        Define verbosity.

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

    # Prepare communication
    global n_skipped
    if num_workers >= 2:
        # Prepare message passing for multi-process data loading
        n_pipes = num_workers
        global rpipe_l
        global wpipe_l
        # Dedicate one communication pipe for each worker
        for pipes in range(n_pipes):
            r, w = os.pipe() 
            os.set_inheritable(r, True) # Compatability with windows
            os.set_inheritable(w, True) # Compatability with windows
            rpipe_l.append(r)
            wpipe_l.append(w)
    else:
        num_workers = 0
    # Create data-loader for protein dataset
    data_loader = DataLoader(dataset, batch_size=batch_size,
                                      num_workers=num_workers,
                                      collate_fn=ds.collate_sequences)
    # Disable tracking of gradients to increase performance
    with torch.no_grad():
        # Do prediction calculations
        if verbose >= 3:
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
        else:
            for i, batch in enumerate(data_loader):
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
    # Collect skipped-sequences messages from workers in the case of 
    # multi-process data-loading
    if num_workers >= 2:
        for workers in range(n_pipes):
            os.close(wpipe_l[workers])
            r = os.fdopen(rpipe_l[workers])
            n_skipped += int(r.read())
    # Check if sequences were skipped due to empty id
    if verbose > 0 and n_skipped > 0:
        print(f'WARNING: Skipped {n_skipped} sequences as no sequence id could'
              +' could be detected.', file=sys.stderr)
    # Merge individual output tensors
    preds = torch.cat(pred_l)
    confs = torch.cat(conf_l)
    return preds, confs, ids, indices


def create_df(class_labels, preds, confs, ids, indices, threshold=None,
              device='cpu', verbose=3):
    """ Creates one dataframe storing all relevant prediction information.

    The rows in the returned dataframe have the same order as the
    original sequences in the data file. First column of the dataframe
    represents the position of the sequence in the datafile. 

    Parameters
    ----------
    class_labels : list
        Store class name corresponding to an output node of the network.
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
    threshold : int
        If given, prediction labels and confidences are set to '' if 
        confidence in prediction is not at least threshold.
    device : torch.device
        Object containing the device type to be used for prediction 
        calculations.
    verbose : int
        If bigger 0, outputs warning if duplicates detected.

    Returns
    -------
    df : pandas.DataFrame
        Stores prediction information about the input protein sequences.
        Duplicates (defined by their sequence_id) have been removed from df.
    """
    confs = confs.cpu().numpy()
    if threshold is not None:
        # Set empty label and confidence if prediction confidence below
        # threshold of model
        labels = [class_labels[pred] if conf >= threshold \
                  else '' for pred, conf in zip(preds, confs)]
        confs = [str(conf) if conf >= threshold \
                  else '' for conf in confs]
    else:
        labels = [class_labels[pred] for pred in preds]
    # Create prediction results frame
    df = pd.DataFrame(data={'index': indices,
                            'sequence_id': ids,
                            'prediction': labels,
                            'confidence': confs})
    df.sort_values(by='index', axis=0, inplace=True)
    # Remove duplicate sequences
    duplicate_mask = df.sequence_id.duplicated(keep='first')
    n_duplicates = sum(duplicate_mask)
    if n_duplicates > 0:
        if verbose > 0:
            print(f'WARNING: Detected {sum(duplicate_mask)} duplicate sequences '
                  +'based on their extracted sequence id. Ignore duplicate '
                  +'sequences in writing prediction output-file.', file=sys.stderr)
        df = df[~duplicate_mask]
    return df


def set_device(user_choice):
    """ Sets calc. device depending on users choices and availability. 
    
    Parameters
    ----------
    user_choice : str
        Device set by user as an argument to DeepNOG call.

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


def main():
    """ DeepNOG command line tool. """
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Sanity check command line arguments
    if args.batch_size <= 0:
        sys.exit(f'ArgumentError: Batch size must be at least one.')
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
    if args.verbose >= 2:
        print(f'Device set to "{device}"')

    # Load neural network parameters
    if args.verbose >= 2:
        print(f'Loading NN-parameters from {weights_path} ...')
    model_dict = torch.load(weights_path, map_location=device)
    # Load neural network model
    model = load_nn(args.architecture, model_dict, device)
    # Load class names
    class_labels = model_dict['classes']

    # Load dataset
    if args.verbose >= 2:
        print(f'Accessing dataset from {args.file} ...')
    dataset = ds.ProteinDataset(args.file, f_format=args.fformat)

    # Predict labels of given data
    if args.verbose >= 2:
        print(f'Predicting protein families ...')
        if args.verbose >= 3:
            print(f'Process {args.batch_size} sequences per iteration: ')
    preds, confs, ids, indices = predict(model, dataset, device,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         verbose=args.verbose)

    # If given, set confidence threshold for prediction
    threshold = None
    if hasattr(model, 'threshold'):
        threshold = model.threshold
    # Construct results dataframe
    df = create_df(class_labels, preds, confs, ids, indices, 
                   threshold=threshold, device=device, verbose=args.verbose)

    # Construct path to save prediction
    if os.path.isdir(args.out):
        save_file = os.path.join(args.out, 'out.csv')
    else:
        save_file = args.out
    # Write to file
    if args.verbose >= 2:
        print(f'Writing prediction to {save_file}')
    columns = ['sequence_id', 'prediction', 'confidence']
    if args.tab:
        df.to_csv(save_file, sep='\t', index=False, columns=columns)
    else:
        df.to_csv(save_file, sep=';', index=False, columns=columns)
    if args.verbose >= 2:
        print(f'Finished magic.')
    return


if __name__ == '__main__':
    main()
