"""
Author: Lukas Gosch
Date: 2019-10-18
Usage: python deepnog.py --help
Description:
    DeepNOG predicts protein families/orthologous groups of given
    protein sequences with deep learning.

    File formats supported:
    Preferred: FASTA
    DeepNOG supports protein sequences stored in all file formats listed in
    https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
    only.

    Architectures supported:

    Databases supported:
        - eggNOG 5.0, taxonomic level 1 (root)
        - eggNOG 5.0, taxonomic level 2 (bacteria)
"""
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import os.path

from . import __version__


def get_parser():
    """ Creates a new argument parser.

    Returns
    -------
    parser : ArgumentParser
        ArgumentParser object to parse program arguments.
    """
    parser = argparse.ArgumentParser(
        usage='%(prog)s proteins.faa --out proteins.csv',
        description=('Predict orthologous groups from protein sequences '
                     'with deep learning.'))
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument("file",
                        metavar='SEQUENCE_FILE',
                        help=("File containing protein sequences for "
                              "classification."))
    parser.add_argument("-o", "--out",
                        metavar='FILE',
                        default='out.csv',
                        help=("Store orthologous group predictions to output"
                              "file (default format CSV)"))
    parser.add_argument("-ff", "--fformat",
                        type=str,
                        default='fasta',
                        help=("File format of protein sequences. Must be "
                              "supported by Biopythons Bio.SeqIO class "
                              "(default: fasta)"))
    parser.add_argument("-db", "--database",
                        type=str,
                        choices=['eggNOG5', ],
                        default='eggNOG5',
                        help="Orthologous group/family database to use "
                             "(default: eggNOG5)")
    parser.add_argument("-t", "--tax",
                        type=int,
                        choices=[1, 2, ],
                        default=2,
                        help="Taxonomic level to use in specified database "
                             "(default: 2 = bacteria)")
    parser.add_argument("--verbose",
                        type=int,
                        default=3,
                        help=("Define verbosity of DeepNOGs output written to "
                              "stdout or stderr. 0 only writes errors to "
                              "stderr which cause DeepNOG to abort and exit. "
                              "1 also writes warnings to stderr if e.g. a "
                              "protein without an ID was found and skipped. "
                              "2 additionally writes general progress "
                              "messages to stdout."
                              "3 includes a dynamic progress bar of the "
                              "prediction stage using tqdm."))
    parser.add_argument("-d", "--device",
                        type=str,
                        default='auto',
                        choices=['auto', 'cpu', 'gpu', ],
                        help=("Define device for calculating protein sequence "
                              "classification. Auto chooses GPU if available, "
                              "otherwise CPU (default: auto)"))
    parser.add_argument("-nw", "--num-workers",
                        type=int,
                        default=0,
                        help=('Number of subprocesses (workers) to use for '
                              'data loading. '
                              'Set to a value <= 0 to use single-process '
                              'data loading. '
                              'Note: Only use multi-process data loading if '
                              'you are calculating on a gpu '
                              '(otherwise inefficient)! Default: 0'))
    parser.add_argument("-a", "--architecture",
                        default='deepencoding',
                        choices=['deepencoding', ],
                        help="Network architecture to use for classification "
                             "(default: deepencoding)")
    parser.add_argument("-w", "--weights",
                        metavar='FILE',
                        help="Custom weights file path (optional)")
    parser.add_argument("--tab", action='store_true',
                        help='Use tab-separation in output')
    parser.add_argument("-bs", "--batch-size",
                        type=int,
                        default=1,
                        help=('Batch size used for prediction.'
                              'Defines how many sequences should be '
                              'forwarded in the network at once. '
                              'With a batch size of one, the protein '
                              'sequences are sequentially classified by '
                              'the network without leveraging parallelism. '
                              'Higher batch-sizes than the default can '
                              'speed up the prediction significantly '
                              'if on a gpu. '
                              'On a cpu, however, they can be slower than '
                              'smaller ones due to the increased average '
                              'sequence length in the convolution step due to '
                              'zero-padding every sequence in each batch.'))
    return parser


def start_prediction(args):
    # Importing here makes CLI more snappy
    import torch
    from .dataset import ProteinDataset
    from .inference import load_nn, predict
    from .io import create_df, get_weights_path
    from .utils import set_device

    # Sanity check command line arguments
    if args.batch_size <= 0:
        sys.exit(f'ArgumentError: Batch size must be at least one.')
    # Construct path to saved parameters of NN
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = get_weights_path(database=args.database,
                                        level=str(args.tax),
                                        architecture=args.architecture,
                                        )
    # Set up device
    try:
        device = set_device(args.device)
    except RuntimeError as err:
        sys.exit(f'RuntimeError: {err} \nLeaving the ship and aborting '
                 f'calculations.')
    if args.verbose >= 2:
        print(f'Device set to "{device}"')

    # Set number of threads to 1 automatic (internal) parallelization is
    # quite inefficient
    torch.set_num_threads(1)

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
    dataset = ProteinDataset(args.file, f_format=args.fformat)

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
                   threshold=threshold, verbose=args.verbose)

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


def main():
    """ DeepNOG command line tool. """
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    start_prediction(args)


if __name__ == '__main__':
    main()
