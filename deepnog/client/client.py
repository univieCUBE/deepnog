#!/usr/bin/env python3
"""
Author: Lukas Gosch
        Roman Feldbauer

Date: 2019-10-18

Usage: python client.py --help

Description:

    Provides the ``deepnog`` command line client and entry point for users.

    DeepNOG predicts protein families/orthologous groups of given
    protein sequences with deep learning.

    Since version 1.2, model training is available as well.

    File formats supported:
    Preferred: FASTA
    DeepNOG supports protein sequences stored in all file formats listed in
    https://biopython.org/wiki/SeqIO but is tested for the FASTA-file format
    only.

    Architectures supported:

    Databases supported:
        - eggNOG 5.0, taxonomic level 1 (root)
        - eggNOG 5.0, taxonomic level 2 (bacteria)
        - Additional databases will be trained on demand/users can add custom
          databases using the training facilities.
"""
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path
import sys

__all__ = ['main',
           ]


def _get_parser():
    """ Create a new argument parser.

    Returns
    -------
    parser : ArgumentParser
        Program arguments including inference/training and many more
    """
    from deepnog import __version__
    parser = argparse.ArgumentParser(
        description=('Assign protein sequences to orthologous groups'
                     'with deep learning.'))
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='phase', required=True)
    parser_train = subparsers.add_parser(
        'train', help='Train a model for a custom database.')
    parser_infer = subparsers.add_parser(
        'infer', help='Infer protein orthologous groups')

    # Arguments for both training and inference
    for p in [parser_train, parser_infer]:
        p.add_argument("-ff", "--fformat",
                       type=str,
                       metavar='STR',
                       default='fasta',
                       help=("File format of protein sequences. Must be "
                             "supported by Biopythons Bio.SeqIO class."))
        p.add_argument("--verbose",
                       type=int,
                       metavar='INT',
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
        p.add_argument("-d", "--device",
                       type=str,
                       default='auto',
                       choices=['auto', 'cpu', 'gpu', ],
                       help=("Define device for calculating protein sequence "
                             "classification. Auto chooses GPU if available, "
                             "otherwise CPU."))
        p.add_argument("-nw", "--num-workers",
                       type=int,
                       metavar='INT',
                       default=0,
                       help=('Number of subprocesses (workers) to use for '
                             'data loading. '
                             'Set to a value <= 0 to use single-process '
                             'data loading. '
                             'Note: Only use multi-process data loading if '
                             'you are calculating on a gpu '
                             '(otherwise inefficient)!'))
        p.add_argument("-a", "--architecture",
                       default='deepencoding',
                       choices=['deepencoding', ],
                       help="Network architecture to use for classification.")
        p.add_argument("-w", "--weights",
                       metavar='FILE',
                       help="Custom weights file path (optional)")
        p.add_argument("-bs", "--batch-size",
                       type=int,
                       metavar='INT',
                       default=64,
                       help=('Batch size used for prediction or training.'
                             'Defines how many sequences should be '
                             'processed in the network at once. '
                             'With a batch size of one, the protein '
                             'sequences are sequentially processed by '
                             'the network without leveraging parallelism. '
                             'Higher batch-sizes than the default can '
                             'speed up the inference and training '
                             'significantly, especially if on a gpu. '
                             'On a cpu, however, they can be slower than '
                             'smaller ones due to the increased average '
                             'sequence length in the convolution step due to '
                             'zero-padding every sequence in each batch.'))

    # Arguments with different help for training vs. inference
    parser_infer.add_argument("-o", "--out",
                              metavar='FILE',
                              default=None,
                              help=("Store orthologous group predictions to output"
                                    "file. Per default, write predictions to stdout."))
    parser_train.add_argument("-o", "--out",
                              metavar='DIR',
                              required=True,
                              help=("Store training results to files in the given "
                                    "directory. Results include the trained model,"
                                    "training/validation loss and accuracy values,"
                                    "and the ground truth plus predicted classes "
                                    "per training epoch."))
    parser_infer.add_argument("-db", "--database",
                              type=str,
                              choices=['eggNOG5', ],
                              default='eggNOG5',
                              help="Orthologous group/family database to use.")
    parser_train.add_argument("-db", "--database",
                              type=str,
                              required=True,
                              help="Orthologous group database name")
    parser_infer.add_argument("-t", "--tax",
                              type=int,
                              choices=[1, 2, ],
                              default=2,
                              help="Taxonomic level to use in specified database "
                                   "(1 = root, 2 = bacteria)")
    parser_train.add_argument("-t", "--tax",
                              type=int,
                              required=True,
                              help="Taxonomic level in specified database")

    # Arguments for INFERENCE only
    parser_infer.add_argument("file",
                              metavar='SEQUENCE_FILE',
                              help=("File containing protein sequences for "
                                    "classification (inference or training)."))
    parser_infer.add_argument("-of", "--outformat",
                              default="csv",
                              choices=["csv", "tsv", "legacy"],
                              help="The file format of the output file produced by deepnog.")
    parser_infer.add_argument("-c", "--confidence-threshold",
                              metavar='FLOAT',
                              type=float,
                              default=None,
                              help="The confidence value below which predictions are "
                                   "masked by deepnog. By default, apply the confidence "
                                   "threshold saved in the model if one exists, "
                                   "and else do not apply a confidence threshold.")

    # Arguments for TRAINING only
    parser_train.add_argument("training_sequences",
                              metavar='TRAIN_SEQUENCE_FILE',
                              help="File containing protein sequences training set.")
    parser_train.add_argument("validation_sequences",
                              metavar='VAL_SEQUENCE_FILE',
                              help="File containing protein sequences validation set.")
    parser_train.add_argument("labels",
                              metavar='LABELS_FILE',
                              help="Orthologous group labels for training and validation set "
                                   "protein sequences. Must be a CSV file and parseable "
                                   "by pandas.read_csv(..., index_col=1). The first column "
                                   "must be a numerical index. The other columns should "
                                   "be named 'protein_id' and 'eggnog_id', or be in order "
                                   "sequence_identifier first, label_identifier second.")
    parser_train.add_argument("-e", "--n-epochs",
                              metavar='N_EPOCHS',
                              type=int,
                              default=15,
                              help="Number of training epochs, that is, "
                                   "passes over the complete data set.")
    parser_train.add_argument("-s", "--shuffle",
                              action='store_true',
                              help=f'Shuffle the training sequences. Note that a shuffle '
                                   f'buffer is used in combination with an iterable dataset. '
                                   f'That is, not all sequences have equal probability to '
                                   f'be chosen. If you have highly structured sequence files '
                                   f'consider shuffling them in advance. '
                                   f'Default buffer size = {2**16}')
    parser_train.add_argument("-lr", "--learning-rate",
                              metavar='LEARNING_RATE',
                              type=float,
                              default=1e-2,
                              help='Initial learning rate, subject to adaptations by '
                                   'chosen optimizer and scheduler.')
    parser_train.add_argument("-r", "--random-seed",
                              metavar='RANDOM_SEED',
                              type=int,
                              default=None,
                              help='Seed the random number generators of numpy and PyTorch '
                                   'during training for reproducibility. Also affects cuDNN '
                                   'determinism. Default: None (disables reproducibility)')
    parser_train.add_argument("--save-each-epoch",
                              action='store_true',
                              default=False,
                              help='Save the model after each epoch.')
    return parser


def _start_prediction_or_training(args):
    # Importing here makes CLI more snappy
    from deepnog.utils import get_logger, set_device

    logger = get_logger(__name__, verbose=args.verbose)
    logger.info('Starting deepnog')

    # Sanity check command line arguments
    if args.batch_size <= 0:
        raise ValueError(f'Batch size must be at least one. '
                         f'Got batch size = {args.batch_size} instead.')

    # Better safe than sorry -- don't overwrite existing files
    if args.out is not None and Path(args.out).is_file():
        raise FileExistsError(f'Output file {args.out} already exists.')

    # Set up device
    args.device = set_device(args.device)

    if args.phase == 'infer':
        return _start_inference(args=args)
    elif args.phase == 'train':
        return _start_training(args=args)


def _start_inference(args):
    import torch
    from deepnog.data import ProteinDataset
    from deepnog.learning import predict
    from deepnog.utils import create_df, get_logger, get_weights_path, load_nn

    logger = get_logger(__name__, verbose=args.verbose)
    # Set number of threads to 1, b/c automatic (internal) parallelization is
    # quite inefficient
    torch.set_num_threads(1)

    # Construct path to saved parameters of NN
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = get_weights_path(database=args.database,
                                        level=str(args.tax),
                                        architecture=args.architecture,
                                        verbose=args.verbose,
                                        )
    # Load neural network parameters
    logger.info(f'Loading NN-parameters from {weights_path} ...')
    model_dict = torch.load(weights_path, map_location=args.device)

    # Load dataset
    logger.info(f'Accessing dataset from {args.file} ...')
    dataset = ProteinDataset(args.file, f_format=args.fformat)

    # Load class names
    try:
        class_labels = model_dict['classes']
    except KeyError:
        class_labels = dataset.label_encoder.classes_

    # Load neural network model
    model = load_nn(architecture=args.architecture,
                    model_dict=model_dict,
                    phase=args.phase,
                    device=args.device)

    # If given, set confidence threshold for prediction
    if args.confidence_threshold is not None:
        if 0.0 < args.confidence_threshold <= 1.0:
            threshold = float(args.confidence_threshold)
        else:
            raise ValueError(f'Invalid confidence threshold specified: '
                             f'{args.confidence_threshold} not in range '
                             f'(0, 1].')
    elif hasattr(model, 'threshold'):
        threshold = float(model.threshold)
        logger.info(f'Applying confidence threshold from model: {threshold}')
    else:
        threshold = None

    # Predict labels of given data
    logger.info('Starting protein sequence group/family inference ...')
    logger.debug(f'Processing {args.batch_size} sequences per iteration (minibatch)')
    preds, confs, ids, indices = predict(model, dataset, args.device,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         verbose=args.verbose)

    # Construct results dataframe
    df = create_df(class_labels, preds, confs, ids, indices, threshold=threshold)

    if args.out is not None:
        save_file = args.out
        logger.info(f'Writing prediction to {save_file}')
    else:
        save_file = sys.stdout
        logger.info('Writing predictions to stdout')

    columns = ['sequence_id', 'prediction', 'confidence']
    separator = {'csv': ',', 'tsv': '\t', 'legacy': ';'}.get(args.outformat)
    df.to_csv(save_file, sep=separator, index=False, columns=columns)
    logger.info('All done.')
    return


def _start_training(args):
    import random
    import string
    import numpy as np
    from pandas import DataFrame
    import torch
    from deepnog.learning import fit
    from deepnog.utils import get_logger

    logger = get_logger(__name__, verbose=args.verbose)

    if args.n_epochs <= 0:
        raise ValueError(f'Number of epochs must be greater than or equal '
                         f'one. Got n_epochs = {args.n_epochs} instead.')
    out_dir = Path(args.out)
    logger.info(f'Output directory: {out_dir} (creating, if necessary)')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Add random letters to files to avoid name collisions
    while True:
        random_letters = ''.join(random.sample(string.ascii_letters, 4))
        if not any([random_letters in str(f) for f in out_dir.iterdir()]):
            break  # if these letters were not used previously
    experiment_name = f'deepnog_custom_model_{args.database}_{args.tax}_{random_letters}'
    model_file = out_dir/f'{experiment_name}_model.pt'
    eval_file = out_dir/f'{experiment_name}_eval.csv'
    classes_file = out_dir/f'{experiment_name}_labels.npz'

    results = fit(architecture=args.architecture,
                  training_sequences=args.training_sequences,
                  validation_sequences=args.validation_sequences,
                  data_loader_params={'batch_size': args.batch_size,
                                      'num_workers': args.num_workers},
                  learning_rate=args.learning_rate,
                  labels=args.labels,
                  device=args.device,
                  verbose=args.verbose,
                  n_epochs=args.n_epochs,
                  shuffle=args.shuffle,
                  random_seed=args.random_seed,
                  out_dir=out_dir,
                  experiment_name=experiment_name,
                  save_each_epoch=args.save_each_epoch,
                  # TODO add the rest of the parameters to the client
                  )

    # Save model to output dir
    logger.info(f'Saving model to {model_file}...')
    torch.save({'classes': results.training_dataset.label_encoder.classes_,
                'model_state_dict': results.model.state_dict()},
               model_file)
    # Save a dataframe of several training/validation statistics
    logger.info(f'Saving evaluation statistics to {eval_file}... '
                f'Load with pandas.read_csv().')
    DataFrame(results.evaluation).to_csv(eval_file)
    # Save ground-truth and predicted classes for further performance analysis
    logger.info(f'Saving ground truth (y_true) and predicted (y_pred) '
                f'labels (from training/validation) to {classes_file}... '
                f'Load with numpy.load().')
    np.savez(classes_file,
             y_train_true=results.y_train_true,
             y_train_pred=results.y_train_pred,
             y_val_true=results.y_val_true,
             y_val_pred=results.y_val_pred)

    logger.info('All done.')
    return


def main():
    """ DeepNOG command line tool. """
    parser = _get_parser()
    args = parser.parse_args()
    _start_prediction_or_training(args)


if __name__ == '__main__':
    main()
