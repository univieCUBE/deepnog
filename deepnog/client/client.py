#!/usr/bin/env python3
"""
Authors
-------
- Roman Feldbauer
- Lukas Gosch

Date
----
2019-10-18

Usage
-----
`python client.py --help`

Description
-----------
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

from deepnog.utils.config import get_config

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
        description=('Assign protein sequences to orthologous groups '
                     'with deep learning.'))
    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'%(prog)s {__version__}')

    # Obtain a list of available models (databases)
    config = get_config()
    available_databases = list(config['database'].keys())
    available_architectures = list(config['architecture'].keys())

    subparsers = parser.add_subparsers(dest='phase', required=True)
    parser_train = subparsers.add_parser(
        'train', help='Train a model for a custom database.')
    parser_infer = subparsers.add_parser(
        'infer', help='Infer protein orthologous groups')

    # Arguments for both training and inference
    for p in [parser_train, parser_infer]:
        p.add_argument("-ff", "--fformat",
                       type=str,
                       metavar='FILEFORMAT',
                       default='fasta',
                       help=("File format of protein sequences. Must be "
                             "supported by Biopythons Bio.SeqIO class."))
        p.add_argument("-V", "--verbose",
                       type=int,
                       metavar='VERBOSE',
                       default=3,
                       help=("Define verbosity of DeepNOGs output written to "
                             "stdout or stderr. 0 only writes errors to "
                             "stderr which cause DeepNOG to abort and exit. "
                             "1 also writes warnings to stderr if e.g. a "
                             "protein without an ID was found and skipped. "
                             "2 additionally writes general progress "
                             "messages to stdout. "
                             "3 includes a dynamic progress bar of the "
                             "prediction stage using tqdm."
                             ))
        p.add_argument("-d", "--device",
                       type=str,
                       default='auto',
                       choices=['auto', 'cpu', 'gpu', ],
                       help=("Define device for calculating protein sequence "
                             "classification. Auto chooses GPU if available, "
                             "otherwise CPU."))
        p.add_argument("-nw", "--num-workers",
                       type=int,
                       metavar='NUM_WORKERS',
                       default=0,
                       help=('Number of subprocesses (workers) to use for '
                             'data loading. '
                             'Set to a value <= 0 to use single-process '
                             'data loading. '
                             'Note: Only use multi-process data loading if '
                             'you are calculating on a gpu '
                             '(otherwise inefficient)!'))
        p.add_argument("-a", "--architecture",
                       default="deepnog",
                       choices=available_architectures,
                       help="Network architecture to use for classification.")
        p.add_argument("-w", "--weights",
                       metavar='WEIGHTS_FILE',
                       help="Custom weights file path (optional)")
        p.add_argument("-bs", "--batch-size",
                       type=int,
                       metavar='BATCH_SIZE',
                       default=64,
                       help=('The batch size determines how many sequences are '
                             'processed by the network at once. '
                             'If 1, process the protein sequences sequentially '
                             '(recommended on CPUs). '
                             'Larger batch sizes speed up the inference '
                             'and training on GPUs. '
                             'Batch size can influence the learning process.'))

    # Arguments with different help for training vs. inference
    parser_infer.add_argument("-o", "--out",
                              metavar='OUT_FILE',
                              default=None,
                              help=("Store orthologous group predictions to output"
                                    "file. Per default, write predictions to stdout."))
    parser_train.add_argument("-o", "--out",
                              metavar='OUT_DIR',
                              required=True,
                              help=("Store training results to files in the given "
                                    "directory. Results include the trained model,"
                                    "training/validation loss and accuracy values,"
                                    "and the ground truth plus predicted classes "
                                    "per training epoch, if requested."))
    parser_infer.add_argument("-db", "--database",
                              type=str,
                              choices=available_databases,
                              default='eggNOG5',
                              help="Orthologous group/family database to use.")
    parser_train.add_argument("-db", "--database",
                              type=str,
                              required=True,
                              metavar='DATABASE_NAME',
                              help="Orthologous group database name")
    parser_infer.add_argument("-t", "--tax",
                              type=str,
                              default='2',
                              metavar='TAXONOMIC_LEVEL',
                              help="Taxonomic level to use in specified database, "
                                   "e.g. 1 = root, 2 = bacteria")
    parser_train.add_argument("-t", "--tax",
                              type=str,
                              required=True,
                              metavar='TAXONOMIC_LEVEL',
                              help="Taxonomic level in specified database")

    # Arguments for INFERENCE only
    parser_infer.add_argument("file",
                              metavar="SEQUENCE_FILE",
                              help=("File containing protein sequences for "
                                    "orthology inference."))
    parser_infer.add_argument("--test_labels",
                              metavar="TEST_LABELS_FILE",
                              required=False,
                              default=None,
                              help="Measure model performance on a test set. If provided, this "
                                   "file must contain the ground-truth labels for the provided "
                                   "sequences. Otherwise, only perform inference.")
    parser_infer.add_argument("-of", "--outformat",
                              default="csv",
                              choices=["csv", "tsv", "legacy"],
                              help="Output file format")
    parser_infer.add_argument("-c", "--confidence-threshold",
                              metavar='CONFIDENCE',
                              type=float,
                              default=None,
                              help="If provided, predictions below the threshold are discarded."
                                   "By default, any confidence threshold stored in the model is "
                                   "applied, if present.")

    # Arguments for TRAINING only
    parser_train.add_argument("training_sequences",
                              metavar='TRAIN_SEQUENCE_FILE',
                              help="File containing protein sequences training set.")
    parser_train.add_argument("validation_sequences",
                              metavar='VAL_SEQUENCE_FILE',
                              help="File containing protein sequences validation set.")
    parser_train.add_argument("training_labels",
                              metavar='TRAIN_LABELS_FILE',
                              help="Orthologous group labels for training set protein sequences.")
    parser_train.add_argument("validation_labels",
                              metavar='VAL_LABELS_FILE',
                              help="Orthologous group labels for training and validation set "
                                   "protein sequences. Both training and validation labels "
                                   "Must be in CSV files that are parseable "
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
    parser_train.add_argument("-g", "--gamma",
                              metavar="LEARNING_RATE_DECAY",
                              type=float,
                              default=0.75,
                              help="Decay for learning rate step scheduler. "
                                   "(lr_epoch_t2 = gamma * lr_epoch_t1)")
    parser_train.add_argument("-l2", "--l2-coeff",
                              metavar="\u03BB",  # lower-case lambda
                              type=float,
                              default=None,
                              help="Regularization coefficient \u03BB for "
                                   "L2 regularization. If None, L2 regularization "
                                   "is disabled.")
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
        logger.error(f'Batch size must be at least one. '
                     f'Got batch size = {args.batch_size} instead.')
        sys.exit(1)

    # Better safe than sorry -- don't overwrite existing files
    if args.out is not None:
        if Path(args.out).is_file():
            logger.error(f'Output file {args.out} already exists.')
            sys.exit(1)
        elif args.phase == 'infer' and (Path(args.out).is_dir() or args.out.endswith('/')):
            logger.error(f'Output path must be a file during inference, '
                         f'but got a directory instead: {args.out}')
            sys.exit(1)

    # Set up device
    args.device = set_device(args.device)

    # Get path to deep network architecture
    config = get_config()
    module = config['architecture'][args.architecture]['module']
    cls = config['architecture'][args.architecture]['class']

    if args.phase == 'infer':
        return _start_inference(args=args, arch_module=module, arch_cls=cls)
    elif args.phase == 'train':
        return _start_training(args=args, arch_module=module, arch_cls=cls)


def _start_inference(args, arch_module, arch_cls):
    from pandas import read_csv, DataFrame
    import torch
    from deepnog.data import ProteinIterableDataset
    from deepnog.learning import predict
    from deepnog.utils import create_df, get_logger, get_weights_path, load_nn
    from deepnog.utils.metrics import estimate_performance

    logger = get_logger(__name__, verbose=args.verbose)
    # Intra-op parallelization appears rather inefficient.
    # Users may override with environmental variable: export OMP_NUM_THREADS=8
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
    dataset = ProteinIterableDataset(args.file,
                                     labels_file=args.test_labels,
                                     f_format=args.fformat)

    # Load class names
    try:
        class_labels = model_dict['classes']
    except KeyError:
        class_labels = dataset.label_encoder.classes_

    # Load neural network model
    model = load_nn(architecture=(arch_module, arch_cls),
                    model_dict=model_dict,
                    phase=args.phase,
                    device=args.device)

    # If given, set confidence threshold for prediction
    if args.confidence_threshold is not None:
        if 0.0 < args.confidence_threshold <= 1.0:
            threshold = float(args.confidence_threshold)
        else:
            logger.error(f'Invalid confidence threshold specified: '
                         f'{args.confidence_threshold} not in range (0, 1].')
            sys.exit(1)
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

    if args.out is None:
        save_file = sys.stdout
        logger.info('Writing predictions to stdout')
    else:
        save_file = args.out
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing prediction to {save_file}')

    columns = ['sequence_id', 'prediction', 'confidence']
    separator = {'csv': ',', 'tsv': '\t', 'legacy': ';'}.get(args.outformat)
    df.to_csv(save_file, sep=separator, index=False, columns=columns)

    # Measure test set performance, if labels were provided
    if args.test_labels is not None:
        if args.out is None:
            perf_file = sys.stderr
            logger.info('Writing test set performance to stderr')
        else:
            perf_file = Path(save_file).with_suffix('.performance.csv')
            logger.info(f'Writing test set performance to {perf_file}')
        # Ensure object dtype to avoid int-str mismatches
        df_true = read_csv(args.test_labels, dtype=object, index_col=0)
        df = df.astype(dtype={columns[1]: object})
        perf = estimate_performance(df_true=df_true, df_pred=df)
        df_perf = DataFrame(data=[perf, ])
        df_perf['experiment'] = args.file
        df_perf.to_csv(perf_file, )
    logger.info('All done.')
    return


def _start_training(args, arch_module, arch_cls):
    import random
    import string
    import numpy as np
    from pandas import DataFrame
    import torch
    from deepnog.learning import fit
    from deepnog.utils import get_logger

    logger = get_logger(__name__, verbose=args.verbose)

    if args.n_epochs <= 0:
        logger.error(f'Number of epochs must be greater than or equal '
                     f'one. Got n_epochs = {args.n_epochs} instead.')
        sys.exit(1)
    out_dir = Path(args.out)
    logger.info(f'Output directory: {out_dir} (creating, if necessary)')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Add random letters to files to avoid name collisions
    while True:
        random_letters = ''.join(random.sample(string.ascii_letters, 4))
        if not any([random_letters in str(f) for f in out_dir.iterdir()]):
            break  # if these letters were not used previously
    experiment_name = f'deepnog_custom_model_{args.database}_{args.tax}_{random_letters}'
    model_file = out_dir/f'{experiment_name}_model.pth'
    eval_file = out_dir/f'{experiment_name}_eval.csv'
    classes_file = out_dir/f'{experiment_name}_labels.npz'

    results = fit(architecture=args.architecture,
                  module=arch_module,
                  cls=arch_cls,
                  training_sequences=args.training_sequences,
                  validation_sequences=args.validation_sequences,
                  training_labels=args.training_labels,
                  validation_labels=args.validation_labels,
                  data_loader_params={'batch_size': args.batch_size,
                                      'num_workers': args.num_workers},
                  learning_rate=args.learning_rate,
                  learning_rate_params={'step_size': 1,
                                        'gamma': args.gamma,
                                        'last_epoch': -1,
                                        },
                  l2_coeff=args.l2_coeff,
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
