"""
Author: Roman Feldbauer

Date: 2020-06-03

Description:

    Training deep networks for protein orthologous group prediction.
"""
# SPDX-License-Identifier: BSD-3-Clause

import copy
from datetime import datetime
from functools import partial
from os import environ
from pathlib import Path
import random
import string
import sys
import tempfile
import time
from typing import Dict, List, Union, NamedTuple
import warnings

import numpy as np
from tqdm.auto import tqdm

from ..data import ProteinDataset, ProteinIterableDataset, ShuffledProteinIterableDataset
from ..data import collate_sequences
from ..utils import count_parameters, get_config, get_logger, load_nn, set_device
from ..utils import try_import_pytorch

torch = try_import_pytorch()
from torch.optim import Adam, lr_scheduler  # noqa
from torch.utils.data import DataLoader  # noqa
from torch.utils.tensorboard import SummaryWriter  # noqa

__all__ = ['fit',
           ]

train_val_result = NamedTuple('train_val_result',
                              [('model', torch.nn.Module),
                               ('training_dataset', torch.utils.data.Dataset),
                               ('validation_dataset', torch.utils.data.Dataset),
                               ('evaluation', List[dict]),
                               ('y_train_true', np.ndarray),
                               ('y_train_pred', np.ndarray),
                               ('y_val_true', np.ndarray),
                               ('y_val_pred', np.ndarray),
                               ])


def _train_and_validate_model(model: torch.nn.Module, criterion, optimizer,
                              scheduler, data_loaders: dict, *,
                              num_epochs=2,
                              tensorboard_exp=None,
                              stop_after: int = None,
                              l2_coeff=None,
                              log_interval: int = 100,
                              device: torch.device = 'cuda',
                              validation_only: bool = False,
                              early_stopping: int = 0,
                              save_each_epoch: bool = True,
                              out_dir: Path = None,
                              experiment_name: str = None,
                              verbose: int = 2) -> train_val_result:
    """ Perform training and validation of a given model, data, and hyperparameters.

    Parameters
    ----------
    model : torch.nn.Module
        Deep network PyTorch module
    criterion
        PyTorch loss function, e.g., CrossEntropyLoss
    optimizer
        PyTorch optimizer instance, e.g., Adam
    scheduler
        PyTorch learning rate scheduler
    data_loaders : dict of torch.utils.DataLoader
        PyTorch DataLoaders (Dataset plus some parameters) for train and val set.
    num_epochs : int
        Maximum number of training passes over the dataset
    tensorboard_exp : [None, str], optional
        If not None, path to Tensorboard files for monitoring training progress
    stop_after : int, optional
        If not None, end training when ``stop_after`` sequences have been processed
    l2_coeff : float, optional
        Optional L2 regularization coefficient
    log_interval : int, optional
        Print intermediary results after ``log_interval`` minibatches
    device : torch.device
        Use either 'cpu' or 'cuda' (GPU) for training/validation.
    validation_only : bool, optional
        Skip the training phase, and only validate the given model
    early_stopping : int, optional
        If > 0, use early stopping: If validation accuracy does not improve for
        ``early_stopping`` epochs, end the training.
    save_each_epoch : bool
        Save the network after each training epoch
    out_dir : Path
        Path to the output directory used to save models during training
    experiment_name : str
        Prefix of model files saved during training
    verbose : int
        Increasing levels of messages

    Returns
    -------
    results : namedtuple
        A namedtuple containing:
         * the trained deep network model
         * training dataset
         * evaluation statistics
         * the ground truth labels (y_true)
         * the predicted labels (y_pred).
    """
    logger = get_logger(__name__, verbose=verbose)
    debug_force_mode = environ.get("DEEPNOG_FORCE_MODE", default=None)
    # Set up tensorboard
    if tensorboard_exp is not None:
        tensorboard_exp = Path(tensorboard_exp)
        tensorboard_writer = SummaryWriter(str(tensorboard_exp))
        logger.info(f'Tensorboard directory: {tensorboard_writer.log_dir}.')
    else:
        tensorboard_writer = None
        logger.info('Tensorboard disabled.')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_acc: float = 0.0
    best_acc: float = 0.0
    best_epoch: int = 0
    evaluation: list = []

    y_true: Dict[str, np.ndarray] = {phase: -np.ones((num_epochs, len(loader.dataset)),
                                                     dtype=np.int32)
                                     for phase, loader in data_loaders.items()}
    y_pred: Dict[str, np.ndarray] = {phase: -np.ones_like(y_true[phase])
                                     for phase, loader in data_loaders.items()}
    tqdm_disable: bool = verbose < 3

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            data_loader = data_loaders[phase]
            dataset = data_loader.dataset
            if phase == 'train':
                logger.debug('Setting model.train() mode')
                model.train()  # Set model to training mode
                if validation_only:
                    continue  # skip training
                else:
                    logger.info(f'Scheduler: learning rate = {scheduler.get_last_lr()}')
            else:
                logger.debug('Setting model.eval() mode')
                model.eval()
            if debug_force_mode is not None:
                if debug_force_mode.lower() == 'train':
                    logger.warning('DEBUG forcing model.train()')
                    model.train()
                elif debug_force_mode.lower() == 'eval':
                    logger.warning('DEBUG forcing model.eval()')
                    model.eval()

            running_loss: torch.float32 = 0.
            running_corrects: torch.int = 0

            log_loss: float = 0.
            log_corrects: int = 0
            log_n_objects: int = 0

            # Iterate over data.
            n_processed_sequences = 0
            tqdm_total = stop_after if stop_after else len(dataset)
            with tqdm(desc=f'deepnog {phase}',
                      total=tqdm_total,
                      mininterval=1.,
                      disable=tqdm_disable,
                      unit='seq',
                      unit_scale=True) as pbar:
                for batch_nr, batch in enumerate(data_loader):
                    sequence = batch.sequences
                    labels = batch.labels

                    inputs = sequence.to(device)
                    labels = labels.to(device)

                    # Update progress for TensorBoard
                    current_batch_size = len(inputs)
                    n_processed_sequences += current_batch_size

                    # Reset gradients
                    optimizer.zero_grad()

                    # forward pass;
                    # track history only during training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if l2_coeff is not None:
                            loss_ce = criterion(outputs, labels.long())
                            loss_reg = model.classification1.weight.pow(2).sum()
                            loss = loss_ce + l2_coeff * loss_reg
                        else:
                            loss = criterion(outputs, labels.long())
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # save to global ground truth and prediction arrays
                        start = n_processed_sequences - current_batch_size
                        end = n_processed_sequences
                        y_true[phase][epoch, start:end] = labels.detach().cpu().numpy()
                        y_pred[phase][epoch, start:end] = preds.detach().cpu().numpy()

                    # statistics
                    batch_loss = loss.item()
                    batch_corrects = torch.sum(preds == labels)

                    log_loss += float(batch_loss)
                    log_corrects += int(batch_corrects)
                    log_n_objects += len(labels)

                    if tensorboard_writer is not None and batch_nr % log_interval == 0:
                        tensorboard_writer.add_scalar(f'{phase}/loss',
                                                      log_loss,
                                                      n_processed_sequences)
                        tensorboard_writer.add_scalar(f'{phase}/accuracy',
                                                      log_corrects / log_n_objects,
                                                      n_processed_sequences)

                        # Reset the log loss/acc variables
                        log_loss = 0.
                        log_corrects = 0
                        log_n_objects = 0

                    running_loss += batch_loss * inputs.size(0)
                    running_corrects += batch_corrects

                    if stop_after and n_processed_sequences >= stop_after:
                        logger.info(f'Stopping after {n_processed_sequences} as '
                                    f'requested (stop_after = {stop_after}).')
                        break

                    # Update progress bar
                    pbar.update(n=current_batch_size)

            # Finishing train or val phase
            if n_processed_sequences == 0:
                logger.error('Empty dataset. Were labels provided for both '
                             'training and validation set?')
                sys.exit(1)

            epoch_loss = running_loss / n_processed_sequences
            epoch_acc = float(running_corrects) / n_processed_sequences
            evaluation.append({'phase': phase,
                               'epoch': epoch,
                               'accuracy': float(epoch_acc),
                               'loss': epoch_loss})
            if n_processed_sequences < len(data_loader):
                logger.warning(f'Not all sequences were processed in epoch {epoch}: '
                               f'n_processed = {n_processed_sequences} < '
                               f'n_total = {len(data_loader)}.')

            logger.info(f'{phase} --- loss: {epoch_loss:.4f}  --- acc: {epoch_acc:.4f}\n')

            if phase == 'train' and scheduler is not None:
                logger.debug('Learning rate scheduler.step()')
                scheduler.step()

            # empty cache if possible
            logger.debug('Emptying CUDA cache')
            torch.cuda.empty_cache()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                logger.debug(f'Validation performance improved in current '
                             f'epoch with accuracy {epoch_acc:.3f} > '
                             f'{best_acc:.3f} (previous best).')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # temporarily save network
        if save_each_epoch:
            if out_dir is None or experiment_name is None:
                logger.warning('Cannot save model after each epoch. '
                               'Please specify "experiment_name" '
                               'and "out_dir" as well.')
            else:
                save_file = out_dir/f'{experiment_name}_epoch{epoch:02d}.pth'
                logger.debug(f'Saving current epoch {epoch} model to {save_file}')
                torch.save({'classes': dataset.label_encoder.classes_,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'l2_coeff': l2_coeff, },
                           save_file)

        # early stopping (regularization)
        if early_stopping and epoch - early_stopping >= best_epoch:
            if epoch_acc < best_acc:
                logger.info(f'Early stopping due to decreasing scores for '
                            f'{early_stopping} epochs after best epoch.')
                break

    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    logger.info(f'Training complete in {minutes:.0f}m {seconds:.0f}s')
    logger.info(f'Best val acc: {best_acc:4f} in epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return train_val_result(model=model,
                            training_dataset=data_loaders['train'].dataset,
                            validation_dataset=data_loaders['val'].dataset,
                            evaluation=evaluation,
                            y_train_true=y_true['train'],
                            y_train_pred=y_pred['train'],
                            y_val_true=y_true['val'],
                            y_val_pred=y_pred['val'],
                            )


def fit(architecture, module, cls,
        training_sequences, validation_sequences,
        training_labels, validation_labels, *,
        data_loader_params: dict = None,
        iterable_dataset: bool = False,
        n_epochs: int = 15,
        shuffle: bool = False,
        learning_rate: float = 1e-2,
        learning_rate_params: dict = None,
        l2_coeff: float = None,
        optimizer_cls=Adam,
        device: Union[str, torch.device] = 'auto',
        tensorboard_dir: Union[None, str] = 'auto',
        log_interval: int = 100,
        random_seed: int = None,
        save_each_epoch: bool = True,
        out_dir: Path = None,
        experiment_name: str = None,
        config_file: str = None,
        verbose: int = 2,
        ) -> train_val_result:
    """ Perform training and validation of a given model, data, and hyperparameters.

        Parameters
        ----------
        architecture : str
            Network architecture, must be available in deepnog/models
        module : str
            Python module containing the network definition
            (inside deepnog/models/).
        cls : str
            Python class name of the network (inside deepnog/models/{module}.py).
        training_sequences : str, Path
            File with training set sequences
        validation_sequences : str, Path
            File with validation set sequences
        training_labels : str, Path
            File with class labels (orthologous groups) of training sequences
        validation_labels : str, Path
            File with class labels (orthologous groups) of validation sequences
        data_loader_params : dict
            Parameters passed to PyTorch DataLoader construction
        iterable_dataset : bool, default False
            Use an iterable dataset that does not load all sequences in advance.
            While this saves memory and does not involve the delay at start,
            random sampling is impaired, and requires a shuffle buffer.
        n_epochs : int
            Number of training passes over the complete training set
        shuffle : bool
            Shuffle the training data. This does NOT shuffle the complete data
            set, which requires having all sequences in memory, but uses a
            shuffle buffer (default size: 2**16), from which sequences are drawn.
        learning_rate : float
            Learning rate, the central hyperparameter of deep network training.
            Too high values may lead to diverging solutions, while too low
            values result in slow learning.
        learning_rate_params : dict
            Parameters passed to the learning rate Scheduler.
        l2_coeff : float
            If not None, regularize training by L2 norm of network weights
        optimizer_cls
            Class of PyTorch optimizer
        device : torch.device
            Use either 'cpu' or 'cuda' (GPU) for training/validation.
        tensorboard_dir : str
            Save online learning statistics for tensorboard in this directory.
        log_interval : int, optional
            Print intermediary results after ``log_interval`` minibatches
        random_seed : int
            Set a random seed for numpy/pytorch for reproducible results.
        save_each_epoch : bool
            Save the network after each training epoch
        out_dir : Path
            Path to the output directory used to save models during training
        experiment_name : str
            Prefix of model files saved during training
        config_file : str
            Override path to config file, e.g. for custom models in unit tests
        verbose : int
            Increasing levels of messages

        Returns
        -------
        results : namedtuple
            A namedtuple containing:
             * the trained deep network model
             * training dataset
             * evaluation statistics
             * the ground truth labels (y_true)
             * the predicted labels (y_pred).
        """
    logger = get_logger(__name__, verbose=verbose)

    device = set_device(device)
    logger.info(f'Training device: {device}')
    if random_seed is None:
        logger.info('Non-deterministic training mode (no random seed given).')
    else:
        logger.info(f'Deterministic training with seed = {random_seed}')
        # See https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PyTorch DataLoader default arguments
    default_data_loader_params = {'batch_size': 32,
                                  'num_workers': 4,
                                  'collate_fn': partial(
                                      collate_sequences,
                                      zero_padding=True),
                                  'pin_memory': True,
                                  }
    if data_loader_params is not None:
        default_data_loader_params.update(data_loader_params)
    data_loader_params = default_data_loader_params
    logger.debug(f'Data loader parameters: {data_loader_params}')
    if learning_rate_params is None:
        learning_rate_params = {'step_size': 1,
                                'gamma': 0.75,
                                'last_epoch': -1,
                                }
    logger.debug(f'Scheduler parameters: {learning_rate_params}')

    # Set up training and validation data set with sequences and labels
    dataset: dict = {}
    if iterable_dataset:
        if shuffle:
            buffer_size = 2 ** 16
            dataset['train'] = ShuffledProteinIterableDataset(file=training_sequences,
                                                              labels_file=training_labels,
                                                              buffer_size=buffer_size)
            logger.info(f'Using iterable dataset with shuffle buffer, '
                        f'and buffer size = {buffer_size}.')
        else:
            dataset['train'] = ProteinIterableDataset(file=training_sequences,
                                                      labels_file=training_labels)
            logger.info('Using iterable dataset without shuffling.')
        dataset['val'] = ProteinIterableDataset(file=validation_sequences,
                                                labels_file=validation_labels,
                                                label_encoder=dataset['train'].label_encoder,
                                                )
    else:
        with_shuffling = 'with shuffling' if shuffle else 'without shuffling'
        logger.info(f'Using in-memory dataset {with_shuffling}.')
        logger.info('Loading training dataset')
        dataset['train'] = ProteinDataset(sequences=training_sequences,
                                          labels=training_labels,
                                          verbose=verbose)
        logger.info('Loading validation dataset')
        dataset['val'] = ProteinDataset(sequences=validation_sequences,
                                        labels=validation_labels,
                                        label_encoder=dataset['train'].label_encoder,
                                        verbose=verbose)
        data_loader_params.update({'shuffle': shuffle})
    data_loader = {phase: DataLoader(d, **data_loader_params)
                   for phase, d in dataset.items()}

    # Deep network hyperparameter default values
    config = get_config(config_file)
    model_dict = config['architecture'][architecture]
    model_dict['n_classes'] = len(dataset['train'].label_encoder.classes_)
    logger.debug(f'Network specs: {model_dict}')

    # Tensorboard experiment name (filename)
    if tensorboard_dir is None:
        experiment = None  # do not use tensorboard
    elif tensorboard_dir == 'auto':
        now = datetime.now().strftime("%Y-%m-%d_%H-%m-%S_%f")
        random_letters = ''.join(random.sample(string.ascii_letters, 4))
        tmp_dir = Path(tempfile.gettempdir())/'tensorboard'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        experiment = tmp_dir/f'deepnog_{now}_{random_letters}'
    else:
        try:
            experiment = Path(tensorboard_dir)
        except TypeError:
            warnings.warn('Invalid value for "tensorboard" argument. '
                          'Must be one of: None, "auto", or a valid path. '
                          'Continuing without tensorboard report.')
            experiment = None

    # Load model, and send to selected device. Set up training.
    model = load_nn(architecture=(module, cls),
                    model_dict=model_dict,
                    phase='train',
                    device=device)
    # NOTE: CrossEntropyLoss is LogSoftmax+NLLoss, that is, no softmax layer
    # should be in the forward pass of the network.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(),
                              lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, **learning_rate_params)

    # Trainable parameters
    logger.info(f'Network architecture: {architecture}')
    logger.info(f'Learning criterion: {criterion}')
    logger.info(f'Optimizer: {optimizer}')
    logger.info(f'Learning rate scheduler: {scheduler}')
    logger.info(f'Number of classes: {model.n_classes}')
    logger.info(f'Tunable parameters: {count_parameters(model)}')

    result = _train_and_validate_model(model=model,
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       data_loaders=data_loader,
                                       num_epochs=n_epochs,
                                       tensorboard_exp=experiment,
                                       l2_coeff=l2_coeff,
                                       log_interval=log_interval,
                                       device=device,
                                       save_each_epoch=save_each_epoch,
                                       out_dir=out_dir,
                                       experiment_name=experiment_name,
                                       verbose=verbose,
                                       )
    return result
