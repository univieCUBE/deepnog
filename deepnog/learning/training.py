"""
Author: Roman Feldbauer

Date: 2020-06-03

Description:

    Traing deep networks for protein orthologous group prediction.
"""
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter, namedtuple
import copy
from datetime import datetime
from functools import partial
from pathlib import Path
import random
import string
import tempfile
import time
from typing import List, Union, NamedTuple
import warnings

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import ProteinDataset, collate_sequences
from ..utils.io_utils import logging
from ..utils.utils import set_device, load_nn, count_parameters

__all__ = ['train_and_validate_model',
           'fit',
           ]

collated_batch = NamedTuple('collated_batch',
                            [('sequence', torch.Tensor),
                             ('label', torch.Tensor)])
train_val_result = NamedTuple('train_val_result',
                              [('model', nn.Module),
                               ('dataset', torch.utils.data.Dataset),
                               ('evaluation', List[dict]),
                               ('y_true', np.ndarray),
                               ('y_pred', np.ndarray)])


def train_and_validate_model(model: nn.Module, criterion, optimizer,
                             scheduler, data_loader, *,
                             num_epochs=2,
                             tensorboard_exp=None,
                             stop_after: int = None,
                             l2_coeff=None,
                             log_interval: int = 100,
                             device: torch.device = 'cuda',
                             validation_only: bool = False,
                             early_stopping: int = 0,
                             save_each_epoch: bool = True,
                             verbose: int = 2) -> train_val_result:
    """ Perform training and validation of a given model, data, and hyperparameters.

    Parameters
    ----------
    model : nn.Module
        Deep network PyTorch module
    criterion
        PyTorch loss function, e.g., CrossEntropyLoss
    optimizer
        PyTorch optimizer instance, e.g., Adam
    scheduler
        PyTorch learning rate scheduler
    data_loader : torch.utils.DataLoader
        PyTorch DataLoader (Dataset plus some parameters)
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
    # Set up tensorboard
    if tensorboard_exp is not None:
        tensorboard_exp = Path(tensorboard_exp)
        tensorboard_writer = SummaryWriter(str(tensorboard_exp))
        logging.info(f'Tensorboard directory: {tensorboard_writer.log_dir}.')
    else:
        tensorboard_writer = None
        logging.info(f'Tensorboard disabled.')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_acc: float = 0.0
    best_acc: float = 0.0
    best_epoch: int = 0
    evaluation: list = []
    batch_size: int = data_loader.batch_size
    y_true: np.ndarray = -np.ones((num_epochs, len(data_loader.dataset)), dtype=np.int32)
    y_pred: np.ndarray = -np.ones_like(y_true)
    tqdm_disable: bool = True if verbose < 2 else False

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                if validation_only:
                    continue  # skip training
                else:
                    model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss: torch.float32 = 0.
            running_corrects: torch.int = 0

            log_loss: float = 0.
            log_corrects: int = 0
            log_n_objects: int = 0

            # Iterate over data.
            n_processed_sequences = 0
            numerator = stop_after if stop_after else len(data_loader.dataset)
            denominator = batch_size if batch_size else None
            tqdm_total = numerator // denominator + 1 if denominator else None
            for batch_nr, batch in enumerate(tqdm(data_loader,
                                                  total=tqdm_total,
                                                  disable=tqdm_disable,
                                                  desc='deepnog training',
                                                  unit=f' minibatches'
                                                  )):
                # About minibatch tuple: ['query', 'hits', 'similarity', 'query_labels', 'hits_labels']
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
                        loss_ce = criterion(outputs, labels)
                        loss_reg = model.classification1.weight.pow(2).sum()
                        loss = loss_ce + l2_coeff * loss_reg
                    else:
                        loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # save to global ground truth and prediction arrays
                    start = n_processed_sequences - current_batch_size
                    end = n_processed_sequences
                    y_true[epoch, start:end] = labels.detach().cpu().numpy()
                    y_pred[epoch, start:end] = preds.detach().cpu().numpy()

                # statistics
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels)

                log_loss += float(batch_loss)
                log_corrects += float(batch_corrects)
                log_n_objects += len(labels)

                if tensorboard_writer is not None and batch_nr % log_interval == 0:
                    tensorboard_writer.add_scalar(f'{phase}/loss',
                                                  log_loss / log_n_objects,
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
                    logging.info(f'Stopping after {n_processed_sequences} as '
                                 f'requested (stop_after = {stop_after}).')
                    break

            # Finishing train or val phase
            epoch_loss = running_loss / n_processed_sequences
            epoch_acc = running_corrects.double() / n_processed_sequences
            evaluation.append({'phase': phase,
                               'epoch': epoch,
                               'accuracy': float(epoch_acc),
                               'loss': epoch_loss})
            if n_processed_sequences < len(data_loader):
                logging.warning(f'Not all sequences were processed in epoch {epoch}: '
                                f'n_processed = {n_processed_sequences} < '
                                f'n_total = {len(data_loader)}.')

            logging.info(f'{phase} --- loss: {epoch_loss:.4f}  --- acc: {epoch_acc:.4f}\n')

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # empty cache if possible
            torch.cuda.empty_cache()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # temporarily save network
        if save_each_epoch:
            save_file = tensorboard_exp/f'_epoch{epoch:02d}.pt'
            torch.save({'classes': data_loader.dataset.label_encoder.classes_,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'l2_coeff': l2_coeff, },
                       save_file)

        # early stopping (regularization)
        if early_stopping and epoch - early_stopping >= best_epoch:
            if epoch_acc < best_acc:
                logging.info(f'Early stopping due to decreasing scores for '
                             f'{early_stopping} epochs after best epoch.')
                break

    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    logging.info(f'Training complete in {minutes:.0f}m {seconds:.0f}s')
    logging.info(f'Best val acc: {best_acc:4f} in epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return train_val_result(model=model,
                            dataset=data_loader.dataset,
                            evaluation=evaluation,
                            y_true=y_true,
                            y_pred=y_pred)


def collate_sequences_with_labels(batch: List[namedtuple],
                                  zero_padding: bool = True,
                                  random_padding: bool = False) -> NamedTuple:
    """ Collate query and (optionally) labels. """

    # Find the longest sequence, in order to zero pad the others; and optionally skip self hits
    max_len, n_features = 0, 1  # batch.query_encoded.shape
    n_data = 0
    for b in batch:
        sequence = b.query_encoded
        n_data += 1
        sequence_len = len(sequence)
        if sequence_len > max_len:
            max_len = sequence_len

    # Collate the sequences
    if zero_padding:
        sequences = np.zeros((n_data, max_len,), dtype=np.int)
        for i, b in enumerate(batch):
            sequence = np.array(b.query_encoded)
            # If selected, choose randomly, where to insert zeros
            if random_padding and len(sequence) < max_len:
                n_zeros = max_len - len(sequence)
                start = np.random.choice(n_zeros + 1)
                end = start + len(sequence)
            else:
                start = 0
                end = len(sequence)

            # Zero pad and / or slice
            sequences[i, start:end] = sequence[:].T
        sequences = default_collate(sequences)
    else:  # no zero-padding, must use minibatches of size 1 downstream!
        # sequences = [torch.from_numpy(x) for x in batch.hits_encoded]
        raise NotImplementedError

    # Collate the labels
    labels = np.array([b.query_labels for b in batch], dtype=np.int)
    labels = default_collate(labels)

    return collated_batch(sequence=sequences, label=labels)


def fit(architecture, sequences, labels, *,
        data_loader_params: dict = None,
        n_epochs: int = 15,
        shuffle: bool = False,
        learning_rate: float = 1e-2,
        learning_rate_params: dict = None,
        optimizer_cls=optim.Adam,
        device: Union[str, torch.device] = 'auto',
        tensorboard_dir: Union[None, str] = 'auto',
        log_interval: int = 100,
        random_state_numpy: int = 0,
        random_state_torch: int = 1,
        save_each_epoch: bool = True,
        verbose: int = 2,
        ):
    device = set_device(device)
    logging.info(f'Training device: {device}')
    rnd_state = np.random.RandomState(random_state_numpy)
    torch.manual_seed(random_state_torch)
    torch.cuda.manual_seed(random_state_torch)

    # PyTorch DataLoader default arguments
    if data_loader_params is None:
        data_loader_params = {'batch_size': 32,
                              # TODO enable shuffling, e.g. via this approach:
                              # https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
                              'shuffle': shuffle,
                              'num_workers': 4,
                              'collate_fn': partial(
                                  collate_sequences,
                                  zero_padding=True),
                              'pin_memory': True,
                              }
    if learning_rate_params is None:
        learning_rate_params = {'step_size': 1,
                                'gamma': 0.75,
                                'last_epoch': -1,
                                }

    # Set up training data set with sequences and labels
    if isinstance(sequences, str):
        dataset = ProteinDataset(file=sequences,
                                 labels_file=labels)
    else:
        dataset = sequences
    data_loader = DataLoader(dataset, **data_loader_params)

    # Deep network hyperparameter default values
    # TODO allow user changes in CLI
    model_dict = {'n_classes': [len(dataset.label_encoder.classes_)],
                  'encoding_dim': 10,
                  'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                  'n_filters': 150,
                  'dropout': 0.3,
                  'pooling_layer_type': 'max',
                  }

    # Tensorboard experiment name (filename)
    if tensorboard_dir is None:
        experiment = None  # do not use tensorboard
    elif tensorboard_dir == 'auto':
        now = datetime.now().strftime("%Y-%m-%d_%H-%m-%S_%f")
        random_letters = ''.join(random.sample(string.ascii_letters, 4))
        tmp_dir = tempfile.mkdtemp(prefix='tensorboard_')
        experiment = Path(tmp_dir)/f'deepnog_{now}_{random_letters}'
    else:
        try:
            experiment = Path(tensorboard_dir)
        except TypeError:
            warnings.warn(f'Invalid value for "tensorboard" argument. '
                          f'Must be one of: None, "auto", or a valid path. '
                          f'Continuing without tensorboard report.')
            experiment = None

    # Load model, and send to selected device. Set up training.
    model = load_nn(architecture=architecture, model_dict=model_dict, phase='train',
                    device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(),
                              lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, **learning_rate_params)

    # Trainable parameters
    logging.info(f'Network architecture: {architecture}')
    logging.info(f'Learning criterion: {criterion}')
    logging.info(f'Optimizer: {optimizer}')
    logging.info(f'Learning rate scheduler: {scheduler}')
    logging.info(f'Number of classes: {model.n_classes}')
    logging.info(f'Tunable parameters: {count_parameters(model)}')

    result = train_and_validate_model(model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      data_loader=data_loader,
                                      num_epochs=n_epochs,
                                      tensorboard_exp=experiment,
                                      log_interval=log_interval,
                                      device=device,
                                      save_each_epoch=save_each_epoch,
                                      verbose=verbose,
                                      )
    return result
