"""
Author: Roman Feldbauer

Date: 2020-06-03

Description:

    Traing deep networks for protein orthologous group prediction.
"""
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple, Counter
import copy
from datetime import datetime
from functools import partial
import logging
import random
import string
import time
from typing import List, Union

import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import default_collate
from torch.nn.modules.loss import MSELoss, L1Loss
from torch.nn.modules import PairwiseDistance
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .dataset import ProteinDataset
from .utils import EXTENDED_IUPAC_PROTEIN_ALPHABET, set_device, load_nn
from .utils import count_parameters
from frenetiq.data_loader import SimapDatasetOnline
from frenetiq.network.embedding import AminoAcidWordEmbedding


# Note: early_stopping is for debugging not regularization technique (the regularization
# technique is implemented so or so
def train_model(model, criterion, optimizer, scheduler, data_loader, num_epochs=2,
                tensorboard_exp=None, batch_size: int = None, early_stopping: int = None,
                l2_coeff=None, log_interval: int = 100, device='cpu', validation_only: bool = False):
    # Try to set up tensorboard
    if tensorboard_exp is not None:
        exp = tensorboard_exp + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        param_str = exp
        tb_dir = f'/proj/cube/deepfam/dev/tensorboard/{exp}'
        tensorboard_writer = SummaryWriter(tb_dir)
        print(f'Tensorboard directory:', tensorboard_writer.log_dir)
    else:
        tensorboard_writer = None

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Don't forget to set SIMAP dataset phase
            data_loader.dataset.phase = phase
            if phase == 'train':
                if validation_only:
                    continue
                else:
                    model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0.

            log_loss = 0.
            log_corrects = 0
            log_n_objects = 0

            # Iterate over data.
            n_processed_sequences = 0
            numerator = early_stopping if early_stopping else len(data_loader.dataset)
            denominator = batch_size if batch_size else None
            tqdm_total = numerator // denominator + 1 if denominator else None
            for batch_nr, batch in enumerate(tqdm(data_loader, total=tqdm_total)):
                # About minibatch tuple: ['query', 'hits', 'similarity', 'query_labels', 'hits_labels']
                sequence = batch.query
                labels = batch.query_labels[:, 0]

                inputs = sequence.to(device)
                labels = labels.to(device)

                # Update progress for TensorBoard
                current_batch_size = len(inputs)
                n_processed_sequences += current_batch_size

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
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

                # statistics
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels)
                batch_acc = batch_corrects.double() / len(labels)

                log_loss += float(batch_loss)
                log_corrects += float(batch_corrects)
                log_n_objects += len(labels)

                if tensorboard_writer is not None and batch_nr % log_interval == 0:
                    tensorboard_writer.add_scalar('data/eggnog_crossentropy_loss',
                                                  log_loss,
                                                  n_processed_sequences)
                    tensorboard_writer.add_scalar('data/eggnog_accuracy',
                                                  log_corrects / log_n_objects,
                                                  n_processed_sequences)

                    # Reset the log loss/acc variables
                    log_loss = 0.
                    log_corrects = 0
                    log_n_objects = 0

                running_loss += batch_loss * inputs.size(0)
                running_corrects += batch_corrects

                if early_stopping and n_processed_sequences > early_stopping:
                    break

            # empty cache if possible
            torch.cuda.empty_cache()

            epoch_loss = running_loss / n_processed_sequences
            epoch_acc = running_corrects.double() / n_processed_sequences

            print(f'{phase} eggnog loss: {epoch_loss:.4f} eggnog acc: {epoch_acc:.4f}\n')

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # temporaly save network
        save_file = tensorboard_exp + f'_epoch{epoch}_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model.save(model, data_loader.dataset, save_file, optimizer, scheduler, l2_coeff)

        # early stopping (regularization)
        if epoch - 2 >= best_epoch:
            print(f'Early stopping due to decreasing validation scores disabled.')
            # break

    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print(f'Training complete in {minutes:.0f}m {seconds:.0f}s')
    print(f'Best val eggnog acc: {best_acc:4f} in epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


collated_batch = namedtuple('collated_batch', ['query', 'hits', 'similarity', 'query_labels', 'hits_labels'])


# Gosch: Note batch is actually a list of namedtuples not a namedtuple
def collate_sequences_with_labels(batch: namedtuple, zero_padding: bool = False, random_padding: bool = False,
                                  skip_identical: bool = True) -> namedtuple:
    """ Collate query and (optionally) labels. """

    # Find the longest sequence, in order to zero pad the others; and optionally skip self hits
    max_len, n_features = 0, 1  # batch.query_encoded.shape
    n_data = 0
    for b in batch:
        query = b.query_encoded
        n_data += 1
        sequence_len = len(query)
        if sequence_len > max_len:
            max_len = sequence_len

    # Collate the sequences
    if zero_padding:
        sequences = np.zeros((n_data, max_len,), dtype=np.int)
        for i, b in enumerate(batch):
            query = np.array(b.query_encoded)
            # If selected, choose randomly, where to insert zeros
            if random_padding and len(query) < max_len:
                n_zeros_1 = max_len - len(query)
                start1 = np.random.choice(n_zeros_1 + 1)
                end1 = start1 + len(query)
            else:
                start1 = 0
                end1 = len(query)

            # Zero pad and / or slice
            sequences[i, start1:end1] = query[:].T
        sequences = default_collate(sequences)
    else:  # no zero-padding, must use minibatches of size 1 downstream!
        raise NotImplementedError
        sequences = [torch.from_numpy(x) for x in batch.hits_encoded]

    # Collate the labels
    labels = np.array([b.query_labels for b in batch], dtype=np.int)
    labels = default_collate(labels)

    return collated_batch(query=sequences,
                          hits=None,
                          similarity=None,
                          query_labels=labels,
                          hits_labels=None)


def create_deepfam_alphabet():
    # Encode 26 letters but use first 21 letters normally
    # treat B,Z,J specially and ignore U, O
    alphabet = ProteinAlphabet()
    alphabet.letters = 'ACDEFGHIKLMNPQRSTVWYXBZJUO'
    alphabet.size = 21
    return alphabet


def validate_model(model, criterion, data_loader, batch_size: int = None,
                   l2_coeff=None, device='cpu'):
    """ Mean accuracy prediction assumes every class is represented
        in validation set at least once! """
    since = time.time()

    print(f'Validate Model')
    print('-' * 10)

    # dataset phase
    data_loader.dataset.phase = 'val'
    model.eval()  # Set model to evaluate mode

    counter_tot = Counter(dict([(key, 0) for key in range(model.classification1.out_features)]))
    counter_corr = Counter(dict([(key, 0) for key in range(model.classification1.out_features)]))
    running_loss = 0.
    running_corrects = 0.

    # Iterate over data.
    n_processed_sequences = 0
    numerator = len(data_loader.dataset)
    denominator = batch_size if batch_size else None
    tqdm_total = numerator // denominator + 1 if denominator else numerator
    for batch_nr, batch in enumerate(tqdm(data_loader, total=tqdm_total)):
        # Get sequences and labels
        sequence = batch.query
        labels = batch.query_labels[:, 0]
        inputs = sequence.to(device)
        labels = labels.to(device)

        # Update counter
        counter_tot.update(labels.tolist())
        current_batch_size = len(inputs)
        n_processed_sequences += current_batch_size

        # forward with disabled gradient
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if l2_coeff is not None:
                loss_ce = criterion(outputs, labels)
                loss_reg = model.classification1.weight.pow(2).sum()
                loss = loss_ce + l2_coeff * loss_reg
            else:
                loss = criterion(outputs, labels)

        corr_mask = preds == labels
        # statistics
        counter_corr.update(labels[corr_mask].tolist())
        batch_loss = loss.item()
        batch_corrects = torch.sum(corr_mask)
        batch_acc = batch_corrects.double() / len(labels)

        running_loss += batch_loss * inputs.size(0)
        running_corrects += batch_corrects

    # empty cache if possible
    torch.cuda.empty_cache()

    epoch_loss = running_loss / n_processed_sequences
    epoch_acc = running_corrects.double() / n_processed_sequences
    mean_acc = 0.
    for key in counter_tot.keys():
        mean_acc += counter_corr[key] * 1.0 / counter_tot[key]
    mean_acc /= model.classification1.out_features

    time_elapsed = time.time() - since
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print(f'Validation complete in {minutes:.0f}m {seconds:.0f}s, loss: {epoch_loss:4f}')
    print(f'Validation total acc: {epoch_acc:4f}, mean acc: {mean_acc:4f}')

    return


def fit(model, sequences, labels, *,
        member_threshold: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        n_epochs: int = 15,
        learning_rate: float = 1e-2,
        learning_rate_params: dict = None,
        optimizer_cls=optim.Adam,
        network_params: dict = None,
        model_dict=None,
        device: Union[str, torch.device] = 'auto',
        tensorboard: Union[None, str] = 'auto',
        log_interval: int = 100,
        random_state_numpy: int = 0,
        random_state_torch: int = 1):
    device = set_device(device)
    logging.info(f'Training device: {device}')
    rnd_state = np.random.RandomState(random_state_numpy)
    torch.manual_seed(random_state_torch)
    torch.cuda.manual_seed(random_state_torch)

    # Set up training data set with sequences and labels
    dataset = ProteinDataset(file=sequences,
                             labels_file=labels)
    # Deep network hyperparameter default values
    if network_params is None:
        network_params = {'encoding_dim': 10,
                          'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                          'n_filters': 150,
                          'dropout': 0.3,
                          'pooling_layer_type': 'max',
                          }
    if learning_rate_params is None:
        learning_rate_params = {'step_size': 1,
                                'gamma': 0.75,
                                'last_epoch': -1,
                                }
    # Load model, and send to selected device. Set up training.
    model = load_nn(architecture=model, model_dict=model_dict, phase='train',
                    device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(),
                              lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, **learning_rate_params)

    # Trainable parameters
    logging.info(f'Tunable parameters: {count_parameters(model)}')

    # Tensorboard name
    if tensorboard is None:
        pass  # do not use tensorboard
    elif tensorboard == 'auto':
        now = datetime.now().strftime("%Y-%m-%d_%H-%m-%S_%f")
        random_letters = ''.join(random.sample(string.ascii_letters, 4))
        tensorboard = f'deepnog_{now}_{random_letters}'
