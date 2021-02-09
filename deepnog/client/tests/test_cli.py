"""
Author: Roman Feldbauer
Date: 2020-02-19
Description:
    Test client (cmd line interface)
"""
import argparse
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import pytest
import shutil
import subprocess
import tempfile
from unittest import mock
import warnings

import numpy as np
import pandas as pd
import torch

from deepnog.client import main
from deepnog.client.client import _start_prediction_or_training  # noqa
from deepnog.utils.io_utils import get_data_home
from deepnog import __version__

DEEPNOG_ROOT = Path(__file__).parent.parent.parent.absolute()
DEEPNOG_TEST = DEEPNOG_ROOT/"tests"
TEST_FILE = DEEPNOG_TEST/"data/test_deepnog.faa"
TEST_FILE_SHORT = DEEPNOG_TEST/"data/test_inference_short.faa"
TEST_LABELS_SHORT = TEST_FILE_SHORT.with_suffix('.csv')
TEST_LABELS_SHORT_COL_RENAME = DEEPNOG_TEST/"data/test_inference_short_wrong_column_names.csv"
TRAINING_FASTA = DEEPNOG_TEST/"data/test_training_dummy.faa"
TRAINING_CSV = DEEPNOG_TEST/"data/test_training_dummy.faa.csv"
Y_TRUE = np.array(
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1],
)
EGGNOG5_BACT_WEIGHTS = get_data_home()/'eggNOG5/2/deepnog.pth'


def try_to_unlink(f: Path):
    try:
        f.unlink()
    except PermissionError:
        pass


def test_entrypoint():
    process = subprocess.run(['deepnog', '--version'], capture_output=True)
    assert process.returncode == 0, (f'Could not invoke deepnog on the '
                                     f'command line: '
                                     f'Return code {process.returncode}')
    assert f'deepnog {__version__}' in process.stdout.decode('ascii'),\
        f'Incorrect version output: {process.stdout}'


@pytest.mark.parametrize('config', [{'weights': None},
                                    {'weights': str(EGGNOG5_BACT_WEIGHTS)},
                                    {'confidence_threshold': 0.99},
                                    ])
def test_run_inference(config):
    """ Also tests column renaming on the fly. """
    with tempfile.TemporaryDirectory(prefix='deepnog_test_') as outdir:
        outfile = Path(outdir)/'pred.out'
        args = argparse.Namespace(phase='infer',
                                  tax='2',
                                  out=str(outfile),
                                  file=TEST_FILE_SHORT,
                                  test_labels=f'{TEST_LABELS_SHORT_COL_RENAME}',
                                  fformat='fasta',
                                  outformat='csv',
                                  database='eggNOG5',
                                  verbose=0,
                                  device='auto',
                                  num_workers=0,
                                  confidence_threshold=config.get('confidence_threshold', None),
                                  architecture='deepnog',
                                  weights=config.get('weights', None),
                                  batch_size=1,
                                  )
        with warnings.catch_warnings():
            # ignore warning due to zero-div in MCC perf measure
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _start_prediction_or_training(args)


@pytest.mark.parametrize('tax', [1, 2, ])
def test_inference_cmd_line_invocation(tax):
    df_true = pd.DataFrame({'sequence_id': [0, 1],
                            'label': ['COG0443', 'COG0443']})
    # Using out file
    with tempfile.TemporaryDirectory(prefix='deepnog_test_') as outdir:
        outfile = Path(outdir)/'pred.out'
        proc = subprocess.run(['deepnog', 'infer',
                               f'{TEST_FILE_SHORT}',
                               '--tax', f'{tax}',
                               '--out', f'{outfile}',
                               '--verbose', f'{0}',
                               '--test_labels', f'{TEST_LABELS_SHORT}',
                               ],
                              capture_output=True,
                              )
        outfile = Path(outfile)
        assert outfile.is_file(), (f'Stdout of call:\n{proc.stdout}\n\n'
                                   f'Stderr of call:\n{proc.stderr}')
        df_pred = pd.read_csv(outfile)
        np.testing.assert_equal(df_pred.sequence_id.values, [0, 1])
        np.testing.assert_equal(df_pred.prediction.values, df_true.label)
        np.testing.assert_allclose(df_pred.confidence.values, [1., 1.], atol=0.05)

        perf_file = outfile.with_suffix('.performance.csv')
        assert perf_file.is_file(), 'No performance.csv file found'
        df_perf = pd.read_csv(perf_file, index_col=0)
        for measure in ['macro_precision', 'micro_precision', 'macro_recall',
                        'micro_recall', 'macro_f1', 'micro_f1', 'accuracy', ]:
            np.testing.assert_allclose(df_perf[measure].values, 1.)
        np.testing.assert_allclose(df_perf.mcc, 0.)  # only TP, no TN/FP/FN

    # Using output to stdout
    proc = subprocess.run(['deepnog', 'infer',
                           f'{TEST_FILE_SHORT}',
                           '--tax', f'{tax}',
                           '--verbose', '4',
                           '--test_labels', f'{TEST_LABELS_SHORT}',
                           ],
                          capture_output=True,
                          )
    for log_str in [b'INFO', b'WARNING', b'DEBUG', b'ERROR', b'CRITICAL']:
        assert log_str not in proc.stdout, 'stdout polluted by logging messages,' \
                                           'when it should only contain predictions.'
    for log_str in [b'INFO', b'DEBUG']:
        assert log_str in proc.stderr, 'missing log messages in stderr'
    # Check the prediction in stdout (omitting volatile confidence values)
    # Iterating over the lines in order to avoid issues with OS-specific linesep
    correct_out = b'sequence_id,prediction,confidence'
    i = 0
    line = "[nothing]"
    for line in BytesIO(proc.stdout):
        if i == 1:
            correct_out = b'0,COG0443'
        elif i == 2:
            correct_out = b'1,COG0443'
        assert correct_out in line, \
            f'Incorrect prediction output: expected {correct_out}, got line: {line}'
        i += 1
    assert i == 3, f'Incorrect number of output lines with i = {i}, and line = {line}.'

    # Check that the performance measures are printed to stderr
    found_header = False
    found_values = False
    header = b",macro_precision,macro_recall,macro_f1,micro_precision," \
             b"micro_recall,micro_f1,accuracy,mcc,experiment"
    values = b"0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,"
    for line in BytesIO(proc.stderr):
        if header in line:
            found_header = True
        elif values in line:
            found_values = True
    assert found_header and found_values, "Missing performance measures in stderr"


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(phase='infer',
                                            tax='2',
                                            out='out.mock.2',
                                            file=TEST_FILE_SHORT,
                                            test_labels=None,
                                            fformat='fasta',
                                            outformat='csv',
                                            database='eggNOG5',
                                            verbose=0,
                                            device='auto',
                                            num_workers=0,
                                            confidence_threshold=None,
                                            architecture='deepnog',
                                            weights=None,
                                            batch_size=1,
                                            # Not necessary to have train args here
                                            ))
def test_main_and_argparsing(mock_args):  # noqa
    main()
    try_to_unlink(Path('out.mock.2'))


def test_args_sanity_check():
    def _assert_exits(func, arguments):
        with pytest.raises(SystemExit) as e:
            func(arguments)
        assert e.type == SystemExit
        assert e.value.code == 1

    _, existing_file = tempfile.mkstemp()
    args = argparse.Namespace(
        phase='infer', tax='2', out='out.mock.2', file=TEST_FILE, fformat='fasta', outformat='csv',
        database='eggNOG5', verbose=0, device='auto', num_workers=0, confidence_threshold=0.5,
        architecture='deepnog', weights=None, batch_size=1, test_labels=None,
        # train only
        training_sequences=None, validation_sequences=None, labels=None, n_epochs=None,
        shuffle=None, learning_rate=None, gamma=None, random_seed=None, save_each_epoch=None,
    )
    args_bs = deepcopy(args)
    args_bs.batch_size = 0
    _assert_exits(_start_prediction_or_training, args_bs)

    args_out = deepcopy(args)
    args_out.out = existing_file
    _assert_exits(_start_prediction_or_training, args_out)

    args_device = deepcopy(args)
    args_device.device = None
    _assert_exits(_start_prediction_or_training, args_device)

    args_train = deepcopy(args)
    args_train.phase = 'train'
    args_train.n_epochs = 0
    _assert_exits(_start_prediction_or_training, args_train)

    try_to_unlink(Path(existing_file))
    args_confidence = deepcopy(args)
    args_confidence.confidence_threshold = 0
    _assert_exits(_start_prediction_or_training, args_confidence)

    args_confidence.confidence_threshold = 1.000001
    _assert_exits(_start_prediction_or_training, args_confidence)

    args_out_dir = deepcopy(args)
    args_out_dir.out = '/tmp/outdir/'
    _assert_exits(_start_prediction_or_training, args_out_dir)


def test_training_cmd_line_invocation():
    outdir = tempfile.mkdtemp(prefix='deepnog_test_')
    tax: int = 2
    n_epochs: int = 3
    proc = subprocess.run(['deepnog', 'train',
                           f'{TRAINING_FASTA}', f'{TRAINING_FASTA}',
                           f'{TRAINING_CSV}', f'{TRAINING_CSV}',
                           '--tax', f'{tax}', '--out', outdir, '--database', 'dummy_db',
                           '--n-epochs', f'{n_epochs}', '--verbose', '0', '--random-seed', '42',
                           ],
                          capture_output=True,
                          )
    outdir = Path(outdir)
    stdout_and_stderr = (f'Stdout of call:\n{proc.stdout}\n\n'
                         f'Stderr of call:\n{proc.stderr}')
    assert outdir.is_dir(), stdout_and_stderr
    out_files = list(outdir.iterdir())
    assert len(out_files) == 3, f'Training files missing. Dir contains: {out_files};' \
                                f'stdout/stderr:\n\n{stdout_and_stderr}'
    for f in outdir.iterdir():
        if str(f).endswith('csv'):
            df = pd.read_csv(f)
            for k in ['phase', 'epoch', 'accuracy', 'loss']:
                assert k in df.columns, f'Column {k} missing in output csv file'
            np.testing.assert_almost_equal(df.accuracy.iloc[-1], 1.0, decimal=3)
            assert df.loss.iloc[-1] < 0.1, "Surprisingly high loss"
            assert df.phase.iloc[-2] == 'train', 'Second last phase was not "train".'
            assert df.phase.iloc[-1] == 'val', 'Last phase was not "val".'
            # Repeat twice: training & validation data
            np.testing.assert_equal(df.epoch, np.repeat(np.arange(n_epochs), repeats=2)),\
                'Wrong number of epochs in csv file'
            try_to_unlink(f)
        elif str(f).endswith('npz'):
            c = np.load(str(f))
            # Here we use the same data for training and validation
            np.testing.assert_equal(c['y_train_true'], np.tile(Y_TRUE, (n_epochs, 1)))
            np.testing.assert_equal(c['y_val_true'], np.tile(Y_TRUE, (n_epochs, 1)))
            # Predictions during training epoch 0 may be anything,
            # especially with regularization.
            np.testing.assert_equal(c['y_train_pred'][-1], Y_TRUE)
            np.testing.assert_equal(c['y_val_pred'][-1], Y_TRUE)
            try_to_unlink(f)
        elif str(f).endswith('pt') or str(f).endswith('pth'):
            model = torch.load(str(f))
            for k in ['classes', 'model_state_dict', ]:
                assert k in model
            np.testing.assert_equal(model['classes'], np.array(['28H52', '99A99', 'ZYX12']))
            assert model['model_state_dict']['classification1.weight'].shape == (3, 1200)
            try_to_unlink(f)
        else:
            assert False, f'Unexpected file in output dir: {f}'

    try:
        shutil.rmtree(outdir)
    except (OSError, PermissionError):
        pass
