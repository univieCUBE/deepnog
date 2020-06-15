"""
Author: Roman Feldbauer
Date: 2020-02-19
Description:
    Test client (cmd line interface)
"""
import argparse
from copy import deepcopy
from pathlib import Path
import pytest
import subprocess
import tempfile
from unittest import mock

from deepnog.client import main
from deepnog.client.client import _start_prediction_or_training  # noqa
from deepnog import __version__

test_file = Path(__file__).parent.absolute() / "data/test_deepencoding.faa"


def test_entrypoint():
    process = subprocess.run(['deepnog', '--version'], capture_output=True)
    assert process.returncode == 0, (f'Could not invoke deepnog on the '
                                     f'command line: '
                                     f'Return code {process.returncode}')
    assert f'deepnog {__version__}' in process.stdout.decode('ascii'),\
        f'Incorrect version output: {process.stdout}'


@pytest.mark.parametrize('tax', [1, 2, ])
def test_cmd_line_invocation(tax):
    # Using out file
    outfile = f'out{tax}.csv'
    proc = subprocess.run(['deepnog', 'infer',
                           f'{test_file}',
                           '--tax', f'{tax}',
                           '--out', f'{outfile}',
                           '--verbose', f'{0}',
                           ],
                          capture_output=True,
                          )
    outfile = Path(outfile)
    assert outfile.is_file(), (f'Stdout of call:\n{proc.stdout}\n\n'
                               f'Stderr of call:\n{proc.stderr}')
    outfile.unlink()

    # Using output to stdout
    proc = subprocess.run(['deepnog', 'infer',
                           f'{test_file}',
                           '--tax', f'{tax}',
                           '--verbose', '3',
                           ],
                          capture_output=True,
                          )
    for log_str in [b'INFO', b'WARNING', b'DEBUG', b'ERROR', b'CRITICAL']:
        assert log_str not in proc.stdout, 'stdout polluted by logging messages,' \
                                           'when it should only contain predictions.'
    for log_str in [b'INFO', b'DEBUG']:
        assert log_str in proc.stderr, 'missing log messages in stderr'
    # Check the prediction in stdout (omitting volatile confidence values)
    correct_out = b'sequence_id,prediction,confidence\n0,COG0443'
    assert correct_out in proc.stdout, f'Incorrect prediction output in stderr: {proc.stderr}'


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(phase='infer',
                                            tax='2',
                                            out='out.mock.2',
                                            file=test_file,
                                            fformat='fasta',
                                            outformat='csv',
                                            database='eggNOG5',
                                            verbose=0,
                                            device='auto',
                                            num_workers=0,
                                            confidence_threshold=None,
                                            architecture='deepencoding',
                                            weights=None,
                                            batch_size=1,
                                            # train only
                                            training_sequences=None,
                                            validation_sequences=None,
                                            labels=None,
                                            n_epochs=None,
                                            shuffle=None,
                                            learning_rate=None,
                                            ))
def test_main_and_argparsing(mock_args):  # noqa
    main()
    Path('out.mock.2').unlink()


def test_args_sanity_check():
    _, existing_file = tempfile.mkstemp()
    args = argparse.Namespace(
        phase='infer', tax='2', out='out.mock.2', file=test_file, fformat='fasta', outformat='csv',
        database='eggNOG5', verbose=0, device='auto', num_workers=0, confidence_threshold=None,
        architecture='deepencoding', weights=None, batch_size=1,
        # train only
        training_sequences=None, validation_sequences=None, labels=None, n_epochs=None, shuffle=None,
        learning_rate=None,
    )
    args_bs = deepcopy(args)
    args_bs.batch_size = 0
    with pytest.raises(ValueError):
        _start_prediction_or_training(args_bs)
    args_out = deepcopy(args)
    args_out.out = existing_file
    with pytest.raises(FileExistsError):
        _start_prediction_or_training(args_out)
    args_device = deepcopy(args)
    args_device.device = None
    with pytest.raises(ValueError):
        _start_prediction_or_training(args_device)
    args_train = deepcopy(args)
    args_train.phase = 'train'
    args_train.n_epochs = 0
    with pytest.raises(ValueError):
        _start_prediction_or_training(args_train)
    Path(existing_file).unlink()
    args_confidence = deepcopy(args)
    args_confidence.confidence_threshold = 0
    with pytest.raises(ValueError):
        _start_prediction_or_training(args_confidence)
    args_confidence.confidence_threshold = 1.000001
    with pytest.raises(ValueError):
        _start_prediction_or_training(args_confidence)
