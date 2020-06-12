"""
Author: Roman Feldbauer
Date: 2020-02-19
Description:
    Test client (cmd line interface)
"""
import argparse
from pathlib import Path
import pytest
from unittest import mock
import subprocess

from deepnog.client import main
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
def test_main_and_argparsing(mock_args):
    main()
    Path('out.mock.2').unlink()
