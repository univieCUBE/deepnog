"""
Author: Roman Feldbauer
Date: 2020-02-19
Description:
    Test client (cmd line interface)
"""
from pathlib import Path
import pytest
import subprocess

test_file = Path(__file__).parent.absolute() / "data/test_deepencoding.faa"


def test_entrypoint():
    process = subprocess.run(['deepnog', '--version'], capture_output=True)
    assert process.returncode == 0, (f'Could not invoke deepnog on the '
                                     f'command line: '
                                     f'Return code {process.returncode}')
    assert b'deepnog' in process.stdout, f'Incorrect version: {process.stdout}'


@pytest.mark.parametrize('tax', [1, 2, ])
def test_cmd_line_invocation(tax):
    outfile = f'out{tax}.csv'
    proc = subprocess.run(['deepnog',
                           f'{test_file}',
                           '--tax', f'{tax}',
                           '--out', f'{outfile}',
                           ],
                          capture_output=True,
                          )
    outfile = Path(outfile)
    assert outfile.is_file(), (f'Stdout of call:\n{proc.stdout}\n\n'
                               f'Stderr of call:\n{proc.stderr}')
    outfile.unlink()
