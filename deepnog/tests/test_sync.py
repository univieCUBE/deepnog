from pathlib import Path
import pytest

import torch

from deepnog.data.dataset import ProteinDataset
from deepnog.learning.inference import predict
from deepnog.utils import load_nn, SynchronizedCounter

current_path = Path(__file__).parent.absolute()
weights_path = current_path/'parameters/test_deepencoding.pthsmall'
data_skip_path = current_path/'data/test_sync_skip_many.faa.gz'


def test_sync_counter_of_many_empty_sequences():
    """ Test if many sequences with empty ids are counted correctly. """
    # Set up device
    torch.set_num_threads(16)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # Start test
    model_dict = torch.load(weights_path, map_location=device)
    model = load_nn('deepencoding', model_dict, phase='infer', device=device)
    dataset = ProteinDataset(data_skip_path, f_format='fasta')
    with pytest.warns(UserWarning, match='no sequence id could be detected'):
        _ = predict(model, dataset, device)

    # Test correct counted skipped sequences
    assert(int(dataset.n_skipped) == 2**16)


def test_sync_counter():
    cnt = SynchronizedCounter(init=0)
    cnt.increment(1)
    other = cnt + 9
    assert other == 10
    cnt += 1
    val = cnt.increment_and_get_value(3)
    assert val == 5

    assert cnt < 6
    assert cnt <= 5
    assert cnt > 4
    assert cnt >= 5
    assert cnt == 5
