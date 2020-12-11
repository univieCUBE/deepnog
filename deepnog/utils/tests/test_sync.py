import pytest

import torch

from deepnog.data.dataset import ProteinIterableDataset
from deepnog.learning.inference import predict
from deepnog.tests.utils import get_deepnog_root
from deepnog.utils import load_nn, SynchronizedCounter

TESTS = get_deepnog_root()/"tests"
WEIGHTS_PATH = TESTS/"parameters/test_deepnog.pthsmall"
DATA_SKIP_PATH = TESTS/"data/test_sync_skip_many.faa.gz"


def test_sync_counter_of_many_empty_sequences():
    """ Test if many sequences with empty ids are counted correctly. """
    # Set up device
    torch.set_num_threads(16)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # Start test
    model_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model = load_nn(['deepnog', 'DeepNOG'], model_dict, phase='infer', device=device)
    dataset = ProteinIterableDataset(DATA_SKIP_PATH, f_format='fasta')
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
