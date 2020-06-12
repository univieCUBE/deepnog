import pytest

import torch

from deepnog.utils import set_device

GPU_AVAILABLE = torch.cuda.is_available()


def test_set_device():
    device = 'tpu'
    msg = f'Unknown device "{device}". Try "auto".'
    with pytest.raises(ValueError, match=msg):
        set_device(device)


def test_auto_device():
    device = set_device('auto')
    print(f'Auto device: {device}')


def test_cpu_device():
    device = 'cpu'
    assert isinstance(set_device(device), torch.device)


@pytest.mark.skipif(not GPU_AVAILABLE, reason='GPU is not available')
def test_gpu_device_available():
    device = 'gpu'
    assert isinstance(set_device(device), torch.device)


@pytest.mark.skipif(GPU_AVAILABLE, reason='GPU is available')
def test_gpu_device_unavailable():
    device = 'gpu'
    msg = 'could not access any CUDA-enabled GPU'
    with pytest.raises(RuntimeError, match=msg):
        set_device(device)
