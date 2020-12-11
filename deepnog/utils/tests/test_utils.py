from itertools import product
import logging
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
import torch
import yaml

from deepnog.utils import count_parameters, load_nn, set_device
from deepnog.utils import get_logger, get_weights_path, get_config
from deepnog.utils import parse

GPU_AVAILABLE = torch.cuda.is_available()
TEST_STR = 'krawutzi'
DEEPNOG_ROOT = Path(__file__).parent.parent.parent.absolute()
TESTS = DEEPNOG_ROOT/"tests"
WEIGHTS_PATH = TESTS/"parameters/test_deepnog.pthsmall"
CONFIG = yaml.safe_load((DEEPNOG_ROOT/"config/deepnog_config.yml").open())


def _assert_exits(func, arguments):
    with pytest.raises(SystemExit) as e:
        func(arguments)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_set_device(caplog):
    device = 'tpu'
    with caplog.at_level(logging.ERROR):
        _assert_exits(set_device, device)
    # FIXME. Currently only the exit status is tested, not the message.
    # msg = f'Unknown device "{device}". Try "auto".'
    # assert msg in caplog.text


def test_auto_device():
    device = set_device('auto')
    assert isinstance(device, torch.device)
    assert str(device) in ['cpu', 'cuda'], f'Unrecognized device: {device}'


def test_cpu_device():
    device = 'cpu'
    assert isinstance(set_device(device), torch.device)


@pytest.mark.skipif(not GPU_AVAILABLE, reason='GPU is not available')
def test_gpu_device_available():
    device = 'gpu'
    assert isinstance(set_device(device), torch.device)


@pytest.mark.skipif(GPU_AVAILABLE, reason='GPU is available')
def test_gpu_device_unavailable(caplog):
    device = 'gpu'
    with caplog.at_level(logging.ERROR):
        _assert_exits(set_device, device)
    # FIXME. Currently only the exit status is tested, not the message.
    # msg = 'could not access any CUDA-enabled GPU'
    # assert msg in caplog.text


@pytest.mark.xfail(reason=("BUG: pytest logging capture does not work. "
                           "Look out for 4 logging lines manually..."))
def test_logger(caplog):
    """ Test that only the correct msg levels are logged according to verbose"""
    with caplog.at_level(logging.DEBUG, logger=__name__):
        for verbose in [True, False]:
            logger = get_logger(__name__, verbose=verbose)
            logger.info(TEST_STR)
            logger.warning(TEST_STR)
        logger = get_logger(__name__, verbose=0)
        logger.error(TEST_STR)
        logger.warning(TEST_STR)
        logger = get_logger(__name__, verbose=1)
        logger.warning(TEST_STR)
        logger.info(TEST_STR)
        logger = get_logger(__name__, verbose=2)
        logger.info(TEST_STR)
        logger.debug(TEST_STR)
        logger = get_logger(__name__, verbose=3)
        logger.debug(TEST_STR)
        lvls = (logging.INFO, logging.WARNING, logging.INFO,
                logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
        assert len(caplog.record_tuples), 'No logging output was captured'
        for i, record in enumerate(caplog.record_tuples):
            assert record == (__name__, lvls[i], TEST_STR)


@pytest.mark.parametrize(
    argnames=["database", "level"],
    argvalues=(*product(["eggNOG5", ], CONFIG["database"]["eggNOG5"]),
               *product(["cog2020", ], CONFIG["database"]["cog2020"]),
               # Add additional databases here, e.g. eggNOG6
               ),
)
def test_get_weights_all(database, level, architecture="deepnog"):
    with TemporaryDirectory(prefix=f"deepnog_test_data_dir_{database}_") as tmp:
        p = get_weights_path(
            database=database,
            level=level,
            architecture=architecture,
            data_home=tmp,
            download_if_missing=True,
            verbose=3,
        )
        p = Path(p)
        assert p.is_file(), "Could not find file. Possibly, download failed."
        assert p.suffix == ".pth", f"Wrong file format: {p.suffix}"
        assert p.stat().st_size, "File is empty"  # File size in bytes


def test_get_weights():
    with TemporaryDirectory(prefix='deepnog_test_data_dir_') as tmpdir:
        p = get_weights_path(database='testdb',
                             level='1',
                             architecture='do_not_delete',
                             data_home=tmpdir,
                             download_if_missing=True,
                             verbose=3)
        assert Path(p).is_file()


def test_get_weights_impossible(capsys):
    with TemporaryDirectory(prefix='deepnog_test_data_dir_') as tmpdir:
        with pytest.raises(IOError, match='Data not found'):
            _ = get_weights_path(database='testdb',
                                 level='1',
                                 architecture='do_not_delete',
                                 data_home=tmpdir,
                                 download_if_missing=False,
                                 verbose=3)
        _ = get_weights_path(database="unavailable_db",
                             level='0',
                             architecture="deepnog",
                             data_home=tmpdir,
                             verbose=3)
        assert "Download failed" in capsys.readouterr().err


@pytest.mark.parametrize("architecture", [('deepnog', 'DeepNOG'), ])
@pytest.mark.parametrize("weights", [WEIGHTS_PATH, ])
def test_count_params(architecture, weights):
    """ Test loading of neural network model. """
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model_dict = torch.load(weights, map_location=device)
    model = load_nn(architecture, model_dict, phase='infer', device=device)
    n_params_tuned = count_parameters(model, tunable_only=True)
    n_params_total = count_parameters(model, tunable_only=False)
    assert n_params_total == n_params_tuned


def test_load_config():
    # Check that default values are present in default config file
    default_config = DEEPNOG_ROOT/'config/deepnog_default_config.yml'
    config = yaml.safe_load(default_config.open())
    assert 'eggNOG5' in config['database'], 'eggNOG5 models missing in config'
    for i in [1, 2]:
        assert i in config['database']['eggNOG5']
    assert 'deepnog' in config['architecture'], 'Standard arch deepnog missing in config'
    for key in ['encoding_dim', 'kernel_size', 'n_filters', 'dropout', 'pooling_layer_type']:
        assert key in config['architecture']['deepnog']

    # Check that default config is loaded when faulty yaml is provided
    broken_config = [TESTS / 'data/broken_config_indent.yml', ]
    for bc in broken_config:
        loaded_config = get_config(bc)
        assert 'default' in loaded_config['config']
        assert 'broken' not in loaded_config['config']

    # Check the customizable config
    config = DEEPNOG_ROOT/"config/deepnog_config.yml"
    config = get_config(config)
    assert 'custom' in config['config']


def test_parse():
    f_plain = TESTS / 'data/GCF_000007025.1.faa'
    f_gz = TESTS / 'data/GCF_000007025.1.faa.gz'
    f_xz = TESTS / 'data/GCF_000007025.1.faa.xz'

    for rec in parse(f_plain, 'fasta'):
        assert len(rec) > 0
        assert str(rec.seq).upper().startswith('M')
    fasta_set = [set(rec.id for rec in parse(f, 'fasta')) for f in [f_plain, f_gz, f_xz]]
    n_sequences = len(fasta_set[0])
    assert n_sequences == 1_289, 'Incorrect number of sequences from plain fasta file'
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        assert len(fasta_set[a].intersection(fasta_set[b])) == n_sequences
