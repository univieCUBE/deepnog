# SPDX-License-Identifier: BSD-3-Clause
from os import environ
from pathlib import Path
from typing import Dict, Union
import yaml

from deepnog.utils.logger import get_logger

__all__ = ['get_config',
           ]

DEEPNOG_CONFIG_PATH = Path(__file__).parent.parent.absolute()/"config/"


def get_config(config_file: Union[Path, str, None] = None) -> Dict:
    """ Get a config dictionary

    If no file is provided, look in the DEEPNOG_CONFIG env variable for
    the path. If this fails, load a default config file (lacking any
    user customization).

    This contains the available models (databases, levels).
    Additional config may be added in future releases.
    """
    if config_file is None:
        config_path = environ.get('DEEPNOG_CONFIG', default=DEEPNOG_CONFIG_PATH)
        config_file = Path(config_path)/'deepnog_config.yml'
    else:
        config_file = Path(config_file)
    try:
        config = yaml.safe_load(config_file.open())
    except yaml.YAMLError as e:
        logger = get_logger(verbose=1)
        logger.warning(f'Could not read config file. Will use default config '
                       f'file (custom models will not be available).\n'
                       f'Error message:\n{e}')
        config_file = DEEPNOG_CONFIG_PATH/"deepnog_default_config.yml"
        config = yaml.safe_load(config_file.open())
    return config
