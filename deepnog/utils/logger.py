# SPDX-License-Identifier: BSD-3-Clause
import logging
import sys

__all__ = ['get_logger',
           ]

logging.addLevelName(logging.DEBUG,
                     "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
logging.addLevelName(logging.INFO,
                     "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
logging.addLevelName(logging.WARNING,
                     "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR,
                     "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))


def get_logger(initname: str = 'deepnog', verbose: int = 0) -> logging.Logger:
    """
    This function provides a nicely formatted logger.

    Parameters
    ----------
    initname : str
        The name of the logger to show up in log.
    verbose : int
        Increasing levels of verbosity

    References
    ----------
    Shamelessly stolen from phenotrex
    """
    logger = logging.getLogger(initname)
    if type(verbose) is bool:
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
    else:
        logger.setLevel(verbose)
    ch = logging.StreamHandler(stream=sys.stderr)
    if verbose <= 0:
        ch.setLevel(logging.ERROR)
    elif verbose == 1:
        ch.setLevel(logging.WARNING)
    elif verbose <= 3:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logstring = ('\033[1;32m[%(asctime)s]\033[1;0m \033[1m%(name)s'
                 '\033[1;0m - %(levelname)s - %(message)s')
    formatter = logging.Formatter(logstring, '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger
