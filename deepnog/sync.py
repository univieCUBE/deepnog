"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Parallel processing helpers
"""
# SPDX-License-Identifier: BSD-3-Clause


__all__ = ['rpipe_l',
           'wpipe_l',
           'n_skipped',
           ]

#: List of read pipes for each worker
rpipe_l: list = []

#: List of write pipes for each worker
wpipe_l: list = []

#: Number of skipped sequences (due to empty ID)
n_skipped: int = 0


def init():
    """ Communicating number of sequences with empty ids in the dataset. """
    global rpipe_l
    global wpipe_l
    global n_skipped
    #rpipe_l = []
    #wpipe_l = []
    #n_skipped = 0
