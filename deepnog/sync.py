"""
Author: Roman Feldbauer
Date: 2020-02-19
"""
# SPDX-License-Identifier: BSD-3-Clause


__all__ = ['rpipe_l',
           'wpipe_l',
           'n_skipped',
           ]


def init():
    """ Communicating number of sequences with empty ids in the dataset. """
    global rpipe_l
    global wpipe_l
    global n_skipped
    rpipe_l = []
    wpipe_l = []
    n_skipped = 0
