"""
Author: Roman Feldbauer

Date: 2020-02-19

Description:

    Parallel processing helpers
"""
# SPDX-License-Identifier: BSD-3-Clause

from multiprocessing import Value

__all__ = ['SynchronizedCounter',
           ]


class SynchronizedCounter:
    """ A multiprocessing-safe counter.

    Parameters
    ----------
    init : int, optional
        Counter starts at init (default: 0)
    """
    def __init__(self, init: int = 0):
        self.val = Value('i', init)

    def __add__(self, other):
        return self.value + other

    def __iadd__(self, other):
        self.increment(n=other)
        return self

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return self.value

    def __gt__(self, other):
        return int(self) > other

    def __ge__(self, other):
        return int(self) >= other

    def __lt__(self, other):
        return int(self) < other

    def __le__(self, other):
        return int(self) <= other

    def __eq__(self, other):
        return int(self) == other

    def increment(self, n=1):
        """ Obtain a lock before incrementing, since += isn't atomic.

        Parameters
        ----------
        n : int, optional
            Increment counter by n (default: 1)
        """
        with self.val.get_lock():
            self.val.value += n
        return

    def increment_and_get_value(self, n=1) -> int:
        """ Obtain a lock before incrementing, since += isn't atomic.

        Parameters
        ----------
        n : int, optional
            Increment counter by n (default: 1)
        """
        with self.val.get_lock():
            self.val.value += n
            return self.val.value

    @property
    def value(self) -> int:
        return self.val.value
