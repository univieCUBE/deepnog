"""
DeepNOG
-------

DeepNOG is a deep learning based command line tool to infer
orthologous groups of given protein sequences.
It provides a number of models for eggNOG orthologous groups,
and allows to train additional models for eggNOG or other databases.

"""
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#

__version__ = '1.2.3'

__all__ = [
    'client',
    'data',
    'learning',
    'models',
    'utils',
    '__version__',
]
