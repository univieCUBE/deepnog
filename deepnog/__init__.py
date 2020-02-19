"""
DeepNOG
=======

DeepNOG is a deep learning based command line tool which predicts the
protein families of given protein sequences based on pretrained neural
networks.

The main module of this tool is defined in deepnog.py. For details about the
usage of the tool, the reader is referred to the documentation as well as 
deepnog.py.

Available subpackages
---------------------
models
    Supported neural network architectures.
tests
    Software tests written for pytest.

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
__version__ = '1.0.2'
