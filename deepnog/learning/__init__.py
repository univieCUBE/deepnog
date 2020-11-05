from .inference import predict
from .training import fit
from ..utils.imports import try_import_pytorch

__all__ = ['fit',
           'predict',
           ]

try_import_pytorch()
