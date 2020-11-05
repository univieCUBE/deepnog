__all__ = ['try_import_pytorch',
           ]


def try_import_pytorch():
    """ Try to import torch, and raise helpful error.

    Notes
    -----
    This is primarily useful for the bioconda install option,
    which does not install PyTorch.
    """
    try:
        import torch
        return torch
    except ImportError as e:
        msg = ("\nIf deepnog was installed via bioconda,"
               "please install its requirement pytorch with:\n"
               "$ conda install pytorch -c pytorch")
        raise ImportError(msg) from e
