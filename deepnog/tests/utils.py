from pathlib import Path

__all__ = ['get_deepnog_root',
           ]


def get_deepnog_root() -> Path:
    return Path(__file__).parent.parent.absolute()
