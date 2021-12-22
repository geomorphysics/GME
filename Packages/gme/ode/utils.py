"""
---------------------------------------------------------------------

Utility functions

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`

---------------------------------------------------------------------

"""
# Library
import warnings

# NumPy
import numpy as np

warnings.filterwarnings("ignore")

__all__ = ['report_progress']


def report_progress(
    i: int,
    n: int,
    progress_was: float = 0.0,
    pc_step: float = 1,
    is_initial_step: bool = False
) -> float:
    """
    Print percentage estimated progress of some ongoing job
    """
    progress_now: float \
        = 100*np.round((100/pc_step)*i/(n-1 if n > 1 else 1)) \
        / np.round(100/pc_step)
    if progress_now > progress_was or is_initial_step:
        print(f'{progress_now:0.0f}% ', end='' if progress_now
              < 100 else '\n', flush=True)
    return progress_now
