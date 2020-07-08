"""
This module defines global constants for the SemSim package.
"""

import os
from pathlib import Path


# --- package paths ---
from typing import Union

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUT_DIR = DATA_DIR / 'out'
CACHE_DIR = OUT_DIR / 'cache'
SEMD_DIR = OUT_DIR / 'SemD'
CORPORA_DIR = DATA_DIR / 'corpora'

# --- typing macros ---
PathLike = Union[Path, str, os.PathLike]
