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
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = DATA_DIR / "out"
NLP_DIR = OUT_DIR / "nlp"
META_DIR = OUT_DIR / "meta"
SEMD_DIR = DATA_DIR / "SemD"
CORPORA_DIR = DATA_DIR / "corpora"
METRICS_DIR = DATA_DIR / "metrics"
VECTORS_DIR = DATA_DIR / "vectors"

# --- typing macros ---
PathLike = Union[Path, str, os.PathLike]


def get_out_dir(corpus: str, make: bool = False) -> Path:
    out_dir = CORPORA_DIR / corpus / "out"
    if make:
        out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir
