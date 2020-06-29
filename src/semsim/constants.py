"""
This module defines global constants for the SemSim package.
"""

from pathlib import Path


# --- package paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUT_DIR = DATA_DIR / 'out'
SEMD_DIR = OUT_DIR / 'SemD'
CORPORA_DIR = DATA_DIR / 'corpora'
BNC_DIR = CORPORA_DIR / 'BNC' / 'ota_20.500.12024_2554'
