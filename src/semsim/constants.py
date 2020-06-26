#!/usr/bin/env python3
#
# Copyright (c) 2019-present, Ella Media GmbH
# All rights reserved.
#
"""
This module defines global constants for the storymodel package
and reads the package.cfg for user specific constants.
"""

from configparser import ConfigParser
from pathlib import Path
from typing import Union


# --- package paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
OUT_DIR = PROJECT_DIR / 'out'
DATA_DIR = PROJECT_DIR / 'data'
CORPORA_DIR = DATA_DIR / 'corpora'
BNC_DIR = CORPORA_DIR / 'BNC' / 'ota_20.500.12024_2554'
