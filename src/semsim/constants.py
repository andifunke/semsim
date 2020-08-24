"""
This module defines global constants for the SemSim package.
"""

import os
from pathlib import Path
from typing import Union


# --- package paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
CORPORA_DIR = DATA_DIR / 'corpora'
CACHE_DIR = DATA_DIR / 'cache'
METRICS_DIR = DATA_DIR / 'metrics'
SEMD_DIR = METRICS_DIR / 'SemD'

# --- typing macros ---
PathLike = Union[Path, str, os.PathLike]

# --- tagsets ---
# - universal tagset -
ADJ = 'ADJ'
ADP = 'ADP'
ADV = 'ADV'
AUX = 'AUX'
CCONJ = 'CCONJ'
DET = 'DET'
INTJ = 'INTJ'
NOUN = 'NOUN'
NUM = 'NUM'
PART = 'PART'
PRON = 'PRON'
PROPN = 'PROPN'
PUNCT = 'PUNCT'
SCONJ = 'SCONJ'
SYM = 'SYM'
VERB = 'VERB'
X = 'X'
# - spacy additions -
CONJ = 'CONJ'
SPACE = 'SPACE'
# - topiclabeling additions -
NPHRASE = 'NPHRASE'

TAGSET = {
    ADJ,
    ADP,
    ADV,
    AUX,
    CCONJ,
    DET,
    INTJ,
    NOUN,
    NUM,
    PART,
    PRON,
    PROPN,
    PUNCT,
    SCONJ,
    SYM,
    VERB,
    X,
    CONJ,
    SPACE,
}

C5_TAGSET = {
    'AJ0': ADJ,
    'AJ0-AV0': ADJ,
    'AJ0-NN1': ADJ,
    'AJ0-VVD': ADJ,
    'AJ0-VVG': ADJ,
    'AJ0-VVN': ADJ,
    'AJC': ADJ,
    'AJS': ADJ,
    'AT0': DET,
    'AV0': ADV,
    'AV0-AJ0': ADV,
    'AVP-PRP': ADP,
    'AVQ-CJS': ADV,
    'CJC': CONJ,
    'CJS': SCONJ,
    'CJS-AVQ': SCONJ,
    'CJS-PRP': SCONJ,
    'CJT': SCONJ,
    'CJT-DT0': SCONJ,
    'CRD': NUM,
    'CRD-PNI': NUM,
    'DPS': DET,
    'DT0': DET,
    'DT0-CJT': DET,
    'DTQ': DET,
    'ITJ': INTJ,
    'NN0': NOUN,
    'NN1': NOUN,
    'NN1-AJ0': NOUN,
    'NN1-NP0': NOUN,
    'NN1-VVB': NOUN,
    'NN1-VVG': NOUN,
    'NN2': NOUN,
    'NN2-VVZ': NOUN,
    'NP0': PROPN,
    'NP0-NN1': PROPN,
    'ORD': NUM,
    'PNI': PRON,
    'PNI-CRD': PRON,
    'PNN': PRON,
    'PNP': PRON,
    'PNQ': PRON,
    'PRF': ADP,
    'PRP': ADP,
    'PRP-AVP': ADP,
    'PRP-CJS': ADP,
    'PUL': PUNCT,
    'PUN': PUNCT,
    'PUQ': PUNCT,
    'PUR': PUNCT,
    'UNC': X,
    'VBB': VERB,
    'VBD': VERB,
    'VBG': VERB,
    'VBI': VERB,
    'VBN': VERB,
    'VBZ': VERB,
    'VDB': VERB,
    'VDD': VERB,
    'VDG': VERB,
    'VDI': VERB,
    'VDN': VERB,
    'VDZ': VERB,
    'VHB': VERB,
    'VHD': VERB,
    'VHG': VERB,
    'VHI': VERB,
    'VHN': VERB,
    'VHZ': VERB,
    'VM0': AUX,
    'VVB': VERB,
    'VVB-NN1': VERB,
    'VVD': VERB,
    'VVD-AJ0': VERB,
    'VVD-VVN': VERB,
    'VVG': VERB,
    'VVG-AJ0': VERB,
    'VVG-NN1': VERB,
    'VVI': VERB,
    'VVN': VERB,
    'VVN-AJ0': VERB,
    'VVN-VVD': VERB,
    'VVZ': VERB,
    'VVZ-NN2': VERB,
    'XX0': PART,
    'ZZ0': X,
    # 'POS': X,
    # 'AVQ': ADV,
    # 'TO0': ADP,
    # 'EX0': PRON,
    # 'AVP': ADV,
}
BNC_TAGSET = {
    'SUBST': NOUN,
    'PUR': PUNCT,
    'UNC': X,
    'PRON': PRON,
    'ADV': ADV,
    'ADJ': ADJ,
    'PREP': ADP,
    'PUN': PUNCT,
    'VERB': VERB,
    'PUQ': PUNCT,
    'CONJ': CONJ,
    'ART': DET,
    'INTERJ': INTJ,
    'PUL': PUNCT,
}

# --- special symbols ---
LF = '</LF>'
