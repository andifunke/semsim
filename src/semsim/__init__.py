__all__ = [
    '__version__',
    'constants',
]
__version__ = '0.1.0a1'

from semsim.corpus.bnc import stream_bnc

DATASET_STREAMS = {
    'bnc': stream_bnc,
}
