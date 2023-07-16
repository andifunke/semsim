__all__ = [
    "__version__",
    "constants",
]
__version__ = "0.2.0a0"

from semsim.corpus.bnc import stream_corpus as stream_bnc
from semsim.corpus.topiclabeling import stream_corpus as stream_tl

DATASET_STREAMS = {
    "bnc": stream_bnc,
    "topiclabeling": stream_tl,
}
