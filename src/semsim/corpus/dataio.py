# --- corpus getter lookup ---
from semsim.corpus.bnc import load_from_cache


def reader(corpus):
    read_fn = dict(
        bnc=load_from_cache,
    )
    return read_fn
