# --- corpus getter lookup ---

from semsim.corpus.bnc import load_from_cache as load_bnc
from semsim.corpus.topiclabeling import load_from_cache as load_tl


def reader(corpus):
    read_fn = dict(
        bnc=load_bnc,
        onlineparticipation=load_tl,
        dewac=load_tl,
    ).get(corpus.lower(), None)

    return read_fn
