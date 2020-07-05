import json
import math
from pathlib import Path
from typing import Generator, List, Union, Tuple

from nltk.corpus.reader.bnc import BNCCorpusReader
from tqdm import tqdm

from semsim.constants import CACHE_DIR, CORPORA_DIR, PathLike

BNC_CORPUS_ID = 'bnc'
BNC_DIR = CORPORA_DIR / 'BNC' / 'ota_20.500.12024_2554' / 'download' / 'Texts'

# TODO: map BNC POS-tags to universal tag set
TAGS = {
    'SUBST', 'PUR', 'UNC', 'PRON', 'ADV', 'ADJ', 'PREP', 'PUN', 'VERB', 'PUQ', 'CONJ', 'ART',
    'INTERJ', 'PUL'
}


def stream_bnc(
        directory: PathLike = None,
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
    """
    Parses documents from the original BNC XML corpus and streams as list of strings or tuples.

    :param directory: Path to a BNC XML corpus. Uses the default path if None.
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param tagged: if False: items of the yielded lists are string tokens.
                   if True: items of the yielded lists are tuples in the format
                   ``(token, pos_tag)``.
    :param lowercase: Convert all tokens to lowercase if True.
    :param tags_blocklist: Remove all tokens from the contexts with pos-tags in this blocklist.

    :returns: Generator that yields documents/contexts as lists of tokens.
    """
    # TODO: There is probably room for performance improvements.
    # TODO: A structured data format might be desirable.

    directory = BNC_DIR if directory is None else directory
    bnc = BNCCorpusReader(root=str(directory), fileids=r'[A-K]/\w*/\w*\.xml')

    read_fn = bnc.tagged_words if tagged else bnc.words
    filter_fn = lambda x: True
    map_fn = lambda x: x

    if tags_blocklist is not None:
        tags_blocklist = set(tags_blocklist)
        read_fn = bnc.tagged_words
        filter_fn = lambda x: x[1] not in tags_blocklist
        if not tagged:
            map_fn = lambda x: x[0]

    if lowercase:
        map_lc = (lambda x: (x[0].lower(), x[1])) if tagged else str.lower
    else:
        map_lc = lambda x: x

    fileids = bnc.fileids()
    for fileid in tqdm(fileids, total=len(fileids)):
        read_id = read_fn(fileids=fileid, strip_space=True)
        filtered = filter(filter_fn, read_id)
        mapped = map(map_fn, filtered)
        mapped_lc = map(map_lc, mapped)
        doc = list(mapped_lc)

        # --- apply chunk_size
        if isinstance(chunk_size, int) and len(doc) > chunk_size:
            nb_chunks = int(math.ceil(len(doc) / chunk_size))
            for i in range(nb_chunks):
                chunk = doc[i*chunk_size:(i+1)*chunk_size]
                yield chunk
        else:
            yield doc


def read_bnc(*args, **kwargs):
    return list(stream_bnc(*args, **kwargs))


def infer_file_path(
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None
) -> Path:
    """Returns a canonical file path for the given corpus and arguments."""

    corpus_name = BNC_CORPUS_ID
    cs_suffix = f'_cs{chunk_size}' if isinstance(chunk_size, int) and chunk_size > 0 else ''
    tagged_suffix = '_tagged' if tagged else ''
    lowercase_suffix = '_lc' if lowercase else ''
    filtered_suffix = '_filtered' if tags_blocklist is not None else ''
    file_name = f'{corpus_name}{cs_suffix}{tagged_suffix}{lowercase_suffix}{filtered_suffix}.txt'
    file_path = CACHE_DIR / file_name

    return file_path


def persist_transformation(
        directory: PathLike = None,
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
):
    """
    Parses documents from the original BNC XML format and writes it as plain text to a file.

    The file is written to ``data/out/cache/bnc[args].txt``.

    :param directory: Path to a BNC XML corpus. Uses the default path if None.
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param tagged:
        if False: items of the yielded lists are string tokens.
        if True: items of the yielded lists are tuples in the format ``(token, pos_tag)``.
    :param lowercase: Converts all tokens to lowercase if True.
    :param tags_blocklist: Removes all tokens from the contexts with pos-tags in this blocklist.
    """

    out_path = infer_file_path(
        chunk_size=chunk_size,
        tagged=tagged,
        lowercase=lowercase,
        tags_blocklist=tags_blocklist,
    )
    contexts = stream_bnc(
        directory=directory,
        chunk_size=chunk_size,
        tagged=tagged,
        lowercase=lowercase,
        tags_blocklist=tags_blocklist,
    )

    # - write corpus as plain text -
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as fp:
        print(f"Writing to {out_path}")
        for context in contexts:
            context = ' '.join(context).replace('\n', '<P>')
            fp.write(context + '\n')

    # - write arguments as meta data -
    with open(out_path.with_suffix('.json'), 'w') as fp:
        args = dict(
            directory=directory,
            chunk_size=chunk_size,
            tagged=tagged,
            lowercase=lowercase,
            tags_blocklist=tags_blocklist,
        )
        json.dump(args, fp)


def load_from_cache(
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        make_if_not_cached: bool = True,
        persist_if_not_cached: bool = True,
) -> List[List]:
    """

    :param chunk_size:
    :param tagged:
    :param lowercase:
    :param tags_blocklist:
    :param make_if_not_cached: Performs a new transformation for the given arguments,
        if loading from cache fails.
    :param persist_if_not_cached: Implies ``make_make_if_not_cached``. Writes the new
        transformation to a plain text file.

    :return: List of lists of tokens.
    """

    out_path = infer_file_path(
        chunk_size=chunk_size,
        tagged=tagged,
        lowercase=lowercase,
        tags_blocklist=tags_blocklist,
    )
    # TODO: read meta data first and verify identity of tags_blocklist

    with open(out_path, 'r') as fp:
        print(f"Loading {out_path}")
        contexts = [c.split() for c in fp.readlines()]

    # TODO: implement fallback modes.

    return contexts


def example():
    docs = stream_bnc(
        chunk_size=1000,
        tagged=True,
        lowercase=True,
        tags_blocklist=['PUL', 'PUN', 'PUQ']
    )
    for doc in docs:
        print(len(doc), doc[:50])


if __name__ == '__main__':
    # example()
    # TODO: refactor as command line arguments
    # persist_transformation(
    #     chunk_size=1000,
    #     tagged=False,
    #     lowercase=True,
    #     tags_blocklist=['PUL', 'PUN', 'PUQ']
    # )
    docs = load_from_cache(
        chunk_size=1000,
        tagged=False,
        lowercase=True,
        tags_blocklist=['PUL', 'PUN', 'PUQ']
    )
    print(docs[:10])
