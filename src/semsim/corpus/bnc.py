import argparse
import json
import math
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


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies and
    returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--window', type=int, required=False, default=1000)
    parser.add_argument('-d', '--directory', type=str, required=False)
    parser.add_argument('--tags-blocklist', nargs='*', type=str, required=False, default=[],
                        choices=TAGS)

    parser.add_argument('--lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
    parser.set_defaults(lowercase=True)

    parser.add_argument('--tagged', action='store_true', required=False)
    parser.add_argument('--no-tagged', dest='tagged', action='store_false', required=False)
    parser.set_defaults(tagged=False)

    args = parser.parse_args()

    return args


def stream_bnc(
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        directory: PathLike = None,
) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
    """
    Parses documents from the original BNC XML corpus and streams as list of strings or tuples.

    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param tagged: if False: items of the yielded lists are string tokens.
                   if True: items of the yielded lists are tuples in the format
                   ``(token, pos_tag)``.
    :param lowercase: Convert all tokens to lowercase if True.
    :param tags_blocklist: Remove all tokens from the contexts with pos-tags in this blocklist.
    :param directory: Optional path to a BNC XML corpus. Uses the default path if None.

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
) -> PathLike:
    """Returns a canonical file path for the given corpus and arguments."""

    corpus_name = BNC_CORPUS_ID
    cs_suffix = f'_cs{chunk_size}' if isinstance(chunk_size, int) and chunk_size > 0 else ''
    tagged_suffix = '_tagged' if tagged else ''
    lowercase_suffix = '_lc' if lowercase else ''
    filtered_suffix = '_filtered' if tags_blocklist else ''
    file_name = f'{corpus_name}{cs_suffix}{tagged_suffix}{lowercase_suffix}{filtered_suffix}.txt'
    file_path = CACHE_DIR / 'corpora' / BNC_CORPUS_ID / file_name

    return file_path


def persist_transformation(
        chunk_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        directory: PathLike = None,
        documents: List[List] = None,
):
    """
    Parses documents from the original BNC XML format and writes it as plain text to a file.

    The file is written to ``data/out/cache/bnc[args].txt``.

    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param tagged:
        if False: items of the yielded lists are string tokens.
        if True: items of the yielded lists are tuples in the format ``(token, pos_tag)``.
    :param lowercase: Converts all tokens to lowercase if True.
    :param tags_blocklist: Removes all tokens from the contexts with pos-tags in this blocklist.
    :param directory: Optional: path to a BNC XML corpus. Uses the default path if None.
    :param documents: Optional: pass an already loaded and parsed corpus as list of lists.
        Omits reading the corpus again.
    """

    out_path = infer_file_path(
        chunk_size=chunk_size,
        tagged=tagged,
        lowercase=lowercase,
        tags_blocklist=tags_blocklist,
    )
    if documents is None:
        documents = stream_bnc(
            chunk_size=chunk_size,
            tagged=tagged,
            lowercase=lowercase,
            tags_blocklist=tags_blocklist,
            directory=directory,
        )

    # - write corpus as plain text -
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as fp:
        print(f"Writing to {out_path}")
        for doc in documents:
            doc = ' '.join(doc).replace('\n', '<P>')
            fp.write(doc + '\n')

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

    try:
        with open(out_path, 'r') as fp:
            print(f"Loading {out_path}")
            docs = [c.split() for c in fp.readlines()]
    except FileNotFoundError:
        make_if_not_cached |= persist_if_not_cached
        if make_if_not_cached:
            docs = read_bnc(
                chunk_size=chunk_size,
                tagged=tagged,
                lowercase=lowercase,
                tags_blocklist=tags_blocklist,
            )
            if persist_if_not_cached:
                persist_transformation(
                    documents=docs,
                    chunk_size=chunk_size,
                    tagged=tagged,
                    lowercase=lowercase,
                    tags_blocklist=tags_blocklist
                )
        else:
            docs = None

    return docs


def example(args, example_='read'):
    if example_ == 'stream':
        docs = stream_bnc(
            chunk_size=args.window,
            tagged=args.tagged,
            lowercase=args.lowercase,
            tags_blocklist=args.tags_blocklist,
            directory=args.directory,
        )
        for doc in docs:
            print(len(doc), doc[:50])
    elif example_ == 'write':
        persist_transformation(
            chunk_size=args.window,
            tagged=args.tagged,
            lowercase=args.lowercase,
            tags_blocklist=args.tags_blocklist,
            directory=args.directory,
        )
    elif example_ == 'load':
        docs = load_from_cache(
            chunk_size=args.window,
            tagged=args.tagged,
            lowercase=args.lowercase,
            tags_blocklist=args.tags_blocklist,
        )
        print(docs[:10])


if __name__ == '__main__':
    args_ = parse_args()
    print(vars(args_))
    example(args_, 'write')
