import argparse
import json
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
    parser.add_argument('-m', '--min-doc-size', type=int, required=False, default=1,
                        help="Discard all documents/chunk smaller than --min-doc-size.")
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


# TODO: wrap in class and move corpus parameter to init

def stream_corpus(
        corpus: str = BNC_CORPUS_ID,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        directory: PathLike = None,
) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
    """
    Parses documents from the original BNC XML corpus and streams as list of strings or tuples.

    :param corpus: fixed value for BNC corpus
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param min_doc_size: Discard all documents/contexts smaller than min_chunk_size.
    :param tagged: if False: items of the yielded lists are string tokens.
                   if True: items of the yielded lists are tuples in the format
                   ``(token, pos_tag)``.
    :param lowercase: Convert all tokens to lowercase if True.
    :param tags_blocklist: Remove all tokens from the contexts with pos-tags in this blocklist.
    :param directory: Optional path to a BNC XML corpus. Uses the default path if None.

    :returns: Generator that yields documents/contexts as lists of tokens.
    """
    # TODO: Benchmark structured data formats for strings/documents/tokens

    directory = BNC_DIR if directory is None else directory
    print(f'Streaming BNC corpus from {directory}')
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

    if not isinstance(chunk_size, int):
        chunk_size = 0
    if not isinstance(min_doc_size, int):
        min_doc_size = 0

    fileids = bnc.fileids()
    for fileid in tqdm(fileids, total=len(fileids)):
        read_id = read_fn(fileids=fileid, strip_space=True)
        filtered = filter(filter_fn, read_id)
        mapped = map(map_fn, filtered)
        mapped_lc = map(map_lc, mapped)
        doc = list(mapped_lc)

        # --- apply chunk_size
        if chunk_size:
            idx = 0
            while idx < len(doc):
                chunk = doc[idx:idx+chunk_size]
                idx += chunk_size
                if len(chunk) >= min_doc_size:
                    yield chunk
        else:
            if len(doc) >= min_doc_size:
                yield doc


def read_corpus(*args, **kwargs):
    return list(stream_corpus(*args, **kwargs))


def infer_file_path(
        corpus: str = BNC_CORPUS_ID,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        with_suffix=True,
) -> PathLike:
    """Returns a canonical file path for the given corpus and arguments."""

    cs_suffix = f'_cs{chunk_size}' if isinstance(chunk_size, int) and chunk_size > 0 else ''
    tagged_suffix = '_tagged' if tagged else ''
    lowercase_suffix = '_lc' if lowercase else ''
    filtered_suffix = '_filtered' if tags_blocklist else ''
    min_doc_size_suffix = (
        f'_minsz{min_doc_size}' if isinstance(min_doc_size, int) and min_doc_size > 0 else ''
    )
    file_suffix = '.txt' if with_suffix else ''
    file_name = (
        f'{corpus}'
        f'{cs_suffix}'
        f'{min_doc_size_suffix}'
        f'{tagged_suffix}'
        f'{lowercase_suffix}'
        f'{filtered_suffix}'
        f'{file_suffix}'
    )
    file_path = CACHE_DIR / 'corpora' / corpus / file_name

    return file_path


def persist_transformation(
        corpus: str = BNC_CORPUS_ID,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        directory: PathLike = None,
        documents: List[List] = None,
):
    """
    Parses documents from the original BNC XML format and writes it as plain text to a file.

    The file is written to ``data/out/cache/bnc[args].txt``.

    :param corpus: fixed value for BNC
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param min_doc_size: Discard all documents/contexts smaller than min_chunk_size.
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
        corpus=corpus,
        chunk_size=chunk_size,
        min_doc_size=min_doc_size,
        tagged=tagged,
        lowercase=lowercase,
        tags_blocklist=tags_blocklist,
    )
    if documents is None:
        documents = stream_corpus(
            corpus=corpus,
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            tags_blocklist=tags_blocklist,
            directory=directory,
        )

    # - write corpus as plain text -
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as fp:
        print(f"Writing to {out_path}")
        for i, doc in enumerate(documents):
            doc = ' '.join(doc).replace('\n', '<P>')
            fp.write(doc + '\n')
        print(f'{i} lines written.')

    # - write arguments as meta data -
    with open(out_path.with_suffix('.json'), 'w') as fp:
        args = dict(
            directory=directory,
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            tags_blocklist=tags_blocklist,
        )
        json.dump(args, fp)


def load_from_cache(
        corpus: str = BNC_CORPUS_ID,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        tags_blocklist: list = None,
        make_if_not_cached: bool = True,
        persist_if_not_cached: bool = True,
        version: str = None,
        as_stream: bool = False,
) -> List[List]:
    """

    :param corpus:
    :param chunk_size:
    :param min_doc_size:
    :param tagged:
    :param lowercase:
    :param tags_blocklist:
    :param make_if_not_cached: Performs a new transformation for the given arguments,
        if loading from cache fails.
    :param persist_if_not_cached: Implies ``make_make_if_not_cached``. Writes the new
        transformation to a plain text file.
    :param version: an optional version id where the version id corresponds to a
        cached corpora file.
    :param as_stream: returns a generator instead of a list.

    :return: List of lists of tokens.
    """

    out_path = CACHE_DIR / 'corpora' / BNC_CORPUS_ID / f'{version}.txt'
    if not out_path.exists():
        out_path = infer_file_path(
            corpus=corpus,
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            tags_blocklist=tags_blocklist,
        )
    # TODO: read meta data first and verify identity of tags_blocklist

    try:
        with open(out_path, 'r') as fp:
            print(f"Loading {out_path}")
            if as_stream:
                raise NotImplementedError ## TODO
            else:
                docs = [c.strip().split() for c in fp.readlines()]
    except FileNotFoundError:
        make_if_not_cached |= persist_if_not_cached
        if make_if_not_cached:
            if as_stream:
                docs = stream_corpus(
                    corpus=corpus,
                    chunk_size=chunk_size,
                    min_doc_size=min_doc_size,
                    tagged=tagged,
                    lowercase=lowercase,
                    tags_blocklist=tags_blocklist,
                )
            else:
                docs = read_corpus(
                    corpus=corpus,
                    chunk_size=chunk_size,
                    min_doc_size=min_doc_size,
                    tagged=tagged,
                    lowercase=lowercase,
                    tags_blocklist=tags_blocklist,
                )
            if persist_if_not_cached:
                persist_transformation(
                    corpus=corpus,
                    documents=docs,
                    chunk_size=chunk_size,
                    min_doc_size=min_doc_size,
                    tagged=tagged,
                    lowercase=lowercase,
                    tags_blocklist=tags_blocklist
                )
        else:
            docs = None

    return docs


def example(args, example_='stream'):
    if example_ == 'stream':
        docs = stream_corpus(
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
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
            min_doc_size=args.min_doc_size,
            tagged=args.tagged,
            lowercase=args.lowercase,
            tags_blocklist=args.tags_blocklist,
            directory=args.directory,
        )
    elif example_ == 'load':
        docs = load_from_cache(
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
            tagged=args.tagged,
            lowercase=args.lowercase,
            tags_blocklist=args.tags_blocklist,
        )
        print(docs[:10])


if __name__ == '__main__':
    args_ = parse_args()
    print(vars(args_))
    example(args_, 'write')
