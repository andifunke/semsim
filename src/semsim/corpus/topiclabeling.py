# coding: utf-8

"""Converts corpora from the topic-labeling package format to the simple semsim format."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Generator, List, Union, Tuple

import pandas as pd
from tqdm import tqdm

from semsim.constants import PathLike, NLP_DIR, CACHE_DIR, META_DIR

tqdm.pandas()


GOOD_IDS_DEWAC = None
GOOD_IDS_DEWIK = None
TAGS = {
    'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ', 'CONJ', 'DET', 'NUM',
    'PART', 'PRON', 'SCONJ',  'PUNCT', 'SYM', 'X', 'SPACE'
}


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies and
    returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--corpus', type=str, required=True,
                        help="ID or unique prefix of a topic-labeling corpus.")
    parser.add_argument('-w', '--window', type=int, required=False, default=1000)
    parser.add_argument('-m', '--min-doc-size', type=int, required=False, default=1,
                        help="Discard all documents/chunk smaller than --min-doc-size.")
    parser.add_argument('-d', '--directory', type=str, required=False, default=NLP_DIR)
    parser.add_argument('--tags-blocklist', nargs='*', type=str, required=False, default=[],
                        choices=TAGS)

    parser.add_argument('--lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
    parser.set_defaults(lowercase=False)

    parser.add_argument('--tagged', action='store_true', required=False)
    parser.add_argument('--no-tagged', dest='tagged', action='store_false', required=False)
    parser.set_defaults(tagged=False)

    parser.add_argument('--lemmatized', action='store_true', required=False)
    parser.add_argument('--no-lemmatized', dest='lemmatized', action='store_false', required=False)
    parser.set_defaults(lemmatized=False)

    args = parser.parse_args()

    return args


def preprocess_dewac(df):
    global GOOD_IDS_DEWAC
    if GOOD_IDS_DEWAC is None:
        GOOD_IDS_DEWAC = pd.read_pickle(META_DIR / 'dewac_good_ids.pickle')
    df = df[df.hash.isin(GOOD_IDS_DEWAC.index)]

    def remove_title(x):
        """Removes the rows up to the first line feed (inclusive), i.e. the document title."""
        ln_argmax = (x.text.to_numpy() == '\n').argmax()
        return x[ln_argmax + 1:]

    df = (
        df
        .groupby('hash', sort=False, as_index=False)
        .progress_apply(remove_title)
        .reset_index(level=0, drop=True)
    )
    return df


def preprocess_dewiki(df):
    global GOOD_IDS_DEWIK
    if GOOD_IDS_DEWIK is None:
        GOOD_IDS_DEWIK = pd.read_pickle(META_DIR / 'dewiki_good_ids.pickle')
    df = df[df.hash.isin(GOOD_IDS_DEWIK.index)]

    return df


# TODO: wrap in class and move corpus parameter to init

def stream_corpus(
        corpus: str,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        lemmatized: bool = True,
        tags_blocklist: list = None,
        directory: PathLike = NLP_DIR,
) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
    """
    Parses a corpus in the topic-labeling package format and streams as list of strings or tuples.

    :param corpus: id or unique prefix of a topic-labeling corpus.
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param min_doc_size: Discard all documents/contexts smaller than min_chunk_size.
    :param tagged: if False: items of the yielded lists are string tokens.
                   if True: items of the yielded lists are tuples in the format
                   ``(token, pos_tag)``.
    :param lemmatized: return a lemma instead of the surface form.
    :param lowercase: Convert all tokens to lowercase if True.
    :param tags_blocklist: Remove all tokens from the contexts with pos-tags in this blocklist.
    :param directory: Optional path to a corpus. Uses the default path if None.

    :returns: Generator that yields documents/contexts as lists of tokens.
    """
    # TODO: Benchmark structured data formats for strings/documents/tokens

    if directory is None:
        directory = NLP_DIR
    else:
        directory = Path(directory)
    print(f"Streaming corpus '{corpus}' from {directory}")

    # filter files for certain prefixes
    prefixes = r'^(' + '|'.join(corpus) + r').'
    pattern = re.compile(prefixes)
    files = sorted([
        f for f in directory.iterdir()
        if f.is_file() and pattern.match(f.name)
    ])

    for file in files:
        corpus_name = file.name.split('_nlp.')[0]
        text_col = 'token' if lemmatized else 'text'
        print(f"reading from {file}")
        df = pd.read_csv(
            file, sep='\t', header=0, usecols=['hash', text_col, 'POS'],
            keep_default_na=False, dtype={'hash': int, 'token': str, 'pos': 'category'},
            lineterminator='\n', quoting=csv.QUOTE_NONE,
        )
        df.columns = ['hash', 'token', 'pos']

        if corpus_name.startswith('dewac'):
            df = preprocess_dewac(df)
        elif corpus_name.startswith('dewiki'):
            df = preprocess_dewiki(df)

        df = df.groupby('hash', sort=False)

        for _, doc in tqdm(df, total=len(df)):
            if tags_blocklist:
                doc = doc[~doc.pos.isin(tags_blocklist)]
            if lowercase:
                doc.token = doc.token.str.lower()
            if tagged:
                doc = list(doc.itertuples(index=False, name=None))
            else:
                doc = doc.token.to_list()

            # --- apply chunk_size
            if chunk_size:
                idx = 0
                while idx < len(doc):
                    chunk = doc[idx:idx + chunk_size]
                    idx += chunk_size
                    if len(chunk) >= min_doc_size:
                        yield chunk
            else:
                if len(doc) >= min_doc_size:
                    yield doc


def read_corpus(*args, **kwargs):
    return list(stream_corpus(*args, **kwargs))


def infer_file_path(
        corpus: str,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        lemmatized: bool = True,
        tags_blocklist: list = None,
        with_suffix=True,
) -> PathLike:
    """Returns a canonical file path for the given corpus and arguments."""

    cs_suffix = f'_cs{chunk_size}' if isinstance(chunk_size, int) and chunk_size > 0 else ''
    tagged_suffix = '_tagged' if tagged else ''
    lowercase_suffix = '_lc' if lowercase else ''
    lemmatized_suffix = '_lemma' if lemmatized else ''
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
        f'{lemmatized_suffix}'
        f'{filtered_suffix}'
        f'{file_suffix}'
    )
    file_path = CACHE_DIR / 'corpora' / corpus / file_name

    return file_path


def persist_transformation(
        corpus: str,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        lemmatized: bool = True,
        tags_blocklist: list = None,
        directory: PathLike = None,
        documents: List[List] = None,
):
    """
    Parses documents from the original BNC XML format and writes it as plain text to a file.

    The file is written to ``data/out/cache/bnc[args].txt``.

    :param corpus: id or unique prefix of a topic-labeling corpus.
    :param chunk_size: Splits the documents in contexts of a maximum window size.
    :param min_doc_size: Discard all documents/contexts smaller than min_chunk_size.
    :param tagged:
        if False: items of the yielded lists are string tokens.
        if True: items of the yielded lists are tuples in the format ``(token, pos_tag)``.
    :param lowercase: Converts all tokens to lowercase if True.
    :param lemmatized: return a lemma instead of the surface form.
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
        lemmatized=lemmatized,
        tags_blocklist=tags_blocklist,
    )
    if documents is None:
        documents = stream_corpus(
            corpus=corpus,
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            lemmatized=lemmatized,
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
            directory=str(directory),
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            lemmatized=lemmatized,
            tags_blocklist=tags_blocklist,
        )
        json.dump(args, fp)


def load_from_cache(
        corpus: str,
        chunk_size: int = None,
        min_doc_size: int = None,
        tagged: bool = False,
        lowercase: bool = False,
        lemmatized: bool = True,
        tags_blocklist: list = None,
        make_if_not_cached: bool = True,
        persist_if_not_cached: bool = True,
        version: str = None,
) -> List[List]:
    """

    :param corpus:
    :param chunk_size:
    :param min_doc_size:
    :param tagged:
    :param lowercase:
    :param lemmatized: return a lemma instead of the surface form.
    :param tags_blocklist:
    :param make_if_not_cached: Performs a new transformation for the given arguments,
        if loading from cache fails.
    :param persist_if_not_cached: Implies ``make_make_if_not_cached``. Writes the new
        transformation to a plain text file.
    :param version: an optional version id where the version id corresponds to a
        cached corpora file.

    :return: List of lists of tokens.
    """

    out_path = CACHE_DIR / 'corpora' / corpus / f'{version}.txt'
    if not out_path.exists():
        out_path = infer_file_path(
            corpus=corpus,
            chunk_size=chunk_size,
            min_doc_size=min_doc_size,
            tagged=tagged,
            lowercase=lowercase,
            lemmatized=lemmatized,
            tags_blocklist=tags_blocklist,
        )
    # TODO: read meta data first and verify identity of tags_blocklist

    try:
        with open(out_path, 'r') as fp:
            print(f"Loading {out_path}")
            docs = [c.strip().split() for c in fp.readlines()]
    except FileNotFoundError:
        make_if_not_cached |= persist_if_not_cached
        if make_if_not_cached:
            docs = read_corpus(
                corpus=corpus,
                chunk_size=chunk_size,
                min_doc_size=min_doc_size,
                tagged=tagged,
                lowercase=lowercase,
                lemmatized=lemmatized,
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
                    lemmatized=lemmatized,
                    tags_blocklist=tags_blocklist
                )
        else:
            docs = None

    return docs


def example(args, example_='stream'):
    if example_ == 'stream':
        docs = stream_corpus(
            corpus=args.corpus,
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
            tagged=args.tagged,
            lowercase=args.lowercase,
            lemmatized=args.lemmatized,
            tags_blocklist=args.tags_blocklist,
            directory=args.directory,
        )
        for doc in docs:
            print(len(doc), doc[:50])
    elif example_ == 'write':
        persist_transformation(
            corpus=args.corpus,
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
            tagged=args.tagged,
            lowercase=args.lowercase,
            lemmatized=args.lemmatized,
            tags_blocklist=args.tags_blocklist,
            directory=args.directory,
        )
    elif example_ == 'load':
        docs = load_from_cache(
            corpus=args.corpus,
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
            tagged=args.tagged,
            lowercase=args.lowercase,
            lemmatized=args.lemmatized,
            tags_blocklist=args.tags_blocklist,
        )
        print(docs[:10])


if __name__ == '__main__':
    args_ = parse_args()
    print(vars(args_))
    example(args_, 'write')
