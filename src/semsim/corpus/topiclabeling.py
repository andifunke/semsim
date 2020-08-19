"""Converts corpora from the topic-labeling package format to the simple semsim format."""

import csv
import re
from pathlib import Path
from typing import Generator, List, Union, Tuple

import pandas as pd
from tqdm import tqdm

from semsim.constants import CORPORA_DIR
from semsim.corpus.interface import CorpusABC

tqdm.pandas()


class TopiclabelingCorpusABC(CorpusABC):

    CORPUS: str = NotImplemented
    CORPUS_DIR_DEFAULT: Path = NotImplemented

    def _stream(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """
        Parses documents from a corpus in the topic-labeling package format and
        streams as list of strings or tuples.

        :returns: Generator that yields documents/contexts as lists of tokens.
        """

        corpus_dir = self.corpus_dir
        print(f"Streaming {self.corpus} corpus from {corpus_dir}")

        # filter files for certain prefixes
        pattern = re.compile(self.corpus, re.IGNORECASE)
        files = sorted([
            f for f in corpus_dir.iterdir()
            if f.is_file() and pattern.match(f.name)
        ])
        if not files:
            raise FileNotFoundError(f"Cannot find any matching file in {corpus_dir}")

        self.nb_documents = 0
        self.nb_contexts = 0

        for file in files:
            text_col = 'token' if self.lemmatized else 'text'
            print(f"reading from {file}")
            df = pd.read_csv(
                file,
                sep='\t',
                header=0,
                usecols=['hash', text_col, 'POS'],
                keep_default_na=False,
                dtype={'hash': int, 'token': str, 'pos': 'category'},
                lineterminator='\n',
                quoting=csv.QUOTE_NONE,
            )
            df.columns = ['hash', 'token', 'pos']
            df = df.groupby('hash', sort=False)

            for _, doc in tqdm(df, total=len(df)):
                if self.blocklist:
                    doc = doc[~doc.pos.isin(self.blocklist)]
                if self.lowercase:
                    doc.token = doc.token.str.lower()
                if self.tagged:
                    doc = list(doc.itertuples(index=False, name=None))
                else:
                    doc = doc.token.to_list()
                if self.as_ids:
                    raise NotImplementedError('`as_ids` is not implemented yet.')

                # --- apply chunk_size
                yield from self._docs2chunks(doc)


class DewikiCorpus(TopiclabelingCorpusABC):
    CORPUS = 'Dewiki'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class DewacCorpus(TopiclabelingCorpusABC):
    CORPUS = 'Dewac'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class EuroparlCorpus(TopiclabelingCorpusABC):
    CORPUS = 'Europarl'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class GermanPoliticalSpeechesCorpus(TopiclabelingCorpusABC):
    CORPUS = 'GermanPoliticalSpeeches'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class SpeechesCorpus(TopiclabelingCorpusABC):
    CORPUS = 'Speeches'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class NewsCorpus(TopiclabelingCorpusABC):
    CORPUS = 'News'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'


class OnlineParticipationCorpus(TopiclabelingCorpusABC):
    CORPUS = 'OnlineParticipation'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / CORPUS / 'nlp'
