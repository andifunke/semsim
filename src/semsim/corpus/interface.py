import hashlib
import json
import re
from pathlib import Path
from typing import Generator, List, Union, Tuple, Any

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from tqdm import tqdm

from semsim.constants import PathLike, CACHE_DIR, LF, TAGSET


def init_blocklist(tags_allowlist, tags_blocklist):
    if tags_allowlist and tags_blocklist:
        raise ValueError(
            'allowlist and blocklist for POS-tags are exclusive. Use only one of them.'
        )

    if tags_blocklist:
        blocklist = set(tags_blocklist) if tags_blocklist else {}
        unknowns = {t for t in blocklist if t not in TAGSET}
        if unknowns:
            raise ValueError(f'POS-tags {unknowns} in blocklist are unknown')
        return blocklist

    if tags_allowlist:
        allowlist = set(tags_allowlist) if tags_allowlist else {}
        unknowns = {t for t in allowlist if t not in TAGSET}
        if unknowns:
            raise ValueError(f'POS-tags {unknowns} in allowlist are unknown')
        blocklist = {t for t in TAGSET if t not in allowlist}
        return blocklist

    return {}


def hexhash(obj: Any) -> str:
    """Hashes a string and returns the MD5 hexadecimal hash as a string."""

    # TODO: catch
    obj = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    story_hash = hashlib.md5(obj.strip().encode('utf8'))
    hex_digest = story_hash.hexdigest()

    return hex_digest


class CorpusABC:

    CORPUS: str = NotImplemented
    CORPUS_DIR_DEFAULT: Path = NotImplemented

    def __init__(
            self,
            name: str = None,
            corpus_dir: PathLike = None,
            cache_dir: PathLike = None,
            chunk_size: int = 0,
            min_doc_size: int = 0,
            tagged: bool = False,
            lowercase: bool = False,
            lemmatized: bool = False,
            tags_allowlist: list = None,
            tags_blocklist: list = None,
            as_ids: bool = False,
    ):
        """
        Parses documents from a original corpus in a common simple format.

        The common format consists of lists per document or context. A context is a chunk
        of a document with a maximum size. Each document list contains either of just the
        tokens or a 2-tuple of ``(token, pos_tag)``.

        The transformation from the original corpus to the common format can be cached
        if the corpus is used multiple times.

        For large corpora two memory efficiency optimizations are available:
        - corpus stream: read one document at a time from disc
        - replacing string tokens with integers, using spaCy's hashing function.
        While reducing the memory footprint both optimizations increase I/O latency and
        computation demands.

        :param name: Specify an optional name for a particular corpus transformation.
            If None, a default name will be inferred on basis of the given parameters.
        :param corpus_dir: Optional path to the corpus. Uses a default path if None.
        :param cache_dir: Optional path to the transformation cache. Uses a default path if None.
        :param chunk_size: Splits the documents in contexts of a maximum window size.
        :param min_doc_size: Discard all documents/contexts smaller than min_chunk_size.
        :param tagged: if False: items of documents/contexts are string tokens.
            if True: items of documents/contexts are tuples in the format ``(token, pos_tag)``.
        :param lowercase: Convert all tokens to lowercase if True.
        :param lemmatized: Replace the surface form with its lemma (if lemmata are available in
            the original corpus)
        :param tags_allowlist: Remove all tokens with pos-tags not in the allowlist.
            `tags_blocklist` must be None.
        :param tags_blocklist: Remove all tokens with pos-tags in the blocklist.
            `tags_allowlist` must be None.
        :param as_ids: replace string tokens by a hash value.
        """

        self.corpus = self.CORPUS

        self.chunk_size = chunk_size if isinstance(chunk_size, int) else 0
        self.min_doc_size = min_doc_size if isinstance(min_doc_size, int) else 0
        self.tagged = tagged
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.blocklist = init_blocklist(tags_allowlist, tags_blocklist)
        self.as_ids = as_ids

        self._name = name
        self._corpus_dir = Path(corpus_dir) if corpus_dir else None
        self._cache_dir = Path(cache_dir) if cache_dir else None

        self.nb_documents = None
        self.nb_contexts = None

        self.cached = True
        self.streamed = True

        self._contexts = dict(
            data=None,
            id=None,
            path=None,
            state=None,
            state_path=None,
        )

        self._dictionary = None
        self._dictionary_state = None
        self._dictionary_id = None
        self._dictionary_path = None

        self._bow = None
        self._bow_state = None
        self._bow_id = None
        self._bow_path = None

        self._tfidf = None
        self._tfidf_state = None
        self._tfidf_id = None
        self._tfidf_path = None

        self._logentropy = None
        self._logentropy_state = None
        self._logentropy_id = None
        self._logentropy_path = None

        self._lsi = None
        self._lsi_state = None
        self._lsi_id = None
        self._lsi_path = None

        self._lda = None
        self._lda_state = None
        self._lda_id = None
        self._lda_path = None

        self._w2v = None
        self._w2v_state = None
        self._w2v_id = None
        self._w2v_path = None

        self._d2v = None
        self._d2v_state = None
        self._d2v_id = None
        self._d2v_path = None

        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def corpus_dir(self) -> Path:
        """Returns the base directory of original corpus files."""

        return self._corpus_dir if self._corpus_dir else self.CORPUS_DIR_DEFAULT

    @property
    def cache_dir(self) -> Path:
        """Returns the cache directory of transformed corpus."""

        return self._cache_dir if self._cache_dir else CACHE_DIR / 'corpora' / self.corpus

    def _docs2chunks(self, doc: List) -> Generator[List, None, None]:
        """
        Returns either the full document or splits the document into chunks
        based on the chunk_size and document length.

        Additionally counts the number of documents and chunks.
        """

        if self.chunk_size:
            idx = 0
            nb_contexts = 0
            while idx < len(doc):
                chunk = doc[idx:idx + self.chunk_size]
                idx += self.chunk_size
                if len(chunk) >= self.min_doc_size:
                    nb_contexts += 1
                    yield chunk
            self.nb_contexts += nb_contexts
            self.nb_documents += bool(nb_contexts)
        else:
            if len(doc) >= self.min_doc_size:
                self.nb_documents += 1
                self.nb_contexts = self.nb_documents
                yield doc

    def _write_cache(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """Transforms original documents on the fly and writes them to a disk cache."""

        # - write corpus as plain text -
        with open(self.cache_file('.txt'), 'w') as fp:
            print(f"Caching corpus to {self.cache_file}")
            for i, doc in enumerate(self._stream()):
                yield doc
                doc = ' '.join(doc).replace('\n', LF)
                fp.write(doc + '\n')
            print(f'{i} lines written.')

        # - write arguments as meta data -
        with open(self.cache_file('.json'), 'w') as fp:
            json.dump(self.contexts_state, fp)

    def _stream_cache(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """Streams transformed documents from a disk cache."""

        with open(self.cache_file('.json')) as fp:
            state = json.load(fp)
            if not self.nb_documents:
                self.nb_documents = state.get('nb_documents')
            if not self.nb_contexts:
                self.nb_contexts = state.get('nb_contexts')

        with open(self.cache_file('.txt')) as fp:
            print(f"Reading corpus from {self.cache_file}")
            for doc in tqdm(fp, unit=' documents'):
                doc = doc.replace(LF, '\n')
                yield doc

    def _load_cache(self) -> List[List[Union[str, Tuple[str, str]]]]:
        """Loads the disk cache into memory."""

        return list(self._stream_cache())

    def _stream(self):
        raise NotImplementedError

    def _load_contexts(self) -> List[List[Union[str, Tuple[str, str]]]]:
        """Loads all documents into memory and returns it as a list."""

        return list(self._stream_contexts())

    def _stream_contexts(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """
        Transforms documents from an original corpus into the simple semsim format.

        Yields one document at a time. A document can be either a list of tokens
        or a list of 2-tuples in the format ``(token, pos_tag)``.
        """

        if self.cached:
            try:
                yield from self._stream_cache()
            except FileNotFoundError:
                yield from self._write_cache()
        else:
            yield from self._stream()

    # --- CONTEXTS ---

    @property
    def contexts_path(self) -> Path:
        if self._contexts['path'] is None:
            self._contexts['path'] = self.cache_dir / self.contexts_id / 'contexts.txt'
        return self._contexts['path']

    @contexts_path.setter
    def contexts_path(self, value):
        self._contexts['path'] = Path(value)

    @property
    def contexts_id(self) -> str:
        """Returns a parameterized or predefined name of the corpus."""

        if self._name:
            return self._name

        if self.blocklist:
            tags = sorted(self.blocklist)
            tags_hash = hexhash(tags)
        else:
            tags_hash = ''

        name = (
            f'_cs{self.chunk_size}' if self.chunk_size else ''
            f'_mds{self.min_doc_size}' if self.min_doc_size else ''
            f'_tg' if self.tagged else ''
            f'_lc' if self.lowercase else ''
            f'_lm' if self.lemmatized else ''
            f'_ids' if self.as_ids else ''
            f'_{tags_hash}' if tags_hash else ''
        )
        name = f'{self.corpus}_default' if not name else f'{self.corpus}_{name}'

        return name

    @property
    def contexts_state(self) -> dict:
        """Returns a dictionary containing the field values of the corpus instance."""

        state = dict(
            corpus=self.corpus,
            name=self.contexts_id,
            corpus_dir=self.corpus_dir.as_posix(),
            cache_dir=self.contexts_path.as_posix(),
            chunk_size=self.chunk_size,
            min_doc_size=self.min_doc_size,
            tagged=self.tagged,
            lowercase=self.lowercase,
            lemmatized=self.lemmatized,
            blocklist=sorted(self.blocklist),
            nb_documents=self.nb_documents,
            nb_contexts=self.nb_contexts,
        )
        return state

    def contexts(self):
        if self.streamed:
            return self._stream_contexts()

        return self._load_contexts()

    # --- DICTIONARY ---

    def _cache_dictionary(self, dictionary):
        # - save dictionary -
        file_path = self.cache_file('.dict')
        print(f"Saving {file_path}")
        dictionary.save(str(file_path))

        # - save dictionary frequencies as plain text -
        dict_table = pd.Series(dictionary.token2id).to_frame(name='idx')
        dict_table['freq'] = dict_table['idx'].map(dictionary.cfs.get)
        dict_table = dict_table.reset_index()
        dict_table = dict_table.set_index('idx', drop=True).rename({'index': 'term'}, axis=1)
        dict_table = dict_table.sort_index()
        file_path = self.cache_file('.dict.csv')
        print(f"Saving {file_path}")
        dict_table.to_csv(file_path, sep='\t')

    def _build_dictionary(
            self, vocab=None, vocab_exclusive=False, min_contexts=0, stopwords=None,
            min_word_freq=0, words_only=False, keep_n=False
    ):
        print(f"Building dictionary.")
        contexts = self.contexts()
        dictionary = Dictionary(contexts, prune_at=None)
        vocab_size = len(dictionary)

        # load allowlist from a predefined vocabulary
        if vocab:
            with open(vocab) as fp:
                print(f'Loading vocab file {vocab}')
                vocab_terms = sorted({line.strip() for line in fp.readlines()})
                print(f'{len(vocab_terms)} terms loaded.')
        else:
            vocab_terms = []

        if vocab_exclusive:
            good_ids = [
                dictionary.token2id[token] for token in vocab_terms
                if token in dictionary.token2id
            ]
            dictionary.filter_tokens(good_ids=good_ids)
            print(
                f"Removing {vocab_size - len(dictionary)} tokens not in predefined vocab."
            )
        else:
            if min_contexts:
                dictionary.filter_extremes(
                    no_below=min_contexts, no_above=1., keep_n=None, keep_tokens=vocab_terms
                )
                print(
                    f"Removing {vocab_size - len(dictionary)} terms "
                    f"appearing in less than {min_contexts} contexts."
                )
                vocab_size = len(dictionary)

            # filter noise (e.g. stopwords, special characters, infrequent words)
            if stopwords:
                bad_ids = [
                    dictionary.token2id[term] for term in stopwords
                    if term not in vocab_terms
                ]
                dictionary.filter_tokens(bad_ids=bad_ids)
                print(
                    f"Removing {len(dictionary) - vocab_size} stopword tokens."
                )
                vocab_size = len(dictionary)

            if min_word_freq > 1:
                bad_ids = [
                    k for k, v in dictionary.cfs.items()
                    if v < min_word_freq and dictionary[k] not in vocab_terms
                ]
                dictionary.filter_tokens(bad_ids=bad_ids)
                print(
                    f"Removing {vocab_size - len(dictionary)} terms with min frequency "
                    f"< {min_word_freq}."
                )
                vocab_size = len(dictionary)

            if words_only:
                re_word = re.compile(r"^[^\d\W]+$")
                bad_ids = [
                    dictionary.token2id[term] for term in dictionary.token2id
                    if re_word.match(term) is None and term not in vocab_terms
                ]
                dictionary.filter_tokens(bad_ids=bad_ids)
                print(
                    f"Removing {vocab_size - len(dictionary)} tokens which are "
                    f"not regular words."
                )
                vocab_size = len(dictionary)

            if keep_n:
                dictionary.filter_extremes(
                    no_below=1, no_above=1., keep_n=keep_n, keep_tokens=vocab_terms
                )
                print(
                    f"Removing {vocab_size - len(dictionary)} terms to keep "
                    f"<= {keep_n} terms."
                )

        dictionary.compactify()
        print(f"Dictionary size: {len(dictionary)}")
        if self.cached:
            self._cache_dictionary(dictionary)

        return dictionary

    @property
    def dictionary(self):
        if self._dictionary is None:
            try:
                if not self.cached:
                    raise FileNotFoundError
                # todo: check whether the dictionary parameters fit with the cached version
                self._dictionary = Dictionary.load(str(self.cache_dir))
            except FileNotFoundError:
                self.dictionary()


    @dictionary.setter
    def dictionary(
            self, vocab=None, vocab_exclusive=False, min_contexts=0, stopwords=None,
            min_word_freq=0, words_only=False, keep_n=False
    ):
        if self._dictionary is None:
            try:
                if not self.cached:
                    raise FileNotFoundError
                # todo: check whether the dictionary parameters fit with the cached version
                self._dictionary = Dictionary.load(str(self.cache_dir))
            except FileNotFoundError:
                self._dictionary = self._build_dictionary(
                    vocab=vocab,
                    vocab_exclusive=vocab_exclusive,
                    min_contexts=min_contexts,
                    stopwords=stopwords,
                    min_word_freq=min_word_freq,
                    words_only=words_only,
                    keep_n=keep_n,
                )

    def _cache_bow(self, bow_corpus):
        # - save bow corpus -
        file_path = directory / f'{file_name}_bow.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), bow_corpus)

    def _build_bow(self):

    def bow(self):
        if

        contexts = self.contexts()
        dictionary = self.dictionary()
        try:
            print(f"Generating bow corpus from {len(contexts)} contexts.")
            bow_corpus = [dictionary.doc2bow(text) for text in contexts]
        except TypeError:
            print(f"Generating bow corpus from {self.nb_contexts} contexts.")
            bow_corpus = map(lambda text: dictionary.doc2bow(text), contexts)

        if self.cached:
            self._cache_bow(bow_corpus)

        return bow_corpus
