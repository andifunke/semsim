import hashlib
import json
from pathlib import Path
from typing import Generator, List, Union, Tuple, Any

from tqdm import tqdm

from semsim.constants import PathLike, CACHE_DIR, LF, TAGSET


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
        self.blocklist = self._init_blocklist(tags_allowlist, tags_blocklist)
        self.as_ids = as_ids

        self._name = name
        self._corpus_dir = Path(corpus_dir) if corpus_dir else None
        self._cache_dir = Path(cache_dir) if cache_dir else None

        self.nb_documents = None
        self.nb_contexts = None

    @staticmethod
    def _init_blocklist(tags_allowlist, tags_blocklist):
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

    @staticmethod
    def hexhash(obj: Any) -> str:
        """Hashes a string and returns the MD5 hexadecimal hash as a string."""

        story_hash = hashlib.md5(str(obj).strip().encode('utf8'))
        hex_digest = story_hash.hexdigest()

        return hex_digest

    @property
    def name(self) -> str:
        """Returns a parameterized or predefined name of the corpus."""

        if self._name:
            return self._name

        if self.blocklist:
            tags = sorted(self.blocklist)
            tags_hash = self.hexhash(tags)
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
    def corpus_dir(self) -> Path:
        """Returns the base directory of original corpus files."""

        return self._corpus_dir if self._corpus_dir else self.CORPUS_DIR_DEFAULT

    @property
    def cache_dir(self) -> Path:
        """Returns the cache directory of transformed corpus."""

        return self._cache_dir if self._cache_dir else CACHE_DIR / 'corpora' / self.corpus

    @property
    def cache_file(self) -> Path:
        """Returns the full file path of the cached corpus transformation."""

        return (self.cache_dir / self.name).with_suffix('.txt')

    @property
    def state(self) -> dict:
        """Returns a dictionary containing the field values of the corpus instance."""

        state = dict(
            corpus=self.corpus,
            name=self.name,
            corpus_dir=self.corpus_dir.as_posix(),
            cache_dir=self.cache_dir.as_posix(),
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

    def _stream(self):
        raise NotImplementedError

    def stream(self, cached=True) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """
        Transforms documents from an original corpus into the simple semsim format.

        Yields one document at a time. A document can be either a list of tokens
        or a list of 2-tuples in the format ``(token, pos_tag)``.

        :param cached: if True: additionally writes the transformation to a disk cache.
        """

        if cached:
            try:
                yield from self.stream_cache()
            except FileNotFoundError:
                yield from self.write_cache()
        else:
            yield from self._stream()

    def load(self, cached=True) -> List[List[Union[str, Tuple[str, str]]]]:
        """
        Loads all documents into memory and returns it as a list.

        :param cached: if True: additionally writes the transformation to a disk cache.
        """

        return list(self.stream(cached=cached))

    def write_cache(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """Transforms original documents on the fly and writes them to a disk cache."""

        # - write corpus as plain text -
        cache_file = self.cache_file
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.cache_file, 'w') as fp:
            print(f"Caching corpus to {self.cache_file}")
            for i, doc in enumerate(self._stream()):
                yield doc
                doc = ' '.join(doc).replace('\n', LF)
                fp.write(doc + '\n')
            print(f'{i} lines written.')

        # - write arguments as meta data -
        with open(self.cache_file.with_suffix('.json'), 'w') as fp:
            json.dump(self.state, fp)

    def stream_cache(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """Streams transformed documents from a disk cache."""

        with open(self.cache_file.with_suffix('.json')) as fp:
            state = json.load(fp)
            if not self.nb_documents:
                self.nb_documents = state.get('nb_documents')
            if not self.nb_contexts:
                self.nb_contexts = state.get('nb_contexts')

        with open(self.cache_file) as fp:
            print(f"Reading corpus from {self.cache_file}")
            for doc in tqdm(fp, unit=' documents'):
                doc = doc.replace(LF, '\n')
                yield doc

    def load_cache(self) -> List[List[Union[str, Tuple[str, str]]]]:
        """Loads the disk cache into memory."""

        return list(self.stream_cache())
