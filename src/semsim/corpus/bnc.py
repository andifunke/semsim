from typing import Generator, List, Union, Tuple

from nltk.corpus.reader.bnc import BNCCorpusReader
from tqdm import tqdm

from semsim.constants import CORPORA_DIR, BNC_TAGSET
from semsim.corpus.interface import CorpusABC


class BNCCorpus(CorpusABC):

    CORPUS = 'BNC'
    CORPUS_DIR_DEFAULT = CORPORA_DIR / 'BNC' / 'ota_20.500.12024_2554' / 'download' / 'Texts'

    def _stream(self) -> Generator[List[Union[str, Tuple[str, str]]], None, None]:
        """
        Parses documents from the original BNC XML corpus and streams as list of strings or tuples.

        :returns: Generator that yields documents/contexts as lists of tokens.
        """

        print(f'Streaming BNC corpus from {self.corpus_dir}')
        bnc = BNCCorpusReader(root=str(self.corpus_dir), fileids=r'[A-K]/\w*/\w*\.xml')

        apply_convert = lambda x: (x[0].lower() if self.lowercase else x[0], BNC_TAGSET[x[1]])
        apply_filter = lambda x: x[1] not in self.blocklist
        apply_tagged = lambda x: x if self.tagged else x[0]

        self.nb_documents = 0
        self.nb_contexts = 0

        fileids = bnc.fileids()
        for fileid in tqdm(fileids, total=len(fileids)):
            raw_doc = bnc.tagged_words(fileid, c5=False, strip_space=True, stem=self.lemmatized)
            raw_doc = map(apply_convert, raw_doc)
            raw_doc = filter(apply_filter, raw_doc)
            raw_doc = map(apply_tagged, raw_doc)
            if self.as_ids:
                raise NotImplementedError('`as_ids` is not implemented yet.')
            doc = list(raw_doc)

            # --- apply chunk_size ---
            yield from self._docs2chunks(doc)
