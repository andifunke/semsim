import math

import spacy
from nltk.corpus.reader.bnc import BNCCorpusReader

from semsim.constants import BNC_DIR


def main():
    # nlp = spacy.load('en')
    root = BNC_DIR / 'download' / 'Texts'
    bnc = BNCCorpusReader(root=str(root), fileids=r'[A-K]/\w*/\w*\.xml')

    docs = []
    for fileid in bnc.fileids():
        words = bnc.words(fileids=fileid, strip_space=True)
        words = list(words)
        doc = words
        # doc = ' '.join(words)
        # doc = nlp(doc)

        # Apply chunking for SemD
        chunk_size = 1000
        if len(doc) <= chunk_size:
            docs.append(doc)
        else:
            chunks = []
            nb_chunks = int(math.ceil(len(doc) / chunk_size))
            for i in range(nb_chunks):
                chunk = doc[i*chunk_size:(i+1)*chunk_size]
                chunks.append(chunk)
            docs += chunks

    # TODO


if __name__ == '__main__':
    main()
