import math

from nltk.corpus.reader.bnc import BNCCorpusReader
from tqdm import tqdm

from semsim.constants import BNC_DIR


def stream_bnc(directory=None, chunk_size=None, tagged=False):
    if directory is None:
        directory = BNC_DIR / 'download' / 'Texts'
    bnc = BNCCorpusReader(root=str(directory), fileids=r'[A-K]/\w*/\w*\.xml')

    fileids = bnc.fileids()
    for fileid in tqdm(fileids, total=len(fileids)):
        if tagged:
            words = bnc.tagged_words(fileids=fileid, strip_space=True)
        else:
            words = bnc.words(fileids=fileid, strip_space=True)
        doc = list(words)

        if isinstance(chunk_size, int) and len(doc) > chunk_size:
            nb_chunks = int(math.ceil(len(doc) / chunk_size))
            for i in range(nb_chunks):
                chunk = doc[i*chunk_size:(i+1)*chunk_size]
                yield chunk
        else:
            yield doc


def read_bnc(directory=None, chunk_size=None, tagged=False):
    return list(stream_bnc(directory=directory, chunk_size=chunk_size, tagged=tagged))


def main():
    docs = stream_bnc(chunk_size=1000, tagged=True)
    for doc in docs:
        print(len(doc), doc[:50])


if __name__ == '__main__':
    main()
