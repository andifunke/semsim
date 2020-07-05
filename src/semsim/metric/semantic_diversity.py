import argparse
from collections import Counter
from pathlib import Path
from time import time
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from semsim import DATASET_STREAMS
from semsim.constants import SEMD_DIR
from semsim.corpus.dataio import reader

tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_STREAMS.keys())
    parser.add_argument('--version', type=str, required=False, default='default')
    parser.add_argument('--window', type=int, required=False, default=1000)
    parser.add_argument('--min-word-freq', type=int, required=False, default=50)
    parser.add_argument('--min-contexts', type=int, required=False, default=40)
    parser.add_argument('--nb-topics', type=int, required=False, default=300)
    parser.add_argument('--pos-tags', nargs='*', type=str, required=False)
    parser.add_argument('--normalization', type=str, required=False,
                        default='entropy', choices=['none', 'entropy', 'tfidf'],
                        help="BOW-frequency normalization method.")
    parser.add_argument('--terms', type=str, required=False,
                        help="File path containing terms")

    parser.add_argument('--lowercase', action='store_true', required=False)
    parser.set_defaults(lowercase=False)

    parser.add_argument('--make-corpus', dest='make_corpus', action='store_true', required=False)
    parser.add_argument('--load-corpus', dest='make_corpus', action='store_false', required=False)
    parser.set_defaults(make_corpus=False)

    parser.add_argument('--make-lsi', dest='make_lsi', action='store_true', required=False)
    parser.add_argument('--load-lsi', dest='make_lsi', action='store_false', required=False)
    parser.set_defaults(make_lsi=False)

    args = parser.parse_args()

    if args.pos_tags is not None:
        args.pos_tags = set(args.pos_tags)

    if args.make_contexts:
        args.make_corpus = True
    if args.make_corpus:
        args.make_lsi = True

    args.input_fn = DATASET_STREAMS[args.dataset]

    return args


def calculate_semantic_diversity(terms, dictionary, corpus, document_vectors):
    csc_matrix = corpus2csc(corpus, dtype=np.float32)

    semd_values = {}
    for term in tqdm(terms, total=len(terms)):
        try:
            term_id = dictionary.token2id[term]
            term_docs_sparse = csc_matrix.getrow(term_id)
            term_doc_ids = term_docs_sparse.nonzero()[1]
            term_doc_vectors = document_vectors[term_doc_ids]
            similarities = cosine_similarity(term_doc_vectors)
            avg_similarity = np.mean(similarities)
            semd = -np.log10(avg_similarity)
            semd_values[term] = semd
        except KeyError:
            semd_values[term] = np.nan

    return pd.Series(semd_values)


def lsi_transform(corpus, dictionary, nb_topics=300, cache_in_memory=False):
    if cache_in_memory:
        print("Loading corpus into memory")
        corpus = list(corpus)
    print(f"Size of train_set={len(corpus)}")

    # --- train ---
    print(f"Training LSI model with {nb_topics} topics")
    model = LsiModel(corpus=corpus, num_topics=nb_topics, id2word=dictionary, dtype=np.float32)

    # --- get vectors ---
    term_vectors = model.projection.u
    term_vectors = pd.DataFrame(term_vectors, index=dictionary.token2id.keys())

    lsi_corpus = model[corpus]
    document_vectors = corpus2dense(lsi_corpus, 300, num_docs=len(corpus)).T
    document_vectors = pd.DataFrame(document_vectors)

    return model, document_vectors, term_vectors


def docs_to_lists(token_series):
    return token_series.tolist()


def entropy_transform(corpus, dictionary, epsilon=1.0):
    # calculate "entropy" per token
    print('Calculating word entropy over all contexts.')
    dfs = pd.Series(dictionary.dfs, name='contexts', dtype='int32').sort_index()
    cfs = pd.Series(dictionary.cfs, name='wordcount', dtype='int32').sort_index()
    df = pd.concat([dfs, cfs], axis=1,)
    wordcount_per_mil = cfs.sum() / 1_000_000
    df['freq'] = (df.wordcount / wordcount_per_mil).astype('float32')
    df['log_freq'] = np.log10(df.freq.values + epsilon, dtype='float32')
    df.loc[:, 'entropy'] = 0.

    for context in tqdm(corpus, total=len(corpus)):
        if len(context) < 2:
            continue
        ctx_wordcount = pd.DataFrame.from_records(context).set_index(0).squeeze()
        corpus_wordcount = df.wordcount[ctx_wordcount.index.values]
        p_c = ctx_wordcount / corpus_wordcount
        ic = -np.log(p_c)
        ctx_ent = p_c * ic
        df.iloc[ctx_wordcount.index.values, :].entropy += ctx_ent

    # calculate transformed values
    print('Normalizing corpus.')
    entropy_corpus = [
        [(i, (np.log(v) + epsilon) / df.entropy[i]) for i, v in context]
        for context in tqdm(corpus, total=len(corpus))
    ]

    return entropy_corpus


def tfidf_transform(bow_corpus):
    tfidf_model = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    return tfidf_corpus


def texts2corpus(contexts, stopwords=None, min_contexts=1, filter_above=1, keep_n=200_000):
    print(f"Generating bow corpus and dictionary")

    dictionary = Dictionary(contexts, prune_at=None)
    dictionary.filter_extremes(no_below=min_contexts, no_above=filter_above, keep_n=keep_n)

    # filter some noise (e.g. special characters)
    if stopwords:
        stopword_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=stopword_ids, good_ids=None)

    bow_corpus = [dictionary.doc2bow(text) for text in contexts]

    return bow_corpus, dictionary


# TODO: deprecated
def chunks_from_documents(documents: Iterable, window_size: int) -> List:
    contexts = []
    for document in documents:
        if len(document) > window_size:
            chunks = [document[x:x+window_size] for x in range(0, len(document), window_size)]
            contexts += chunks
        else:
            contexts.append(document)

    return contexts


def get_contexts(args):
    read_fn = reader(args.corpus)
    contexts = read_fn(
        chunk_size=args.window,
        tagged=False,
        lowercase=args.lowercase,
        tags_blocklist=['PUL', 'PUN', 'PUQ'],  # TODO: add to args
        make_if_not_cached=True,
        persist_if_not_cached=True,
    )
    return contexts


def normalize(bow_corpus, dictionary, normalization, directory, file_name):
    if normalization == 'tfidf':
        # - tfidf transform corpus -
        tfidf_corpus = tfidf_transform(bow_corpus)

        # - save tf-idf corpus -
        file_path = directory / f'{file_name}_tfidf.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), tfidf_corpus)
        corpus = tfidf_corpus
    elif normalization == 'entropy':
        # - log transform and entropy-normalize corpus -
        entropy_corpus = entropy_transform(bow_corpus, dictionary)

        # - save entropy-normalized corpus -
        file_path = directory / f'{file_name}_entropy.mm'
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), entropy_corpus)
        corpus = entropy_corpus
    else:
        corpus = bow_corpus

    return corpus


def make_corpus(args, directory, file_name):
    contexts = get_contexts(args, directory, file_name)
    bow_corpus, dictionary = texts2corpus(contexts, min_contexts=args.min_contexts, stopwords=None)

    # - save dictionary -
    file_path = directory / f'{file_name}.dict'
    print(f"Saving {file_path}")
    dictionary.save(str(file_path))

    # - save dictionary frequencies as plain text -
    dict_table = pd.Series(dictionary.token2id).to_frame(name='idx')
    dict_table['freq'] = dict_table['idx'].map(dictionary.cfs.get)
    dict_table = dict_table.reset_index()
    dict_table = dict_table.set_index('idx', drop=True).rename({'index': 'token'}, axis=1)
    dict_table = dict_table.sort_index()
    file_path = directory / f'{file_name}_dict.csv'
    print(f"Saving {file_path}")
    # dictionary.save_as_text(file_path, sort_by_word=False)
    dict_table.to_csv(file_path, sep='\t')

    # - save bow corpus -
    file_path = directory / f'{file_name}_bow.mm'
    print(f"Saving {file_path}")
    MmCorpus.serialize(str(file_path), bow_corpus)
    corpus = normalize(bow_corpus, dictionary, args.normalization, directory, file_name)

    return corpus, dictionary


def load_bow_corpus(directory, file_name):
    file_name += '_bow'
    file_path = directory / f'{file_name}.mm'
    print(f"Loading BOW corpus from {file_path}")
    corpus = MmCorpus(str(file_path))

    return corpus


def load_corpus(args, directory, file_name):
    # - load dictionary -
    file_path = directory / f'{file_name}.dict'
    print(f"Loading dictionary from {file_path}")
    dictionary = Dictionary.load(str(file_path))

    # - load corpus -
    if args.normalization is None:
        return load_bow_corpus(directory, file_name)

    try:
        if args.normalization == 'tfidf':
            file_path = directory / f'{file_name}_tfidf.mm'
        elif args.normalization == 'entropy':
            file_path = directory / f'{file_name}_entropy.mm'

        print(f"Loading corpus from {file_path}")
        corpus = MmCorpus(str(file_path))
    except FileNotFoundError as e:
        print(e)
        bow_corpus = load_bow_corpus(directory, file_name)
        corpus = normalize(bow_corpus, dictionary, args.normalization, directory, file_name)

    return corpus, dictionary


def get_corpus(args, directory, file_name):
    if args.make_corpus:
        corpus, dictionary = make_corpus(args, directory, file_name)
    else:
        try:
            corpus, dictionary = load_corpus(args, directory, file_name)
        except FileNotFoundError as e:
            print(e)
            corpus, dictionary = make_corpus(args, directory, file_name)

    return corpus, dictionary


def get_document_vectors(corpus, dictionary, args, directory, file_name):
    if args.make_lsi:
        model, document_vectors, term_vectors = lsi_transform(
            corpus=corpus, dictionary=dictionary, nb_topics=args.nb_topics,
            cache_in_memory=True
        )

        # --- save model ---
        file_path = directory / f'{file_name}_lsi.model'
        print(f"Saving model to {file_path}")
        model.save(str(file_path))

        # --- save document vectors ---
        file_path = directory / f'{file_name}_lsi_document_vectors.csv'
        print(f"Saving document vectors to {file_path}")
        document_vectors.to_csv(file_path)

        # --- save term vectors ---
        file_path = directory / f'{file_name}_lsi_term_vectors.csv'
        print(f"Saving document vectors to {file_path}")
        term_vectors.to_csv(file_path)
    else:
        # --- load document vectors ---
        file_path = directory / f'{file_name}_lsi_document_vectors.csv'
        print(f"Loading document vectors from {file_path}")
        document_vectors = pd.read_csv(file_path, index_col=0)

    return document_vectors


def main():
    args = parse_args()
    print(args)

    file_name = f'{args.dataset}_{args.version}'
    directory = SEMD_DIR / args.version
    directory.mkdir(exist_ok=True, parents=True)

    corpus, dictionary = get_corpus(args, directory, file_name)
    document_vectors = get_document_vectors(corpus, dictionary, args, directory, file_name)

    # --- calculate semd for vocabulary ---
    if args.terms:
        terms_path = Path(args.terms).resolve()
        with open(terms_path) as fp:
            terms = [line.strip() for line in fp.readlines()]
            print(terms)
        file_path = terms_path.with_suffix('.semd')
    else:
        terms = dictionary.token2id.keys()
        file_path = directory / f'{file_name}.semd'
    semd_values = calculate_semantic_diversity(terms, dictionary, corpus, document_vectors.values)

    # - save SemD values for vocabulary -
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path)

    print(semd_values)


if __name__ == '__main__':
    main()
