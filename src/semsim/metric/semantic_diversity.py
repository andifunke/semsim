import argparse
import warnings

# TODO: remove when tqdm fully supports pandas >= 0.25
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel
from tqdm import tqdm

from semsim import DATASET_STREAMS
from semsim.constants import SEMD_DIR, CACHE_DIR
from semsim.corpus.dataio import reader

tqdm.pandas()
np.random.seed(42)


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies and
    returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--corpus', type=str, required=True, choices=DATASET_STREAMS.keys())
    parser.add_argument('-v', '--version', type=str, required=False, default='default')
    parser.add_argument('-w', '--window', type=int, required=False, default=1000)
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
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_true', required=False)
    parser.set_defaults(lowercase=True)

    parser.add_argument('--make-corpus', dest='make_corpus', action='store_true', required=False)
    parser.add_argument('--load-corpus', dest='make_corpus', action='store_false', required=False)
    parser.set_defaults(make_corpus=False)

    parser.add_argument('--make-lsi', dest='make_lsi', action='store_true', required=False)
    parser.add_argument('--load-lsi', dest='make_lsi', action='store_false', required=False)
    parser.set_defaults(make_lsi=False)

    args = parser.parse_args()

    if args.pos_tags is not None:
        args.pos_tags = set(args.pos_tags)

    if args.make_corpus:
        args.make_lsi = True

    args.input_fn = DATASET_STREAMS[args.corpus]

    return args


def calculate_semantic_diversity(terms, dictionary, corpus, document_vectors):
    print('Calculate Semantic Diversity.')

    csc_matrix = corpus2csc(corpus, dtype=np.float32)

    mean_cos = {}
    semd_values = {}
    for term in tqdm(terms, total=len(terms)):  # TODO: vectorize
        try:
            term_id = dictionary.token2id[term]
            term_docs_sparse = csc_matrix.getrow(term_id)
            term_doc_ids = term_docs_sparse.nonzero()[1]

            # if target appears in >2000 documents, subsample 2000 at random
            if len(term_doc_ids) > 2000:
                term_doc_ids = np.random.choice(term_doc_ids, size=2000, replace=False)

            term_doc_vectors = document_vectors[term_doc_ids]
            similarities = cosine_similarity(term_doc_vectors)
            lower_tri = np.tril_indices(similarities.shape[0], k=1)
            similarities = similarities[lower_tri]
            avg_similarity = np.mean(similarities)
            semd = -np.log10(avg_similarity)
            mean_cos[term] = avg_similarity
            semd_values[term] = semd
        except KeyError:
            semd_values[term] = np.nan

    semd = pd.DataFrame([mean_cos, semd_values], index=['mean_cos', 'SemD']).T

    return semd


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


def calc_entropy(corpus, corpus_freqs):
    print('Calculating word entropy over all contexts.')

    entropy = np.zeros_like(corpus_freqs, dtype=np.float32)
    for context in tqdm(corpus, total=len(corpus)):
        context = np.asarray(context, dtype=np.int32).T
        token_ids = context[0]
        context_freq = context[1]
        corpus_freq = corpus_freqs[token_ids]
        p_c = context_freq / corpus_freq
        ic = -np.log(p_c)
        ctx_ent = p_c * ic
        entropy[token_ids] += ctx_ent

    return entropy


def calc_entropy_normalization(corpus, word_entropies, epsilon):
    print('Normalizing corpus.')

    word_entropies = word_entropies.astype(np.float32)
    epsilon = np.float32(epsilon)
    transformed_corpus = []
    for context in tqdm(corpus, total=len(corpus)):
        context_arr = np.asarray(context, dtype=np.float32).T
        token_ids = context_arr[0].astype(np.int32)
        context_freq = context_arr[1]
        context_entropy = word_entropies[token_ids]
        transformations = (np.log(context_freq) + epsilon) / context_entropy
        transformed_context = [x for x in zip(token_ids, transformations)]
        transformed_corpus.append(transformed_context)

    return transformed_corpus


def entropy_transform(corpus, dictionary, epsilon=1.0, use_cache=True):

    # TODO: individualize file name based on args
    file_name = 'entropy_transform.csv'
    dir_path = CACHE_DIR / 'SemD'
    dir_path.mkdir(exist_ok=True, parents=True)
    file_path = dir_path / file_name

    df = None
    if use_cache:
        try:
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            print(f'Could not read cache from {file_path}')

    # calculate "entropy" per token
    if df is None:
        dfs = pd.Series(dictionary.dfs, name='context_freq', dtype='int32').sort_index()
        cfs = pd.Series(dictionary.cfs, name='corpus_freq', dtype='int32').sort_index()
        df = pd.concat([dfs, cfs], axis=1, )
        wordcount_per_mil = cfs.sum() / 1_000_000
        df['freq'] = (df.corpus_freq / wordcount_per_mil).astype('float32')
        df['log_freq'] = np.log10(df.freq.values + epsilon, dtype='float32')
        df['entropy'] = calc_entropy(corpus, corpus_freqs=df.corpus_freq.values)
        df['token_id'] = df.index
        df['token'] = df.token_id.map(lambda x: dictionary[x])
        df = df.set_index('token')
        df.to_csv(file_path, sep='\t')

    # calculate transformation
    transformed_corpus = calc_entropy_normalization(
        corpus, word_entropies=df.entropy.values, epsilon=epsilon
    )

    return transformed_corpus


def tfidf_transform(bow_corpus):
    tfidf_model = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    return tfidf_corpus


def texts2corpus(
        contexts, stopwords=None, min_word_freq=1, min_contexts=1, filter_above=1, keep_n=1_000_000
):
    print(f"Generating dictionary.")

    dictionary = Dictionary(contexts, prune_at=None)

    vocab_size = len(dictionary)
    dictionary.filter_extremes(no_below=min_contexts, no_above=filter_above, keep_n=keep_n)
    print(
        f"Removing {vocab_size - len(dictionary)} tokens "
        f"appearing in less than {min_contexts} contexts."
    )
    vocab_size = len(dictionary)

    # filter noise (e.g. stopwords, special characters, infrequent words)
    if stopwords:
        bad_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=bad_ids, good_ids=None)
        print(f"Removing {len(dictionary) - vocab_size} stopword tokens.")
        vocab_size = len(dictionary)

    if min_word_freq > 1:
        bad_ids = [k for k, v in dictionary.cfs.items() if v < min_word_freq]
        dictionary.filter_tokens(bad_ids=bad_ids, good_ids=None)
        print(
            f"Removing {vocab_size - len(dictionary)} tokens with min frequency < {min_word_freq}."
        )

    dictionary.compactify()
    print(f"Dictionary size: {len(dictionary)}")

    print(f"Generating bow corpus from {len(contexts)} contexts.")
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
        # tags_allowlist=args.pos_tags,  # TODO: implement
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
    contexts = get_contexts(args)
    bow_corpus, dictionary = texts2corpus(
        contexts, min_word_freq=args.min_word_freq, min_contexts=args.min_contexts, stopwords=None
    )

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


def get_sparse_corpus(args, directory, file_name):
    if args.make_corpus:
        corpus, dictionary = make_corpus(args, directory, file_name)
    else:
        try:
            corpus, dictionary = load_corpus(args, directory, file_name)
        except FileNotFoundError as e:
            print(e)
            corpus, dictionary = make_corpus(args, directory, file_name)

    return corpus, dictionary


def make_lsi(corpus, dictionary, args, directory, file_name):
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

    return document_vectors


def load_lsi(directory, file_name):
    # --- load document vectors ---
    file_path = directory / f'{file_name}_lsi_document_vectors.csv'
    print(f"Loading document vectors from {file_path}")
    document_vectors = pd.read_csv(file_path, index_col=0, dtype=np.float32)

    return document_vectors


def get_lsi_corpus(corpus, dictionary, args, directory, file_name):
    if args.make_lsi:
        lsi_vectors = make_lsi(corpus, dictionary, args, directory, file_name)
    else:
        try:
            lsi_vectors = load_lsi(directory, file_name)
        except FileNotFoundError as e:
            print(e)
            lsi_vectors = make_lsi(corpus, dictionary, args, directory, file_name)

    return lsi_vectors


def main():
    args = parse_args()
    print(args)

    file_name = f'{args.corpus}_{args.version}'
    directory = SEMD_DIR / args.version
    directory.mkdir(exist_ok=True, parents=True)

    # --- create a sparse corpus ---
    corpus, dictionary = get_sparse_corpus(args, directory, file_name)
    lsi_vectors = get_lsi_corpus(corpus, dictionary, args, directory, file_name)

    # --- calculate SemD for vocabulary ---
    if args.terms:
        terms_path = Path(args.terms).resolve()
        with open(terms_path) as fp:
            terms = [line.strip() for line in fp.readlines()]
            print(terms)
        file_path = terms_path.with_suffix('.semd')
    else:
        terms = dictionary.token2id.keys()
        file_path = directory / f'{file_name}.semd'
    semd_values = calculate_semantic_diversity(terms, dictionary, corpus, lsi_vectors.values)

    # - save SemD values for vocabulary -
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path, sep='\t')

    print(semd_values)


if __name__ == '__main__':
    main()
