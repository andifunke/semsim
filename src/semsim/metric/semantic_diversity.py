import argparse
import warnings
from pathlib import Path

# TODO: remove when tqdm fully supports pandas >= 0.25
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from semsim import DATASET_STREAMS
from semsim.constants import SEMD_DIR
from semsim.corpus.dataio import reader
from semsim.corpus.bnc import infer_file_path

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
    parser.add_argument('-v', '--version', type=str, required=False, default=None)
    parser.add_argument('-p', '--project', type=str, required=False, default='')
    parser.add_argument('-w', '--window', type=int, required=False, default=1000)
    parser.add_argument('-m', '--min-doc-size', type=int, required=False, default=1,
                        help="Discard all documents/chunk smaller than --min-doc-size.")
    parser.add_argument('--min-word-freq', type=int, required=False, default=50)
    parser.add_argument('--min-contexts', type=int, required=False, default=40)
    parser.add_argument('--keep-n', type=int, required=False, default=None)
    parser.add_argument('--nb-topics', type=int, required=False, default=300)
    parser.add_argument('--epsilon', type=float, required=False, default=0.0,
                        help="Add an offset to each BOW-matrix entry before taking the log.")
    parser.add_argument('--pos-tags', nargs='*', type=str, required=False)
    parser.add_argument('--normalization', type=str, required=False,
                        default='entropy', choices=['none', 'entropy', 'tfidf'],
                        help="BOW-frequency normalization method.")
    parser.add_argument('--terms', type=str, required=False,
                        help="File path containing the terms per line to calculate SemD values "
                             "for.")
    parser.add_argument('--vocab', type=str, required=False,
                        help="File path containing terms per line to include in the "
                             "term-document-matrix. Any other term is excluded.")

    parser.add_argument('--lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
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

    if args.project and not args.version:
        args.version = args.project

    if not args.version:
        args.version = infer_file_path(
            chunk_size=args.window,
            min_doc_size=args.min_doc_size,
            tagged=False,
            lowercase=args.lowercase,
            tags_blocklist=[],
            with_suffix=False,
        )

    args.input_fn = DATASET_STREAMS[args.corpus]

    return args


def calculate_semantic_diversity(terms, dictionary, corpus, document_vectors):
    print('Calculate Semantic Diversity.')

    csc_matrix = corpus2csc(corpus, dtype=np.float32)
    assert csc_matrix.shape[0] == len(dictionary)
    assert csc_matrix.shape[1] == len(corpus)

    mean_cos = {}
    semd_values = {}
    for term in tqdm(terms, total=len(terms)):  # TODO: vectorize
        try:
            term_id = dictionary.token2id[term]
            term_docs_sparse = csc_matrix.getrow(term_id)
            term_doc_ids = term_docs_sparse.nonzero()[1]
            # TODO: why is one entry in the dictionary missing from the tdm?
            # TODO: check plausibility of term_doc_ids

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
        except (KeyError, ValueError, IndexError):
            semd_values[term] = np.nan

    semd = pd.DataFrame([mean_cos, semd_values], index=['mean_cos', 'SemD']).T

    return semd


def lsi_projection(corpus, dictionary, nb_topics=300, cache_in_memory=False):
    if cache_in_memory and not isinstance(corpus, list):
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


def lsi_projection_sklearn(corpus, nb_topics=300):
    csc_matrix = corpus2csc(corpus, dtype=np.float32).T
    print(f"Size of train_set={csc_matrix.shape[0]}")

    # --- train ---
    print(f"Training LSI model with {nb_topics} topics")
    svd = TruncatedSVD(n_components=nb_topics)
    svdMatrix = svd.fit_transform(csc_matrix)

    # --- get vectors ---
    # term_vectors = svd.projection.u
    # term_vectors = pd.DataFrame(term_vectors, index=dictionary.token2id.keys())

    term_vectors = None
    document_vectors = svdMatrix
    # document_vectors = corpus2dense(lsi_corpus, 300, num_docs=len(corpus)).T
    document_vectors = pd.DataFrame(document_vectors)

    return svd, document_vectors, term_vectors


def docs_to_lists(token_series):
    return token_series.tolist()


def calc_entropy(corpus, corpus_freqs):
    print('Calculating word entropy over all contexts.')

    entropy = np.zeros_like(corpus_freqs, dtype=np.float32)
    for context in tqdm(corpus, total=len(corpus)):
        if not context:
            continue
        context_arr = np.asarray(context, dtype=np.int32).T
        term_ids = context_arr[0]
        context_freq = context_arr[1]
        corpus_freq = corpus_freqs[term_ids]
        p_c = context_freq / corpus_freq
        ic = -np.log10(p_c)
        ctx_ent = p_c * ic
        entropy[term_ids] += ctx_ent

    return entropy


def calc_entropy_normalization(corpus, word_entropies, epsilon=0.0):
    print('Normalizing corpus.')

    word_entropies = word_entropies.astype(np.float32)
    epsilon = np.float32(epsilon)
    transformed_corpus = []
    for context in tqdm(corpus, total=len(corpus)):
        if not context:
            transformed_corpus.append([])
        else:
            context_arr = np.asarray(context, dtype=np.float32).T
            term_ids = context_arr[0].astype(np.int32)
            context_freq = context_arr[1]
            context_entropy = word_entropies[term_ids]
            transformations = (np.log(context_freq) + epsilon) / context_entropy
            transformed_context = [x for x in zip(term_ids, transformations)]
            transformed_corpus.append(transformed_context)

    return transformed_corpus


def entropy_transform(corpus, dictionary, directory, epsilon=0.0, use_cache=True):

    # TODO: individualize file name based on args
    file_name = 'entropy_transform.csv'
    file_path = directory / file_name

    df = None
    if use_cache:
        try:
            df = pd.read_csv(file_path, sep='\t')
            print(f'Read from cache {file_path}.')
        except FileNotFoundError:
            print(f'Could not read cache from {file_path}')

    # calculate "entropy" per term
    if df is None:
        dfs = pd.Series(dictionary.dfs, name='context_freq', dtype='int32').sort_index()
        cfs = pd.Series(dictionary.cfs, name='corpus_freq', dtype='int32').sort_index()
        df = pd.concat([dfs, cfs], axis=1, )
        wordcount_per_mil = cfs.sum() / 1_000_000
        df['freq'] = (df.corpus_freq / wordcount_per_mil).astype('float32')
        df['log_freq'] = np.log10(df.freq.values + epsilon, dtype='float32')
        df['entropy'] = calc_entropy(corpus, corpus_freqs=df.corpus_freq.values)
        df['term_id'] = df.index
        df['term'] = df.term_id.map(lambda x: dictionary[x])
        df = df.set_index('term')
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
        contexts, stopwords=None, vocab=None, min_word_freq=1, min_contexts=1, filter_above=1,
        keep_n=None
):
    print(f"Generating dictionary.")

    dictionary = Dictionary(contexts, prune_at=None)

    vocab_size = len(dictionary)
    dictionary.filter_extremes(no_below=min_contexts, no_above=filter_above, keep_n=keep_n)
    print(
        f"Removing {vocab_size - len(dictionary)} terms "
        f"appearing in less than {min_contexts} contexts."
    )
    vocab_size = len(dictionary)

    # apply allowlist by a predefined vocabulary
    if vocab:
        with open(vocab, 'r') as fp:
            print(f'Loading vocab file {vocab}')
            vocab_ = {line.strip() for line in fp.readlines()}
            print(f'{len(vocab_)} terms loaded.')

        good_ids = [
            dictionary.token2id[token] for token in vocab_ if token in dictionary.token2id
        ]
        dictionary.filter_tokens(good_ids=good_ids)
        print(f"Removing {vocab_size - len(dictionary)} tokens not in predefined vocab.")
        vocab_size = len(dictionary)

    # filter noise (e.g. stopwords, special characters, infrequent words)
    if stopwords:
        bad_ids = [dictionary.token2id[token] for token in stopwords]
        dictionary.filter_tokens(bad_ids=bad_ids)
        print(f"Removing {len(dictionary) - vocab_size} stopword tokens.")
        vocab_size = len(dictionary)

    if min_word_freq > 1:
        bad_ids = [k for k, v in dictionary.cfs.items() if v < min_word_freq]
        dictionary.filter_tokens(bad_ids=bad_ids)
        print(
            f"Removing {vocab_size - len(dictionary)} terms with min frequency < {min_word_freq}."
        )

    dictionary.compactify()
    print(f"Dictionary size: {len(dictionary)}")

    print(f"Generating bow corpus from {len(contexts)} contexts.")
    bow_corpus = [dictionary.doc2bow(text) for text in contexts]

    return bow_corpus, dictionary


def get_contexts(args):
    read_fn = reader(args.corpus)
    contexts = read_fn(
        chunk_size=args.window,
        min_doc_size=args.min_doc_size,
        tagged=False,
        lowercase=args.lowercase,
        # tags_allowlist=args.pos_tags,  # TODO: implement
        tags_blocklist=[],  # TODO: add to args
        make_if_not_cached=True,
        persist_if_not_cached=True,
    )
    return contexts


def normalize(bow_corpus, dictionary, normalization, directory, file_name, epsilon=0.0):
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
        entropy_corpus = entropy_transform(bow_corpus, dictionary, directory, epsilon)
        assert len(entropy_corpus) == len(bow_corpus)

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
        contexts, stopwords=None, vocab=args.vocab,
        min_word_freq=args.min_word_freq, min_contexts=args.min_contexts, keep_n=args.keep_n
    )
    assert len(contexts) == len(bow_corpus)

    # - save dictionary -
    file_path = directory / f'{file_name}.dict'
    print(f"Saving {file_path}")
    dictionary.save(str(file_path))

    # - save dictionary frequencies as plain text -
    dict_table = pd.Series(dictionary.token2id).to_frame(name='idx')
    dict_table['freq'] = dict_table['idx'].map(dictionary.cfs.get)
    dict_table = dict_table.reset_index()
    dict_table = dict_table.set_index('idx', drop=True).rename({'index': 'term'}, axis=1)
    dict_table = dict_table.sort_index()
    file_path = directory / f'{file_name}_dict.csv'
    print(f"Saving {file_path}")
    # dictionary.save_as_text(file_path, sort_by_word=False)
    dict_table.to_csv(file_path, sep='\t')

    # - save bow corpus -
    file_path = directory / f'{file_name}_bow.mm'
    print(f"Saving {file_path}")
    MmCorpus.serialize(str(file_path), bow_corpus)
    corpus = normalize(
        bow_corpus, dictionary, args.normalization, directory, file_name,
        epsilon=args.epsilon
    )

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
        corpus = normalize(
            bow_corpus, dictionary, args.normalization, directory, file_name,
            epsilon=args.epsilon
        )

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
    if 'sklearn':  # TODO: parameterize
        model, document_vectors, term_vectors = lsi_projection_sklearn(
            corpus=corpus, nb_topics=args.nb_topics
        )
        assert len(document_vectors) == len(corpus)

        # --- save document vectors ---
        file_path = directory / f'{file_name}_lsi_sklearn_document_vectors.csv'
        print(f"Saving document vectors to {file_path}")
        document_vectors.to_csv(file_path)
    else:
        model, document_vectors, term_vectors = lsi_projection(
            corpus=corpus, dictionary=dictionary, nb_topics=args.nb_topics,
            cache_in_memory=True
        )
        assert len(document_vectors) == len(corpus)
        assert len(term_vectors) == len(dictionary)

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
        print(f"Saving term vectors to {file_path}")
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

    file_name = f'{args.corpus}'
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
        project_suffix = f'_{args.project}' if args.project else ''
        file_path = directory / f'{file_name}{project_suffix}.semd'
    semd_values = calculate_semantic_diversity(terms, dictionary, corpus, lsi_vectors.values)

    # - save SemD values for vocabulary -
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path, sep='\t')

    print(semd_values)


if __name__ == '__main__':
    main()
