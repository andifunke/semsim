import argparse
import csv
import warnings
from pathlib import Path

# TODO: remove when tqdm fully supports pandas >= 0.25
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel, LsiModel, LogEntropyModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from semsim import DATASET_STREAMS
from semsim.constants import SEMD_DIR
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

    parser.add_argument('-c', '--corpus', type=str, required=True)
    parser.add_argument('-v', '--version', type=str, required=False, default=None,
                        help="Specify a corpus version. A corpus version points to a cached "
                             "corpus file containing. The corpus version may contain special"
                             "pre-processing like pos-filtering. Without specifying a corpus"
                             "version the cached corpus file will be inferred from other"
                             "CLI arguments if possible.")
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
    parser.add_argument('--normalization', type=str, required=False, default='log-entropy-norm',
                        choices=['none', 'entropy', 'tfidf', 'log-entropy', 'log-entropy-norm'],
                        help="BOW-frequency normalization method.")
    parser.add_argument('--terms', type=str, required=False,
                        help="File path containing the terms per line to calculate SemD values "
                             "for.")
    parser.add_argument('--vocab', type=str, required=False,
                        help="File path containing terms per line to be included in the "
                             "term-document-matrix. "
                             "Terms will only be in the tdm, if found in the corpus.")
    parser.add_argument('--vocab-exclusive', action='store_true', required=False,
                        help="Remove all terms not included in the external vocab.")
    parser.set_defaults(vocab_exclusive=False)

    parser.add_argument('--document-vectors', type=str, required=False,
                        help="File path to a csv file containing document vectors.")
    parser.add_argument('--corpus-path', type=str, required=False,
                        help="File path containing an MatrixMarket corpus.")
    parser.add_argument('--dictionary-path', type=str, required=False,
                        help="File path containing an dictionary (gensim or csv).")
    parser.add_argument('--tags-blocklist', nargs='*', type=str, required=False, default=[],
                        help='List of part-of-speech tags to remove from corpus.')

    parser.add_argument('--center', action='store_true', required=False)
    parser.add_argument('--no-center', dest='center', action='store_false', required=False)
    parser.set_defaults(center=True)

    parser.add_argument('--lowercase', action='store_true', required=False)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', required=False)
    parser.set_defaults(lowercase=False)

    parser.add_argument('--lemmatized', action='store_true', required=False)
    parser.add_argument('--no-lemmatized', dest='lemmatized', action='store_false', required=False)
    parser.set_defaults(lemmatized=False)

    parser.add_argument('--make-corpus', dest='make_corpus', action='store_true', required=False)
    parser.add_argument('--load-corpus', dest='make_corpus', action='store_false', required=False)
    parser.set_defaults(make_corpus=False)

    parser.add_argument('--make-lsi', dest='make_lsi', action='store_true', required=False)
    parser.add_argument('--load-lsi', dest='make_lsi', action='store_false', required=False)
    parser.set_defaults(make_lsi=False)

    parser.add_argument('--memory-efficient', dest='streamed', action='store_true', required=False,
                        help="Recommended for large corpora and limited memory capacity."
                             "May be slower.")
    parser.set_defaults(streamed=False)

    args = parser.parse_args()

    if args.pos_tags is not None:
        args.pos_tags = set(args.pos_tags)

    if args.make_corpus:
        args.make_lsi = True

    if args.streamed and args.normalization in ['entropy', 'log-entropy', 'log-entropy-norm']:
        print(f"WARNING: {args.normalization} does not fully support streamed corpora. "
              "The normalization may require more memory than expected.")

    try:
        args.input_fn = DATASET_STREAMS[args.corpus]
    except KeyError:
        args.input_fn = DATASET_STREAMS['topiclabeling']

    return args


def calculate_semantic_diversity(terms, dictionary, corpus, document_vectors, min_contexts=2):
    print('Calculate Semantic Diversity.')

    if isinstance(dictionary, Dictionary):
        dictionary = dictionary.token2id

    csc_matrix = corpus2csc(corpus, dtype=np.float32)
    if csc_matrix.shape[0] != len(dictionary):
        print(
            f"TDM vocabulary shape {csc_matrix.shape[0]} "
            f"differs from vocabulary length {len(dictionary)}"
        )
    assert csc_matrix.shape[1] == len(corpus)

    mean_cos = {}
    semd_values = {}
    nb_contexts = {}
    for term in tqdm(terms, total=len(terms)):
        try:
            term_id = dictionary[term]
            term_docs_sparse = csc_matrix.getrow(term_id)
            current_docs = term_docs_sparse.nonzero()[1]
            nb_contexts[term] = len(current_docs)

            # can only calculate similarity if word appears in multiple documents
            if len(current_docs) < max(2, min_contexts):
                raise ValueError

            # if target appears in >2000 documents, subsample 2000 at random
            if len(current_docs) > 2000:
                current_docs = np.random.choice(current_docs, size=2000, replace=False)

            term_doc_vectors = document_vectors[current_docs]
            similarities = cosine_similarity(term_doc_vectors)
            lower_tri = np.tril_indices(similarities.shape[0], k=-1)
            similarities = similarities[lower_tri]
            avg_similarity = np.mean(similarities)
            semd = -np.log10(avg_similarity) if avg_similarity > 0. else np.nan
            mean_cos[term] = avg_similarity
            semd_values[term] = semd
        except (KeyError, ValueError, IndexError):
            mean_cos[term] = np.nan
            semd_values[term] = np.nan

    semd = pd.DataFrame(
        [mean_cos, semd_values, nb_contexts], index=['mean_cos', 'SemD', 'nb_contexts']
    ).T

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
    print('Applying entropy transform')

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
        df['log_freq'] = np.log10(df.freq.to_numpy() + epsilon, dtype='float32')
        df['entropy'] = calc_entropy(corpus, corpus_freqs=df.corpus_freq.to_numpy())
        df['term_id'] = df.index
        df['term'] = df.term_id.map(lambda x: dictionary[x])
        df = df.set_index('term')
        df.to_csv(file_path, sep='\t')

    # calculate transformation
    transformed_corpus = calc_entropy_normalization(
        corpus, word_entropies=df.entropy.to_numpy(), epsilon=epsilon
    )

    return transformed_corpus


def tfidf_transform(bow_corpus):
    print('Applying tfidf transform')

    tfidf_model = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]

    return tfidf_corpus


def log_entropy_transform(bow_corpus, normalize=True):
    print(f"Applying log-entropy{'-norm' if normalize else ''} transform")

    # train model
    try:
        log_entropy_model = LogEntropyModel(bow_corpus, normalize=normalize)
    except ValueError:
        print('Loading corpus into memory')
        bow_corpus = list(tqdm(bow_corpus, unit=' documents'))
        log_entropy_model = LogEntropyModel(bow_corpus, normalize=normalize)

    # apply model
    log_entropy_corpus = log_entropy_model[bow_corpus]

    return log_entropy_corpus


def texts2corpus(args):
    stopwords = []

    print(f"Generating dictionary.")
    contexts = get_contexts(args)
    dictionary = Dictionary(contexts, prune_at=None)
    vocab_size = len(dictionary)

    # load allowlist from a predefined vocabulary
    if args.vocab:
        with open(args.vocab) as fp:
            print(f'Loading vocab file {args.vocab}')
            vocab_terms = sorted({line.strip() for line in fp.readlines()})
            print(f'{len(vocab_terms)} terms loaded.')
    else:
        vocab_terms = []

    if args.vocab_exclusive:
        good_ids = [
            dictionary.token2id[token] for token in vocab_terms
            if token in dictionary.token2id
        ]
        dictionary.filter_tokens(good_ids=good_ids)
        print(f"Removing {vocab_size - len(dictionary)} tokens not in predefined vocab.")
    else:
        dictionary.filter_extremes(
            no_below=args.min_contexts, no_above=1., keep_n=args.keep_n, keep_tokens=vocab_terms
        )
        print(
            f"Removing {vocab_size - len(dictionary)} terms "
            f"appearing in less than {args.min_contexts} contexts."
        )
        vocab_size = len(dictionary)

        # filter noise (e.g. stopwords, special characters, infrequent words)
        if stopwords:
            bad_ids = [
                dictionary.token2id[token] for token in stopwords
                if token not in vocab_terms
            ]
            dictionary.filter_tokens(bad_ids=bad_ids)
            print(f"Removing {len(dictionary) - vocab_size} stopword tokens.")
            vocab_size = len(dictionary)

        if args.min_word_freq > 1:
            bad_ids = [
                k for k, v in dictionary.cfs.items()
                if v < args.min_word_freq and dictionary[k] not in vocab_terms
            ]
            dictionary.filter_tokens(bad_ids=bad_ids)
            print(
                f"Removing {vocab_size - len(dictionary)} terms with min frequency "
                f"< {args.min_word_freq}."
            )

    dictionary.compactify()
    print(f"Dictionary size: {len(dictionary)}")

    try:
        print(f"Generating bow corpus from {len(contexts)} contexts.")
        bow_corpus = [dictionary.doc2bow(text) for text in contexts]
    except TypeError:
        print(f"Generating bow corpus from contexts.")
        contexts = get_contexts(args)
        bow_corpus = map(lambda text: dictionary.doc2bow(text), contexts)

    return bow_corpus, dictionary


def get_contexts(args):
    read_fn = reader(args.corpus)
    contexts = read_fn(
        corpus=args.corpus,
        chunk_size=args.window,
        min_doc_size=args.min_doc_size,
        tagged=False,
        lowercase=args.lowercase,
        lemmatized=args.lemmatized,
        # tags_allowlist=args.pos_tags,  # TODO: implement
        tags_blocklist=args.tags_blocklist,
        make_if_not_cached=True,
        persist_if_not_cached=not args.streamed,
        version=args.version,
        as_stream=args.streamed,
    )
    return contexts


def normalize_weights(bow_corpus, dictionary, normalization, directory, file_name, epsilon=0.0):
    if normalization:

        if normalization == 'tfidf':
            corpus = tfidf_transform(bow_corpus)
            file_path = directory / f'{file_name}_tfidf.mm'

        elif normalization == 'log-entropy':
            corpus = log_entropy_transform(bow_corpus, normalize=False)
            file_path = directory / f'{file_name}_log-entropy.mm'

        elif normalization == 'log-entropy-norm':
            corpus = log_entropy_transform(bow_corpus, normalize=True)
            file_path = directory / f'{file_name}_log-entropy-norm.mm'

        elif normalization == 'entropy':
            corpus = entropy_transform(bow_corpus, dictionary, directory, epsilon)
            file_path = directory / f'{file_name}_entropy.mm'

        else:
            raise ValueError(f'{normalization} unknown.')

        assert len(corpus) == len(bow_corpus)
        print(f"Saving {file_path}")
        MmCorpus.serialize(str(file_path), corpus)

    else:
        corpus = bow_corpus

    return corpus


def make_corpus(args, directory, file_name):
    bow_corpus, dictionary = texts2corpus(args)

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

    if args.streamed:
        bow_corpus = load_bow_corpus(directory, file_name)

    corpus = normalize_weights(
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
        elif args.normalization == 'log-entropy':
            file_path = directory / f'{file_name}_log-entropy.mm'
        elif args.normalization == 'log-entropy-norm':
            file_path = directory / f'{file_name}_log-entropy-norm.mm'

        print(f"Loading corpus from {file_path}")
        corpus = MmCorpus(str(file_path))
    except FileNotFoundError as e:
        print(e)
        bow_corpus = load_bow_corpus(directory, file_name)
        corpus = normalize_weights(
            bow_corpus, dictionary, args.normalization, directory, file_name,
            epsilon=args.epsilon
        )

    return corpus, dictionary


def get_sparse_corpus(args, directory, file_name):
    if args.make_corpus:
        corpus, dictionary = make_corpus(args, directory, file_name)
    else:
        try:
            if args.corpus_path and args.dictionary_path:
                corpus = MmCorpus(args.corpus_path)
                try:
                    dictionary = Dictionary.load(str(args.dictionary_path))
                except Exception as e:
                    print(e)
                    with open(args.dictionary_path) as fp:
                        dictionary = {line.strip(): i for i, line in enumerate(fp.readlines())}
            else:
                corpus, dictionary = load_corpus(args, directory, file_name)
        except FileNotFoundError as e:
            print(e)
            corpus, dictionary = make_corpus(args, directory, file_name)

    return corpus, dictionary


def make_lsi(corpus, dictionary, args, directory, file_name):
    center = '_cent' if args.center else ''

    model, document_vectors, term_vectors = lsi_projection(
        corpus=corpus, dictionary=dictionary, nb_topics=args.nb_topics,
        cache_in_memory=not args.streamed
    )
    assert len(document_vectors) == len(corpus)
    assert len(term_vectors) == len(dictionary)

    # --- save model ---
    file_path = directory / f'{file_name}_lsi_gensim.model'
    print(f"Saving model to {file_path}")
    model.save(str(file_path))

    # --- save term vectors ---
    file_path = directory / f'{file_name}_lsi_word_vectors.vec'
    print(f"Saving term vectors to {file_path}")
    with open(file_path, 'w') as fp:
        fp.write(f'{term_vectors.shape[0]} {term_vectors.shape[1]}\n')
        term_vectors.to_csv(fp, sep=' ', header=False, quoting=csv.QUOTE_NONE)

    V_file_path = directory / f'{file_name}_lsi_gensim{center}_document_vectors.csv'

    if center:
        print('Centering LSI document vectors.')
        dv_mean = document_vectors.mean()
        document_vectors -= dv_mean

    # --- save document vectors ---
    print(f"Saving document vectors to {V_file_path}")
    document_vectors.to_csv(V_file_path)

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
            if args.document_vectors:
                lsi_vectors = pd.read_csv(args.document_vectors, index_col=0)
            else:
                lsi_vectors = load_lsi(directory, file_name)
        except FileNotFoundError as e:
            print(e)
            lsi_vectors = make_lsi(corpus, dictionary, args, directory, file_name)

    return lsi_vectors


def main():
    args = parse_args()
    print(args)

    if args.project:
        print('Project:', args.project)

    file_name = f'{args.corpus}'
    if args.project:
        directory = SEMD_DIR / args.project
    elif args.version:
        directory = SEMD_DIR / args.version
    else:
        directory = SEMD_DIR
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
        if args.project:
            file_path = directory / f'{args.project}.semd'
        else:
            file_path = directory / f'{terms_path.stem}.semd'
    else:
        try:
            terms = dictionary.token2id.keys()
        except AttributeError:
            terms = dictionary.keys()
        project_suffix = f'_{args.project}' if args.project else ''
        file_path = directory / f'{file_name}{project_suffix}.semd'
    semd_values = calculate_semantic_diversity(terms, dictionary, corpus, lsi_vectors.to_numpy())

    # - save SemD values for vocabulary -
    print(f"Saving SemD values to {file_path}")
    semd_values.to_csv(file_path, sep='\t')

    print(semd_values)


if __name__ == '__main__':
    main()
