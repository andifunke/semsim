#!/usr/bin/env python
# coding: utf-8
import pickle

import pandas as pd

from gensim import models
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec


# --- load ground truth ---

# loading gt EN
from semsim.constants import DATA_DIR, TMP_DIR

sj_file_en = TMP_DIR / 'Psycho-Paper/synonym_judgement/cueing study stimuli for distribution.csv'
sj_en_full = pd.read_csv(sj_file_en)
sj_en = sj_en_full[['Probe', 'Target', 'Foil1', 'Foil2']]
sj_en = sj_en[~sj_en.isna().any(axis=1)]

# loading gt DE
sj_file_de = TMP_DIR / 'Psycho-Paper/synonym_judgement/SJT_stimuli.csv'
sj_de_full = pd.read_csv(sj_file_de)
sj_de = sj_de_full[['probe', 'target', 'foil1', 'foil2']]
sj_de = sj_de[~sj_de.isna().any(axis=1)]


def closest_match(terms, vectors, verbose=False):
    """
    Returns the index of the term closest to the first term in a list of words.
    
    Note that index 0 is taken as the probe and all words with index > 0 are tested.
    """
    
    terms = terms.to_list()
    # print(terms)
    try:
        distances = vectors.distances(terms[0], terms[1:])
        # print(distances)
        min_dist = distances.argmin() + 1
        return min_dist
    except KeyError:
        if verbose:
            for term in terms:
                if term not in vectors:
                    print(f"missing in vectors: '{term}'")
        return -1

    
def synonym_judgement_accuracy(word_vectors, tests, target_idx=1):
    pred = tests.apply(lambda x: closest_match(x, word_vectors), axis=1)
    pred = pred[pred > 0]
    correct = (pred == target_idx).sum()
    acc = correct / len(pred)
    print(f"Accuracy: {round(acc, 3)}")
    print(f"Number of tests omitted due to unknown words: {len(tests) - len(pred)}")


# -- evaluate word vectors on SJT --

def convert_csv_to_w2v_format(csv_file_path, w2v_file_path):
    lsi_wv = pd.read_csv(csv_file_path, index_col=0)

    with open(w2v_file_path, 'w') as fp:
        fp.write(f'{lsi_wv.shape[0]} {lsi_wv.shape[1]}\n')
        lsi_wv.to_csv(fp, sep=' ', header=False)


def example_vectors_en():
    # - pretrained vectors -
    google_w2v = models.KeyedVectors.load_word2vec_format(
        str(DATA_DIR / 'vectors/GoogleNews-vectors-negative300.bin'),
        binary=True
    )
    synonym_judgement_accuracy(google_w2v, sj_en)

    # - bnc lsi vectors -
    file = 'bnc_lsi_gensim_term_vectors.csv'
    dir_path = DATA_DIR / 'out/SemD/bnc_cs1000_minsz50_lc_filtered'
    csv_file_path = dir_path / file
    w2v_file_path = csv_file_path.with_suffix('.w2v')
    convert_csv_to_w2v_format(csv_file_path, w2v_file_path)

    print(f"Loading {w2v_file_path}")
    bnc_lsi = models.KeyedVectors.load_word2vec_format(str(w2v_file_path))
    synonym_judgement_accuracy(bnc_lsi, sj_en)


def example_vectors_de():
    file = DATA_DIR / 'vectors/d2v'
    print(f"Loading {file}")
    d2v = Doc2Vec.load(str(file))
    synonym_judgement_accuracy(d2v.wv, sj_de)

    file = DATA_DIR / 'vectors/w2v'
    print(f"Loading {file}")
    w2v = Word2Vec.load(str(file))
    synonym_judgement_accuracy(w2v.wv, sj_de)

    file = DATA_DIR / 'out/SemD/OP2/OnlineParticipation_lsi_gensim_term_vectors.csv'
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = DATA_DIR / 'data/out/SemD/DEWAC_1000_40k/dewac_lsi_word_vectors.vec'
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = DATA_DIR / 'out/SemD/DEWAC_1000/dewac_lsi_word_vectors.vec'
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = DATA_DIR / 'out/SemD/DEWAC/dewac_lsi_word_vectors.vec'
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = DATA_DIR / 'out/SemD/DEWAC_1000_40k_v2/dewac_lsi_word_vectors.vec'
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)


def evaluate_dewac_d2v_vectors(vec_path):

    for file in sorted(vec_path.iterdir()):
        try:
            d2v = Doc2Vec.load(str(file))
            print(f'{file} loaded')
            synonym_judgement_accuracy(d2v.wv, sj_de)
            print()
        except pickle.UnpicklingError:
            pass


if __name__ == '__main__':
    #example_vectors_de()
    evaluate_dewac_d2v_vectors(vec_path=DATA_DIR / 'out/models/d2v_dewac')
    evaluate_dewac_d2v_vectors(vec_path=DATA_DIR / 'out/models/d2v_dewac_vocab')
    evaluate_dewac_d2v_vectors(vec_path=DATA_DIR / 'out/models/d2v_test_vocab_B')
