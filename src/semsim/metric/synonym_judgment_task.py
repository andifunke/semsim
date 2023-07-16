#!/usr/bin/env python3

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import models
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec

# loading gt EN
from semsim.constants import METRICS_DIR, VECTORS_DIR, SEMD_DIR

# --- load ground truth ---

sj_file_en = (
    METRICS_DIR / "synonym_judgement" / "cueing_study_stimuli_for_distribution.csv"
)
sj_en_full = pd.read_csv(sj_file_en)
sj_en = sj_en_full[["Probe", "Target", "Foil1", "Foil2"]]
sj_en = sj_en[~sj_en.isna().any(axis=1)]

# loading gt DE
sj_file_de = METRICS_DIR / "synonym_judgement" / "SJT_stimuli.csv"
sj_de_full = pd.read_csv(sj_file_de)
sj_de = sj_de_full[["probe", "target", "foil1", "foil2"]]
sj_de = sj_de[~sj_de.isna().any(axis=1)]


def similarities(terms, vectors):
    terms = terms.to_list()
    probe = terms[0]
    targets = terms[1:]

    try:
        distances = vectors.distances(probe, targets)
        sims = 1 - distances
    except KeyError:
        if probe not in vectors:
            sims = [np.nan] * 3

        else:
            sims = []
            for term in targets:
                if term in vectors:
                    similarity = vectors.similarity(probe, term)
                    sims.append(similarity)
                else:
                    sims.append(np.nan)

    return pd.Series(sims)


def synonym_judgement_accuracy(word_vectors, tests, target_idx=0, file=None):
    sim_cols = ["target_sim", "foil1_sim", "foil2_sim"]
    tests = tests.copy()

    # calculating similarities
    tests[sim_cols] = tests.apply(similarities, vectors=word_vectors, axis=1)

    # default values for OOV tests
    tests["pred"] = -1
    tests["correct"] = np.nan

    # predictions for in-vocab test
    in_vocab = ~tests[sim_cols].isna().any(axis=1)
    tests.loc[in_vocab, "pred"] = tests.loc[in_vocab, sim_cols].apply(np.argmax, axis=1)
    pred = tests.loc[in_vocab, "pred"]
    tests.loc[in_vocab, "correct"] = pred == target_idx

    # calculating accuracy
    correct = tests.loc[in_vocab, "correct"].sum()
    acc = correct / len(pred)
    print(f"Accuracy: {acc:.03f}")
    print(f"Number of tests omitted due to unknown words: {len(tests) - len(pred)}")

    # writing results
    if file is not None:
        file = Path(file)
        file = file.parent / f"{file.name}_acc{acc:.03f}.sjt"
        print(f"Saving SJT predictions to {file}")
        tests.to_csv(file, sep="\t")


# -- evaluate word vectors on SJT --


def convert_csv_to_w2v_format(csv_file_path, w2v_file_path):
    lsi_wv = pd.read_csv(csv_file_path, index_col=0)

    with open(w2v_file_path, "w") as fp:
        fp.write(f"{lsi_wv.shape[0]} {lsi_wv.shape[1]}\n")
        lsi_wv.to_csv(fp, sep=" ", header=False)


def example_vectors_en():
    # - pretrained vectors -
    google_w2v = models.KeyedVectors.load_word2vec_format(
        str(VECTORS_DIR / "GoogleNews-vectors-negative300.bin"), binary=True
    )
    synonym_judgement_accuracy(google_w2v, sj_en)

    # - bnc lsi vectors -
    file = "bnc_lsi_gensim_term_vectors.csv"
    dir_path = SEMD_DIR / "bnc_cs1000_minsz50_lc_filtered"
    csv_file_path = dir_path / file
    w2v_file_path = csv_file_path.with_suffix(".w2v")
    convert_csv_to_w2v_format(csv_file_path, w2v_file_path)

    print(f"Loading {w2v_file_path}")
    bnc_lsi = models.KeyedVectors.load_word2vec_format(str(w2v_file_path))
    synonym_judgement_accuracy(bnc_lsi, sj_en)


def example_vectors_de():
    file = VECTORS_DIR / "d2v"
    print(f"Loading {file}")
    d2v = Doc2Vec.load(str(file))
    synonym_judgement_accuracy(d2v.wv, sj_de)

    file = VECTORS_DIR / "w2v"
    print(f"Loading {file}")
    w2v = Word2Vec.load(str(file))
    synonym_judgement_accuracy(w2v.wv, sj_de)

    file = VECTORS_DIR / "OnlineParticipation_lsi_gensim_term_vectors.csv"
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = SEMD_DIR / "DEWAC_1000_40k/dewac_lsi_word_vectors.vec"
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = SEMD_DIR / "DEWAC_1000/dewac_lsi_word_vectors.vec"
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)

    file = SEMD_DIR / "DEWAC/dewac_lsi_word_vectors.vec"
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de)


def evaluate_lsi_vectors(vec_path):
    file = Path(vec_path)
    print(f"Loading {file}")
    op_lsi = models.KeyedVectors.load_word2vec_format(str(file))
    synonym_judgement_accuracy(op_lsi, sj_de, file=file)


def evaluate_d2v_vectors(vec_path):
    for file in sorted(vec_path.iterdir()):
        try:
            d2v = Doc2Vec.load(str(file))
            print(f"{file} loaded")
            synonym_judgement_accuracy(d2v.wv, sj_de, file=file)
            print()
        except pickle.UnpicklingError:
            pass


if __name__ == "__main__":
    # example_vectors_de()
    # evaluate_d2v_vectors(DATA_DIR / 'out/models/d2v_dewac')
    # evaluate_dewac_d2v_vectors(DATA_DIR / 'out/models/d2v_dewac_vocab')
    # evaluate_d2v_vectors(DATA_DIR / 'out/models/d2v_test_vocab_B')
    evaluate_lsi_vectors(SEMD_DIR / "DEWAC_1000_40k_v2/dewac_lsi_word_vectors.vec")
