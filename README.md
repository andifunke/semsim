# SemSim

## Introduction

SemSim is a project for calculating semantic similarity of word pairs. In particular, it carefully 
implements the *Semantic Diversity* metric (*SemD*) from 
[Hoffman et al. 2013](https://link.springer.com/article/10.3758/s13428-012-0278-x) in Python. 
Additionally, it provides word and document vector models for German and English, optimized for 
semantic diversity which also perform very well on the *Synonym Judgement Task*. The models have 
been trained on the **[British National Corpus (BNC)](http://www.natcorp.ox.ac.uk/)** (English) as 
well as on the **[DeWaC](https://wacky.sslmit.unibo.it/)** corpus (German).

## Installation

```bash
conda env create -f environment.yml
conda activate semsim
pip install -r requirements.txt
pip install -e .
```

## Corpora

For downloading the **BNC** and **DeWaC** corpora visit:

- [BNC](http://www.natcorp.ox.ac.uk/)
- [DeWaC](https://wacky.sslmit.unibo.it/)

## Models

In addition, download models and evaluation content from this 
[Google Drive folder](https://drive.google.com/drive/folders/10X8mn6J6REUH-wdxkNZ6BxpW7U9MpzN0) and 
extract it do `semsim/data`.

The models referenced in [Bechtold et al. 2023](https://psyarxiv.com/grwa3/) are located in

- `semsin/data/SemD/DEWAC_1000_40k_v2` (LSI based word and document vectors + SemD values)
- `semsin/data/SemD/DEWAC_1000_40k_d2v` (doc2vec based word and document vectors + SemD values)

## Corpus preparation

This section describes how to convert the **BNC** and **DeWaC** corpora into a common format for 
further processing. If you are only interested in the vector models or in calculating *SemD* or the 
*Synonym Judgement Task* with pre-trained models, you can skip this section.

### BNC

Download the full **BNC** (XML edition) from the Oxford Text Archive via: 
**[British National Corpus (BNC)](http://www.natcorp.ox.ac.uk/)**.
The download format is a `.zip` file which can be extracted anywhere, preferably into 
`semsim/data/corpora/BNC/<corpus-version>`, where `<corpus-version>` is usually the stem of the 
`.zip` file. The path is now referenced as `$BNC_DIR` and should contain a `download` directory.

Run the extraction script via:

```bash
python -m semsim.corpus.bnc -i $BNC_DIR
```

For our reference model we were using

```bash
python -m semsim.corpus.bnc -i $BNC_DIR \
   --window 1000 \
   --min-doc-size 50 \
   --lowercase \
   --tags-blocklist PUN PUL PUR UNC PUQ
```

Filtering for uninformative POS tags made no significant difference with respect to model 
performance, but helps to improve the efficiency of the downstream pipeline.

For additional options run 

```bash
python -m semsim.corpus.bnc --help
```

### DeWaC

coming soon

## Corpus to Vector Model

In this step we extract the term-document-matrix from the pre-processed corpus and apply a 
normalization such as tf-idf or log-entropy-norm to the matrix. Hereafter we convert the sparse
matrix into two dense matrices representing word and document vectors using latent semantic 
indexing (LSI). We can use these latent representations later to calculate SemD for a given set of 
words.

Run for details:

```bash
python -m semsim.metric.semantic_diversity --help
```

## Calculating Metrics

### SemD

coming soon

### Synonym Judgement Task

coming soon
