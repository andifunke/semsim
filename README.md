# SemSim

## Introduction

SemSim is a project for calculating semantic similarity of word pairs. In particular it carefully implements the semantic diversity metric (SemD) from [Hoffman et al. 2013](https://link.springer.com/article/10.3758/s13428-012-0278-x) in Python. Additionally it provides word and document vector models for German and English, optimized for semantic diversity which also perform very well on the synonym judgement task. The models have been trained on the [British National Corpus (BNC)](http://www.natcorp.ox.ac.uk/) (English) as well as on the [DeWaC](https://wacky.sslmit.unibo.it/) corpus (German).

## Installation

```bash
conda env create -f environment.yml
conda activate semsim
pip install -r requirements.txt
pip install -e .
```

## Data and Models

For downloading the BNC and DeWaC corpora visit:

- [BNC](http://www.natcorp.ox.ac.uk/)
- [DeWaC](https://wacky.sslmit.unibo.it/)

In additional download the content from this [Google Drive folder](https://drive.google.com/drive/folders/10X8mn6J6REUH-wdxkNZ6BxpW7U9MpzN0) and extract it do `semsim/data`.
