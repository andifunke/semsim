# SemSim

**This project is still in early alpha stage.**

...

SemSim is intended to be a Python package for applying and 
evaluating metrics of semantic similarity and semantic relatedness measures
on word-pairs or sets of words such as the representation terms derived from
topic models.

SemSim will examine topological, statistical, vector-based and other similarity metrics
that can be calculated on words. It will apply freely available algorithms and packages
as well as re-implemented and self-developed metrics.

SemSim will test the effectiveness of these measures on
a diverse set of English and German semantic similarity/relatedness
datasets by calculating the correlation of these human annotated gold-standards.

```bash
conda env create -f environment.yml
conda activate semsim
pip install -r requirements.txt
python setup.py develop
```
