"""
The recommended way to install the w2v-pos-tagger package in an existing
environment is:

```bash
python setup.py develop
```
"""

import subprocess
from pathlib import Path

import setuptools


# --- constants ---

URL = "https://github.com/andifunke/semsim"
README = "README.md"
PACKAGE = "semsim"
PACKAGE_DIR = Path('./src') / PACKAGE


# --- functions ---

def install_spacy_model():
    try:
        import de_core_news_md
    except ImportError:
        subprocess.run(['python', '-m', 'spacy', 'download', 'de'])
    try:
        import en_core_web_sm
    except ImportError:
        subprocess.run(['python', '-m', 'spacy', 'download', 'en'])


def install_nltk():
    try:
        import nltk
        print("installing nltk punkt tokenizer")
        nltk.download('punkt')
    except ImportError:
        print('Could not import NLTK. Please install the package via `pip install nltk`.')


def read_version():
    print('inferring version')
    try:
        with open(PACKAGE_DIR.resolve() / '__init__.py') as fp:
            for line in fp.readlines():
                if line.startswith('__version__'):
                    version = line.lstrip("__version__ = '").rstrip("'\n")
                    print('version:', version)
                    return version
    except FileNotFoundError as e:
        print('info:', e)
        return None


def read_readme():
    try:
        with open(README, 'r') as fp:
            print('reading', README)
            readme = fp.read()
    except OSError:
        print("README.md not found.")
        readme = ""
    return readme


# --- main ---

setuptools.setup(
    name=PACKAGE,
    version=read_version(),
    author="Andreas Funke",
    author_email="andreas.funke@uni-duesseldorf.de",
    description="The SemSim python package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={'Source': URL},
    python_requires=">=3.7",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
    platforms=['any'],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={PACKAGE: ['*.cfg']},
    exclude_package_data={'': ['setup.cfg']},
    entry_points={
        'console_scripts': [
        ],
    },
)

# --- post-install ---

install_spacy_model()
install_nltk()
