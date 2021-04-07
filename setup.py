"""
TakeLab Podium is an open source library for natural language processing.

Podium accelerates data loading, preprocessing & batching to enable faster development of NLP models.
See http://takelab.fer.hr/podium/ for complete documentation.
"""
import re
from pathlib import Path

from setuptools import find_packages, setup


def _get_version():
    project_root_init = Path(__file__).parent / "podium" / "__init__.py"
    with open(project_root_init, "r") as f:
        version = re.search(r'__version__ = \"(.*)\"', f.read()).group(1)
    return version

DISTNAME = 'podium-nlp'
VERSION = _get_version()
DOCLINES = __doc__.split('\n')

INSTALL_REQUIRES = [
    # for numericalization in batching
    "numpy;python_version>='3.7'",
    "numpy<=1.19;python_version<'3.7'",
    # for improved dataset pickling
    "dill",
    # for tokenization and data encoded in tree structure
    "nltk>=3.0,<3.6",
    # for improved csv parsing
    "pandas;python_version>='3.7'",
    "pandas<1.2.0;python_version<'3.7'",
    # for downloading datasets over HTTP
    "paramiko",
    "requests",
    # for models and model selection
    "scikit-learn",
    # for sparse storage
    "scipy;python_version>='3.7'",
    "scipy<1.6.0;python_version<'3.7'",
    # progress bar in download and model selection
    "tqdm",
    # for nodes in HierarhicalDataset
    "dataclasses;python_version<'3.7'",
]


TESTS_REQUIRE = [
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "urllib3",
    # for preprocessing (tokenization, hooks, etc.)
    "spacy",
    "spacy-lookups-data"
]


QUALITY_REQUIRE = [
    "black",
    "flake8",
    "isort",
    "docformatter",
]


DATASETS_REQUIRE = [
    # to transform CONLL-U datasets to our dataset type
    "conllu",
    # to support HF Datasets conversion
    "datasets",
    # to support saving/loading datasets from a disk
    "pyarrow>=1.0.0",
    # to tokenize the input in the IMDB dataset
    "spacy",
]


PREPROC_REQUIRE = [
    # for normalization and tokenization
    "sacremoses",
    # for text cleanup (url removal, currency removal, etc.)
    "clean-text",
    # for truecasing
    "truecase",
    # for keyword extraction
    "rake-nltk",
]


DOCS_REQUIRE = [
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx-copybutton',
    'recommonmark',
    'nbformat',
    'datasets',
]


EXTRAS_REQUIRE = {
    # for training and evaluation of PyTorch models
    "torch": ["torch"],
    # dependencies for all dataset implementations (including the ones in dataload)
    "datasets": DATASETS_REQUIRE,

    "docs": DOCS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE + DATASETS_REQUIRE + PREPROC_REQUIRE,
}
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["tests"] + QUALITY_REQUIRE


setup(
    name=DISTNAME,
    version=VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES),
    author="TakeLab",
    author_email="takelab@fer.hr",
    url="https://github.com/TakeLab/podium",
    download_url="https://github.com/TakeLab/podium/tags",
    license="BSD-3",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests",
            "tests.*",
            "examples",
            "examples.*",
        ]
    ),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.6",
    classifiers=[
        # maturity level
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    keywords="podium nlp natural-language-processing machine learning",
)
