from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'dill',
    'nltk>=3.0',
    'numpy',
    'pandas',
    'paramiko',
    'requests',
    'scikit-learn',
    'scipy',
    'spacy',
    'tqdm',
    'yake @ https://github.com/LIAAD/yake/archive/v0.4.2.tar.gz',
]


TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pytest-mock',
    'urllib3',
]


QUALITY_REQUIRE = [
    'black',
    'flake8',
    'isort',
]


DATASETS_REQUIRE = [
    'conllu',
    'datasets',
    'pyarrow',
    'xlrd',
]


DOCS_REQUIRE = [
    'sphinx',
    'sphinx_rtd_theme',
    'datasets',
    'keras==2.2.4',
    'tensorflow==1.15',
]


EXTRAS_REQUIRE = {
    # for blcc model
    'blcc': ['keras==2.2.4', 'tensorflow==1.15'],
    # for training/evaluation PyTorch models
    'torch': ['torch'],
    # dependencies for all dataset implementations (including the ones in dataload)
    'datasets': DATASETS_REQUIRE,

    'docs': DOCS_REQUIRE,
    'quality': QUALITY_REQUIRE,
    'tests': TESTS_REQUIRE + DATASETS_REQUIRE,
}
EXTRAS_REQUIRE['dev'] = EXTRAS_REQUIRE['tests'] + QUALITY_REQUIRE


setup(
    name='podium',
    version='1.0.1',
    description='TakeLab podium project',
    author='TakeLab',
    author_email='takelab@fer.hr',
    license='MIT',
    packages=find_packages(
        exclude=[
            '*.tests',
            '*.tests.*',
            'tests',
            'tests.*',
            'examples',
            'examples.*',
        ]
    ),
    url='https://github.com/mttk/podium',  # Change to Takelab/podium on release
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={
        'podium': [
            'preproc/stemmer/data/*.txt'
        ]},
    python_requires=">=3.6",
    zip_safe=False
)

# TODO: (before release)
# 1. add setup.py args: long description (via __doc__), download_url (point to tag),
#    keywords, classifiers
# 2. add dependency comments (so it's more clear why we use them)
