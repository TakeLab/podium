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
    'six',
    'spacy',
    'tqdm',
]


TESTS_REQUIRE = [
    'pytest==3.10.1',
    'pytest-cov==2.6.0',
    'pytest-mock==1.10.1',
    'urllib3',
    'yake @ https://github.com/LIAAD/yake/archive/v0.4.2.tar.gz',
]


QUALITY_REQUIRE = [
    'black',
    'flake8',
    'isort',
]


DATASETS_REQUIRE = [
    'conllu',
    'datasets',
    'xlrd',
]


EXTRAS_REQUIRE = {
    # for blcc model
    'keras': ['keras==2.2.4'],
    'torch': ['torch'],
    # for preprocessing
    'yake': ['yake @ git+https://github.com/LIAAD/yake/archive/v0.4.2.tar.gz'],

    # dependencies for all dataset implementations (including the ones in dataload)
    'datasets': DATASETS_REQUIRE,

    'docs': ['sphinx'],
    'dev': TESTS_REQUIRE + QUALITY_REQUIRE,
    'quality': QUALITY_REQUIRE,
    'tests': TESTS_REQUIRE + DATASETS_REQUIRE,
}


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
    python_requires=">=3.6.0",
    zip_safe=False
)

# before release:
# 1. add setup.py args: long description (via __doc__), download_url (point to tag),
#    keywords, classifiers
# 2. add dependency comments (so it's more clear why we use them)
