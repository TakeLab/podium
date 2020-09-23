import itertools
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'dill',
    'numpy',
    'nltk>=3.0',
    'pandas',
    'paramiko',
    'requests',
    'spacy',
    'scikit-learn',
    'scipy',
    'six',
    'tqdm',
]


TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pytest-mock',
    'urllib3',
    'conllu',
    'xlrd',
    'yake',
]


QUALITY_REQUIRE = [
    'black'
    'flake8',
    'isort',
]


EXTRAS_REQUIRE = {
    'conllu': ['conllu'],
    # for blcc model
    'keras': ['keras==2.2.4'],
    'tensorflow': ['tensorflow=1.15'],
    'tensorflow_gpu': ['tensorflow-gpu=1.15'],
    'torch': ['torch'],
    'xlrd': ['xlrd'],
    'yake': ['https://github.com/LIAAD/yake/archive/v0.4.2.tar.gz'],

    'ner': ['keras==2.2.4', 'tensorflow-gpu==1.15'],

    'tests': TESTS_REQUIRE,
    'dev': TESTS_REQUIRE + QUALITY_REQUIRE,
    'docs': ['sphinx'],
}


EXTRAS_REQUIRE['all'] = list(set(itertools.chain.from_iterable(EXTRAS_REQUIRE.values())))


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
# 2. add dependency descriptions
