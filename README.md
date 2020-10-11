# TakeLab Podium

Home of the **TakeLab Podium** project. Podium is a framework agnostic Python natural language processing library which standardizes data loading and preprocessing as well as model training and selection, among others.
Our goal is to accelerate users' development of NLP models whichever aspect of the library they decide to use.

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage examples](#usage-examples)
  - [Loading datasets](#loading-datasets)
  - [Define your preprocessing](#define-your-preprocessing)
  - [Use preprocessing from other libraries](#use-preprocessing-from-other-libraries)
- [Contributing](#contributing)
  - [Building and running unit tests](#building-and-running-unit-tests)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For building this project, system needs to have installed the following:
- [```git```](https://git-scm.com/)
- [```python3.6```](https://www.python.org/downloads/release/python-360/) or higher
- [```pip```](https://pypi.org/project/pip/)

We also recommend usage of a virtual environment:
- [```conda```](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#virtual-environments)
- [```virtualenv```](https://virtualenv.pypa.io/en/latest/installation/)

### Installing from source

To install `podium`, in your terminal
1. Clone the repository: `git clone git@github.com:mttk/podium.git && cd podium`
2. Install podium: `pip install .`

### Installing package from pip/wheel
Coming soon!

## Usage examples
For detailed usage examples see [examples](https://github.com/mttk/podium/tree/master/examples)

### Loading datasets

Use some of our pre-defined datasets:

```python
>>> from podium.datasets import SST
>>> sst_train, sst_test, sst_dev = SST.get_dataset_splits()
>>> print(sst_train)
SST[Size: 6920, Fields: ['text', 'label']]
>>> print(sst_train[222]) # A short example
Example[label: ('positive', None); text: (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])]
```

Load your own dataset from a standardized format (`csv`, `tsv` or `jsonl`):

```python
>>> from podium.datasets import TabularDataset
>>> from podium.storage import Vocab, Field, LabelField
>>> fields = {'premise':   Field('premise', vocab=Vocab()),
              'hypothesis':Field('hypothesis', vocab=Vocab()),
              'label':     LabelField('label')}
>>> dataset = TabularDataset('my_dataset.csv', format='csv', fields=fields)
>>> print(dataset)
TabularDataset[Size: 1, Fields: ['premise', 'hypothesis', 'label']]
```

Or define your own `Dataset` subclass (tutorial coming soon)

### Define your preprocessing

We wrap dataset pre-processing in customizable `Field` classes. Each `Field` has an optional `Vocab` instance which automatically handles token-to-index conversion.

```python
>>> from podium.storage import Vocab, Field, LabelField
>>> vocab = Vocab(max_size=5000, min_freq=2)
>>> text = Field(name='text', vocab=vocab)
>>> label = LabelField(name='label')
>>> fields = {'text': text, 'label': label}
>>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
>>> print(vocab)
Vocab[finalized: True, size: 5000]
```

Each `Field` allows the user full flexibility modify the data in multiple stages:
- Prior to tokenization (by using pre-tokenization `hooks`)
- During tokenization (by using your own `tokenizer`)
- Post tokenization (by using post-tokenization `hooks`)

You can also completely disregard our preprocessing and define your own:
- Set your `custom_numericalize`

You could decide to lowercase all the characters and filter out all non-alphanumeric tokens:

```python
>>> def lowercase(raw):
>>>     return raw.lower()
>>> def filter_alnum(raw, tokenized):
>>>     filtered_tokens = [token for token in tokenized if
                           any([char.isalnum() for char in token])]
>>>     return raw, filtered_tokens
>>> text.add_pretokenize_hook(lowercase)
>>> text.add_posttokenize_hook(filter_alnum)
>>> # ...
>>> print(sst_train[222])
Example[label: ('positive', None); text: (None, ['a', 'slick', 'engrossing', 'melodrama'])]
```
**Pre-tokenization** hooks do not see the tokenized data and are applied (and modify) only `raw` data. 
**Post-tokenization** hooks have access to tokenized data, and can be applied to either `raw` or `tokenized` data.


### Use preprocessing from other libraries

A common use-case is to incorporate existing components of pretrained language models, such as BERT. This is extremely simple to incorporate as part of our `Field`s. This snippet requires installation of the `transformers` (`pip install transformers`) library.

```python
>>> from transformers import BertTokenizer
>>> # Load the tokenizer and fetch pad index
>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
>>> pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
>>> # Define a BERT subword Field
>>> bert_field = Field("subword",
                       vocab=None,
                       padding_token=pad_index,
                       tokenizer=tokenizer.tokenize,
                       custom_numericalize=tokenizer.convert_tokens_to_ids)
>>> # ...
>>> print(sst_train[222])
Example[label: ('positive', None); subword: (None, ['a', 'slick', ',', 'eng', '##ross', '##ing', 'mel', '##od', '##rama', '.'])]
```

## Contributing

To learn more about making a contribution to Podium, please see our [Contribution page](CONTRIBUTING.md).

### Code style standards
In this repository we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) as a standard for documentation and Black, Flake8 and isort for code sytle. Code style references are [Black](https://black.readthedocs.io/en/stable/), [Flake8](http://flake8.pycqa.org/en/latest/), [isort](https://pycqa.github.io/isort/) and [PEP8](https://www.python.org/dev/peps/pep-0008/).

Commands to check black, isort and flake8 compliance for written code and tests.
```
black --check --line-length 90 --target-version py36 podium tests examples
isort --check-only podium tests examples
flake8 podium tests examples
```

### Building and running unit tests

If you are installing from source, you will need Python 3.6 or later. We highly recommend installing an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#virtual-environments) environment.

Commands to setup virtual environment and run tests.
```
conda create --name env python=3.6
conda activate env
pip install .[tests]
pytest tests
```

If you intend to develop part of podium you should use following command to install podium.
```
pip install -e .[dev]
```
In other cases it should be enough to run ```pip install .``` for podium to be added to python environment.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](./tags). 

## Authors

* Podium is currently maintained by [Ivan Smoković](https://github.com/ivansmokovic), [Silvije Skudar](https://github.com/sskudar), [Mario Šaško](https://github.com/mariosasko), [Filip Boltužić](https://github.com/FilipBolt) and [Martin Tutek](https://github.com/mttk). A non-exhaustive but growing list of collaborators needs to mention: [Domagoj Pluščec](https://github.com/domi385), [Marin Kačan](https://github.com/mkacan), [Dunja Vesinger](https://github.com/dunja-v), [Mate Mijolović](https://github.com/matemijolovic).
* Project made as part of [TakeLab](http://takelab.fer.hr) at Faculty of Electrical Engineering and Computing, University of Zagreb

See also the list of [contributors](../graphs/contributors) who participated in this project.

## License

This project is licensed under the BSD 3-Clause - see the [LICENSE](LICENSE) file for details.
