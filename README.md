# TakeLab Podium

Home of the **TakeLab Podium** project. Podium is a framework agnostic Python natural language processing library which standardizes **data loading** and **preprocessing** as well as **model training** and **selection**, among others.
Our goal is to accelerate users' development of NLP models whichever aspect of the library they decide to use.

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage examples](#usage-examples)
- [Contributing](#contributing)
  - [Building and running unit tests](#building-and-running-unit-tests)
  - [Adding new dependencies](#adding-new-dependencies)
  - [Windows specifics](#windows-specifics)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For building this project system needs to have installed the following:
- [```git```](https://git-scm.com/)
- [```python3.6```](https://www.python.org/downloads/release/python-360/)
- [```pip```](https://pypi.org/project/pip/)
We also recommend usage of a virtual environment:
- [```conda```](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#virtual-environments)
- [```virtualenv```](https://virtualenv.pypa.io/en/latest/installation/)

### Installing from source

To install `podium`, in your terminal
1. Clone the repository: `git clone git@github.com:mttk/takepod.git && cd takepod`
3. Install requirements: `pip install -r requirements.txt`
4. Install podium: `python setup.py install`

### Installing package from pip/wheel
Coming soon!

## Usage examples
For detailed usage examples see examples in [takepod/examples](https://github.com/mttk/takepod/tree/master/takepod/examples)

### Loading datasets

Use some of our pre-defined datasets:

```python
>>> from takepod.datasets import SST
>>> sst_train, sst_test, sst_dev = SST.get_dataset_splits()
>>> print(sst_train[222]) # A short example
Example[label: ('positive', None); text: (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])]
```

Load your own dataset from a standardized format (`csv`, `tsv` or `jsonl`):

```python
>>> from takepod.datasets import TabularDataset
>>> from takepod.storage import Vocab, Field, LabelField
>>> vocab = Vocab()  # Shared vocab
>>> premise = Field('premise', vocab=vocab)
>>> hypothesis = Field('hypothesis', vocab=vocab)
>>> label = LabelField('label')
>>> fields = {"premise" :premise,
              "hypothesis" :hypothesis,
              "label" :label}
>>> dataset = TabularDataset('my_dataset.csv', format='csv',fields=fields)
>>> print(f"{dataset}\n{vocab}")
TabularDataset[Size: 1, Fields: ['premise', 'hypothesis', 'label']]
Vocab[finalized: True, size: 10]
```

Or define your own `Dataset` subclass by using our 

Define your own pre-processing for your dataset trough customizing `Field` and `Vocab` classes:

```python
>>> from takepod.storage import Vocab, Field, LabelField
>>> max_vocab_size = 10000
>>> min_frequency = 5
>>> vocab = Vocab(max_size=max_vocab_size, min_freq=min_frequency)
>>> text = Field(name='text', vocab=vocab, tokenizer='spacy')
>>> label = LabelField(name='label')
>>> fields = {'text': text, 'label':label}
>>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
>>> print(sst_train)
SST[Size: 6920, Fields: ['text', 'label']]
```

## Code style standards
In this repository we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) as a standard for documentation and Flake8 for code sytle. Code style references are [Flake8](http://flake8.pycqa.org/en/latest/) and [PEP8](https://www.python.org/dev/peps/pep-0008/).

Commands to check flake8 compliance for written code and tests.
```
flake8 takepod
flake8 test
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

### Building and running unit tests

You will work in a virtual environment and keep a list of required
dependencies in a ```requirements.txt``` file. The master branch of the 
project **must** be buildable with passing tests **all the time**. 
Code coverage should be kept as high as possible (preferably >95%). 

Commands to setup virtual environment and run tests.
```
virtualenv -p python3.6 env
source env/bin/activate
python setup.py install
py.test --cov-report=term-missing --cov=takepod
```

If you intend to develop part of podium you should use following command to install podium.
```
python setup.py develop
```
In other cases it should be enough to run ```python setup.py``` for podium to be added to python environment.


The project is packaged according to official Python packaging [guidelines](https://packaging.python.org/tutorials/packaging-projects/).

We recommend use of [pytest](https://docs.pytest.org/en/latest/) and [pytest-mock](https://pypi.org/project/pytest-mock/) library for testing when developing new parts of the library.

### Adding new dependencies

Adding a new library to a project should be done via ```pip install <new_framework>```. **Don't forget to add it to requirements.txt** 

The best thing to do is to manually add dependencies to the
```requirements.txt``` file instead of using 
```pip freeze > requirements.txt```. 
See [here](https://medium.com/@tomagee/pip-freeze-requirements-txt-considered-harmful-f0bce66cf895)
why.

### Windows specifics
1. install python 3.6 64 bit with pip
(if needed update pip ``` python3 -m pip install --upgrade pip ```)
2. install and create virtual environment  
```
pip3 install virtualenv
virtualenv -p python3 env
```
3. Activate environment  
```
\path\to\env\Scripts\activate.bat -- using CMD
\path\to\env\Scripts\activate.ps1 -- using PowerShell
```

Note: To create a virtualenv under a path with spaces in it on Windows, you’ll need the win32api library installed.

3. Install requirements and run tests
Install pytorch.  
```
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
```
Install other requirements.  
```
pip install -r requirements.txt
python setup.py install
```

4. Deactivate environment when needed  
```
.\env\Scripts\deactivate.bat
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mttk/takepod/tags). 

## Authors

* Podium is currently maintained by [Ivan Smoković](https://github.com/ivansmokovic), [Silvije Skudar](https://github.com/sskudar), [Filip Boltužić](https://github.com/FilipBolt) and [Martin Tutek](https://github.com/mttk). A non-exhaustive but growing list of collaborators needs to mention: [Domagoj Pluščec](https://github.com/domi385), [Marin Kačan](https://github.com/mkacan), [Mate Mijolović](https://github.com/matemijolovic).
* Project made as part of TakeLab at Faculty of Electrical Engineering and Computing, University of Zagreb
* Laboratory url: http://takelab.fer.hr

See also the list of [contributors](https://github.com/mttk/takepod/graphs/contributors) who participated in this project.

## License

This project is licensed under the (TODO) - see the [LICENSE.md](LICENSE.md) file for details

## Project package TODOs

- The project still needs a to have a license picked. 
- OSX instructions on installation and building
- Add used references
- Add deployment notes
- Add small examples
- For automatic flake8 compliance we currently don't recommend use of [Black](https://github.com/ambv/black).

