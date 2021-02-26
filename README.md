# TakeLab Podium

## What is Podium?

Podium is a framework agnostic Python natural language processing library which standardizes data loading and preprocessing.
Our goal is to accelerate users' development of NLP models whichever aspect of the library they decide to use.

### Contents

- [Installation](#installation)
- [Usage examples](#usage-examples)
- [Contributing](#contributing)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)

## Installation

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

We also recommend usage of a virtual environment:
- [```conda```](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#virtual-environments)
- [```virtualenv```](https://virtualenv.pypa.io/en/latest/installation/)

### Installing from source

Commands to install `podium` from source

```bash
git clone git@github.com:mttk/podium.git && cd podium
pip install .
```

### Installing from wheel

The following release wheels are available for Podium:

- Version **1.0.1**:
```bash
pip install http://takelab.fer.hr/podium/releases/podium-1.0.1-py3-none-any.whl
```

- Version **1.0.0**:
```bash
pip install http://takelab.fer.hr/podium/releases/podium-1.0.0-py3-none-any.whl
```

### Installing from pip

**[Coming soon]** You can also install `podium` using pip

```bash
pip install podium-nlp
```

For more detailed installation instructions, check the [installation page](http://takelab.fer.hr/podium/installation.html) in the documentation.

## Usage examples

For usage examples see the documentation pages [walkthrough](http://takelab.fer.hr/podium/walkthrough.html) and [examples](https://github.com/mttk/podium/tree/master/examples)

### Loading datasets

Use some of our pre-defined datasets:

```python
>>> from podium.datasets import SST
>>> sst_train, sst_dev, sst_test = SST.get_dataset_splits()
>>> print(sst_train)
SST({
    size: 6920,
    fields: [
        Field({
            name: text,
            keep_raw: False,
            is_target: False,
            vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, finalized: True, size: 16284})
        }),
        LabelField({
            name: label,
            keep_raw: False,
            is_target: True,
            vocab: Vocab({specials: (), eager: False, finalized: True, size: 2})
        })
    ]
})
>>> print(sst_train[222]) # A short example
Example({'text': (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.']), 'label': (None, 'positive')})

```

Load datasets from [🤗/datasets](https://github.com/huggingface/datasets):

```python

  >>> from podium.datasets.hf import HFDatasetConverter
  >>> import datasets
  >>> # Load the huggingface dataset
  >>> imdb = datasets.load_dataset('imdb')
  >>> print(imdb.keys())
  dict_keys(['train', 'test', 'unsupervised'])
  >>> # Wrap it so it can be used in Podium (without being loaded in memory!)
  >>> imdb_train, imdb_test, imdb_unsupervised = HFDatasetConverter.from_dataset_dict(imdb).values()
  >>> # We need to trigger Vocab construction
  >>> imdb_train.finalize_fields()
  >>> print(imdb_train)
  HFDatasetConverter({
    size: 25000,
    fields: [
        Field({
            name: text,
            keep_raw: False,
            is_target: False,
            vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, finalized: True, size: 280619})
        }),
        LabelField({
            name: label,
            keep_raw: False,
            is_target: True})
    ]
  })
```

Load your own dataset from a standardized tabular format (e.g. `csv`, `tsv`, `jsonl`):

```python
>>> from podium.datasets import TabularDataset
>>> from podium import Vocab, Field, LabelField
>>> fields = {'premise':   Field('premise', numericalizer=Vocab()),
...           'hypothesis':Field('hypothesis', numericalizer=Vocab()),
...           'label':     LabelField('label')}
>>> dataset = TabularDataset('my_dataset.csv', format='csv', fields=fields)
>>> print(dataset)
TabularDataset({
    size: 1,
    fields: [
        Field({
            name: premise,
            is_target: False, 
            vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, finalized: True, size: 19})
        }),
        Field({
            name: hypothesis,
            is_target: False, 
            vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, finalized: True, size: 19})
        }),
        LabelField({
            name: label,
            is_target: True, 
            vocab: Vocab({specials: (), eager: False, finalized: True, size: 1})
        }),
    ]
})
```

Or define your own `Dataset` subclass (tutorial coming soon)

### Define your preprocessing

We wrap dataset pre-processing in customizable `Field` classes. Each `Field` has an optional `Vocab` instance which automatically handles token-to-index conversion.

```python
>>> from podium import Vocab, Field, LabelField
>>> vocab = Vocab(max_size=5000, min_freq=2)
>>> text = Field(name='text', numericalizer=vocab)
>>> label = LabelField(name='label')
>>> fields = {'text': text, 'label': label}
>>> sst_train, sst_dev, sst_test = SST.get_dataset_splits(fields=fields)
>>> print(vocab)
Vocab({specials: ('<UNK>', '<PAD>'), eager: True, finalized: True, size: 5000})

```

Each `Field` allows the user full flexibility modify the data in multiple stages:
- Prior to tokenization (by using pre-tokenization `hooks`)
- During tokenization (by using your own `tokenizer`)
- Post tokenization (by using post-tokenization `hooks`)

You can also completely disregard our preprocessing and define your own by setting your own `numericalizer`.

You could decide to lowercase all the characters and filter out all non-alphanumeric tokens:

```python
>>> def lowercase(raw):
...     return raw.lower()
>>> def filter_alnum(raw, tokenized):
...     filtered_tokens = [token for token in tokenized if
...                        any([char.isalnum() for char in token])]
...     return raw, filtered_tokens
>>> text.add_pretokenize_hook(lowercase)
>>> text.add_posttokenize_hook(filter_alnum)
>>> fields = {'text': text, 'label': label}
>>> sst_train, sst_dev, sst_test = SST.get_dataset_splits(fields=fields)
>>> print(sst_train[222])
Example({'text': (None, ['a', 'slick', 'engrossing', 'melodrama']), 'label': (None, 'positive')})

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
>>> subword_field = Field(name="subword",
...                       padding_token=pad_index,
...                       tokenizer=tokenizer.tokenize,
...                       numericalizer=tokenizer.convert_tokens_to_ids)
>>> fields = {'text': subword_field, 'label': label}
>>> sst_train, sst_dev, sst_test = SST.get_dataset_splits(fields=fields)
>>> print(sst_train[222])
Example({'subword': (None, ['a', 'slick', ',', 'eng', '##ross', '##ing', 'mel', '##od', '##rama', '.']), 'label': (None, 'positive')})

```

## Contributing

To learn more about making a contribution to Podium, please see our [Contribution page](CONTRIBUTING.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](../../tags). 

## Authors

* Podium is currently maintained by [Ivan Smoković](https://github.com/ivansmokovic), [Silvije Skudar](https://github.com/sskudar), [Mario Šaško](https://github.com/mariosasko), [Filip Boltužić](https://github.com/FilipBolt) and [Martin Tutek](https://github.com/mttk). A non-exhaustive but growing list of collaborators needs to mention: [Domagoj Pluščec](https://github.com/domi385), [Marin Kačan](https://github.com/mkacan), [Dunja Vesinger](https://github.com/dunja-v), [Mate Mijolović](https://github.com/matemijolovic).
* Project made as part of [TakeLab](http://takelab.fer.hr) at Faculty of Electrical Engineering and Computing, University of Zagreb

See also the list of [contributors](../../graphs/contributors) who participated in this project.

## License

This project is licensed under the BSD 3-Clause - see the [LICENSE](LICENSE) file for details.
