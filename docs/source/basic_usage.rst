
Basic usage
============

The core component of Podium is the :class:`podium.Dataset` class, a shallow wrapper which contains instances of a machine learning dataset and the preprocessing pipeline for each data field. 

Podium datasets come in three flavors:

- **Predefined datasets**: Podium contains data load and download functionality for some commonly used datasets in separate classes.
- **Tabular datasets**: Podium allows you to load datasets in standardized format through :class:`podium.TabularDataset` and :class:`podium.arrow.ArrowTabularDataset` classes.
  - Regular tabular datasets are memory-backed, while the arrow version is disk-backed
- **HuggingFace datasets**: Podium wraps the popular `huggingface/datasets <https://github.com/huggingface/datasets>`__ library and allows you to convert every 🤗 dataset to a Podium dataset.

Loading predefined datasets
----------------------------

One predefined dataset available in Podium is the `Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/treebank.html>`__. In order to load the dataset, it is enough to call the :meth:`get_dataset_splits` method.

.. code-block:: python

  >>> from podium.datasets import SST
  >>> sst_train, sst_test, sst_valid = SST.get_dataset_splits()
  >>> print(sst_train)
  SST[Size: 6920, Fields: ['text', 'label']]
  >>> print(sst_train[222]) # A short example
  Example[label: (None, 'positive'); text: (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])]

Each predefined Podium dataset has a :meth:`get_dataset_splits` method, which returns the `train`, `test` and `validation` split of that dataset, if available.

Iterating over datasets
------------------------

Podium contains methods to iterate over data. Let's take a look at :class:`podium.Iterator`, the simplest data iterator. The default batch size of the iterator is `32` but we will reduce it for the sake of space.

.. code-block:: python

  >>> from podium import Iterator
  >>> train_iter = Iterator(sst_train, batch_size=2)
  >>> batch_x, batch_y = next(iter(train_iter))
  >>> print(f"{batch_x}\n{batch_y}")
  InputBatch(text=array([[ 1390,   193,  3035,    12,     4,   652, 13874,   310,    11,
          101, 13875,    12,    31,    14,   729,  1733,     5,     9,
          144,  7287,     8,  3656,   193,  7357,   700,     2,     1,
            1,     1,     1],
       [   29,  1659,   827,     8,    27,     7,  6115,     3,  4635,
           63,     3,    19,     4,    55, 15634,   231,   170,     9,
          128,    48,   123,   656,   130,   190,  2047,     8,   803,
           74,    79,     2]])) 
  TargetBatch(label=array([[1],
       [1]]))


There are a couple of things we need to unpack here. Firstly, our textual input data and class labels were converted to indices. This happened without our intervention -- predefined datasets have a default preprocessing pipeline, which handles text tokenization and numericalization.
Secondly, while iterating we obtained two `namedtuple` instances: an :class:`InputBatch` and a :class:`TargetBatch`. By default, Podium Iterators group input and target data Fields during iteration. If your dataset contains multiple input or target fields, they will also be present as attributes of the namedtuples.

Let us now take a closer look how Podium does the text to index conversion.

The Vocabulary
---------------

We saw earlier that our dataset has two Fields: text and label. We will touch on what exactly Fields are later, but for now let's retrieve and print them out.

.. code-block:: python
  :emphasize-lines: 3

  >>> text_field, label_field = sst_train.fields
  >>> print(f"{text_field}\n{label_field}")
  Field[name: text, is_target: False, vocab: Vocab[finalized: True, size: 16284]]
  LabelField[name: label, is_target: True, vocab: Vocab[finalized: True, size: 2]]

Inside each of these two fields we can see a :class:`podium.storage.Vocab` class, which is used for numericalization (converting tokens to indices). A Vocab is mainly defined by two maps: the string-to-index mapping :attr:`podium.storage.Vocab.stoi` and the index-to-string mapping :attr:`podium.storage.Vocab.itos`.

In the previous code block we can see that the Vocab for the ``text`` field has a size of 16282. The Vocab by default includes all the tokens present in the dataset, whichever their frequency might be. There are two ways to control the size of your vocabulary:

1. Setting the minimum frequency (inclusive) for a token to be used in a Vocab: the :attr:`podium.storage.Vocab.min_freq` argument
2. Setting the maximum size of the Vocab: the :attr:`podium.storage.Vocab.max_size` argument

You might want to limit the size of your Vocab for larger datasets. To do so, define your own vocabulary as follows:

.. code-block:: python
  
  >>> from podium import Vocab
  >>> small_vocabulary = Vocab(max_size=5000, min_freq=3)

In order to use this new Vocab with a dataset, we need to get familiar with Fields.


Customizing the preprocessing pipeline with Fields
--------------------------------------------------

Data processing in Podium is wholly encapsulated in the flexible :class:`podium.storage.Field` class. Default Fields are already defined in the `get_dataset_splits` method of SST, but you can easily redefine and customize them.

LabelField
^^^^^^^^^^^^^^^^^^^^^^^^^^^


Loading your custom dataset
----------------------------




