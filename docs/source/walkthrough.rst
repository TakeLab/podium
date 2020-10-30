
Walkthrough
============

The core component of Podium is the :class:`podium.Dataset` class, a shallow wrapper which contains instances of a machine learning dataset and the preprocessing pipeline for each data field. 

Podium datasets come in three flavors:

- **Built-in datasets**: Podium contains data load and download functionality for some commonly used datasets in separate classes. See how to load built-in datasets here: :ref:`builtin-loading`.
- **Tabular datasets**: Podium allows you to load datasets in standardized format through :class:`podium.TabularDataset` and :class:`podium.arrow.ArrowTabularDataset` classes. See how to load tabular datasets here: :ref:`custom-loading`.

  - Regular tabular datasets are memory-backed, while the arrow version is disk-backed.
- **HuggingFace datasets**: Podium wraps the popular `huggingface/datasets <https://github.com/huggingface/datasets>`__ library and allows you to convert every 🤗 dataset to a Podium dataset. See how to load 🤗 datasets here: :ref:`hf-loading`.

.. _builtin-loading:

Loading built-in datasets
----------------------------

One built-in dataset available in Podium is the `Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/treebank.html>`__. In order to load the dataset, it is enough to call the :meth:`get_dataset_splits` method.

.. code-block:: python

  >>> from podium.datasets import SST
  >>> sst_train, sst_test, sst_valid = SST.get_dataset_splits()
  >>> print(sst_train)
  SST[Size: 6920, Fields: ['text', 'label']]
  >>> print(sst_train[222]) # A short example
  Example[label: (None, 'positive'); text: (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])]

Each built-in Podium dataset has a :meth:`get_dataset_splits` method, which returns the `train`, `test` and `validation` split of that dataset, if available.

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


There are a couple of things we need to unpack here. Firstly, our textual input data and class labels were converted to indices. This happened without our intervention -- built-in datasets have a default preprocessing pipeline, which handles text tokenization and numericalization.
Secondly, while iterating we obtained two `namedtuple` instances: an :class:`InputBatch` and a :class:`TargetBatch`. By default, Podium Iterators group input and target data Fields during iteration. If your dataset contains multiple input or target fields, they will also be present as attributes of the namedtuples.

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
  >>> small_vocabulary = Vocab(max_size=5000, min_freq=2)

In order to use this new Vocab with a dataset, we first need to get familiar with Fields.


Customizing the preprocessing pipeline with Fields
--------------------------------------------------

Data processing in Podium is wholly encapsulated in the flexible :class:`podium.storage.Field` class. Default Fields for the SST dataset are defined in the :meth:`podium.datasets.impl.SST.get_dataset_splits` method, but you can easily redefine and customize them. We will only scratch the surface of customizing Fields in this section.

Fields have a number of constructor arguments, only some of which we will enumerate here:

  - :obj:`name` (str): The name under which the Field's data will be stored in the dataset's example attributes.
  - :obj:`tokenizer` (str | callable | optional): The tokenizer for sequential data. You can pass a string to use a predefined tokenizer or pass a python callable which performs tokenization (e.g. a function or a class which implements ``__call__``). For predefined tokenizers, you can use ``'split'`` for ``str.split`` tokenizer or ``'spacy-lang'`` for the spacy tokenizer in ``lang`` language. For the spacy english tokenizer, this argument would be ``'spacy-en'``. If the data Field should not be tokenized, this argument should be None. Defaults to ``'split'``.
  - :obj:`numericalizer` (Vocab | callable | optional): The method to convert tokens to indices. Traditionally, this argument should be a Vocab instance but users can define their own numericalization function and pass it as an argument. Custom numericalization can be used when you want to ensure that a certain token has a certain index for consistency with other work. If ``None``, numericalization won't e attempted.
  - :obj:`is_target` (bool): Whether this data Field is a target field (will be used as a label during prediction). This flag serves merely as a convenience, to separate batches into input and target data during iteration.
  - :obj:`fixed_length`: (int, optional): Usually, text batches are padded to the maximum length of an instance in batch (default behavior). However, if you are using a fixed-size model (e.g. CNN without pooling) you can use this argument to force each instance of this Field to be of ``fixed_length``. Longer instances will be right-truncated, shorter instances will be padded.

The SST dataset has two textual data columns (fields): (1) the input text of the movie review and (2) the label. We need to define a ``podium.Field`` for each of these.

.. code-block:: python

  >>> from podium import Field, LabelField
  >>> text = Field(name='text', numericalizer=small_vocabulary)
  >>> label = LabelField(name='label')
  >>> print(f"{text}\n{label}")
  Field[name: text, is_target: False, vocab: Vocab[finalized: False, size: 0]]
  LabelField[name: label, is_target: True, vocab: Vocab[finalized: False, size: 0]]

That's it! We have defined our Fields. In order for them to be initialized, we need to `show` them a dataset. For built-in datasets, this is done behind the scenes in the ``get_dataset_splits`` method. We will elaborate how to do this yourself in :ref:`custom-loading`.

.. code-block:: python

  >>> fields = {'text': text, 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(small_vocabulary)
  Vocab[finalized: True, size: 5000]

Our new Vocab has been limited to the 5000 most frequent words. The remaining words will be replaced by the unknown (``<UNK>``) token, which is one of the default `special` tokens in the Vocab.

You might have noticed that we used a different type of Field: :class:`podium.storage.LabelField` for the label. LabelField is one of the predefined custom Field classes with sensible default constructor arguments for its concrete use-case. We'll take a closer look at LabelFields in the following subsection.


LabelField
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common case in datasets is a data Field which contains a label, represented as a string (e.g. positive/negative, a news document category). For defining such a Field, you would need to set a number of its arguments which would lead to a lot of repetetive code.

For convenience, ``LabelField`` sets the required defaults for you, and all you need to define is its name. LabelFields always have a ``fixed_length`` of 1, are not tokenized and are by default set as the target for batching.


.. _custom-loading:

Loading your custom dataset
----------------------------

We have covered loading built-in datasets. However, it is often the case that you want to work on a dataset that you either constructed or we have not yet implemented the loading function for. If that dataset is in a simple tabular format, you can use :class:`podium.datasets.TabularDataset`.

Let's take an example of a natural language inference (NLI) dataset. In NLI, datasets have two input fields: the `premise` and the `hypothesis` and a single, multi-class label. The first two rows of such a dataset written in comma-separated-values (`csv`) format could like as follows:

.. code-block:: bash

  premise,hypothesis,label
  A man inspects the uniform of a figure in some East Asian country.,The man is sleeping,contradiction

For this dataset, we need to define three Fields. We also might want the fields for `premise` and `hypothesis` to share their Vocab.


.. code-block:: python

  >>> from podium import TabularDataset, Vocab, Field, LabelField
  >>> shared_vocab = Vocab()
  >>> fields = {'premise':   Field('premise', numericalizer=shared_vocab, tokenizer="spacy-en"),
  >>>         'hypothesis':Field('hypothesis', numericalizer=shared_vocab, tokenizer="spacy-en"),
  >>>         'label':     LabelField('label')}
  >>> dataset = TabularDataset('my_dataset.csv', format='csv', fields=fields)
  >>> print(dataset)
  TabularDataset[Size: 1, Fields: ['premise', 'hypothesis', 'label']]
  >>> print(shared_vocab.itos)
  [<SpecialVocabSymbols.UNK: '<unk>'>, <SpecialVocabSymbols.PAD: '<pad>'>, 'man', 'A', 'inspects', 'the', 'uniform', 'of', 'a', 'figure', 'in', 'some', 'East', 'Asian', 'country', '.', 'The', 'is', 'sleeping']


.. _hf-loading:

Loading 🤗 datasets
--------------------

The recently released `huggingface/datasets <https://github.com/huggingface/datasets>`__ library implements a large number of NLP datasets. For your convenience (and not to reimplement data loading for each one of them), we have created a wrapper for 🤗/datasets, which allows you to map all of the 140+ datasets directly to your Podium pipeline.

You can load a dataset in 🤗/datasets and then convert it to a Podium dataset as follows:

.. code-block:: python

  >>> from podium import HuggingFaceDatasetConverter as hfd
  >>> import datasets
  >>> # Loading a huggingface dataset returns an instance of DatasetDict
  >>> # which contains the dataset splits (usually: train, valid, test, 
  >>> # but other splits can also be contained such as in the case of IMDB)
  >>> imdb = datasets.load_dataset('imdb')
  >>> print(imdb.keys())
  dict_keys(['train', 'test', 'unsupervised'])
  >>> # We create an adapter for huggingface dataset schema to podium Fields.
  >>> # These are not yet Podium datasets, but behave as such (you can iterate
  >>> # over them as if they were).
  >>> imdb = hfd.from_dataset_dict(imdb)
  >>> imdb_train, imdb_test, imdb_unsupervised = hfd.from_dataset_dict(imdb).values()
  >>> print(imdb_train.fields)
  {'text': Field[name: text, is_target: False, vocab: Vocab[finalized: False, size: 0]], 'label': LabelField[name: label, is_target: True]}

