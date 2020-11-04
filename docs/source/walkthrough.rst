
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
  >>> print(batch_x, batch_y, sep='\n')
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
  >>> print(text_field, label_field, sep='\n')
  Field[name: text, is_target: False, vocab: Vocab[finalized: True, size: 16284]]
  LabelField[name: label, is_target: True, vocab: Vocab[finalized: True, size: 2]]

Inside each of these two fields we can see a :class:`podium.storage.Vocab` class, which is used for numericalization (converting tokens to indices). A Vocab is mainly defined by two maps: the string-to-index mapping :attr:`podium.storage.Vocab.stoi` and the index-to-string mapping :attr:`podium.storage.Vocab.itos`.

In highligted code block we can see that the Vocab for the ``text`` field has a size of 16282. The Vocab by default includes all the tokens present in the dataset, whichever their frequency might be. There are two ways to control the size of your vocabulary:

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

You can think of Fields as the path your data takes from the input to your model. In order for Fields to be able to process data, you need to which input data columns will pass through which Fields.

.. image:: _static/field_visual.png
    :alt: Field visualisation
    :align: center

Looking at the image, your job is to define the color-coding between input data columns and Fields. If the columns in your dataset are named (as they are in the SST dataset), you should define this mapping as a **dictionary** where the keys are the names of the input data columns, while the values are Fields. The name of the Field affects only the attribute where the data for that Field will be stored, and not the input column! This is due to the fact that it more complex datasets, you might want to map a single input column to multiple Fields.

Fields have a number of constructor arguments, only some of which we will enumerate here:

  - :obj:`name` (str): The name under which the Field's data will be stored in the dataset's Examples.
  - :obj:`tokenizer` (str | callable | optional): The tokenizer for sequential data. You can pass a string to use a predefined tokenizer or pass a python callable which performs tokenization (e.g. a function or a class which implements ``__call__``). For predefined tokenizers, you should follow the ``name-args`` argument formatting convention. You can use ``'split'`` for the ``str.split`` tokenizer (has no additional args) or ``'spacy-en'`` for the spacy english tokenizer. If the data Field should not be tokenized, this argument should be None. Defaults to ``'split'``.
  - :obj:`numericalizer` (Vocab | callable | optional): The method to convert tokens to indices. Traditionally, this argument should be a Vocab instance but users can define their own numericalization function and pass it as an argument. Custom numericalization can be used when you want to ensure that a certain token has a certain index for consistency with other work. If ``None``, numericalization won't be attempted.
  - :obj:`is_target` (bool): Whether this data Field is a target field (will be used as a label during prediction). This flag serves merely as a convenience, to separate batches into input and target data during iteration.
  - :obj:`fixed_length`: (int, optional): Usually, text batches are padded to the maximum length of an instance in batch (default behavior). However, if you are using a fixed-size model (e.g. CNN without pooling) you can use this argument to force each instance of this Field to be of ``fixed_length``. Longer instances will be right-truncated, shorter instances will be padded.

The SST dataset has two textual data columns (fields): (1) the input text of the movie review and (2) the label. We need to define a ``podium.Field`` for each of these.

.. code-block:: python

  >>> from podium import Field, LabelField
  >>> text = Field(name='text', numericalizer=small_vocabulary)
  >>> label = LabelField(name='label')
  >>> print(text, label, sep='\n')
  Field[name: text, is_target: False, vocab: Vocab[finalized: False, size: 0]]
  LabelField[name: label, is_target: True, vocab: Vocab[finalized: False, size: 0]]

That's it! We have defined our Fields. In order for them to be initialized, we need to `show` them a dataset. For built-in datasets, this is done behind the scenes in the ``get_dataset_splits`` method. We will elaborate how to do this yourself in :ref:`custom-loading`.

.. code-block:: python

  >>> fields = {'text': text, 'label': label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(small_vocabulary)
  Vocab[finalized: True, size: 5000]

Our new Vocab has been limited to the 5000 most frequent words. The remaining words will be replaced by the unknown (``<UNK>``) token, which is one of the default `special` tokens in the Vocab.

You might have noticed that we used a different type of Field: :class:`podium.storage.LabelField` for the label. LabelField is one of the predefined custom Field classes with sensible default constructor arguments for its concrete use-case. We'll take a closer look at LabelFields in the following subsection.


LabelField
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common case in datasets is a data Field which contains a label, represented as a string (e.g. positive/negative, a news document category). For defining such a Field, you would need to set a number of its arguments which would lead to a lot of repetetive code.

For convenience, ``LabelField`` sets the required defaults for you, and all you need to define is its name. LabelFields always have a ``fixed_length`` of 1, are not tokenized and are by default set as the target for batching.


Loading pretrained word vectors
-------------------------------

With most deep learning models, we want to use pre-trained word embeddings. In Podium, this process is very simple. If your field uses a vocabulary, it has already built an inventory of tokens for your dataset.

A number of predefined vectorizers are available (:class:`podium.storage.vectorizers.GloVe`, :class:`podium.storage.vectorizers.NlplVectorizer`, :class:`podium.storage.vectorizers.TfIdfVectorizer`), as well as a standardized loader :class:`podium.storage.vectorizers.BasicVectorStorage` for loading word2vec-style format of word embeddings from disk.

For example, we will use the `GloVe <https://nlp.stanford.edu/projects/glove/>`__ vectors. The procedure to load these vectors has two steps:

1. Initialize the vector class, which sets all the required paths.
   The vectors are not yet loaded from disk as you usually don't want to load the full file in memory.
2. Obtain vectors for a pre-defined list of words by calling ``load_vocab``.
   The argument can be a ``Vocab`` object (which is itself an `iterable` of strings), or any sequence of strings.

The output of the function call is a numpy matrix of word embeddings which you can then pass to your model to initialize the embedding matrix or to be used otherwise. The word embeddings are in the same order as the tokens in the Vocab.

.. code-block:: python

  >>> from takepod.storage.vectorizers import GloVe
  >>> vocab = fields['text'].vocab
  >>> glove = GloVe()
  >>> embeddings = glove.load_vocab(vocab)
  >>> print(f"For vocabulary of size: {len(vocab)} loaded embedding matrix of shape: {embeddings.shape}")
  >>>
  >>> # We can obtain vectors for a single word (given the word is loaded) like this:
  >>> word = "sport"
  >>> print(f"Vector for {word}: {glove.token_to_vector(word)}")
  For vocabulary of size: 21701 loaded embedding matrix of shape: (21701, 300)
  Vector for sport: [ 0.34566    0.15934    0.48444   -0.13693    0.18737    0.2678
   -0.39159    0.4931    -0.76111   -1.4586     0.41475    0.55837
   ...
   -0.050651  -0.041129   0.15092    0.22084    0.52252   -0.27224  ]

Using TF-IDF or count vectorization
-----------------------------------
In the case you wish to use a standard shallow model, Podium also supports TF-IDF or count vectorization. We'll now briefly demonstrate how to obtain a TF-IDF matrix for your dataset. We will first load the SST dataset with a limited size Vocab in order to not blow up our RAM. 

As we intend to use the whole dataset at once, we will also set ``disable_batch_matrix=True`` in the constructor for the text Field. This option will return our dataset as a list of numericalized instances during batching instead of a numpy matrix. The benefit here is that if returned as a numpy matrix, all of the instances have to be padded, using up a lot of memory.

.. code-block:: python

  >>> from podium.datasets import SST
  >>> from podium import Vocab, Field, LabelField
  >>> vocab = Vocab(max_size=5000)
  >>> text = Field(name='text', numericalizer=vocab, disable_batch_matrix=True)
  >>> label = LabelField(name='label')
  >>> fields = {'text': text, 'label': label}
  >>> sst_train, sst_test, sst_valid = SST.get_dataset_splits(fields=fields)

Since the Tf-Idf vectorizer needs information from the dataset to compute the inverse document frequency, we first need to fit it on the dataset.

.. code-block:: python

  >>> from podium.storage.vectorizers.tfidf import TfIdfVectorizer
  >>> tfidf_vectorizer = TfIdfVectorizer()
  >>> tfidf_vectorizer.fit(dataset=sst_train, field=text)

Now our vectorizer has seen the dataset as well as the vocabulary and has all the required information to compute Tf-Idf value for each instance. As is standard in using shallow models, we want to convert all of the instances in a dataset to a Tf-Idf matrix which can then be used with a support vector machine (SVM) model.

.. code-block:: python

  >>> # Obtain the whole dataset as a batch
  >>> x, y = sst_train.batch()
  >>> tfidf_batch = tfidf_vectorizer.transform(x.text)
  >>>
  >>> print(type(tfidf_batch), tfidf_batch.shape)
  >>> print(tfidf_batch[222])
  <class 'scipy.sparse.csr.csr_matrix'> (6920, 4998)
  (0, 1658) 0.617113703893198
  (0, 654)  0.5208201737884445
  (0, 450)  0.5116152860290002
  (0, 20) 0.2515101839877878
  (0, 1)  0.12681755258500052
  (0, 0)  0.08262419651916046

The Tf-Idf counts are highly sparse since not all words from the vocabulary are present in every instance. To reduce the memory footprint of count-based numericalization, we store the values in a `SciPy <https://www.scipy.org/>`__ sparse matrix, which can be used in various `scikit-learn <https://scikit-learn.org/stable/>`__ models.

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
  >>>           'hypothesis':Field('hypothesis', numericalizer=shared_vocab, tokenizer="spacy-en"),
  >>>           'label':     LabelField('label')}
  >>>
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
  >>>
  >>> # We create an adapter for huggingface dataset schema to podium Fields.
  >>> # These are not yet Podium datasets, but behave as such (you can iterate
  >>> # over them as if they were).
  >>> imdb = hfd.from_dataset_dict(imdb)
  >>> imdb_train, imdb_test, imdb_unsupervised = hfd.from_dataset_dict(imdb).values()
  >>>
  >>> print(imdb_train.fields)
  {'text': Field[name: text, is_target: False, vocab: Vocab[finalized: False, size: 0]], 'label': LabelField[name: label, is_target: True]}

