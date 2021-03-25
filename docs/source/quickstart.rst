
.. testsetup:: *

  from podium import Field, LabelField, Vocab, Iterator, TabularDataset
  from podium.datasets import SST
  from podium.vectorizers import GloVe, TfIdfVectorizer


Quickstart
============

Throughout the quickstart, we will use a sample from a movie review classification dataset stored in a comma-separated value (``csv``) file named ``sample_dataset.csv``:

.. code-block:: python

  text, label
  Absorbing character study, positive
  Amazingly lame, negative

This dataset has a header line defining the names of the input columns (features).

Mapping data to Fields
-----------------------

Data preprocessing in Podium is done in pipelines we call Fields. Columns of your dataset are mapped to one or more :class:`podium.Field` instances, which then handle tokenization and any additional data transform. What **you** need to do is define how the input data maps to Fields. 

.. code-block:: python

  >>> from podium import Field, LabelField, Vocab, TabularDataset
  >>> # Define Fields for each column
  >>> text = Field(name='input_text', tokenizer="split", numericalizer=Vocab())
  >>> label = LabelField(name='target')
  >>> # Map the column names to Fields
  >>> fields = {'text':text, 'label':label}
  >>>
  >>> dataset = TabularDataset('sample_dataset.csv', fields=fields, format='csv')
  >>> print(dataset[1])
  Example({'input_text': (None, ['Amazingly', 'lame']),
           'target': (None, 'negative')})

We used the built-in :class:`podium.TabularDataset` loader and defined that the dataset is in ``csv`` format. The loader reads the dataset and uses the ``fields`` dictionary to determine the preprocessing for each data column and under which attribute name to store it. Each dataset instance is stored in a :class:`podium.Example`, a ``dict`` with some convenience functions.

You might wonder, why not use the column names from the header to store data. This is because you might want to map a single input to multiple Fields, like so:

.. code-block:: python

  >>> # ...
  >>> char = Field(name='input_chars', tokenizer=list, numericalizer=Vocab())
  >>> fields = {'text':(text, char), 'label':label}
  >>>
  >>> dataset_with_chars = TabularDataset('sample_dataset.csv', fields=fields, format='csv')
  >>> print(dataset_with_chars[1])
  Example({'input_text': (None, ['Amazingly', 'lame']),
           'input_chars': (None, ['A', 'm', 'a', 'z', 'i', 'n', 'g', 'l', 'y', ' ', 'l', 'a', 'm', 'e']),
           'target': (None, 'negative')})

The ``tokenizer`` can be a keyword for some of our predefined tokenizers or any *callable* that for an input ``str`` outputs an iterable of elements. By default, ``str.split`` is used for tokenization.

Mapping tokens to indices
--------------------------

Something else happened when we loaded the dataset -- apart from the tokenization, each Field also constructed a :class:`podium.Vocab` instance which maps tokens to indices.

.. code-block:: python

  >>> text_vocab = dataset.field('input_text').vocab
  >>> print(text_vocab)
  Vocab({specials: ('<UNK>', '<PAD>'), eager: True, is_finalized: True, size: 7})
  >>> print(text_vocab.stoi) # String-to-integer
  {'<UNK>': 0, '<PAD>': 1, 'Absorbing': 2, 'character': 3, 'study': 4, 'Amazingly': 5, 'lame': 6}

A Field's ``Vocab`` instance automatically collects frequencies of tokens upon data loading and, when signaled, constructs **string-to-integer** (stoi) and **index-to-string** (itos) ``dict`` and ``list``, respectively. Once ``stoi`` and ``itos`` are constructed the ``Vocab`` is finalized, cannot be updated and will raise warnings if you attempt to do so.

Apart from using the ``Vocab`` class to perform numericalization, you can pass any callable which maps a token to an index or use a static ``Vocab`` constructor (:meth:`podium.Vocab.from_stoi`, :meth:`podium.Vocab.from_itos`) to define your desired mapping.


Retrieving processed data
--------------------------

In case structured preprocessing and data loading is the only thing you need from Podium, you can easily retrieve your data and use it elsewhere. If you want to retrieve the tokens, you can obtain a data generator for each Field in a dataset through the field name attribute:

.. code-block:: python

  >>> print(list(dataset.input_text))
  [(None, ['Absorbing', 'character', 'study']), (None, ['Amazingly', 'lame'])]

You might wonder what the ``None``\s we've been seeing represent. For each Field, we store raw and processed data as a tuple. The first element of the tuple is reserved for raw data, by default blank to preserve memory. Setting the ``store_raw=True`` in the ``Field`` constructor will cause raw data to be stored (and can be used for debugging purposes).

To obtain the entire dataset in dict-based format, you can use :meth:`podium.Dataset.as_dict`, which by default doesn't return raw data:

.. code-block:: python

  >>> from pprint import pprint
  >>> pprint(dataset.as_dict())
  {'input_text': [['Absorbing', 'character', 'study'], ['Amazingly', 'lame']],
   'target': ['positive', 'negative']}

If you are after numericalized data, we've got you covered. Use :meth:`podium.Dataset.batch`, which will provide the entire dataset in batched format.

.. code-block:: python

  >>> batch_x, batch_y = dataset.batch(add_padding=True)
  >>> print(batch_x, batch_y, sep="\n")
  {'input_text': array([[2, 3, 4],
         [5, 6, 1]])}
  {'target': array([[0],
         [1]])}

Since our dataset is small, we can set ``add_padding=True``, in which case the output of each Field is padded to the same length and packed into a matrix (in this case concretely, a numpy array). 

.. note::
  When obtaining larger datasets as a single batch, we recommend leaving ``add_padding=False`` (default) or your entire dataset will be padded to the length of the longest instance, causing memory issues.


Batching data
-----------------------

We have seen that 



.. testcleanup::

  import shutil
  shutil.rmtree('sst')
