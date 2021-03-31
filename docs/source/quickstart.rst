
Quickstart
============

To get you started with Podium, we will use a sample from a movie review classification dataset stored in a ``csv`` file named ``sample_dataset.csv``:

.. testsetup:: quickstart

  # Run this cell to create the dataset on disk
  import csv
  dataset_path = 'sample_dataset.csv'
  field_names = ('text', 'label')
  with open(dataset_path, 'w', newline='') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=field_names)
      writer.writeheader()
      writer.writerow({
          'text': 'Absorbing character study .',
          'label': 'positive',
      })
      writer.writerow({
          'text': 'Amazingly lame .',
          'label': 'negative',
      })

.. code-block::

  text, label
  Absorbing character study ., positive
  Amazingly lame ., negative

The header of this dataset defines the names of the input columns (features).

Preprocessing data with Fields
-------------------------------

Data preprocessing in Podium is done in pipelines called Fields. Each dataset column is mapped to one or more :class:`podium.Field` instances, which handle tokenization, numericalization and all additional data transforms. What **you** need to do is define how input data maps to Fields. 

.. doctest:: quickstart

  >>> from podium import Field, LabelField, Vocab, TabularDataset
  >>> # Define Fields for each column
  >>> text = Field(name='input_text', tokenizer="split", numericalizer=Vocab())
  >>> label = LabelField(name='target')
  >>> # Map the column names to Fields
  >>> fields = {'text': text, 'label': label}
  >>>
  >>> dataset = TabularDataset('sample_dataset.csv', fields=fields, format='csv')
  >>> dataset.finalize_fields()
  >>> print(dataset[1])
  Example({
      input_text: (None, ['Amazingly', 'lame', '.']),
      target: (None, 'negative')
  })

In this example, we used the built-in :class:`podium.TabularDataset` loader to load our ``csv`` dataset. The loader reads the dataset and uses the ``fields`` dictionary to determine how input data columns map to Fields. Each dataset instance is stored in a :class:`podium.Example`, with the data for each Field stored under that Field's name.

You might wonder, why not simply use the input column names from the header to store data in Examples. This is because you might want to map a single input to multiple Fields, like so:

.. doctest:: quickstart
  :options: +NORMALIZE_WHITESPACE

  >>> text = Field(name='input_text', tokenizer="split", numericalizer=Vocab())
  >>> char = Field(name='input_chars', tokenizer=list, numericalizer=Vocab())
  >>> label = LabelField(name='target')
  >>> fields = {'text': (text, char), 'label': label}
  >>>
  >>> dataset_with_chars = TabularDataset('sample_dataset.csv', fields=fields, format='csv')
  >>> dataset_with_chars.finalize_fields()
  >>> print(dataset_with_chars[1])
  Example({
    input_text: (None, ['Amazingly', 'lame', '.']),
    input_chars: (None, ['A', 'm', 'a', 'z', 'i', 'n', 'g', 'l', 'y', ' ', 'l', 'a', 'm', 'e', ' ', '.']),
    target: (None, 'negative')
  })

You might wonder what the ``None``\s we've been seeing represent. For each Field, we store raw and processed data as a tuple. The first element of the tuple is reserved for raw data, by default blank to preserve memory. For a detailed overview of the Field constructor arguments and how to use them, check :ref:`fields`.

Adding your own preprocessing with hooks
-----------------------------------------

The main way to customize data preprocessing in Podium is with functions we call *hooks*.
Briefly, hooks are python callables that modify data which passes through Fields. They come in two flavors: pre-tokenization and post-tokenization. Pre-tokenization hooks mdoify only raw data, while post-tokenization hooks modify both raw and tokenized data.

Looking at our dataset, we might want to lowercase the data and remove punctuation. We will make lowercasing a pre-tokenization hook and punctuation removal a post-tokenization hook. Please be aware that tokenizers (e.g. ``spacy``, ``nltk``) are commonly sensitive to word casing so lowercasing might be best done in post-tokenization.

.. doctest:: quickstart

  >>> import string
  >>> class RemovePunct:
  ...     def __init__(self):
  ...         self.punct = set(string.punctuation)
  ...     def __call__(self, raw, tokenized):
  ...         """Remove punctuation from tokenized data"""
  ...         return raw, [tok for tok in tokenized if tok not in self.punct]
  >>>
  >>> def lowercase(raw):
  ...    """Lowercases the input string"""
  ...    return raw.lower()

We can add these hooks to the Field constructor and load the dataset again, appying the new preprocessing:

.. doctest:: quickstart

  >>> text = Field(name='input_text', numericalizer=Vocab(),
  ...              keep_raw=True,
  ...              pretokenize_hooks=[lowercase],
  ...              posttokenize_hooks=[RemovePunct()]
  ...        )
  >>> label = LabelField(name='target')
  >>> fields = {'text': text, 'label': label}
  >>> filtered_dataset = TabularDataset('sample_dataset.csv', fields=fields, format='csv')
  >>> filtered_dataset.finalize_fields()
  >>> print(filtered_dataset[1])
  Example({
      input_text: ('amazingly lame .', ['amazingly', 'lame']),
      target: (None, 'negative')
  })

As we have set ``keep_raw=True`` in our input text Field, we can see the effect the tokenization and post-tokenization had on our raw data.
For a more detailed overview of what hooks are and how to use them, check out :ref:`fields` and :ref:`interact_fields`.

Mapping tokens to indices
--------------------------

Apart from the tokenization, each Field also constructed a :class:`podium.Vocab` instance, which maps tokens to indices.

.. doctest:: quickstart

  >>> text_vocab = dataset.field('input_text').vocab
  >>> print(text_vocab)
  Vocab({specials: ('<UNK>', '<PAD>'), eager: False, is_finalized: True, size: 8})
  >>> print(text_vocab.stoi) # String-to-integer
  {'<UNK>': 0, '<PAD>': 1, '.': 2, 'Absorbing': 3, 'character': 4, 'study': 5, 'Amazingly': 6, 'lame': 7}

When loading data, a Field automatically collects frequencies of tokens and relays them to its Vocab. When signaled, the Vocab constructs a **string-to-integer** (stoi) ``dict`` and **index-to-string** (itos) ``list``. Once ``stoi`` and ``itos`` are constructed the Vocab is finalized, cannot be updated and will raise an error if you attempt to do so.
The vocabularies are finalized **by you** -- you need to call :meth:`Dataset.finalize_fields` which subsequently tells every Field in the dataset to finalize its Vocab, if it has one. Please check :ref:`finalizing_vocab` for a more detailed explanation.

Apart from using our ``Vocab`` class to perform numericalization, you can also pass your own callable which maps tokens to indices. Vocabularies (:ref:`vocab`) contain special tokens, which we designed to be easily customizable (:ref:`specials`).


Retrieving processed data
--------------------------

In case structured preprocessing and data loading is the only thing you need from Podium, you can easily retrieve your data and use it elsewhere. You can obtain a generator for each Field's data through the field name attribute:

.. doctest:: quickstart

  >>> print(list(dataset.input_text))
  [(None, ['Absorbing', 'character', 'study', '.']), (None, ['Amazingly', 'lame', '.'])]

To obtain the entire dataset in dict-based format, you can use :meth:`podium.Dataset.as_dict`, which by default doesn't return raw data:

.. doctest:: quickstart

  >>> from pprint import pprint
  >>> pprint(dataset.as_dict())
  {'input_text': [['Absorbing', 'character', 'study', '.'],
                  ['Amazingly', 'lame', '.']],
   'target': ['positive', 'negative']}

If you are only after the full numericalized dataset, we've got you covered. Use :meth:`podium.Dataset.batch`, which will provide the **entire** dataset as a single numericalized batch.

.. doctest:: quickstart
  :options: +NORMALIZE_WHITESPACE

  >>> batch_x, batch_y = dataset.batch(add_padding=True)
  >>> print(batch_x, batch_y, sep="\n")
  {'input_text': array([[3, 4, 5, 2],
        [6, 7, 2, 1]])}
  {'target': array([[0],
        [1]])}

We can easily validate that the numericalized instances correspond to the input data:

.. doctest:: quickstart

  >>> vocab = dataset.field('input_text').vocab
  >>> for instance in batch_x.input_text:
  ...     print(vocab.reverse_numericalize(instance))
  ['Absorbing', 'character', 'study', '.']
  ['Amazingly', 'lame', '.', '<PAD>']

Since our example dataset is small, we can set ``add_padding=True``, which causes output of each Field to be padded to the same length and packed into a matrix (in this case concretely, a numpy array).

.. note::
  When obtaining larger datasets as a single batch, we recommend leaving ``add_padding=False`` (default) or your entire dataset will be padded to the length of the longest instance, causing memory issues.
  When set to ``False``, the output of each Field will be a list instead of a matrix.


Minibatching data
-----------------------

If you want to use the data to train a machine learning model, this can also be done with Podium.

.. doctest:: quickstart

  >>> from podium import Iterator
  >>> 
  >>> train_iter = Iterator(dataset, batch_size=2)
  >>> for batch_x, batch_y in train_iter:
  ...     print(batch_x, batch_y, sep="\n")
  {'input_text': array([[6, 7, 2, 1],
         [3, 4, 5, 2]])}
  {'target': array([[1],
         [0]])}

Each element yielded by Podium iterators is a ``tuple`` of input data and response variable(s). Response variables can be marked as such by setting ``is_target=True`` in their Field constructor. Both elements of the tuple are instances of our ``Batch`` class, a dict-tuple hybrid which unpacks by value rather than by key (as standard python dictionaries do).

For a comprehensive overview of data prep for models, check :ref:`iterating` and the subsequent documentation chapters. For the recommended way of iterating over NLP data, check :ref:`bucketing`

.. testcleanup:: quickstart

  import os
  try:
    os.remove(dataset_path)
  except OSError:
    pass
