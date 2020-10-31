

The Podium data flow
====================

In Podium, data exists in three states: **raw** (as read from the dataset), **processed** (once the tokenizer and additional postprocessing have been applied) and **numericalized** (converted to indices).

The data is processed immediately when the instance is loaded from disk and then stored in the Example class. Each instance of an Example (a shallow wrapper of a python dictionary) contains one instance of the dataset. Both the `raw` and `processed` data are stored as a tuple attribute under the name of the Field in an Example. You can see this in the SST example:


.. code-block:: python

  >>> from podium.datasets import SST
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits()
  >>> print(sst_train[222]) 
  Example[label: (None, 'positive'); text: (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])]

We can unpack the Example class with the bracket notation, as you would a dictionary.

.. code-block:: python

  >>> text_raw, text_processed = sst_train[222]['text']
  >>> label_raw, label_processed = sst_train[222]['label']
  >>> print(text_raw, text_processed)
  >>> print(label_raw, label_processed)
  None ['A', 'slick', ',', 'engrossing', 'melodrama', '.']
  None positive

What are the ``None`` s? This is the `raw` data, which by default isn't stored in Examples to save memory. If you want to keep the raw data as well (which might be required when applying some transformations), you have to set the ``keep_raw=True`` in the corresponding Field.

.. code-block:: python

  >>> from podium import Vocab, Field, LabelField
  >>> text = Field(name='text', numericalizer=Vocab(), keep_raw=True)
  >>> label = LabelField(name='label')
  >>> fields = {'text': text, 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(sst_train[222].text)
  ('A slick , engrossing melodrama .', ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])

We can see that now we also have the pre-tokenized text available to us. In the case of SST this might not be as useful because the tokenizer is simply ``str.split``, in case of non-reversible tokenizers (ones in ``spacy``), you might want to keep the raw instance for reference.


How to interact with Fields
===========================

In the previous section, we could see that text from the SST dataset is both in uppercase as well as lowercase. Apart from that, we might not want to keep punctuation tokens, which we can also see in the processed data. These are two cases for which we have designed pretokenization and posttokenization **hooks**.

As we said earlier, data in Podium exists in three states: raw, processed and numericalized. You can intervene and add a custom transformation between each of these three states. Functions which modify raw data prior to tokenization are called **pretokenization hooks**, while functions which modify processed data prior to numericalization are called **posttokenization hooks**.

Pretokenization hooks have the following signature:

.. code-block:: python

  >>> def pretokenization_hook(raw):
  >>>   raw = do_something(raw)
  >>>   return raw

Each pretokenization hook accepts one argument, the raw data for that instance and returns one output, the modified raw data. The raw data is then updated accordingly in the Example instance. Posttokenization hooks follow a similar signature:

.. code-block:: python

  >>> def posttokenization_hook(raw, processed):
  >>>   processed = do_something(raw, processed)
  >>>   return raw, processed

Each posttokenization hook accepts two arguments, the raw and processed data for that instance and returns two outputs, which are the modified raw and tokenized data. Both of those are then updated in the Example instance for that data Field in each dataset instance.
If we want to define some text processing which requires some external attribute (e.g. storing the list of stop words for removing stop words), our hook can be a class as long as it implements the ``__call__`` method.


.. code-block:: python

  >>> class Pretokenization_hook:
  >>>   def __init__(self, metadata):
  >>>     self.metadata = metadata
  >>>
  >>>   def __call__(self, raw):
  >>>     raw = do_something(raw, metadata)
  >>>     return raw

Let's now define a few concrete hooks and use them in our dataset.

Lowercase as a pretokenization hook
-----------------------------------

We will first implement a pretokenization hook which will lowercase our raw data. Please beware that casing might influence your tokenizer, so it might be wiser to implement this as a posttokenization hook. In our case however, the tokenizer is ``str.split``, so we are safe. This hook is going to be very simple:

.. code-block:: python

  >>> def lowercase(raw: str) -> str:
  >>>   """Lowercases the input string"""
  >>>   return raw.lower()

And we're done! We can now add our hook to the text field either through the :meth:`podium.storage.Field.add_pretokenize_hook` method of the Field or through the ``pretokenize_hooks`` constructor argument. We will first define a posttokenization hook which removes punctuation and then apply them both to our text Field.

Removing punctuation as a posttokenization hook
-----------------------------------------------

We will now similarly define a posttokenization hook to remove punctuation. We will use the punctuation list from python's built-in ``string`` module, which we will store as an attribute of our hook.

.. code-block:: python

  >>> import string
  >>> class RemovePunct:
  >>>   def __init__(self):
  >>>     self.punct = set(string.punctuation)
  >>>
  >>>   def __call__(self, raw, tokenized):
  >>>     """Remove punctuation from tokenized data"""
  >>>     return raw, [tok for tok in tokenized if tok not in self.punct]

Putting it all together
-----------------------

.. code-block:: python

  >>> text = Field(name='text', numericalizer=Vocab(), 
  >>>              keep_raw=True,
  >>>              pretokenize_hooks=[lowercase],
  >>>              posttokenize_hooks=[RemovePunct()]
  >>>        )
  >>> label = LabelField(name='label')
  >>> fields = {'text':text, 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(sst_train[222])
  ('a slick , engrossing melodrama .', ['a', 'slick', 'engrossing', 'melodrama'])

We can see that our hooks worked: the raw data was lowercased prior to tokenization, and the punctuation is not present in the processed data. You can similarly define other hooks and pass them as arguments to your Fields. It is important to take care of the order in which you pass the hooks -- they will be executed in the same order as you passed them to the constructor, so take care that you don't modify some aspect of data crucial for your next hook.

Defining a custom numericalization function
===========================================

It is often the case you want to do something very specific during numericalization which is not covered by our vocabulary. Examples of this could be mapping 


Fields with multiple outputs
============================

Handling datasets with missing data
===================================

Bucketing instances when iterating
==================================


