
.. _predefined-hooks:

Hooks
======

Podium contains a number of predefined hook classes which you can instantiate and use in your Fields. Most of these hooks (if they have the ``as_pretokenization`` constructor parameter) are customizable and can work both as pretokenization hooks as well as posttokenization hooks.

.. note::
   If you apply a hook as posttokenization, it will be called for each element in the tokenized sequence!

   Hooks should be cast to posttokenization **only** if their application would otherwise influence the tokenization process. Setting a hook to posttokenization is expected to take longer than the same hook being used during pretokenization.


Moses Normalizer
-----------------

A hook that wraps ``MosesPunctNormalizer`` from `sacremoses <https://github.com/alvations/sacremoses>`__. Accepts the language for the punctuation normalizer to be applied on. Normalizes whitespace, unicode punctuations, numbers and special control characters.

.. doctest:: moses
   :options: +NORMALIZE_WHITESPACE

   >>> from podium.preproc import MosesNormalizer
   >>> moses = MosesNormalizer(language="en")
   >>> text = "A                 _very_     spaced   sentence"
   >>> print(moses(text))
   A _very_ spaced sentence

By default, MosesNormalizer is a pretokenization hook, which means it expects a single string as an argument. We can cast it to a post-tokenization hook by setting ``as_pretokenization=False`` in the constructor. As a result, the hook now expectes two arguments.

.. doctest:: moses
   :options: +NORMALIZE_WHITESPACE

   >>> moses = MosesNormalizer(language="en", as_pretokenization=False)
   >>> raw_text = None
   >>> tokenized_text = ["A        ","         _very_","     spaced  "," sentence"]
   >>> print(moses(raw_text, tokenized_text))
   (None, ['A ', ' _very_', ' spaced ', ' sentence'])


Regex Replace
--------------

A **pretokenization** hook that applies regex replacements. As an example, we can replace all non-alphanumeric characters from SST instances. First, we will setup loading of the SST dataset, which we will use throughout the following examples. For reference, we will now print out the instance we will apply the transformation on:

.. doctest:: regex
   :options: +NORMALIZE_WHITESPACE

   >>> from podium import Field, LabelField, Vocab
   >>> from podium.datasets import SST
   >>> 
   >>> text = Field('text', numericalizer=Vocab())
   >>> label = LabelField('label')
   >>> fields={'text':text, 'label':label}
   >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
   >>> print(sst_train[222])
   Example({'text': (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.']), 'label': (None, 'positive')})

Now, we need to define our replacements, each a ``(Pattern, str)`` tuple where the pattern matched is replaced with the string.

.. doctest:: regex
   :options: +NORMALIZE_WHITESPACE

   >>> from podium.preproc import RegexReplace
   >>> non_alpha = r"[^a-zA-Z\d\s]"
   >>> replacements = RegexReplace([
   ...     (non_alpha, '')
   ... ])
   >>> text = Field('text', numericalizer=Vocab(),
   ...              pretokenize_hooks=[replacements],
   ...              keep_raw=True)
   >>> fields={'text':text, 'label':label}
   >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
   >>> print(sst_train[222])
   Example({'text': ('A slick  engrossing melodrama ', ['A', 'slick', 'engrossing', 'melodrama']), 'label': (None, 'positive')})

As we can see, the non-alphanumeric characters have been removed from the sequence. Similarly, you can pass a list of regex replacements which will then be executed in the order given. Please do take note that regular expressions are not known for their speed and if you can perform a replacement without using one, it might be beneficial.

Text Cleanup
------------


.. testcleanup::

  import shutil
  shutil.rmtree('sst')
