
.. _predefined-hooks:

Hooks
======

Podium contains a number of predefined hook classes which you can instantiate and use in your Fields. Most of these hooks (if they have the ``as_pretokenization`` constructor parameter) are customizable and can work both as pretokenization hooks as well as posttokenization hooks.

.. note::
   If you apply a hook as posttokenization, it will be called for each element in the tokenized sequence!

   Hooks should be cast to posttokenization **only** if their application would otherwise influence the tokenization process. Setting a hook to posttokenization is expected to take longer than the same hook being used during pretokenization.


Moses Normalizer
-----------------

:class:`podium.preproc.MosesNormalizer` is a hook that wraps ``MosesPunctNormalizer`` from `sacremoses <https://github.com/alvations/sacremoses>`__. Accepts the language for the punctuation normalizer to be applied on. Normalizes whitespace, unicode punctuations, numbers and special control characters.

.. code-block:: python

   >>> from podium.preproc import MosesNormalizer
   >>> moses = MosesNormalizer(language="en")
   >>> text = "A                 _very_     spaced   sentence"
   >>> print(moses(text))
   A _very_ spaced sentence

By default, MosesNormalizer is a pretokenization hook, which means it expects a single string as an argument. We can cast it to a post-tokenization hook by setting ``as_pretokenization=False`` in the constructor. As a result, the hook now expectes two arguments.

.. code-block:: python

   >>> moses = MosesNormalizer(language="en", as_pretokenization=False)
   >>> raw_text = None
   >>> tokenized_text = ["A        ","         _very_","     spaced  "," sentence"]
   >>> print(moses(raw_text, tokenized_text))
   (None, ['A ', ' _very_', ' spaced ', ' sentence'])


Regex Replace
--------------

:class:`podium.preproc.RegexReplace` is a hook that applies regex replacements. As an example, we can replace all non-alphanumeric characters from SST instances. First, we will setup loading of the SST dataset, which we will use throughout the following examples. For reference, we will now print out the instance we will apply the transformation on:

.. code-block:: python

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

.. code-block:: python

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
-------------

:class:`podium.preproc.TextCleanUp` is a hook wrapper of a versatile library that can perform a number of text cleaning operations. For full options, we refer the reader to the
`cleantext <https://github.com/jfilter/clean-text>`__ repository . In Podium, :class:`podium.preproc.TextCleanUp` can be used as follows:

.. code-block:: python

   >>> from podium.preproc import TextCleanUp
   >>> cleanup = TextCleanUp(remove_punct=True)
   >>> text = Field('text', numericalizer=Vocab(), pretokenize_hooks=[cleanup], keep_raw=True)
   >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields={'text':text, 'label':label})
   >>> print(sst_train[222])
   Example({'text': ('A slick engrossing melodrama', ['A', 'slick', 'engrossing', 'melodrama']), 'label': (None, 'positive')})


NLTK Stemmer
------------

:class:`podium.preproc.NLTKStemmer` is a **post-tokenization** hook that applies the NLTK stemmer to the tokenized sequence. This hook, for obvious reasons, cannot be used as a pretokenization hook.

.. code-block:: python

   >>> from podium.preproc import NLTKStemmer
   >>> stemmer = NLTKStemmer(language="en", ignore_stopwords=True)
   >>> text = Field('text', numericalizer=Vocab(), posttokenize_hooks=[stemmer])
   >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields={'text':text, 'label':label})
   >>> print(sst_train[222])
   Example({'text': (None, ['a', 'slick', ',', 'engross', 'melodrama', '.']), 'label': (None, 'positive')})

Spacy Lemmatizer
----------------

:class:`podium.preproc.SpacyLemmatizer` is a **post-tokenization** hook that applies the Spacy lemmatizer to the tokenized sequence. This hook, for obvious reasons, cannot be used as a pretokenization hook.

.. code-block:: python

   >>> from podium.preproc import SpacyLemmatizer
   >>> lemmatizer = SpacyLemmatizer(language="en")
   >>> text = Field('text', numericalizer=Vocab(), posttokenize_hooks=[stemmer])
   >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields={'text':text, 'label':label})
   >>> print(sst_train[222])
   Example({'text': (None, ['a', 'slick', ',', 'engross', 'melodrama', '.']), 'label': (None, 'positive')})

Truecase
--------

:meth:`podium.preproc.truecase` is a **pre-tokenization** hook that applies `truecasing <https://github.com/daltonfury42/truecase>`__ the the input strings. The ``oov`` argument controls how the library handles out-of-vocabulary tokens, the options being ``{"title", "lower", "as-is"}``.

.. code-block:: python

   >>> from podium.preproc import truecase
   >>> apply_truecase = truecase(oov='as-is')
   >>> print(apply_truecase('hey, what is the weather in new york?'))
   Hey, what is the weather in New York?

Stopword removal
-----------------

:meth:`podium.preproc.remove_stopwords` is a **post-tokenization** hook that removes stop words from the tokenized sequence. The list of stop words is provided by `SpaCy <https://spacy.io/>`__ and the language is controlled by the ``language`` parameter.

.. warning::
   The spacy stopword list is in lowercase, so it is recommended to lowercase your tokens prior to stopword removal to avoid unexpected behavior.

.. code-block:: python

   >>> remove_stopwords_hook = remove_stopwords('en')
   >>> raw_text = None
   >>> tokenized_text = ['in', 'my', 'opinion', 'an', 'exciting', 'and', 'funny', 'movie']
   >>> print(remove_stopwords_hook(raw_text, tokenized_text))
   (None, [opinion', 'exciting', 'funny', 'movie'])

.. testcleanup::

  import shutil
  shutil.rmtree('sst')

Utilities
=========

Various tools that can be used for preprocessing textual datasets, not necessarily intended to be used as hooks.

The SpaCy sentencizer
----------------------

:class:`podium.preproc.SpacySentencizer` can bse used to split input strings into sentences prior to tokenization.


Yet another keyword extractor
-----------------------------

:class:`podium.preproc.YAKE` can be used to extract keywords from input strings.
