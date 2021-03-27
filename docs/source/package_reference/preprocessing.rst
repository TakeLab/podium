Built-in preprocessing tools
============================

Hooks
------

Moses Normalizer
^^^^^^^^^^^^^^^^
.. autoclass:: podium.preproc.MosesNormalizer
   :members:
   :no-undoc-members:

Regex Replace
^^^^^^^^^^^^^
.. autoclass:: podium.preproc.RegexReplace
   :members:
   :no-undoc-members:


Keyword Extractor
^^^^^^^^^^^^^^^^^
.. autoclass:: podium.preproc.KeywordExtractor
   :members:
   :no-undoc-members:


Text Cleanup
^^^^^^^^^^^^
.. autoclass:: podium.preproc.TextCleanUp
   :members:
   :no-undoc-members:

NLTK Stemmer
^^^^^^^^^^^^
.. autoclass:: podium.preproc.NLTKStemmer
   :members:
   :no-undoc-members:

Spacy Lemmatizer
^^^^^^^^^^^^^^^^
.. autoclass:: podium.preproc.SpacyLemmatizer
   :members:
   :no-undoc-members:

Truecasing
^^^^^^^^^^
.. autofunction:: podium.preproc.truecase

Stopwords removal
^^^^^^^^^^^^^^^^^
.. autofunction:: podium.preproc.remove_stopwords

Tokenizers
-----------

.. autofunction:: podium.preproc.get_tokenizer

.. autoclass:: podium.preproc.SpacySentencizer
   :members:
   :no-undoc-members:

Utilities
----------

.. autofunction:: podium.preproc.as_posttokenize_hook
