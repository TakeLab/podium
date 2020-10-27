Core components
================

podium.Field
-------------

A :class:`podium.storage.Field` is the main class in which the data processing logic
is implemented. Data from datasets can exist in three modes in Fields: _raw_ (as-is in
datasets), *tokenized* (after tokenization) and _numericalized_ (tokens replaced with 
vocabulary indices).

To construct a Field, you need to define its name, as follows:

.. code-block:: python

   >>> text = Field(name='text', tokenize='spacy-en')
   >>> label = LabelField(name='label')

.. autoclass:: podium.storage.Field
   :members:
   :no-undoc-members:

podium.Vocab
-------------

.. autoclass:: podium.storage.Vocab
   :members:
   :no-undoc-members:

podium.Dataset
--------------
.. autoclass:: podium.datasets.Dataset
   :members: __getitem__, get, __len__, __iter__, __getattr__, filter


podium.storage.example\_factory module
---------------------------------------

.. autoclass:: podium.storage.example_factory
   :members:
   :no-undoc-members:
