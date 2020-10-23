podium.storage package
=======================

podium.storage.Field
----------------------------

A :class:`podium.storage.Field` is the main class in which the data processing logic
is implemented. Data from datasets can exist in three modes in Fields: _raw_ (as-is in
datasets), *tokenized* (after tokenization) and _numericalized_ (tokens replaced with 
vocabulary indices).

To construct a Field, you need to define its name, as follows:

::

   >>> text = Field(name='text', tokenize='spacy-en')
   >>> label = LabelField(name='label')

podium.storage.field module
----------------------------

.. automodule:: podium.storage.field
   :members:
   :no-undoc-members:
   :show-inheritance:

podium.storage.vocab module
----------------------------

.. automodule:: podium.storage.vocab
   :members:
   :no-undoc-members:
   :show-inheritance:

podium.storage.example\_factory module
---------------------------------------

.. automodule:: podium.storage.example_factory
   :members:
   :no-undoc-members:
   :show-inheritance:


Subpackages
-----------

.. toctree::

   podium.storage.resources
   podium.storage.vectorizers

