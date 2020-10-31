Core components
================

Fields and Vocab
----------------

Field
^^^^^^

A :class:`podium.storage.Field` is the main class in which the data processing logic
is implemented. Data from datasets can exist in three modes in Fields: _raw_ (as-is in
datasets), *processed* (after tokenization) and _numericalized_ (tokens replaced with their indices in a vocabulary or via custom numericalization).

To construct a Field, you need to define its name, as follows:

.. autoclass:: podium.storage.Field
   :members:
   :no-undoc-members:

MultioutputField
^^^^^^^^^^^^^^^^^
.. autoclass:: podium.storage.MultioutputField
   :members:
   :no-undoc-members:

Vocab
^^^^^^
.. autoclass:: podium.storage.Vocab
   :members:
   :no-undoc-members:

Dataset classes
---------------

Dataset
^^^^^^^^
.. autoclass:: podium.datasets.Dataset
   :members:
   :no-undoc-members:

TabularDataset
^^^^^^^^^^^^^^^
.. autoclass:: podium.datasets.Dataset
   :members:
   :no-undoc-members:

Iterators
---------

Iterator
^^^^^^^^^
.. autoclass:: podium.datasets.Iterator
   :members:
   :no-undoc-members:

BucketIterator
^^^^^^^^^^^^^^^
.. autoclass:: podium.datasets.BucketIterator
   :members:
   :no-undoc-members:

SingleBatchIterator
^^^^^^^^^^^^^^^^^^^
.. autoclass:: podium.datasets.SingleBatchIterator
   :members:
   :no-undoc-members:
