FAQ
====

**Q: Can I just load and retrieve a numericalized dataset from Podium?**


**A:** Yes, you can use the :meth:`Podium.datasets.Dataset.batch()` method. You still need to define the Fields if you are not using a built-in dataset.


.. code-block:: python

  >>> from podium.datasets import SST
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits()
  >>> x, y = sst_train.batch()
  >>> print(x.text.shape, y.label.shape, sep='\n')
  (6920, 52)
  (6920, 1)

Be aware that you will get a dataset as a matrix by default -- meaning that all the instances will be padded to the maximum length. If you wish to get a list of instances instead, set ``disable_batch_matrix=True`` in the constructor for the corresponding Field.

.. code-block:: python

  >>> from podium.datasets import SST
  >>> from podium import Vocab, Field, LabelField
  >>> text = Field(name='text', numericalizer=Vocab(), disable_batch_matrix=True)
  >>> label = LabelField(name='label')
  >>> fields = {'text':text, 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> x, y = sst_train.batch()
  >>> print(type(x.text), len(x.text))
  <class 'list'> 6920


**Your question here!**
